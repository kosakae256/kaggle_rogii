"""
v9 パーティクルフィルタ GPU高速化 比較スクリプト (ancc PF)
=========================================================

目的:
  v9 の粒子フィルタ `_pf_ancc`(n_seeds 回まわす方=コストの主)を GPU(PyTorch)で
  ベクトル化し、CPU(numba)版と「速度」「精度」を比較する。

方針:
  - 時間(MDステップ)方向は逐次依存なので並列化しない。
  - 「シード × 粒子」次元を (B, N) テンソルで一括演算する。
    さらに GPU を活かすため「複数坑井」もバッチ次元(B = 坑井数 × シード数)に積む版を用意。
  - 乱数列が CPU と変わるため結果は完全一致しない(=シード変動)。ユーザー要件どおり OK。
    重要なのは「精度(真値 TVT に対する RMSE)が CPU より悪化しないこと」。本スクリプトで検証する。

注意:
  - GPU化したのは ancc PF のみ(`run_pf_ancc` 相当)。`_pf_z` はシングルシード1回なので
    n_seeds 比較の対象外。必要なら同じ要領で移植可能。
  - dtype は float32(GPU標準)。numba CPU は float64。粒子フィルタは確率的なので
    float32 でも精度は十分なはず、というのを RMSE で確認する。
  - このスクリプトはまだ実行確認していない。初回実行時に挙動を確認のこと。

使い方:
  rogii_claude/.venv/Scripts/python.exe v9/pf_gpu_benchmark.py
  下の CONFIG を書き換えて手で実験する(坑井数 K、シード数 S、パラメータセットなど)。
"""

import os, time, glob
import numpy as np
import pandas as pd
from numba import njit
import torch

# ============================== CONFIG ==============================
# データ場所(このプロジェクトの train を使う)
DATA_DIR = r"C:\Users\kosaka256\Documents\rogii_claude\rogii-wellbore-geology-prediction\train"

N_WELLS   = 100     # 比較に使う坑井数(手で増やして速度差を見る)
N_SEEDS   = 8     # ★ n_seeds=8 で比較
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE     = torch.float32
PARAM_SET = "pf_1"   # "pf_1" / "pf_2" / "pf_3"

MEASURE_GPU_SINGLE = False  # GPU単坑井ループ(遅い)も測るか。Trueでも先頭6坑井だけ計測。
RUN_SEED_CHECK     = True    # シード変動チェック(CPU/GPUを別シードで複数回回し、RMSEのバラつきを見る)
SEED_CHECK_REPEATS = 4       # 各方式を何回まわすか
# ===================================================================


# ----------------------- PF パラメータ定義 -----------------------
# v9 ノートブックの PF_PARAM_SETS_RAW から抜粋(代表3セット)。
PF_PARAM_SETS_RAW = {
    "pf_1": {"gr_power":2.1947763195255123,"gr_sig_def":114.11439615641156,"gr_sig_max":32.79346108797473,
             "gr_sig_min":7.495879205722816,"gr_sig_mult":2.7482077023565075,"hgr_smooth_r":3,
             "init_pos_std":0.43824435375436277,"init_rate_std":0.001840222365629066,"jump_prob":0.002,
             "jump_std":5.0,"likelihood_scale":20.0,"mom":0.998833416480357,"n_particles":600,"n_seeds":32,
             "pn":0.006788762926604484,"rate_clip":2.0,"resamp":0.060642973044944336,"rp":0.6864944651997384,
             "rr":0.00015873788792956448,"tvt_clip_margin":113.08092552242186,"tw_gr_smooth_r":0,
             "vn":0.0013871256161405515},
    "pf_2": {"gr_sig_def":71.37105679394953,"gr_sig_max":44.7373292550486,"gr_sig_min":6.520200142977573,
             "gr_sig_mult":1.9000654781073127,"init_pos_std":1.1039311766848976,"init_rate_std":0.009213697911348712,
             "likelihood_scale":18.584958539746772,"mom":0.9994265753132063,"n_particles":600,"n_seeds":32,
             "pn":0.011257078147676399,"resamp":0.30834451379977984,"rp":0.49711567429512854,
             "rr":0.00031680690540158854,"tvt_clip_margin":63.752292532272826,"vn":0.0010650490363481595,
             "rate_clip":0.0,"gr_power":2.0,"hgr_smooth_r":0,"tw_gr_smooth_r":0,"jump_prob":0.0,"jump_std":10.0},
    "pf_3": {"gr_power":2.074149546524975,"gr_sig_def":66.97913258656452,"gr_sig_max":50.843656140023995,
             "gr_sig_min":9.427442041004895,"gr_sig_mult":2.4485083695548218,"hgr_smooth_r":0,
             "init_pos_std":2.3279966476603975,"init_rate_std":0.03938758391370206,"jump_prob":0.002,
             "jump_std":5.0,"likelihood_scale":14.890876838217093,"mom":0.9999769474035675,"n_particles":600,
             "n_seeds":32,"pn":0.04394815893193551,"rate_clip":0.0,"resamp":0.054606961294962585,
             "rp":0.42322457711970995,"rr":0.002148810696831248,"tvt_clip_margin":91.4794941498565,
             "tw_gr_smooth_r":0,"vn":0.0006883646035942229},
}

def make_p(r):
    """raw dict -> 内部パラメータ名(ノートブックの _make_pf_params 相当の必要分)。"""
    p = {}
    p["ANCC_N"]            = int(r.get("n_particles", 600))
    p["PF_SEED0"]         = 0
    p["PF_LIKELIHOOD_SCALE"] = float(r.get("likelihood_scale", 20.0))
    p["PF_INIT_SPR"]      = float(r.get("init_pos_std", 0.5))
    p["PF_INIT_V_STD"]    = float(r.get("init_rate_std", 0.01))
    p["PF_MOM"]           = float(r.get("mom", 0.999))
    p["PF_VN"]            = float(r.get("vn", 0.001))
    p["PF_PN"]            = float(r.get("pn", 0.005))
    p["PF_ROUGH_P"]       = float(r.get("rp", 0.5))
    p["PF_ROUGH_V"]       = float(r.get("rr", 0.001))
    p["PF_RESAMP"]        = float(r.get("resamp", 0.5))
    p["PF_GR_SIG_MIN"]    = float(r.get("gr_sig_min", 5.0))
    p["PF_GR_SIG_MAX"]    = float(r.get("gr_sig_max", 50.0))
    p["PF_GR_SIG_DEF"]    = float(r.get("gr_sig_def", 100.0))
    p["PF_GR_SIG_MULT"]   = float(r.get("gr_sig_mult", 2.0))
    p["PF_TVT_CLIP_MARGIN"] = float(r.get("tvt_clip_margin", 50.0))
    p["PF_RATE_CLIP"]     = float(r.get("rate_clip", 0.0))
    p["PF_GR_POWER"]      = float(r.get("gr_power", 2.0))
    p["PF_HGR_SMOOTH_R"]  = int(r.get("hgr_smooth_r", 0))
    p["PF_TW_GR_SMOOTH_R"] = int(r.get("tw_gr_smooth_r", 0))
    p["PF_JUMP_PROB"]     = float(r.get("jump_prob", 0.0))
    p["PF_JUMP_STD"]      = float(r.get("jump_std", 0.0))
    # ancc は PF の遷移/初期/リサンプリングを流用
    p["ANCC_ALPHA"]=p["PF_MOM"]; p["ANCC_RN"]=p["PF_VN"]; p["ANCC_PN"]=p["PF_PN"]
    p["ANCC_IR"]=p["PF_INIT_V_STD"]; p["ANCC_IS"]=p["PF_INIT_SPR"]
    p["ANCC_RP"]=p["PF_ROUGH_P"]; p["ANCC_RR"]=p["PF_ROUGH_V"]
    return p


# ----------------------- 入力作成ヘルパ(ノートブックから) -----------------------
def _smooth_radius_values(vals, fb, r):
    r = int(r)
    s = pd.Series(vals, dtype="float32").interpolate(limit_direction="both").fillna(float(fb))
    return s.to_numpy(np.float32) if r <= 0 else s.rolling(2*r+1, center=True, min_periods=1).mean().to_numpy(np.float32)

def _grid(tw_tvt, tw_gr, step=0.2):
    tmin = float(tw_tvt.min()); tmax = float(tw_tvt.max())
    tvt_g = np.arange(tmin, tmax+step, step)
    return np.interp(tvt_g, tw_tvt, tw_gr).astype(np.float64), float(tmin), float(step)

def _gr_sig(hw, tw_tvt, tw_gr, p):
    gr_fb = float(np.nanmean(tw_gr)) if np.isfinite(np.nanmean(tw_gr)) else 0.
    hgr = _smooth_radius_values(hw["GR"].astype(float).to_numpy(), gr_fb, p["PF_HGR_SMOOTH_R"])
    kn = hw[hw["TVT_input"].notna()]
    if len(kn) < 20:
        gs = float(p["PF_GR_SIG_DEF"])
    else:
        kpos = hw.index.get_indexer(kn.index)
        resid = hgr[kpos] - np.interp(kn["TVT_input"].values, tw_tvt, tw_gr)
        gs = float(np.nanstd(resid))
        if not np.isfinite(gs) or gs <= 0:
            gs = float(p["PF_GR_SIG_DEF"])
    return float(np.clip(gs*p["PF_GR_SIG_MULT"], p["PF_GR_SIG_MIN"], p["PF_GR_SIG_MAX"]))

def build_inputs(hw, tw_tvt, tw_gr, p):
    """run_pf_ancc のシードループ手前までを再現し、PF への入力一式を返す。"""
    tw_fb = float(np.nanmean(tw_gr)) if np.isfinite(np.nanmean(tw_gr)) else 0.
    tw_gr_pf = _smooth_radius_values(tw_gr, tw_fb, p["PF_TW_GR_SMOOTH_R"]).astype(np.float64)
    gs = _gr_sig(hw, tw_tvt, tw_gr_pf, p)
    kn = hw[hw["TVT_input"].notna()]; ev = hw[hw["TVT_input"].isna()]
    if len(ev) == 0:
        return None
    ls = float(kn["TVT_input"].iloc[-1] + kn["Z"].iloc[-1])
    tail = kn.tail(30)
    dt = np.diff(tail["TVT_input"].values); dz = np.diff(tail["Z"].values); dm = np.diff(tail["MD"].values); m = dm > 0
    ir = float(np.median((dt+dz)[m]/dm[m])) if m.sum() >= 3 else 0.
    gg, gmin, gst = _grid(tw_tvt, tw_gr_pf)
    md_ev = ev["MD"].values.astype(np.float64); z_ev = ev["Z"].values.astype(np.float64)
    gr_full = hw["GR"].astype(float).interpolate(limit_direction="both").fillna(tw_fb).to_numpy(float)
    gr_proc = _smooth_radius_values(gr_full, tw_fb, p["PF_HGR_SMOOTH_R"])
    ev_pos = hw.index.get_indexer(ev.index)
    gr_ev = gr_proc[ev_pos].astype(np.float64)
    return dict(md=md_ev, z=z_ev, gr=gr_ev, gg=gg, gmin=gmin, gst=gst, gs=gs, ls=ls, ir=ir,
                N=int(p["ANCC_N"]), tvt_true=ev["TVT"].values.astype(np.float64))


# ============================== CPU 版(numba・忠実コピー) ==============================
@njit(cache=True)
def _interp1(grid, v, vmin, step):
    i = int((v-vmin)/step)
    if i < 0: return grid[0]
    n = len(grid)-1
    if i >= n: return grid[n]
    t = (v-vmin)/step - i
    return grid[i]*(1.-t) + grid[i+1]*t

@njit(cache=True)
def _resamp(pos, aux, w, N, rp, rv):
    cum = np.zeros(N+1)
    for j in range(N): cum[j+1] = cum[j]+w[j]
    u0 = np.random.uniform(0., 1./N); np2 = np.empty(N); na = np.empty(N); ci = 0
    for j in range(N):
        u = u0 + j/N
        while ci < N-1 and cum[ci+1] < u: ci += 1
        np2[j] = pos[ci] + rp*np.random.randn()
        na[j]  = aux[ci] + rv*np.random.randn()
    return np2, na

@njit(cache=True)
def _pf_ancc(md_v, z_v, gr_v, gg, vmin, step, gs, ls, ir, N, ALPHA, RN, PN, IR, IS, RP, RR, RESAMP,
             RATE_CLIP, GR_POWER, JUMP_PROB, JUMP_STD, CLIP_MARGIN, SEED_VALUE):
    if SEED_VALUE >= 0:
        np.random.seed(SEED_VALUE)
    pos = np.empty(N); rate = np.empty(N); w = np.ones(N)/N
    for j in range(N):
        pos[j] = ls + IS*np.random.randn()
        rate[j] = ir + IR*np.random.randn()
    pts = np.empty(len(md_v)); std_ = np.empty(len(md_v)); pm = md_v[0]-1.; log_lik = 0.
    tvt_lo = vmin-CLIP_MARGIN; tvt_hi = vmin+(len(gg)-1)*step+CLIP_MARGIN
    for i in range(len(md_v)):
        dm = md_v[i]-pm; dm = max(dm, 1.)
        for j in range(N):
            rate[j] = ALPHA*rate[j] + RN*np.random.randn()
            if RATE_CLIP > 0.:
                if rate[j] > RATE_CLIP: rate[j] = RATE_CLIP
                elif rate[j] < -RATE_CLIP: rate[j] = -RATE_CLIP
            pos[j] += rate[j]*dm + PN*np.random.randn()
            if JUMP_PROB > 0. and JUMP_STD > 0. and np.random.random() < JUMP_PROB:
                pos[j] += JUMP_STD*np.random.randn()
            tvt_j = pos[j]-z_v[i]; tvt_j = max(tvt_j, tvt_lo); tvt_j = min(tvt_j, tvt_hi); pos[j] = tvt_j+z_v[i]
        if not np.isnan(gr_v[i]):
            ws = 0.; avg_lk = 0.
            for j in range(N):
                eg = _interp1(gg, pos[j]-z_v[i], vmin, step)
                d = abs((gr_v[i]-eg)/gs); dp = d**GR_POWER
                lk = max(np.exp(-0.5*dp) if dp < 600. else 0., 1e-300)
                old_w = w[j]; avg_lk += old_w*lk; w[j] = old_w*lk; ws += w[j]
            log_lik += np.log(max(avg_lk, 1e-300))
            if ws > 0.:
                for j in range(N): w[j] /= ws
            else:
                for j in range(N): w[j] = 1./N
        ne = 0.
        for j in range(N): ne += w[j]*w[j]
        if 1./ne < RESAMP*N:
            pos, rate = _resamp(pos, rate, w, N, RP, RR)
            if RATE_CLIP > 0.:
                for j in range(N):
                    if rate[j] > RATE_CLIP: rate[j] = RATE_CLIP
                    elif rate[j] < -RATE_CLIP: rate[j] = -RATE_CLIP
            for j in range(N): w[j] = 1./N
        tv = 0.
        for j in range(N): tv += w[j]*(pos[j]-z_v[i])
        pts[i] = tv; va = 0.
        for j in range(N): va += w[j]*(pos[j]-z_v[i]-tv)**2
        std_[i] = va**0.5; pm = md_v[i]
    return pts, std_, log_lik

def _ensemble(preds, liks, p):
    """シード方向に尤度加重平均(run_pf_ancc と同じ式)。"""
    pred = np.stack(preds, 0)
    lik = np.asarray(liks, float); lik = lik - np.nanmax(lik)
    ww = np.exp(lik / max(float(p["PF_LIKELIHOOD_SCALE"]), 1e-6)); ww /= max(ww.sum(), 1e-300)
    return (ww[:, None]*pred).sum(0)

def run_cpu(inp, p, S, seed0=None):
    if seed0 is None:
        seed0 = int(p["PF_SEED0"])
    preds = []; liks = []
    for s in range(S):
        pts, _, ll = _pf_ancc(inp["md"], inp["z"], inp["gr"], inp["gg"], inp["gmin"], inp["gst"],
                              inp["gs"], inp["ls"], inp["ir"], inp["N"],
                              p["ANCC_ALPHA"], p["ANCC_RN"], p["ANCC_PN"], p["ANCC_IR"], p["ANCC_IS"],
                              p["ANCC_RP"], p["ANCC_RR"], p["PF_RESAMP"], p["PF_RATE_CLIP"], p["PF_GR_POWER"],
                              p["PF_JUMP_PROB"], p["PF_JUMP_STD"], p["PF_TVT_CLIP_MARGIN"],
                              int(seed0)+s)
        preds.append(pts); liks.append(ll)
    return _ensemble(preds, liks, p)


# ============================== GPU 版(PyTorch・坑井×シードをバッチ化) ==============================
def _pf_ancc_gpu_core(md, z, gr, valid, grid, glen, vmin, step, gs, ls, ir, p, N, device, gen, dtype=DTYPE):
    """
    各行(row)が独立な N 粒子アンサンブル。row = 坑井 × シード。
    md,z,gr,valid: (B,Tmax)   grid:(B,Gmax)  glen,vmin,step,gs,ls,ir:(B,)
    時間方向のみ Python ループ。各ステップは (B,N) のベクトル演算。
    """
    B, T = md.shape
    ALPHA=p["ANCC_ALPHA"]; RN=p["ANCC_RN"]; PN=p["ANCC_PN"]; IR=p["ANCC_IR"]; IS=p["ANCC_IS"]
    RP=p["ANCC_RP"]; RR=p["ANCC_RR"]; RESAMP=p["PF_RESAMP"]; RATE_CLIP=p["PF_RATE_CLIP"]
    GR_POWER=p["PF_GR_POWER"]; JUMP_PROB=p["PF_JUMP_PROB"]; JUMP_STD=p["PF_JUMP_STD"]; CLIP=p["PF_TVT_CLIP_MARGIN"]

    def rn():
        return torch.randn((B, N), generator=gen, device=device, dtype=dtype)

    pos = ls[:, None] + IS*rn()
    rate = ir[:, None] + IR*rn()
    w = torch.full((B, N), 1.0/N, device=device, dtype=dtype)
    log_lik = torch.zeros(B, device=device, dtype=torch.float64)
    pts = torch.zeros((B, T), device=device, dtype=dtype)

    tvt_lo = vmin - CLIP
    tvt_hi = vmin + (glen.to(dtype)-1)*step + CLIP
    glast = torch.gather(grid, 1, (glen-1).clamp_min(0).unsqueeze(1))  # (B,1) 各行のグリッド末尾値
    g0col = grid[:, 0:1]
    pm = md[:, 0] - 1.0
    arN = torch.arange(N, device=device, dtype=dtype)
    zero = torch.zeros((), device=device, dtype=dtype)

    for i in range(T):
        act = valid[:, i]                      # (B,) このステップが有効な行
        cur = md[:, i]
        dm = (cur - pm).clamp_min(1.0)         # (B,)
        # --- 伝播 ---
        rate_n = ALPHA*rate + RN*rn()
        if RATE_CLIP > 0:
            rate_n = rate_n.clamp(-RATE_CLIP, RATE_CLIP)
        pos_n = pos + rate_n*dm[:, None] + PN*rn()
        if JUMP_PROB > 0 and JUMP_STD > 0:
            jm = torch.rand((B, N), generator=gen, device=device, dtype=dtype) < JUMP_PROB
            pos_n = pos_n + torch.where(jm, JUMP_STD*rn(), zero)
        zi = z[:, i][:, None]
        tvt = (pos_n - zi).clamp(tvt_lo[:, None], tvt_hi[:, None])
        pos_n = tvt + zi
        am = act[:, None]
        pos = torch.where(am, pos_n, pos)
        rate = torch.where(am, rate_n, rate)
        # --- 尤度(GR 補間)---
        gri = gr[:, i]
        obs = act & ~torch.isnan(gri)
        v = pos - zi
        ii = (v - vmin[:, None]) / step[:, None]
        i0 = torch.floor(ii).long()
        below = i0 < 0
        above = i0 >= (glen-1)[:, None]
        i0c = i0.clamp_min(0)
        i0c = torch.minimum(i0c, (glen-2).clamp_min(0)[:, None])
        t = ii - i0c.to(dtype)
        gA = torch.gather(grid, 1, i0c)
        gB = torch.gather(grid, 1, i0c+1)
        eg = gA*(1-t) + gB*t
        eg = torch.where(below, g0col, eg)
        eg = torch.where(above, glast, eg)
        d = ((gri[:, None] - eg) / gs[:, None]).abs()
        dp = d**GR_POWER
        lk = torch.where(dp < 600.0, torch.exp(-0.5*dp), zero).clamp_min(1e-300)
        avg_lk = (w*lk).sum(1)
        log_lik = log_lik + torch.where(obs, torch.log(avg_lk.clamp_min(1e-300).double()),
                                        torch.zeros((), device=device, dtype=torch.float64))
        w_new = w*lk
        ws = w_new.sum(1, keepdim=True)
        w_upd = torch.where(ws > 0, w_new / ws.clamp_min(1e-300), torch.full_like(w, 1.0/N))
        w = torch.where(obs[:, None], w_upd, w)
        # --- リサンプリング(系統的)---
        ne = (w*w).sum(1)
        need = act & ((1.0/ne) < (RESAMP*N))
        if bool(need.any()):
            cumw = torch.cumsum(w, 1)
            u0 = torch.rand((B, 1), generator=gen, device=device, dtype=dtype) * (1.0/N)
            u = u0 + arN[None, :]/N
            idx = torch.searchsorted(cumw, u, right=False).clamp(0, N-1)
            pos_rs = torch.gather(pos, 1, idx) + RP*rn()
            rate_rs = torch.gather(rate, 1, idx) + RR*rn()
            if RATE_CLIP > 0:
                rate_rs = rate_rs.clamp(-RATE_CLIP, RATE_CLIP)
            nm = need[:, None]
            pos = torch.where(nm, pos_rs, pos)
            rate = torch.where(nm, rate_rs, rate)
            w = torch.where(nm, torch.full_like(w, 1.0/N), w)
        # --- 出力(重み付き平均 TVT)---
        tvt_out = pos - zi
        pts[:, i] = (w*tvt_out).sum(1)
        pm = torch.where(act, cur, pm)
    return pts, log_lik

def _pad_inputs(inps, device, dtype=DTYPE):
    """可変長の坑井入力を (W,Tmax)/(W,Gmax) にパディングし、有効マスクを作る。"""
    W = len(inps)
    Tmax = max(len(x["md"]) for x in inps)
    Gmax = max(len(x["gg"]) for x in inps)
    md = torch.zeros((W, Tmax), dtype=dtype); z = torch.zeros((W, Tmax), dtype=dtype)
    gr = torch.full((W, Tmax), float("nan"), dtype=dtype); valid = torch.zeros((W, Tmax), dtype=torch.bool)
    grid = torch.zeros((W, Gmax), dtype=dtype); glen = torch.zeros(W, dtype=torch.long)
    vmin = torch.zeros(W, dtype=dtype); step = torch.zeros(W, dtype=dtype); gs = torch.zeros(W, dtype=dtype)
    ls = torch.zeros(W, dtype=dtype); ir = torch.zeros(W, dtype=dtype)
    for b, x in enumerate(inps):
        T = len(x["md"]); G = len(x["gg"])
        md[b, :T] = torch.from_numpy(x["md"].astype("float32"))
        if T < Tmax: md[b, T:] = md[b, T-1]
        z[b, :T] = torch.from_numpy(x["z"].astype("float32"))
        gr[b, :T] = torch.from_numpy(x["gr"].astype("float32")); valid[b, :T] = True
        grid[b, :G] = torch.from_numpy(x["gg"].astype("float32"))
        if G < Gmax: grid[b, G:] = grid[b, G-1]
        glen[b] = G; vmin[b] = x["gmin"]; step[b] = x["gst"]; gs[b] = x["gs"]; ls[b] = x["ls"]; ir[b] = x["ir"]
    to = lambda tt: tt.to(device)
    return dict(md=to(md), z=to(z), gr=to(gr), valid=to(valid), grid=to(grid), glen=to(glen),
                vmin=to(vmin), step=to(step), gs=to(gs), ls=to(ls), ir=to(ir), Tmax=Tmax, W=W)

def run_gpu_batched(inps, p, S, device, seed=0):
    """複数坑井 × S シードを1つのバッチ(B=W*S)で同時実行。坑井ごとのアンサンブル予測を返す。"""
    N = int(p["ANCC_N"])
    pad = _pad_inputs(inps, device); W = pad["W"]
    rep = lambda tt: tt.repeat_interleave(S, dim=0)   # 行順 = 坑井メジャー(w0s0,w0s1,...)
    gen = torch.Generator(device=device); gen.manual_seed(seed)
    pts, log_lik = _pf_ancc_gpu_core(
        rep(pad["md"]), rep(pad["z"]), rep(pad["gr"]), rep(pad["valid"]), rep(pad["grid"]),
        pad["glen"].repeat_interleave(S), pad["vmin"].repeat_interleave(S), pad["step"].repeat_interleave(S),
        pad["gs"].repeat_interleave(S), pad["ls"].repeat_interleave(S), pad["ir"].repeat_interleave(S),
        p, N, device, gen)
    pts = pts.view(W, S, -1).double().cpu().numpy()
    ll = log_lik.view(W, S).cpu().numpy()
    outs = []
    for w in range(W):
        T = len(inps[w]["md"]); pr = pts[w, :, :T]
        lk = ll[w] - np.nanmax(ll[w])
        ww = np.exp(lk / max(float(p["PF_LIKELIHOOD_SCALE"]), 1e-6)); ww /= max(ww.sum(), 1e-300)
        outs.append((ww[:, None]*pr).sum(0))
    return outs

def run_gpu_single(inp, p, S, device, seed=0):
    """1坑井のみをバッチ(B=S)で実行。"""
    return run_gpu_batched([inp], p, S, device, seed)[0]


# ============================== 比較ドライバ ==============================
def rmse(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.sqrt(np.mean((a[m]-b[m])**2)))

def mean_rmse_over_wells(preds, inps):
    return float(np.mean([rmse(preds[i], inps[i]["tvt_true"]) for i in range(len(inps))]))

def seed_variation_check(inps, p, S, device, repeats):
    """
    CPU/GPU をそれぞれ別シードで repeats 回まわし、全坑井平均RMSEの分布を出す。
    CPU自身のバラつき幅に GPU の値が収まっていれば、CPU/GPU差は乱数変動であり優劣なしと判断できる。
    """
    print(f"\n===== seed variation check (repeats={repeats}) =====")
    cpu_scores = []
    for r in range(repeats):
        preds = [run_cpu(x, p, S, seed0=r*10000) for x in inps]
        cpu_scores.append(mean_rmse_over_wells(preds, inps))
    gpu_scores = []
    for r in range(repeats):
        preds = run_gpu_batched(inps, p, S, device, seed=r)
        if device == "cuda": torch.cuda.synchronize()
        gpu_scores.append(mean_rmse_over_wells(preds, inps))
    cs = np.array(cpu_scores); gs = np.array(gpu_scores)
    print(f"CPU mean-RMSE per run: {[f'{v:.3f}' for v in cpu_scores]}")
    print(f"    min={cs.min():.3f} max={cs.max():.3f} mean={cs.mean():.3f} std={cs.std():.3f}")
    print(f"GPU mean-RMSE per run: {[f'{v:.3f}' for v in gpu_scores]}")
    print(f"    min={gs.min():.3f} max={gs.max():.3f} mean={gs.mean():.3f} std={gs.std():.3f}")
    overlap = (gs.min() <= cs.max()) and (gs.max() >= cs.min())
    print(f"判定: GPUの分布がCPUの分布と{'重なる→乱数変動(優劣なし)' if overlap else '重ならない→要調査'}")

def main():
    p = make_p(PF_PARAM_SETS_RAW[PARAM_SET])
    S = N_SEEDS
    print(f"device={DEVICE}  param_set={PARAM_SET}  seeds={S}  N={p['ANCC_N']}")
    if DEVICE == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    wells = sorted({os.path.basename(q).split("__")[0] for q in glob.glob(DATA_DIR+"/*__horizontal_well.csv")})
    inps = []; names = []
    for w in wells[:N_WELLS]:
        hw = pd.read_csv(f"{DATA_DIR}/{w}__horizontal_well.csv")
        tw = pd.read_csv(f"{DATA_DIR}/{w}__typewell.csv")
        inp = build_inputs(hw, tw["TVT"].values.astype(float), tw["GR"].values.astype(float), p)
        if inp is not None:
            inps.append(inp); names.append(w)
    print(f"loaded wells: {len(inps)}")

    # ウォームアップ(numba コンパイル / CUDA 初期化は計測から除外)
    _ = run_cpu(inps[0], p, 1)
    _ = run_gpu_single(inps[0], p, S, DEVICE)
    if DEVICE == "cuda": torch.cuda.synchronize()

    # CPU
    t0 = time.perf_counter()
    cpu_preds = [run_cpu(x, p, S) for x in inps]
    t_cpu = time.perf_counter() - t0

    # GPU 単坑井(坑井ループ・遅いので任意。測る場合も先頭6坑井だけで1坑井あたりに換算)
    t_gs_per = None
    if MEASURE_GPU_SINGLE:
        sub = inps[:min(6, len(inps))]
        t0 = time.perf_counter()
        _ = [run_gpu_single(x, p, S, DEVICE) for x in sub]
        if DEVICE == "cuda": torch.cuda.synchronize()
        t_gs_per = (time.perf_counter() - t0) / len(sub)

    # GPU バッチ(全坑井同時)
    t0 = time.perf_counter()
    gb_preds = run_gpu_batched(inps, p, S, DEVICE)
    if DEVICE == "cuda": torch.cuda.synchronize()
    t_gb = time.perf_counter() - t0

    # 精度比較(真値 TVT に対する RMSE)+ CPU/GPU 一致度
    print(f"\n{'well':10}{'RMSE_CPU':>10}{'RMSE_GPU':>10}{'CPUvsGPU':>10}")
    rc, rg = [], []
    for i, nm in enumerate(names):
        tr = inps[i]["tvt_true"]
        a = rmse(cpu_preds[i], tr); b = rmse(gb_preds[i], tr); c = rmse(cpu_preds[i], gb_preds[i])
        rc.append(a); rg.append(b)
        if len(inps) <= 12:
            print(f"{nm:10}{a:10.3f}{b:10.3f}{c:10.3f}")
    print(f"\nmean RMSE vs TVT : CPU={np.mean(rc):.4f}   GPU(batched)={np.mean(rg):.4f}")
    print(f"mean |CPU-GPU| 一致度RMSE(坑井平均): {np.mean([rmse(cpu_preds[i], gb_preds[i]) for i in range(len(inps))]):.4f}")
    print(f"\ntime: CPU={t_cpu:.2f}s   GPU_batched={t_gb:.2f}s   (wells={len(inps)})")
    print(f"speedup  batched vs CPU : {t_cpu/max(t_gb,1e-9):.2f}x")
    if t_gs_per is not None:
        print(f"GPU_single: {t_gs_per:.3f}s/well  (= 全{len(inps)}坑井換算 {t_gs_per*len(inps):.1f}s, 遅い)")

    if RUN_SEED_CHECK:
        seed_variation_check(inps, p, S, DEVICE, SEED_CHECK_REPEATS)

if __name__ == "__main__":
    main()
