"""
v9 パーティクルフィルタ GPU実行(最速バッチ・GPUのみ)
====================================================

ancc PF を GPU(PyTorch)で、複数坑井 × シードを1バッチにまとめて実行する。
比較用コード(CPU/numba・シードチェック)は含まない。実運用・本番推論向けの素の実行版。

- 並列化: 時間(MDステップ)方向のみ逐次。各ステップは (B, N) のベクトル演算。
  B = 坑井数 × シード数。坑井ごとに尤度加重でアンサンブルした TVT 予測を返す。
- 坑井数が多くても VRAM に収まるよう WELL_CHUNK 単位でチャンク処理。
- 出力: 坑井ごとの予測(評価ゾーンの TVT)。必要なら下の SAVE_PATH に保存。

使い方:
  rogii_claude/.venv/Scripts/python.exe v9/pf_gpu_run.py
"""

import os, time, glob
import numpy as np
import pandas as pd
import torch

# ============================== CONFIG ==============================
DATA_DIR   = r"C:\Users\kosaka256\Documents\rogii_claude\rogii-wellbore-geology-prediction\train"
N_WELLS    = 773          # None=全坑井, または整数で先頭N坑井
N_SEEDS    = 32
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE      = torch.float32
PARAM_SET  = "pf_1"        # "pf_1" / "pf_2" / "pf_3"
SEED       = 0
WELL_CHUNK = 773            # 1回のGPUバッチに載せる坑井数(VRAMに応じて調整)
SAVE_PATH  = None          # 例: r"...\pf_ancc_pred.npz" を入れると坑井名→予測を保存
# ===================================================================


# ----------------------- PF パラメータ定義 -----------------------
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
    p = {}
    p["ANCC_N"]            = int(r.get("n_particles", 600))
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
                N=int(p["ANCC_N"]),
                ev_index=ev.index.to_numpy())  # 提出 id 用に評価ゾーンの行インデックス


# ============================== GPU コア(坑井×シードをバッチ化) ==============================
def _pf_ancc_gpu_core(md, z, gr, valid, grid, glen, vmin, step, gs, ls, ir, p, N, device, gen, dtype=DTYPE):
    """
    各行(row)= 坑井 × シード、独立な N 粒子アンサンブル。
    md,z,gr,valid:(B,Tmax)  grid:(B,Gmax)  glen,vmin,step,gs,ls,ir:(B,)
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
    glast = torch.gather(grid, 1, (glen-1).clamp_min(0).unsqueeze(1))
    g0col = grid[:, 0:1]
    pm = md[:, 0] - 1.0
    arN = torch.arange(N, device=device, dtype=dtype)
    zero = torch.zeros((), device=device, dtype=dtype)

    for i in range(T):
        act = valid[:, i]
        cur = md[:, i]
        dm = (cur - pm).clamp_min(1.0)
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
        # 尤度
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
        # リサンプリング(系統的)
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
        tvt_out = pos - zi
        pts[:, i] = (w*tvt_out).sum(1)
        pm = torch.where(act, cur, pm)
    return pts, log_lik

def _pad_inputs(inps, device, dtype=DTYPE):
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
                vmin=to(vmin), step=to(step), gs=to(gs), ls=to(ls), ir=to(ir))

def _run_chunk(inps, p, S, device, seed):
    """1チャンク(複数坑井)を B=W*S で実行し、坑井ごとのアンサンブル予測を返す。"""
    N = int(p["ANCC_N"])
    pad = _pad_inputs(inps, device); W = len(inps)
    rep = lambda tt: tt.repeat_interleave(S, dim=0)
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
        outs.append((ww[:, None]*pr).sum(0).astype(np.float32))
    return outs

def run_gpu_ancc(inps, p, S, device, seed=0, well_chunk=64):
    """全坑井を well_chunk ごとに分けて GPU 実行。坑井順の予測リストを返す。"""
    preds = []
    for c in range(0, len(inps), well_chunk):
        preds.extend(_run_chunk(inps[c:c+well_chunk], p, S, device, seed))
    return preds


# ============================== main ==============================
def main():
    p = make_p(PF_PARAM_SETS_RAW[PARAM_SET])
    S = N_SEEDS
    print(f"device={DEVICE}  param_set={PARAM_SET}  seeds={S}  N={p['ANCC_N']}  chunk={WELL_CHUNK}")
    if DEVICE == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    wells = sorted({os.path.basename(q).split("__")[0] for q in glob.glob(DATA_DIR+"/*__horizontal_well.csv")})
    if N_WELLS is not None:
        wells = wells[:N_WELLS]

    inps = []; names = []
    for w in wells:
        hw = pd.read_csv(f"{DATA_DIR}/{w}__horizontal_well.csv")
        tw = pd.read_csv(f"{DATA_DIR}/{w}__typewell.csv")
        inp = build_inputs(hw, tw["TVT"].values.astype(float), tw["GR"].values.astype(float), p)
        if inp is not None:
            inps.append(inp); names.append(w)
    print(f"loaded wells: {len(inps)}")

    # ウォームアップ(CUDA初期化を計測から除外)
    _ = run_gpu_ancc(inps[:1], p, S, DEVICE, seed=SEED, well_chunk=WELL_CHUNK)
    if DEVICE == "cuda": torch.cuda.synchronize()

    t0 = time.perf_counter()
    preds = run_gpu_ancc(inps, p, S, DEVICE, seed=SEED, well_chunk=WELL_CHUNK)
    if DEVICE == "cuda": torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    print(f"done: {len(preds)} wells in {dt:.2f}s  ({dt/max(len(preds),1)*1000:.1f} ms/well)")

    if SAVE_PATH:
        np.savez_compressed(SAVE_PATH, **{nm: preds[i] for i, nm in enumerate(names)})
        print(f"saved -> {SAVE_PATH}")

if __name__ == "__main__":
    main()
