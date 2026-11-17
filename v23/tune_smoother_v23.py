"""
v23 Particle Smoother (固定ラグ系統スムーザー) カテゴリ別チューナー(6分割・各カテゴリ独立・再開可能)
================================================================================================
やること:
  v22 と同じ枠組み(selector の 6カテゴリ code 0..5 ごとに well を分け、各カテゴリで Optuna で
  overall RMSE 最小の PFパラメータを探索)で、**前向きPFに「後ろ向き(未来を使う)平滑」を足した**もの。

Particle Smoother とは:
  前向きPF(filter)は各ステップ t で「過去〜現在の観測 y_{1:t}」しか使わない。
  このコンペは **評価ゾーンの後方(toe側)も観測がある=look-ahead 合法**なので、
  ある中間ステップ t の予測に **未来 y_{t+1:t+L} の GR も使える**(= 平滑 smoothing)。
  → forward の弱い「中間ステップ(アンカーから遠いが、さらに先のGRで縛れる所)」を未来情報で締める狙い。

実装 = 固定ラグ系統スムーザー(fixed-lag genealogy smoother, 標準的で GPU軽量):
  - 前向きPFを普通に走らせる。リサンプリング時に粒子の**系譜(どの祖先由来か)**で過去Lステップ分の
    位置バッファを並べ替え、現在の粒子identityに整列させ続ける。
  - ステップ i まで進んだら、Lステップ前の t=i-L の平滑事後平均を
      smoothed_geo_t = Σ_n w_i^n · (現在粒子nの t時点の位置)
    で読み出す(= 現在の filter重み w_i で、未来 y_{1:i} を反映した重みで過去位置を平均)。
  - 末尾Lステップは最終重みで flush(ラグが縮むだけ、未来全部を使う)。
  - **L=平滑の強さ(先読みステップ数)**。L大ほど未来を多く使うが、系譜の退化(祖先が少数化)で
    効果は頭打ち/逆効果になりうる → Optuna で探索。

パラメータ調整込み:
  - v22(=v19) の PF全パラメータ空間 + スムーザー専用 2つ:
      smooth_lag   = L(先読みステップ数)
      smooth_blend = 最終予測 = (1-blend)*forward + blend*smoothed のブレンド比(0=前向きのみ,1=平滑のみ)
  - **v22 の best(per_category_pf.json)を warm-start で enqueue**(既知の良いPF設定から出発)。
  - objective は **blended の overall RMSE**。記録には forward-only / smoothed-only も残す(平滑の寄与が見える)。

出力: v23/artifacts_v23/per_category_smoother.json
実行: rogii_claude/.venv/Scripts/python.exe v23/tune_smoother_v23.py
※ build_cache/well_code/カテゴリ定義/IO は v22 と同一。エンジン(_pf_smoother_core)だけ平滑付き。
※ メモリ: バッファ (B=chunk*seeds, L, N)。OOM なら SEARCH_CHUNK / n_particles / smooth_lag 上限を下げる。
※ 別db(optuna_smoother.db)/state_smoother.json を使うので他チューナーと衝突しない。
"""

import os, time, json, glob, math, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import optuna

warnings.filterwarnings("ignore")
np.seterr(all="ignore")

# ============================== CONFIG ==============================
PROJECT   = Path(r"C:\Users\kosaka256\Documents\rogii_claude")
DATA_DIR  = PROJECT / "rogii-wellbore-geology-prediction" / "train"
OUT_DIR   = PROJECT / "v23" / "artifacts_v23"; OUT_DIR.mkdir(parents=True, exist_ok=True)
V22_PARAMS = PROJECT / "v22" / "artifacts_v22" / "per_category_pf.json"   # warm-start元(あれば)
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE     = torch.float32

SEARCH_SEEDS = 16         # 探索シード数(平滑はバッファでメモリ食うので v22(32) より控えめ)
EVAL_SEEDS   = 32         # カテゴリ best の記録用
SEED0        = 0
SEARCH_CHUNK = 200        # 探索GPUバッチwell数(バッファ B*L*N があるので控えめ。OOMなら下げる)
EVAL_CHUNK   = 120        # 評価分割
N_WELLS      = None       # None=全well, 整数で先頭N(デバッグ)

N_TRIALS     = 150        # 全パラメータ同時調整(SMOOTH_ONLY=False)時の1カテゴリ探索回数
USE_PHYS     = True

# ---- 2段チューニング ----
# True : PF を v22(per_category_pf.json)で固定し、smooth_lag/smooth_blend だけを調整(速い・推奨)。
# False: v22 と同じく PF全パラメータ + smooth を同時調整(重い)。
SMOOTH_ONLY        = True
SMOOTH_ONLY_TRIALS = 50   # SMOOTH_ONLY時の1カテゴリ探索回数(2次元なので少なくて足りる)

# ---- カテゴリ定義(7.191 selector と一致, v22 と同一)----
N_EVAL_THRESHOLD  = 4840.0
Z_SPAN_THRESHOLDS = (136.73000000000016, 185.5133333333342)
N_CATS = 6

# SMOOTH_ONLY か否かで出力/DBを分け、探索空間の食い違いを避ける(再開も独立)
_tag = "smoother_only" if SMOOTH_ONLY else "smoother"
STORAGE      = f"sqlite:///{(OUT_DIR / f'optuna_{_tag}.db').as_posix()}"
STATE_PATH   = OUT_DIR / f"state_{_tag}.json"
PER_CAT_PATH = OUT_DIR / f"per_category_{_tag}.json"
CUR_PATH     = OUT_DIR / f"current_cat_{_tag}.json"
# ===================================================================

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ----------------------- データ(v22と同一) -----------------------
def list_wells():
    return sorted({Path(p).stem.replace("__horizontal_well", "")
                   for p in glob.glob(str(DATA_DIR / "*__horizontal_well.csv"))})

def build_cache(wid):
    hw = pd.read_csv(DATA_DIR / f"{wid}__horizontal_well.csv")
    tw = pd.read_csv(DATA_DIR / f"{wid}__typewell.csv").sort_values("TVT")
    if "TVT" not in hw.columns:
        return None
    kn = hw[hw["TVT_input"].notna()]; ev = hw[hw["TVT_input"].isna()]
    if len(ev) == 0 or len(kn) < 10:
        return None
    tw_tvt = tw["TVT"].to_numpy(float); tw_gr = tw["GR"].to_numpy(float)
    if len(tw_tvt) < 3:
        return None
    tw_fb = float(np.nanmean(tw_gr)) if np.isfinite(np.nanmean(tw_gr)) else 0.
    gr_filled = hw["GR"].astype(float).interpolate(limit_direction="both").fillna(tw_fb).to_numpy(float)
    ev_pos = hw.index.get_indexer(ev.index); kpos = hw.index.get_indexer(kn.index)
    ls = float(kn["TVT_input"].iloc[-1] + kn["Z"].iloc[-1])
    tail = kn.tail(30); dt = np.diff(tail["TVT_input"].values); dz = np.diff(tail["Z"].values)
    dm = np.diff(tail["MD"].values); m = dm > 0
    ir = float(np.median((dt + dz)[m] / dm[m])) if m.sum() >= 3 else 0.
    tv = ev["TVT"].values.astype(np.float64) if "TVT" in ev.columns else None
    z_ev = ev["Z"].values.astype(np.float64)
    n_eval = int(len(z_ev))
    z_span = float(np.nanmax(z_ev) - np.nanmin(z_ev)) if len(z_ev) else 0.0
    return dict(md=ev["MD"].values.astype(np.float64), z=z_ev,
                gr_filled=gr_filled, ev_pos=ev_pos, kpos=kpos,
                kn_tvtin=kn["TVT_input"].values.astype(float),
                tw_tvt=tw_tvt, tw_gr=tw_gr, tw_fb=tw_fb, ls=ls, ir=ir, tvt_true=tv,
                n_eval=n_eval, z_span=z_span)

def well_code(c):
    n_bin = int(c["n_eval"] > N_EVAL_THRESHOLD)
    z_bin = int(np.searchsorted(Z_SPAN_THRESHOLDS, c["z_span"], side="right"))
    return n_bin + 2 * z_bin

def _smooth(vals, fb, r):
    r = int(r); s = pd.Series(vals, dtype="float32").interpolate(limit_direction="both").fillna(float(fb))
    return s.to_numpy(np.float32) if r <= 0 else s.rolling(2 * r + 1, center=True, min_periods=1).mean().to_numpy(np.float32)

def _grid(tw_tvt, tw_gr, step=0.2):
    a, b = float(tw_tvt.min()), float(tw_tvt.max()); g = np.arange(a, b + step, step)
    return np.interp(g, tw_tvt, tw_gr).astype(np.float64), float(a), float(step)

def make_inputs(c, P):
    tw_gr_pf = _smooth(c["tw_gr"], c["tw_fb"], P["tw_gr_smooth_r"]).astype(np.float64)
    gg, gmin, gst = _grid(c["tw_tvt"], tw_gr_pf)
    gr_sm = _smooth(c["gr_filled"], c["tw_fb"], P["hgr_smooth_r"])
    if len(c["kpos"]) < 20:
        gs = P["gr_sig_def"]
    else:
        resid = gr_sm[c["kpos"]] - np.interp(c["kn_tvtin"], c["tw_tvt"], tw_gr_pf)
        gs = float(np.nanstd(resid)); gs = P["gr_sig_def"] if (not np.isfinite(gs) or gs <= 0) else gs
    gs = float(np.clip(gs * P["gr_sig_mult"], P["gr_sig_min"], max(P["gr_sig_max"], P["gr_sig_min"] + 1e-6)))
    return dict(md=c["md"], z=c["z"], gr=gr_sm[c["ev_pos"]].astype(np.float64),
                gg=gg, gmin=gmin, gst=gst, gs=gs, ls=c["ls"], ir=c["ir"], N=int(P["n_particles"]))


# ----------------------- 物理PF + 固定ラグ系統スムーザー GPU -----------------------
def _pf_smoother_core(md, z, gr, valid, grid, glen, vmin, step, gs, ls, ir, P, N, device, gen, dtype=DTYPE):
    """
    v22 の物理PF(前向き)に、固定ラグ系統スムーザーを足したもの。
    返り値:
      pts_f : (B,T) 前向き filter の事後平均 TVT(= v22 と同じ)
      pts_s : (B,T) 固定ラグ平滑の事後平均 TVT
      log_lik: (B,) 前向き周辺尤度の対数和(シード加重アンサンブル用, v22 と同じ)
    """
    B, T = md.shape
    ALPHA = P["mom"]; RN = P["vn"]; PN = P["pn"]; IR = P["init_rate_std"]; IS = P["init_pos_std"]
    RP = P["rp"]; RR = P["rr"]; RESAMP = P["resamp"]; RATE_CLIP = P["rate_clip"]
    GR_POWER = P["gr_power"]; JUMP_PROB = P["jump_prob"]; JUMP_STD = P["jump_std"]; CLIP = P["tvt_clip_margin"]
    PHYS_SIG = P["phys_sig"]; USE_P = P["use_phys"]
    L = int(P.get("smooth_lag", 32)); L = max(0, min(L, T))   # 固定ラグ(先読みステップ数)

    def rn(): return torch.randn((B, N), generator=gen, device=device, dtype=dtype)
    pos = ls[:, None] + IS * rn(); rate = ir[:, None] + IR * rn()
    w = torch.full((B, N), 1.0 / N, device=device, dtype=dtype)
    log_lik = torch.zeros(B, device=device, dtype=torch.float64)
    pts_f = torch.zeros((B, T), device=device, dtype=dtype)
    pts_s = torch.zeros((B, T), device=device, dtype=dtype)
    tvt_lo = vmin - CLIP; tvt_hi = vmin + (glen.to(dtype) - 1) * step + CLIP
    glast = torch.gather(grid, 1, (glen - 1).clamp_min(0).unsqueeze(1)); g0 = grid[:, 0:1]
    pm = md[:, 0] - 1.0; arN = torch.arange(N, device=device, dtype=dtype)
    zero = torch.zeros((), device=device, dtype=dtype); dipp = ir[:, None]

    # 固定ラグ用バッファ: buf[:, k, :] = 現在の粒子identityに整列した「過去の geo位置」(円環バッファ)
    use_sm = L > 0
    if use_sm:
        buf = torch.full((B, L, N), float("nan"), device=device, dtype=dtype)
        ptr = 0
        arNl = arN.long()

    for i in range(T):
        act = valid[:, i]; cur = md[:, i]; dm = (cur - pm).clamp_min(1.0)
        rate_n = ALPHA * rate + RN * rn()
        if RATE_CLIP > 0: rate_n = rate_n.clamp(-RATE_CLIP, RATE_CLIP)
        pos_n = pos + rate_n * dm[:, None] + PN * rn()
        if JUMP_PROB > 0 and JUMP_STD > 0:
            jm = torch.rand((B, N), generator=gen, device=device, dtype=dtype) < JUMP_PROB
            pos_n = pos_n + torch.where(jm, JUMP_STD * rn(), zero)
        zi = z[:, i][:, None]; tvt = (pos_n - zi).clamp(tvt_lo[:, None], tvt_hi[:, None]); pos_n = tvt + zi
        am = act[:, None]; pos = torch.where(am, pos_n, pos); rate = torch.where(am, rate_n, rate)
        gri = gr[:, i]; obs = act & ~torch.isnan(gri)
        v = pos - zi; ii = (v - vmin[:, None]) / step[:, None]; i0 = torch.floor(ii).long()
        below = i0 < 0; above = i0 >= (glen - 1)[:, None]
        i0c = i0.clamp_min(0); i0c = torch.minimum(i0c, (glen - 2).clamp_min(0)[:, None])
        t = ii - i0c.to(dtype); gA = torch.gather(grid, 1, i0c); gB = torch.gather(grid, 1, i0c + 1)
        eg = gA * (1 - t) + gB * t; eg = torch.where(below, g0, eg); eg = torch.where(above, glast, eg)
        d = ((gri[:, None] - eg) / gs[:, None]).abs(); dp = d ** GR_POWER
        lk = torch.where(dp < 600.0, torch.exp(-0.5 * dp), zero).clamp_min(1e-300)
        avg = (w * lk).sum(1)
        log_lik = log_lik + torch.where(obs, torch.log(avg.clamp_min(1e-300).double()),
                                        torch.zeros((), device=device, dtype=torch.float64))
        if USE_P:
            dphys = (rate - dipp) / PHYS_SIG; lkp = torch.exp(-0.5 * dphys * dphys).clamp_min(1e-300)
        else:
            lkp = torch.ones_like(w)
        w_new = torch.where(obs[:, None], w * lk * lkp, w * lkp)
        ws = w_new.sum(1, keepdim=True)
        w = torch.where(ws > 0, w_new / ws.clamp_min(1e-300), torch.full_like(w, 1.0 / N))
        ne = (w * w).sum(1); need = act & ((1.0 / ne) < (RESAMP * N))
        if bool(need.any()):
            cumw = torch.cumsum(w, 1); u0 = torch.rand((B, 1), generator=gen, device=device, dtype=dtype) * (1.0 / N)
            u = u0 + arN[None, :] / N; idx = torch.searchsorted(cumw, u, right=False).clamp(0, N - 1)
            pos_rs = torch.gather(pos, 1, idx) + RP * rn(); rate_rs = torch.gather(rate, 1, idx) + RR * rn()
            if RATE_CLIP > 0: rate_rs = rate_rs.clamp(-RATE_CLIP, RATE_CLIP)
            nm = need[:, None]; pos = torch.where(nm, pos_rs, pos); rate = torch.where(nm, rate_rs, rate)
            w = torch.where(nm, torch.full_like(w, 1.0 / N), w)
            if use_sm:
                # 系譜で過去バッファを並べ替え(リサンプリングした行だけ)。idx:(B,N)
                buf_g = torch.gather(buf, 2, idx[:, None, :].expand(B, L, N))
                buf = torch.where(need[:, None, None], buf_g, buf)

        # 前向き filter 読み出し
        pts_f[:, i] = (w * (pos - zi)).sum(1)

        # 固定ラグ平滑: 現在重み w で「Lステップ前の整列済み位置」を平均 → t=i-L を確定
        if use_sm:
            if i >= L:
                old = buf[:, ptr, :]                       # = step i-L の geo位置(現在粒子に整列)
                pts_s[:, i - L] = (w * old).sum(1) - z[:, i - L]
            buf[:, ptr, :] = pos                            # 現在(step i)の geo位置を書き込む
            ptr = (ptr + 1) % L

        pm = torch.where(act, cur, pm)

    # 末尾Lステップを最終重みで flush(ラグ縮小=未来全部使用)
    if use_sm:
        for j in range(max(0, T - L), T):
            pts_s[:, j] = (w * buf[:, j % L, :]).sum(1) - z[:, j]
    else:
        pts_s = pts_f
    return pts_f, pts_s, log_lik


def _pad(inps, device, dtype=DTYPE):
    W = len(inps); Tmax = max(len(x["md"]) for x in inps); Gmax = max(len(x["gg"]) for x in inps)
    md = torch.zeros((W, Tmax), dtype=dtype); z = torch.zeros((W, Tmax), dtype=dtype)
    gr = torch.full((W, Tmax), float("nan"), dtype=dtype); valid = torch.zeros((W, Tmax), dtype=torch.bool)
    grid = torch.zeros((W, Gmax), dtype=dtype); glen = torch.zeros(W, dtype=torch.long)
    vmin = torch.zeros(W, dtype=dtype); step = torch.zeros(W, dtype=dtype); gs = torch.zeros(W, dtype=dtype)
    ls = torch.zeros(W, dtype=dtype); ir = torch.zeros(W, dtype=dtype)
    for b, x in enumerate(inps):
        Tn = len(x["md"]); G = len(x["gg"])
        md[b, :Tn] = torch.from_numpy(x["md"].astype("float32"))
        if Tn < Tmax: md[b, Tn:] = md[b, Tn - 1]
        z[b, :Tn] = torch.from_numpy(x["z"].astype("float32"))
        gr[b, :Tn] = torch.from_numpy(x["gr"].astype("float32")); valid[b, :Tn] = True
        grid[b, :G] = torch.from_numpy(x["gg"].astype("float32"))
        if G < Gmax: grid[b, G:] = grid[b, G - 1]
        glen[b] = G; vmin[b] = x["gmin"]; step[b] = x["gst"]; gs[b] = x["gs"]; ls[b] = x["ls"]; ir[b] = x["ir"]
    to = lambda tt: tt.to(device)
    return dict(md=to(md), z=to(z), gr=to(gr), valid=to(valid), grid=to(grid), glen=to(glen),
                vmin=to(vmin), step=to(step), gs=to(gs), ls=to(ls), ir=to(ir))

def run_smoother_ens(inps, P, device, seed, n_seeds):
    """シード尤度加重アンサンブル後の forward / smoothed 予測(各 well の eval長配列)を返す。"""
    N = int(P["n_particles"]); S = n_seeds; W = len(inps); pad = _pad(inps, device)
    rep = lambda tt: tt.repeat_interleave(S, 0); gen = torch.Generator(device=device); gen.manual_seed(seed)
    pf, ps, ll = _pf_smoother_core(
        rep(pad["md"]), rep(pad["z"]), rep(pad["gr"]), rep(pad["valid"]), rep(pad["grid"]),
        pad["glen"].repeat_interleave(S), pad["vmin"].repeat_interleave(S), pad["step"].repeat_interleave(S),
        pad["gs"].repeat_interleave(S), pad["ls"].repeat_interleave(S), pad["ir"].repeat_interleave(S),
        P, N, device, gen)
    pf = pf.view(W, S, -1).double().cpu().numpy(); ps = ps.view(W, S, -1).double().cpu().numpy()
    ll = ll.view(W, S).cpu().numpy()
    out_f = []; out_s = []
    for w in range(W):
        T = len(inps[w]["md"]); llw = ll[w]
        if np.isfinite(llw).any():
            mx = np.nanmax(llw[np.isfinite(llw)])
            lk = np.where(np.isfinite(llw), llw - mx, -np.inf)
            ww = np.exp(lk / max(P["likelihood_scale"], 1e-6)); s = ww.sum()
            ww = ww / s if s > 1e-300 else np.full(S, 1.0 / S)
        else:
            ww = np.full(S, 1.0 / S)
        out_f.append((ww[:, None] * pf[w, :, :T]).sum(0))
        out_s.append((ww[:, None] * ps[w, :, :T]).sum(0))
    return out_f, out_s


# ----------------------- 評価 -----------------------
def predict_errs(P, subset, seed, n_seeds, chunk):
    """各 well の (forward誤差, smoothed誤差) を返す。"""
    ef = {}; es = {}
    for c0 in range(0, len(subset), chunk):
        sub = subset[c0:c0 + chunk]
        inps = [make_inputs(CACHE[w], P) for w in sub]
        pf, ps = run_smoother_ens(inps, P, DEVICE, seed, n_seeds)
        for k, w in enumerate(sub):
            ef[w] = pf[k] - CACHE[w]["tvt_true"]; es[w] = ps[k] - CACHE[w]["tvt_true"]
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    return ef, es

def _rmse(errs, subset):
    e = np.concatenate([errs[w] for w in subset]); e = e[np.isfinite(e)]
    if e.size == 0 or not np.isfinite(e).all():
        return 1e3
    return float(np.sqrt(np.mean(e ** 2)))

def eval_blend(P, subset, seed, n_seeds, chunk):
    """(blended_rmse, forward_rmse, smoothed_rmse) を返す。blend = P['smooth_blend']。"""
    ef, es = predict_errs(P, subset, seed, n_seeds, chunk)
    bl = float(P.get("smooth_blend", 1.0))
    eb = {w: (1 - bl) * ef[w] + bl * es[w] for w in subset}
    return _rmse(eb, subset), _rmse(ef, subset), _rmse(es, subset)


# ----------------------- Optuna 探索空間(v22=v19 と同一 + スムーザー2つ) -----------------------
def suggest_smooth(trial):
    return dict(
        smooth_lag   = trial.suggest_categorical("smooth_lag", [8, 16, 32, 48, 64, 96]),
        smooth_blend = trial.suggest_float("smooth_blend", 0.0, 1.0),
    )

def suggest_params(trial, fixed=None):
    # fixed が与えられたら PF は固定し smooth だけ探索(2段チューニングの②)
    if fixed is not None:
        P = {k: v for k, v in fixed.items() if k != "use_phys"}
        P.update(suggest_smooth(trial)); P["use_phys"] = USE_PHYS
        return P
    return dict(
        n_particles      = trial.suggest_categorical("n_particles", [400, 500, 600, 800]),
        likelihood_scale = trial.suggest_float("likelihood_scale", 6.0, 60.0, log=True),
        init_pos_std     = trial.suggest_float("init_pos_std", 0.1, 4.0, log=True),
        init_rate_std    = trial.suggest_float("init_rate_std", 0.0003, 0.03, log=True),
        mom              = trial.suggest_float("mom", 0.990, 0.99998),
        vn               = trial.suggest_float("vn", 0.0003, 0.012, log=True),
        pn               = trial.suggest_float("pn", 0.002, 0.06, log=True),
        rp               = trial.suggest_float("rp", 0.1, 1.5, log=True),
        rr               = trial.suggest_float("rr", 0.00005, 0.004, log=True),
        resamp           = trial.suggest_float("resamp", 0.03, 0.6, log=True),
        gr_sig_min       = trial.suggest_float("gr_sig_min", 4.0, 12.0),
        gr_sig_max       = trial.suggest_float("gr_sig_max", 22.0, 60.0),
        gr_sig_def       = trial.suggest_float("gr_sig_def", 70.0, 160.0),
        gr_sig_mult      = trial.suggest_float("gr_sig_mult", 1.6, 4.0),
        tvt_clip_margin  = trial.suggest_float("tvt_clip_margin", 50.0, 200.0),
        rate_clip        = trial.suggest_categorical("rate_clip", [0.0, 1.5, 2.0, 2.5, 3.0, 5.0]),
        gr_power         = trial.suggest_float("gr_power", 1.4, 3.2),
        hgr_smooth_r     = trial.suggest_categorical("hgr_smooth_r", [0, 1, 2, 3, 4, 6]),
        tw_gr_smooth_r   = trial.suggest_categorical("tw_gr_smooth_r", [0, 1, 2]),
        jump_prob        = trial.suggest_categorical("jump_prob", [0.0, 0.002, 0.005, 0.01, 0.02]),
        jump_std         = trial.suggest_categorical("jump_std", [5.0, 10.0, 15.0, 20.0]),
        phys_sig         = trial.suggest_float("phys_sig", 0.003, 5.0, log=True),
        # ---- スムーザー専用 ----
        smooth_lag       = trial.suggest_categorical("smooth_lag", [8, 16, 32, 48, 64, 96]),  # L=先読みステップ数
        smooth_blend     = trial.suggest_float("smooth_blend", 0.0, 1.0),                     # forward↔smoothed ブレンド
        use_phys         = USE_PHYS,
    )

def full_params(best_params):
    return {**best_params, "use_phys": USE_PHYS}

def warmstart_params():
    """v22 per_category_pf.json から {code: params} を読む(無ければ {})。"""
    if not V22_PARAMS.exists():
        return {}
    try:
        d = json.loads(V22_PARAMS.read_text())
        out = {}
        for code, rec in d.get("categories", {}).items():
            bp = rec.get("best_params", {})
            if bp:
                out[int(code)] = bp
        return out
    except Exception:
        return {}


# ----------------------- 状態(再開用) -----------------------
def load_state():
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {"done_codes": [], "results": {}}

def save_state(state):
    STATE_PATH.write_text(json.dumps(state, indent=2))
    PER_CAT_PATH.write_text(json.dumps(dict(
        algorithm="fixed_lag_genealogy_smoother", n_eval_threshold=N_EVAL_THRESHOLD,
        z_span_thresholds=list(Z_SPAN_THRESHOLDS), eval_seeds=EVAL_SEEDS, use_phys=USE_PHYS,
        categories={str(code): state["results"][str(code)] for code in range(N_CATS)
                    if str(code) in state["results"]},
    ), indent=2))


# ----------------------- カテゴリ実行 -----------------------
def run_category(code, wells, warm):
    seed = SEED0 + 100 * code
    # 2段チューニング: SMOOTH_ONLY なら PF を v22 best で固定し smooth だけ探索
    fixed = None; n_trials = N_TRIALS
    if SMOOTH_ONLY:
        if code in warm:
            fixed = {k: v for k, v in warm[code].items() if k != "use_phys"}
            n_trials = SMOOTH_ONLY_TRIALS
        else:
            print(f"[cat {code}] SMOOTH_ONLYだが v22 best 無し → 全パラメータ調整にフォールバック", flush=True)
    study = optuna.create_study(direction="minimize", study_name=f"cat{code}",
                                storage=STORAGE, load_if_exists=True,
                                sampler=optuna.samplers.TPESampler(seed=seed))
    done = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)
    # warm-start: 全パラメータ調整時のみ v22 best を初回 enqueue(SMOOTH_ONLY時はPF固定なので不要)
    if fixed is None and done == 0 and code in warm:
        wp = dict(warm[code]); wp.pop("use_phys", None)
        wp.setdefault("smooth_lag", 32); wp.setdefault("smooth_blend", 0.5)
        try:
            study.enqueue_trial(wp, skip_if_exists=True)
            print(f"[cat {code}] warm-start enqueued from v22 best (+lag32/blend0.5)", flush=True)
        except Exception as e:
            print(f"[cat {code}] warm-start enqueue 失敗: {e}", flush=True)
    remaining = max(0, n_trials - done)
    mode = "SMOOTH_ONLY(PF固定)" if fixed is not None else "全パラメータ"
    print(f"[cat {code}] wells={len(wells)} seed={seed} [{mode}] optuna {done}/{n_trials} done, run {remaining} more", flush=True)

    def merged(study_params):
        base = dict(fixed) if fixed is not None else {}
        return full_params({**base, **study_params})

    def objective(trial):
        P = suggest_params(trial, fixed=fixed)
        try:
            rb, rf, rs = eval_blend(P, wells, seed, SEARCH_SEEDS, SEARCH_CHUNK)
        except Exception:
            return 1e3
        trial.set_user_attr("forward_rmse", rf); trial.set_user_attr("smoothed_rmse", rs)
        return rb if math.isfinite(rb) else 1e3

    def cb(study, trial):
        try:
            bv = study.best_value
        except Exception:
            return
        nd = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)
        CUR_PATH.write_text(json.dumps(dict(code=code, seed=seed, mode=mode, trials_done=nd, n_wells=len(wells),
                                            best_value=bv, best_params=merged(study.best_params)), indent=2))
        tv = trial.value if trial.value is not None else float("nan")
        rf = trial.user_attrs.get("forward_rmse", float("nan"))
        rs = trial.user_attrs.get("smoothed_rmse", float("nan"))
        print(f"   [cat{code}] {nd}/{n_trials} blend={tv:.4f} (fwd={rf:.4f} smo={rs:.4f}) best={bv:.4f}", flush=True)

    if remaining > 0:
        study.optimize(objective, n_trials=remaining, callbacks=[cb])
    bp = merged(study.best_params)
    rb, rf, rs = eval_blend(bp, wells, seed, EVAL_SEEDS, EVAL_CHUNK)
    print(f"[cat {code}] best(search)={study.best_value:.4f}  EVAL blend={rb:.4f} fwd={rf:.4f} smo={rs:.4f}", flush=True)
    return dict(code=code, seed=seed, n_wells=len(wells), best_value_search=float(study.best_value),
                overall_blend=float(rb), overall_forward=float(rf), overall_smoothed=float(rs), best_params=bp)


def main():
    global CACHE
    if DEVICE != "cuda":
        print("WARNING: CUDA not available, will be slow.")
    print(f"device={DEVICE}  storage={STORAGE}")

    wells = list_wells()
    if N_WELLS:
        wells = wells[:N_WELLS]
    print(f"loading {len(wells)} wells ...")
    CACHE = {}; t0 = time.time()
    for i, w in enumerate(wells):
        c = build_cache(w)
        if c is not None and c["tvt_true"] is not None:
            CACHE[w] = c
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(wells)} ({time.time()-t0:.0f}s)")
    all_wells = [w for w in wells if w in CACHE]

    cats = {code: [] for code in range(N_CATS)}
    for w in all_wells:
        cats[well_code(CACHE[w])].append(w)
    print(f"usable wells: {len(all_wells)}")
    print("カテゴリ別 well 数:", {code: len(cats[code]) for code in range(N_CATS)})

    warm = warmstart_params()
    print(f"warm-start(v22): codes = {sorted(warm.keys())}")
    state = load_state()
    print(f"resume: done codes = {state['done_codes']}")

    for code in range(N_CATS):
        if code in state["done_codes"]:
            continue
        if len(cats[code]) == 0:
            print(f"[cat {code}] well 0 件 → スキップ")
            state["done_codes"].append(code); save_state(state); continue
        rec = run_category(code, cats[code], warm)
        state["results"][str(code)] = rec
        state["done_codes"].append(code)
        save_state(state)
        print(f"==> cat {code} saved. blend={rec['overall_blend']:.4f} (fwd={rec['overall_forward']:.4f} smo={rec['overall_smoothed']:.4f})\n")

    print("ALL CATEGORIES DONE.")
    for code in range(N_CATS):
        r = state["results"].get(str(code))
        if r:
            print(f"  cat {code}: wells={r['n_wells']:4d}  blend={r['overall_blend']:.4f} "
                  f"fwd={r['overall_forward']:.4f} smo={r['overall_smoothed']:.4f}")
    print(f"6つの smoother パラメータ -> {PER_CAT_PATH}")


if __name__ == "__main__":
    main()
