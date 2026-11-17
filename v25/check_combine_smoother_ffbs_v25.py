"""
v25 診断: smoother(固定ラグ) と FFBS を「多重で組み合わせる」価値があるか(相補性の天井)
================================================================================
背景(LB): smoother(lag192)+beam+hold = 6.786(ベスト) / FFBS+beam+hold = 6.886。両方 v9(6.946)突破。
問い: 2つは「別の坑井で当たってる(相補)」か「同じ坑井で当たってる(冗長)」か。
  相補なら合成/選択で 6.786 を割れる。冗長なら片方でよい。→ **oracle(各坑井で良い方を選ぶ)の天井**を測る。

各坑井で両方の最終予測(平滑 → カテゴリ別 beam+hold → apply_pp)を出し、per-well の二乗誤差和で比較:
  smoother_overall  … 固定ラグ+beam+hold の全行プールRMSE
  ffbs_overall      … FFBS+beam+hold の全行プールRMSE
  oracle_overall    … 各坑井で SSE の小さい方を採用してプール(=選択完璧の天井)
  + どちらが何坑井で勝つか / 予測の不一致 mean|smoother-ffbs| の分布

読み: oracle << min(smoother, ffbs) なら相補=合成に伸びしろ大。oracle ≈ min なら冗長。

出力: v25/artifacts_v25/combine_smoother_ffbs.json
実行: rogii_claude/.venv/Scripts/python.exe v25/check_combine_smoother_ffbs_v25.py
※ 提出ノートと同じ n_seed=32 で両エンジンを回す。FFBSは坑井ごとで重い→ N_WELLS で絞ってよい。
"""

import time, json, glob, math, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.signal import savgol_filter
from numba import njit

warnings.filterwarnings("ignore")
np.seterr(all="ignore")

# ============================== CONFIG ==============================
PROJECT   = Path(r"C:\Users\kosaka256\Documents\rogii_claude")
DATA_DIR  = PROJECT / "rogii-wellbore-geology-prediction" / "train"
OUT_DIR   = PROJECT / "v25" / "artifacts_v25"; OUT_DIR.mkdir(parents=True, exist_ok=True)
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE     = torch.float32

SEED     = 0
N_SEED   = 32            # ★提出ノート(submit_*_v25)と揃える
N_WELLS  = 250           # None=全773。FFBSが重いので絞り推奨

# smoother(固定ラグ) 側 = submit_smoother_v25 と同じ
SMOOTH_LAG   = 192
SMOOTH_BLEND = 1.0
SMOOTH_CLAMP = None      # submit_smoother 既定 None

# FFBS 側 = submit_ffbsi_v25 と同じ
INFLATE   = 32.0
N_BWD     = 128
FFBS_CLAMP = 60.0
FFBS_BLEND = 1.0

# カテゴリ別 beam + hold(両方に同一適用)
N_EVAL_THRESHOLD  = 4840.0
Z_SPAN_THRESHOLDS = (136.73000000000016, 185.5133333333342)
BEAM_W = {0: 0.0, 1: 0.0, 2: 0.2, 3: 0.0, 4: 0.05, 5: 0.2}
HOLD_W = {0: 0.2, 1: 0.15, 2: 0.15, 3: 0.15, 4: 0.05, 5: 0.05}
BEAM_CONFIGS = [
    (10, 20.0, 144.0, 2), (10, 8.0, 64.0, 2), (8, 35.0, 220.0, 1), (10, 14.0, 90.0, 5),
    (20, 4.0, 36.0, 3), (12, 12.0, 100.0, 3), (15, 25.0, 180.0, 2), (20, 30.0, 200.0, 2),
    (15, 10.0, 80.0, 4), (25, 6.0, 50.0, 3), (10, 40.0, 300.0, 1), (12, 18.0, 120.0, 5),
    (30, 8.0, 70.0, 2), (10, 50.0, 400.0, 0),
]
APPLY_PP  = True
PP_TAU = 85.0; PP_ALPHA = 1.0; SG_W, SG_P = 17, 3

OUT_PATH = OUT_DIR / "combine_smoother_ffbs.json"

R0 = {
    "n_particles": 400, "likelihood_scale": 10.536319667317654,
    "init_pos_std": 0.9331541486020234, "init_rate_std": 0.00047753793163756304,
    "mom": 0.9997911117425017, "vn": 0.0006297672017716754, "pn": 0.005014009140727263,
    "rp": 0.7627309890952053, "rr": 0.00045663430474523125, "resamp": 0.032049849143995945,
    "gr_sig_min": 11.865793233968713, "gr_sig_max": 44.98927964612316, "gr_sig_def": 119.25397464636981,
    "gr_sig_mult": 3.058318309943017, "tvt_clip_margin": 150.56111180234504, "rate_clip": 2.0,
    "gr_power": 1.7826896851395453, "hgr_smooth_r": 0, "tw_gr_smooth_r": 0,
    "jump_prob": 0.002, "jump_std": 5.0, "phys_sig": 0.013666550145707291, "use_phys": False,
}
# ===================================================================


# ----------------------- データ -----------------------
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
    n_eval = int(len(z_ev)); z_span = float(np.nanmax(z_ev) - np.nanmin(z_ev)) if len(z_ev) else 0.0
    return dict(md=ev["MD"].values.astype(np.float64), z=z_ev, gr_filled=gr_filled, ev_pos=ev_pos, kpos=kpos,
                kn_tvtin=kn["TVT_input"].values.astype(float), tw_tvt=tw_tvt, tw_gr=tw_gr, tw_fb=tw_fb,
                ls=ls, ir=ir, tvt_true=tv, n_eval=n_eval, z_span=z_span,
                last_known_md=float(kn["MD"].iloc[-1]), last_known_tvt=float(kn["TVT_input"].iloc[-1]))

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


# ----------------------- FFBS engine -----------------------
def ffbs_forward_well(inp, P, seed, S, device, dtype=DTYPE):
    N = int(P["n_particles"]); T = len(inp["md"])
    ALPHA = P["mom"]; RN = P["vn"]; PN = P["pn"]; IR = P["init_rate_std"]; IS = P["init_pos_std"]
    RP = P["rp"]; RR = P["rr"]; RESAMP = P["resamp"]; RATE_CLIP = P["rate_clip"]
    GR_POWER = P["gr_power"]; JUMP_PROB = P["jump_prob"]; JUMP_STD = P["jump_std"]; CLIP = P["tvt_clip_margin"]
    PHYS_SIG = P["phys_sig"]; USE_P = P["use_phys"]
    md = torch.from_numpy(inp["md"].astype("float32")).to(device)
    z  = torch.from_numpy(inp["z"].astype("float32")).to(device)
    gr = torch.from_numpy(inp["gr"].astype("float32")).to(device)
    grid = torch.from_numpy(inp["gg"].astype("float32")).to(device); G = grid.numel()
    vmin = float(inp["gmin"]); step = float(inp["gst"]); gs = float(inp["gs"])
    ls = float(inp["ls"]); ir = float(inp["ir"])
    gen = torch.Generator(device=device); gen.manual_seed(seed)
    def rn(): return torch.randn((S, N), generator=gen, device=device, dtype=dtype)
    pos = ls + IS * rn(); rate = ir + IR * rn()
    w = torch.full((S, N), 1.0 / N, device=device, dtype=dtype)
    loglik = torch.zeros(S, device=device, dtype=torch.float64)
    Xpos = torch.empty((S, T, N), device=device, dtype=dtype)
    Xrate = torch.empty((S, T, N), device=device, dtype=dtype)
    Wst = torch.empty((S, T, N), device=device, dtype=dtype)
    DM = torch.empty(T, device=device, dtype=dtype)
    tvt_lo = vmin - CLIP; tvt_hi = vmin + (G - 1) * step + CLIP
    g0 = grid[0]; glast = grid[G - 1]
    zero = torch.zeros((), device=device, dtype=dtype); arN = torch.arange(N, device=device, dtype=dtype)
    pm = float(inp["md"][0]) - 1.0
    for i in range(T):
        cur = float(inp["md"][i]); dm = max(cur - pm, 1.0); DM[i] = dm
        rate_n = ALPHA * rate + RN * rn()
        if RATE_CLIP > 0: rate_n = rate_n.clamp(-RATE_CLIP, RATE_CLIP)
        pos_n = pos + rate_n * dm + PN * rn()
        if JUMP_PROB > 0 and JUMP_STD > 0:
            jm = torch.rand((S, N), generator=gen, device=device, dtype=dtype) < JUMP_PROB
            pos_n = pos_n + torch.where(jm, JUMP_STD * rn(), zero)
        zi = z[i]; tvt = (pos_n - zi).clamp(tvt_lo, tvt_hi); pos_n = tvt + zi
        pos = pos_n; rate = rate_n
        gri = gr[i]; obs = not bool(torch.isnan(gri))
        v = pos - zi; ii = (v - vmin) / step; i0 = torch.floor(ii).long()
        below = i0 < 0; above = i0 >= (G - 1)
        i0c = i0.clamp(0, G - 2); t = ii - i0c.to(dtype)
        gA = grid[i0c]; gB = grid[i0c + 1]
        eg = gA * (1 - t) + gB * t; eg = torch.where(below, g0, eg); eg = torch.where(above, glast, eg)
        d = ((gri - eg) / gs).abs(); dp = d ** GR_POWER
        lk = torch.where(dp < 600.0, torch.exp(-0.5 * dp), zero).clamp_min(1e-300)
        if obs:
            avg = (w * lk).sum(1); loglik = loglik + torch.log(avg.clamp_min(1e-300).double())
        if USE_P:
            dphys = (rate - ir) / PHYS_SIG; lkp = torch.exp(-0.5 * dphys * dphys).clamp_min(1e-300)
        else:
            lkp = torch.ones_like(w)
        w_new = (w * lk * lkp) if obs else (w * lkp)
        ws = w_new.sum(1, keepdim=True)
        w = torch.where(ws > 0, w_new / ws.clamp_min(1e-300), torch.full_like(w, 1.0 / N))
        Xpos[:, i, :] = pos; Xrate[:, i, :] = rate; Wst[:, i, :] = w
        ne = (w * w).sum(1); need = (1.0 / ne) < (RESAMP * N)
        if bool(need.any()):
            cumw = torch.cumsum(w, 1); u0 = torch.rand((S, 1), generator=gen, device=device, dtype=dtype) * (1.0 / N)
            u = u0 + arN[None, :] / N; idx = torch.searchsorted(cumw, u, right=False).clamp(0, N - 1)
            pos_rs = torch.gather(pos, 1, idx) + RP * rn(); rate_rs = torch.gather(rate, 1, idx) + RR * rn()
            if RATE_CLIP > 0: rate_rs = rate_rs.clamp(-RATE_CLIP, RATE_CLIP)
            nm = need[:, None]; pos = torch.where(nm, pos_rs, pos); rate = torch.where(nm, rate_rs, rate)
            w = torch.where(nm, torch.full_like(w, 1.0 / N), w)
        pm = cur
    return Xpos, Xrate, Wst, DM, loglik, z

def reduce_cloud(Xpos, Xrate, Wst, n_bwd, gen, device, dtype=DTYPE):
    S, T, N = Xpos.shape
    cumw = torch.cumsum(Wst, dim=2)
    ar = torch.arange(n_bwd, device=device, dtype=dtype)
    u0 = torch.rand((S, T, 1), generator=gen, device=device, dtype=dtype) * (1.0 / n_bwd)
    u = u0 + ar / n_bwd
    idx = torch.searchsorted(cumw, u, right=False).clamp(0, N - 1)
    return torch.gather(Xpos, 2, idx), torch.gather(Xrate, 2, idx), \
        torch.full((S, T, n_bwd), 1.0 / n_bwd, device=device, dtype=dtype)

def ffbsm_backward_well(Xpos, Xrate, Wst, DM, z, P, inflate, device, dtype=DTYPE):
    S, T, N = Xpos.shape
    ALPHA = P["mom"]; RN = max(P["vn"] * inflate, 1e-9); PN = max(P["pn"] * inflate, 1e-9)
    sm = torch.empty((S, T), device=device, dtype=dtype); eps = 1e-30
    wb = Wst[:, T - 1, :].clone(); wb = wb / wb.sum(1, keepdim=True).clamp_min(eps)
    sm[:, T - 1] = (wb * (Xpos[:, T - 1, :] - z[T - 1])).sum(1)
    for t in range(T - 2, -1, -1):
        dm = DM[t + 1]
        r_i = Xrate[:, t, :]; p_i = Xpos[:, t, :]
        r_j = Xrate[:, t + 1, :]; p_j = Xpos[:, t + 1, :]
        A = (r_j[:, None, :] - ALPHA * r_i[:, :, None]) / RN
        Bm = (p_j[:, None, :] - p_i[:, :, None] - r_j[:, None, :] * dm) / PN
        logf = -0.5 * (A * A + Bm * Bm)
        m = logf.amax(dim=1, keepdim=True); f = torch.exp(logf - m)
        w_t = Wst[:, t, :]
        vj = (w_t[:, :, None] * f).sum(1).clamp_min(eps)
        ratio = wb / vj
        w_sm = w_t * (f * ratio[:, None, :]).sum(2)
        w_sm = w_sm / w_sm.sum(1, keepdim=True).clamp_min(eps)
        wb = w_sm
        sm[:, t] = (wb * (p_i - z[t])).sum(1)
    return sm

def ensemble_seed(vals_ST, loglik_S, P):
    S = vals_ST.shape[0]; llw = loglik_S.detach().cpu().numpy()
    if np.isfinite(llw).any():
        mx = np.nanmax(llw[np.isfinite(llw)])
        lk = np.where(np.isfinite(llw), llw - mx, -np.inf)
        ww = np.exp(lk / max(P["likelihood_scale"], 1e-6)); s = ww.sum()
        ww = ww / s if s > 1e-300 else np.full(S, 1.0 / S)
    else:
        ww = np.full(S, 1.0 / S)
    ww_t = torch.from_numpy(ww.astype("float32")).to(vals_ST.device)
    return (ww_t[:, None] * vals_ST).sum(0).double().cpu().numpy()

def ffbs_predict(inp, c, gen_bwd):
    """FFBS(clamp,blend) の生予測 TVT (T,) を返す(beam/hold前)。"""
    Xpos, Xrate, Wst, DM, loglik, z = ffbs_forward_well(inp, R0, SEED, N_SEED, DEVICE)
    fwd = ensemble_seed((Wst * (Xpos - z[None, :, None])).sum(2), loglik, R0)
    if N_BWD is not None and N_BWD < Xpos.shape[2]:
        Xpos, Xrate, Wst = reduce_cloud(Xpos, Xrate, Wst, int(N_BWD), gen_bwd, DEVICE)
    sm = ensemble_seed(ffbsm_backward_well(Xpos, Xrate, Wst, DM, z, R0, INFLATE, DEVICE), loglik, R0)
    del Xpos, Xrate, Wst
    if DEVICE == "cuda": torch.cuda.empty_cache()
    if FFBS_CLAMP is not None:
        bad = np.abs(sm - fwd) > FFBS_CLAMP
        if bad.any(): sm = np.where(bad, fwd, sm)
    return (1.0 - FFBS_BLEND) * fwd + FFBS_BLEND * sm


# ----------------------- smoother(固定ラグ) engine -----------------------
def _pf_smoother_core(md, z, gr, valid, grid, glen, vmin, step, gs, ls, ir, P, N, device, gen, dtype=DTYPE):
    B, T = md.shape
    ALPHA = P["mom"]; RN = P["vn"]; PN = P["pn"]; IR = P["init_rate_std"]; IS = P["init_pos_std"]
    RP = P["rp"]; RR = P["rr"]; RESAMP = P["resamp"]; RATE_CLIP = P["rate_clip"]
    GR_POWER = P["gr_power"]; JUMP_PROB = P["jump_prob"]; JUMP_STD = P["jump_std"]; CLIP = P["tvt_clip_margin"]
    PHYS_SIG = P["phys_sig"]; USE_P = P["use_phys"]
    L = int(P.get("smooth_lag", 32)); L = max(0, min(L, T))
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
    use_sm = L > 0
    if use_sm:
        buf = torch.full((B, L, N), float("nan"), device=device, dtype=dtype); ptr = 0
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
                buf_g = torch.gather(buf, 2, idx[:, None, :].expand(B, L, N))
                buf = torch.where(need[:, None, None], buf_g, buf)
        pts_f[:, i] = (w * (pos - zi)).sum(1)
        if use_sm:
            if i >= L:
                old = buf[:, ptr, :]; pts_s[:, i - L] = (w * old).sum(1) - z[:, i - L]
            buf[:, ptr, :] = pos; ptr = (ptr + 1) % L
        pm = torch.where(act, cur, pm)
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

def smoother_predict(inp, P):
    """固定ラグ平滑(clamp,blend) の生予測 TVT (T,) を返す(beam/hold前)。1坑井。"""
    N = int(P["n_particles"]); S = N_SEED; pad = _pad([inp], DEVICE)
    rep = lambda tt: tt.repeat_interleave(S, 0); gen = torch.Generator(device=DEVICE); gen.manual_seed(SEED)
    pf, ps, ll = _pf_smoother_core(
        rep(pad["md"]), rep(pad["z"]), rep(pad["gr"]), rep(pad["valid"]), rep(pad["grid"]),
        pad["glen"].repeat_interleave(S), pad["vmin"].repeat_interleave(S), pad["step"].repeat_interleave(S),
        pad["gs"].repeat_interleave(S), pad["ls"].repeat_interleave(S), pad["ir"].repeat_interleave(S),
        P, N, DEVICE, gen)
    T = len(inp["md"]); pf = pf.view(S, -1).double().cpu().numpy(); ps = ps.view(S, -1).double().cpu().numpy()
    llw = ll.view(S).cpu().numpy()
    if np.isfinite(llw).any():
        mx = np.nanmax(llw[np.isfinite(llw)]); lk = np.where(np.isfinite(llw), llw - mx, -np.inf)
        ww = np.exp(lk / max(P["likelihood_scale"], 1e-6)); s = ww.sum(); ww = ww / s if s > 1e-300 else np.full(S, 1.0 / S)
    else:
        ww = np.full(S, 1.0 / S)
    fwd = (ww[:, None] * pf[:, :T]).sum(0); sm = (ww[:, None] * ps[:, :T]).sum(0)
    if DEVICE == "cuda": torch.cuda.empty_cache()
    if SMOOTH_CLAMP is not None:
        bad = np.abs(sm - fwd) > SMOOTH_CLAMP
        if bad.any(): sm = np.where(bad, fwd, sm)
    return (1.0 - SMOOTH_BLEND) * fwd + SMOOTH_BLEND * sm


# ----------------------- beam + hold + pp -----------------------
@njit(cache=False)
def _beam_jit(sgr, tw_gr, si, BS, mc, es):
    n = len(sgr); nt = len(tw_gr); MAX = BS * 6
    bidx = np.zeros(BS, np.int64); bidx[0] = si
    bcost = np.full(BS, 1e30); bcost[0] = 0.; bn = np.int64(1)
    hI = np.zeros((n, BS), np.int64); hP = np.zeros((n, BS), np.int64)
    cI = np.zeros(MAX, np.int64); cC = np.full(MAX, 1e30); cP = np.zeros(MAX, np.int64)
    for step in range(n):
        gv = sgr[step]; nc = np.int64(0)
        for bi in range(bn):
            idx = bidx[bi]; cost = bcost[bi]
            for d in range(-2, 3):
                ni = idx + d
                if ni < 0 or ni >= nt: continue
                tot = cost + (gv - tw_gr[ni]) ** 2 / es + mc * (d if d >= 0 else -d)
                fnd = np.int64(-1)
                for ci in range(nc):
                    if cI[ci] == ni: fnd = ci; break
                if fnd >= 0:
                    if tot < cC[fnd]: cC[fnd] = tot; cP[fnd] = bi
                else:
                    if nc < MAX: cI[nc] = ni; cC[nc] = tot; cP[nc] = bi; nc += 1
        kept = min(BS, nc)
        for i in range(kept):
            mi = i
            for j in range(i + 1, nc):
                if cC[j] < cC[mi]: mi = j
            if mi != i:
                cI[i], cI[mi] = cI[mi], cI[i]; cC[i], cC[mi] = cC[mi], cC[i]; cP[i], cP[mi] = cP[mi], cP[i]
        hI[step, :kept] = cI[:kept]; hP[step, :kept] = cP[:kept]
        bidx[:kept] = cI[:kept]; bcost[:kept] = cC[:kept]; bn = kept
    best = np.int64(0)
    for b in range(1, bn):
        if bcost[b] < bcost[best]: best = b
    path = np.zeros(n, np.int64); b = best
    for s in range(n - 1, -1, -1): path[s] = hI[s, b]; b = hP[s, b]
    return path

def _nn(arr, val): return int(np.abs(arr - val).argmin())

def run_beam(c):
    tw_tvt = c["tw_tvt"].astype(np.float64)
    fb = float(np.nanmean(c["tw_gr"])) if np.isfinite(np.nanmean(c["tw_gr"])) else 0.0
    tw_gr = np.where(np.isfinite(c["tw_gr"]), c["tw_gr"], fb).astype(np.float64)
    hgr = c["gr_filled"][c["ev_pos"]]; last_tvt = c["last_known_tvt"]
    res = []
    for (bs, mc, es, r) in BEAM_CONFIGS:
        si = _nn(tw_tvt, last_tvt); sgr = _smooth(hgr, float(np.nanmean(tw_gr)), r).astype(np.float64)
        path = _beam_jit(sgr, tw_gr, si, int(bs), float(mc), float(es)); res.append(tw_tvt[path].astype(np.float32))
    return np.stack(res, 0).mean(0).astype(np.float64)

def finalize(pred, c, code, beam):
    """生予測 → カテゴリ別 beam blend → hold → apply_pp。"""
    pred = np.asarray(pred, dtype=np.float64)
    bw = BEAM_W.get(code, 0.0); hwt = HOLD_W.get(code, 0.0)
    if bw > 0 and beam is not None:
        pred = pred.copy(); m = min(len(pred), len(beam)); pred[:m] = (1.0 - bw) * pred[:m] + bw * beam[:m]
    if hwt > 0:
        pred = (1.0 - hwt) * pred + hwt * c["last_known_tvt"]
    if APPLY_PP:
        md_since = np.maximum(c["md"] - c["last_known_md"], 0.0)
        d = (pred - c["last_known_tvt"]) * (1.0 - np.exp(-md_since / PP_TAU)) * PP_ALPHA
        pred = c["last_known_tvt"] + d
        n = len(pred); wl = min(SG_W, n)
        if wl % 2 == 0: wl -= 1
        if wl >= SG_P + 2: pred = savgol_filter(pred, wl, SG_P)
    return pred


def _sse(err):
    e = err[np.isfinite(err)]; return float(np.sum(e ** 2)), int(e.size)


def main():
    print(f"device={DEVICE} n_seed={N_SEED} smoother(lag={SMOOTH_LAG}) vs FFBS(inflate={INFLATE}) +beam+hold")
    wells = list_wells()
    if N_WELLS: wells = wells[:N_WELLS]
    CACHE = {}
    for w in wells:
        c = build_cache(w)
        if c is not None and c["tvt_true"] is not None:
            CACHE[w] = c
    aw = [w for w in wells if w in CACHE]
    print(f"usable wells: {len(aw)}")

    Ps = {**R0, "smooth_lag": SMOOTH_LAG}
    gen_bwd = torch.Generator(device=DEVICE); gen_bwd.manual_seed(SEED + 987654321)
    acc = {"smoother": [0.0, 0], "ffbs": [0.0, 0], "oracle": [0.0, 0]}
    win = {"smoother": 0, "ffbs": 0}; disagree = []
    n = len(aw); t0 = time.time()
    for k, wid in enumerate(aw):
        c = CACHE[wid]; true = c["tvt_true"]; inp = make_inputs(c, Ps); code = well_code(c)
        beam = run_beam(c) if BEAM_W.get(code, 0.0) > 0 else None
        sm_pred = finalize(smoother_predict(inp, Ps), c, code, beam)
        ff_pred = finalize(ffbs_predict(inp, c, gen_bwd), c, code, beam)
        es_s, cs = _sse(sm_pred - true); es_f, cf = _sse(ff_pred - true)
        acc["smoother"][0] += es_s; acc["smoother"][1] += cs
        acc["ffbs"][0] += es_f; acc["ffbs"][1] += cf
        # oracle: 坑井単位で SSE 小さい方
        if es_s <= es_f:
            acc["oracle"][0] += es_s; acc["oracle"][1] += cs; win["smoother"] += 1
        else:
            acc["oracle"][0] += es_f; acc["oracle"][1] += cf; win["ffbs"] += 1
        m = min(len(sm_pred), len(ff_pred)); disagree.append(float(np.nanmean(np.abs(sm_pred[:m] - ff_pred[:m]))))
        if (k + 1) % 10 == 0 or k + 1 == n:
            rs = math.sqrt(acc["smoother"][0] / acc["smoother"][1]); rf = math.sqrt(acc["ffbs"][0] / acc["ffbs"][1])
            ro = math.sqrt(acc["oracle"][0] / acc["oracle"][1])
            el = time.time() - t0; eta = el / (k + 1) * (n - (k + 1))
            print(f"[{k+1}/{n}] smoother={rs:.3f} ffbs={rf:.3f} oracle={ro:.3f} "
                  f"(win s/f={win['smoother']}/{win['ffbs']}) | {el:.0f}s ETA {eta/60:.0f}m", flush=True)

    rs = math.sqrt(acc["smoother"][0] / acc["smoother"][1]); rf = math.sqrt(acc["ffbs"][0] / acc["ffbs"][1])
    ro = math.sqrt(acc["oracle"][0] / acc["oracle"][1]); mn = min(rs, rf)
    dis = float(np.nanmean(disagree))
    print("\n===== 相補性(全行プールRMSE) =====")
    print(f"  smoother+beam+hold = {rs:.4f}")
    print(f"  ffbs+beam+hold     = {rf:.4f}")
    print(f"  oracle(坑井選択完璧) = {ro:.4f}   (Δ vs min単体 {ro-mn:+.4f})")
    print(f"  勝ち坑井 smoother/ffbs = {win['smoother']}/{win['ffbs']}   予測不一致 mean|s-f| = {dis:.2f} ft")
    verdict = "相補=合成に伸びしろ大" if ro < mn - 0.05 else ("やや相補" if ro < mn - 0.01 else "ほぼ冗長=合成しても伸びない")
    print(f"  判定: {verdict}")
    OUT_PATH.write_text(json.dumps(dict(
        n_seed=N_SEED, n_wells=len(aw), smooth_lag=SMOOTH_LAG, inflate=INFLATE,
        smoother_overall=rs, ffbs_overall=rf, oracle_overall=ro,
        wins=win, mean_disagree_ft=dis, verdict=verdict,
        note="oracle << min(smoother,ffbs) なら相補=mode-aware合成/選択で伸びる。ほぼ同なら冗長。",
    ), indent=2))
    print(f"saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
