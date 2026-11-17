"""
v25 診断: FFBS の上に「今のヒューリスティック後処理」を載せる意味があるか
================================================================================
問い: FFBS(r0固定, inflate=32, clamp, blend=1) が良い平滑軌跡を出す前提で、
      今の提出ヒューリスティック(apply_pp = tauランプ + sg_smooth, hold 収縮)を
      **その上に重ねると overall RMSE が改善するか / 冗長 or 有害か** を train(真値)で測る。

比較する予測(すべて全行プール overall RMSE):
  forward           … r0 素
  ffbs              … r0 + FFBS(基準)
  ffbs+sg           … FFBS に savgol だけ
  ffbs+ramp         … FFBS に tauランプだけ
  ffbs+pp           … FFBS に apply_pp(ランプ→sg) 全部
  ffbs+hold{05,10,15} … FFBS を last_known_tvt へ収縮(hold)
  fwd+pp            … 参考: 素forward に apply_pp(7.098ヒューリスティックに近い側)

読み方: 'ffbs' より下がる後処理があれば「意味あり」、'ffbs' が最小なら「FFBSが後処理を不要にした」。

出力: v25/artifacts_v25/heuristic_on_ffbs.json
実行: rogii_claude/.venv/Scripts/python.exe v25/check_heuristic_on_ffbs_v25.py
※ コストは FFBS 1回/坑井(+ 後処理は解析的に軽い)。まず N_WELLS で絞ってよい。
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
N_SEED   = 128
N_WELLS  = None          # None=全773。まず 200 等で絞ってもよい
INFLATE  = 32.0
N_BWD    = 128
CLAMP_FT = 60.0
BLEND    = 1.0           # 最終 ffbs = (1-BLEND)*fwd + BLEND*smoothed

# 後処理パラメータ(提出ノートと同一)
PP_TAU   = 85.0
PP_ALPHA = 1.0
SG_W, SG_P = 17, 3
HOLD_WS  = [0.05, 0.10, 0.15]          # 一律hold比較用(参考)

# --- 本物のヒューリスティック後処理(tuned-heuristic-only-cat6.ipynb cell3 と同一) ---
N_EVAL_THRESHOLD  = 4840.0
Z_SPAN_THRESHOLDS = (136.73000000000016, 185.5133333333342)
BEAM_W = {0: 0.0, 1: 0.0, 2: 0.2, 3: 0.0, 4: 0.05, 5: 0.2}   # PFとbeamのブレンド(codes 2,4,5)
HOLD_W = {0: 0.2, 1: 0.15, 2: 0.15, 3: 0.15, 4: 0.05, 5: 0.05}  # last_known収縮(カテゴリ別)
BEAM_CONFIGS = [
    (10, 20.0, 144.0, 2), (10, 8.0, 64.0, 2), (8, 35.0, 220.0, 1), (10, 14.0, 90.0, 5),
    (20, 4.0, 36.0, 3), (12, 12.0, 100.0, 3), (15, 25.0, 180.0, 2), (20, 30.0, 200.0, 2),
    (15, 10.0, 80.0, 4), (25, 6.0, 50.0, 3), (10, 40.0, 300.0, 1), (12, 18.0, 120.0, 5),
    (30, 8.0, 70.0, 2), (10, 50.0, 400.0, 0),
]

OUT_PATH = OUT_DIR / "heuristic_on_ffbs.json"

R0_PARAMS = {
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


# ----------------------- データ(自己完結。last_known を追加) -----------------------
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
    return dict(md=ev["MD"].values.astype(np.float64), z=z_ev,
                gr_filled=gr_filled, ev_pos=ev_pos, kpos=kpos,
                kn_tvtin=kn["TVT_input"].values.astype(float),
                tw_tvt=tw_tvt, tw_gr=tw_gr, tw_fb=tw_fb, ls=ls, ir=ir, tvt_true=tv,
                n_eval=n_eval, z_span=z_span,
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


# ----------------------- FFBS engine(tune_ffbsi_r0_v25.py と同一) -----------------------
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


# ----------------------- beam search(7.191移植, numba, 提出ノートと同一) -----------------------
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

def beam_search(gr_h, tw_tvt, tw_gr, start_tvt, bs, mc, es, r):
    si = _nn(tw_tvt, start_tvt)
    sgr = _smooth(gr_h, float(np.nanmean(tw_gr)), r).astype(np.float64)
    path = _beam_jit(sgr, tw_gr.astype(np.float64), si, int(bs), float(mc), float(es))
    return tw_tvt[path].astype(np.float32)

def run_beam(c):
    tw_tvt = c["tw_tvt"].astype(np.float64)
    fb = float(np.nanmean(c["tw_gr"])) if np.isfinite(np.nanmean(c["tw_gr"])) else 0.0
    tw_gr = np.where(np.isfinite(c["tw_gr"]), c["tw_gr"], fb).astype(np.float64)
    hgr = c["gr_filled"][c["ev_pos"]]; last_tvt = c["last_known_tvt"]
    res = [beam_search(hgr, tw_tvt, tw_gr, last_tvt, bs, mc, es, r) for (bs, mc, es, r) in BEAM_CONFIGS]
    return np.stack(res, 0).mean(0).astype(np.float64)

def blend_beam(pred, beam, bw):
    if bw <= 0: return pred
    q = np.asarray(pred, dtype=np.float64).copy(); m = min(len(q), len(beam))
    q[:m] = (1.0 - bw) * q[:m] + bw * beam[:m]
    return q


# ----------------------- 後処理(提出ノートと同一) -----------------------
def apply_ramp(pred, c, tau=PP_TAU, alpha=PP_ALPHA):
    md_since = np.maximum(c["md"] - c["last_known_md"], 0.0)
    d = (pred - c["last_known_tvt"]) * (1.0 - np.exp(-md_since / tau)) * alpha
    return c["last_known_tvt"] + d

def apply_sg(pred):
    out = np.asarray(pred, dtype=np.float64); n = len(out); wl = min(SG_W, n)
    if wl % 2 == 0: wl -= 1
    return savgol_filter(out, wl, SG_P) if wl >= SG_P + 2 else out

def apply_hold(pred, c, hw):
    return (1.0 - hw) * pred + hw * c["last_known_tvt"]


def _rmse(errs):
    e = np.concatenate(errs); e = e[np.isfinite(e)]
    return float(np.sqrt(np.mean(e ** 2))) if e.size else 1e3


def main():
    print(f"device={DEVICE} PF=r0_seed32 n_seed={N_SEED} inflate={INFLATE} N_BWD={N_BWD} clamp={CLAMP_FT} blend={BLEND}")
    wells = list_wells()
    if N_WELLS: wells = wells[:N_WELLS]
    CACHE = {}
    for w in wells:
        c = build_cache(w)
        if c is not None and c["tvt_true"] is not None:
            CACHE[w] = c
    aw = [w for w in wells if w in CACHE]
    print(f"usable wells: {len(aw)}")

    P = {**R0_PARAMS}; gen_bwd = torch.Generator(device=DEVICE); gen_bwd.manual_seed(SEED + 987654321)
    names = ["forward", "ffbs", "ffbs+pp",
             "ffbs+beam", "ffbs+holdW", "ffbs+beam+holdW", "ffbs+beam+holdW+pp",
             "fwd+beam+holdW", "fwd+beam+holdW+pp"] \
            + [f"ffbs+hold{int(h*100):02d}" for h in HOLD_WS]
    E = {k: [] for k in names}
    n = len(aw); t0 = time.time()
    for k, wid in enumerate(aw):
        c = CACHE[wid]; true = c["tvt_true"]; inp = make_inputs(c, P)
        Xpos, Xrate, Wst, DM, loglik, z = ffbs_forward_well(inp, P, SEED, N_SEED, DEVICE)
        fwd = ensemble_seed((Wst * (Xpos - z[None, :, None])).sum(2), loglik, P)
        if N_BWD is not None and N_BWD < Xpos.shape[2]:
            Xpos, Xrate, Wst = reduce_cloud(Xpos, Xrate, Wst, int(N_BWD), gen_bwd, DEVICE)
        sm = ensemble_seed(ffbsm_backward_well(Xpos, Xrate, Wst, DM, z, P, INFLATE, DEVICE), loglik, P)
        if CLAMP_FT is not None:
            bad = np.abs(sm - fwd) > CLAMP_FT
            if bad.any(): sm = np.where(bad, fwd, sm)
        ffbs = (1.0 - BLEND) * fwd + BLEND * sm
        del Xpos, Xrate, Wst
        if DEVICE == "cuda": torch.cuda.empty_cache()
        # --- 本物のヒューリスティック後処理(カテゴリ別 beam/hold) ---
        code = well_code(c); bw = BEAM_W.get(code, 0.0); hwt = HOLD_W.get(code, 0.0)
        beam = run_beam(c) if bw > 0 else None
        ffbs_b = blend_beam(ffbs, beam, bw) if bw > 0 else ffbs        # ffbs + per-cat beam
        ffbs_bh = apply_hold(ffbs_b, c, hwt)                          # + per-cat hold
        fwd_b = blend_beam(fwd, beam, bw) if bw > 0 else fwd
        fwd_bh = apply_hold(fwd_b, c, hwt)
        # --- 予測バリアント ---
        preds = {
            "forward":              fwd,
            "ffbs":                 ffbs,
            "ffbs+pp":              apply_sg(apply_ramp(ffbs, c)),
            "ffbs+beam":            ffbs_b,
            "ffbs+holdW":           apply_hold(ffbs, c, hwt),
            "ffbs+beam+holdW":      ffbs_bh,
            "ffbs+beam+holdW+pp":   apply_sg(apply_ramp(ffbs_bh, c)),
            "fwd+beam+holdW":       fwd_bh,                            # ★ベースライン(forward)にbeam+hold
            "fwd+beam+holdW+pp":    apply_sg(apply_ramp(fwd_bh, c)),   # ≈7.098側(per-cat scale除く)
        }
        for h in HOLD_WS:
            preds[f"ffbs+hold{int(h*100):02d}"] = apply_hold(ffbs, c, h)
        for name in names:
            E[name].append(preds[name] - true)
        # --- 逐次(1件ずつ)表示: 素 vs beam+hold を forward/ffbs で並べる ---
        run = {nm: _rmse(E[nm]) for nm in ["forward", "ffbs", "fwd+beam+holdW", "ffbs+beam+holdW"]}
        el = time.time() - t0; eta = el / (k + 1) * (n - (k + 1))
        print(f"[{k+1}/{n}] {c['tvt_true'].shape[0] if hasattr(c['tvt_true'],'shape') else 0}r "
              f"| fwd={run['forward']:.3f} ffbs={run['ffbs']:.3f} "
              f"fwd+bh={run['fwd+beam+holdW']:.3f} ffbs+bh={run['ffbs+beam+holdW']:.3f} "
              f"| {el:.0f}s ETA {eta/60:.0f}m", flush=True)

    print("\n===== 最終 overall(全行プールRMSE) =====")
    res = {nm: _rmse(E[nm]) for nm in names}
    base = res["ffbs"]
    for nm in names:
        tag = "  <= 基準" if nm == "ffbs" else f"  (Δ vs ffbs {res[nm]-base:+.4f})"
        print(f"  {nm:14s} = {res[nm]:.4f}{tag}")
    best = min(names, key=lambda x: res[x])
    print(f"\n最良 = {best} ({res[best]:.4f})")
    if best == "ffbs":
        print("→ FFBS が後処理を不要にした(ヒューリスティックは載せる意味なし/むしろ有害)")
    else:
        print(f"→ '{best}' が FFBS単体より良い=その後処理は載せる意味あり")
    OUT_PATH.write_text(json.dumps(dict(
        n_seed=N_SEED, inflate=INFLATE, blend=BLEND, n_bwd=N_BWD, clamp_ft=CLAMP_FT,
        pp={"tau": PP_TAU, "alpha": PP_ALPHA, "sg": [SG_W, SG_P]}, hold_ws=HOLD_WS,
        n_wells=len(aw), overall=res, best=best,
        note="'ffbs'が最小ならヒューリスティック後処理は不要。下回る後処理があればそれは載せる価値あり。",
    ), indent=2))
    print(f"saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
