"""
v25 診断: forward / smooth PF を1回ずつ入力し、ノートの各段を通した値の推移(各段RMSE)を見る
================================================================================
線形パイプライン(PFが各段を流れる):
  s0 PF-selector            … r0_seed32 の forward / smooth を multi-scale → 6分割variant(scale/beam/hold)
  s1 + grid FB平滑(構造無)   … grid 前向き後ろ向き平滑。**前段(s0)をdrift priorに**、w_struct=0(linear/GRのみ)
  s2 + 構造prior            … 同上で prior に FormationPlaneKNN構造(近傍dip面)を w_struct で混ぜる
  s3 + PP                   … apply_pp(tauランプ + sg)
各段の後で overall RMSE を出す。forward入力 と smooth入力 を並べて「smoothが各段で良いまま来るか」を見る。

出力: v25/artifacts_v25/pipeline_stages.json
実行: rogii_claude/.venv/Scripts/python.exe v25/check_pipeline_stages_v25.py
※ FormationPlaneKNN は train全坑井の X,Y + formation列(ANCC..BUDA)から近傍dip面を作る。
  クエリ坑井は self 除外(リーク防止)。grid平滑は坑井ごとnumpy(gaussian_filter1d)。
"""

import time, json, glob, math, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree
from numba import njit

warnings.filterwarnings("ignore")
np.seterr(all="ignore")

# ============================== CONFIG ==============================
PROJECT   = Path(r"C:\Users\kosaka256\Documents\rogii_claude")
DATA_DIR  = PROJECT / "rogii-wellbore-geology-prediction" / "train"
OUT_DIR   = PROJECT / "v25" / "artifacts_v25"; OUT_DIR.mkdir(parents=True, exist_ok=True)
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE     = torch.float32

SEED       = 0
N_SEED     = 32
N_WELLS    = None        # None=全773。まず 200 等で絞ってよい
SMOOTH_LAG = 192
CHUNK      = 24

# selector(cell4 と同一)
N_EVAL_THRESHOLD  = 4840.0
Z_SPAN_THRESHOLDS = (136.73000000000016, 185.5133333333342)
SELECTOR_SCALES   = (3.0, 5.0, 8.0, 12.0)
SELECTOR_BIN_VARIANTS = {
    0: 'pf_scale_5_hold_0.2', 1: 'pf_scale_3_hold_0.15', 2: 'pf_scale_12_beam_0.2_hold_0.15',
    3: 'pf_scale_5_hold_0.15', 4: 'pf_scale_5_beam_0.05_hold_0.05', 5: 'pf_scale_12_beam_0.2_hold_0.05',
}
SELECTOR_GLOBAL_VARIANT = 'pf_scale_8_hold_0.2'
BEAM_CONFIGS = [
    (10, 20.0, 144.0, 2), (10, 8.0, 64.0, 2), (8, 35.0, 220.0, 1), (10, 14.0, 90.0, 5),
    (20, 4.0, 36.0, 3), (12, 12.0, 100.0, 3), (15, 25.0, 180.0, 2), (20, 30.0, 200.0, 2),
    (15, 10.0, 80.0, 4), (25, 6.0, 50.0, 3), (10, 40.0, 300.0, 1), (12, 18.0, 120.0, 5),
    (30, 8.0, 70.0, 2), (10, 50.0, 400.0, 0),
]
# 構造prior / grid FB平滑器(cell31/41 と同一系)
FORMATIONS = ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]
PLANE_K    = 10
W_STRUCT   = 0.5         # s2 の構造priorブレンド比(eval行のみ。1.0はPFを完全に消すので0.3-0.5推奨)
GRID_HALF  = 50.0; GRID_STEP = 0.5; Q_WALK = 0.35
GR_SIG_MIN = 8.0; GR_SIG_MAX = 60.0; GR_SIG_DEF = 30.0
PIN_SIG    = 0.5; LL_CAP = 80.0; TAIL = 50
# PP
PP_TAU = 85.0; PP_ALPHA = 1.0; SG_W, SG_P = 17, 3

OUT_PATH = OUT_DIR / "pipeline_stages.json"

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
    gr_full = hw["GR"].astype(float).interpolate(limit_direction="both").fillna(tw_fb).to_numpy(float)
    ev_pos = hw.index.get_indexer(ev.index); kpos = hw.index.get_indexer(kn.index)
    ls = float(kn["TVT_input"].iloc[-1] + kn["Z"].iloc[-1])
    tail = kn.tail(30); dt = np.diff(tail["TVT_input"].values); dz = np.diff(tail["Z"].values)
    dm = np.diff(tail["MD"].values); m = dm > 0
    ir = float(np.median((dt + dz)[m] / dm[m])) if m.sum() >= 3 else 0.
    tv = ev["TVT"].values.astype(np.float64) if "TVT" in ev.columns else None
    z_ev = ev["Z"].values.astype(np.float64)
    n_eval = int(len(z_ev)); z_span = float(np.nanmax(z_ev) - np.nanmin(z_ev)) if len(z_ev) else 0.0
    md_full = hw["MD"].to_numpy(float); z_full = hw["Z"].to_numpy(float)
    ti_full = hw["TVT_input"].to_numpy(float); known = np.isfinite(ti_full)
    x_full = hw["X"].to_numpy(float); y_full = hw["Y"].to_numpy(float)
    return dict(md=ev["MD"].values.astype(np.float64), z=z_ev, gr_filled=gr_full, ev_pos=ev_pos, kpos=kpos,
                kn_tvtin=kn["TVT_input"].values.astype(float), tw_tvt=tw_tvt, tw_gr=tw_gr, tw_fb=tw_fb,
                ls=ls, ir=ir, tvt_true=tv, n_eval=n_eval, z_span=z_span,
                last_known_md=float(kn["MD"].iloc[-1]), last_known_tvt=float(kn["TVT_input"].iloc[-1]),
                md_full=md_full, z_full=z_full, ti_full=ti_full, gr_full_arr=gr_full,
                x_full=x_full, y_full=y_full, known=known, ev_index=ev_pos)

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


# ---------- 固定ラグ系統スムーザー(tune_smoother_r0_v25 と同一) + multi-scale ----------
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

def run_smoother_scales(inps, P, seed, n_seeds, chunk, scales):
    N = int(P["n_particles"]); S = n_seeds; W = len(inps); out = [None] * W
    for c0 in range(0, W, chunk):
        sub = inps[c0:c0 + chunk]; w = len(sub); pad = _pad(sub, DEVICE)
        rep = lambda tt: tt.repeat_interleave(S, 0); gen = torch.Generator(device=DEVICE); gen.manual_seed(seed)
        pf, ps, ll = _pf_smoother_core(
            rep(pad["md"]), rep(pad["z"]), rep(pad["gr"]), rep(pad["valid"]), rep(pad["grid"]),
            pad["glen"].repeat_interleave(S), pad["vmin"].repeat_interleave(S), pad["step"].repeat_interleave(S),
            pad["gs"].repeat_interleave(S), pad["ls"].repeat_interleave(S), pad["ir"].repeat_interleave(S),
            P, N, DEVICE, gen)
        pf = pf.view(w, S, -1).double().cpu().numpy(); ps = ps.view(w, S, -1).double().cpu().numpy()
        ll = ll.view(w, S).cpu().numpy()
        for j in range(w):
            T = len(sub[j]["md"]); llw = ll[j]; d = {}
            for sc in scales:
                if np.isfinite(llw).any():
                    mx = np.nanmax(llw[np.isfinite(llw)])
                    ww = np.exp(np.where(np.isfinite(llw), llw - mx, -np.inf) / max(sc, 1e-6))
                    ww = ww / ww.sum() if ww.sum() > 1e-300 else np.full(S, 1.0 / S)
                else:
                    ww = np.full(S, 1.0 / S)
                d[sc] = ((ww[:, None] * pf[j, :, :T]).sum(0), (ww[:, None] * ps[j, :, :T]).sum(0))
            out[c0 + j] = d
    if DEVICE == "cuda": torch.cuda.synchronize()
    return out


# ---------- beam(cell4) ----------
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
            for dmv in range(-2, 3):
                ni = idx + dmv
                if ni < 0 or ni >= nt: continue
                tot = cost + (gv - tw_gr[ni]) ** 2 / es + mc * (dmv if dmv >= 0 else -dmv)
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
        res.append(tw_tvt[_beam_jit(sgr, tw_gr, si, int(bs), float(mc), float(es))].astype(np.float32))
    return np.stack(res, 0).mean(0).astype(np.float64)

def parse_variant(name):
    parts = name.split('_'); scale = float(parts[2]); bw = hw_ = 0.0
    if 'beam' in parts: bw = float(parts[parts.index('beam') + 1])
    if 'hold' in parts: hw_ = float(parts[parts.index('hold') + 1])
    return scale, bw, hw_

def apply_selector(pf_by_scale, variant, beam, last_known):
    scale, bw, hw_ = parse_variant(variant)
    pred = (1.0 - bw) * pf_by_scale.get(scale, next(iter(pf_by_scale.values())))
    if bw > 0 and beam is not None:
        m = min(len(pred), len(beam)); pred = pred.copy(); pred[:m] = pred[:m] + bw * beam[:m]
    return (1.0 - hw_) * pred + hw_ * last_known


# ---------- 構造prior FormationPlaneKNN(cell31 と同一系) ----------
class FormationPlaneKNN:
    def __init__(self, well_ids, data_dir):
        rows = []
        for wid in well_ids:
            try:
                df = pd.read_csv(data_dir / f"{wid}__horizontal_well.csv", usecols=["X", "Y"] + FORMATIONS).dropna()
            except Exception:
                continue
            if len(df) == 0: continue
            row = {"wid": wid, "x": float(df.X.median()), "y": float(df.Y.median())}
            for c in FORMATIONS: row[f"{c}_m"] = float(df[c].median())
            rows.append(row)
        self.df = pd.DataFrame(rows); self.wmap = {w: i for i, w in enumerate(self.df.wid)}
        xy = self.df[["x", "y"]].to_numpy(); self.scale = np.where(xy.std(0) < 1e-3, 1., xy.std(0))
        self.tree = cKDTree(xy / self.scale); self.xa = self.df.x.to_numpy(); self.ya = self.df.y.to_numpy()
        self.fa = self.df[[f"{c}_m" for c in FORMATIONS]].to_numpy(np.float64)
    def impute(self, xy_q, self_wid=None, k=PLANE_K):
        q = xy_q / self.scale; nf = min(k + 5, len(self.df)); dist, idx = self.tree.query(q, k=nf, workers=-1)
        if self_wid in self.wmap: dist = np.where(idx == self.wmap[self_wid], np.inf, dist)
        ordr = np.argpartition(dist, min(k - 1, nf - 1), 1)[:, :k]
        dk = np.take_along_axis(dist, ordr, 1); ik = np.take_along_axis(idx, ordr, 1)
        vk = np.isfinite(dk); w = np.where(vk, 1. / (dk + 1e-3), 0.).astype(np.float64)
        xn = self.xa[ik]; yn = self.ya[ik]; fn = self.fa[ik]; wx = w * xn; wy = w * yn
        A = np.zeros((len(q), 3, 3))
        A[:, 0, 0] = (wx * xn).sum(1); A[:, 0, 1] = (wx * yn).sum(1); A[:, 0, 2] = wx.sum(1)
        A[:, 1, 0] = A[:, 0, 1]; A[:, 1, 1] = (wy * yn).sum(1); A[:, 1, 2] = wy.sum(1)
        A[:, 2, 0] = A[:, 0, 2]; A[:, 2, 1] = A[:, 1, 2]; A[:, 2, 2] = w.sum(1)
        A[:, 0, 0] += 1e-9; A[:, 1, 1] += 1e-9; A[:, 2, 2] += 1e-9
        rhs = np.stack([(wx[:, :, None] * fn).sum(1), (wy[:, :, None] * fn).sum(1), (w[:, :, None] * fn).sum(1)], 1)
        try:
            coef = np.linalg.solve(A, rhs)
        except Exception:
            coef = np.zeros((len(q), 3, len(FORMATIONS)))
            for r in range(len(q)):
                try: coef[r] = np.linalg.pinv(A[r]) @ rhs[r]
                except Exception: pass
        Xq = xy_q[:, 0]; Yq = xy_q[:, 1]
        pred = (Xq[:, None] * coef[:, 0, :] + Yq[:, None] * coef[:, 1, :] + coef[:, 2, :]).astype(np.float32)
        pred[~vk.any(1)] = self.fa.mean(0)
        return pred

_FI = None

def robust_slope(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float); m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 2 or np.std(x[m]) < 1e-6: return 0.
    return float(np.polyfit(x[m], y[m], 1)[0])

def _struct_u_prior(c, self_wid):
    """U=TVT+Z の構造prior(近傍dip面 + formation別坑井offset), full長。無効なら None。"""
    if _FI is None: return None
    try:
        xy = np.column_stack([c["x_full"], c["y_full"]])
        if not np.isfinite(xy).all(): return None
        form_all = _FI.impute(xy, self_wid=self_wid)              # (T, n_form)
        known = c["known"]; U_known = c["ti_full"][known] + c["z_full"][known]
        Us = []
        for f in range(form_all.shape[1]):
            fv = form_all[:, f].astype(float)
            if not np.isfinite(fv[known]).any(): continue
            bk = U_known - fv[known]; bk = bk[np.isfinite(bk)]
            if len(bk) == 0: continue
            b_f = float(np.median(bk))
            if np.isfinite(b_f): Us.append(fv + b_f)
        if not Us: return None
        U = np.nanmedian(np.stack(Us, 0), axis=0)
        return U if np.isfinite(U).any() else None
    except Exception:
        return None


# ---------- grid FB平滑器(cell41 と同一系, prior=前段PFを流す) ----------
def _dense_gr(tw_tvt, tw_gr, pad=250.0, step=0.1):
    lo = float(np.min(tw_tvt)) - pad; hi = float(np.max(tw_tvt)) + pad
    grid = np.arange(lo, hi + step, step)
    return np.interp(grid, tw_tvt, tw_gr).astype(np.float32), lo, step

def _gr_at(gg, lo, step, tvt):
    idx = ((tvt - lo) / step).astype(np.int64); np.clip(idx, 0, len(gg) - 1, out=idx); return gg[idx]

def grid_fb(c, w_struct, U_struct):
    """
    ★cell41 bayes_smoother_well 忠実移植。PFとは独立。
    drift prior = linear(ヒールdip外挿 last_U + ir*(MD-last_MD)) [+ w_struct*構造(近傍dip面)]。
    grid 前向き後ろ向き平滑 → eval行の平滑後 TVT。防御的: 例外/NaN時は last_known フォールバック。
    """
    md = c["md_full"]; Z = c["z_full"]; ti = c["ti_full"]; gr = c["gr_full_arr"]
    known = c["known"]; T = len(md); ev_pos = c["ev_index"]; lastk = c["last_known_tvt"]
    tw_tvt = c["tw_tvt"]
    m_tw = float(np.nanmean(c["tw_gr"])) if np.isfinite(np.nanmean(c["tw_gr"])) else 0.0
    tw_gr = np.where(np.isfinite(c["tw_gr"]), c["tw_gr"], m_tw).astype(float)
    fb = np.full(len(ev_pos), lastk, float)                        # フォールバック=last_known
    if len(tw_tvt) < 3 or T < 3 or known.sum() < 5:
        return fb
    try:
        # --- linear drift prior on U = TVT+Z (cell41) ---
        kpos = np.where(known)[0]; last = kpos[-1]
        last_U = ti[last] + Z[last]; last_MD = md[last]
        tk = kpos[-min(TAIL, len(kpos)):]
        ir = robust_slope(md[tk], ti[tk] + Z[tk])                  # dU/dMD = prefix tail の傾き
        u_prior = last_U + ir * (md - last_MD)
        if w_struct > 0 and U_struct is not None:
            ok = np.isfinite(U_struct); u_prior = u_prior.copy()
            u_prior[ok] = (1.0 - w_struct) * u_prior[ok] + w_struct * U_struct[ok]
        u_prior = np.where(np.isfinite(u_prior), u_prior, last_U)
        # --- GR emission noise(既知prefix残差) ---
        res = gr[known] - np.interp(ti[known], tw_tvt, tw_gr); res = res[np.isfinite(res)]
        gr_sig = float(np.clip(np.std(res) if res.size > 20 else GR_SIG_DEF, GR_SIG_MIN, GR_SIG_MAX))
        if not np.isfinite(gr_sig) or gr_sig <= 0: gr_sig = GR_SIG_DEF
        rr = np.arange(-GRID_HALF, GRID_HALF + GRID_STEP, GRID_STEP).astype(np.float32); K = len(rr)
        gg, glo, gstep = _dense_gr(tw_tvt, tw_gr)
        tvt_cells = (u_prior[:, None] + rr[None, :]) - Z[:, None]
        gr_pred = _gr_at(gg, glo, gstep, tvt_cells)
        resid = (gr[:, None] - gr_pred) / gr_sig
        ll = (-0.5 * np.minimum(resid * resid, LL_CAP)).astype(np.float32)
        kp = (tvt_cells - ti[:, None]) / PIN_SIG
        ll_known = (-0.5 * np.minimum(kp * kp, 1e6)).astype(np.float32)
        ll = np.where(known[:, None], ll_known, ll)
        ll = np.where(np.isfinite(ll), ll, -1e30)
        E = np.exp(ll - ll.max(1, keepdims=True)); np.maximum(E, 1e-30, out=E)
        dm = np.empty(T); dm[1:] = np.maximum(md[1:] - md[:-1], 1e-3); dm[0] = dm[1] if T > 1 else 1.0
        sig_cells = np.clip(Q_WALK * np.sqrt(np.maximum(dm, 1e-6)) / GRID_STEP, 1e-3, K / 3.0)
        A = np.empty((T, K), np.float32); a = E[0] / E[0].sum(); A[0] = a
        for t in range(1, T):
            a = gaussian_filter1d(a, float(sig_cells[t]), mode="constant"); a = a * E[t]; s = a.sum()
            a = a / s if s > 0 else np.full(K, 1.0 / K, np.float32); A[t] = a
        B = np.empty((T, K), np.float32); b = np.full(K, 1.0 / K, np.float32); B[T - 1] = b
        for t in range(T - 2, -1, -1):
            bb = b * E[t + 1]; bb = gaussian_filter1d(bb, float(sig_cells[t + 1]), mode="constant")
            s = bb.sum(); b = bb / s if s > 0 else np.full(K, 1.0 / K, np.float32); B[t] = b
        G = A * B; gsum = G.sum(1, keepdims=True); G = np.where(gsum > 0, G / np.maximum(gsum, 1e-30), 1.0 / K)
        r_hat = G @ rr
        tvt_hat = (u_prior + r_hat) - Z; tvt_hat[known] = ti[known]
        out = tvt_hat[ev_pos]
        return np.where(np.isfinite(out), out, fb)
    except Exception:
        return fb


def apply_pp(pred, c):
    pred = np.asarray(pred, dtype=np.float64)
    md_since = np.maximum(c["md"] - c["last_known_md"], 0.0)
    d = (pred - c["last_known_tvt"]) * (1.0 - np.exp(-md_since / PP_TAU)) * PP_ALPHA
    out = c["last_known_tvt"] + d
    n = len(out); wl = min(SG_W, n)
    if wl % 2 == 0: wl -= 1
    return savgol_filter(out, wl, SG_P) if wl >= SG_P + 2 else out


def _sse(err):
    e = err[np.isfinite(err)]; return float(np.sum(e ** 2)), int(e.size)
def _rmse(a): return math.sqrt(a[0] / a[1]) if a[1] else float("nan")


def main():
    global _FI
    print(f"device={DEVICE} PF=r0_seed32 smooth_lag={SMOOTH_LAG} n_seed={N_SEED} w_struct={W_STRUCT}")
    all_wids = list_wells()
    wells = all_wids[:N_WELLS] if N_WELLS else all_wids
    print("building FormationPlaneKNN on all train wells ...")
    _FI = FormationPlaneKNN(all_wids, DATA_DIR)                    # 近傍dip面は全train坑井(self除外で使う)
    CACHE = {}
    for w in wells:
        c = build_cache(w)
        if c is not None and c["tvt_true"] is not None:
            CACHE[w] = c
    aw = [w for w in wells if w in CACHE]
    print(f"usable wells: {len(aw)}")

    P = {**R0, "smooth_lag": SMOOTH_LAG}
    BLEND_WS = [0.15, 0.25, 0.35, 0.5]                              # cell42 の smoother blend 候補
    src_stages = ["pf_sel"] + [f"blend{int(w*100):02d}" for w in BLEND_WS]
    Esrc = {src: {s: [0.0, 0] for s in src_stages} for src in ("forward", "smooth")}
    Ecom = {s: [0.0, 0] for s in ["grid_lin", "grid_struct", "struct_only"]}   # PF非依存
    n_struct_ok = 0
    n = len(aw); t0 = time.time()
    for c0 in range(0, n, CHUNK):
        sub = aw[c0:c0 + CHUNK]
        inps = [make_inputs(CACHE[w], P) for w in sub]
        scaled = run_smoother_scales(inps, P, SEED, N_SEED, len(inps), SELECTOR_SCALES)
        for w, d in zip(sub, scaled):
            c = CACHE[w]; true = c["tvt_true"]; code = well_code(c); lastk = c["last_known_tvt"]
            variant = SELECTOR_BIN_VARIANTS.get(code, SELECTOR_GLOBAL_VARIANT)
            _, bw, _ = parse_variant(variant); beam = run_beam(c) if bw > 0 else None
            U_struct = _struct_u_prior(c, self_wid=w)              # self除外(リーク防止)
            # --- PF非依存: grid FB平滑(cell41忠実, linear prior) と 構造単体 ---
            grid_lin = grid_fb(c, 0.0, None)                       # 構造無(linear prior)
            grid_str = grid_fb(c, W_STRUCT, U_struct)              # +構造prior
            for s, pred in (("grid_lin", grid_lin), ("grid_struct", grid_str)):
                ss, cnt = _sse(pred - true); Ecom[s][0] += ss; Ecom[s][1] += cnt
            if U_struct is not None:
                st = U_struct[c["ev_index"]] - c["z_full"][c["ev_index"]]
                st = np.where(np.isfinite(st), st, lastk); n_struct_ok += 1
            else:
                st = grid_lin
            ss, cnt = _sse(st - true); Ecom["struct_only"][0] += ss; Ecom["struct_only"][1] += cnt
            # --- PF依存: pf_sel(forward/smooth) と grid_str との blend ---
            for src, sel_idx in (("forward", 0), ("smooth", 1)):
                pf_by = {sc: d[sc][sel_idx] for sc in SELECTOR_SCALES}
                s0 = apply_selector(pf_by, variant, beam, lastk)
                s0 = np.where(np.isfinite(s0), s0, lastk)
                preds = {"pf_sel": s0}
                for wgt in BLEND_WS:
                    preds[f"blend{int(wgt*100):02d}"] = (1.0 - wgt) * s0 + wgt * grid_str
                for s in src_stages:
                    ss, cnt = _sse(preds[s] - true); Esrc[src][s][0] += ss; Esrc[src][s][1] += cnt
        done = min(c0 + CHUNK, n); el = time.time() - t0; eta = el / done * (n - done)
        rp = _rmse(Esrc["smooth"]["pf_sel"]); rg = _rmse(Ecom["grid_lin"]); rgs = _rmse(Ecom["grid_struct"])
        print(f"[{done}/{n}] smo pf_sel={rp:.3f} | grid_lin={rg:.3f} grid_str={rgs:.3f} "
              f"struct={_rmse(Ecom['struct_only']):.3f} | {el:.0f}s ETA {eta/60:.0f}m", flush=True)
        if DEVICE == "cuda": torch.cuda.empty_cache()
    print(f"構造prior 利用可能坑井: {n_struct_ok}/{len(aw)}")

    print("\n===== PF非依存(grid FB平滑器, cell41忠実) =====")
    com = {s: _rmse(Ecom[s]) for s in Ecom}
    for s in ["grid_lin", "grid_struct", "struct_only"]:
        print(f"  {s:12s} = {com[s]:.4f}")
    print("\n===== PF依存: pf_sel と grid_struct の blend (forward / smooth) =====")
    print(f"  {'stage':10s}  {'forward':>9s}  {'smooth':>9s}")
    res = {src: {s: _rmse(Esrc[src][s]) for s in src_stages} for src in ("forward", "smooth")}
    for s in src_stages:
        print(f"  {s:10s}  {res['forward'][s]:9.4f}  {res['smooth'][s]:9.4f}")
    OUT_PATH.write_text(json.dumps(dict(
        pf="r0_seed32", smooth_lag=SMOOTH_LAG, n_seed=N_SEED, w_struct=W_STRUCT, blend_ws=BLEND_WS,
        n_wells=len(aw), pf_independent=com, forward=res["forward"], smooth=res["smooth"],
        note="cell41忠実: grid FB平滑器は linear prior で PF独立。pf_sel(forward/smooth) と grid_struct を blend。",
    ), indent=2))
    print(f"\nsaved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
