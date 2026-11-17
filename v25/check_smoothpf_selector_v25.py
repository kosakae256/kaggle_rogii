"""
v25 診断: 7.191ノートの selector フローの PF を「smooth PF(r0_seed32)」に差し替えるとどうなるか
================================================================================
やること(ユーザー方針):
  7.191ノートの selector フロー(マルチスケールPF → カテゴリ別 variant(scale/beam/hold) → apply_pp)を
  そのまま踏襲し、**PFの部分だけ v25 smooth PF(固定ラグ系統スムーザー, r0_seed32)に差し替え**て、
  train(真値)で overall RMSE を確認する。

比較:
  forward_sel … selector を **forward PF** に適用(=ノート現行相当)
  smooth_sel  … selector を **smooth PF** に適用(=今回の差し替え)
  参考: forward_raw / smooth_raw (variant の scale だけ, beam/hold無)
  → smooth_sel < forward_sel なら「PFを平滑に替えると selector フローでも効く」。
     カテゴリ別も出す(どの bin で効く/効かないか)。overfit方針: scale/beam/hold は本来この土台で再導出すべき。

selector(cell4 と同一):
  code = int(n_eval>4840) + 2*searchsorted((136.73,185.51), z_span)
  bin -> variant('pf_scale_X_beam_Y_hold_Z'): base=PF@scaleX → (1-Y)*base+Y*beam → (1-Z)*.+Z*last_known
  マルチスケール = seed尤度加重の温度 scale∈{3,5,8,12}(smooth側は smoothed 予測を同様に温度加重)

出力: v25/artifacts_v25/smoothpf_selector.json
実行: rogii_claude/.venv/Scripts/python.exe v25/check_smoothpf_selector_v25.py
※ 固定ラグ+マルチスケールを1回のPFで出す(前向き/平滑を全seed保持→scaleごとに加重)。
"""

import time, json, glob, math, warnings
from collections import defaultdict
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

SEED       = 0
N_SEED     = 128         # r0_seed32 regime(6.786と同じ)。ノートselectorは128だが土台に合わせ32
N_WELLS    = None        # None=全773。まず 200 等で絞ってよい
SMOOTH_LAG = 192         # smooth PF = 固定ラグ系統スムーザー(6.786と同じ L=192)
CHUNK      = 24          # GPUバッチ坑井数。バッファ B*L*N=CHUNK*N_SEED*L*N。OOMなら下げる

# --- selector(cell4 と同一) ---
N_EVAL_THRESHOLD  = 4840.0
Z_SPAN_THRESHOLDS = (136.73000000000016, 185.5133333333342)
SELECTOR_SCALES   = (3.0, 5.0, 8.0, 12.0)
SELECTOR_BIN_VARIANTS = {
    0: 'pf_scale_5_hold_0.2',
    1: 'pf_scale_3_hold_0.15',
    2: 'pf_scale_12_beam_0.2_hold_0.15',
    3: 'pf_scale_5_hold_0.15',
    4: 'pf_scale_5_beam_0.05_hold_0.05',
    5: 'pf_scale_12_beam_0.2_hold_0.05',
}
SELECTOR_GLOBAL_VARIANT = 'pf_scale_8_hold_0.2'
BEAM_CONFIGS = [
    (10, 20.0, 144.0, 2), (10, 8.0, 64.0, 2), (8, 35.0, 220.0, 1), (10, 14.0, 90.0, 5),
    (20, 4.0, 36.0, 3), (12, 12.0, 100.0, 3), (15, 25.0, 180.0, 2), (20, 30.0, 200.0, 2),
    (15, 10.0, 80.0, 4), (25, 6.0, 50.0, 3), (10, 40.0, 300.0, 1), (12, 18.0, 120.0, 5),
    (30, 8.0, 70.0, 2), (10, 50.0, 400.0, 0),
]
APPLY_PP = True
PP_TAU = 85.0; PP_ALPHA = 1.0; SG_W, SG_P = 17, 3

OUT_PATH = OUT_DIR / "smoothpf_selector.json"

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


# ----------------------- 固定ラグ系統スムーザー(tune_smoother_r0_v25 と同一) -----------------------
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
    """各 well について {scale: (forward予測, smoothed予測)} を返す(seed尤度加重, 温度=scale)。"""
    N = int(P["n_particles"]); S = n_seeds; W = len(inps)
    out = [None] * W
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
            T = len(sub[j]["md"]); llw = ll[j]
            d = {}
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


# ----------------------- beam(cell4 と同一, numba) -----------------------
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


# ----------------------- selector(cell4 と同一) + apply_pp -----------------------
def parse_selector_variant(name):
    parts = name.split('_'); scale = float(parts[2]); bw = hw_ = 0.0
    if 'beam' in parts: bw = float(parts[parts.index('beam') + 1])
    if 'hold' in parts: hw_ = float(parts[parts.index('hold') + 1])
    return scale, bw, hw_

def apply_selector(pf_by_scale, variant, beam, last_known_tvt):
    scale, bw, hw_ = parse_selector_variant(variant)
    base = pf_by_scale.get(scale, next(iter(pf_by_scale.values())))
    pred = (1.0 - bw) * base
    if bw > 0 and beam is not None:
        m = min(len(pred), len(beam)); pred = pred.copy(); pred[:m] = pred[:m] + bw * beam[:m]
    pred = (1.0 - hw_) * pred + hw_ * last_known_tvt
    return pred

def apply_pp(pred, c):
    pred = np.asarray(pred, dtype=np.float64)
    if not APPLY_PP: return pred
    md_since = np.maximum(c["md"] - c["last_known_md"], 0.0)
    d = (pred - c["last_known_tvt"]) * (1.0 - np.exp(-md_since / PP_TAU)) * PP_ALPHA
    out = c["last_known_tvt"] + d
    n = len(out); wl = min(SG_W, n)
    if wl % 2 == 0: wl -= 1
    return savgol_filter(out, wl, SG_P) if wl >= SG_P + 2 else out


def _sse(err):
    e = err[np.isfinite(err)]; return float(np.sum(e ** 2)), int(e.size)
def _rmse_acc(a): return math.sqrt(a[0] / a[1]) if a[1] else float("nan")


def main():
    print(f"device={DEVICE} PF=r0_seed32 smooth_lag={SMOOTH_LAG} n_seed={N_SEED} scales={SELECTOR_SCALES}")
    wells = list_wells()
    if N_WELLS: wells = wells[:N_WELLS]
    CACHE = {}
    for w in wells:
        c = build_cache(w)
        if c is not None and c["tvt_true"] is not None:
            CACHE[w] = c
    aw = [w for w in wells if w in CACHE]
    print(f"usable wells: {len(aw)}")

    P = {**R0, "smooth_lag": SMOOTH_LAG}
    names = ["forward_raw", "smooth_raw", "forward_sel", "smooth_sel"]
    E = {k: [0.0, 0] for k in names}                    # 全体 (sumsq,count)
    Ecat = {code: {k: [0.0, 0] for k in names} for code in range(6)}
    n = len(aw); t0 = time.time()
    for c0 in range(0, n, CHUNK):
        sub = aw[c0:c0 + CHUNK]
        inps = [make_inputs(CACHE[w], P) for w in sub]
        scaled = run_smoother_scales(inps, P, SEED, N_SEED, len(inps), SELECTOR_SCALES)
        for w, d in zip(sub, scaled):
            c = CACHE[w]; true = c["tvt_true"]; code = well_code(c)
            variant = SELECTOR_BIN_VARIANTS.get(code, SELECTOR_GLOBAL_VARIANT)
            sc_var, bw, _ = parse_selector_variant(variant)
            fwd_by = {sc: d[sc][0] for sc in SELECTOR_SCALES}
            sm_by  = {sc: d[sc][1] for sc in SELECTOR_SCALES}
            beam = run_beam(c) if bw > 0 else None
            preds = {
                "forward_raw": fwd_by[sc_var],
                "smooth_raw":  sm_by[sc_var],
                "forward_sel": apply_pp(apply_selector(fwd_by, variant, beam, c["last_known_tvt"]), c),
                "smooth_sel":  apply_pp(apply_selector(sm_by,  variant, beam, c["last_known_tvt"]), c),
            }
            for k in names:
                s, cnt = _sse(preds[k] - true)
                E[k][0] += s; E[k][1] += cnt; Ecat[code][k][0] += s; Ecat[code][k][1] += cnt
        done = min(c0 + CHUNK, n); el = time.time() - t0; eta = el / done * (n - done)
        rf = _rmse_acc(E["forward_sel"]); rs = _rmse_acc(E["smooth_sel"])
        print(f"[{done}/{n}] forward_sel={rf:.3f} smooth_sel={rs:.3f} (Δ {rs-rf:+.3f}) | {el:.0f}s ETA {eta/60:.0f}m", flush=True)
        if DEVICE == "cuda": torch.cuda.empty_cache()

    print("\n===== 全体 overall(全行プールRMSE) =====")
    res = {k: _rmse_acc(E[k]) for k in names}
    for k in names: print(f"  {k:14s} = {res[k]:.4f}")
    print(f"  Δ(smooth_sel - forward_sel) = {res['smooth_sel']-res['forward_sel']:+.4f}  "
          f"({'smoothが良い' if res['smooth_sel']<res['forward_sel'] else 'forwardが良い'})")

    print("\n===== カテゴリ別(forward_sel vs smooth_sel) =====")
    cat_tbl = {}
    for code in range(6):
        if Ecat[code]["forward_sel"][1] == 0: continue
        rf = _rmse_acc(Ecat[code]["forward_sel"]); rs = _rmse_acc(Ecat[code]["smooth_sel"])
        var = SELECTOR_BIN_VARIANTS.get(code, SELECTOR_GLOBAL_VARIANT)
        nrows = Ecat[code]["forward_sel"][1]
        print(f"  cat{code} [{var:32s}] rows={nrows:6d}  forward={rf:.3f} smooth={rs:.3f} (Δ {rs-rf:+.3f})")
        cat_tbl[str(code)] = dict(variant=var, rows=nrows, forward_sel=rf, smooth_sel=rs)

    OUT_PATH.write_text(json.dumps(dict(
        pf="r0_seed32", smooth_lag=SMOOTH_LAG, n_seed=N_SEED, scales=list(SELECTOR_SCALES),
        n_wells=len(aw), overall=res, per_category=cat_tbl,
        note="selectorフローのPFをforward->smoothに差替。smooth_sel<forward_selなら平滑がselectorでも効く。"
             "scale/beam/holdは本来この土台で再導出すべき(overfit_policy)。",
    ), indent=2))
    print(f"\nsaved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
