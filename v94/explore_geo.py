# -*- coding: utf-8 -*-
"""★Z幾何配置(TVT=面−Z, 面=prefix外挿; GR錨不使用)で horizontal を TVT配置した後、
   band-pass より良い信号変換を探索。各変換の【信号強度②(真TVTで一致)】＋【振幅と脱相関】。
   真TVTは評価のみ。出力: explore_geo.txt"""
import sys, io, os
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pf_engine_v94 as pf
from scipy.signal import savgol_filter, hilbert
from scipy.ndimage import gaussian_filter1d


def cc(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    return np.corrcoef(a[m], b[m])[0, 1] if m.sum() > 20 and np.std(a[m]) > 1e-9 and np.std(b[m]) > 1e-9 else np.nan


def zsc(a):
    s = np.nanstd(a); return (a - np.nanmean(a)) / (s + 1e-9) if s > 0 else a * 0


def gbp(g, lo, hi):
    return gaussian_filter1d(g, lo) - gaussian_filter1d(g, hi)


def cwt1(g, a):
    n = int(8 * a) | 1; x = np.arange(-(n // 2), n // 2 + 1)
    k = 2 / (np.sqrt(3 * a) * np.pi ** .25) * (1 - (x / a) ** 2) * np.exp(-x ** 2 / (2 * a ** 2))
    return np.convolve(g, k, mode="same")


# grid 0.5ft: σ[sample]*0.5=ft
TF = {
    "amp": lambda g: g,
    "hp_g4(>2ft除)": lambda g: g - gaussian_filter1d(g, 4),
    "hp_g8(>4ft除)": lambda g: g - gaussian_filter1d(g, 8),
    "hp_g16(>8ft除)": lambda g: g - gaussian_filter1d(g, 16),
    "hp_g32(>16ft除)": lambda g: g - gaussian_filter1d(g, 32),
    "hp_g64(>32ft除)": lambda g: g - gaussian_filter1d(g, 64),
    "bp_1_16": lambda g: gbp(g, 2, 32),
    "bp_2_32": lambda g: gbp(g, 4, 64),
    "bp_2_64": lambda g: gbp(g, 4, 128),
    "bp_4_64": lambda g: gbp(g, 8, 128),
    "dGR": lambda g: savgol_filter(g, 11, 3, deriv=1),
    "cwt6": lambda g: cwt1(g, 6),
    "cwt12": lambda g: cwt1(g, 12),
    "hilb_env": lambda g: np.abs(hilbert(gbp(g, 4, 64))),
    "bp+dGR": lambda g: savgol_filter(gbp(g, 4, 64), 11, 3, deriv=1),
}
GRID = 0.5


def resample(te, gr, lo, hi):
    grid = np.arange(lo, hi + GRID, GRID)
    idx = np.clip(np.digitize(te, grid), 0, len(grid) - 1)
    cnt = np.bincount(idx, minlength=len(grid)).astype(float)
    rs = np.bincount(idx, weights=gr, minlength=len(grid)) / np.maximum(cnt, 1.0)
    fm = cnt > 0
    if fm.sum() > 5:
        rs = np.interp(np.arange(len(grid)), np.where(fm)[0], rs[fm])
    return grid, rs


ws = [w for w in pf.list_wells()[:200] if pf.CACHE.setdefault(w, pf.build_cache(w)) is not None]
dgrid = np.arange(-64, 65, 2.0)
sig = {k: [] for k in TF}; ds = {k: [] for k in TF}
for w in ws:
    c = pf.CACHE[w]; tt = c["tvt_true"]
    if tt is None:
        continue
    hw = pd.read_csv(pf.DATA_DIR / f"{w}__horizontal_well.csv")
    Z = hw["Z"].to_numpy(float); MD = hw["MD"].to_numpy(float)
    gr = c["gr_filled"]; ev = c["ev_pos"]; kp = c["kpos"]
    s_kn = c["kn_tvtin"] + Z[kp]
    if len(kp) < 5:
        continue
    A = np.polyfit(MD[kp], s_kn, 1)                       # ★面をprefixで1次外挿(GR錨不使用)
    tvt_geo = np.polyval(A, MD) - Z
    ok = np.isfinite(tvt_geo) & np.isfinite(gr)
    if ok.sum() < 100:
        continue
    lo = float(np.percentile(tvt_geo[ok], 1)); hi = float(np.percentile(tvt_geo[ok], 99))
    if hi - lo < 30:
        continue
    grid, hrs = resample(tvt_geo[ok], gr[ok], lo, hi)     # 幾何配置 horizontal
    twt = c["tw_tvt"]; twg = c["tw_gr"]
    tw_grid = np.arange(twt.min(), twt.max() + GRID, GRID); tws = np.interp(tw_grid, twt, twg)
    geo_ev = np.clip(tvt_geo[ev], lo, hi)
    for k, T in TF.items():
        try:
            TH = T(hrs); TT = T(tws)
        except Exception:
            continue
        th_ev = np.interp(geo_ev, grid, TH)
        tt_ev = np.interp(tt, tw_grid, TT)
        s = cc(zsc(th_ev), zsc(tt_ev))
        if np.isfinite(s):
            sig[k].append(s)
        cs = np.array([cc(zsc(th_ev), zsc(np.interp(tvt_geo[ev] + d, tw_grid, TT))) for d in dgrid])
        ds[k].append(dgrid[int(np.nanargmax(cs))] if np.isfinite(cs).sum() > 20 else np.nan)

amp_ds = np.array(ds["amp"])
out = [f"=== Z幾何配置での変換探索(GR錨不使用) {len([x for x in sig['amp'] if np.isfinite(x)])}井 ==="]
out.append(f"{'変換':16s} {'信号強度②':>9s} {'脱相関vs振幅':>11s}  判定")
rows = []
for k in TF:
    s = np.nanmedian(sig[k]) if sig[k] else np.nan
    d = np.array(ds[k]); n = min(len(d), len(amp_ds))
    dc = np.corrcoef(d[:n], amp_ds[:n])[0, 1] if n > 10 and np.isfinite(d[:n]).sum() > 10 else np.nan
    rows.append((k, s, dc))
bp_sig = dict((k, s) for k, s, _ in rows).get("bp_2_32", 0.45)
for k, s, dc in sorted(rows, key=lambda r: -(r[1] if np.isfinite(r[1]) else -9)):
    v = ""
    if k == "amp":
        v = "基準(点照合0.72だがresample込みの値)"
    elif np.isfinite(s) and s > bp_sig + 0.02 and np.isfinite(dc) and abs(dc) < 0.7:
        v = "★★band-pass超え(信号強+脱相関)"
    elif np.isfinite(s) and s > 0.4 and np.isfinite(dc) and abs(dc) < 0.7:
        v = "○使える脱相関脚"
    out.append(f"{k:16s} {s:>+9.3f} {dc:>+11.3f}  {v}")
txt = "\n".join(out); print("\n" + txt, flush=True)
open("v94/artifacts_v94/explore_geo.txt", "w", encoding="utf-8").write(txt)
