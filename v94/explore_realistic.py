# -*- coding: utf-8 -*-
"""★真TVT不使用の現実的探索: 錨(known=TVT_input, eval=anchor)で horizontal を TVT配置し、
   各変換の【信号強度(錨配置で真アライメントに残る一致度=②)】と【振幅との脱相関(δ*)】を測る。
   真TVTは"評価"にのみ使用(表現の構築には使わない)。band-pass以外の使える別表現を探す。
   出力: explore_realistic.txt"""
import sys, io, os
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pf_engine_v94 as pf   # 振幅engine(生cache取得用)
from scipy.signal import savgol_filter, hilbert
from scipy.ndimage import gaussian_filter1d, uniform_filter1d


def cc(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    return np.corrcoef(a[m], b[m])[0, 1] if m.sum() > 20 and np.std(a[m]) > 1e-9 and np.std(b[m]) > 1e-9 else np.nan


def zsc(a):
    s = np.nanstd(a); return (a - np.nanmean(a)) / (s + 1e-9) if s > 0 else a * 0


def rstd(a, w):
    m = uniform_filter1d(a, w); return np.sqrt(np.maximum(uniform_filter1d(a * a, w) - m * m, 0))


def gbp(g, lo, hi):
    return gaussian_filter1d(g, lo) - gaussian_filter1d(g, hi)


# 変換(uniform TVGグリッド上, 真TVT不要)
TF = {
    "amp": lambda g: g,
    "bp_2_16": lambda g: gbp(g, 4, 32),        # grid0.5ft: σ4-32samples=2-16ft
    "bp_2_32": lambda g: gbp(g, 4, 64),        # 2-32ft(現行bp)
    "bp_4_32": lambda g: gbp(g, 8, 64),        # 4-32ft
    "dGR": lambda g: savgol_filter(g, 11, 3, deriv=1),
    "hilb_env": lambda g: np.abs(hilbert(gbp(g, 4, 64))),
    "hilb_cos": lambda g: np.cos(np.angle(hilbert(gbp(g, 4, 64)))),
    "rstd11": lambda g: rstd(g, 11),           # 局所テクスチャ(荒さ)
    "rstd31": lambda g: rstd(g, 31),
    "highpass": lambda g: g - gaussian_filter1d(g, 8),   # 高域(細ベッド)
    "cwt8": lambda g: np.convolve(g, (lambda a: (lambda x: 2 / (np.sqrt(3 * a) * np.pi ** .25) * (1 - (x / a) ** 2) * np.exp(-x ** 2 / (2 * a ** 2)))(np.arange(-(int(8 * a) // 2), int(8 * a) // 2 + 1)))(8), mode="same"),
    "logamp": lambda g: np.log(np.clip(g - np.nanmin(g) + 1, 1, None)),
}
GRID = 0.5


def resample(tvt_est, gr, lo, hi):
    grid = np.arange(lo, hi + GRID, GRID)
    idx = np.clip(np.digitize(tvt_est, grid), 0, len(grid) - 1)
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
    if tt is None or float(c["ancs"][0]) > 1e8:
        continue
    gr_filled = c["gr_filled"]; ev = c["ev_pos"]; kp = c["kpos"]
    tvt_est = np.full(len(gr_filled), np.nan)
    tvt_est[kp] = c["kn_tvtin"]; tvt_est[ev] = c["anc"]
    ok = np.isfinite(tvt_est) & np.isfinite(gr_filled)
    if ok.sum() < 80:
        continue
    lo = float(np.percentile(tvt_est[ok], 1)); hi = float(np.percentile(tvt_est[ok], 99))
    if hi - lo < 30:
        continue
    grid, hrs = resample(tvt_est[ok], gr_filled[ok], lo, hi)   # 錨配置 horizontal (uniform grid)
    twt = c["tw_tvt"]; twg = c["tw_gr"]
    tw_grid = np.arange(twt.min(), twt.max() + GRID, GRID)
    tws = np.interp(tw_grid, twt, twg)                          # typewell (uniform grid, 真TVT)
    ttev = tt; ancev = c["anc"]   # tvt_true/anc は既にeval順(長さnev)
    for k, T in TF.items():
        try:
            TH = T(hrs); TT = T(tws)
        except Exception:
            continue
        # ② 信号: 錨配置 horizontal(各eval行=anchor位置) vs typewell(真TVT)
        th_ev = np.interp(np.clip(ancev, lo, hi), grid, TH)
        tt_ev = np.interp(ttev, tw_grid, TT)
        sig[k].append(cc(zsc(th_ev), zsc(tt_ev)))
        # δ* 脱相関(錨配置での datum探索)
        cs = np.array([cc(zsc(th_ev), zsc(np.interp(ancev + d, tw_grid, TT))) for d in dgrid])
        ds[k].append(dgrid[int(np.nanargmax(cs))] if np.isfinite(cs).sum() > 20 else np.nan)

amp_ds = np.array(ds["amp"])
out = [f"=== 現実的探索(錨配置, 真TVT不使用) {len([1 for x in sig['amp'] if np.isfinite(x)])}井 ==="]
out.append(f"{'変換':10s} {'信号強度②':>9s} {'脱相関vs振幅':>11s}  判定")
rows = []
for k in TF:
    s = np.nanmedian(sig[k]); d = np.array(ds[k]); n = min(len(d), len(amp_ds))
    dc = np.corrcoef(d[:n], amp_ds[:n])[0, 1] if n > 10 and np.isfinite(d[:n]).sum() > 10 else np.nan
    rows.append((k, s, dc))
for k, s, dc in sorted(rows, key=lambda r: -(r[1] if np.isfinite(r[1]) else -9)):
    v = ""
    if k != "amp" and np.isfinite(s) and s > 0.4 and np.isfinite(dc) and abs(dc) < 0.7:
        v = "★使える候補(錨で信号残る+脱相関)"
    elif k == "amp":
        v = "基準"
    out.append(f"{k:10s} {s:>+9.3f} {dc:>+11.3f}  {v}")
txt = "\n".join(out); print("\n" + txt, flush=True)
open("v94/artifacts_v94/explore_realistic.txt", "w", encoding="utf-8").write(txt)
