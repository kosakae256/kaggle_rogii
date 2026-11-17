# -*- coding: utf-8 -*-
"""波形変換の大量探索: GRの各種変換(Fourier/wavelet/Hilbert/band-pass/微分/…)が
   PFに有利か = ①判別マージン(真vs decoy) ②振幅/dGRとの脱相関(δ* datum推定) で評価。
   さらに事前区間(既知prefix)較正版も試す。ダメもと大量探索。出力: explore_transforms.txt"""
import sys, io, glob
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
import numpy as np, pandas as pd
from scipy.signal import savgol_filter, hilbert
from scipy.ndimage import gaussian_filter1d, median_filter


def _ricker(a):
    n = int(8 * a) | 1
    x = np.arange(-(n // 2), n // 2 + 1)
    A = 2 / (np.sqrt(3 * a) * np.pi ** 0.25)
    return A * (1 - (x / a) ** 2) * np.exp(-x ** 2 / (2 * a ** 2))


def cwt1(g, a):
    return np.convolve(g, _ricker(a), mode="same")   # ricker wavelet 1スケール

fs = sorted(glob.glob("rogii-wellbore-geology-prediction/train/*__horizontal_well.csv"))
GRID = 0.5
SHIFTS = [-30, -15, 15, 30]


def cc(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 20 or np.std(a[m]) < 1e-9 or np.std(b[m]) < 1e-9:
        return np.nan
    return float(np.corrcoef(a[m], b[m])[0, 1])


def zsc(a):
    s = np.nanstd(a); return (a - np.nanmean(a)) / (s + 1e-9) if s > 0 else a * 0


# 変換群(1D系列 -> 1D系列)
def bandpass(a, s_lo, s_hi):
    return gaussian_filter1d(a, s_lo) - gaussian_filter1d(a, s_hi)   # DoG(帯域)


TRANSFORMS = {
    "amp": lambda g: g,
    "dGR": lambda g: savgol_filter(g, 11, 3, deriv=1),
    "d2GR": lambda g: savgol_filter(g, 11, 3, deriv=2),
    "absdGR": lambda g: np.abs(savgol_filter(g, 11, 3, deriv=1)),
    "bp_low": lambda g: gaussian_filter1d(g, 20),
    "bp_mid": lambda g: bandpass(g, 3, 12),      # サイクル帯
    "bp_midlo": lambda g: bandpass(g, 6, 20),
    "bp_high": lambda g: g - gaussian_filter1d(g, 4),   # 高域(細ベッド)
    "hilb_env": lambda g: np.abs(hilbert(g - gaussian_filter1d(g, 30))),   # 包絡(振幅変調)
    "hilb_cos": lambda g: np.cos(np.angle(hilbert(g - gaussian_filter1d(g, 30)))),  # 瞬時位相cos
    "rank": lambda g: (np.argsort(np.argsort(g)) / len(g)),
    "log": lambda g: np.log(np.clip(g, 1, None)),
    "medfilt": lambda g: median_filter(g, 7),
    "cwt3": lambda g: cwt1(g, 3),
    "cwt8": lambda g: cwt1(g, 8),
    "cwt15": lambda g: cwt1(g, 15),
}


def bin_med(x, y, g):
    idx = np.digitize(x, g); out = np.full(len(g), np.nan)
    for i in range(len(g)):
        v = y[idx == i]
        if len(v):
            out[i] = np.median(v)
    m = np.isfinite(out)
    return np.interp(g, g[m], out[m]) if m.sum() > 2 else out


margin = {k: [] for k in TRANSFORMS}
dstar = {k: [] for k in TRANSFORMS}
dgrid = np.arange(-40, 41, 2.0)
n_ok = 0
for hf in fs[:150]:
    tf = hf.replace("__horizontal_well", "__typewell")
    try:
        h = pd.read_csv(hf); tw = pd.read_csv(tf).sort_values("TVT")
    except Exception:
        continue
    if "GR" not in h or "TVT" not in h:
        continue
    tvt = h["TVT"].to_numpy(float); gh = h["GR"].to_numpy(float)
    fin = np.isfinite(tvt) & np.isfinite(gh); tvt = tvt[fin]; gh = gh[fin]
    twt = tw["TVT"].to_numpy(float); twg = tw["GR"].to_numpy(float)
    if len(gh) < 300:
        continue
    lo = max(np.percentile(tvt, 5), twt.min()); hi = min(np.percentile(tvt, 95), twt.max())
    if hi - lo < 60:
        continue
    grid = np.arange(lo, hi, GRID)
    ghg = bin_med(tvt, gh, grid)
    # 真+decoyの typewell(grid上)
    def twgrid(shift): return np.interp(grid + shift, twt, twg)
    for k, T in TRANSFORMS.items():
        try:
            th = zsc(T(ghg))
            tt_true = zsc(T(twgrid(0)))
            ct = cc(th, tt_true)
            cds = [cc(th, zsc(T(twgrid(s)))) for s in SHIFTS]; cds = [x for x in cds if np.isfinite(x)]
            if np.isfinite(ct) and cds:
                margin[k].append(ct - max(cds))
            # δ*(datum推定): 各offsetでの一致度の最大位置
            cs = [cc(th, zsc(T(np.interp(grid + d, twt, twg)))) for d in dgrid]
            cs = [x if np.isfinite(x) else -9 for x in cs]
            dstar[k].append(dgrid[int(np.argmax(cs))])
        except Exception:
            pass
    n_ok += 1

out = [f"=== 波形変換 大量探索 ({n_ok}井) : マージン(判別力) と 脱相関(振幅/dGRと別物か) ==="]
amp_d = np.array(dstar["amp"]); dgr_d = np.array(dstar["dGR"])
rows = []
for k in TRANSFORMS:
    mg = np.nanmedian(margin[k]) if margin[k] else np.nan
    d = np.array(dstar[k])
    n = min(len(d), len(amp_d))
    dc_amp = np.corrcoef(d[:n], amp_d[:n])[0, 1] if n > 10 else np.nan
    dc_dgr = np.corrcoef(d[:n], dgr_d[:n])[0, 1] if n > 10 else np.nan
    rows.append((k, mg, dc_amp, dc_dgr))
out.append(f"{'変換':10s} {'マージン':>8s} {'脱相関vs振幅':>12s} {'脱相関vs dGR':>12s}  判定")
for k, mg, da, dd in sorted(rows, key=lambda r: -(r[1] if np.isfinite(r[1]) else -9)):
    verdict = ""
    if np.isfinite(mg) and mg > 0.25:
        if k != "amp" and np.isfinite(da) and abs(da) < 0.7 and abs(dd) < 0.7:
            verdict = "★脱相関脚候補(判別力あり+振幅/dGR両方と別物)"
        elif k == "amp":
            verdict = "基準"
        elif np.isfinite(da) and abs(da) < 0.7:
            verdict = "○振幅と別物"
    out.append(f"{k:10s} {mg:>+8.3f} {da:>+12.3f} {dd:>+12.3f}  {verdict}")
txt = "\n".join(out); print("\n" + txt, flush=True)
open("v94/artifacts_v94/explore_transforms.txt", "w", encoding="utf-8").write(txt)
