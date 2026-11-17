# -*- coding: utf-8 -*-
"""★錨・TVT配置 不要の局所特徴を探索: horizontalは生MD空間で各行の局所特徴を計算(配置不要)、
   typewellはTVT空間で計算し、真アライメントで一致するか(信号強度)＋振幅と脱相関か を測る。
   Fourier系(局所スペクトル重心/帯域幅)・texture系(rolling std/skew)・ゼロ交差率 等。
   真TVTは評価にのみ使用。出力: explore_noanchor.txt"""
import sys, io, os
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pf_engine_v94 as pf
from scipy.ndimage import uniform_filter1d, gaussian_filter1d


def cc(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    return np.corrcoef(a[m], b[m])[0, 1] if m.sum() > 20 and np.std(a[m]) > 1e-9 and np.std(b[m]) > 1e-9 else np.nan


def zsc(a):
    s = np.nanstd(a); return (a - np.nanmean(a)) / (s + 1e-9) if s > 0 else a * 0


def rstd(a, w):
    m = uniform_filter1d(a, w); return np.sqrt(np.maximum(uniform_filter1d(a * a, w) - m * m, 0))


def rskew(a, w):
    m = uniform_filter1d(a, w); s = rstd(a, w) + 1e-6
    return uniform_filter1d((a - m) ** 3, w) / s ** 3


def zcr(a, w):
    d = a - gaussian_filter1d(a, w)          # 局所detrend
    z = (np.abs(np.diff(np.sign(d))) > 0).astype(float)
    z = np.concatenate([z, z[-1:]])
    return uniform_filter1d(z, w)            # 局所ゼロ交差率(=局所周波数proxy)


def spec_feat(a, win, which):
    """スライド窓局所FFT: which='cent'(重心)/'bw'(帯域幅)。窓内detrend+Hann。"""
    n = len(a); out = np.full(n, np.nan); h = win // 2
    ham = np.hanning(win); freqs = np.fft.rfftfreq(win)
    ap = np.pad(a, h, mode="reflect")
    for i in range(n):
        seg = ap[i:i + win].copy(); seg = (seg - seg.mean()) * ham
        P = np.abs(np.fft.rfft(seg)) ** 2; s = P.sum()
        if s < 1e-9:
            continue
        cent = (freqs * P).sum() / s
        if which == "cent":
            out[i] = cent
        else:
            out[i] = np.sqrt(((freqs - cent) ** 2 * P).sum() / s)
    return out


TF = {
    "amp": lambda g: g,
    "rstd15": lambda g: rstd(g, 15),
    "rstd31": lambda g: rstd(g, 31),
    "rskew21": lambda g: rskew(g, 21),
    "zcr21": lambda g: zcr(g, 21),
    "spec_cent31": lambda g: spec_feat(g, 31, "cent"),
    "spec_cent61": lambda g: spec_feat(g, 61, "cent"),
    "spec_bw31": lambda g: spec_feat(g, 31, "bw"),
    "range21": lambda g: uniform_filter1d(np.maximum.accumulate(g * 0), 1) if False else
        (lambda m: uniform_filter1d(np.abs(g - m), 21))(uniform_filter1d(g, 21)),
}

ws = [w for w in pf.list_wells()[:200] if pf.CACHE.setdefault(w, pf.build_cache(w)) is not None]
dgrid = np.arange(-64, 65, 2.0)
sig = {k: [] for k in TF}; ds = {k: [] for k in TF}
for w in ws:
    c = pf.CACHE[w]; tt = c["tvt_true"]
    if tt is None:
        continue
    gh = c["gr_filled"][c["ev_pos"]]                 # ★生MD(eval行, 配置不要)
    twt = c["tw_tvt"]; twg = c["tw_gr"]
    m = np.isfinite(gh) & np.isfinite(tt)
    if m.sum() < 100:
        continue
    for k, T in TF.items():
        try:
            fh = T(gh)                                # horizontal局所特徴(MD)
            ftw_grid = T(twg)                         # typewell局所特徴(TVT samples)
        except Exception:
            continue
        ftw_at = np.interp(tt, twt, ftw_grid)         # typewell特徴@真TVT
        s = cc(zsc(fh[m]), zsc(ftw_at[m]))
        if np.isfinite(s):
            sig[k].append(s)
        anc = tt  # δ*探索は真TVT中心(配置不要特徴なので、真周辺での判別を見る)
        cs = np.array([cc(zsc(fh[m]), zsc(np.interp(tt[m] + d, twt, ftw_grid))) for d in dgrid])
        ds[k].append(dgrid[int(np.nanargmax(cs))] if np.isfinite(cs).sum() > 20 else np.nan)

amp_ds = np.array(ds["amp"])
out = [f"=== 錨不要の局所特徴 探索(生MD計算) {len([x for x in sig['amp'] if np.isfinite(x)])}井 ==="]
out.append(f"{'特徴':12s} {'信号強度':>8s} {'脱相関vs振幅':>11s}  判定")
rows = [(k, np.nanmedian(sig[k]) if sig[k] else np.nan,
         (lambda d: np.corrcoef(d[:min(len(d), len(amp_ds))], amp_ds[:min(len(d), len(amp_ds))])[0, 1]
          if min(len(d), len(amp_ds)) > 10 and np.isfinite(d[:min(len(d), len(amp_ds))]).sum() > 10 else np.nan)(np.array(ds[k])))
        for k in TF]
for k, s, dc in sorted(rows, key=lambda r: -(r[1] if np.isfinite(r[1]) else -9)):
    v = "★錨不要で使える(信号+脱相関)" if (k != "amp" and np.isfinite(s) and s > 0.35 and np.isfinite(dc) and abs(dc) < 0.7) else ("基準" if k == "amp" else "")
    out.append(f"{k:12s} {s:>+8.3f} {dc:>+11.3f}  {v}")
txt = "\n".join(out); print("\n" + txt, flush=True)
open("v94/artifacts_v94/explore_noanchor.txt", "w", encoding="utf-8").write(txt)
