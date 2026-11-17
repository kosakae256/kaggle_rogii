# -*- coding: utf-8 -*-
"""placement改善の探索: high-pass のマージンは"配置精度"で決まる(配置誤差±15ftが細ベッドを汚す)。
   複数の配置法(linear幾何/quadratic幾何/錨raw/錨平滑/hybrid)でhigh-passのマージン+脱相関を比較。
   真TVTは評価のみ。出力: explore_placement.txt"""
import sys, io, os
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pf_engine_v94 as pf
from scipy.ndimage import gaussian_filter1d


def cc(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    return np.corrcoef(a[m], b[m])[0, 1] if m.sum() > 20 and np.std(a[m]) > 1e-9 and np.std(b[m]) > 1e-9 else np.nan


def zsc(a):
    s = np.nanstd(a); return (a - np.nanmean(a)) / (s + 1e-9) if s > 0 else a * 0


def hp(g, sigft, step=0.5):
    return g - gaussian_filter1d(g, max(sigft / step, 1.0))


def resample(te, gr, lo, hi, step=0.5):
    grid = np.arange(lo, hi + step, step); idx = np.clip(np.digitize(te, grid), 0, len(grid) - 1)
    cnt = np.bincount(idx, minlength=len(grid)).astype(float); rs = np.bincount(idx, weights=gr, minlength=len(grid)) / np.maximum(cnt, 1)
    fm = cnt > 0
    if fm.sum() > 5:
        rs = np.interp(np.arange(len(grid)), np.where(fm)[0], rs[fm])
    return grid, rs


ws = [w for w in pf.list_wells()[:150] if pf.CACHE.setdefault(w, pf.build_cache(w)) is not None]
shifts = [-30, -15, 15, 30]
methods = ["geo_lin", "geo_quad", "anc_raw", "anc_sm5", "anc_sm15", "hybrid"]
sig = {k: [] for k in methods}; mg = {k: [] for k in methods}; dstar = {k: [] for k in methods}
dgrid = np.arange(-64, 65, 2.0)
for w in ws:
    c = pf.CACHE[w]; tt = c["tvt_true"]
    if tt is None:
        continue
    hw = pd.read_csv(pf.DATA_DIR / f"{w}__horizontal_well.csv"); Z = hw["Z"].to_numpy(float); MD = hw["MD"].to_numpy(float)
    gr = c["gr_filled"]; ev = c["ev_pos"]; kp = c["kpos"]
    if len(kp) < 8:
        continue
    surf_kn = c["kn_tvtin"] + Z[kp]
    A1 = np.polyfit(MD[kp], surf_kn, 1); A2 = np.polyfit(MD[kp], surf_kn, 2)
    anc = c["anc"]; has_anc = float(c["ancs"][0]) < 1e8
    # 各配置の eval行TVT(＋known行TVTも配置に必要 → 全行版を作る)
    def full(evvals):
        te = np.full(len(gr), np.nan); te[kp] = c["kn_tvtin"]; te[ev] = evvals; return te
    P = {
        "geo_lin": full((np.polyval(A1, MD) - Z)[ev]),
        "geo_quad": full((np.polyval(A2, MD) - Z)[ev]),
    }
    if has_anc:
        P["anc_raw"] = full(anc)
        P["anc_sm5"] = full(gaussian_filter1d(anc, 5))
        P["anc_sm15"] = full(gaussian_filter1d(anc, 15))
        # hybrid: 錨surface(anc+Z_ev)を平滑→−Z (錨の構造+Zの細部, 行ノイズ除去)
        surf_anc = gaussian_filter1d(anc + Z[ev], 8)
        P["hybrid"] = full(surf_anc - Z[ev])
    twt = c["tw_tvt"]; twg = c["tw_gr"]; tw_grid = np.arange(twt.min(), twt.max(), 0.5); tws = np.interp(tw_grid, twt, twg)
    TT = hp(tws, 32)
    for k, te in P.items():
        okr = np.isfinite(te) & np.isfinite(gr)
        if okr.sum() < 80:
            continue
        lo = np.percentile(te[okr], 1); hi = np.percentile(te[okr], 99)
        if hi - lo < 30:
            continue
        grid, rs = resample(te[okr], gr[okr], lo, hi); TH = hp(rs, 32)
        th = np.interp(np.clip(te[ev], lo, hi), grid, TH)
        ct = cc(zsc(th), zsc(np.interp(tt, tw_grid, TT)))
        cds = [cc(zsc(th), zsc(np.interp(tt + s, tw_grid, TT))) for s in shifts]; cds = [x for x in cds if np.isfinite(x)]
        if np.isfinite(ct):
            sig[k].append(ct)
        if np.isfinite(ct) and cds:
            mg[k].append(ct - max(cds))
        cs = np.array([cc(zsc(th), zsc(np.interp(te[ev] + d, tw_grid, TT))) for d in dgrid])
        dstar[k].append(dgrid[int(np.nanargmax(cs))] if np.isfinite(cs).sum() > 20 else np.nan)

# 振幅点照合の参考
amp_ds = []
for w in ws:
    c = pf.CACHE[w]; tt = c["tvt_true"]
    if tt is None:
        continue
    ge = c["gr_filled"][c["ev_pos"]]; twt = c["tw_tvt"]; twg = c["tw_gr"]
    m = np.isfinite(ge) & np.isfinite(tt)
    if m.sum() < 80:
        continue
    cs = np.array([cc(zsc(ge[m]), zsc(np.interp(tt[m] + d, twt, twg))) for d in dgrid])
    amp_ds.append(dgrid[int(np.nanargmax(cs))] if np.isfinite(cs).sum() > 20 else np.nan)
amp_ds = np.array(amp_ds)
out = ["=== placement改善 探索(high-pass>32ft, マージン+脱相関) ==="]
out.append("参考: 振幅点照合 マージン+0.462")
out.append(f"{'配置':10s} {'信号':>7s} {'マージン':>8s} {'脱相関vs振幅':>11s}")
for k in methods:
    if not mg[k]:
        continue
    d = np.array(dstar[k]); n = min(len(d), len(amp_ds))
    dc = np.corrcoef(d[:n], amp_ds[:n])[0, 1] if n > 10 and np.isfinite(d[:n]).sum() > 10 else np.nan
    out.append(f"{k:10s} {np.nanmedian(sig[k]):>+7.3f} {np.nanmedian(mg[k]):>+8.3f} {dc:>+11.3f}")
txt = "\n".join(out); print("\n" + txt, flush=True)
open("v94/artifacts_v94/explore_placement.txt", "w", encoding="utf-8").write(txt)
