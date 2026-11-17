# -*- coding: utf-8 -*-
"""
v78 P24 — K16錨のフィールド適応θ0(方位)
=========================================
原典はθ0=118.4°全域固定。フィールド(XYクラスタ)ごとにθをLOO掃引して適応させる。
θはドリフト場の脱射影(c/proj)と予測射影の両方に効く。クラスタ内で1パラメータ=過学習低リスク。
出力: 各クラスタのθ*とLOO改善、採用時の全体RMSE。改善時は k16_anchor_train_ftheta.pkl を保存。
使い方: python v77/p24_fieldtheta_v77.py
"""
import os, io, sys, glob, pickle, time
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
from pathlib import Path
import numpy as np, pandas as pd

P0 = Path(r"C:\Users\kosaka256\Documents\rogii_claude")
DATA = P0 / "rogii-wellbore-geology-prediction"
ART = P0 / "v77" / "artifacts_v77"
K = 16; KNN_K, KNN_H = 15, 500.0; GATE = 0.35; FIELD_MIN_PROJ = 0.3
KBINS = [0.0, 750.0, 1500.0, 2500.0, 4000.0, 1e18]
NBIN = len(KBINS) - 1
KAPPA = np.array([0.812, 0.722, 0.618, 0.372, 0.109])   # 原典fit値を固定(θのみ動かす)
THETAS = np.arange(0, 180, 10.0)
CLTH = float(os.environ.get("CLTH", "30000"))            # クラスタ距離閾値(ft)

t0 = time.time()
wells = []
for fp in sorted(glob.glob(str(DATA / "train" / "*__horizontal_well.csv"))):
    w = pd.read_csv(fp, usecols=["X", "Y", "Z", "TVT", "TVT_input"])
    X, Y, Z = w["X"].to_numpy(), w["Y"].to_numpy(), w["Z"].to_numpy()
    tvt, ti = w["TVT"].to_numpy(), w["TVT_input"].to_numpy()
    fin = np.where(np.isfinite(ti))[0]
    if len(fin) < 10: continue
    s = int(fin.max()); ndz = -np.diff(Z)[s:]; n = len(ndz)
    if n < 30: continue
    U = np.cumsum(ndz); R0 = tvt[s + 1:] - tvt[s]
    t = np.arange(1, n + 1.0); edges = np.linspace(0, n, K + 1)
    phi = np.column_stack([np.clip(t - edges[j], 0, edges[j + 1] - edges[j]) for j in range(K)])
    mfin = np.isfinite(R0)
    c = np.linalg.lstsq(phi[mfin], (R0 - U)[mfin], rcond=None)[0]
    segid = np.clip(np.searchsorted(edges[1:], np.arange(1, n + 1.0), side="left"), 0, K - 1)
    mid = np.empty((K, 2)); az = np.empty(K)
    for j in range(K):
        f0 = s + 1 + int(edges[j]); f1 = min(s + 1 + max(int(edges[j + 1]) - 1, int(edges[j])), len(X) - 1)
        az[j] = np.degrees(np.arctan2(Y[f1] - Y[f0], X[f1] - X[f0])) % 360
        mid[j] = ((X[f0] + X[f1]) / 2, (Y[f0] + Y[f1]) / 2)
    wid = os.path.basename(fp).split("__")[0]
    wells.append(dict(wid=wid, wi=len(wells), ndz=ndz, R0=R0, c=c, segid=segid, mid=mid, az=az,
                      cx=float(np.nanmean(X[s:])), cy=float(np.nanmean(Y[s:])), ti_s=float(ti[s]), s=s))
print(f"{len(wells)}井 ({time.time()-t0:.0f}s)", flush=True)

# XYクラスタ(単純連結: 30000ft)
cid = -np.ones(len(wells), int)
cc = 0
pts = np.array([(w["cx"], w["cy"]) for w in wells])
for i in range(len(wells)):
    if cid[i] >= 0: continue
    stack = [i]; cid[i] = cc
    while stack:
        j = stack.pop()
        d = np.hypot(pts[:, 0] - pts[j, 0], pts[:, 1] - pts[j, 1])
        for k2 in np.where((d < CLTH) & (cid < 0))[0]:
            cid[k2] = cc; stack.append(int(k2))
    cc += 1
sizes = np.bincount(cid)
print(f"クラスタ {cc}個 sizes上位: {sorted(sizes,reverse=True)[:8]}", flush=True)


def loo_errs(theta, sub):
    """クラスタ内LOO誤差(行連結)を返す(場・予測ともθで)"""
    fld = []
    for wd in sub:
        proj = np.cos(np.radians(wd["az"] - theta))
        for j in range(K):
            if abs(proj[j]) > FIELD_MIN_PROJ:
                fld.append((wd["mid"][j, 0], wd["mid"][j, 1], wd["c"][j] / proj[j], wd["wi"]))
    if len(fld) < KNN_K + 1: return None
    fld = np.array(fld)
    es = []
    for wd in sub:
        proj = np.cos(np.radians(wd["az"] - theta))
        keep = fld[:, 3] != wd["wi"]
        fx, fy, fD = fld[keep, 0], fld[keep, 1], fld[keep, 2]
        if len(fx) < KNN_K: return None
        Dh = np.empty(K); dd = np.empty(K)
        for j in range(K):
            d2 = (fx - wd["mid"][j, 0]) ** 2 + (fy - wd["mid"][j, 1]) ** 2
            sel = np.argpartition(d2, KNN_K)[:KNN_K]
            wgt = np.exp(np.maximum(-d2[sel] / (2 * KNN_H ** 2), -700))
            Dh[j] = np.sum(wgt * fD[sel]) / np.sum(wgt)
            dd[j] = np.sqrt(np.median(d2[sel]))
        g = wd["ndz"] + (Dh * proj)[wd["segid"]]
        g = np.where(np.abs(proj[wd["segid"]]) < GATE, 0.0, g)
        kstep = KAPPA[np.digitize(dd, KBINS[1:-1])][wd["segid"]]
        e = np.cumsum(kstep * g) - wd["R0"]
        es.append(e[np.isfinite(e)])
    return np.concatenate(es)


tot_base, tot_best = [], []
theta_map = {}
for c_ in range(cc):
    sub = [w for w in wells if cid[w["wi"]] == c_]
    if len(sub) < 30:                                    # 小クラスタはθ適応しない(選択バイアス回避)
        e = loo_errs(118.4, sub)
        if e is not None: tot_base.append(e); tot_best.append(e)
        theta_map[c_] = 118.4
        continue
    base = loo_errs(118.4, sub)
    if base is None:
        theta_map[c_] = 118.4; continue
    rb = np.sqrt(np.mean(base ** 2))
    best = (rb, 118.4, base)
    for th in THETAS:
        e = loo_errs(float(th), sub)
        if e is None: continue
        r = np.sqrt(np.mean(e ** 2))
        if r < best[0]: best = (r, float(th), e)
    theta_map[c_] = best[1]
    tot_base.append(base); tot_best.append(best[2])
    print(f"cluster{c_}({len(sub)}井): θ118.4→{best[1]:.0f}  LOO {rb:.3f}→{best[0]:.3f}", flush=True)
eb = np.concatenate(tot_base); ee = np.concatenate(tot_best)
print(f"\n全体LOO: base(118.4固定)={np.sqrt(np.mean(eb**2)):.4f} → field-θ={np.sqrt(np.mean(ee**2)):.4f}  ({time.time()-t0:.0f}s)")
print("(注: θはクラスタ内LOOで選択=軽い選択バイアスあり。大幅改善時のみ採用検討)")
