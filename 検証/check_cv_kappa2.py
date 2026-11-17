# -*- coding: utf-8 -*-
"""検証: rogii_k16_spline_kernel_knn_adaptive_kappa_2.py の LOO CV を同一コードパスで測る。
   ヘッダ主張: pooled per-point RMSE 9.67 ft (~500ft donors) / 11.53 ft (>=1500ft)。
   手順: 本体と同じ field/κ(全well一括fit=本体準拠) → 各train井をown除外でLOO予測 →
         overall pooled + ドナー距離帯別 + 参考: Phase G(fold-honest 9.832)との比較。"""
import io, sys, os, glob
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
import numpy as np
import pandas as pd

DATA = r"c:\Users\kosaka256\Documents\rogii_claude\rogii-wellbore-geology-prediction"
THETA0 = 118.4
K = 16
KNN_K, KNN_H = 15, 500.0
GATE = 0.35
FIELD_MIN_PROJ = 0.3
KBINS = [0.0, 750.0, 1500.0, 2500.0, 4000.0, 1e18]
KAPPA_REGIMES = [0.0, 1000.0, 1500.0, 2000.0]
NBIN = len(KBINS) - 1

def segment_geometry(X, Y, s, n):
    edges = np.linspace(0, n, K + 1)
    segid = np.clip(np.searchsorted(edges[1:], np.arange(1, n + 1.0), side="left"), 0, K - 1)
    mid = np.empty((K, 2)); proj = np.empty(K)
    for j in range(K):
        f0 = s + 1 + int(edges[j])
        f1 = min(s + 1 + max(int(edges[j + 1]) - 1, int(edges[j])), len(X) - 1)
        az = np.degrees(np.arctan2(Y[f1] - Y[f0], X[f1] - X[f0])) % 360
        mid[j] = ((X[f0] + X[f1]) / 2, (Y[f0] + Y[f1]) / 2)
        proj[j] = np.cos(np.radians(az - THETA0))
    return segid, mid, proj

def kernel_knn(field, own_wi, mid, min_dist=0.0):
    keep = field[:, 3] != own_wi
    fx, fy, fD = field[keep, 0], field[keep, 1], field[keep, 2]
    Dh = np.empty(K); dd = np.empty(K)
    for j in range(K):
        d2 = (fx - mid[j, 0]) ** 2 + (fy - mid[j, 1]) ** 2
        cand = np.where(d2 >= min_dist * min_dist)[0] if min_dist else np.arange(len(d2))
        sel = cand[np.argpartition(d2[cand], KNN_K)[:KNN_K]]
        w = np.exp(np.maximum(-d2[sel] / (2 * KNN_H ** 2), -700))
        Dh[j] = np.sum(w * fD[sel]) / np.sum(w)
        dd[j] = np.sqrt(np.median(d2[sel]))
    return Dh, dd

train = []
truth = []
for fp in sorted(glob.glob(os.path.join(DATA, "train", "*__horizontal_well.csv"))):
    w = pd.read_csv(fp, usecols=["X", "Y", "Z", "TVT", "TVT_input"])
    X, Y, Z = w["X"].to_numpy(), w["Y"].to_numpy(), w["Z"].to_numpy()
    tvt, ti = w["TVT"].to_numpy(), w["TVT_input"].to_numpy()
    s = int(np.where(np.isfinite(ti))[0].max())
    ndz = -np.diff(Z)[s:]
    n = len(ndz)
    U = np.cumsum(ndz)
    R0 = tvt[s + 1:] - tvt[s]
    t = np.arange(1, n + 1.0)
    edges = np.linspace(0, n, K + 1)
    phi = np.column_stack([np.clip(t - edges[j], 0, edges[j + 1] - edges[j]) for j in range(K)])
    c = np.linalg.lstsq(phi, np.nan_to_num(R0 - U), rcond=None)[0]
    segid, mid, proj = segment_geometry(X, Y, s, n)
    ev = (~np.isfinite(ti[s + 1:])) & np.isfinite(tvt[s + 1:])
    train.append(dict(wi=len(train), ndz=ndz, R0=np.nan_to_num(R0), c=c,
                      segid=segid, mid=mid, proj=proj,
                      tvt_tail=tvt[s + 1:], anchor=float(ti[s]), ev=ev))
field = np.array([(wd["mid"][j, 0], wd["mid"][j, 1], wd["c"][j] / wd["proj"][j], wd["wi"])
                  for wd in train for j in range(K) if abs(wd["proj"][j]) > FIELD_MIN_PROJ])
print(f"wells={len(train)}  field points={len(field)}")

A = np.zeros((NBIN, NBIN)); y = np.zeros(NBIN)
for R in KAPPA_REGIMES:
    for wd in train:
        Dh, dd = kernel_knn(field, wd["wi"], wd["mid"], min_dist=R)
        g = wd["ndz"] + (Dh * wd["proj"])[wd["segid"]]
        g = np.where(np.abs(wd["proj"][wd["segid"]]) < GATE, 0.0, g)
        bstep = np.digitize(dd, KBINS[1:-1])[wd["segid"]]
        G = np.column_stack([np.cumsum(np.where(bstep == b, g, 0.0)) for b in range(NBIN)])
        A += G.T @ G; y += G.T @ wd["R0"]
kappa = np.linalg.solve(A, y)
print("kappa:", np.round(kappa, 3))

ES = []; perw = []
for wd in train:
    Dh, dd = kernel_knn(field, wd["wi"], wd["mid"])
    g = wd["ndz"] + (Dh * wd["proj"])[wd["segid"]]
    g = np.where(np.abs(wd["proj"][wd["segid"]]) < GATE, 0.0, g)
    kstep = kappa[np.digitize(dd, KBINS[1:-1])][wd["segid"]]
    pred = wd["anchor"] + np.cumsum(kstep * g)
    e = (pred - wd["tvt_tail"])[wd["ev"]]
    e = e[np.isfinite(e)]
    if len(e):
        ES.append(e)
        perw.append((float(np.median(dd)), float(np.sqrt(np.mean(e ** 2))), len(e)))
es = np.concatenate(ES)
overall = float(np.sqrt(np.mean(es ** 2)))
pw = np.array(perw)
dm = float(np.sqrt(np.mean(np.concatenate([e - np.median(e) for e in ES]) ** 2)))
print(f"\nLOO overall pooled RMSE = {overall:.4f}  (demean {dm:.4f})")
print("ドナー距離帯別(井戸のmedian dd, 行重みプール):")
bins = [0, 300, 600, 900, 1200, 1500, 2500, 1e9]
for a, b in zip(bins[:-1], bins[1:]):
    m = (pw[:, 0] >= a) & (pw[:, 0] < b)
    if m.sum() == 0:
        continue
    r = float(np.sqrt(np.sum(pw[m, 2] * pw[m, 1] ** 2) / np.sum(pw[m, 2])))
    print(f"  dd {a:>5.0f}-{b if b < 1e9 else 9999:>5.0f}ft: wells={int(m.sum()):3d}  RMSE={r:6.2f}")
m = pw[:, 0] >= 1500
if m.sum():
    r = float(np.sqrt(np.sum(pw[m, 2] * pw[m, 1] ** 2) / np.sum(pw[m, 2])))
    print(f"  [ヘッダ主張チェック] dd>=1500ft: {r:.2f} (主張 11.53)")
m = (pw[:, 0] >= 350) & (pw[:, 0] < 700)
if m.sum():
    r = float(np.sqrt(np.sum(pw[m, 2] * pw[m, 1] ** 2) / np.sum(pw[m, 2])))
    print(f"  [ヘッダ主張チェック] dd~500ft(350-700): {r:.2f} (主張 9.67)")
print("\n参考: Phase G(fold-honest κ, 弧長セグメント, α/β分離+リッジ) overall 9.832 / 本家struct 9.703")
