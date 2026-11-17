# -*- coding: utf-8 -*-
"""
v78 — test側 K16錨生成(隠しtest対応)
=====================================
場・κ・σ(dd)は train全体から構築(test井は自分がドナーに居ないので除外不要=LOO不要)。
κとσ(dd)ビンは train LOO時と同一手順で再フィット(=k16_anchor_v77.pyと同じ値になる)。
test各井: TVT_input既知区間の最終点を錨に、eval行(TVT_input NaN)の tvt/sig/dd を出力。
出力: v77/artifacts_v77/k16_anchor_test.pkl {wid:{tvt,sig,dd,s}}
使い方: python v77/k16_anchor_test_v77.py [TESTDIR=official test dir]
"""
import os, io, sys, glob, pickle, time
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
from pathlib import Path
import numpy as np, pandas as pd

P0 = Path(r"C:\Users\kosaka256\Documents\rogii_claude")
DATA = P0 / "rogii-wellbore-geology-prediction"
TESTDIR = Path(os.environ.get("TESTDIR", str(DATA / "test")))
ART = P0 / "v77" / "artifacts_v77"; ART.mkdir(parents=True, exist_ok=True)

THETA0 = 118.4; K = 16; KNN_K, KNN_H = 15, 500.0; GATE = 0.35; FIELD_MIN_PROJ = 0.3
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
    keep = field[:, 3] != own_wi          # own_wi=-1 でtest(除外なし)
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


t0 = time.time()
# --- train場+κ+σ(dd): k16_anchor_v77.py と同一 ---
train = []
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
    segid, mid, proj = segment_geometry(X, Y, s, n)
    train.append(dict(wi=len(train), ndz=ndz, R0=R0, c=c, segid=segid, mid=mid, proj=proj))
field = np.array([(wd["mid"][j, 0], wd["mid"][j, 1], wd["c"][j] / wd["proj"][j], wd["wi"])
                  for wd in train for j in range(K) if abs(wd["proj"][j]) > FIELD_MIN_PROJ])
A = np.zeros((NBIN, NBIN)); yv = np.zeros(NBIN)
for R in KAPPA_REGIMES:
    for wd in train:
        Dh, dd = kernel_knn(field, wd["wi"], wd["mid"], min_dist=R)
        g = wd["ndz"] + (Dh * wd["proj"])[wd["segid"]]
        g = np.where(np.abs(wd["proj"][wd["segid"]]) < GATE, 0.0, g)
        bstep = np.digitize(dd, KBINS[1:-1])[wd["segid"]]
        G = np.column_stack([np.cumsum(np.where(bstep == b, g, 0.0)) for b in range(NBIN)])
        mfin = np.isfinite(wd["R0"])
        A += G[mfin].T @ G[mfin]; yv += G[mfin].T @ wd["R0"][mfin]
kappa = np.linalg.solve(A, yv)
res_by_bin = [[] for _ in range(NBIN)]
for wd in train:                                     # σ(dd)較正はLOO残差(train側)
    Dh, dd = kernel_knn(field, wd["wi"], wd["mid"])
    g = wd["ndz"] + (Dh * wd["proj"])[wd["segid"]]
    g = np.where(np.abs(wd["proj"][wd["segid"]]) < GATE, 0.0, g)
    kstep = kappa[np.digitize(dd, KBINS[1:-1])][wd["segid"]]
    e = np.cumsum(kstep * g) - wd["R0"]
    dd_row = dd[wd["segid"]]
    for b in range(NBIN):
        m = (np.digitize(dd_row, KBINS[1:-1]) == b) & np.isfinite(e)
        if m.any(): res_by_bin[b].extend(e[m].tolist())
sig_bin = np.array([np.sqrt(np.mean(np.array(r) ** 2)) if len(r) > 100 else 30.0 for r in res_by_bin])
print(f"train場 {len(field)}点  kappa={np.round(kappa,3)}  σ={np.round(sig_bin,2)} ({time.time()-t0:.0f}s)", flush=True)

# --- test井へ適用(除外なし=own_wi -1) ---
out = {}
for fp in sorted(glob.glob(str(TESTDIR / "*__horizontal_well.csv"))):
    wid = os.path.basename(fp).split("__")[0]
    w = pd.read_csv(fp, usecols=lambda c: c in ("X", "Y", "Z", "TVT_input"))
    X, Y, Z = w["X"].to_numpy(), w["Y"].to_numpy(), w["Z"].to_numpy()
    ti = w["TVT_input"].to_numpy()
    fin = np.where(np.isfinite(ti))[0]
    if len(fin) < 10: continue
    s = int(fin.max()); ndz = -np.diff(Z)[s:]; n = len(ndz)
    if n < 1: continue
    segid, mid, proj = segment_geometry(X, Y, s, n)
    Dh, dd = kernel_knn(field, -1, mid)
    g = ndz + (Dh * proj)[segid]
    g = np.where(np.abs(proj[segid]) < GATE, 0.0, g)
    kstep = kappa[np.digitize(dd, KBINS[1:-1])][segid]
    pred = float(ti[s]) + np.cumsum(kstep * g)
    dd_row = dd[segid]
    out[wid] = dict(tvt=pred.astype(np.float64),
                    sig=sig_bin[np.digitize(dd_row, KBINS[1:-1])].astype(np.float32),
                    dd=dd_row.astype(np.float32), s=s)
fp = ART / "k16_anchor_test.pkl"
fp.write_bytes(pickle.dumps(out, protocol=4))
print(f"[saved] {fp.name} ({len(out)}井, {time.time()-t0:.0f}s)")
