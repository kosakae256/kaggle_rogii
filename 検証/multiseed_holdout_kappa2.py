# -*- coding: utf-8 -*-
"""多シード3分割: script エンジンで seed 0..9 の train(LOO)/valid/test を測定。
   pooled(行重み)と井戸中央値の両方を記録し、CV↔疑似testの一致度を分布で見る。"""
import io, sys, os, glob
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
import numpy as np
import pandas as pd

DATA = r"c:\Users\kosaka256\Documents\rogii_claude\rogii-wellbore-geology-prediction"
THETA0 = 118.4; K = 16; KNN_K, KNN_H = 15, 500.0
GATE = 0.35; FIELD_MIN_PROJ = 0.3
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
        kk = min(KNN_K, len(cand))
        sel = cand[np.argpartition(d2[cand], kk - 1)[:kk]]
        w = np.exp(np.maximum(-d2[sel] / (2 * KNN_H ** 2), -700))
        Dh[j] = np.sum(w * fD[sel]) / np.sum(w)
        dd[j] = np.sqrt(np.median(d2[sel]))
    return Dh, dd

wells = []
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
    wells.append(dict(wi=len(wells), ndz=ndz, R0=np.nan_to_num(R0), c=c, segid=segid,
                      mid=mid, proj=proj, tvt_tail=tvt[s + 1:], anchor=float(ti[s]), ev=ev))
n_all = len(wells)

def run_seed(seed):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_all)
    n_tr = int(n_all * 0.6); n_va = (n_all - n_tr) // 2
    idx_tr = set(perm[:n_tr].tolist())
    idx_va = set(perm[n_tr:n_tr + n_va].tolist())
    idx_te = set(perm[n_tr + n_va:].tolist())
    field = np.array([(wd["mid"][j, 0], wd["mid"][j, 1], wd["c"][j] / wd["proj"][j], wd["wi"])
                      for wd in wells if wd["wi"] in idx_tr
                      for j in range(K) if abs(wd["proj"][j]) > FIELD_MIN_PROJ])
    A = np.zeros((NBIN, NBIN)); yv = np.zeros(NBIN)
    for R in KAPPA_REGIMES:
        for wd in wells:
            if wd["wi"] not in idx_tr:
                continue
            Dh, dd = kernel_knn(field, wd["wi"], wd["mid"], min_dist=R)
            g = wd["ndz"] + (Dh * wd["proj"])[wd["segid"]]
            g = np.where(np.abs(wd["proj"][wd["segid"]]) < GATE, 0.0, g)
            bstep = np.digitize(dd, KBINS[1:-1])[wd["segid"]]
            G = np.column_stack([np.cumsum(np.where(bstep == b, g, 0.0)) for b in range(NBIN)])
            A += G.T @ G; yv += G.T @ wd["R0"]
    kappa = np.linalg.solve(A, yv)

    def ev_part(part, own_excl):
        ES = []; pw = []
        for wd in wells:
            if wd["wi"] not in part:
                continue
            Dh, dd = kernel_knn(field, wd["wi"] if own_excl else -1, wd["mid"])
            g = wd["ndz"] + (Dh * wd["proj"])[wd["segid"]]
            g = np.where(np.abs(wd["proj"][wd["segid"]]) < GATE, 0.0, g)
            kstep = kappa[np.digitize(dd, KBINS[1:-1])][wd["segid"]]
            pred = wd["anchor"] + np.cumsum(kstep * g)
            e = (pred - wd["tvt_tail"])[wd["ev"]]
            e = e[np.isfinite(e)]
            if len(e):
                ES.append(e); pw.append(float(np.sqrt(np.mean(e ** 2))))
        es = np.concatenate(ES)
        return float(np.sqrt(np.mean(es ** 2))), float(np.median(pw))
    tr = ev_part(idx_tr, True)
    va = ev_part(idx_va, False)
    te = ev_part(idx_te, False)
    return tr, va, te

rows = []
for seed in range(10):
    tr, va, te = run_seed(seed)
    rows.append((seed, tr[0], va[0], te[0], tr[1], va[1], te[1]))
    print(f"seed{seed}: pooled train={tr[0]:6.2f} valid={va[0]:6.2f} test={te[0]:6.2f} | "
          f"井戸med train={tr[1]:5.2f} valid={va[1]:5.2f} test={te[1]:5.2f}", flush=True)
R = np.array(rows)
print("\n== 集計(10 seeds) ==")
print(f"pooled : train {R[:,1].mean():.2f}±{R[:,1].std():.2f}  valid {R[:,2].mean():.2f}±{R[:,2].std():.2f}  test {R[:,3].mean():.2f}±{R[:,3].std():.2f}")
print(f"井戸med: train {R[:,4].mean():.2f}±{R[:,4].std():.2f}  valid {R[:,5].mean():.2f}±{R[:,5].std():.2f}  test {R[:,6].mean():.2f}±{R[:,6].std():.2f}")
print(f"pooled: valid-test相関 {np.corrcoef(R[:,2],R[:,3])[0,1]:.2f}  |valid−test|平均 {np.mean(np.abs(R[:,2]-R[:,3])):.2f}")
print(f"train(CV)→test転移率: {(R[:,3]/R[:,1]).mean():.3f}±{(R[:,3]/R[:,1]).std():.3f}  (min {(R[:,3]/R[:,1]).min():.2f} / max {(R[:,3]/R[:,1]).max():.2f})")
