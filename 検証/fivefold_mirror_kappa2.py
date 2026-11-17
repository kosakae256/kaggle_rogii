# -*- coding: utf-8 -*-
"""提出パイプライン鏡写し検証: test20%隔離 → trainval80%で5-fold κ → OOF CV vs test(5モデル平均推論)。
   10シードで CV↔test の一致度を分布として測る。scriptエンジン(kappa_2)使用。"""
import io, sys, os, glob
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

DATA = r"c:\Users\kosaka256\Documents\rogii_claude\rogii-wellbore-geology-prediction"
THETA0 = 118.4; K = 16; KNN_K, KNN_H = 15, 500.0
GATE = 0.35; FIELD_MIN_PROJ = 0.3
KBINS = [0.0, 750.0, 1500.0, 2500.0, 4000.0, 1e18]
KAPPA_REGIMES = [0.0, 1000.0, 1500.0, 2000.0]
NBIN = len(KBINS) - 1
N_SEEDS = int(os.environ.get("N_SEEDS", "10"))

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

def predict_one(wd, field, kappa, own):
    Dh, dd = kernel_knn(field, own, wd["mid"])
    g = wd["ndz"] + (Dh * wd["proj"])[wd["segid"]]
    g = np.where(np.abs(wd["proj"][wd["segid"]]) < GATE, 0.0, g)
    kstep = kappa[np.digitize(dd, KBINS[1:-1])][wd["segid"]]
    return wd["anchor"] + np.cumsum(kstep * g)

def run_seed(seed):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_all)
    n_te = n_all // 5
    idx_te = perm[:n_te].tolist()
    idx_tv = perm[n_te:].tolist()
    tv_set = set(idx_tv)
    field = np.array([(wd["mid"][j, 0], wd["mid"][j, 1], wd["c"][j] / wd["proj"][j], wd["wi"])
                      for wd in wells if wd["wi"] in tv_set
                      for j in range(K) if abs(wd["proj"][j]) > FIELD_MIN_PROJ])
    # 設計ブロック(場=trainval-LOO)を各wellで1回 → fold別に加算
    blocks = {}
    for wi in idx_tv:
        wd = wells[wi]
        A = np.zeros((NBIN, NBIN)); yv = np.zeros(NBIN)
        for R in KAPPA_REGIMES:
            Dh, dd = kernel_knn(field, wd["wi"], wd["mid"], min_dist=R)
            g = wd["ndz"] + (Dh * wd["proj"])[wd["segid"]]
            g = np.where(np.abs(wd["proj"][wd["segid"]]) < GATE, 0.0, g)
            bstep = np.digitize(dd, KBINS[1:-1])[wd["segid"]]
            G = np.column_stack([np.cumsum(np.where(bstep == b, g, 0.0)) for b in range(NBIN)])
            A += G.T @ G; yv += G.T @ wd["R0"]
        blocks[wi] = (A, yv)
    folds = list(KFold(5, shuffle=True, random_state=seed).split(np.arange(len(idx_tv))))
    kset = []
    fold_of = {}
    for fi, (tr_j, va_j) in enumerate(folds):
        A = np.zeros((NBIN, NBIN)); yv = np.zeros(NBIN)
        for j in tr_j:
            wi = idx_tv[j]
            A += blocks[wi][0]; yv += blocks[wi][1]
        kset.append(np.linalg.solve(A, yv))
        for j in va_j:
            fold_of[idx_tv[j]] = fi
    # OOF CV(場=trainval-LOO, κ=自foldモデル)
    ES = []; pw = []
    for wi in idx_tv:
        wd = wells[wi]
        pred = predict_one(wd, field, kset[fold_of[wi]], wd["wi"])
        e = (pred - wd["tvt_tail"])[wd["ev"]]
        e = e[np.isfinite(e)]
        if len(e):
            ES.append(e); pw.append(float(np.sqrt(np.mean(e ** 2))))
    cv_pool = float(np.sqrt(np.mean(np.concatenate(ES) ** 2)))
    cv_med = float(np.median(pw))
    # test: 5モデル全部で推論して平均
    ES = []; pw = []
    for wi in idx_te:
        wd = wells[wi]
        preds = np.mean([predict_one(wd, field, kp, -1) for kp in kset], axis=0)
        e = (preds - wd["tvt_tail"])[wd["ev"]]
        e = e[np.isfinite(e)]
        if len(e):
            ES.append(e); pw.append(float(np.sqrt(np.mean(e ** 2))))
    te_pool = float(np.sqrt(np.mean(np.concatenate(ES) ** 2)))
    te_med = float(np.median(pw))
    return cv_pool, cv_med, te_pool, te_med

rows = []
for seed in range(N_SEEDS):
    cvp, cvm, tep, tem = run_seed(seed)
    rows.append((cvp, cvm, tep, tem))
    print(f"seed{seed}: CV(5foldOOF)={cvp:6.2f} (med {cvm:5.2f})  "
          f"test(5モデル平均)={tep:6.2f} (med {tem:5.2f})  比={tep/cvp:.3f}", flush=True)
R = np.array(rows)
print("\n== 集計 ==")
print(f"CV   : pooled {R[:,0].mean():.2f}±{R[:,0].std():.2f}  med {R[:,1].mean():.2f}±{R[:,1].std():.2f}")
print(f"test : pooled {R[:,2].mean():.2f}±{R[:,2].std():.2f}  med {R[:,3].mean():.2f}±{R[:,3].std():.2f}")
r = R[:, 2] / R[:, 0]
print(f"CV→test比: {r.mean():.3f}±{r.std():.3f}  (min {r.min():.2f} / max {r.max():.2f})")
print(f"med比    : {(R[:,3]/R[:,1]).mean():.3f}±{(R[:,3]/R[:,1]).std():.3f}")
