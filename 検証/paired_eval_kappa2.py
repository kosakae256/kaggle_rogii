# -*- coding: utf-8 -*-
"""採用判定・決定版: 全773井を5-fold部外者方式で両エンジンに予測させ、ペア比較(分割ガチャ排除)。
   fold外の井戸のみで場+κを構築 → val foldの井戸を部外者として予測(隠しtest同条件)。"""
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
wids = []
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
    wids.append(os.path.basename(fp).split("__")[0])

os.environ.update(SEG_LEN="0", VEC="1", QW="0", ROBUST="0", STEREO="0",
                  REGIMES_EXT="0", BACK_DA="0")
sys.path.insert(0, r"c:\Users\kosaka256\Documents\rogii_claude\v74")
import phaseG_struct2_v74 as G
gws = []
for fp in sorted(glob.glob(os.path.join(DATA, "train", "*__horizontal_well.csv"))):
    wd = G.load_well(fp)
    wd["wi"] = len(gws)
    wd["seg"] = G.segment_well(wd)
    gws.append(wd)

kf = list(KFold(5, shuffle=True, random_state=42).split(np.arange(len(wells))))
ES_s, ES_g, PER = [], [], []
for fold, (tr_i, va_i) in enumerate(kf):
    tr = set(tr_i.tolist())
    field = np.array([(wd["mid"][j, 0], wd["mid"][j, 1], wd["c"][j] / wd["proj"][j], wd["wi"])
                      for wd in wells if wd["wi"] in tr
                      for j in range(K) if abs(wd["proj"][j]) > FIELD_MIN_PROJ])
    A = np.zeros((NBIN, NBIN)); yv = np.zeros(NBIN)
    for R in KAPPA_REGIMES:
        for wd in wells:
            if wd["wi"] not in tr:
                continue
            Dh, dd = kernel_knn(field, wd["wi"], wd["mid"], min_dist=R)
            g = wd["ndz"] + (Dh * wd["proj"])[wd["segid"]]
            g = np.where(np.abs(wd["proj"][wd["segid"]]) < GATE, 0.0, g)
            bstep = np.digitize(dd, KBINS[1:-1])[wd["segid"]]
            Gm = np.column_stack([np.cumsum(np.where(bstep == b, g, 0.0)) for b in range(NBIN)])
            A += Gm.T @ Gm; yv += Gm.T @ wd["R0"]
    kappa = np.linalg.solve(A, yv)
    F2 = G.build_field(gws, tr)
    K2 = G.fit_kappa(F2, gws, tr)
    for i in va_i:
        wd = wells[i]
        Dh, dd = kernel_knn(field, -1, wd["mid"])
        g = wd["ndz"] + (Dh * wd["proj"])[wd["segid"]]
        g = np.where(np.abs(wd["proj"][wd["segid"]]) < GATE, 0.0, g)
        kstep = kappa[np.digitize(dd, KBINS[1:-1])][wd["segid"]]
        pred_s = wd["anchor"] + np.cumsum(kstep * g)
        gw = gws[i]
        pred_g, _, _, _ = G.predict(F2, gw, K2)
        ev = wd["ev"]
        es = (pred_s - wd["tvt_tail"])[ev]; es = es[np.isfinite(es)]
        eg = (pred_g - gw["tvt_tail"])[gw["ev_tail"] & np.isfinite(gw["tvt_tail"])]
        eg = eg[np.isfinite(eg)]
        if len(es) and len(eg):
            ES_s.append(es); ES_g.append(eg)
            PER.append((float(np.sqrt(np.mean(es ** 2))), float(np.sqrt(np.mean(eg ** 2))),
                        len(es)))
    print(f"fold{fold} done", flush=True)
es = np.concatenate(ES_s); eg = np.concatenate(ES_g)
rmse = lambda e: float(np.sqrt(np.mean(e ** 2)))
P = np.array(PER)
print(f"\n[部外者5-fold ペア評価, 全{len(PER)}井]")
print(f"  script: overall {rmse(es):.4f}")
print(f"  PhaseG: overall {rmse(eg):.4f}")
win = int((P[:, 0] < P[:, 1]).sum())
print(f"  井戸別勝敗: script勝ち {win}/{len(P)}")
d = P[:, 0] - P[:, 1]
print(f"  井戸別RMSE差(script−G): median {np.median(d):+.3f}  mean {np.mean(d):+.3f}")
# 50:50ブレンドも参考に
eb = 0.5 * es + 0.5 * eg if len(es) == len(eg) else None
if eb is not None:
    print(f"  等重みブレンド: {rmse(eb):.4f}  (誤差相関 {np.corrcoef(es, eg)[0, 1]:.3f})")
