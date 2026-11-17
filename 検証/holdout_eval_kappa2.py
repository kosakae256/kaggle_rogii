# -*- coding: utf-8 -*-
"""採用判定: rogii_k16_spline_kernel_knn_adaptive_kappa_2 の train/valid/test 3分割評価。
   - 井戸ランダム3分割(465/154/154, seed=42)
   - 場ドナー・κとも train のみから構築(valid/test は完全部外者=隠しtest同条件)
   - 対照: Phase G エンジン(同一プロトコル)
   出力: 分割別 overall + dd構成 + dd重み付き(τ2279)LB予想
"""
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
SEED = int(os.environ.get("SEED", "42"))

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

# ---------- 全井戸ロード(script流) ----------
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
rng = np.random.default_rng(SEED)
perm = rng.permutation(n_all)
n_tr = int(n_all * 0.6); n_va = (n_all - n_tr) // 2
idx_tr = set(perm[:n_tr].tolist())
idx_va = set(perm[n_tr:n_tr + n_va].tolist())
idx_te = set(perm[n_tr + n_va:].tolist())
print(f"split: train={len(idx_tr)} valid={len(idx_va)} test={len(idx_te)}  (seed={SEED})")

# ---------- script エンジン: 場・κ = train のみ ----------
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
print("script κ(trainのみfit):", np.round(kappa, 3))

def eval_script(part, own_excl):
    ES = []; DD = []
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
            ES.append(e); DD.append((float(np.median(dd)), len(e)))
    es = np.concatenate(ES)
    dd_arr = np.array(DD)
    return (float(np.sqrt(np.mean(es ** 2))),
            float(np.sum(dd_arr[:, 0] * dd_arr[:, 1]) / np.sum(dd_arr[:, 1])), ES, dd_arr)

r_tr = eval_script(idx_tr, True)
r_va = eval_script(idx_va, False)
r_te = eval_script(idx_te, False)
print(f"\n[script] train(LOO)={r_tr[0]:.4f} (平均dd {r_tr[1]:.0f}ft)")
print(f"[script] valid     ={r_va[0]:.4f} (平均dd {r_va[1]:.0f}ft)")
print(f"[script] test      ={r_te[0]:.4f} (平均dd {r_te[1]:.0f}ft)")

# ---------- Phase G エンジン: 同一プロトコル ----------
os.environ.update(SEG_LEN="0", VEC="1", QW="0", ROBUST="0", STEREO="0",
                  REGIMES_EXT="0", BACK_DA="0")
sys.path.insert(0, r"c:\Users\kosaka256\Documents\rogii_claude\v74")
import phaseG_struct2_v74 as G
gws = []
for fp in sorted(glob.glob(os.path.join(DATA, "train", "*__horizontal_well.csv"))):
    wd = G.load_well(fp)
    if wd is None:
        continue
    wd["wi"] = len(gws)
    wd["seg"] = G.segment_well(wd)
    gws.append(wd)
F2 = G.build_field(gws, idx_tr)
K2 = G.fit_kappa(F2, gws, idx_tr)

def eval_g(part):
    ES = []
    for wd in gws:
        if wd["wi"] not in part or wd.get("seg") is None:
            continue
        pred, _, _, _ = G.predict(F2, wd, K2)
        ev = wd["ev_tail"] & np.isfinite(wd["tvt_tail"])
        e = (pred - wd["tvt_tail"])[ev]
        e = e[np.isfinite(e)]
        if len(e):
            ES.append(e)
    es = np.concatenate(ES)
    return float(np.sqrt(np.mean(es ** 2)))

print(f"\n[PhaseG]  train(LOO)={eval_g(idx_tr):.4f}")
print(f"[PhaseG]  valid     ={eval_g(idx_va):.4f}")
print(f"[PhaseG]  test      ={eval_g(idx_te):.4f}")
