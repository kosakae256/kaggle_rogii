"""ROGII wellbore-geology prediction: K16 spline + kernel kNN + adaptive kappa(d) + proj gate.

Developed and LOO-validated locally (eda.ipynb): pooled per-point RMSE 9.67 ft at ~500 ft
donor distances, 11.53 ft at >=1500 ft. Test wells' per-segment donors sit at ~480-1130 ft.

  TVT(row) = TVT_input(PS) + cumsum( kappa_seg * ( -dZ + c_hat_seg ) )
  c_hat    = D_hat * cos(az_seg - THETA0);  D_hat = Gaussian-kernel kNN (k=15, h=500 ft)
             over the deprojected per-segment drift field of all training laterals
  kappa    = shrinkage per donor-distance bin, fit jointly over buffered-LOO regimes
  gate     = kappa 0 where |cos(az - THETA0)| < 0.35 (never fires on the 3 test wells)
"""
import glob
import os

import numpy as np
import pandas as pd

DATA = "/kaggle/input/rogii-wellbore-geology-prediction"
OUT = "/kaggle/working/submission.csv"
if not os.path.isdir(DATA):                                # local fallback
    import kagglehub
    DATA = kagglehub.competition_download("rogii-wellbore-geology-prediction")
    OUT = "submission.csv"

THETA0 = 118.4          # regional up-dip heading (deg, atan2 frame), fit from ANCC plane
K = 16                  # spline segments per lateral
KNN_K, KNN_H = 15, 500.0
GATE = 0.35
FIELD_MIN_PROJ = 0.3
KBINS = [0.0, 750.0, 1500.0, 2500.0, 4000.0, 1e18]
KAPPA_REGIMES = [0.0, 1000.0, 1500.0, 2000.0]
NBIN = len(KBINS) - 1


def segment_geometry(X, Y, s, n):
    edges = np.linspace(0, n, K + 1)
    segid = np.clip(np.searchsorted(edges[1:], np.arange(1, n + 1.0), side="left"), 0, K - 1)
    mid = np.empty((K, 2))
    proj = np.empty(K)
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
    Dh = np.empty(K)
    dd = np.empty(K)
    for j in range(K):
        d2 = (fx - mid[j, 0]) ** 2 + (fy - mid[j, 1]) ** 2
        cand = np.where(d2 >= min_dist * min_dist)[0] if min_dist else np.arange(len(d2))
        sel = cand[np.argpartition(d2[cand], KNN_K)[:KNN_K]]
        w = np.exp(np.maximum(-d2[sel] / (2 * KNN_H ** 2), -700))
        Dh[j] = np.sum(w * fD[sel]) / np.sum(w)
        dd[j] = np.sqrt(np.median(d2[sel]))
    return Dh, dd


# ---------- 1) training wells -> K=16 drift coefficients -> deprojected donor field ----------
train = []
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
    c = np.linalg.lstsq(phi, R0 - U, rcond=None)[0]
    assert np.isfinite(c).all(), fp
    segid, mid, proj = segment_geometry(X, Y, s, n)
    train.append(dict(wi=len(train), ndz=ndz, R0=R0, c=c, segid=segid, mid=mid, proj=proj))
field = np.array([(wd["mid"][j, 0], wd["mid"][j, 1], wd["c"][j] / wd["proj"][j], wd["wi"])
                  for wd in train for j in range(K) if abs(wd["proj"][j]) > FIELD_MIN_PROJ])
print(f"training wells: {len(train)}   field points: {len(field)}")

# ---------- 2) kappa per donor-distance bin: joint LOO fit across buffer regimes ----------
A = np.zeros((NBIN, NBIN))
y = np.zeros(NBIN)
for R in KAPPA_REGIMES:
    for wd in train:
        Dh, dd = kernel_knn(field, wd["wi"], wd["mid"], min_dist=R)
        g = wd["ndz"] + (Dh * wd["proj"])[wd["segid"]]
        g = np.where(np.abs(wd["proj"][wd["segid"]]) < GATE, 0.0, g)
        bstep = np.digitize(dd, KBINS[1:-1])[wd["segid"]]
        G = np.column_stack([np.cumsum(np.where(bstep == b, g, 0.0)) for b in range(NBIN)])
        A += G.T @ G
        y += G.T @ wd["R0"]
kappa = np.linalg.solve(A, y)
print("kappa per donor-distance bin:", np.round(kappa, 3))

# ---------- 3) predict the test wells (full field, no exclusion) ----------
preds = {}
for fp in sorted(glob.glob(os.path.join(DATA, "test", "*__horizontal_well.csv"))):
    wid = os.path.basename(fp).split("__")[0]
    w = pd.read_csv(fp, usecols=["X", "Y", "Z", "TVT_input"])
    X, Y, Z = w["X"].to_numpy(), w["Y"].to_numpy(), w["Z"].to_numpy()
    ti = w["TVT_input"].to_numpy()
    s = int(np.where(np.isfinite(ti))[0].max())
    ndz = -np.diff(Z)[s:]
    n = len(ndz)
    segid, mid, proj = segment_geometry(X, Y, s, n)
    Dh, dd = kernel_knn(field, -1, mid)
    g = ndz + (Dh * proj)[segid]
    g = np.where(np.abs(proj[segid]) < GATE, 0.0, g)
    kstep = kappa[np.digitize(dd, KBINS[1:-1])][segid]
    preds[wid] = dict(s=s, tvt=ti[s] + np.cumsum(kstep * g))
    print(f"{wid}: rows {s + 1}..{s + n}  anchor {ti[s]:.1f}  "
          f"donor dist [{dd.min():.0f}..{dd.max():.0f}] ft  gate fires {int((np.abs(proj) < GATE).sum())}/16")

# ---------- 4) align to sample_submission and write ----------
sub = pd.read_csv(os.path.join(DATA, "sample_submission.csv"))
parts = sub["id"].str.rsplit("_", n=1)
wids = parts.str[0].to_numpy()
ris = parts.str[1].astype(int).to_numpy()
out = np.empty(len(sub))
for wid, p in preds.items():
    m = wids == wid
    out[m] = p["tvt"][ris[m] - (p["s"] + 1)]
assert np.isfinite(out).all() and set(wids) == set(preds)
sub["tvt"] = out
sub.to_csv(OUT, index=False)
print(f"wrote {OUT} ({len(sub)} rows)")
