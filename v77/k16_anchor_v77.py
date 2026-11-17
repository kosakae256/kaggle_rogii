# -*- coding: utf-8 -*-
"""
v77 — K16物理錨の前計算(検証/rogii_k16_spline_kernel_knn_adaptive_kappa_2 の忠実移植, train LOO)
出力: v77/artifacts_v77/k16_anchor_train.pkl = {wid: {tvt(eval行,), sig(eval行,), dd(eval行,)}}
 - tvt: K16モデルの行TVT予測(錨=最終既知TVT_input + cumsum(κ(dd)·g)) — 自坑井除外kNN(LOO)
 - sig: σ(dd)カーブ(LOO残差をddビンで較正)を行ddに適用
 - v38/v66 struct場は一切不使用(kappa_2のコードのみが源)。
使い方: python v77/k16_anchor_v77.py
"""
import os, io, sys, glob, pickle, time
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
from pathlib import Path
import numpy as np, pandas as pd

P0 = Path(r"C:\Users\kosaka256\Documents\rogii_claude")
DATA = P0 / "rogii-wellbore-geology-prediction"
ART = P0 / "v77" / "artifacts_v77"; ART.mkdir(parents=True, exist_ok=True)

THETA0 = float(os.environ.get("K16_TH", "118.4"))
K = int(os.environ.get("K16_SEG", "16"))
KNN_K = int(os.environ.get("K16_K", "15")); KNN_H = float(os.environ.get("K16_H", "500"))
GATE = 0.35; FIELD_MIN_PROJ = 0.3
SUFFIX = os.environ.get("K16_SUFFIX", "")
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


t0 = time.time()
train = []
for fp in sorted(glob.glob(str(DATA / "train" / "*__horizontal_well.csv"))):
    w = pd.read_csv(fp, usecols=["X", "Y", "Z", "TVT", "TVT_input"])
    X, Y, Z = w["X"].to_numpy(), w["Y"].to_numpy(), w["Z"].to_numpy()
    tvt, ti = w["TVT"].to_numpy(), w["TVT_input"].to_numpy()
    fin = np.where(np.isfinite(ti))[0]
    if len(fin) < 10: continue
    s = int(fin.max())
    ndz = -np.diff(Z)[s:]
    n = len(ndz)
    if n < 30: continue
    U = np.cumsum(ndz)
    R0 = tvt[s + 1:] - tvt[s]
    if not np.isfinite(R0).all():
        # 真値欠損行がある場合はfitからその行を除く(まれ)
        pass
    t = np.arange(1, n + 1.0)
    edges = np.linspace(0, n, K + 1)
    phi = np.column_stack([np.clip(t - edges[j], 0, edges[j + 1] - edges[j]) for j in range(K)])
    mfin = np.isfinite(R0)
    c = np.linalg.lstsq(phi[mfin], (R0 - U)[mfin], rcond=None)[0]
    segid, mid, proj = segment_geometry(X, Y, s, n)
    wid = os.path.basename(fp).split("__")[0]
    train.append(dict(wid=wid, wi=len(train), ndz=ndz, R0=R0, c=c, segid=segid, mid=mid, proj=proj,
                      s=s, ti_s=float(ti[s]), nrows=len(w)))
field = np.array([(wd["mid"][j, 0], wd["mid"][j, 1], wd["c"][j] / wd["proj"][j], wd["wi"])
                  for wd in train for j in range(K) if abs(wd["proj"][j]) > FIELD_MIN_PROJ])
print(f"train {len(train)}井  field {len(field)}点 ({time.time()-t0:.0f}s)", flush=True)

# κ: buffered-LOO 結合最小二乗(原典どおり)
A = np.zeros((NBIN, NBIN)); y = np.zeros(NBIN)
for R in KAPPA_REGIMES:
    for wd in train:
        Dh, dd = kernel_knn(field, wd["wi"], wd["mid"], min_dist=R)
        g = wd["ndz"] + (Dh * wd["proj"])[wd["segid"]]
        g = np.where(np.abs(wd["proj"][wd["segid"]]) < GATE, 0.0, g)
        bstep = np.digitize(dd, KBINS[1:-1])[wd["segid"]]
        G = np.column_stack([np.cumsum(np.where(bstep == b, g, 0.0)) for b in range(NBIN)])
        mfin = np.isfinite(wd["R0"])
        A += G[mfin].T @ G[mfin]; y += G[mfin].T @ wd["R0"][mfin]
kappa = np.linalg.solve(A, y)
print("kappa(ddビン別):", np.round(kappa, 3), flush=True)

# LOO予測(min_dist=0=通常LOO / K16_MINDIST>0でspace-blockゲート) + 残差→σ(dd)較正
MINDIST = float(os.environ.get("K16_MINDIST", "0"))
out = {}; res_by_bin = [[] for _ in range(NBIN)]
errs_all = []
for wd in train:
    Dh, dd = kernel_knn(field, wd["wi"], wd["mid"], min_dist=MINDIST)
    g = wd["ndz"] + (Dh * wd["proj"])[wd["segid"]]
    g = np.where(np.abs(wd["proj"][wd["segid"]]) < GATE, 0.0, g)
    kstep = kappa[np.digitize(dd, KBINS[1:-1])][wd["segid"]]
    pred = wd["ti_s"] + np.cumsum(kstep * g)                     # 行TVT予測(s+1..)
    true = wd["ti_s"] + wd["R0"]
    e = pred - true
    dd_row = dd[wd["segid"]]
    for b in range(NBIN):
        m = (np.digitize(dd_row, KBINS[1:-1]) == b) & np.isfinite(e)
        if m.any(): res_by_bin[b].extend(e[m].tolist())
    errs_all.append(e[np.isfinite(e)])
    out[wd["wid"]] = dict(tvt=pred.astype(np.float64), dd=dd_row.astype(np.float32), s=wd["s"])
sig_bin = np.array([np.sqrt(np.mean(np.array(r) ** 2)) if len(r) > 100 else 30.0 for r in res_by_bin])
print("σ(dd)ビン別:", np.round(sig_bin, 2), flush=True)
e = np.concatenate(errs_all)
print(f"K16 LOO pooled RMSE = {np.sqrt(np.mean(e**2)):.3f}ft (原典: 9.67@500ft)")
for wid in out:
    out[wid]["sig"] = sig_bin[np.digitize(out[wid]["dd"], KBINS[1:-1])].astype(np.float32)
fp = ART / f"k16_anchor_train{SUFFIX}.pkl"
fp.write_bytes(pickle.dumps(out, protocol=4))
print(f"[saved] {fp.name} ({len(out)}井, {time.time()-t0:.0f}s)")

# blend評価 vs 素PF(bare0, 773井)
bare = pickle.loads((P0 / "v73" / "artifacts_v73" / "seed_dump_bare0.pkl").read_bytes())
LS = 10.536319667317654
em, ep = [], []
for wid, d in bare.items():
    if wid not in out: continue
    ll = d["ll"].astype(np.float64)
    ww = np.exp((ll - ll.max()) / LS); ww /= ww.sum()
    epf = (ww[:, None] * d["err"].astype(np.float64)).sum(0)
    tv = d["true"]; a = out[wid]["tvt"]
    T = min(len(a), len(tv), len(epf))
    ea = a[:T] - tv[:T]; p = epf[:T]
    m = np.isfinite(ea) & np.isfinite(p)
    em.append(ea[m]); ep.append(p[m])
em = np.concatenate(em); ep = np.concatenate(ep)
sm = np.sqrt(np.mean(em ** 2)); sp = np.sqrt(np.mean(ep ** 2)); rho = np.corrcoef(em, ep)[0, 1]
best = min((float(np.sqrt(np.mean(((1 - w) * ep + w * em) ** 2))), float(w)) for w in np.linspace(0, 1, 101))
print(f"[blend] K16={sm:.4f} ρ={rho:.3f} w={best[1]:.2f} → {best[0]:.4f} (素PF{sp:.3f}比 {best[0]-sp:+.4f})")
