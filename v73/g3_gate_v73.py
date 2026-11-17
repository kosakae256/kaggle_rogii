# -*- coding: utf-8 -*-
"""
v73 段g-3 — 学習モードゲート(v34式 soft posterior-mean のチェーン版)
====================================================================
モード特徴→P(正解モード)をロジスティックで学習(fold-OOF, 坑井sorted%5)、
全モードを P^λ·mass^(1−λ) で再加重した posterior-mean(hard選択しない=v34の生き残り形)。
使い方: DUMPTAG=champ2 python v73/g3_gate_v73.py
"""
import os, io, sys, pickle
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression

ART = Path(r"C:\Users\kosaka256\Documents\rogii_claude\v73\artifacts_v73")
TAG = os.environ.get("DUMPTAG", "champ2")
GAP = 6.0; LS = 10.536319667317654
D = pickle.loads((ART / f"seed_dump_{TAG}.pkl").read_bytes())
wells = sorted(D.keys()); fold_of = {w: i % 5 for i, w in enumerate(wells)}


def wsoft(ll):
    m = np.isfinite(ll); mx = np.nanmax(ll[m]) if m.any() else 0.0
    lw = np.where(m, ll - mx, -np.inf)
    ww = np.exp(lw / LS); s = ww.sum()
    return ww / s if s > 1e-300 else np.full(len(ll), 1.0 / len(ll))


# ---- モード分解 + 特徴 + ラベル(=そのモードがRMSE最小か) ----
rows = []; well_modes = {}
for w in wells:
    d = D[w]; err = d["err"].astype(np.float32); ll = d["ll"]; S, T = err.shape
    ww = wsoft(ll); ec = (ww[:, None] * err).sum(0); fin = np.isfinite(ec)
    if fin.sum() == 0: continue
    off = np.nanmedian(err - ec[None, :], axis=1)
    order = np.argsort(off); gaps = np.diff(off[order]); cuts = set(np.where(gaps > GAP)[0])
    mode_id = np.zeros(S, int); mid = 0
    for j, si in enumerate(order):
        if j > 0 and (j - 1) in cuts: mid += 1
        mode_id[si] = mid
    M = mid + 1
    ems = []; feats = []; rms = []
    for m_ in range(M):
        sel = mode_id == m_
        mass = float(ww[sel].sum())
        wwm = ww[sel] / max(ww[sel].sum(), 1e-12)
        em = (wwm[:, None] * err[sel]).sum(0)
        offm = float(np.nanmedian(off[sel]))
        spread = float(np.nanstd(off[sel])) if sel.sum() > 1 else 0.0
        llm = ll[sel]
        f = [mass, np.log(mass + 1e-4), abs(offm), spread, sel.sum() / S,
             float(np.nanmax(llm) - np.nanmax(ll)), M, np.log1p(T) / 10.0]
        ems.append(em); feats.append(f); rms.append(float(np.sqrt(np.nanmean(em[fin] ** 2))))
    best = int(np.argmin(rms))
    well_modes[w] = dict(ems=ems, fin=fin, feats=np.array(feats, np.float32), M=M,
                         mass=np.array([f[0] for f in feats]))
    for m_ in range(M):
        rows.append((w, m_, feats[m_], 1.0 if m_ == best else 0.0))

X = np.array([r[2] for r in rows], np.float32); y = np.array([r[3] for r in rows])
wl = [r[0] for r in rows]
print(f"モード総数 {len(rows)} (坑井{len(well_modes)})  正例率 {y.mean():.2f}")

# ---- fold-OOF ロジスティック ----
P = np.zeros(len(rows))
for f in range(5):
    tr = [i for i, w in enumerate(wl) if fold_of[w] != f]
    va = [i for i, w in enumerate(wl) if fold_of[w] == f]
    if not va: continue
    mdl = LogisticRegression(max_iter=1000, C=1.0)
    mdl.fit(X[tr], y[tr])
    P[va] = mdl.predict_proba(X[va])[:, 1]
from sklearn.metrics import roc_auc_score
print(f"OOF AUC(正解モード判別) = {roc_auc_score(y, P):.3f}  / mass単独AUC = {roc_auc_score(y, X[:,0]):.3f}")

# ---- 再加重合成: w_m ∝ mass^(1−λ)·P^λ ----
pmap = {}
for i, (w, m_, _, _) in enumerate(rows):
    pmap.setdefault(w, {})[m_] = P[i]
def overall(lam):
    se = 0.0; n = 0
    for w, dm in well_modes.items():
        M = dm["M"]; fin = dm["fin"]
        pw = np.array([pmap[w][m_] for m_ in range(M)])
        wt = (dm["mass"] ** (1 - lam)) * np.maximum(pw, 1e-4) ** lam
        wt = wt / wt.sum()
        e = sum(wt[m_] * dm["ems"][m_] for m_ in range(M))
        se += np.nansum(e[fin] ** 2); n += fin.sum()
    return np.sqrt(se / n)
print("\n=== λ掃引 (0=現行mass加重, 1=純学習P加重) ===")
for lam in (0.0, 0.25, 0.5, 0.75, 1.0):
    print(f"  λ={lam:4.2f}: {overall(lam):.4f}")
