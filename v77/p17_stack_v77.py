# -*- coding: utf-8 -*-
"""
v77 P17 — 行レベル学習コンバイナ(非逐次・アンサンブル学習)
==========================================================
入力(行特徴): 各脚のオフセット(PF/P5/P6−錨), 脚間不一致, σ(dd), dd, postd等 → 真オフセット回帰。
GroupKFold5(井戸, ソート順%5) OOF。固定重みblend(7.134@773)を学習で超えられるかの判定。
使い方: python v77/p17_stack_v77.py [p5pkl p6pkl]
"""
import io, sys, pickle, time
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
from pathlib import Path
import numpy as np

P0 = Path(r"C:\Users\kosaka256\Documents\rogii_claude")
ART = P0 / "v77" / "artifacts_v77"
LS = 10.536319667317654
P5N = sys.argv[1] if len(sys.argv) > 1 else "p5_out_dip03_773w"
P6N = sys.argv[2] if len(sys.argv) > 2 else "p6_out_lp13_773w"
anc = pickle.loads((ART / "k16_anchor_train.pkl").read_bytes())
bare = pickle.loads((P0 / "v73" / "artifacts_v73" / "seed_dump_bare0.pkl").read_bytes())
p5 = pickle.loads((ART / f"{P5N}.pkl").read_bytes())
p6 = pickle.loads((ART / f"{P6N}.pkl").read_bytes())

wids = sorted(set(bare) & set(anc) & set(p5) & set(p6))
print(f"{len(wids)}井")
F, Y, G, EFIX = [], [], [], []
t0 = time.time()
for gi, wid in enumerate(wids):
    d = bare[wid]; ll = d["ll"].astype(np.float64)
    ww = np.exp((ll - ll.max()) / LS); ww /= ww.sum()
    epf_pred = (ww[:, None] * d["err"].astype(np.float64)).sum(0) + d["true"]   # PF予測TVT
    sspread = d["err"].astype(np.float64).std(0)                                # 32seedばらつき
    tv = d["true"]; A = anc[wid]["tvt"]; sdd = anc[wid]["sig"]; dd = anc[wid]["dd"]
    o5 = np.asarray(p5[wid]["pred"], np.float64); o6 = np.asarray(p6[wid]["pred"], np.float64)
    postd = np.asarray(p6[wid].get("postd", np.zeros(len(o6))), np.float64)
    T = min(len(tv), len(A), len(o5), len(o6), len(epf_pred))
    fpf = epf_pred[:T] - A[:T]; f5 = o5[:T] - A[:T]; f6 = o6[:T] - A[:T]
    y = tv[:T] - A[:T]
    pos = np.arange(T) / max(T - 1, 1)
    X = np.column_stack([fpf, f5, f6, np.abs(fpf - f5), np.abs(fpf - f6), np.abs(f5 - f6),
                         np.abs(fpf), sdd[:T], dd[:T] / 1000.0, sspread[:T], postd[:T],
                         pos, np.full(T, T / 1000.0)])
    m = np.isfinite(X).all(1) & np.isfinite(y)
    F.append(X[m].astype(np.float32)); Y.append(y[m].astype(np.float32))
    G.append(np.full(m.sum(), gi, np.int32))
    EFIX.append((0.5 * fpf + 0.2 * (0.0) + 0.2 * f5 + 0.1 * f6)[m] - y[m])  # 参考: 固定w(錨=0成分含む)
X = np.concatenate(F); y = np.concatenate(Y); g = np.concatenate(G)
efix = np.concatenate(EFIX)
print(f"rows {len(X)}  固定w再現: {np.sqrt(np.mean(efix**2)):.4f}  ({time.time()-t0:.0f}s)")

import os
import lightgbm as lgb
RESID = os.environ.get("RESID", "0")
blendfix = 0.5 * X[:, 0] + 0.2 * X[:, 1] + 0.1 * X[:, 2]        # w: PF0.5/P5 0.2/P6 0.1/錨0.2(=off0)
tgt = (y - blendfix) if RESID == "1" else y
oof = np.full(len(y), np.nan, np.float32)
for f in range(5):
    va = (g % 5) == f; tr = ~va
    ds = lgb.Dataset(X[tr], tgt[tr])
    mdl = lgb.train(dict(objective="regression", metric="rmse", learning_rate=0.05,
                         num_leaves=63, min_data_in_leaf=200, feature_fraction=0.9,
                         bagging_fraction=0.8, bagging_freq=1, verbosity=-1),
                    ds, num_boost_round=400)
    oof[va] = mdl.predict(X[va]).astype(np.float32)
    print(f"fold{f} done ({time.time()-t0:.0f}s)", flush=True)
pred = (blendfix + oof) if RESID == "1" else oof
e = pred - y
print(f"\n=== P17 学習コンバイナ (LightGBM, {len(wids)}井 OOF, RESID={RESID}) legs={P5N},{P6N} ===")
print(f"  P17 OOF pooled = {np.sqrt(np.nanmean(e**2)):.4f}  (固定blend基準 {np.sqrt(np.mean((blendfix-y)**2)):.4f})")
for sc in (0.25, 0.5, 0.75):
    if RESID == "1":
        print(f"  縮小適用 {sc}: {np.sqrt(np.nanmean((blendfix + sc*oof - y)**2)):.4f}")
imp = mdl.feature_importance("gain")
names = ["fpf", "f5", "f6", "|pf-5|", "|pf-6|", "|5-6|", "|pf|", "sdd", "dd", "sspread", "postd", "pos", "len"]
top = sorted(zip(imp, names), reverse=True)
print("  imp:", [(n, int(v)) for v, n in top])
