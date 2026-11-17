# -*- coding: utf-8 -*-
"""v74 (b) — Phase D(GRフリー)OOF誤差 × 既存チェーン(v69_4 combo)OOF誤差の相関測定。
   目的: GRフリー脚の「多様性価値」の定量(誤差相関が低いほどブレンド素材として有効)。
   整列: well + md_since(v69_4 parquet) ↔ CSV行位置(Phase D rows − (t0+1))。
   出力: 行レベル相関 / 井戸レベルRMSE相関 / 仮想ブレンド RMSE(a走査) / png"""
import io, sys, os, json
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

P0 = Path(r"C:\Users\kosaka256\Documents\rogii_claude")
ART69 = P0 / "v69" / "artifacts_v69_4"
ART = P0 / "v74" / "artifacts_v74"
TRAIN = P0 / "rogii-wellbore-geology-prediction" / "train"

# --- v69_4 combo OOF誤差
g = np.load(ART69 / "gbm_v41_oof.npy").astype(np.float64)
z = np.load(ART69 / "nn_oof_combo.npz")
nn = np.mean([z[k] for k in ("mdn2_s0", "mdn2_s1", "mdn2_s2")], axis=0)
combo = 0.54 * nn + 0.46 * g
tb = pq.read_table(ART69 / "data" / "train.parquet",
                   columns=["well", "target", "md_since"]).to_pandas()
tb["e69"] = combo - tb["target"].to_numpy(np.float64)
print(f"v69_4 rows={len(tb)}  overall(pre-pp)={np.sqrt(np.nanmean(tb.e69**2)):.4f}")

# --- Phase D OOF
dz = np.load(ART / "phaseD_oof.npz")
wells = sorted(set(k[:-6] for k in dz.files if k.endswith("__rows")))
print(f"phaseD wells={len(wells)}")

# --- t0(最終既知行)をCSVから
E69, ED = [], []
perw = []
for w in wells:
    sub = tb[tb.well == w]
    if not len(sub):
        continue
    hw = pd.read_csv(TRAIN / f"{w}__horizontal_well.csv", usecols=["TVT_input"])
    tin = hw["TVT_input"].to_numpy(float)
    kn = np.isfinite(tin)
    if not kn.any():
        continue
    t0 = int(np.where(kn)[0][-1])
    rows = dz[w + "__rows"]; errD = dz[w + "__err"]
    posD = rows - (t0 + 1)                        # 評価帯内位置(ft≈行)
    m69 = np.round(sub["md_since"].to_numpy(float)).astype(int)
    e69 = sub["e69"].to_numpy(float)
    lut = {p: i for i, p in enumerate(m69)}
    idx = np.array([lut.get(int(p), -1) for p in posD])
    ok = (idx >= 0) & np.isfinite(errD)
    if ok.sum() < 50:
        continue
    a = e69[idx[ok]]; b = errD[ok]
    fin = np.isfinite(a) & np.isfinite(b)
    E69.append(a[fin]); ED.append(b[fin])
    if fin.sum() > 100:
        perw.append((w, float(np.sqrt(np.mean(a[fin] ** 2))),
                     float(np.sqrt(np.mean(b[fin] ** 2)))))
e69 = np.concatenate(E69); eD = np.concatenate(ED)
n = len(e69)
r69 = float(np.sqrt(np.mean(e69 ** 2))); rD = float(np.sqrt(np.mean(eD ** 2)))
corr = float(np.corrcoef(e69, eD)[0, 1])
print(f"整列行={n}  v69_4={r69:.4f}  phaseD={rD:.4f}  行レベル誤差相関={corr:.3f}")
pw = np.array([[a, b] for _, a, b in perw])
print(f"井戸レベルRMSE相関={np.corrcoef(pw[:,0], pw[:,1])[0,1]:.3f}  (n={len(perw)})")
# 仮想ブレンド(誤差の線形結合=予測の線形結合と同値)
best = (0.0, r69)
curve = []
for a in np.round(np.arange(0, 0.51, 0.05), 2):
    rb = float(np.sqrt(np.mean(((1 - a) * e69 + a * eD) ** 2)))
    curve.append((a, rb))
    if rb < best[1]:
        best = (a, rb)
print("ブレンド走査 (a=GRフリー重み):")
for a, rb in curve:
    print(f"  a={a:.2f}: {rb:.4f}")
print(f"best a={best[0]} → {best[1]:.4f}  (Δ{best[1]-r69:+.4f})")
json.dump(dict(n=n, r69=r69, rD=rD, corr=corr, best_a=best[0], best=best[1],
               curve=curve), open(ART / "corr_chain.json", "w"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Meiryo"
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].scatter(pw[:, 0], pw[:, 1], s=8)
ax[0].plot([0, 30], [0, 30], "r--", lw=0.6)
ax[0].set_xlabel("v69_4 combo 井戸RMSE"); ax[0].set_ylabel("Phase D(GRフリー) 井戸RMSE")
ax[0].set_xlim(0, 30); ax[0].set_ylim(0, 40)
ax[0].set_title(f"井戸レベル (corr={np.corrcoef(pw[:,0],pw[:,1])[0,1]:.2f})")
ax[0].grid(alpha=0.3)
aa = [c[0] for c in curve]; rr = [c[1] for c in curve]
ax[1].plot(aa, rr, marker="o", ms=3)
ax[1].axhline(r69, color="gray", ls="--", lw=0.7, label=f"v69_4 {r69:.3f}")
ax[1].set_xlabel("a(GRフリー脚の重み)"); ax[1].set_ylabel("overall RMSE")
ax[1].set_title(f"仮想ブレンド (行相関={corr:.2f})")
ax[1].legend(); ax[1].grid(alpha=0.3)
fig.suptitle("v74(b) — GRフリー脚 × 既存チェーンの誤差相関とブレンド価値", fontsize=12)
fig.tight_layout()
fig.savefig(ART / "corr_chain.png", dpi=130)
print("saved:", ART / "corr_chain.png")
