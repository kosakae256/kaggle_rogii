# -*- coding: utf-8 -*-
"""per-well PF選択モデル(CPU): oracle-per-well を取りに行く。
5表現(バンク平均)を well単位で選択(soft/hard)。特徴=provenance+不一致(well集約)。
過去: v33 gate6.749 / v34 OOF6.529(LB5.930,転移11%)。新情報で超えるか。
真TVTはラベル/評価のみ。出力: wellsel_v95.txt"""
import sys, io
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
from pathlib import Path
import numpy as np, pandas as pd, pyarrow.parquet as pq
from sklearn.model_selection import GroupKFold
import lightgbm as lgb

B = Path(r"C:\Users\kosaka256\Documents\rogii_claude\v95\artifacts_v95b")
PQ = B / "data" / "train.parquet"
REPS = ["ancc", "self", "nbr", "nbr_gr5", "self_graft"]
sch = set(pq.read_schema(PQ).names)
PROV = [c for c in sch if c.startswith("prov_")]
DIS = [c for c in sch if c.startswith("pf_agree")]
PREDc = {r: [f"pf_{r}_delta_{k}" for k in range(1, 7) if f"pf_{r}_delta_{k}" in sch] for r in REPS}
allpred = [c for cs in PREDc.values() for c in cs]
df = pq.read_table(PQ, columns=list(dict.fromkeys(allpred + PROV + DIS + ["well", "target"]))).to_pandas()
y = df["target"].to_numpy(np.float64); well = df["well"].astype(str).to_numpy()
# 各表現のバンク平均予測(per-row)
REPPRED = {r: np.nanmean(df[PREDc[r]].to_numpy(np.float64), 1) for r in REPS}

# ==== well集約 ====
uw, inv = np.unique(well, return_inverse=True)
NW = len(uw)
def wagg(a):   # well平均
    s = np.bincount(inv, np.nan_to_num(a), minlength=NW); c = np.bincount(inv, np.isfinite(a).astype(float), minlength=NW)
    return s / np.maximum(c, 1)
# well特徴 = provenance平均 + 不一致平均 + 各表現予測のwell std(自信のなさ)
featcols = PROV + DIS
Wfeat = np.column_stack([wagg(df[c].to_numpy(np.float64)) for c in featcols])
# 各表現の per-well RMSE(ラベル=どれが良いか)
def well_rmse(pred):
    e2 = (pred - y) ** 2; s = np.bincount(inv, np.nan_to_num(e2), minlength=NW); c = np.bincount(inv, np.isfinite(e2).astype(float), minlength=NW)
    return np.sqrt(s / np.maximum(c, 1))
RMSE = {r: well_rmse(REPPRED[r]) for r in REPS}       # (NW,) 各表現のwell RMSE
RMAT = np.stack([RMSE[r] for r in REPS], 1)            # (NW,5)
best_rep = np.argmin(RMAT, 1)

# ==== 選択器: 各表現のwell RMSE を予測 → soft/hard 選択 ====
gkf = GroupKFold(5); wg = uw
predR = np.full((NW, len(REPS)), np.nan)
for j, r in enumerate(REPS):
    oof = np.full(NW, np.nan)
    for tr, va in gkf.split(Wfeat, RMSE[r], groups=wg):
        m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=30,
                              subsample=0.8, colsample_bytree=0.7, device="cpu", n_jobs=-1, verbose=-1)
        m.fit(Wfeat[tr], RMSE[r][tr]); oof[va] = m.predict(Wfeat[va])
    predR[:, j] = oof
# hard: 予測RMSE最小の表現
sel_hard = np.argmin(predR, 1)
# 各行に、その井の選択表現の予測を割当てて pooled RMSE
def pooled_from_wellsel(sel):
    selrep = np.array(REPS)[sel][inv]            # 各行の選択表現名
    p = np.empty(len(y))
    for r in REPS:
        mask = (selrep == r); p[mask] = REPPRED[r][mask]
    m = np.isfinite(p) & np.isfinite(y); return float(np.sqrt(np.mean((p[m] - y[m]) ** 2)))
# soft: weight ∝ exp(-β predRMSE) を well毎に、rep予測をblend
def pooled_soft(beta):
    w = np.exp(-beta * (predR - np.nanmin(predR, 1, keepdims=True)))
    w = w / np.nansum(w, 1, keepdims=True)       # (NW,5)
    wr = w[inv]                                    # (N,5)
    p = np.nansum(np.stack([REPPRED[r] for r in REPS], 1) * wr, 1)
    m = np.isfinite(p) & np.isfinite(y); return float(np.sqrt(np.mean((p[m] - y[m]) ** 2)))

def R(p): m = np.isfinite(p) & np.isfinite(y); return float(np.sqrt(np.mean((p[m] - y[m]) ** 2)))
single = {r: R(REPPRED[r]) for r in REPS}
# oracle per-well(真の最良表現)
sel_orac = best_rep
orac_well = pooled_from_wellsel(sel_orac)
# equal blend
eq = R(np.nanmean(np.stack([REPPRED[r] for r in REPS], 1), 1))
# hard選択の的中率
hit = float(np.mean(sel_hard == best_rep))

out = ["=== per-well PF選択モデル(5表現, v95b nn) ==="]
out.append(f"  単体: " + " ".join(f"{r}={single[r]:.3f}" for r in REPS))
out.append(f"  等重blend = {eq:.4f}")
out.append(f"  ★per-well oracle(真の最良表現) = {orac_well:.4f}")
out.append(f"  --- 学習選択器 ---")
out.append(f"  hard選択(予測RMSE最小)   = {pooled_from_wellsel(sel_hard):.4f}  (最良表現的中率{hit*100:.0f}%, chance20%)")
for b in [0.3, 0.7, 1.5, 3.0]:
    out.append(f"  soft選択 β={b}          = {pooled_soft(b):.4f}")
out.append(f"  → oracle({orac_well:.2f})にどこまで近づくか。過去 v33 gate6.749/v34 6.529(転移11%)。")
txt = "\n".join(out); print("\n" + txt, flush=True)
(B / "wellsel_v95.txt").write_text(txt, encoding="utf-8")
