# -*- coding: utf-8 -*-
"""枝選択の検証: ①med30抜きでAUC保つか(循環リーク排除) ②方向付き枝(±15)→RMSE補正(実利)。
真TVTはラベル/評価のみ。出力: branch_v95b.txt"""
import os, sys, io
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
from pathlib import Path
import numpy as np, pandas as pd, pyarrow.parquet as pq
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

ART = Path(r"C:\Users\kosaka256\Documents\rogii_claude\v95\artifacts_v95b")
PQ = ART / "data" / "train.parquet"
REPS = ["ancc", "self", "nbr", "nbr_gr5", "self_graft"]
sch = set(pq.read_schema(PQ).names)
PRED30 = [f"pf_{r}_delta_{k}" for r in REPS for k in range(1, 7) if f"pf_{r}_delta_{k}" in sch]
PROV = [c for c in sch if c.startswith("prov_")]
DISAG = [c for c in sch if c.startswith("pf_agree_std_") or c.startswith("pf_agree_range_")]
tb = pq.read_table(PQ, columns=list(dict.fromkeys(PRED30 + PROV + DISAG + ["well", "target"])))
df = tb.to_pandas().iloc[::3].reset_index(drop=True)
y = df["target"].to_numpy(np.float64); g = df["well"].astype(str).to_numpy()
P = df[PRED30].to_numpy(np.float64)
med30 = np.nanmedian(P, 1); std30 = np.nanstd(P, 1)
iqr30 = np.nanpercentile(P, 75, 1) - np.nanpercentile(P, 25, 1)
rng30 = np.nanmax(P, 1) - np.nanmin(P, 1); mad30 = np.nanmedian(np.abs(P - med30[:, None]), 1)
d = P - med30[:, None]
frac_near = np.mean(np.abs(d) < 5, 1); frac_p15 = np.mean((d > 10) & (d < 20), 1); frac_m15 = np.mean((d < -10) & (d > -20), 1)
# ★med30を除いた分布特徴のみ(循環排除)
SHAPE = np.column_stack([std30, iqr30, rng30, mad30, frac_near, frac_p15, frac_m15])
shape_names = ["std30", "iqr30", "rng30", "mad30", "frac_near", "frac_p15", "frac_m15"]
PROVD = df[PROV + DISAG].to_numpy(np.float64)
Xnomed = np.column_stack([SHAPE, PROVD]); names = shape_names + PROV + DISAG
gkf = GroupKFold(5)

def auc_cv(X, yb):
    oof = np.full(len(yb), np.nan)
    for tr, va in gkf.split(X, yb, groups=g):
        m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=63, min_child_samples=200,
                               subsample=0.8, colsample_bytree=0.6, device="cpu", n_jobs=-1, verbose=-1)
        m.fit(X[tr], yb[tr]); oof[va] = m.predict_proba(X[va])[:, 1]
    return roc_auc_score(yb, oof), oof

y_br = (np.round((y - med30) / 15.0) != 0).astype(int)
auc_nomed, _ = auc_cv(Xnomed, y_br)

# ===== 方向付き枝(±15 up/down/0) -> RMSE補正 =====
brdir = np.clip(np.round((y - med30) / 15.0), -1, 1).astype(int)      # -1/0/+1
# 3クラス分類(方向) を回帰で近似(OOF予測を四捨五入)
oof_dir = np.full(len(y), np.nan)
for tr, va in gkf.split(Xnomed, brdir, groups=g):
    m = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=63, min_child_samples=200,
                          subsample=0.8, colsample_bytree=0.6, device="cpu", n_jobs=-1, verbose=-1)
    m.fit(Xnomed[tr], brdir[tr].astype(float)); oof_dir[va] = m.predict(Xnomed[va])
pred_dir = np.clip(np.round(oof_dir), -1, 1)
corrected = med30 + 15.0 * pred_dir
def R(p): mm = np.isfinite(p) & np.isfinite(y); return float(np.sqrt(np.mean((p[mm] - y[mm]) ** 2)))
# 方向当たり率(誤枝行での)
mask_wrong = (brdir != 0)
dir_acc = float(np.mean(pred_dir[mask_wrong] == brdir[mask_wrong])) if mask_wrong.sum() else 0.0

out = ["=== 枝選択 検証(med30抜きAUC / 方向付き補正) ==="]
out.append(f"  ★med30抜き 枝判別AUC = {auc_nomed:.4f}  (分布+provのみ。0.7超なら循環でなく実信号)")
out.append(f"  素med30 RMSE      = {R(med30):.4f}")
out.append(f"  ★枝補正後 RMSE    = {R(corrected):.4f}   (差 {R(corrected)-R(med30):+.4f})")
out.append(f"  誤枝行での方向当たり率 = {dir_acc*100:.1f}%  (誤枝{mask_wrong.mean()*100:.1f}%)")
out.append(f"  (参考)per-well oracle=3.56 / per-row=2.22")
txt = "\n".join(out); print("\n" + txt, flush=True)
(ART / "branch_v95b.txt").write_text(txt, encoding="utf-8")
