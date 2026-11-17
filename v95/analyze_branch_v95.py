# -*- coding: utf-8 -*-
"""30 PF出力(6バンク×5表現)で【選択問題】と【枝選択問題】を検証(CPU, GPU非使用)。
A: 30出力の per-well/per-row oracle(選択天井)。
B: 枝選択= 真値が「30出力のconsensus(median)」か「±15ftずれた枝」かを二値分類。
   材料=30出力の分布(spread/多峰性)+provenance+不一致。焦点=v47の枝判別壁AUC0.56を超えるか。
真TVTは"評価/ラベル"にのみ使用(モデル入力は予測+provenanceのみ=valid)。
出力: branch_v95.txt  実行: ./.venv/Scripts/python.exe v95/analyze_branch_v95.py"""
import os, sys, io
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
from pathlib import Path
import numpy as np, pandas as pd, pyarrow.parquet as pq
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

ART = Path(r"C:\Users\kosaka256\Documents\rogii_claude\v95\artifacts_v95b")
PQ = ART / "data" / "train.parquet"
REPS = ["ancc", "self", "nbr", "nbr_gr5", "self_graft"]     # ancc=tw
sch = set(pq.read_schema(PQ).names)
PRED30 = [f"pf_{r}_delta_{k}" for r in REPS for k in range(1, 7) if f"pf_{r}_delta_{k}" in sch]
PROV = [c for c in sch if c.startswith("prov_")]
DISAG = [c for c in sch if (c.startswith("pf_agree_std_") or c.startswith("pf_agree_range_"))]
need = list(dict.fromkeys(PRED30 + PROV + DISAG + ["well", "target"]))
print(f"[branch] PF出力={len(PRED30)}個 prov={len(PROV)} disag={len(DISAG)}", flush=True)

tb = pq.read_table(PQ, columns=need); df = tb.to_pandas()
STEP = 3                                                     # 行間引き(メモリ/速度)
df = df.iloc[::STEP].reset_index(drop=True)
y = df["target"].to_numpy(np.float64); g = df["well"].astype(str).to_numpy()
P = df[PRED30].to_numpy(np.float64)                          # (N,30) 予測残差
print(f"[branch] rows={len(df)}(間引き{STEP}) wells={len(set(g))}", flush=True)

# ===== 実験A: 選択天井(oracle over 30) =====
err = np.abs(P - y[:, None])
row_orac = float(np.sqrt(np.nanmean(np.nanmin(err, 1) ** 2)))
med30 = np.nanmedian(P, 1)
single_med = float(np.sqrt(np.nanmean((med30 - y) ** 2)))     # 30の中央値(素の集約)
# per-well oracle(各井で30個から最良1個)
tmp = pd.DataFrame(P, columns=[f"p{i}" for i in range(P.shape[1])]); tmp["g"] = g; tmp["y"] = y
owell = np.empty(len(y))
for gg, sub in tmp.groupby("g"):
    yy = sub["y"].to_numpy(); rk = [np.sqrt(np.mean((sub[f"p{i}"].to_numpy() - yy) ** 2)) for i in range(P.shape[1])]
    bi = int(np.argmin(rk)); owell[sub.index.to_numpy()] = sub[f"p{bi}"].to_numpy()
well_orac = float(np.sqrt(np.mean((owell - y) ** 2)))

# ===== 実験B: 枝選択(consensus median から真値が±15ftずれるか) =====
branch = np.round((y - med30) / 15.0)                        # 0=consensus正, ±1=±15ずれ
y_br = (branch != 0).astype(int)                             # 二値: consensusが誤枝か
# 枝判別の材料(真TVT不使用): 30出力の分布 + provenance + 不一致
std30 = np.nanstd(P, 1); iqr30 = np.nanpercentile(P, 75, 1) - np.nanpercentile(P, 25, 1)
rng30 = np.nanmax(P, 1) - np.nanmin(P, 1); mad30 = np.nanmedian(np.abs(P - med30[:, None]), 1)
# 多峰性: median から ±(10-20)ft に第2クラスタがあるか(枝の兆候)
d = P - med30[:, None]
frac_near = np.mean(np.abs(d) < 5, 1); frac_p15 = np.mean((d > 10) & (d < 20), 1); frac_m15 = np.mean((d < -10) & (d > -20), 1)
BF = np.column_stack([std30, iqr30, rng30, mad30, frac_near, frac_p15, frac_m15, med30])
BFcols = ["std30", "iqr30", "rng30", "mad30", "frac_near", "frac_p15", "frac_m15", "med30"]
Xb = np.column_stack([BF, df[PROV + DISAG].to_numpy(np.float64)])
feat_names = BFcols + PROV + DISAG

gkf = GroupKFold(5); oof_p = np.full(len(y), np.nan)
for k, (tr, va) in enumerate(gkf.split(Xb, y_br, groups=g)):
    m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=63,
                           subsample=0.8, colsample_bytree=0.6, min_child_samples=200,
                           device="cpu", n_jobs=-1, verbose=-1)
    m.fit(Xb[tr], y_br[tr]); oof_p[va] = m.predict_proba(Xb[va])[:, 1]
auc = roc_auc_score(y_br, oof_p)
# 特徴importance top
mm = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=63, min_child_samples=200, verbose=-1)
mm.fit(Xb, y_br); imp = sorted(zip(feat_names, mm.feature_importances_), key=lambda t: -t[1])[:8]

out = ["=== 30 PF出力の 選択/枝選択 実験(v95b nn_emission, 間引き{}) ===".format(STEP)]
out.append("--- 実験A: 選択天井 ---")
out.append(f"  30の中央値(素集約) = {single_med:.4f}")
out.append(f"  ★per-well oracle(30から井ごと最良) = {well_orac:.4f}")
out.append(f"  ★per-row oracle(30から行ごと最良)  = {row_orac:.4f}")
out.append("--- 実験B: 枝選択(consensusが±15ft誤枝か 二値分類) ---")
out.append(f"  誤枝の割合(chance) = {y_br.mean()*100:.1f}%")
out.append(f"  ★枝判別 AUC = {auc:.4f}   (v47壁≈0.56。超えれば増えた情報で枝が読める)")
out.append(f"  重要特徴 top8 = {[(n, int(v)) for n, v in imp]}")
txt = "\n".join(out); print("\n" + txt, flush=True)
(ART / "branch_v95.txt").write_text(txt, encoding="utf-8")
