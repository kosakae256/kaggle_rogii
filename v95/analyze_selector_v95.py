# -*- coding: utf-8 -*-
"""選択器テスト(GPU軽量, CPU LightGBM): 増えた情報(provenance/loglik)で「表現選択」が改善するか。
基本特徴=5表現の予測delta+std+不一致。それに provenance/loglik を足す/足さないで OOF比較。
oracle(per-well/per-row=選択天井)とも比較。v95b(nn_emission)parquet使用。
出力: selector_v95.txt  実行: ./.venv/Scripts/python.exe v95/analyze_selector_v95.py"""
import os, sys, io, json
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
from pathlib import Path
import numpy as np, pyarrow.parquet as pq
from sklearn.model_selection import GroupKFold
import lightgbm as lgb

ART = Path(r"C:\Users\kosaka256\Documents\rogii_claude\v95\artifacts_v95b")
PQ = ART / "data" / "train.parquet"
REPS = ["ancc", "self", "nbr", "nbr_gr5", "self_graft"]   # ancc=tw
sch = set(pq.read_schema(PQ).names)

def cols_for(pats):
    return [c for c in pats if c in sch]

# 予測delta(5表現×6バンク) / std / 不一致 / loglik / provenance
PRED = cols_for([f"pf_{r}_delta_{k}" for r in REPS for k in range(1, 7)]
                + [f"pf_ancc_delta_{k}" for k in range(1, 7)])   # ancc は pf_ancc_delta
PRED = cols_for([f"pf_ancc_delta_{k}" for k in range(1, 7)]
                + [f"pf_{r}_delta_{k}" for r in ["self", "nbr", "nbr_gr5", "self_graft"] for k in range(1, 7)])
STD = cols_for([f"pf_ancc_std_{k}" for k in range(1, 7)]
               + [f"pf_{r}_std_{k}" for r in ["self", "nbr", "nbr_gr5", "self_graft"] for k in range(1, 7)])
DISAG = cols_for([f"pf_{r}_vs_tw_{k}" for r in ["self", "nbr", "nbr_gr5", "self_graft"] for k in range(1, 7)]
                 + [f"pf_agree_std_{k}" for k in range(1, 7)] + [f"pf_agree_range_{k}" for k in range(1, 7)])
LOGLIK = cols_for([f"pf_{r}_loglik_{k}" for r in ["self", "nbr", "nbr_gr5", "self_graft"] for k in range(1, 7)])
PROV = cols_for([c for c in sch if c.startswith("prov_")])
BASE = PRED + STD + DISAG                       # 予測+不確実性+不一致(旧来型の選択材料)
NEWINFO = LOGLIK + PROV                          # ★増えた情報

print(f"[selector] BASE={len(BASE)}列 (pred{len(PRED)}/std{len(STD)}/disag{len(DISAG)}) NEWINFO={len(NEWINFO)}列 (loglik{len(LOGLIK)}/prov{len(PROV)})", flush=True)

need = list(dict.fromkeys(BASE + NEWINFO + ["well", "target", "last_known_tvt"]))
tb = pq.read_table(PQ, columns=need)
import pandas as pd
df = tb.to_pandas(); y = df["target"].to_numpy(np.float64); g = df["well"].astype(str).to_numpy()
print(f"[selector] rows={len(df)} wells={len(set(g))}", flush=True)

def run_lgb(feats, tag):
    X = df[feats].to_numpy(np.float32)
    gkf = GroupKFold(5); oof = np.full(len(y), np.nan)
    for k, (tr, va) in enumerate(gkf.split(X, y, groups=g)):
        m = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=63,
                              subsample=0.8, colsample_bytree=0.6, min_child_samples=200,
                              device="cpu", n_jobs=-1, verbose=-1)
        m.fit(X[tr], y[tr]); oof[va] = m.predict(X[va])
    r = float(np.sqrt(np.mean((oof - y) ** 2)))
    print(f"  {tag:28s} OOF RMSE = {r:.4f}", flush=True); return r

# oracle(5表現のバンク平均予測で per-well/per-row 選択天井)
def rep_pred(r):
    cs = cols_for([f"pf_{'ancc' if r=='ancc' else r}_delta_{k}" for k in range(1, 7)])
    return np.nanmean(df[cs].to_numpy(np.float64), 1)
REPPRED = {r: rep_pred(r) for r in REPS}
M = np.stack([REPPRED[r] for r in REPS], 1)          # (N,5) 各表現のバンク平均予測
err = np.abs(M - y[:, None])
row_orac = float(np.sqrt(np.mean(np.min(err, 1) ** 2)))
# per-well oracle(各井で最良表現を選ぶ)
tmp = pd.DataFrame(M, columns=REPS); tmp["g"] = g; tmp["y"] = y
owell = np.empty(len(y))
for gg, sub in tmp.groupby("g"):
    rk = {r: np.sqrt(np.mean((sub[r] - sub["y"]) ** 2)) for r in REPS}
    br = min(rk, key=rk.get); owell[sub.index.to_numpy()] = sub[br].to_numpy()
well_orac = float(np.sqrt(np.mean((owell - y) ** 2)))
single = {r: float(np.sqrt(np.mean((REPPRED[r] - y) ** 2))) for r in REPS}

out = ["=== 選択器テスト(CPU LightGBM, v95b nn_emission) ==="]
out.append(f"  単体(バンク平均): " + " ".join(f"{r}={single[r]:.3f}" for r in REPS))
out.append(f"  per-well oracle={well_orac:.4f}  per-row oracle={row_orac:.4f}  (選択天井)")
r_base = run_lgb(BASE, "BASE(pred+std+不一致)")
r_new = run_lgb(BASE + NEWINFO, "BASE+新情報(loglik+prov)")
out.append(f"  BASE(pred+std+不一致)      = {r_base:.4f}")
out.append(f"  ★BASE+新情報(loglik+prov)  = {r_new:.4f}   (差 {r_new-r_base:+.4f})")
out.append(f"  → 新情報で下がれば「増えた情報が選択に効く」。oracle({well_orac:.2f})にどこまで近づくか。")
txt = "\n".join(out); print("\n" + txt, flush=True)
(ART / "selector_v95.txt").write_text(txt, encoding="utf-8")
