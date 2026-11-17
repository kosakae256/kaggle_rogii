# -*- coding: utf-8 -*-
"""GBDT枝選択器 achievability probe(坑井単位)。
各(坑井,候補)について「その候補の坑井内RMSE」を prov/一致度/候補ID から GBDTで予測 →
OOF(GroupKFold by well)で坑井ごと最良候補を選択 → プールRMSE。
現L5(5.80)/坑井oracle(2.51)/単体ベスト(6.96)と比較 = 選択の伸びしろが学習で取れるか。
CPU(LightGBM)。省メモリ(候補列を1本ずつstream→坑井集計、行配列は保持しない)。
実行: .venv/Scripts/python.exe v100/gbdt_selector_probe.py
"""
import os, sys, io, re
try:
    sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
except Exception:
    pass
import numpy as np, pyarrow.parquet as pq
import lightgbm as lgb
from sklearn.model_selection import GroupKFold

P = os.environ.get("PARQUET", r"C:\Users\kosaka256\Documents\rogii_claude\v99\artifacts_v99\data\train_combo.parquet")
REPS = ("ancc", "self", "nbr", "nbr_gr5", "self_graft")
REP2PROV = {"ancc": "tw", "self": "self", "nbr": "nbr", "nbr_gr5": "nbr_gr5", "self_graft": "self_graft"}


def main():
    sch = pq.read_schema(P); names = set(sch.names)
    cands = []
    for pre in ("", "v97_"):
        for rep in REPS:
            k = 1
            while f"{pre}pf_{rep}_delta_{k}" in names:
                cands.append((f"{pre}pf_{rep}_delta_{k}", rep, 1 if pre else 0, k)); k += 1
    C = len(cands)
    print(f"候補数={C}")
    y = pq.read_table(P, columns=["target"]).column("target").to_numpy(zero_copy_only=False).astype(np.float64)
    well = pq.read_table(P, columns=["well"]).column("well").to_pandas().to_numpy()
    uw, inv = np.unique(well, return_inverse=True); W = len(uw)
    cntw = np.bincount(inv, minlength=W).astype(np.float64)
    finy = np.isfinite(y); N = len(y)
    print(f"坑井={W} 行={N}")

    # prov(坑井定数): 各repのgrfit/grcorr/grfit_vs_tw + global
    provcols = [c for c in names if c.startswith("prov_")]
    provtab = pq.read_table(P, columns=provcols + ["well"])
    import pandas as pd
    pdf = provtab.to_pandas().groupby("well").first()   # 坑井→prov(定数)
    pdf = pdf.loc[uw]                                    # uw順

    # pass1: 候補ごと 坑井SSE / 坑井mean / 坑井std
    sse = np.zeros((W, C)); smean = np.zeros((W, C)); sstd = np.zeros((W, C))
    for ci, (col, rep, leg, k) in enumerate(cands):
        v = pq.read_table(P, columns=[col]).column(col).to_numpy(zero_copy_only=False).astype(np.float64)
        e = v - y; se = np.where(np.isfinite(e), e * e, 0.0)
        sse[:, ci] = np.bincount(inv, weights=se, minlength=W)
        vv = np.where(np.isfinite(v), v, 0.0)
        sm = np.bincount(inv, weights=vv, minlength=W) / np.maximum(cntw, 1)
        s2 = np.bincount(inv, weights=vv * vv, minlength=W) / np.maximum(cntw, 1)
        smean[:, ci] = sm; sstd[:, ci] = np.sqrt(np.maximum(s2 - sm * sm, 0))
    rmse_wc = np.sqrt(sse / np.maximum(cntw[:, None], 1))    # (W,C) 坑井×候補 RMSE = ラベル
    cross_mean = smean.mean(1)                                # 坑井の候補横断平均
    cross_med = np.median(smean, 1)

    # 特徴テーブル (W*C, F)
    rows = []
    for ci, (col, rep, leg, k) in enumerate(cands):
        pv = REP2PROV[rep]
        feat = {
            "rep": np.full(W, REPS.index(rep)), "leg": np.full(W, leg), "bank": np.full(W, k),
            "mean_delta": smean[:, ci], "std_delta": sstd[:, ci],
            "agree_mean": np.abs(smean[:, ci] - cross_mean),
            "agree_med": smean[:, ci] - cross_med,
            "grfit": pdf.get(f"prov_{pv}_grfit", pd.Series(np.nan, index=pdf.index)).to_numpy(),
            "grcorr": pdf.get(f"prov_{pv}_grcorr", pd.Series(np.nan, index=pdf.index)).to_numpy(),
            "grfit_vs_tw": pdf.get(f"prov_{pv}_grfit_vs_tw", pd.Series(0.0, index=pdf.index)).to_numpy(),
            "tw_grfit": pdf["prov_tw_grfit"].to_numpy(), "tw_grcorr": pdf["prov_tw_grcorr"].to_numpy(),
            "nbr_dist_min": pdf["prov_nbr_dist_min"].to_numpy(),
            "nbr_gr5_sim": pdf["prov_nbr_gr5_sim_top1"].to_numpy(),
            "self_cover": pdf["prov_self_cover"].to_numpy(),
            "nrows": np.log1p(cntw),
            "label": rmse_wc[:, ci],
        }
        rows.append(feat)
    FEATS = [k for k in rows[0].keys() if k != "label"]
    X = np.concatenate([np.stack([r[f] for f in FEATS], 1) for r in rows], 0)   # (W*C, F)
    Y = np.concatenate([r["label"] for r in rows], 0)
    Gw = np.tile(np.arange(W), C)                              # 各行の坑井
    cand_of = np.repeat(np.arange(C), W)                       # 各行の候補
    print(f"学習行(坑井×候補)={len(Y)} 特徴={len(FEATS)}")

    # GroupKFold by well: 坑井ごと候補RMSEを予測 → 坑井ごと min予測候補をOOF選択
    gkf = GroupKFold(5)
    pred = np.full(len(Y), np.nan)
    for tr, va in gkf.split(X, Y, groups=Gw):
        m = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=31,
                              subsample=0.8, colsample_bytree=0.8, min_child_samples=40, verbosity=-1)
        m.fit(X[tr], Y[tr])
        pred[va] = m.predict(X[va])
    # 坑井ごと OOF選択(min予測RMSEの候補)
    pred_wc = np.full((W, C), np.inf)
    pred_wc[Gw, cand_of] = pred
    sel = np.argmin(pred_wc, 1)                                # 各坑井の選択候補
    sel_sse = sse[np.arange(W), sel].sum()
    sel_rmse = float(np.sqrt(sel_sse / cntw.sum()))
    # 参考: soft blend(予測RMSEでsoftmax重み)
    T = np.nanstd(pred_wc[np.isfinite(pred_wc)])
    w = np.exp(-(pred_wc - pred_wc.min(1, keepdims=True)) / max(T * 0.5, 1e-6))
    w /= w.sum(1, keepdims=True)
    soft_sse = (w * sse).sum(1).sum()     # 近似(候補独立SSE加重≠真のblend分散だが目安)
    print("\n=== 結果 ===")
    print(f"  単体ベスト候補         : 6.9569")
    print(f"  現 L5                  : 5.7961")
    print(f"  ★ GBDT坑井選択(OOF hard): {sel_rmse:.4f}")
    print(f"  坑井oracle(上限)       : 2.5136")
    print(f"  → L5比 {sel_rmse-5.7961:+.4f} / oracleまで {sel_rmse-2.5136:+.4f}")
    # 特徴重要度
    m = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=31, verbosity=-1).fit(X, Y)
    imp = sorted(zip(FEATS, m.feature_importances_), key=lambda t: -t[1])
    print("\n特徴重要度 top10:", [(f, int(v)) for f, v in imp[:10]])


if __name__ == "__main__":
    main()
