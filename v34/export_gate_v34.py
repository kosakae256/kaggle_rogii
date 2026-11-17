# -*- coding: utf-8 -*-
"""v34 モードゲートの提出用エクスポート: mode_table_v34.csv の全行で Logistic を学習し、
係数・スケーラ統計・ルール定数を gate_pack_v34.json に保存する(提出側は pickle 不要)。
実行: .venv/Scripts/python.exe v34/export_gate_v34.py(数秒)
→ artifacts_v34/gate_pack_v34.json を Kaggle dataset(rogii-v34-artifacts)にアップロードする。
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

PROJECT = Path(r"C:\Users\kosaka256\Documents\rogii_claude")
ART = PROJECT / "v34" / "artifacts_v34"

# 診断ノートの FEATS と完全同一(順序込み)。提出側はこの順でベクトルを組む。
FEATS = ["vote", "ext_vote", "stay_share", "n_src", "has_nbr", "has_self", "has_tw", "n_pset_ref",
         "tw_seed_mass", "ext_seed_mass", "sm_seed_mass", "fw_seed_mass",
         "seed_frac", "n_src_seed", "n_pset_seed", "n_flt",
         "c_abs", "spread", "has_opp", "gnn_align",
         "xm_self", "xm_nbr", "xm_tw",
         "rough_ratio", "dip_dev", "dip_absmax", "geo_jump", "smfw_gap", "edge_frac", "tail_dev", "ll_margin",
         "nrefs", "nndist", "n_rows", "md_len", "md_step", "d_gnn"]

TBL = pd.read_csv(ART / "mode_table_v34.csv")
X = np.nan_to_num(TBL[FEATS].to_numpy(dtype=float), nan=-999.0)   # ★NaN→-999 の後にスケーラ(診断と同一手順)
y = TBL["y"].to_numpy()
sc = StandardScaler().fit(X)
lo = LogisticRegression(C=0.3, max_iter=2000).fit(sc.transform(X), y)
p_full = lo.predict_proba(sc.transform(X))[:, 1]
print(f"full-train: rows={len(TBL)} y=1:{int(y.sum())}  AUC(in-sample)={roc_auc_score(y, p_full):.3f}"
      f"(OOF実測は 0.924 — in-sample はそれよりやや高くて正常)")

pack = dict(
    feats=FEATS,
    scaler_mean=sc.mean_.tolist(), scaler_scale=sc.scale_.tolist(),
    coef=lo.coef_[0].tolist(), intercept=float(lo.intercept_[0]),
    rule=dict(agg="post", lam=1.0),                     # OOF best: p_log post λ1.0 = 6.5288
    const=dict(gap_link=8.0, mode_min=4.0, mode_max=60.0, nan_fill=-999.0),
    provenance=dict(oof_overall=6.5288, oof_auc_log=0.924, base_blend=6.8711,
                    table_rows=int(len(TBL)), created="2026-07-07"))
(ART / "gate_pack_v34.json").write_text(json.dumps(pack, indent=1))
print("saved:", ART / "gate_pack_v34.json")
