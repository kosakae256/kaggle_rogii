# -*- coding: utf-8 -*-
"""7番bank(newpf, smooth)の派生フルセット(tabular用)を生成。newpf_bank_train.npz の
   tw/self/nbr delta+std(train.parquet行整列済)から mean を復元し、features_v93 の派生器で
   pf_7_* / pf_ancc_7 / pf_self_7 / pf_nbr_7 の派生特徴を作る。GPU不使用(CPU)。
   出力: newpf_full_train.npz(全派生列)。既存ファイル上書きなし。"""
import os, sys, io
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
from pathlib import Path
import numpy as np, pandas as pd, pyarrow.parquet as pq
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pf_banks_v93 as pf
import features_v93 as featv

ART = pf.ART; DATA = pf.DATA_DIR / "train"
OFFS = featv.PF_OFFS
P7 = pf.bank_param("pf_1")   # hgr_smooth_r 等は pf_1 相当(gr整形の一貫性用)
d = np.load(ART / "newpf_bank_train.npz", allow_pickle=True)
ANC_D, ANC_S = d["pf_ancc_delta_7"], d["pf_ancc_std_7"]
SLF_D, SLF_S = d["pf_self_delta_7"], d["pf_self_std_7"]
NBR_D, NBR_S = d["pf_nbr_delta_7"], d["pf_nbr_std_7"]

PF_OFFS = featv.PF_OFFS
# 既存bank列を読み込み(tdpf/vs/dev/z を bank7 でも同構造で作るため)
GCOLS = ["id", "well", "last_known_tvt", "pf_z_1", "pf_z_delta_1", "pf_ancc_1",
         "pf_vs_spatial_1", "pf_vs_dense_1", "pf_ancc_mean", "pf_ancc_med"]
tb = pq.read_table(ART / "data" / "train.parquet", columns=GCOLS)
ids = tb.column("id").to_pandas().astype(str).to_numpy()
well = tb.column("well").to_pandas().astype(str).to_numpy()
lk = tb.column("last_known_tvt").to_numpy(zero_copy_only=False).astype(np.float64)
G = {c: tb.column(c).to_numpy(zero_copy_only=False).astype(np.float64) for c in GCOLS[3:]}
# tvt_f_ancc / tvt_dense を bank1 の pf_vs_* から逆算(グローバル予測子)
TVT_FANCC = G["pf_ancc_1"] - G["pf_vs_spatial_1"]
TVT_DENSE = G["pf_ancc_1"] - G["pf_vs_dense_1"]
N = len(ids)
OUT = {}


def put(name, ii, arr):
    a = OUT.get(name)
    if a is None:
        a = np.full(N, np.nan, np.float32); OUT[name] = a
    a[ii] = np.asarray(arr, np.float32)


uniq = pd.unique(well); n_ok = 0
for wid in uniq:
    idx = np.where(well == wid)[0]; m = len(idx)
    ad = ANC_D[idx]
    if not np.isfinite(ad).any():
        continue
    lkw = lk[idx]
    mean_tw = ad + lkw; mean_self = SLF_D[idx] + lkw; mean_nbr = NBR_D[idx] + lkw
    try:
        hw = pd.read_csv(DATA / f"{wid}__horizontal_well.csv"); tw = pd.read_csv(DATA / f"{wid}__typewell.csv").sort_values("TVT")
        x = pf.build_smoother_inputs(hw, tw["TVT"].to_numpy(float), tw["GR"].to_numpy(float), P7)
    except Exception:
        x = None
    if x is None:
        continue
    hgr = np.asarray(x["gr"], np.float32)[:m]; hmd = np.asarray(x["md"], np.float32)[:m]
    tw_tvt = tw["TVT"].to_numpy(float); tw_gr = tw["GR"].to_numpy(float)
    k = min(m, len(hgr)); ii = idx[:k]
    f = {}
    # ancc(tw)
    f["pf_ancc_7"] = mean_tw[:k]; f["pf_ancc_std_7"] = ANC_S[idx][:k]; f["pf_ancc_delta_7"] = ad[:k]
    featv._add_gr_match_features(f, "pf_7", mean_tw[:k], hgr[:k], tw_tvt, tw_gr, OFFS)   # pf_7_gr_*
    featv._add_path_shape_features(f, "pf_7", mean_tw[:k], hmd[:k])                       # pf_7_rate/acc..
    # self
    f["pf_self_7"] = mean_self[:k]; f["pf_self_std_7"] = SLF_S[idx][:k]; f["pf_self_delta_7"] = SLF_D[idx][:k]
    f["pf_self_vs_tw_7"] = (mean_self[:k] - mean_tw[:k]).astype(np.float32)
    # nbr
    f["pf_nbr_7"] = mean_nbr[:k]; f["pf_nbr_std_7"] = NBR_S[idx][:k]; f["pf_nbr_delta_7"] = NBR_D[idx][:k]
    f["pf_nbr_vs_tw_7"] = (mean_nbr[:k] - mean_tw[:k]).astype(np.float32)
    f["pf_self_vs_nbr_7"] = (mean_self[:k] - mean_nbr[:k]).astype(np.float32)
    st = np.stack([mean_tw[:k].astype(float), mean_self[:k].astype(float), mean_nbr[:k].astype(float)], 1)
    f["pf_agree_std_7"] = np.nanstd(st, 1).astype(np.float32)
    f["pf_agree_range_7"] = (np.nanmax(st, 1) - np.nanmin(st, 1)).astype(np.float32)
    # --- 残り18列: z / vs_spatial / vs_dense / tdpf×11 / dev_mean/med(既存列から同構造で) ---
    iiw = idx[:k]; pa = mean_tw[:k].astype(np.float64)
    f["pf_z_7"] = G["pf_z_1"][iiw]                       # pf_z はほぼ全バンク共通(グローバルz推定)
    f["pf_z_delta_7"] = G["pf_z_delta_1"][iiw]
    f["pf_vs_z_7"] = (pa - G["pf_z_1"][iiw]).astype(np.float32)
    f["pf_vs_spatial_7"] = (pa - TVT_FANCC[iiw]).astype(np.float32)
    f["pf_vs_dense_7"] = (pa - TVT_DENSE[iiw]).astype(np.float32)
    f["pf_dev_mean_7"] = (pa - G["pf_ancc_mean"][iiw]).astype(np.float32)
    f["pf_dev_med_7"] = (pa - G["pf_ancc_med"][iiw]).astype(np.float32)
    for o in PF_OFFS:
        f[f"tdpf{int(o)}_7"] = (hgr[:k] - np.interp(pa + o, tw_tvt, tw_gr)).astype(np.float32)
    for name, arr in f.items():
        put(name, ii, arr)
    n_ok += 1

cols = sorted(OUT.keys())
np.savez_compressed(ART / "newpf_full_train.npz", id=ids, **{c: OUT[c] for c in cols})
print(f"saved newpf_full_train.npz  井={n_ok}  派生列数={len(cols)}")
print("列:", ", ".join(cols[:12]), "...")
