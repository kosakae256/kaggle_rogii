# v56_feature.npy 生成 — lowo_rows.npz(eval行CSV順)を v52 train.parquet の行順に整列
# 列: [surf_delta(=tvt_surf−last_known_tvt), surf_dist, surf_slope, surf_fmad]
# ガード: parquet井戸数<700で異常中断(v52部分キャッシュ事故の教訓)
import os, sys
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

P0 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART52 = os.path.join(P0, "v52", "artifacts_v52")
ART56 = os.path.join(P0, "v56", "artifacts_v56")
PQ = os.path.join(ART52, "data", "train.parquet")

def main():
    d = np.load(os.path.join(ART56, "lowo_rows.npz"))
    tb = pq.read_table(PQ, columns=["id", "well", "last_known_tvt"])
    ids = tb.column("id").to_pandas().astype(str)
    wname = tb.column("well").to_pandas().astype(str).to_numpy()
    lk = tb.column("last_known_tvt").to_numpy(zero_copy_only=False).astype(np.float64)
    nw = pd.unique(wname).size
    assert nw >= 700, f"parquet井戸数 {nw} < 700(部分キャッシュ疑い。createやり直し)"
    rn = ids.str.rsplit("_", n=1, expand=True)[1].astype(int).to_numpy()
    N = len(ids)
    F = np.full((N, 4), np.nan, np.float32); miss = 0
    for wn in pd.unique(wname):
        ii = np.where(wname == wn)[0]
        o = np.argsort(rn[ii], kind="stable"); iis = ii[o]
        if wn not in d.files: miss += 1; continue
        A = d[wn]
        n0 = min(len(iis), len(A))
        F[iis[:n0], 0] = A[:n0, 0] - lk[iis[:n0]]
        F[iis[:n0], 1] = A[:n0, 1]; F[iis[:n0], 2] = A[:n0, 2]; F[iis[:n0], 3] = A[:n0, 3]
    np.save(os.path.join(ART56, "v56_feature.npy"), F)
    print(f"saved v56_feature.npy [{N},4] miss井戸={miss} NaN率={np.isnan(F[:,0]).mean():.2%}"
          f"  (cols: surf_delta/surf_dist/surf_slope/surf_fmad)")

if __name__ == "__main__":
    main()
