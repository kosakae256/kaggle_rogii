# v56→v52 特徴注入 A/B(LightGBMのみ, CPU, 高速プローブ構成)
#   ARM=base : v52特徴(features.json 788列: parquet781+gf3+v50b4)
#   ARM=v56  : 上に v56 4列(surf_delta/surf_dist/surf_slope/surf_fmad)を追加
#   fold: GroupKFold(5, groups=well) — v52 tabular と同一。FOLDS env で部分実行(既定 "0")。
#   注意: 本家はGPU LGBM(lr0.03,5000本)。ここはCPUプローブ(lr0.05,3000本,ES200)=両腕同一条件のΔ判定専用。
import os, sys, json, time, gc
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

P0 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART52 = os.path.join(P0, "v52", "artifacts_v52")
ART56 = os.path.join(P0, "v56", "artifacts_v56")
FWD_PQ = os.path.join(ART52, "data", "train.parquet")
ARM = os.environ.get("ARM", "v56")
FOLDS = [int(x) for x in os.environ.get("FOLDS", "0").split(",")]
V56COLS = ["surf_delta", "surf_dist", "surf_slope", "surf_fmad"]

def build_v56_feature(N, ids, wname, lk):
    d = np.load(os.path.join(ART56, "lowo_rows.npz"))
    rn = pd.Series(ids).str.rsplit("_", n=1, expand=True)[1].astype(int).to_numpy()
    F = np.full((N, 4), np.nan, np.float32); miss = 0
    for wn in pd.unique(wname):
        ii = np.where(wname == wn)[0]
        o = np.argsort(rn[ii], kind="stable"); iis = ii[o]
        if wn not in d.files: miss += 1; continue
        A = d[wn]                                    # (nev,4)=[tvt_surf,dist,slope,fmad] eval行CSV順
        n0 = min(len(iis), len(A))
        F[iis[:n0], 0] = A[:n0, 0] - lk[iis[:n0]]    # surf_delta
        F[iis[:n0], 1] = A[:n0, 1]; F[iis[:n0], 2] = A[:n0, 2]; F[iis[:n0], 3] = A[:n0, 3]
    print(f"v56特徴: 井戸miss={miss}  NaN率={np.isnan(F[:,0]).mean():.2%}")
    return np.nan_to_num(F, nan=0.0)

def main():
    t0 = time.time()
    import lightgbm as lgb
    from sklearn.model_selection import GroupKFold
    feat52 = json.loads(open(os.path.join(ART52, "features.json")).read())
    schema = set(pq.read_schema(FWD_PQ).names)
    NPY = {"gf_delta": 0, "gf_std": 1, "gf_vs_struct": 2}
    NPY2 = {"v50b_delta": 0, "v50b_std": 1, "v50b_margin": 2, "v50b_maxp": 3}
    gfA = np.load(os.path.join(P0, "v48", "artifacts_v48", "gf_feature.npy"))
    bA = np.load(os.path.join(ART52, "v50b_feature.npy"))
    N = pq.ParquetFile(FWD_PQ).metadata.num_rows
    cols = feat52 + (V56COLS if ARM == "v56" else [])
    X = np.empty((N, len(cols)), np.float32)
    for j0 in range(0, len(feat52), 24):
        sub = [c for c in feat52[j0:j0 + 24] if c in schema]
        tb = pq.read_table(FWD_PQ, columns=sub) if sub else None
        for jj, c in enumerate(feat52[j0:j0 + 24]):
            j = j0 + jj
            if c in schema:
                X[:, j] = tb.column(c).to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
            elif c in NPY:
                X[:, j] = np.nan_to_num(gfA[:N, NPY[c]], nan=0.0)
            elif c in NPY2:
                X[:, j] = np.nan_to_num(bA[:N, NPY2[c]], nan=0.0)
            else:
                X[:, j] = np.nan
        del tb; gc.collect()
    sf = pq.read_table(FWD_PQ, columns=["id", "well", "target", "last_known_tvt"])
    ids = sf.column("id").to_pandas().astype(str)
    g = sf.column("well").to_pandas().to_numpy()
    y = sf.column("target").to_numpy(zero_copy_only=False).astype(np.float32)
    lk = sf.column("last_known_tvt").to_numpy(zero_copy_only=False).astype(np.float64)
    del sf
    if ARM == "v56":
        X[:, len(feat52):] = build_v56_feature(N, ids, g, lk)
    print(f"[{ARM}] X {X.shape} {X.nbytes/1e9:.1f}GB  ({time.time()-t0:.0f}s)", flush=True)
    params = dict(objective="regression", metric="rmse", num_leaves=255, min_data_in_leaf=15,
                  bagging_fraction=0.8, bagging_freq=1, feature_fraction=0.8,
                  lambda_l2=3.0, lambda_l1=0.05, verbose=-1, num_threads=max(os.cpu_count() - 4, 2),
                  max_bin=255, learning_rate=0.05, seed=123)
    gkf = GroupKFold(n_splits=5); folds = list(gkf.split(np.zeros(len(y)), y, groups=g))
    # ★省メモリ: 一度ビン化して生配列を解放。foldはビンレベルsubset(コピー無し)
    ds = lgb.Dataset(X, label=y, feature_name=list(cols), free_raw_data=True)
    ds.construct()
    del X; gc.collect()
    print(f"[{ARM}] Dataset構築済・生配列解放 ({time.time()-t0:.0f}s)", flush=True)
    res = {}
    for k in FOLDS:
        tr, va = folds[k]
        dtr = ds.subset(sorted(tr)); dva = ds.subset(sorted(va))
        evals = {}
        bst = lgb.train(params, dtr, num_boost_round=3000, valid_sets=[dva],
                        callbacks=[lgb.early_stopping(200, verbose=False),
                                   lgb.record_evaluation(evals), lgb.log_evaluation(0)])
        r = float(min(evals["valid_0"]["rmse"]))
        res[k] = dict(rmse=r, best_iter=int(bst.best_iteration))
        print(f"[{ARM}] fold{k}: rmse={r:.4f} (iter {bst.best_iteration}, {time.time()-t0:.0f}s)", flush=True)
        if ARM == "v56":
            imp = dict(zip(cols, bst.feature_importance("gain")))
            order = sorted(cols, key=lambda c: -imp[c])
            print("  v56列gain順位:", {c: order.index(c) + 1 for c in V56COLS}, flush=True)
        del bst, dtr, dva; gc.collect()
    json.dump(res, open(os.path.join(ART56, f"gbm_ab_{ARM}_{'-'.join(map(str,FOLDS))}.json"), "w"), indent=1)
    print(f"[{ARM}] done {res}")

if __name__ == "__main__":
    main()
