# 自由形式メタ(脚予測+特徴を同列入力): ridge加法 / 浅いGBM / 中容量GBM(GroupKFoldメタOOF)
import os, json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import sys
sys.stdout = open(1, "w", encoding="utf-8", closefd=False)

ART = "v62/artifacts_v62"; ART52 = "v52/artifacts_v52"
LEGS = ["GBM", "TCN", "L3img", "L4", "L5", "L6", "L7"]

def main():
    tb = pq.read_table(f"{ART52}/data/train.parquet",
                       columns=["id", "well", "target", "md_since", "pf_ancc", "last_known_tvt",
                                "pf_ancc_std_1", "pf_ancc_std_4"])
    ids = tb.column("id").to_pandas().astype(str)
    gw = tb.column("well").to_pandas().astype(str).to_numpy()
    y = tb.column("target").to_numpy(zero_copy_only=False).astype(np.float64)
    mds = np.maximum(tb.column("md_since").to_numpy(zero_copy_only=False).astype(np.float64), 0)
    pfa = tb.column("pf_ancc").to_numpy(zero_copy_only=False).astype(np.float64)
    lk = tb.column("last_known_tvt").to_numpy(zero_copy_only=False).astype(np.float64)
    s1 = np.nan_to_num(tb.column("pf_ancc_std_1").to_numpy(zero_copy_only=False).astype(np.float64), nan=0)
    s4 = np.nan_to_num(tb.column("pf_ancc_std_4").to_numpy(zero_copy_only=False).astype(np.float64), nan=0)
    gfA = np.load("v48/artifacts_v48/gf_feature.npy"); vbA = np.load(f"{ART52}/v50b_feature.npy")
    gbm = np.load(f"{ART52}/gbm_v41_oof.npy").astype(np.float64)
    nn = np.mean([np.load(f"{ART52}/nn_oof_combo.npz")[k] for k in ("mdn2_s0", "mdn2_s1", "mdn2_s2")], 0).astype(np.float64)
    pp = json.loads(open(f"{ART52}/pp_params.json").read())
    def ppe(r): return (r * (1 - pp["w_pf"]) + (pfa - lk) * pp["w_pf"]) * (1 - np.exp(-mds / pp["tau"])) * pp["alpha"] - y
    eg = ppe(gbm); et = ppe(nn)
    E = {"L3img": {w: v.astype(np.float64) for w, v in np.load(f"{ART}/v62_oof_err.npz").items()},
         "L4": {w: v.astype(np.float64) for w, v in np.load(f"{ART}/l4_oof_err.npz").items()},
         "L5": {w: v.astype(np.float64) for w, v in np.load(f"{ART}/l5_oof_err.npz").items()},
         "L6": {w: v.astype(np.float64) for w, v in np.load(f"{ART}/l6_oof_err.npz").items()},
         "L7": {w: v.astype(np.float64) for w, v in np.load(f"{ART}/l7_oof_err.npz").items()}}
    rn = ids.str.rsplit("_", n=1, expand=True)[1].astype(int).to_numpy()
    WELLS = sorted(E["L5"])
    Pw, Fw, Yw, LKw, Wid = [], [], [], [], []
    for w in WELLS:
        ii = np.where(gw == w)[0]; o = np.argsort(rn[ii], kind="stable"); iio = ii[o]
        n0 = min([len(E[n][w]) for n in E] + [len(iio)])
        ms = np.ones(n0, bool)
        for n in E: ms &= np.isfinite(E[n][w][:n0])
        ms &= np.isfinite(eg[iio[:n0]])
        iiom = iio[:n0][ms]
        errs = np.stack([eg[iiom], et[iiom]] + [E[n][w][:n0][ms] for n in ("L3img", "L4", "L5", "L6", "L7")], 1)
        Pw.append(errs)
        f = np.stack([mds[iiom] / 1000.0, np.full(ms.sum(), np.log1p(n0)),
                      errs.std(1), np.percentile(errs, 75, 1) - np.percentile(errs, 25, 1),
                      s1[iiom], s4[iiom],
                      np.nan_to_num(gfA[iiom, 1], nan=0), np.nan_to_num(vbA[iiom, 1], nan=0),
                      np.nan_to_num(vbA[iiom, 2], nan=0)], 1)
        Fw.append(f); Yw.append(y[iiom]); LKw.append(lk[iiom]); Wid += [w] * int(ms.sum())
    P = np.concatenate(Pw); F = np.concatenate(Fw)
    Yr = np.concatenate(Yw); LKr = np.concatenate(LKw); Wid = np.array(Wid)
    fold_of = {w: f for f in range(5) for w in WELLS[f::5]}
    fid = np.array([fold_of[w] for w in Wid])
    PRED = P + Yr[:, None] - LKr[:, None]                 # 脚予測(残差スケール, test計算可)
    TGT = Yr - LKr
    X = np.concatenate([PRED, F], 1).astype(np.float32)
    N = len(X)
    print(f"rows={N} X={X.shape}")
    def pooled(e): return float(np.sqrt(np.mean(e ** 2)))
    res = {}
    from numpy.linalg import solve
    A = np.concatenate([X, np.ones((N, 1), np.float32)], 1)
    e_oof = np.zeros(N)
    lam = 1e-3 * N
    for f in range(5):
        tr = fid != f; va = ~tr
        At = A[tr].astype(np.float64)
        w_ = solve(At.T @ At + lam * np.eye(A.shape[1]), At.T @ TGT[tr])
        e_oof[va] = A[va] @ w_ - TGT[va]
    res["ridge_add"] = pooled(e_oof)
    import lightgbm as lgb
    for tag, prm, nr in (("gbm_shallow", dict(num_leaves=15, min_data_in_leaf=2000, learning_rate=0.05,
                                              feature_fraction=0.8, bagging_fraction=0.7, bagging_freq=1,
                                              lambda_l2=50.0, verbose=-1, num_threads=os.cpu_count() - 2, seed=7), 400),
                         ("gbm_mid", dict(num_leaves=63, min_data_in_leaf=500, learning_rate=0.05,
                                          feature_fraction=0.8, bagging_fraction=0.7, bagging_freq=1,
                                          lambda_l2=10.0, verbose=-1, num_threads=os.cpu_count() - 2, seed=7), 800)):
        e_oof = np.zeros(N)
        for f in range(5):
            tr = np.where(fid != f)[0]; va = np.where(fid == f)[0]
            m = lgb.train(dict(objective="regression", **prm), lgb.Dataset(X[tr], TGT[tr]), num_boost_round=nr)
            e_oof[va] = m.predict(X[va]) - TGT[va]
        res[tag] = pooled(e_oof)
        print(f"{tag} fold別:", [round(pooled(e_oof[fid == f2]), 4) for f2 in range(5)])
    print("\n=== 自由形式メタ(脚+特徴を同列入力, メタOOF) ===")
    print("  fixed(基準)     : 5.8253")
    for k, v in sorted(res.items(), key=lambda t: t[1]):
        print(f"  {k:16s}: {v:.4f}")

if __name__ == "__main__":
    main()
