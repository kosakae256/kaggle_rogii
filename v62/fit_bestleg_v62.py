# 「どの脚が最良か」判別モデル(ユーザー案): 7クラス分類(行レベル/井戸レベル)
#   入力=特徴+各脚予測(相対化)。出力=best-leg確率 → hard argmax / soft確率重み の両方を評価。
#   GroupKFold(5) メタOOF。基準: fixed 5.8253 / oracle井戸別 4.891。
import os, json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import sys
sys.stdout = open(1, "w", encoding="utf-8", closefd=False)

ART = "v62/artifacts_v62"; ART52 = "v52/artifacts_v52"
LEGS = ["GBM", "TCN", "L3img", "L4", "L5", "L6", "L7"]

def main():
    import lightgbm as lgb
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
    Pw, Fw, Wid = [], [], []
    for w in WELLS:
        ii = np.where(gw == w)[0]; o = np.argsort(rn[ii], kind="stable"); iio = ii[o]
        n0 = min([len(E[n][w]) for n in E] + [len(iio)])
        ms = np.ones(n0, bool)
        for n in E: ms &= np.isfinite(E[n][w][:n0])
        ms &= np.isfinite(eg[iio[:n0]])
        iiom = iio[:n0][ms]
        errs = np.stack([eg[iiom], et[iiom]] + [E[n][w][:n0][ms] for n in ("L3img", "L4", "L5", "L6", "L7")], 1)
        Pw.append(errs)
        med = np.median(errs, 1)
        f = np.concatenate([errs - med[:, None],                       # 脚の相対配置(y相殺=合法)
                            np.stack([mds[iiom] / 1000.0, errs.std(1),
                                      np.percentile(errs, 75, 1) - np.percentile(errs, 25, 1),
                                      s1[iiom], s4[iiom],
                                      np.nan_to_num(gfA[iiom, 1], nan=0),
                                      np.nan_to_num(vbA[iiom, 1], nan=0),
                                      np.nan_to_num(vbA[iiom, 2], nan=0)], 1)], 1)
        Fw.append(f); Wid += [w] * int(ms.sum())
    P = np.concatenate(Pw); F = np.concatenate(Fw).astype(np.float32); Wid = np.array(Wid)
    fold_of = {w: f for f in range(5) for w in WELLS[f::5]}
    fid = np.array([fold_of[w] for w in Wid])
    N, K = P.shape
    def pooled(e): return float(np.sqrt(np.mean(e ** 2)))
    print(f"rows={N}")
    # ---- 行レベル: best-leg 7クラス分類 ----
    lab = np.argmin(np.abs(P), 1)
    prm = dict(objective="multiclass", num_class=K, num_leaves=31, min_data_in_leaf=1000,
               learning_rate=0.07, feature_fraction=0.8, bagging_fraction=0.7, bagging_freq=1,
               lambda_l2=20.0, verbose=-1, num_threads=4, seed=7)
    e_hard = np.zeros(N); e_soft = np.zeros(N); acc = np.zeros(N, bool)
    for f in range(5):
        tr = np.where(fid != f)[0]; va = np.where(fid == f)[0]
        m = lgb.train(prm, lgb.Dataset(F[tr], lab[tr]), num_boost_round=300)
        pr = m.predict(F[va])                                    # (n,K)
        pick = pr.argmax(1)
        e_hard[va] = P[va, pick]
        e_soft[va] = (P[va] * pr).sum(1)
        acc[va] = pick == lab[va]
    print(f"[行レベル分類] top1精度={acc.mean():.3f} (base={np.bincount(lab).max()/N:.3f})")
    print(f"  hard選択: {pooled(e_hard):.4f} / soft確率重み: {pooled(e_soft):.4f}")
    # ---- 井戸レベル: best-leg 分類(773サンプル) ----
    Wl = []; Flw = []; labw = []
    for i, w in enumerate(WELLS):
        m = Wid == w
        errs = P[m]
        rmses = np.sqrt(np.mean(errs ** 2, 0))
        labw.append(int(np.argmin(rmses)))
        dev = errs - np.median(errs, 1)[:, None]
        Flw.append(np.concatenate([np.abs(dev).mean(0), dev.std(0),
                                   [errs.std(1).mean(), mds[gw == w].max() / 1000.0, np.log1p(m.sum())]]))
        Wl.append(w)
    Flw = np.array(Flw, np.float32); labw = np.array(labw)
    e_hw = np.zeros(N); e_sw = np.zeros(N); accw = np.zeros(len(WELLS), bool)
    for f in range(5):
        trw = [i for i, w in enumerate(WELLS) if fold_of[w] != f]
        vaw = [i for i, w in enumerate(WELLS) if fold_of[w] == f]
        m = lgb.train(dict(objective="multiclass", num_class=K, num_leaves=7, min_data_in_leaf=30,
                           learning_rate=0.07, feature_fraction=0.8, lambda_l2=10.0, verbose=-1, num_threads=4, seed=7),
                      lgb.Dataset(Flw[trw], labw[trw]), num_boost_round=200)
        pr = m.predict(Flw[vaw])
        for j, wi_ in enumerate(vaw):
            w = WELLS[wi_]; msk = Wid == w
            pick = int(pr[j].argmax())
            e_hw[msk] = P[msk, pick]
            e_sw[msk] = (P[msk] * pr[j][None, :]).sum(1)
            accw[wi_] = pick == labw[wi_]
    print(f"[井戸レベル分類] top1精度={accw.mean():.3f} (base={np.bincount(labw).max()/len(WELLS):.3f})")
    print(f"  hard選択: {pooled(e_hw):.4f} / soft確率重み: {pooled(e_sw):.4f}")
    print(f"\n基準: fixed 5.8253 / oracle井戸別 4.891 / oracle行別 4.400")

if __name__ == "__main__":
    main()
