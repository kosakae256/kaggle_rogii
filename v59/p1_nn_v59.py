# v59 P1 — GR-free 学習補正: target = y − st を LightGBM(GroupKFold OOF)で学習
#   特徴(全てGR-free): 面系(p_abs−st, p_b2−st, D−st, dist, σ(dist))/ 幾何(dz累積, 蛇行, md_since,
#   prefix長, prefix dip(ir), 井戸長)/ struct系(st−lk, st勾配)。
#   併せて井戸レベルST×面ブレンド(w定数/ndゲート)も測る。参照: ST(このプロトコル)=10.368。
import os, sys, json, time
import numpy as np
import pandas as pd

P0 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(P0, "v51"))
TRAIN = os.path.join(P0, "rogii-wellbore-geology-prediction", "train")
ART = os.path.join(P0, "v59", "artifacts_v59")

def main():
    import nnpf_v51 as npf
    import lightgbm as lgb
    from sklearn.model_selection import GroupKFold
    t0 = time.time()
    D = np.load(os.path.join(ART, "geofree_rows.npz"))
    wells = sorted(D.files)
    ST = {w: np.asarray(npf.SIMD[w]["st"], np.float64) for w in npf.SIMD if w in set(wells)}
    rows = {"w": [], "y": [], "st": [], "feat": []}
    BLEND = {f"w{wv}": {} for wv in (0.15, 0.3, 0.5)}
    BLEND["ndgate"] = {}
    YW = {}; STP = {}
    for w in wells:
        A = D[w].astype(np.float64)             # [p_abs, p_c, p_b2, D_mid, dist]
        hw = pd.read_csv(os.path.join(TRAIN, f"{w}__horizontal_well.csv"),
                         usecols=["MD", "X", "Y", "Z", "TVT", "TVT_input"])
        kn = hw["TVT_input"].notna().to_numpy(); ev = ~kn
        y = hw["TVT"].to_numpy(float)[ev]
        md = hw["MD"].to_numpy(float); z = hw["Z"].to_numpy(float)
        x = hw["X"].to_numpy(float); yy = hw["Y"].to_numpy(float)
        ilast = np.where(kn)[0][-1]
        mds = np.maximum(md[ev] - md[ilast], 0)
        stw = ST.get(w)
        if stw is None: continue
        nev = int(ev.sum()); m2 = min(nev, len(stw), len(A))
        stv = stw[:m2]; yv = y[:m2]
        YW[w] = yv; STP[w] = stv
        pb2 = A[:m2, 2]; pabs = A[:m2, 0]; dmid = A[:m2, 3]; dist = A[:m2, 4]
        # 井戸レベルブレンド
        for wv in (0.15, 0.3, 0.5):
            BLEND[f"w{wv}"][w] = (1 - wv) * stv + wv * pb2
        ndw = float(np.nanmedian(dist))
        wg = 0.35 if ndw < 300 else (0.25 if ndw < 800 else (0.12 if ndw < 2000 else 0.0))
        BLEND["ndgate"][w] = (1 - wg) * stv + wg * pb2
        # P1特徴
        tail = hw[kn].tail(30)
        dt = np.diff(tail["TVT_input"].to_numpy(float)); dm = np.diff(tail["MD"].to_numpy(float))
        ir = float(np.median(dt[dm > 0] / dm[dm > 0])) if (dm > 0).sum() >= 3 else 0.0
        lkv = float(hw.loc[kn, "TVT_input"].iloc[-1])
        stg = np.gradient(stv) if m2 > 1 else np.zeros(1)
        zev = z[ev][:m2]
        F = np.stack([
            pabs - stv, pb2 - stv, dmid - stv, np.log1p(np.maximum(dist, 0)),
            mds[:m2] / 1000.0, np.full(m2, ir * 1000), stv - lkv, stg * 10,
            np.gradient(zev) if m2 > 1 else np.zeros(1),
            np.full(m2, np.log1p(nev)),
            np.hypot(x[ev][:m2] - x[ilast], yy[ev][:m2] - yy[ilast]) / 1000.0,
        ], 1).astype(np.float32)
        rows["w"] += [w] * m2; rows["feat"].append(F)
        rows["y"].append(yv - stv)              # 残差ターゲット
    X = np.concatenate(rows["feat"]); yres = np.concatenate(rows["y"]).astype(np.float32)
    gw = np.array(rows["w"])
    fin = np.isfinite(yres)
    print(f"rows={len(yres)} wells={len(YW)} ({time.time()-t0:.0f}s)", flush=True)
    def pooled(dh):
        es = [np.asarray(dh[w], np.float64) - YW[w] for w in dh]
        ee = np.concatenate(es); ee = ee[np.isfinite(ee)]
        return float(np.sqrt(np.mean(ee ** 2)))
    print(f"ST(基準) = {pooled(STP):.4f}")
    for k in BLEND:
        print(f"  blend {k:8s}: {pooled(BLEND[k]):.4f}", flush=True)
    # P1: LightGBM OOF
    gkf = GroupKFold(5)
    oof = np.zeros(len(yres), np.float32)
    names = ["abs_st", "b2_st", "d_st", "logdist", "mds", "ir", "st_lk", "stg", "dzg", "logT", "xy1k"]
    for k, (tr, va) in enumerate(gkf.split(X, yres, groups=gw)):
        trf = tr[np.isfinite(yres[tr])]
        m = lgb.train(dict(objective="regression", learning_rate=0.05, num_leaves=63,
                           min_data_in_leaf=200, feature_fraction=0.9, bagging_fraction=0.8,
                           bagging_freq=1, lambda_l2=5.0, verbose=-1,
                           num_threads=max(os.cpu_count() - 2, 2), seed=7),
                      lgb.Dataset(X[trf], yres[trf], feature_name=names),
                      num_boost_round=600)
        oof[va] = m.predict(X[va])
        print(f"  P1 fold{k} done ({time.time()-t0:.0f}s)", flush=True)
    P1 = {}
    i0 = 0
    for w in YW:
        m2 = len(YW[w])
        P1[w] = STP[w] + oof[i0:i0 + m2]
        i0 += m2
    r1 = pooled(P1)
    print(f"\nP1(ST + NN補正) = {r1:.4f}  (ST {pooled(STP):.4f})")
    for f in range(5):
        val = set(sorted(YW)[f::5])
        es = np.concatenate([P1[w] - YW[w] for w in YW if w in val])
        es = es[np.isfinite(es)]
        print(f"  P1 fold{f}: {np.sqrt(np.mean(es**2)):.3f}")
    imp = sorted(zip(names, m.feature_importance("gain")), key=lambda t: -t[1])[:8]
    print("gain上位:", [(n, int(g)) for n, g in imp])
    json.dump(dict(st=pooled(STP), p1=r1, blends={k: pooled(BLEND[k]) for k in BLEND}),
              open(os.path.join(ART, "p1_result.json"), "w"), indent=1)

if __name__ == "__main__":
    main()
