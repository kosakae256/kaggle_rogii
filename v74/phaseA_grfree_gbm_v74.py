# v74 Phase A — GRフリー GBM(既存struct資産は一切不使用・ゼロから)
#   target: drift d(t) = TVT(t) − TVT_last(hold からのドリフト)。TVT̂ = TVT_last + pred。
#   特徴(全て test 実在列 MD,X,Y,Z,TVT_input と「他坑井のtrainデータ」のみ):
#     [自分prefix] dip外挿デルタ(500/1000/2000)・操舵κ・prefix TVT std・境界Z勾配
#     [軌道]       md_since・Zバンド差分(301/1001/2001/4001)・HPバンド値・局所勾配・方位sin/cos・方位変化
#     [近傍K=3]    形状転写 dg_k = g_k(現在位置への射影) − g_k(境界位置への射影)(絶対レベル不使用)
#                  ・距離・相対方位・近傍局所dip・境界レベル差
#   CV: GroupKFold(5) by well(プロジェクト標準)。overall pooled RMSE + md_since帯 + 井戸別ECDF。
# 出力: v74/artifacts_v74/phaseA_result.json / phaseA.png / oof npz
# env: NW(0=全), STRIDE_TR(学習行間引き, 既定3), NBK(近傍数3), NONBR=1で近傍特徴オフ(アブレーション)
import os, json, time
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

P0 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN = os.path.join(P0, "rogii-wellbore-geology-prediction", "train")
ART = os.path.join(P0, "v74", "artifacts_v74"); os.makedirs(ART, exist_ok=True)
NW = int(os.environ.get("NW", "0"))
STRIDE_TR = int(os.environ.get("STRIDE_TR", "3"))
NBK = int(os.environ.get("NBK", "3"))
NONBR = os.environ.get("NONBR", "0") == "1"
WINS = [301, 1001, 2001, 4001]

def lowpass(x, win):
    return pd.Series(x).rolling(win, center=True, min_periods=1).mean().to_numpy()

def robust_slope(md, y):
    if len(md) < 30:
        return 0.0
    c = np.polyfit(md, y, 1)
    r = y - np.polyval(c, md)
    s = np.std(r) + 1e-9
    m = np.abs(r) < 2.5 * s
    if m.sum() >= 20:
        c = np.polyfit(md[m], y[m], 1)
    return float(c[0])

def load_wells():
    wells = sorted(set(f.split("__")[0] for f in os.listdir(TRAIN)
                       if f.endswith("__horizontal_well.csv")))
    if NW:
        wells = wells[:NW]
    D = {}
    for w in wells:
        hw = pd.read_csv(os.path.join(TRAIN, f"{w}__horizontal_well.csv"),
                         usecols=["MD", "X", "Y", "Z", "TVT", "TVT_input"])
        md = hw["MD"].to_numpy(float); x = hw["X"].to_numpy(float)
        y = hw["Y"].to_numpy(float); z = hw["Z"].to_numpy(float)
        tvt = hw["TVT"].to_numpy(float); tin = hw["TVT_input"].to_numpy(float)
        kn = np.isfinite(tin) & np.isfinite(tvt)
        ev = (~np.isfinite(tin)) & np.isfinite(tvt)
        if kn.sum() < 100 or ev.sum() < 50:
            continue
        D[w] = dict(md=md, x=x, y=y, z=z, tvt=tvt, tin=tin, kn=kn, ev=ev,
                    g=tvt + z, t0=int(np.where(kn)[0][-1]))
    return D

def build_features(D):
    wells = sorted(D.keys())
    # 近傍井戸(centroid距離)と各井戸トラックのKDTree
    cent = np.array([[np.nanmean(D[w]["x"]), np.nanmean(D[w]["y"])] for w in wells])
    trees = {}
    for w in wells:
        d = D[w]
        # ラテラル部のみ射影対象(ビルド区間のg巨大変動の混入を排除)
        lat = np.abs(np.gradient(lowpass(d["z"], 301))) < 0.3
        lidx = np.where(lat)[0][::2]
        if len(lidx) > 100:
            d["lidx"] = lidx
            trees[w] = cKDTree(np.c_[d["x"][lidx], d["y"][lidx]])
    nbrs = {}
    for i, w in enumerate(wells):
        dd = np.hypot(cent[:, 0] - cent[i, 0], cent[:, 1] - cent[i, 1])
        dd[i] = 1e18
        nbrs[w] = [wells[j] for j in np.argsort(dd) if wells[j] in trees][:NBK]
    X_all, y_all, grp, wid_rows, msb_rows = [], [], [], [], []
    names = None
    for gi, w in enumerate(wells):
        d = D[w]
        md, x, y, z, g, tin = d["md"], d["x"], d["y"], d["z"], d["g"], d["tin"]
        t0 = d["t0"]; ei = np.where(d["ev"])[0]
        n = len(md)
        tvt_last = float(tin[t0])
        # --- prefix 統計
        ki = np.where(d["kn"])[0]
        dips = {}
        for L in (500, 1000, 2000):
            tl = ki[-min(len(ki), L):]
            dips[L] = robust_slope(md[tl], g[tl])
        zh301 = z - lowpass(z, 301)
        tl = ki[-min(len(ki), 1000):]
        gt = (g - lowpass(g, 301))[tl]; zt = zh301[tl]
        vz = float(np.dot(zt, zt))
        k_pref = float(np.clip(np.dot(zt, gt) / vz, 0, 1.5)) if vz > 1e-9 else 0.0
        pre_std = float(np.std(np.diff(tin[ki[-500:]][np.isfinite(tin[ki[-500:]])])
                        )) if len(ki) > 50 else 0.0
        # --- 軌道
        dx = np.gradient(x); dy = np.gradient(y)
        az = np.arctan2(dy, dx)
        az_s = np.arctan2(lowpass(np.sin(az), 301), lowpass(np.cos(az), 301))
        dz = np.gradient(lowpass(z, 101))
        bands = {W: lowpass(z, W) for W in WINS}
        # 掘削角度(ユーザー指摘): インクリネーション(水平からの傾き)・ビルドレート・ドッグレッグ累積
        dh = np.hypot(np.gradient(lowpass(x, 101)), np.gradient(lowpass(y, 101))) + 1e-9
        incl = np.arctan2(np.gradient(lowpass(z, 301)), dh)          # rad, 0=水平
        build = np.gradient(lowpass(incl, 201)) * 1000               # ビルドレート(mrad/ft)
        dogleg = np.cumsum(np.abs(np.gradient(az_s)) + np.abs(np.gradient(incl)))
        # --- 近傍 形状転写
        nb_feats = []
        if not NONBR:
            p0 = np.array([x[t0], y[t0]])
            for k, nw in enumerate(nbrs[w]):
                nd = D[nw]
                tr = trees[nw]; lidx = nd["lidx"]
                _, j0 = tr.query(p0)
                j0 = int(lidx[int(j0)])
                dd_t, jj = tr.query(np.c_[x[ei], y[ei]])
                jj = lidx[np.asarray(jj, int)]
                gk = nd["g"]
                # 近傍gの平滑(row-levelノイズ除去)とローカルdip
                gks = lowpass(gk, 201)
                dipk = np.gradient(gks, nd["md"], edge_order=1)
                dgk = np.clip(gks[jj] - gks[j0], -80, 80)
                lvl0 = float(gks[j0] - g[t0])
                raz = az_s[ei] - np.arctan2(np.gradient(nd["y"])[jj], np.gradient(nd["x"])[jj])
                nb_feats.append(dict(dist=dd_t, dg=dgk, dip=dipk[jj] * 1000,
                                     lvl0=np.full(len(ei), lvl0),
                                     cos=np.cos(raz), sin=np.sin(raz)))
        # --- 行列組み立て
        msb = (np.arange(n) - t0).astype(float)
        cols = {
            "md_since": msb[ei],
            "frac": msb[ei] / max(n - t0, 1),
            "dz_local": dz[ei] * 1000,
            "az_sin": np.sin(az_s[ei]), "az_cos": np.cos(az_s[ei]),
            "daz": np.unwrap(az_s)[ei] - az_s[t0],
            "incl": incl[ei] * 1000, "build": build[ei],
            "incl_d0": (incl[ei] - incl[t0]) * 1000,
            "dogleg_d": (dogleg[ei] - dogleg[t0]).astype(float),
            "k_pref": np.full(len(ei), k_pref),
            "pre_std": np.full(len(ei), pre_std),
            "tvt_last": np.full(len(ei), tvt_last),
        }
        for L, sl in dips.items():
            cols[f"dip{L}_delta"] = sl * (md[ei] - md[t0])          # dip外挿のドリフト予測
        for W in WINS:
            cols[f"zlp{W}_d"] = bands[W][ei] - bands[W][t0]          # 軌道低周波の境界からの変化
            cols[f"zhp{W}"] = (z - bands[W])[ei]                     # 高周波バンド値
        if not NONBR:
            for k, nf in enumerate(nb_feats):
                for key, v in nf.items():
                    cols[f"nb{k}_{key}"] = v
        if names is None:
            names = list(cols.keys())
        X_all.append(np.c_[tuple(cols[c] for c in names)].astype(np.float32))
        y_all.append((d["tvt"][ei] - tvt_last).astype(np.float32))
        grp.append(np.full(len(ei), gi, np.int32))
        wid_rows.append(w); msb_rows.append(msb[ei].astype(np.float32))
    return (wells, names, np.concatenate(X_all), np.concatenate(y_all),
            np.concatenate(grp), np.concatenate(msb_rows))

def main():
    t0 = time.time()
    import lightgbm as lgb
    from sklearn.model_selection import GroupKFold
    D = load_wells()
    wells, names, X, yv, grp, msb = build_features(D)
    print(f"wells={len(wells)}  rows={len(yv)}  feats={len(names)}  ({time.time()-t0:.0f}s)",
          flush=True)
    oof = np.full(len(yv), np.nan, np.float32)
    gkf = GroupKFold(n_splits=5)
    params = dict(objective="l2", learning_rate=0.05, num_leaves=127,
                  min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8,
                  bagging_freq=1, lambda_l2=1.0, verbosity=-1, num_threads=os.cpu_count())
    imp = np.zeros(len(names))
    for fi, (tr, va) in enumerate(gkf.split(X, yv, grp)):
        tr = tr[::STRIDE_TR]
        m = lgb.train(params, lgb.Dataset(X[tr], yv[tr]), num_boost_round=3000,
                      valid_sets=[lgb.Dataset(X[va[::10]], yv[va[::10]])],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[va] = m.predict(X[va], num_iteration=m.best_iteration).astype(np.float32)
        imp += m.feature_importance("gain")
        e = oof[va] - yv[va]
        print(f"fold{fi}: n_tr={len(tr)} best_it={m.best_iteration} "
              f"fold RMSE={np.sqrt(np.mean(e**2)):.4f}  ({time.time()-t0:.0f}s)", flush=True)
    err = oof - yv
    overall = float(np.sqrt(np.mean(err ** 2)))
    hold = float(np.sqrt(np.mean(yv ** 2)))
    bands = [(0, 1000), (1000, 2000), (2000, 4000), (4000, 8000), (8000, 1e18)]
    band_rmse = [float(np.sqrt(np.mean(err[(msb >= a) & (msb < b)] ** 2))) for a, b in bands]
    perw = {}
    for gi, w in enumerate(wells):
        m = grp == gi
        perw[w] = float(np.sqrt(np.mean(err[m] ** 2)))
    print(f"\n[v74 Phase A GRフリーGBM] overall = {overall:.4f}  (hold {hold:.4f} / 参考struct 9.70)")
    print("  md_since帯:", " ".join(f"{v:6.3f}" for v in band_rmse))
    print("  井戸別 med =", float(np.median(list(perw.values()))))
    top = np.argsort(imp)[::-1][:15]
    for i in top:
        print(f"    {names[i]:16s} {imp[i]/imp.sum()*100:5.1f}%")
    res = dict(overall=overall, hold=hold, band_rmse=band_rmse,
               per_well_median=float(np.median(list(perw.values()))),
               n_wells=len(wells), n_rows=int(len(yv)),
               importance={names[i]: float(imp[i] / imp.sum()) for i in top},
               nonbr=NONBR, stride_tr=STRIDE_TR)
    with open(os.path.join(ART, "phaseA_result.json"), "w", encoding="utf-8") as f:
        json.dump(dict(res, per_well=perw), f, ensure_ascii=False, indent=1)
    np.savez_compressed(os.path.join(ART, "phaseA_oof.npz"),
                        oof=oof, y=yv, grp=grp, msb=msb, wells=np.array(wells))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "Meiryo"
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 5))
    vals = np.sort(list(perw.values()))
    ax[0].plot(vals, np.arange(1, len(vals) + 1) / len(vals), label="Phase A GBM")
    ax[0].axvline(overall, color="tab:blue", ls=":", lw=1)
    ax[0].axvline(9.70, color="gray", ls="--", lw=0.8, label="参考: struct 9.70")
    ax[0].set_xlim(0, 30); ax[0].set_xlabel("井戸RMSE (ft)")
    ax[0].set_title(f"井戸別RMSE ECDF (overall={overall:.3f})")
    ax[0].legend(); ax[0].grid(alpha=0.3)
    ctr = [(a + min(b, 12000)) / 2 for a, b in bands]
    ax[1].plot(ctr, band_rmse, marker="o")
    ax[1].set_xlabel("md_since (ft)"); ax[1].set_ylabel("RMSE (ft)")
    ax[1].set_title("距離帯別RMSE(ドリフトの残り方)")
    ax[1].grid(alpha=0.3)
    fig.suptitle("v74 Phase A — GRフリーGBM(struct不使用・ゼロから)", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(ART, "phaseA.png"), dpi=130)
    print("saved:", os.path.join(ART, "phaseA.png"))

if __name__ == "__main__":
    main()
