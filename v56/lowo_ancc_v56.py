# v56 LOWO — ANCC面の leave-one-well-out 局所平面補間エンジン
#   予測: TVT̂ = b̂ + ANCĈ(x,y) − Z,  b̂ = median_prefix(TVT_input + Z − ANCĈ)
#   面補間: 自井戸を除く点群のKNN(≤K)→Gaussian距離重み付き平面fit(ロバスト1回再重み)
# 出力: artifacts_v56/lowo_result.json / lowo_per_well.npz / lowo_graphs.png
# env: NW(井戸数上限, smoke用), STRIDE(点群間引き, 既定10), QSTRIDE(予測点間引き, 既定5),
#      K(近傍点数128), RMAX(探索半径ft 4000), SIG0(重み距離スケール下限 300)
import os, sys, json, time
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

P0 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN = os.path.join(P0, "rogii-wellbore-geology-prediction", "train")
ART = os.path.join(P0, "v56", "artifacts_v56"); os.makedirs(ART, exist_ok=True)
NW = int(os.environ.get("NW", "0"))
STRIDE = int(os.environ.get("STRIDE", "10")); QSTRIDE = int(os.environ.get("QSTRIDE", "5"))
K = int(os.environ.get("K", "128")); RMAX = float(os.environ.get("RMAX", "4000"))
SIG0 = float(os.environ.get("SIG0", "300")); LAMS = float(os.environ.get("LAMS", "5.0"))

def local_plane_predict(tree, PXY, PA, PW, qxy, own, own_npts):
    """qxy (n,2) の各点で自井戸(own)除外KNN→加重平面fit→ANCC予測。返り値(pred, med_dist)
    ★kは自井戸の点数分を上乗せして問い合わせる(自井戸が最近傍を独占するため)。"""
    n = len(qxy)
    kq = min(K + own_npts + 64, len(PXY))
    pred = np.full(n, np.nan); mdist = np.full(n, np.nan)
    slope = np.full(n, np.nan); fmad = np.full(n, np.nan)
    for c0 in range(0, n, 512):
        q = qxy[c0:c0 + 512]
        dd, ii = tree.query(q, k=kq, distance_upper_bound=RMAX)
        for bi in range(len(q)):
            d = dd[bi]; j = ii[bi]
            ok = np.isfinite(d) & (j < len(PXY))
            j = j[ok]; d = d[ok]
            keep = PW[j] != own
            j = j[keep]; d = d[keep]
            if len(j) < 12: continue
            # ★井戸多様性: 近い外部井戸 NBW 本から各 ≤PPW 点(1本の平行井戸独占=直線縮退を防ぐ)
            NBW, PPW = 4, 48
            sel = []
            seen = {}
            for idx in range(len(j)):
                wid = PW[j[idx]]
                c = seen.get(wid, 0)
                if c == 0 and len(seen) >= NBW: continue
                if c >= PPW: continue
                seen[wid] = c + 1; sel.append(idx)
                if len(sel) >= NBW * PPW: break
            j = j[sel]; d = d[sel]
            i = c0 + bi
            sig = max(SIG0, float(np.median(d)))
            w = np.exp(-0.5 * (d / sig) ** 2) + 1e-6
            dx = (PXY[j, 0] - qxy[i, 0]) * 1e-3; dy = (PXY[j, 1] - qxy[i, 1]) * 1e-3   # kft
            A = np.c_[np.ones(len(j)), dx, dy]; yv = PA[j]
            RIDGE = np.diag([0.0, LAMS, LAMS])                  # スロープのみ正則化(縮退→局所定数へ)
            for _ in range(2):
                W_ = w
                AtW = A.T * W_
                try:
                    coef = np.linalg.solve(AtW @ A + RIDGE, AtW @ yv)
                except np.linalg.LinAlgError:
                    coef = np.array([np.average(yv, weights=W_), 0, 0])
                r = yv - A @ coef
                mad = np.median(np.abs(r - np.median(r))) + 1e-6
                w = W_ * np.clip(1.0 - (r / (4.5 * mad)) ** 2, 0.05, None)
            pred[i] = coef[0]; mdist[i] = float(np.median(d))
            slope[i] = float(np.hypot(coef[1], coef[2]))        # ft/kft
            fmad[i] = float(np.median(np.abs(r - np.median(r))))
    return pred, mdist, slope, fmad

def main():
    t0 = time.time()
    wells = sorted(set(f.split("__")[0] for f in os.listdir(TRAIN) if f.endswith("__horizontal_well.csv")))
    eval_wells = wells[:NW] if NW else wells   # ★点群は常に全井戸、評価だけ NW で絞る(スモーク用)
    # 点群構築
    PX, PA_, PWID = [], [], []
    HW = {}
    for wi, w in enumerate(wells):
        hw = pd.read_csv(os.path.join(TRAIN, f"{w}__horizontal_well.csv"),
                         usecols=["MD", "X", "Y", "Z", "ANCC", "TVT", "TVT_input"])
        HW[w] = hw
        hh = hw.iloc[::STRIDE]
        m = hh[["X", "Y", "ANCC"]].notna().all(axis=1)
        PX.append(hh.loc[m, ["X", "Y"]].to_numpy(float))
        PA_.append(hh.loc[m, "ANCC"].to_numpy(float))
        PWID.append(np.full(int(m.sum()), wi, np.int32))
    PXY = np.concatenate(PX); PA = np.concatenate(PA_); PW = np.concatenate(PWID)
    tree = cKDTree(PXY)
    print(f"wells={len(wells)}  points={len(PXY)}  ({time.time()-t0:.0f}s)", flush=True)
    res_rows = {}; es_all = []; es_orc = []
    SAVEP = os.environ.get("SAVEP", "0") == "1"; DUMP = {}
    for wi, w in enumerate(wells):
        if w not in set(eval_wells): continue
        hw = HW[w]
        x = hw["X"].to_numpy(float); y = hw["Y"].to_numpy(float); z = hw["Z"].to_numpy(float)
        tvt = hw["TVT"].to_numpy(float); tin = hw["TVT_input"].to_numpy(float)
        anc = hw["ANCC"].to_numpy(float)
        n = len(hw)
        qi = np.arange(0, n, QSTRIDE)
        own_npts = int((PW == wi).sum())
        pred_q, mdist_q, slope_q, fmad_q = local_plane_predict(tree, PXY, PA, PW, np.c_[x[qi], y[qi]], wi, own_npts)
        # 全行へ線形補間(NaNは前後埋め)
        good = np.isfinite(pred_q)
        if good.sum() < 5:
            continue
        ancp = np.interp(np.arange(n), qi[good], pred_q[good])
        mdist = np.interp(np.arange(n), qi[good], mdist_q[good])
        slp = np.interp(np.arange(n), qi[good], np.nan_to_num(slope_q[good]))
        fmd = np.interp(np.arange(n), qi[good], np.nan_to_num(fmad_q[good]))
        kn = np.isfinite(tin); ev = ~kn
        if kn.sum() < 5 or ev.sum() < 10: continue
        # ★圏外フォールバック: 外部点が遠すぎる井戸は面を信じず prefix-hold(最後の既知TVT保持)
        FMAX = float(os.environ.get("FMAX", "1500"))
        if not np.isfinite(np.nanmedian(mdist)) or np.nanmedian(mdist) > FMAX:
            hold = tin[kn][-1]
            pred = np.full(n, hold)
            e = (pred - tvt)[ev & np.isfinite(tvt)]
            es_all.append(e); es_orc.append(e.copy())
            res_rows[w] = dict(rmse=float(np.sqrt(np.mean(e ** 2))), rmse_ob=None,
                               nd=float(np.nanmedian(mdist)) if np.isfinite(np.nanmedian(mdist)) else 1e9,
                               nev=int(len(e)), db=None, fallback=True)
            if SAVEP:
                DUMP[w] = np.stack([pred[ev], np.full(ev.sum(), 9e3), np.zeros(ev.sum()), np.zeros(ev.sum())], 1).astype(np.float32)
            continue
        bhat = np.median(tin[kn] + z[kn] - ancp[kn])
        btru = np.median(tvt[np.isfinite(tvt)] + z[np.isfinite(tvt)] - anc[np.isfinite(tvt)]) if np.isfinite(anc).sum() > 10 else np.nan
        pred = bhat + ancp - z
        if SAVEP:
            DUMP[w] = np.stack([pred[ev], mdist[ev], slp[ev], fmd[ev]], 1).astype(np.float32)
        e = (pred - tvt)[ev & np.isfinite(tvt)]
        es_all.append(e)
        if np.isfinite(btru):
            eo = (btru + ancp - z - tvt)[ev & np.isfinite(tvt)]
            es_orc.append(eo)
        res_rows[w] = dict(rmse=float(np.sqrt(np.mean(e ** 2))),
                           rmse_ob=float(np.sqrt(np.mean(eo ** 2))) if np.isfinite(btru) else None,
                           nd=float(np.nanmedian(mdist)), nev=int(len(e)),
                           db=float(bhat - btru) if np.isfinite(btru) else None)
        if (wi + 1) % 100 == 0:
            ee = np.concatenate(es_all)
            print(f"  {wi+1}/{len(wells)}  途中pooled={np.sqrt(np.mean(ee**2)):.3f} ({time.time()-t0:.0f}s)", flush=True)
    ee = np.concatenate(es_all); eo = np.concatenate(es_orc)
    total = float(np.sqrt(np.mean(ee ** 2))); total_ob = float(np.sqrt(np.mean(eo ** 2)))
    W = sorted(res_rows)
    per_fold = {}
    wl = [w for w in wells if w in res_rows]
    for f in range(5):
        val = set(sorted(wells)[f::5])
        idx = [i for i, w in enumerate(wl) if w in val]
        if idx:
            ef = np.concatenate([es_all[i] for i in idx])
            per_fold[f] = float(np.sqrt(np.mean(ef ** 2)))
    print(f"\n=== v56 LOWO 面補間エンジン: overall = {total:.4f}  (oracle-b {total_ob:.4f}) ===")
    print("fold別:", {k: round(v, 3) for k, v in per_fold.items()})
    print("  参照: struct(v38式)~9.7 / 純PF 8.1-8.6 / v52 GBM 6.28")
    rs = np.array([res_rows[w]["rmse"] for w in W]); nd = np.array([res_rows[w]["nd"] for w in W])
    for tag, m in (("近接<300ft", nd < 300), ("300-800", (nd >= 300) & (nd < 800)), ("800-2000", (nd >= 800) & (nd < 2000)), (">2000ft", nd >= 2000)):
        if m.sum():
            print(f"  {tag:11s}: n={m.sum():3d}  井戸RMSE med={np.median(rs[m]):6.2f}  P90={np.percentile(rs[m],90):6.2f}")
    json.dump(dict(total=total, total_oracle_b=total_ob, per_fold=per_fold, n_wells=len(W)),
              open(os.path.join(ART, "lowo_result.json"), "w"), indent=1)
    if SAVEP:
        np.savez_compressed(os.path.join(ART, "lowo_rows.npz"), **DUMP)
        print(f"saved lowo_rows.npz ({len(DUMP)} wells, cols=[tvt_surf,dist,slope,fmad], eval行のみCSV順)")
    np.savez_compressed(os.path.join(ART, "lowo_per_well.npz"),
                        wells=np.array(W), rmse=rs, nd=nd,
                        db=np.array([res_rows[w]["db"] if res_rows[w]["db"] is not None else np.nan for w in W]))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "Meiryo"
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    ax[0].scatter(nd, rs, s=12, alpha=0.6)
    ax[0].set_xscale("log"); ax[0].set_xlabel("近傍点までの距離 med (ft)"); ax[0].set_ylabel("井戸RMSE (ft)")
    ax[0].set_title(f"距離 vs 誤差 (overall {total:.2f})")
    xs = np.sort(rs); ax[1].plot(xs, np.arange(1, len(xs) + 1) / len(xs))
    ax[1].set_xlabel("井戸RMSE (ft)"); ax[1].set_ylabel("ECDF"); ax[1].set_xlim(0, 25); ax[1].set_title("井戸RMSE ECDF")
    db = np.array([res_rows[w]["db"] for w in W if res_rows[w]["db"] is not None])
    ax[2].hist(db, bins=40, range=(-10, 10)); ax[2].set_xlabel("b̂ − b_true (ft)"); ax[2].set_title("prefix由来 b 推定誤差")
    plt.tight_layout(); plt.savefig(os.path.join(ART, "lowo_graphs.png"), dpi=110)
    print(f"saved artifacts_v56/lowo_graphs.png ({time.time()-t0:.0f}s)")

if __name__ == "__main__":
    main()
