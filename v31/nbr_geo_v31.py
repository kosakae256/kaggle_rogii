# -*- coding: utf-8 -*-
"""
v31-B: 学習なし・GR解析なし・PFなしの「近傍幾何転写」推定
==========================================================
原理: TVT+Z = 地層面(井戸ごと定数オフセットを除く, error_anatomy §1)。
  1) 全train井の点(X,Y)→geo_z=TVT+Z を KD-tree 化(対象井は除外=リーク無し)
  2) 対象井の各行(x,y)で半径R内k近傍の geo_z を逆距離加重平均(地層面のローカル転写)
  3) オフセット b は自分の既知帯で較正: b = median(真geo_z − 転写geo_z)(末尾CAL_N行)
  4) TVT_pred = 転写geo_z + b − Z。近傍なし行はヒールdip直線にフォールバック
※ cross-field kriging(死亡済)との違い = 半径制限つき最近傍のみ・面のフィッティングをしない・自己較正。
評価: overall / wells>10 / 近傍カバレッジ別。R を3水準だけ見る(チューニングはしない)。
出力: artifacts_v31/nbr_geo_preds.npz(最良Rの全行予測=後段の特徴/blend素材)
実行: rogii_claude/.venv/Scripts/python.exe v31/nbr_geo_v31.py
"""
import glob, json, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

warnings.filterwarnings("ignore")

PROJECT  = Path(r"C:\Users\kosaka256\Documents\rogii_claude")
DATA_DIR = PROJECT / "rogii-wellbore-geology-prediction" / "train"
ART = PROJECT / "v31" / "artifacts_v31"; ART.mkdir(parents=True, exist_ok=True)

RADII = [300.0, 800.0, 1500.0]   # 転写半径の3水準(粗く見るだけ)
K_NN  = 16                        # 近傍点数上限
CAL_N = 200                       # 較正に使う既知帯の末尾行数
DIP_N = 30                        # フォールバックdipの回帰行数


def load_wells():
    wells = {}
    for p in sorted(glob.glob(str(DATA_DIR / "*__horizontal_well.csv"))):
        w = Path(p).stem.replace("__horizontal_well", "")
        hw = pd.read_csv(p, usecols=["MD", "X", "Y", "Z", "GR", "TVT", "TVT_input"])
        kn = hw[hw["TVT_input"].notna()]; ev = hw[hw["TVT_input"].isna()]
        if len(ev) == 0 or len(kn) < 10 or hw["TVT"].isna().all():
            continue
        wells[w] = dict(
            kn_xy=kn[["X", "Y"]].to_numpy(float), kn_z=kn["Z"].to_numpy(float),
            kn_md=kn["MD"].to_numpy(float), kn_tvt=kn["TVT_input"].to_numpy(float),
            ev_xy=ev[["X", "Y"]].to_numpy(float), ev_z=ev["Z"].to_numpy(float),
            ev_md=ev["MD"].to_numpy(float), ev_tvt=ev["TVT"].to_numpy(float),
            all_xy=hw[["X", "Y"]].to_numpy(float),
            all_geo=(hw["TVT"].to_numpy(float) + hw["Z"].to_numpy(float)),
        )
    return wells


DIP_TAU = 500.0   # dip外挿の減衰スケール[ft](線形外挿は暴走するため drift を ir*TAU で有界化)

def heel_dip_line(d):
    """フォールバック: ヒールdip の減衰付き外挿(drift = ir*TAU*(1-exp(-dMD/TAU)))。"""
    t = d["kn_tvt"][-DIP_N:]; z = d["kn_z"][-DIP_N:]; m = d["kn_md"][-DIP_N:]
    dt = np.diff(t + z); dm = np.diff(m); ok = dm > 0
    ir = float(np.median(dt[ok] / dm[ok])) if ok.sum() >= 3 else 0.0
    geo0 = float(d["kn_tvt"][-1] + d["kn_z"][-1]); md0 = float(d["kn_md"][-1])
    drift = ir * DIP_TAU * (1.0 - np.exp(-np.maximum(d["ev_md"] - md0, 0.0) / DIP_TAU))
    return geo0 + drift - d["ev_z"]     # TVT予測


def main():
    t0 = time.time()
    wells = load_wells()
    wl = sorted(wells)
    print(f"wells: {len(wl)} ({time.time()-t0:.0f}s)")

    # 井戸ごとの点群/バウンディングボックス(自坑井除外は候補井選定で行う)
    PW = {}; BOX = {}
    for w in wl:
        d = wells[w]
        ok = np.isfinite(d["all_geo"])
        PW[w] = (d["all_xy"][ok], d["all_geo"][ok])
        xy = PW[w][0]
        BOX[w] = (xy[:, 0].min(), xy[:, 0].max(), xy[:, 1].min(), xy[:, 1].max())
    print(f"pool: {sum(len(v[0]) for v in PW.values())} pts / {len(PW)} wells")

    def transfer(xy, self_w, R):
        """各点の 転写geo_z(自坑井除外・bbox候補井の点だけで小tree) と 最近傍距離。"""
        est = np.full(len(xy), np.nan); nnd = np.full(len(xy), np.inf)
        x0, x1 = xy[:, 0].min() - R, xy[:, 0].max() + R
        y0, y1 = xy[:, 1].min() - R, xy[:, 1].max() + R
        cands = [w2 for w2 in wl if w2 != self_w
                 and BOX[w2][0] <= x1 and BOX[w2][1] >= x0
                 and BOX[w2][2] <= y1 and BOX[w2][3] >= y0]
        if not cands: return est, nnd
        P = np.concatenate([PW[w2][0] for w2 in cands])
        GEO = np.concatenate([PW[w2][1] for w2 in cands])
        tree = cKDTree(P)
        dist, idx = tree.query(xy, k=min(K_NN, len(P)), distance_upper_bound=R)
        if dist.ndim == 1: dist = dist[:, None]; idx = idx[:, None]
        for j in range(len(xy)):
            ok = np.isfinite(dist[j])
            if not ok.any(): continue
            dj = dist[j][ok]; ij = idx[j][ok]
            wgt = 1.0 / np.maximum(dj, 1.0)
            est[j] = float(np.sum(wgt * GEO[ij]) / np.sum(wgt))
            nnd[j] = float(dj.min())
        return est, nnd

    results = {}
    best = None
    for R in RADII:
        tR = time.time()
        errs = []; per = {}; preds = {}; covs = []
        for w in wl:
            d = wells[w]
            fb = heel_dip_line(d)                      # フォールバック(dip直線)
            est_ev, nnd_ev = transfer(d["ev_xy"], w, R)
            est_kn, _ = transfer(d["kn_xy"][-CAL_N:], w, R)
            geo_kn = d["kn_tvt"][-CAL_N:] + d["kn_z"][-CAL_N:]
            ok_cal = np.isfinite(est_kn)
            pred = fb.copy()
            if ok_cal.sum() >= 10:
                b = float(np.median(geo_kn[ok_cal] - est_kn[ok_cal]))   # 自己較正オフセット
                ok_ev = np.isfinite(est_ev)
                pred[ok_ev] = est_ev[ok_ev] + b - d["ev_z"][ok_ev]
                cov = float(ok_ev.mean())
            else:
                cov = 0.0
            covs.append(cov); preds[w] = pred
            e = pred - d["ev_tvt"]; e = e[np.isfinite(e)]
            errs.append(e); per[w] = float(np.sqrt(np.mean(e ** 2))) if e.size else np.nan
        e = np.concatenate(errs); overall = float(np.sqrt(np.mean(e ** 2)))
        n10 = sum(1 for v in per.values() if np.isfinite(v) and v > 10)
        covs = np.array(covs)
        full_cov = [w for w, c in zip(wl, covs) if c > 0.9]
        e_fc = np.concatenate([preds[w] - wells[w]["ev_tvt"] for w in full_cov]) if full_cov else np.array([])
        e_fc = e_fc[np.isfinite(e_fc)]
        rmse_fc = float(np.sqrt(np.mean(e_fc ** 2))) if e_fc.size else float("nan")
        results[R] = dict(overall=overall, over10=n10, cov_full=len(full_cov),
                          rmse_covered=rmse_fc, preds=preds, per=per)
        print(f"[R={R:.0f}] overall={overall:.4f} >10={n10} | カバー>90%: {len(full_cov)}井戸"
              f" rmse={rmse_fc:.4f} | ({time.time()-tR:.0f}s)")
        if best is None or overall < results[best]["overall"]: best = R

    # フォールバック単体(dip直線)の床も表示
    e_fb = []
    for w in wl:
        e = heel_dip_line(wells[w]) - wells[w]["ev_tvt"]; e_fb.append(e[np.isfinite(e)])
    e_fb = np.concatenate(e_fb)
    print(f"\n(参考) dip直線のみ: overall={np.sqrt(np.mean(e_fb**2)):.4f} / carry-last≈15.9")

    r = results[best]
    np.savez_compressed(ART / "nbr_geo_preds.npz",
                        **{f"p_{w}": r["preds"][w] for w in wl})
    (ART / "nbr_geo_summary.json").write_text(json.dumps(
        {str(R): {k: v for k, v in res.items() if k in ("overall", "over10", "cov_full", "rmse_covered")}
         for R, res in results.items()} | {"best_R": best}, indent=1))
    print(f"\nbest R={best} → nbr_geo_preds.npz 保存。overall={r['overall']:.4f}"
          f" / カバー井戸のみ {r['rmse_covered']:.4f}")
    print("読み方: カバー井戸のみの rmse が本命(全体は非カバー井戸のdip直線で薄まる)。")


if __name__ == "__main__":
    main()
