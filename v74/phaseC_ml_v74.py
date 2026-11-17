# v74 Phase C-ML(v8) — 集約器の学習化: セグメント特徴 → 真のrateをLGBM回帰(OOF) → 積分
#   v7までの手動ノブ(ゲートwg/ga・信頼度縮小・prefixフォールバック)を全部モデルに置換。
#   特徴は全てGRフリー&test合法: 場のrate推定(r_field)・qdir・mad・距離・n_eff・
#   操舵rate(r_flat=dZ/dh)・prefix dip・md_since・方位・incl・操舵κ。
#   target: r_true = d(lowpass(g,201))/dh(trainの全行で利用可)。GroupKFold(5) by well。
# 評価: OOF rateを補間→クリップ→t0から積分→評価帯 overall pooled RMSE。
# env: NW(0=全), SEG(25), K(128), RMAX(3000), SIG0(300), SIGCAP(800)
import os, json, time, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import phaseC_field2_v74 as F

P0 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART = os.path.join(P0, "v74", "artifacts_v74"); os.makedirs(ART, exist_ok=True)
SEG = int(os.environ.get("SEG", "25"))
K = int(os.environ.get("K", "128"))
RMAX = float(os.environ.get("RMAX", "3000"))
SIG0 = float(os.environ.get("SIG0", "300"))
SIGCAP = float(os.environ.get("SIGCAP", "800"))

FEATS = ["r_field", "qdir", "mad", "dmed", "n_eff", "r_flat", "r_pref", "md_since",
         "az_sin", "az_cos", "incl", "k_pref", "dh", "df_flat", "df_pref"]

def collect(D, ws, donors, pair):
    from scipy.spatial import cKDTree
    tp = cKDTree(donors["pos"])
    n_don = len(donors["pos"])
    rows, tgt, grp = [], [], []
    seg_index = {}          # well → (centers, row_slice into rows)
    t0all = time.time()
    for wi, w in enumerate(ws):
        d = D[w]
        n = len(d["md"]); t0 = d["t0"]
        own_np = int((donors["well"] == wi).sum())
        kq = min(K + own_np + 64, n_don)
        g_true_s = F.lowpass((d["tvt"] + d["z"]), 201)
        r_true = np.gradient(g_true_s) / np.maximum(d["dh"], 1e-6)
        r_flat_all = np.gradient(d["zs"]) / np.maximum(d["dh"], 1e-6)
        ki = np.where(d["kn"])[0]
        tl = ki[(ki >= t0 - 1500) & (d["dh"][ki] > 0.5)]
        r_pref = float(np.clip(np.median(r_true[tl]), -0.08, 0.08)) if len(tl) > 100 else 0.0
        # 操舵κ(prefixでのHP相関, phaseA/Bと同じ)
        zh301 = d["z"] - F.lowpass(d["z"], 301)
        tl2 = ki[-min(len(ki), 1000):]
        gt = ((d["tvt"] + d["z"]) - F.lowpass(d["tvt"] + d["z"], 301))[tl2]
        zt = zh301[tl2]
        vz = float(np.dot(zt, zt))
        k_pref = float(np.clip(np.dot(zt, gt) / vz, 0, 1.5)) if vz > 1e-9 else 0.0
        az_all = np.arctan2(d["dyr"], d["dxr"])
        incl = np.arctan2(np.gradient(F.lowpass(d["z"], 301)),
                          np.maximum(d["dh"], 1e-9))
        centers = np.arange(0, n, SEG)
        i0 = len(rows)
        for c0 in centers:
            p = np.array([d["x"][c0], d["y"][c0]])
            dd, jj = tp.query(p, k=kq, distance_upper_bound=RMAX)
            ok = np.isfinite(dd) & (jj < n_don)
            jj = jj[ok]; dd = dd[ok]
            keep = donors["well"][jj] != wi
            jj = jj[keep][:K]; dd = dd[keep][:K]
            if len(jj) < 8:
                r_field, madv, qdir, dmed, n_eff = 0.0, 1.0, 0.0, 9e3, 0.0
            else:
                cs, azs, wts = donors["c"][jj], donors["az"][jj], donors["w"][jj]
                sig = min(max(SIG0, np.median(dd)), SIGCAP)
                wts = wts * np.exp(-0.5 * (dd / sig) ** 2)
                ux, uy = d["dxr"][c0], d["dyr"][c0]
                hh = max(np.hypot(ux, uy), 1e-6)
                u = np.array([ux, uy]) / hh
                (gx, gy), madv, qdir = F.solve_grad(cs, azs, wts, F.RIDGE, u)
                r_field = float(np.clip(gx * u[0] + gy * u[1], -0.12, 0.12))
                dmed = float(np.median(dd)); n_eff = float(wts.sum())
            rf = float(np.clip(r_flat_all[c0], -0.15, 0.15))
            rows.append([r_field, qdir, madv, dmed, n_eff, rf, r_pref,
                         float(c0 - t0), np.sin(az_all[c0]), np.cos(az_all[c0]),
                         float(incl[c0]), k_pref, float(d["dh"][c0]),
                         r_field - rf, r_field - r_pref])
            tgt.append(float(np.clip(r_true[c0], -0.2, 0.2)))
            grp.append(wi)
        seg_index[w] = (centers, (i0, len(rows)))
        if (wi + 1) % 100 == 0:
            print(f"  collect {wi+1}/{len(ws)}  ({time.time()-t0all:.0f}s)", flush=True)
    return (np.array(rows, np.float32), np.array(tgt, np.float32),
            np.array(grp, np.int32), seg_index)

def main():
    t0 = time.time()
    import lightgbm as lgb
    from sklearn.model_selection import GroupKFold
    D = F.load()
    ws, donors, pair = F.build_donors(D)
    print(f"wells={len(ws)}  donors={len(donors['pos'])}  ({time.time()-t0:.0f}s)", flush=True)
    X, yv, grp, seg_index = collect(D, ws, donors, pair)
    print(f"segments={len(yv)}  feats={len(FEATS)}  ({time.time()-t0:.0f}s)", flush=True)
    oof = np.full(len(yv), np.nan, np.float32)
    params = dict(objective="l2", learning_rate=0.05, num_leaves=63,
                  min_data_in_leaf=100, feature_fraction=0.85, bagging_fraction=0.8,
                  bagging_freq=1, lambda_l2=1.0, verbosity=-1, num_threads=os.cpu_count())
    imp = np.zeros(len(FEATS))
    for fi, (tr, va) in enumerate(GroupKFold(5).split(X, yv, grp)):
        m = lgb.train(params, lgb.Dataset(X[tr], yv[tr]), num_boost_round=2000,
                      valid_sets=[lgb.Dataset(X[va], yv[va])],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[va] = m.predict(X[va], num_iteration=m.best_iteration).astype(np.float32)
        imp += m.feature_importance("gain")
        print(f"fold{fi}: it={m.best_iteration} rate_rmse={np.sqrt(np.mean((oof[va]-yv[va])**2)):.5f}",
              flush=True)
    # 積分評価
    ES, perw = [], {}
    for wi, w in enumerate(ws):
        d = D[w]
        n = len(d["md"]); t0w = d["t0"]
        centers, (i0, i1) = seg_index[w]
        r_seg = np.clip(oof[i0:i1], -0.08, 0.08)
        rates = np.interp(np.arange(n), centers, r_seg)
        step = rates * d["dh"]
        g0 = float(d["tin"][t0w] + d["z"][t0w])
        ei_all = np.arange(t0w + 1, n)
        g_hat = g0 + np.cumsum(np.r_[0.0, step[t0w + 1:]])[1:]
        pred = g_hat - d["z"][ei_all]
        m = d["ev"][ei_all]
        e = (pred - d["tvt"][ei_all])[m]
        e = e[np.isfinite(e)]
        if len(e):
            ES.append(e); perw[w] = float(np.sqrt(np.mean(e ** 2)))
    es = np.concatenate(ES)
    overall = float(np.sqrt(np.mean(es ** 2)))
    dm = float(np.sqrt(np.mean(np.concatenate([e - np.median(e) for e in ES]) ** 2)))
    print(f"\n[v8 学習集約器] overall = {overall:.4f}  (井戸med {np.median(list(perw.values())):.3f}, "
          f"demean {dm:.4f})  [v7best 11.56 / 既存struct 9.70]")
    order = np.argsort(imp)[::-1]
    for i in order[:10]:
        print(f"    {FEATS[i]:10s} {imp[i]/imp.sum()*100:5.1f}%")
    with open(os.path.join(ART, "phaseC_ml_result.json"), "w", encoding="utf-8") as f:
        json.dump(dict(overall=overall, demean=dm,
                       per_well_median=float(np.median(list(perw.values()))),
                       importance={FEATS[i]: float(imp[i] / imp.sum()) for i in order},
                       per_well=perw), f, ensure_ascii=False, indent=1)

if __name__ == "__main__":
    main()
