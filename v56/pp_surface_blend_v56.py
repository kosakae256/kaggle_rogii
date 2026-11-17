# v56 後処理ブレンド探索 — GBM(pp後)に面予測を距離ゲートで少量ブレンド(v27距離連動αの面版)
#   pred' = pred_pp + w(dist)·(tvt_surf − pred_pp)
#   形: A) 2段ステップ w=W0(d<300)/W1(300-800)/0  B) 指数 w=W0·exp(−d/τd) (d≥2000は0)
# OOF(=valid)でチューニング(2026-07-12 ユーザー方針)。粗いグリッド限定(overfit_policy)。
import os, json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.model_selection import GroupKFold

P0 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART52 = os.path.join(P0, "v52", "artifacts_v52")
ART56 = os.path.join(P0, "v56", "artifacts_v56")
PQ = os.path.join(ART52, "data", "train.parquet")

def main():
    tb = pq.read_table(PQ, columns=["target", "well", "last_known_tvt", "md_since", "pf_ancc"])
    y = tb.column("target").to_numpy(zero_copy_only=False).astype(np.float64)
    g = tb.column("well").to_pandas().to_numpy()
    lk = tb.column("last_known_tvt").to_numpy(zero_copy_only=False).astype(np.float64)
    mds = tb.column("md_since").to_numpy(zero_copy_only=False).astype(np.float64)
    pfa = tb.column("pf_ancc").to_numpy(zero_copy_only=False).astype(np.float64)
    del tb
    # ★target/oof/pf_res は残差スケール(TVT − last_known)。ppも残差のまま(cell4 準拠)
    oof = np.load(os.path.join(ART52, "gbm_v41_oof.npy")).astype(np.float64)
    pp = json.loads(open(os.path.join(ART52, "pp_params.json")).read())
    pf_res = pfa - lk
    pred = (oof * (1 - pp["w_pf"]) + pf_res * pp["w_pf"]) * (1 - np.exp(-np.maximum(mds, 0.0) / pp["tau"])) * pp["alpha"]
    fin = np.isfinite(y) & np.isfinite(pred)
    def overall(p): return float(np.sqrt(np.mean((p[fin] - y[fin]) ** 2)))
    base = overall(pred)
    print(f"baseline GBM pp = {base:.4f}  (pp={pp})", flush=True)
    F = np.load(os.path.join(ART56, "v56_feature.npy")).astype(np.float64)
    tsurf = F[:, 0]; dist = F[:, 1]          # surf_delta = tvt_surf − lk(残差スケール一致)
    ok_s = np.isfinite(tsurf)
    folds = list(GroupKFold(5).split(np.zeros(len(y)), y, groups=g))
    def fold_rmses(p):
        return [round(float(np.sqrt(np.mean((p[va][fin[va]] - y[va][fin[va]]) ** 2))), 3) for _, va in folds]
    def apply_w(w):
        p2 = pred.copy()
        m = ok_s & (w > 0)
        p2[m] = pred[m] + w[m] * (tsurf[m] - pred[m])
        return p2
    results = {}
    print("--- A) 2段ステップ ---", flush=True)
    for W0 in (0.1, 0.2, 0.3, 0.5):
        for W1 in (0.0, 0.05, 0.1, 0.2):
            w = np.where(dist < 300, W0, np.where(dist < 800, W1, 0.0))
            r = overall(apply_w(w)); results[f"step_{W0}_{W1}"] = r
            print(f"  W0={W0} W1={W1}: {r:.4f} ({r-base:+.4f})", flush=True)
    print("--- B) 指数 w=W0*exp(-d/tau_d) ---", flush=True)
    for W0 in (0.1, 0.2, 0.3, 0.5):
        for td in (200.0, 400.0, 600.0):
            w = W0 * np.exp(-dist / td); w[dist >= 2000] = 0.0
            r = overall(apply_w(w)); results[f"exp_{W0}_{td:.0f}"] = r
            print(f"  W0={W0} τd={td:.0f}: {r:.4f} ({r-base:+.4f})", flush=True)
    best = min(results, key=results.get)
    print(f"\nbest={best} {results[best]:.4f} (Δ{results[best]-base:+.4f})")
    # bestのfold別
    if best.startswith("step"):
        _, W0, W1 = best.split("_"); w = np.where(dist < 300, float(W0), np.where(dist < 800, float(W1), 0.0))
    else:
        _, W0, td = best.split("_"); w = float(W0) * np.exp(-dist / float(td)); w[dist >= 2000] = 0.0
    pb = apply_w(w)
    print("fold別 base :", fold_rmses(pred))
    print("fold別 blend:", fold_rmses(pb))
    json.dump(dict(base=base, results=results, best=best), open(os.path.join(ART56, "pp_blend_result.json"), "w"), indent=1)

if __name__ == "__main__":
    main()
