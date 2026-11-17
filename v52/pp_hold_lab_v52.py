# hold 実験 — pp の lk 錨(残差縮小)の「形×長さ×α」を OOF で掃引
#   pred_res' = pred_res_mix · (1 − w_hold(mds)) · α,  w_hold(0)=1 → pred=lk
#   形: exp: exp(-m/τ) / sexp: exp(-(m/τ)^β) (β<1=長裾) / rat: 1/(1+m/τ) (最長裾) / gauss: exp(-(m/τ)²)
#   ベース: v52 GBM OOF(gbm_v41_oof.npy, 残差スケール) + w_pf=0.05 混合(固定)
import os, json
import numpy as np
import pyarrow.parquet as pq
from sklearn.model_selection import GroupKFold

P0 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART52 = os.path.join(P0, "v52", "artifacts_v52")
PQ = os.path.join(ART52, "data", "train.parquet")

def main():
    tb = pq.read_table(PQ, columns=["target", "well", "md_since", "pf_ancc", "last_known_tvt"])
    y = tb.column("target").to_numpy(zero_copy_only=False).astype(np.float64)
    g = tb.column("well").to_pandas().to_numpy()
    mds = np.maximum(tb.column("md_since").to_numpy(zero_copy_only=False).astype(np.float64), 0.0)
    pfa = tb.column("pf_ancc").to_numpy(zero_copy_only=False).astype(np.float64)
    lk = tb.column("last_known_tvt").to_numpy(zero_copy_only=False).astype(np.float64)
    del tb
    oof = np.load(os.path.join(ART52, "gbm_v41_oof.npy")).astype(np.float64)
    W_PF = 0.05
    mix = oof * (1 - W_PF) + (pfa - lk) * W_PF
    fin = np.isfinite(y) & np.isfinite(mix)
    def overall(p): return float(np.sqrt(np.mean((p[fin] - y[fin]) ** 2)))
    folds = list(GroupKFold(5).split(np.zeros(len(y)), y, groups=g))
    def foldr(p):
        return [round(float(np.sqrt(np.mean((p[va][fin[va]] - y[va][fin[va]]) ** 2))), 3) for _, va in folds]
    def hold(shape, tau, beta=1.0):
        m = mds / tau
        if shape == "exp":   return np.exp(-m)
        if shape == "sexp":  return np.exp(-np.power(m, beta))
        if shape == "rat":   return 1.0 / (1.0 + m)
        if shape == "gauss": return np.exp(-m * m)
        raise ValueError(shape)
    ramp40 = 1 - np.exp(-mds / 40.0)
    base = overall(mix * ramp40 * 1.03)
    print(f"baseline(現行 exp τ40 α1.03) = {base:.4f}", flush=True)
    # 井戸長(最大mds)
    import pandas as pd
    wl = pd.Series(mds).groupby(pd.Series(g)).transform("max").to_numpy()
    results = {}; PRED = {}
    # A) 全域一律縮小(=α走査。α<1がholdの向き)
    for alpha in (0.85, 0.90, 0.94, 0.97, 1.00, 1.03, 1.06):
        k = f"A_uniform_a{alpha}"
        PRED[k] = mix * ramp40 * alpha; results[k] = overall(PRED[k])
    # B) 逆ランプ: 遠いほど強くlkへ(現行チェーンの上に)
    for lam in (0.03, 0.06, 0.10, 0.15, 0.25):
        for tl in (300.0, 800.0, 2000.0):
            k = f"B_far_l{lam}_t{tl:.0f}"
            PRED[k] = mix * ramp40 * 1.03 * (1 - lam * (1 - np.exp(-mds / tl)))
            results[k] = overall(PRED[k])
    # C) 井戸長比例の一律縮小(長い井戸ほど強く)
    for lam0 in (0.05, 0.10, 0.20):
        for L0 in (2000.0, 4000.0, 8000.0):
            lam_w = lam0 * np.minimum(1.0, wl / L0)
            k = f"C_len_l{lam0}_L{L0:.0f}"
            PRED[k] = mix * ramp40 * 1.03 * (1 - lam_w)
            results[k] = overall(PRED[k])
    # D) 参考: 近傍ランプの長さ違い(旧仮説)
    for t in (85, 300):
        k = f"D_ramp_t{t}"
        PRED[k] = mix * (1 - np.exp(-mds / t)) * 1.03; results[k] = overall(PRED[k])
    top = sorted(results.items(), key=lambda t: t[1])[:14]
    for k, v in top: print(f"  {k:24s} {v:.4f} ({v-base:+.4f})")
    bk = top[0][0]; pb = PRED[bk]
    print(f"\nbest={bk}: {results[bk]:.4f} (Δ{results[bk]-base:+.4f})")
    print("fold別 base:", foldr(mix * (1 - np.exp(-mds / 40.0)) * 1.03))
    print("fold別 best:", foldr(pb))
    # mds帯別の寄与(baseとの差)
    eb = (mix * (1 - np.exp(-mds / 40.0)) * 1.03 - y); e2 = (pb - y)
    for lo, hi in ((0, 100), (100, 300), (300, 800), (800, 2000), (2000, 99999)):
        m = fin & (mds >= lo) & (mds < hi)
        if m.sum():
            print(f"  mds {lo:>5}-{hi:<5}: base {np.sqrt(np.mean(eb[m]**2)):7.3f} → best {np.sqrt(np.mean(e2[m]**2)):7.3f}  (n={m.sum()})")
    json.dump(dict(base=base, results=results, best=bk), open(os.path.join(ART52, "pp_hold_lab.json"), "w"), indent=1)

if __name__ == "__main__":
    main()
