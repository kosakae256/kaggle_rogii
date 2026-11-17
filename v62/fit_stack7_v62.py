# L1-L7 スタック: ローカルで重み確定(粗座標降下)+ oracle score(井戸別/行別)算出
# 出力: artifacts_v62/stack7_config.json(重み・pp・列順など提出側が読む全て)
import os, json, pickle
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import sys
sys.stdout = open(1, "w", encoding="utf-8", closefd=False)

ART = "v62/artifacts_v62"; ART52 = "v52/artifacts_v52"
LEGS = ["GBM", "TCN", "L3img", "L4", "L5", "L6", "L7"]

def main():
    tb = pq.read_table(f"{ART52}/data/train.parquet",
                       columns=["id", "well", "target", "md_since", "pf_ancc", "last_known_tvt"])
    ids = tb.column("id").to_pandas().astype(str)
    gw = tb.column("well").to_pandas().astype(str).to_numpy()
    y = tb.column("target").to_numpy(zero_copy_only=False).astype(np.float64)
    mds = np.maximum(tb.column("md_since").to_numpy(zero_copy_only=False).astype(np.float64), 0)
    pfa = tb.column("pf_ancc").to_numpy(zero_copy_only=False).astype(np.float64)
    lk = tb.column("last_known_tvt").to_numpy(zero_copy_only=False).astype(np.float64)
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
    X = {n: {} for n in LEGS}
    for w in WELLS:
        ii = np.where(gw == w)[0]; o = np.argsort(rn[ii], kind="stable"); iio = ii[o]
        n0 = min([len(E[n][w]) for n in E] + [len(iio)])
        ms = np.ones(n0, bool)
        for n in E: ms &= np.isfinite(E[n][w][:n0])
        ms &= np.isfinite(eg[iio[:n0]])
        X["GBM"][w] = eg[iio[:n0]][ms]; X["TCN"][w] = et[iio[:n0]][ms]
        for n in E: X[n][w] = E[n][w][:n0][ms]
    def pooled(errs):
        ee = np.concatenate(errs); return float(np.sqrt(np.mean(ee ** 2)))
    print("単体 overall:", {n: round(pooled([X[n][w] for w in WELLS]), 4) for n in LEGS})
    # ---- 重みフィット(粗座標降下, 0.05刻み) ----
    wt = {n: 0.0 for n in LEGS}; wt["TCN"] = 0.5; wt["L5"] = 1.0
    def sc(wd, ws=WELLS):
        s = sum(wd.values())
        return pooled([sum(wd[n] * X[n][w] for n in LEGS) / s for w in ws])
    cur = sc(wt)
    for _ in range(4):
        for n in LEGS:
            bv, bw = cur, wt[n]
            for v in np.arange(0, 1.01, 0.05):
                w2 = dict(wt); w2[n] = float(v)
                if sum(w2.values()) < 1e-9: continue
                r = sc(w2)
                if r < bv - 1e-6: bv, bw = r, float(v)
            wt[n] = bw; cur = bv
    print(f"\nスタック重み(全OOF, 0.05刻み): {json.dumps({k: round(v,2) for k,v in wt.items()})}")
    print(f"スタック overall = {cur:.4f}")
    for f in range(5):
        ws = WELLS[f::5]
        print(f"  fold{f}: {sc(wt, ws):.4f}")
    # ---- oracle scores ----
    orc_well = []
    orc_row = []
    for w in WELLS:
        errs = np.stack([X[n][w] for n in LEGS])          # (7, n)
        rmses = np.sqrt(np.mean(errs ** 2, 1))
        orc_well.append(errs[int(np.argmin(rmses))])
        orc_row.append(errs[np.argmin(np.abs(errs), 0), np.arange(errs.shape[1])])
    print(f"\noracle(井戸別に最良脚を選択) = {pooled(orc_well):.4f}")
    print(f"oracle(行別に最良脚を選択)   = {pooled(orc_row):.4f}  (参考: 過度に楽観的な下限)")
    # 井戸別: どの脚が最良になるかの分布
    cnt = {}
    for w in WELLS:
        errs = np.stack([X[n][w] for n in LEGS])
        best = LEGS[int(np.argmin(np.sqrt(np.mean(errs ** 2, 1))))]
        cnt[best] = cnt.get(best, 0) + 1
    print("井戸別最良脚の分布:", cnt)
    json.dump(dict(weights=wt, stack_overall=cur, pp=pp,
                   oracle_well=pooled(orc_well), oracle_row=pooled(orc_row)),
              open(f"{ART}/stack7_config.json", "w"), indent=1)
    print("saved stack7_config.json")

if __name__ == "__main__":
    main()
