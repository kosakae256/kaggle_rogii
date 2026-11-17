# -*- coding: utf-8 -*-
"""v30 診断: 閉ループ出力は prior のコピー(エコー)か、別物の信号か。
各 preds_sig*.npz について 誤差相関ρ / RMS(cl−prior) / 補正場の井戸別統計 を出す。
実行: rogii_claude/.venv/Scripts/python.exe v30/diag_echo_v30.py
"""
import glob, json
from pathlib import Path
import numpy as np

import closed_loop_v30 as cl

def main():
    wells = cl.list_wells()
    cache = {}
    for wid in wells:
        c = cl.build_cache(wid)
        if c is not None and c["tvt_true"] is not None:
            cache[wid] = c
    priors, meta = cl.build_priors(cache)
    wl = [w for w in wells if w in priors]
    tt = {w: cache[w]["tvt_true"] for w in wl}
    ep = np.concatenate([priors[w] - tt[w] for w in wl])          # prior 誤差
    fin = np.isfinite(ep)

    print(f"\n{'sig':>6s} {'rmse_cl':>8s} {'rho(err)':>9s} {'RMS(cl-prior)':>14s} "
          f"{'med|corr|/well':>15s} {'p90|corr|/well':>15s}")
    for fp in sorted(glob.glob(str(cl.OUT_DIR / "preds_sig*.npz"))):
        key = Path(fp).stem.replace("preds_sig", "")
        z = np.load(fp, allow_pickle=True)
        pred = {w: z[f"s_{w}"] for w in wl if f"s_{w}" in z}
        if len(pred) != len(wl):
            print(f"{key:>6s}  (wells不足 {len(pred)}/{len(wl)} skip)"); continue
        ec = np.concatenate([pred[w] - tt[w] for w in wl])        # 閉ループ誤差
        dc = np.concatenate([pred[w] - priors[w] for w in wl])    # 補正場
        m = fin & np.isfinite(ec)
        rho = float(np.corrcoef(ec[m], ep[m])[0, 1])
        rms_d = float(np.sqrt(np.nanmean(dc[m] ** 2)))
        rmse_c = float(np.sqrt(np.mean(ec[m] ** 2)))
        # 井戸別: 補正場の大きさ(RMS)分布 → 補正が「一部の井戸に局在」か「全体に薄く」か
        per = np.array([float(np.sqrt(np.nanmean((pred[w] - priors[w]) ** 2))) for w in wl])
        print(f"{key:>6s} {rmse_c:8.4f} {rho:9.4f} {rms_d:14.3f} "
              f"{np.median(per):15.3f} {np.percentile(per, 90):15.3f}")
    print("\n読み方: rho→1 = エコー(priorコピー)。RMS(cl-prior)が大きく rho が低いσほど『別物の信号』。")
    print("per-well p90 ≫ median なら補正は難井戸に局在(=特徴として情報が濃い)。")

if __name__ == "__main__":
    main()
