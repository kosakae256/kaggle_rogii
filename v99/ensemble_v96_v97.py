# -*- coding: utf-8 -*-
"""v97 train 完了(submit_v97_config.json)を待って、v96(LB6.173)との OOF アンサンブルCVを確認。
v96/v97 は同一773井・同一target。id で行整列して blend(nnls)。
出力: v97/artifacts_v97/ensemble_v96_v97.txt"""
import os, sys, io, json, time
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
from pathlib import Path
import numpy as np, pyarrow.parquet as pq
from scipy.optimize import nnls

PROJ = Path(r"C:\Users\kosaka256\Documents\rogii_claude")
A95 = PROJ / "v95" / "artifacts_v95"; A95B = PROJ / "v95" / "artifacts_v95b"
A97 = PROJ / "v97" / "artifacts_v97"
V97CFG = A97 / "submit_v97_config.json"


def load_blend(pq_path, gbm_oof, tcn_oof, l5_oof, weights):
    """(ids, blend_pred(残差), y) を返す。weights=[gbm,tcn,l5cf] 固定。"""
    tb = pq.read_table(pq_path, columns=["id", "target"]).to_pandas()
    ids = tb["id"].astype(str).to_numpy(); y = tb["target"].to_numpy(float)
    d1 = np.load(gbm_oof); gbm = np.mean([d1[k] for k in d1.files if k not in ("y", "well")], 0)
    tcn = np.mean([v for k, v in np.load(tcn_oof).items() if k.startswith("mdn2")], 0)
    l5 = np.load(l5_oof)["l5_mean"].astype(float)
    M = np.stack([gbm, tcn, l5], 1); w = np.asarray(weights, float)
    return ids, M @ w, y


def main():
    print(f"[wait] v97 train 完了({V97CFG.name})待ち...", flush=True)
    while not V97CFG.exists():
        time.sleep(120)
    time.sleep(20)
    cfg97 = json.loads(V97CFG.read_text())
    print(f"[ok] v97 完了。CV={cfg97.get('cv_rmse')} weights={cfg97.get('blend_weights')}", flush=True)

    # v96(現最良, LB6.173): gbm=artifacts_v95, tcn/l5cf=artifacts_v95b, 固定重み
    w96 = [0.047378426301733896, 0.5093755025195342, 0.44324607117873205]
    id96, b96, y96 = load_blend(A95B / "data" / "train.parquet",
                                A95 / "l1_oof.npz", A95B / "nn_oof_combo.npz", A95B / "l5_cf_oof.npz", w96)
    # v97
    w97 = cfg97["blend_weights"]
    l5_97 = A97 / "l5_cf_oof.npz"
    if not l5_97.exists():   # L5cf 失敗時は gbm/tcn 2脚
        w97 = w97[:2];
    id97, b97, y97 = load_blend(A97 / "data" / "train.parquet",
                                A97 / "l1_oof.npz", A97 / "nn_oof_combo.npz",
                                l5_97 if l5_97.exists() else A97 / "nn_oof_combo.npz", w97 if l5_97.exists() else w97)

    # id で整列
    m96 = {i: k for k, i in enumerate(id96)}; common = [i for i in id97 if i in m96]
    idx97 = {i: k for k, i in enumerate(id97)}
    j96 = np.array([m96[i] for i in common]); j97 = np.array([idx97[i] for i in common])
    B96 = b96[j96]; B97 = b97[j97]; Y = y96[j96]
    assert np.allclose(Y, y97[j97], equal_nan=True), "target 不一致(整列失敗)"
    fin = np.isfinite(B96) & np.isfinite(B97) & np.isfinite(Y)
    B96, B97, Y = B96[fin], B97[fin], Y[fin]

    def R(p): return float(np.sqrt(np.mean((p - Y) ** 2)))
    out = ["=== v96(LB6.173) × v97 アンサンブル(OOF, 共通{}行) ===".format(len(Y))]
    out.append(f"  v96 単体OOF = {R(B96):.4f}  (LB=6.173)")
    out.append(f"  v97 単体OOF = {R(B97):.4f}")
    out.append(f"  ★誤差相関 ρ(v96,v97) = {np.corrcoef(B96-Y,B97-Y)[0,1]:.3f}")
    # nnls 2脚
    M = np.stack([B96, B97], 1); w, _ = nnls(M, Y); w = w / w.sum()
    out.append(f"  ★nnls 2脚 blend = {R(M@w):.4f}  重み v96={w[0]:.3f}/v97={w[1]:.3f}  (v96比 {R(M@w)-R(B96):+.4f})")
    for fw in [0.2, 0.3, 0.4, 0.5]:
        p = (1 - fw) * B96 + fw * B97
        out.append(f"    固定 v97={fw}: {R(p):.4f} ({R(p)-R(B96):+.4f})")
    txt = "\n".join(out); print("\n" + txt, flush=True)
    (A97 / "ensemble_v96_v97.txt").write_text(txt, encoding="utf-8")


if __name__ == "__main__":
    main()
