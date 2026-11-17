# -*- coding: utf-8 -*-
"""v98 差し込み実験の評価: TCN/L5/GRU/Transformer のOOFで 脱相関(誤差相関) と blend増分 を測定。
OOF は全て target(δ=TVT-last_known)単位。存在する脚だけ使う。GPU不要。
出力: コンソール(脚別RMSE / 誤差相関行列 / blend RMSE と増分)。"""
import os, sys, io, glob
try:
    sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
except Exception:
    pass
from pathlib import Path
import numpy as np, pyarrow.parquet as pq

ART = Path("v98/artifacts_v99")
LEGS = {
    "TCN":   ART / "nn_oof_combo_fcombo.npz",
    "L5":    ART / "l5_fcombo_oof.npz",
    "GRU":   ART / "nn_oof_combo_fcombo_alt-gru.npz",
    "TFMR":  ART / "nn_oof_combo_fcombo_alt-transformer.npz",
}


def load_leg(p):
    if not p.exists():
        return None
    d = np.load(p)
    ks = [k for k in d.files if k.startswith("mdn2_s") or k.startswith("l5_s")]
    if not ks:
        return None
    return np.mean([d[k] for k in ks], 0).astype(np.float64)   # seed平均(target単位)


def main():
    # y(target)
    y = pq.read_table(str(ART / "data" / "train_combo.parquet"), columns=["target"]).column("target").to_numpy(zero_copy_only=False).astype(np.float64)
    legs = {}
    for name, p in LEGS.items():
        v = load_leg(p)
        if v is not None and len(v) == len(y):
            legs[name] = v
    print(f"[blend比較] y行={len(y)} 利用可能脚={list(legs.keys())}", flush=True)
    if len(legs) < 2:
        print("脚が2つ未満 -> 比較不可(脚の学習完了待ち)"); return
    # 共通マスク(全脚 finite & !=0)
    m = np.isfinite(y)
    for v in legs.values():
        m &= np.isfinite(v) & (v != 0)
    y = y[m]; L = {k: v[m] for k, v in legs.items()}
    names = list(L.keys())
    rmse = lambda e: float(np.sqrt(np.mean(e ** 2)))
    print("\n=== 脚別 OOF RMSE(target単位, raw) ===", flush=True)
    for n in names:
        print(f"  {n:6s}: {rmse(L[n]-y):.4f}", flush=True)
    print("\n=== 誤差相関(低いほど脱相関=blend価値大) ===", flush=True)
    E = {n: L[n]-y for n in names}
    print("        " + " ".join(f"{n:>6s}" for n in names), flush=True)
    for a in names:
        row = " ".join(f"{np.corrcoef(E[a],E[b])[0,1]:6.3f}" for b in names)
        print(f"  {a:6s} {row}", flush=True)

    def blend_rmse(sub):
        """非負最小二乗で最良線形blend(OOF上)。"""
        A = np.stack([L[n] for n in sub], 1)
        try:
            from scipy.optimize import nnls
            w, _ = nnls(A, y); pred = A @ w
        except Exception:
            w = np.linalg.lstsq(A, y, rcond=None)[0]; pred = A @ w
        return rmse(pred - y), w
    print("\n=== blend(非負最小二乗) RMSE と増分 ===", flush=True)
    base = [n for n in ["TCN", "L5"] if n in names]
    if len(base) >= 1:
        r0, w0 = blend_rmse(base)
        print(f"  基準 {'+'.join(base)}: {r0:.4f}  w={np.round(w0,3)}", flush=True)
        for alt in [n for n in ["GRU", "TFMR"] if n in names]:
            r1, w1 = blend_rmse(base + [alt])
            print(f"  +{alt:5s}: {r1:.4f}  (増分 {r1-r0:+.4f})  w={np.round(w1,3)}", flush=True)
        alln = base + [n for n in ["GRU", "TFMR"] if n in names]
        if len(alln) > len(base):
            rA, wA = blend_rmse(alln)
            print(f"  全脚 {'+'.join(alln)}: {rA:.4f}  (基準比 {rA-r0:+.4f})  w={np.round(wA,3)}", flush=True)
    print("\n※ raw target RMSE。pp(後処理)未適用なので絶対値は combo blend(5.8881)と直接比較不可。増分/相関で判断。", flush=True)


if __name__ == "__main__":
    main()
