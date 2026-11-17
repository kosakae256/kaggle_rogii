# -*- coding: utf-8 -*-
"""bwd を「blend脚」として v100(TCN + L5-K71)に外から混ぜたら効くか。OOF演算のみ(GPU不要)。
脚: TCN(nn_oof mdn2 5seed mean) / L5prod(l5_fcombo K71 5seed) / L5bwd(l5_bwd3 K101 3seed) / L5base3(K71 3seed)。
convex重み(≥0,和1)グリッドで最小overall RMSE。誤差相関も出す。"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import numpy as np
from scipy.optimize import nnls
from itertools import product

A = "v100/artifacts_v100/"
tcn = np.mean([np.load(A + "nn_oof_combo_fcombo.npz")[k] for k in ["mdn2_s%d" % i for i in range(5)]], 0)
dl = np.load(A + "l5_fcombo_oof.npz"); l5p = dl["l5_mean"]; y = dl["y"]
l5b3 = np.load(A + "l5_base3_oof.npz")["l5_mean"]
l5bw = np.load(A + "l5_bwd3_oof.npz")["l5_mean"]

ok = np.isfinite(tcn) & np.isfinite(l5p) & np.isfinite(l5b3) & np.isfinite(l5bw) & np.isfinite(y) & (l5p != 0) & (l5bw != 0)
tcn, l5p, l5b3, l5bw, y = tcn[ok], l5p[ok], l5b3[ok], l5bw[ok], y[ok]
print(f"[blend] rows={ok.sum()}  ", flush=True)


def rmse(p): return float(np.sqrt(np.mean((p - y) ** 2)))
print(f"単体: TCN={rmse(tcn):.4f}  L5prod(K71,5s)={rmse(l5p):.4f}  L5base3(K71,3s)={rmse(l5b3):.4f}  L5+bwd(K101,3s)={rmse(l5bw):.4f}")

# 誤差相関(bwd脚が脱相関してるか)
def corr(a, b): return float(np.corrcoef(a - y, b - y)[0, 1])
print(f"誤差相関: L5bwd×L5base3={corr(l5bw, l5b3):.4f}  L5bwd×TCN={corr(l5bw, tcn):.4f}  L5bwd×L5prod={corr(l5bw, l5p):.4f}")


def best_convex(legs, names, step=0.02):
    ws = np.arange(0, 1 + 1e-9, step); best = (1e9, None)
    if len(legs) == 2:
        for a in ws:
            p = a * legs[0] + (1 - a) * legs[1]; r = rmse(p)
            if r < best[0]: best = (r, (a, 1 - a))
    else:  # 3脚 simplex
        for a in ws:
            for b in ws:
                if a + b > 1 + 1e-9: continue
                c = 1 - a - b
                p = a * legs[0] + b * legs[1] + c * legs[2]; r = rmse(p)
                if r < best[0]: best = (r, (a, b, c))
    return best


print("\n===== convex blend(≥0, 和1) =====")
r2, w2 = best_convex([tcn, l5p], ["TCN", "L5prod"])
print(f"[v100 baseline] TCN+L5prod           = {r2:.4f}  w={tuple(round(x,3) for x in w2)}")
r3, w3 = best_convex([tcn, l5p, l5bw], ["TCN", "L5prod", "L5bwd"])
print(f"[+bwd脚]        TCN+L5prod+L5bwd      = {r3:.4f}  w={tuple(round(x,3) for x in w3)}   Δ={r3-r2:+.4f}")

# seed一致版(3seed同士でフェア比較)
r2b, w2b = best_convex([tcn, l5b3], ["TCN", "L5base3"])
r3b, w3b = best_convex([tcn, l5b3, l5bw], ["TCN", "L5base3", "L5bwd"])
print(f"\n[seed一致 baseline] TCN+L5base3        = {r2b:.4f}  w={tuple(round(x,3) for x in w2b)}")
print(f"[seed一致 +bwd脚]   TCN+L5base3+L5bwd   = {r3b:.4f}  w={tuple(round(x,3) for x in w3b)}   Δ={r3b-r2b:+.4f}")

# NNLS(非負・和自由=v93方式)参考
Amat = np.stack([tcn, l5p, l5bw], 1); w, _ = nnls(Amat, y)
print(f"\n[NNLS 参考] w(TCN,L5prod,L5bwd)={tuple(round(x,3) for x in w)}  RMSE={rmse(Amat@w):.4f}")
