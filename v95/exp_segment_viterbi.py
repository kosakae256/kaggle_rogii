# -*- coding: utf-8 -*-
"""夜間実験A(CPU): per-segment枝をglobal Viterbi(dip連続transition + PF支持emission)で解けるか。
research OQ4検証。segment枝oracle(4.3prize)をViterbiがどこまで実現するか、per-well backboneを超えるか。
真TVTはoracle/評価のみ。出力: v95/artifacts_v95b/exp_segment_viterbi.txt"""
import io, sys
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
import numpy as np, pandas as pd, pyarrow.parquet as pq
from scipy.optimize import nnls
B = "v95/artifacts_v95b/"
REPS = ["ancc", "self", "nbr", "nbr_gr5", "self_graft"]
sch = set(pq.read_schema(B + "data/train.parquet").names)
PC = [f"pf_{r}_delta_{k}" for r in REPS for k in range(1, 7) if f"pf_{r}_delta_{k}" in sch]
tb = pq.read_table(B + "data/train.parquet", columns=PC + ["well", "target", "last_known_tvt", "md_since", "z"]).to_pandas()
y = tb["target"].to_numpy(float); g = tb["well"].astype(str).to_numpy()
lk = tb["last_known_tvt"].to_numpy(float); md = tb["md_since"].to_numpy(float); z = tb["z"].to_numpy(float)
P = tb[PC].to_numpy(float)
d1 = np.load(B.replace("v95b", "v95") + "l1_oof.npz")
gbm = np.mean([d1[k] for k in d1.files if k not in ("y", "well")], 0)
tcn = np.mean([v for k, v in np.load(B + "nn_oof_combo.npz").items() if k.startswith("mdn2")], 0)
l5 = np.load(B + "l5_ext_oof.npz")["l5_mean"].astype(float)
M = np.stack([gbm, tcn, l5], 1); mf = np.isfinite(M).all(1) & np.isfinite(y)
w, _ = nnls(M[mf], y[mf]); w = w / w.sum(); blend = M @ w   # 残差空間 blend
def R(p): m = np.isfinite(p) & np.isfinite(y); return float(np.sqrt(np.mean((p[m] - y[m]) ** 2)))
print(f"blend(base)={R(blend):.4f}", flush=True)
ch = np.where(g[1:] != g[:-1])[0] + 1; starts = np.concatenate([[0], ch]); ends = np.concatenate([ch, [len(g)]])
STR = 16; BR = np.array([-2, -1, 0, 1, 2]); NB = len(BR)

oracle = blend.copy(); vit = blend.copy(); wellmaj = blend.copy()
for s, e in zip(starts, ends):
    o = np.argsort(md[s:e], kind="stable"); idx = np.arange(s, e)[o]
    T = e - s; nb = (T + STR - 1) // STR
    seg_of = np.minimum(np.arange(T) // STR, nb - 1)
    bl_s = blend[idx]; y_s = y[idx]; P_s = P[idx]; z_s = z[idx]
    # per-segment blend中央値 + segment代表z(dip連続用)
    seg_bl = np.array([np.nanmedian(bl_s[seg_of == b]) for b in range(nb)])
    seg_z = np.array([np.nanmean(z_s[seg_of == b]) for b in range(nb)])
    # emission E[seg, branch] = 30PF出力が (val - seg_bl) ≈ br*15 を支持する割合(log)
    E = np.full((nb, NB), -20.0)
    for b in range(nb):
        rows = np.where(seg_of == b)[0]
        dd = P_s[rows] - seg_bl[b]                       # (n,30) blendからの差
        for j, br in enumerate(BR):
            supp = np.mean(np.abs(dd - br * 15.0) < 7.5)  # br枝を支持する割合
            E[b, j] = np.log(supp + 1e-3)
    # oracle: 各segment真の最良枝
    for b in range(nb):
        rows = np.where(seg_of == b)[0]
        errs = [np.sqrt(np.nanmean(((seg_bl[b] + br * 15.0) - y_s[rows]) ** 2)) for br in BR]
        best = BR[int(np.argmin(errs))]; oracle[idx[rows]] = seg_bl[b] + best * 15.0
    # Viterbi: emission + dip連続transition(枝跳びを幾何不連続としてペナルティ)
    LAM = 3.0                                            # transition強度
    dp = np.full((nb, NB), -1e9); bp = np.zeros((nb, NB), int); dp[0] = E[0]
    for b in range(1, nb):
        for j in range(NB):
            trans = -LAM * np.abs(BR - BR[j])            # 枝変化=15ft*|Δ| の不連続ペナルティ
            val = dp[b - 1] + trans; k = int(np.argmax(val))
            dp[b, j] = val[k] + E[b, j]; bp[b, j] = k
    path = np.zeros(nb, int); path[-1] = int(np.argmax(dp[-1]))
    for b in range(nb - 2, -1, -1): path[b] = bp[b + 1, path[b + 1]]
    for b in range(nb):
        rows = np.where(seg_of == b)[0]; vit[idx[rows]] = seg_bl[b] + BR[path[b]] * 15.0
    # 参考: per-well多数派枝(emission argmaxの井戸最頻)
    segbr = BR[np.argmax(E, 1)]; vals, cnts = np.unique(segbr, return_counts=True); mb = vals[np.argmax(cnts)]
    for b in range(nb):
        rows = np.where(seg_of == b)[0]; wellmaj[idx[rows]] = seg_bl[b] + mb * 15.0

out = ["=== 実験A: per-segment枝 Viterbi (OQ4) ==="]
out.append(f"  blend(base)              = {R(blend):.4f}")
out.append(f"  ★per-segment枝 oracle    = {R(oracle):.4f}  (4.3prize相当の天井)")
out.append(f"  Viterbi(emission+dip連続) = {R(vit):.4f}  ({R(vit)-R(blend):+.4f})")
out.append(f"  per-well多数派枝          = {R(wellmaj):.4f}  ({R(wellmaj)-R(blend):+.4f})")
out.append(f"  → Viterbiがoracleに迫れば4.x射程/blend未超なら壁確認(emission=両義consensusで枝割れず)")
txt = "\n".join(out); print("\n" + txt, flush=True)
open(B + "exp_segment_viterbi.txt", "w", encoding="utf-8").write(txt)
