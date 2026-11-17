# -*- coding: utf-8 -*-
"""bwd の"生"予測(逆方向コンセンサス, 学習なし)を脚にして v100 に混ぜ、脱相関信号が残っているか診断。
bpf_{rep}_delta は target空間(TVT_bwd − last_known)= OOF の y と直接比較可。
生でも blend に非ゼロ重みが付けば → bwd主役モデルを作る価値あり。付かなければ bwd は forward と本質的に重複。"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import numpy as np
import pyarrow.parquet as pq
from scipy.optimize import nnls

A = "v100/artifacts_v100/"
# --- v100 OOF 脚 ---
tcn = np.mean([np.load(A + "nn_oof_combo_fcombo.npz")[f"mdn2_s{i}"] for i in range(5)], 0)
dl = np.load(A + "l5_fcombo_oof.npz"); l5 = dl["l5_mean"]; y = dl["y"]

# --- bwd 生予測(train_combo_bwd から bpf delta を読む。行はOOFとpositional一致) ---
REPS = ["ancc", "self", "nbr", "nbr_gr5", "self_graft"]
tw_cols = [f"bpf_ancc_delta_{k}" for k in range(1, 7)]                        # 逆方向tw 6バンク
all_cols = [f"bpf_{r}_delta_{k}" for r in REPS for k in range(1, 7)]          # 全30
tb = pq.read_table(A + "data/train_combo_bwd.parquet", columns=tw_cols + [f"bpf_{r}_delta_{k}" for r in REPS[1:] for k in range(1, 7)])
def col(c): return tb.column(c).to_numpy(zero_copy_only=False).astype(np.float64)
bwd_tw = np.nanmean(np.stack([col(c) for c in tw_cols], 1), 1)               # 逆方向tw consensus
bwd_all = np.nanmean(np.stack([col(c) for c in all_cols], 1), 1)             # 逆方向 全表現 consensus

ok = np.isfinite(tcn) & np.isfinite(l5) & np.isfinite(y) & (l5 != 0) & np.isfinite(bwd_tw) & np.isfinite(bwd_all)
tcn, l5, y, bwd_tw, bwd_all = tcn[ok], l5[ok], y[ok], bwd_tw[ok], bwd_all[ok]
print(f"[probe] rows={ok.sum()}")

def rmse(p): return float(np.sqrt(np.mean((p - y) ** 2)))
def corr(a, b): return float(np.corrcoef(a - y, b - y)[0, 1])
print(f"単体 RMSE: TCN={rmse(tcn):.3f}  L5={rmse(l5):.3f}  bwd_tw(生)={rmse(bwd_tw):.3f}  bwd_all(生)={rmse(bwd_all):.3f}")
print(f"誤差相関 vs L5: bwd_tw={corr(bwd_tw,l5):.3f}  bwd_all={corr(bwd_all,l5):.3f}  (TCN×L5={corr(tcn,l5):.3f} 参考)")

# baseline v100 blend
def cvx2(a, b, step=0.02):
    best = (1e9, None)
    for w in np.arange(0, 1 + 1e-9, step):
        r = rmse(w * a + (1 - w) * b)
        if r < best[0]: best = (r, w)
    return best
r0, w0 = cvx2(tcn, l5)
print(f"\n[v100 baseline] TCN+L5 = {r0:.4f} (w_tcn={w0:.2f})")

# NNLS で bwd生脚を足す
for name, bwd in [("bwd_tw", bwd_tw), ("bwd_all", bwd_all)]:
    M = np.stack([tcn, l5, bwd], 1); w, _ = nnls(M, y)
    print(f"[+{name}] NNLS w(TCN,L5,{name})=({w[0]:.3f},{w[1]:.3f},{w[2]:.3f})  RMSE={rmse(M@w):.4f}  Δ={rmse(M@w)-rmse(np.stack([tcn,l5],1)@nnls(np.stack([tcn,l5],1),y)[0]):+.4f}")
