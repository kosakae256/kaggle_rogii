# -*- coding: utf-8 -*-
"""artifacts_v69_2 の4本PF(pf_ancc_1..4)の 誤差相関 と oracle 調査。
overall pooled RMSE(全eval行, Kaggle指標)。true_TVT = last_known + target(=残差)。
出力: コンソール(相関行列/RMSE表) + artifacts_v69_2/pf4_corr_oracle_v69_2.png(Meiryoグラフ)。"""
import io, sys
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

ART = Path(r"C:\Users\kosaka256\Documents\rogii_claude\v69\artifacts_v69_2")
BANKS = [1, 2, 3, 4]
cols = ["well", "target", "last_known_tvt"] + [f"pf_ancc_{i}" for i in BANKS]
tb = pq.read_table(ART / "data" / "train.parquet", columns=cols).to_pandas()
lk = tb["last_known_tvt"].to_numpy(np.float64)
true = lk + tb["target"].to_numpy(np.float64)
P = np.stack([tb[f"pf_ancc_{i}"].to_numpy(np.float64) for i in BANKS], 0)   # [4, N]
err = P - true[None, :]                                                     # [4, N]
fin = np.isfinite(true) & np.isfinite(err).all(0)
print(f"rows={len(true)}  finite={fin.sum()}")


def rmse(e): e = e[np.isfinite(e)]; return float(np.sqrt(np.mean(e ** 2))) if e.size else float("nan")


# --- 単体 / 平均 / 中央 / oracle ---
per = [rmse(err[k]) for k in range(4)]
mean_b = rmse(P.mean(0) - true)
med_b = rmse(np.median(P, 0) - true)
hard = rmse(np.min(np.abs(err), 0))                          # 各行 最良バンク選択
lo, hi = P.min(0), P.max(0)
convex = rmse(np.clip(true, lo, hi) - true)                  # 4本が真値を挟めば0、外なら最近傍境界まで
inrange = float(np.mean((true[fin] >= lo[fin]) & (true[fin] <= hi[fin])))
print("\n=== overall pooled RMSE ===")
for i, r in zip(BANKS, per): print(f"  pf_ancc_{i} = {r:.4f}")
print(f"  mean(4)   = {mean_b:.4f}")
print(f"  median(4) = {med_b:.4f}")
print(f"  hard-oracle(行別最良) = {hard:.4f}")
print(f"  convex-oracle(範囲内clip) = {convex:.4f}   (真値がバンク範囲内 = {inrange*100:.1f}%)")
best_single = min(per)
print(f"  → 最良単体={best_single:.4f} / median={med_b:.4f} / hard-oracle={hard:.4f}  余地(median−oracle)={med_b-hard:.3f}ft")

# --- 誤差相関(4x4) ---
E = err[:, fin]
C = np.corrcoef(E)
print("\n=== 誤差相関行列(Pearson) ===")
print("        " + "  ".join(f"pf{ i}" for i in BANKS))
for i in range(4):
    print(f"  pf{BANKS[i]}  " + "  ".join(f"{C[i,j]:.3f}" for j in range(4)))
# oracle 選択率(各バンクが行別最良だった割合)
oidx = np.argmin(np.abs(E), 0)
share = [float(np.mean(oidx == k)) for k in range(4)]
print("\n=== oracle選択率(その行で最良だった割合) ===")
for i, s in zip(BANKS, share): print(f"  pf_ancc_{i}: {s*100:.1f}%")

# --- グラフ ---
import matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Meiryo"
fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
im = ax[0].imshow(C, vmin=0, vmax=1, cmap="viridis")
ax[0].set_xticks(range(4)); ax[0].set_yticks(range(4))
ax[0].set_xticklabels([f"pf{i}" for i in BANKS]); ax[0].set_yticklabels([f"pf{i}" for i in BANKS])
for i in range(4):
    for j in range(4):
        ax[0].text(j, i, f"{C[i,j]:.2f}", ha="center", va="center",
                   color="white" if C[i, j] < 0.7 else "black", fontsize=10)
ax[0].set_title(f"4本PF 誤差相関(平均={np.mean(C[np.triu_indices(4,1)]):.3f})")
fig.colorbar(im, ax=ax[0], fraction=0.046)
labels = [f"pf{i}" for i in BANKS] + ["mean", "median", "hard\noracle", "convex\noracle"]
vals = per + [mean_b, med_b, hard, convex]
colors = ["#4363d8"] * 4 + ["#3cb44b", "#3cb44b", "#e6194B", "#f58231"]
b = ax[1].bar(range(len(vals)), vals, color=colors)
ax[1].set_xticks(range(len(vals))); ax[1].set_xticklabels(labels)
ax[1].set_ylabel("overall RMSE (ft)"); ax[1].set_title("単体 vs ブレンド vs oracle")
ax[1].grid(axis="y", alpha=0.3)
for r, v in zip(b, vals): ax[1].text(r.get_x()+r.get_width()/2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
fig.suptitle("artifacts_v69_2 4本PF(pf_ancc_1..4) 相関 & oracle")
fig.tight_layout(); fig.savefig(ART / "pf4_corr_oracle_v69_2.png", dpi=115)
print(f"\n[saved] {ART/'pf4_corr_oracle_v69_2.png'}")
