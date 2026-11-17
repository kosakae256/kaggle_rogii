# L4-L7 3seed(+旧s0) の fold別RMSE分散と、スタック構成比較の図
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Meiryo"

R = {  # [seed][fold] 今回の再学習ラン(VFRAC0.85)
 "L4": {0: [5.1848, 5.5890, 6.3658, 7.9068, 6.3654], 1: [5.1407, 5.5939, 6.2573, 7.7090, 6.4563], 2: [5.1076, 5.6267, 6.3427, 7.8934, 6.3411]},
 "L5": {0: [4.9769, 5.5790, 6.1629, 6.7824, 6.0153], 1: [4.9549, 5.4895, 6.2618, 6.7281, 6.7429], 2: [4.9207, 5.5173, 6.2301, 7.3951, 6.2839]},
 "L6": {0: [5.1679, 5.2509, 6.3555, 7.2599, 6.4623], 1: [5.2003, 5.3150, 6.2769, 7.6514, 6.4508], 2: [5.1428, 5.3412, 6.3503, 7.7293, 6.4557]},
 "L7": {0: [5.0423, 5.5339, 6.2025, 6.8968, 6.6498], 1: [4.9993, 5.4979, 6.2996, 6.7122, 6.6348], 2: [4.9801, 5.5293, 6.2328, 7.1434, 6.4496]},
}
MEAN4 = {"L4": 6.2756, "L5": 5.9322, "L6": 6.1482, "L7": 6.0553}   # 4member平均OOFのoverall
SINGLE = {"L4": 6.35, "L5": 5.9278, "L6": 6.15, "L7": 6.10}          # 旧s0単発(VFRAC0.5)

fig, axes = plt.subplots(1, 4, figsize=(16, 4.2), sharey=True)
cols = {0: "#1f77b4", 1: "#ff7f0e", 2: "#2ca02c"}
for ax, leg in zip(axes, ["L4", "L5", "L6", "L7"]):
    for sd in (0, 1, 2):
        ax.plot(range(5), R[leg][sd], "o-", ms=5, lw=1, alpha=0.75, color=cols[sd], label=f"seed{sd}")
    mu = np.mean([R[leg][sd] for sd in (0, 1, 2)], 0)
    ax.plot(range(5), mu, "k--", lw=1.2, label="fold平均(参考)")
    ax.set_title(f"{leg}  4member={MEAN4[leg]:.3f}(旧s0単発 {SINGLE[leg]:.3f})", fontsize=10)
    ax.set_xlabel("fold"); ax.grid(alpha=0.3)
    ax.set_xticks(range(5))
axes[0].set_ylabel("fold RMSE")
axes[0].legend(fontsize=8)
# seed事故の注記
axes[1].annotate("s2事故 7.40", xy=(3, 7.3951), xytext=(2.0, 7.65), fontsize=9, color="#2ca02c",
                 arrowprops=dict(arrowstyle="->", color="#2ca02c"))
axes[1].annotate("s1事故 6.74", xy=(4, 6.7429), xytext=(2.6, 7.05), fontsize=9, color="#ff7f0e",
                 arrowprops=dict(arrowstyle="->", color="#ff7f0e"))
plt.suptitle("v62 L4-L7 seed×fold RMSE(再学習3seed, VFRAC0.85)— L4/L6のfold3は全seed悪(アーキ由来)、L5はfold単位のseed事故あり", fontsize=11)
plt.tight_layout()
plt.savefig("v62/fig_seed3_v62.png", dpi=130)
print("saved v62/fig_seed3_v62.png")
