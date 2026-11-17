# -*- coding: utf-8 -*-
"""ファイル名(ID)順で並べた時、隣接する井戸(file-order neighbor)が空間的に近いか。
比較: 隣接ペア距離 vs ランダムペア vs 真の最近傍。file order が空間近傍の手がかりになるか判定。
出力: コンソール + artifacts_v69_2/fileorder_proximity_v69.png。"""
import io, sys, pickle
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

ART = Path(r"C:\Users\kosaka256\Documents\rogii_claude\v69\artifacts_v69_2")
W = pickle.loads((ART / "wells_xyz_ancc_cache.pkl").read_bytes())
wids = sorted(W)                                    # ★ファイル名(ID)順
cent = np.array([[np.nanmean(W[w]["x"]), np.nanmean(W[w]["y"])] for w in wids])
N = len(wids)
tree = cKDTree(cent)

# 隣接(file-order)ペア距離
consec = np.hypot(*(cent[1:] - cent[:-1]).T)
# ランダムペア距離
rng = np.random.RandomState(0)
ridx = rng.randint(0, N, (20000, 2)); ridx = ridx[ridx[:, 0] != ridx[:, 1]]
randd = np.hypot(*(cent[ridx[:, 0]] - cent[ridx[:, 1]]).T)
# 真の最近傍距離
dd, _ = tree.query(cent, k=2); nn = dd[:, 1]

# file-order neighbor(i+1)が i の最近傍から何番目か(rank)
ranks = []
for i in range(N - 1):
    d_all = np.hypot(*(cent - cent[i]).T); order = np.argsort(d_all)  # order[0]=self
    r = int(np.where(order == i + 1)[0][0])                          # 1=最近傍
    ranks.append(r)
ranks = np.array(ranks)

def q(a): return np.round(np.percentile(a, [10, 25, 50, 75, 90]), 0)
print(f"井戸 {N} 本  (ID順)")
print(f"\n距離(ft) 分位 [10,25,50,75,90]:")
print(f"  隣接(file-order) : {q(consec)}")
print(f"  ランダムペア     : {q(randd)}")
print(f"  真の最近傍       : {q(nn)}")
print(f"\n隣接ペア距離の中央値 {np.median(consec):.0f}ft  vs ランダム {np.median(randd):.0f}ft  "
      f"→ {'近い(file orderに空間性あり)' if np.median(consec) < 0.6*np.median(randd) else '差なし(file order≒ランダム)'}")
print(f"隣接<500ft: {100*np.mean(consec<500):.0f}%  / ランダム<500ft: {100*np.mean(randd<500):.0f}%")
print(f"\nfile-order neighbor(i+1) の最近傍rank: 中央={int(np.median(ranks))}位  "
      f"(1位=真の最近傍だった割合 {100*np.mean(ranks==1):.0f}% / 上位5位内 {100*np.mean(ranks<=5):.0f}%)")
print(f"  ※ランダムなら rank 中央 ~{N//2}位。1位割合 ~{100/N:.1f}%")

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Meiryo"
fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
bins = np.linspace(0, np.percentile(randd, 95), 60)
ax[0].hist(randd, bins, density=True, alpha=0.5, label=f"ランダムペア (中央{np.median(randd):.0f})")
ax[0].hist(consec, bins, density=True, alpha=0.6, label=f"隣接(file-order) (中央{np.median(consec):.0f})")
ax[0].hist(nn, bins, density=True, alpha=0.5, label=f"真の最近傍 (中央{np.median(nn):.0f})")
ax[0].set_xlabel("距離 (ft)"); ax[0].set_ylabel("密度"); ax[0].legend(); ax[0].set_title("隣接ペア距離の分布")
# 空間配置を file-order で色付け(順序が地理を辿るか)
sc = ax[1].scatter(cent[:, 0], cent[:, 1], c=np.arange(N), cmap="turbo", s=14)
ax[1].set_xlabel("X"); ax[1].set_ylabel("Y"); ax[1].set_aspect("equal", "datalim")
ax[1].set_title("井戸重心(色=ID順index)  連続色が固まれば order≒地理")
fig.colorbar(sc, ax=ax[1], label="ID順 index")
fig.suptitle("ファイル(ID)順 と 空間近接の関係 — ROGII train"); fig.tight_layout()
fig.savefig(ART / "fileorder_proximity_v69.png", dpi=115)
print(f"\n[saved] {ART/'fileorder_proximity_v69.png'}")
