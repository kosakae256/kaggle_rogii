# -*- coding: utf-8 -*-
"""診断A: モデルが使う192特徴の重複構造を見る。相関でクラスタリングし「実効的に何グループの独立信号か」を可視化。
   さらに各特徴を意味グループ(pf候補15本/証拠sc/空間prior/dip…)にラベル付けし、
   群内の冗長(相関クラスタ数)と 群→targetの相関(独立予測力) を出す。
   出力: コンソール + artifacts_v69_4/diag_feat_cluster.png"""
import io, sys, json, re
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform

ART = Path(r"C:\Users\kosaka256\Documents\rogii_claude\v69\artifacts_v69_4")
SUB = 40
feats = json.loads((ART / "features.json").read_text())
sch = set(f.name for f in pq.read_schema(ART / "data" / "train.parquet"))
feats = [f for f in feats if f in sch]
print(f"使用特徴 {len(feats)} 列を相関クラスタリング(行 ::{SUB} subsample)", flush=True)

# --- 特徴+target を subsample 読み(列一括→::SUB) ---
tb = pq.read_table(ART / "data" / "train.parquet", columns=feats + ["target"])
import pandas as pd
df = tb.slice(0, tb.num_rows).to_pandas()
del tb
X = df[feats].to_numpy(np.float32)[::SUB]
y = df["target"].to_numpy(np.float32)[::SUB]
del df
m = np.isfinite(X).all(1) & np.isfinite(y)
X = X[m]; y = y[m]
print(f"  行 {X.shape[0]}  列 {X.shape[1]}", flush=True)

# --- |Pearson| 相関行列 → 距離 1-|r| で階層クラスタ ---
C = np.corrcoef(X.T)
C = np.nan_to_num(C, nan=0.0)
D = 1 - np.abs(C); np.fill_diagonal(D, 0.0); D = (D + D.T) / 2
Z = linkage(squareform(D, checks=False), method="average")

print("\n=== 実効独立グループ数(相関しきい別) ===")
for thr in (0.95, 0.9, 0.8, 0.7, 0.5):
    ncl = fcluster(Z, t=1 - thr, criterion="distance").max()
    print(f"  |corr|>{thr}: {ncl:3d} グループ  ({len(feats)}列が実効{ncl}信号に畳める)")

# --- 意味グループへのラベル付け ---
def grp(name):
    n = name.lower()
    if re.match(r"pf_(ancc|self|nbr)", n) or re.match(r"pf_\d", n) or "pf_mean" in n or "pf_med" in n: return "PF候補(tw/self/nbr×5)"
    if "vs_spatial" in n or "vs_dense" in n or "vs_sc" in n or "vs_beam" in n or "vs_slp" in n: return "候補間の不一致(手作り)"
    if n.startswith("tvtf") or "_ancc" in n or "_astn" in n or "_egfd" in n or "_buda" in n: return "地層面prior(tvtF_*)"
    if "dense" in n: return "dense構造prior"
    if "spatial" in n: return "spatial prior"
    if n.startswith("sc") or "tdsc" in n or "_sc" in n: return "GR証拠(sc)"
    if "beam" in n: return "beam"
    if "slp" in n or "dzdmd" in n or "dip" in n or "z_grad" in n: return "dip/傾き"
    if "gr" in n: return "GR素"
    return "その他"

G = [grp(f) for f in feats]
uniq = sorted(set(G))
cl90 = fcluster(Z, t=1 - 0.9, criterion="distance")
print("\n=== 意味グループ別: 列数 / 相関クラスタ数(群内冗長) / target|相関|(中央,最大) ===")
tcorr = np.array([abs(np.corrcoef(X[:, i], y)[0, 1]) for i in range(len(feats))])
rows = []
for g in uniq:
    idx = [i for i, gg in enumerate(G) if gg == g]
    ncl = len(set(cl90[idx]))
    tc = tcorr[idx]
    print(f"  {g:22s}: {len(idx):3d}列 → 冗長圧縮後{ncl:2d}クラスタ  target|r| 中央{np.median(tc):.3f} 最大{tc.max():.3f}")
    rows.append((g, len(idx), ncl, np.median(tc), tc.max()))

# --- 図 ---
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Meiryo"
fig = plt.figure(figsize=(16, 9))
# (a) dendrogram
ax = fig.add_subplot(2, 2, 1)
dendrogram(Z, no_labels=True, color_threshold=1 - 0.8, ax=ax)
ax.axhline(1 - 0.9, c="r", ls="--", lw=.8); ax.axhline(1 - 0.8, c="orange", ls="--", lw=.8)
ax.set_title("① 192特徴の階層クラスタ(距離=1-|corr|)\n赤=|corr|0.9 橙=0.8"); ax.set_ylabel("1-|corr|")
# (b) 実効グループ数 vs しきい
ax = fig.add_subplot(2, 2, 2)
thrs = [0.95, 0.9, 0.8, 0.7, 0.5]; ncls = [fcluster(Z, t=1 - t, criterion="distance").max() for t in thrs]
ax.plot(thrs, ncls, "o-"); ax.set_xlabel("|corr| しきい"); ax.set_ylabel("実効グループ数")
ax.set_title(f"② 192列 → 実効独立信号数\n(|corr|>0.9で{ncls[1]}グループ)"); ax.invert_xaxis(); ax.grid(alpha=.3)
for t, nc in zip(thrs, ncls): ax.text(t, nc + 2, str(nc), ha="center")
# (c) 意味グループ: 列数 vs 冗長圧縮後
ax = fig.add_subplot(2, 2, 3)
gg = [r[0] for r in rows]; ncol = [r[1] for r in rows]; nclr = [r[2] for r in rows]
yp = np.arange(len(gg))
ax.barh(yp, ncol, color="#bcd", label="列数")
ax.barh(yp, nclr, color="#369", label="冗長圧縮後クラスタ")
ax.set_yticks(yp); ax.set_yticklabels(gg, fontsize=8); ax.invert_yaxis(); ax.legend()
ax.set_title("③ 意味グループ別 列数と実効信号数(群内冗長)")
# (d) 意味グループ: target相関
ax = fig.add_subplot(2, 2, 4)
tcm = [r[3] for r in rows]; tcx = [r[4] for r in rows]
ax.barh(yp, tcx, color="#fca", label="最大|r|")
ax.barh(yp, tcm, color="#e63", label="中央|r|")
ax.set_yticks(yp); ax.set_yticklabels(gg, fontsize=8); ax.invert_yaxis(); ax.legend()
ax.set_xlabel("target との |相関|"); ax.set_title("④ 意味グループ別 target予測力")
fig.suptitle("診断A: v69_4 使用192特徴の重複構造(似ているもの/別物)", fontsize=13)
fig.tight_layout()
fig.savefig(ART / "diag_feat_cluster.png", dpi=115)
print(f"\n[saved] {ART/'diag_feat_cluster.png'}")
