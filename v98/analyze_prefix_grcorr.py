# -*- coding: utf-8 -*-
"""事前区間(prefix)の GR相関(horizontal GR vs typewell/self/nbr GR) が、eval区間のPF精度を予測するか検証。
既存 train_combo.parquet の prov_*_grcorr(事前区間相関) と pf_*_delta(各表現PF予測) と target から、
井戸単位で「表現別 事前grcorr」vs「表現別 eval RMSE」の関係を測る。GPU不要。
出力: v98/prefix_grcorr_analysis.png + コンソール相関表。"""
import os, sys, io
try:
    sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
except Exception:
    pass
from pathlib import Path
import numpy as np, pandas as pd, pyarrow.parquet as pq

PQ = "v97/artifacts_v97/data/train_combo.parquet"
names = pq.read_schema(PQ).names
REPS = ["tw", "self", "nbr", "nbr_gr5", "self_graft"]
# 表現別 PF delta 列(各バンクの平滑delta)。tw は base(pf_ancc_delta*)
DELTA = {
    "tw":         [c for c in names if c == "pf_ancc_delta" or (c.startswith("pf_ancc_delta_") and c.split("_")[-1].isdigit())],
    "self":       [c for c in names if c.startswith("pf_self_delta_") and not c.startswith("pf_self_graft")],
    "nbr":        [c for c in names if c.startswith("pf_nbr_delta_")],
    "nbr_gr5":    [c for c in names if c.startswith("pf_nbr_gr5_delta_")],
    "self_graft": [c for c in names if c.startswith("pf_self_graft_delta_")],
}
GRCORR = {r: f"prov_{r}_grcorr" for r in REPS}
need = ["well", "target"] + list(GRCORR.values()) + sum(DELTA.values(), [])
need = [c for c in dict.fromkeys(need) if c in names]
print("表現別 delta列数:", {r: len(DELTA[r]) for r in REPS}, flush=True)

t = pq.read_table(PQ, columns=need)
df = t.to_pandas(); del t
g = df["well"].to_numpy(); y = df["target"].to_numpy(np.float64)
uniq = pd.unique(g); idx_of = {w: np.where(g == w)[0] for w in uniq}
print(f"読込: 行={len(df)} 井={len(uniq)}", flush=True)

# 井戸単位: 表現別 事前grcorr(先頭値) と eval RMSE(バンク平均delta vs target)
rows = []
for w in uniq:
    ix = idx_of[w]; yy = y[ix]; m = np.isfinite(yy)
    if m.sum() < 20:
        continue
    rec = {"well": w, "n": int(m.sum())}
    for r in REPS:
        gc = df[GRCORR[r]].to_numpy()[ix[0]]                          # 事前区間相関(well定数)
        dc = DELTA[r]
        if dc:
            pred = df.iloc[ix][dc].to_numpy(np.float64).mean(1)       # バンク平均delta
            e = pred[m] - yy[m]; rmse = float(np.sqrt(np.nanmean(e ** 2)))
        else:
            rmse = np.nan
        rec[f"grcorr_{r}"] = float(gc) if np.isfinite(gc) else np.nan
        rec[f"rmse_{r}"] = rmse
    rows.append(rec)
W = pd.DataFrame(rows)
print(f"有効井={len(W)}\n", flush=True)

# 相関: 各表現で 事前grcorr vs eval RMSE(井戸横断)。負=高相関ほど低誤差=仮説成立
from scipy.stats import spearmanr, pearsonr
print("=== 表現別: 事前区間grcorr vs eval RMSE の井戸横断相関 ===", flush=True)
print(f"{'rep':11s} {'n_well':>6s} {'grcorr平均':>9s} {'Pearson':>8s} {'Spearman':>9s}  解釈", flush=True)
for r in REPS:
    x = W[f"grcorr_{r}"].to_numpy(); yv = W[f"rmse_{r}"].to_numpy()
    m = np.isfinite(x) & np.isfinite(yv)
    if m.sum() < 30:
        print(f"{r:11s} 井不足"); continue
    pr = pearsonr(x[m], yv[m])[0]; sp = spearmanr(x[m], yv[m])[0]
    tag = "★予測力あり(高grcorr→低誤差)" if sp < -0.15 else ("弱い" if sp < -0.05 else "予測力なし")
    print(f"{r:11s} {int(m.sum()):6d} {np.nanmean(x[m]):9.3f} {pr:8.3f} {sp:9.3f}  {tag}", flush=True)

# grcorr五分位ごとの平均RMSE(tw)で傾向を可視化
print("\n=== tw: 事前grcorr 五分位 → eval RMSE 平均(単調に下がれば選択に有効) ===", flush=True)
x = W["grcorr_tw"].to_numpy(); yv = W["rmse_tw"].to_numpy(); m = np.isfinite(x) & np.isfinite(yv)
xb, yb = x[m], yv[m]; q = np.quantile(xb, [0, .2, .4, .6, .8, 1.0])
for i in range(5):
    sel = (xb >= q[i]) & (xb <= q[i + 1])
    print(f"  Q{i+1} grcorr[{q[i]:.2f},{q[i+1]:.2f}] : 平均RMSE={yb[sel].mean():.3f} (n={sel.sum()})", flush=True)

# 選択検証: 井ごとに grcorr最大の表現を選ぶと、固定tw/平均より良いか(oracle比較)
print("\n=== 選択検証: 事前grcorr最大の表現を採用 vs 各固定 ===", flush=True)
gr_mat = W[[f"grcorr_{r}" for r in REPS]].to_numpy(); rm_mat = W[[f"rmse_{r}" for r in REPS]].to_numpy()
valid = np.isfinite(gr_mat).all(1) & np.isfinite(rm_mat).all(1)
gm, rm = gr_mat[valid], rm_mat[valid]
pick = np.nanargmax(gm, 1)                                   # grcorr最大の表現
pooled = lambda col: rm[:, col]
sel_rmse = rm[np.arange(len(rm)), pick]
oracle = rm.min(1);
print(f"  対象井={len(rm)}", flush=True)
for j, r in enumerate(REPS):
    print(f"    固定{r:11s}: 平均RMSE={rm[:,j].mean():.3f}", flush=True)
print(f"  ★grcorr最大選択      : 平均RMSE={sel_rmse.mean():.3f}", flush=True)
print(f"  oracle(誤差最小選択)  : 平均RMSE={oracle.mean():.3f}", flush=True)
best_fixed = min(range(len(REPS)), key=lambda j: rm[:, j].mean())
print(f"  => grcorr選択 {'有効(最良固定より改善)' if sel_rmse.mean() < rm[:,best_fixed].mean() else '無効(最良固定に劣る)'}"
      f"  (最良固定={REPS[best_fixed]} {rm[:,best_fixed].mean():.3f})", flush=True)

# グラフ
try:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "Meiryo"
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for r in REPS:
        x = W[f"grcorr_{r}"].to_numpy(); yv = W[f"rmse_{r}"].to_numpy(); m = np.isfinite(x) & np.isfinite(yv)
        ax[0].scatter(x[m], yv[m], s=6, alpha=0.3, label=r)
    ax[0].set_xlabel("事前区間 GR相関 (grcorr)"); ax[0].set_ylabel("eval RMSE (表現別)")
    ax[0].set_ylim(0, 30); ax[0].legend(); ax[0].set_title("事前grcorr vs eval精度(表現別)")
    # tw五分位
    xb = W["grcorr_tw"].to_numpy(); yb = W["rmse_tw"].to_numpy(); m = np.isfinite(xb) & np.isfinite(yb)
    xb, yb = xb[m], yb[m]; q = np.quantile(xb, np.linspace(0, 1, 6))
    mids = [(q[i] + q[i + 1]) / 2 for i in range(5)]; means = [yb[(xb >= q[i]) & (xb <= q[i + 1])].mean() for i in range(5)]
    ax[1].plot(mids, means, "o-"); ax[1].set_xlabel("事前grcorr(tw) 五分位中央"); ax[1].set_ylabel("平均 eval RMSE")
    ax[1].set_title("tw: 事前grcorr↑ で精度↑ か")
    fig.tight_layout(); fig.savefig("v98/prefix_grcorr_analysis.png", dpi=110)
    print("\nsaved v98/prefix_grcorr_analysis.png", flush=True)
except Exception as e:
    print("plot skip:", e, flush=True)
