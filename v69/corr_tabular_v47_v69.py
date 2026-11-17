# -*- coding: utf-8 -*-
"""
v69 tabular(GBM) OOF × v47 tabular(GBM) OOF の相関(行レベル=overall)。
両 OOF は残差 d=TVT-lk(行整合, 3783989行)。誤差相関=ブレンド多様性の判定。
実行: .venv/Scripts/python.exe v69/corr_tabular_v47_v69.py
"""
import sys, re
sys.stdout = open(1, "w", encoding="utf-8", closefd=False)
import numpy as np, pyarrow.parquet as pq
from pathlib import Path
P = Path(r"C:\Users\kosaka256\Documents\rogii_claude")

o47 = np.load(P / "v47/artifacts_v47/gbm_v41_oof.npy").astype(np.float64)
o69 = np.load(P / "v69/artifacts_v69/gbm_v41_oof.npy").astype(np.float64)
tgt = pq.read_table(P / "v69/artifacts_v69/data/train.parquet", columns=["target"])["target"].to_numpy().astype(np.float64)

# v69 OOF がどのバンク構成のビルドか(parquet の pf_ancc suffix)
sch = pq.read_schema(P / "v69/artifacts_v69/data/train.parquet")
sufs = sorted(set(re.findall(r"pf_ancc(_\d+)", " ".join(sch.names))))
print(f"[v69 parquet] pf_ancc suffixes = {sufs}  (この OOF が使ったバンク数)")

n = min(len(o47), len(o69), len(tgt))
o47, o69, tgt = o47[:n], o69[:n], tgt[:n]
e47 = o47 - tgt; e69 = o69 - tgt
fin = np.isfinite(e47) & np.isfinite(e69)
e47, e69, o47f, o69f = e47[fin], e69[fin], o47[fin], o69[fin]
N = fin.sum()

def rmse(e): return float(np.sqrt(np.mean(e ** 2)))
r47, r69 = rmse(e47), rmse(e69)
pear_pred = float(np.corrcoef(o47f, o69f)[0, 1])
pear_err = float(np.corrcoef(e47, e69)[0, 1])
# 50:50 と最適重み(誤差の線形結合 e(w)=w*e47+(1-w)*e69, Σ=1)
G = np.array([[np.mean(e47 * e47), np.mean(e47 * e69)], [np.mean(e47 * e69), np.mean(e69 * e69)]])
ws = np.linspace(0, 1, 101)
rs = [np.sqrt(max(w * w * G[0, 0] + 2 * w * (1 - w) * G[0, 1] + (1 - w) ** 2 * G[1, 1], 0)) for w in ws]
wbest = ws[int(np.argmin(rs))]; rbest = float(min(rs))
r5050 = float(np.sqrt(0.25 * G[0, 0] + 0.5 * G[0, 1] + 0.25 * G[1, 1]))

print(f"\n有効行 {N}")
print(f"[個別 overall RMSE]  v47_tab={r47:.4f}   v69_tab={r69:.4f}")
print(f"[相関] 予測 Pearson={pear_pred:.4f}   誤差 Pearson={pear_err:.4f}  ← 低いほどブレンド多様")
print(f"[50:50 ブレンド] RMSE={r5050:.4f}  (最良単体 {min(r47,r69):.4f} 比 {r5050-min(r47,r69):+.4f})")
print(f"[最適重み] w_v47={wbest:.2f} → RMSE={rbest:.4f}")

try:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "Meiryo"
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].bar(["v47_tab", "v69_tab", "50:50", f"最適\nw47={wbest:.2f}"], [r47, r69, r5050, rbest],
              color=["#4C72B0", "#55A868", "#8172B3", "#CCB974"])
    for i, v in enumerate([r47, r69, r5050, rbest]): ax[0].text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)
    ax[0].set_ylabel("overall RMSE"); ax[0].set_ylim(min(r47, r69, rbest) - 0.3, max(r47, r69, r5050) + 0.3)
    ax[0].set_title(f"tabular OOF: v47 vs v69  (誤差相関 {pear_err:.3f})"); ax[0].grid(axis="y", alpha=0.3)
    idx = np.random.RandomState(0).choice(N, min(40000, N), replace=False)
    ax[1].scatter(e47[idx], e69[idx], s=2, alpha=0.15, color="#333")
    lim = np.percentile(np.abs(np.concatenate([e47[idx], e69[idx]])), 99)
    ax[1].plot([-lim, lim], [-lim, lim], "r--", lw=1); ax[1].set_xlim(-lim, lim); ax[1].set_ylim(-lim, lim)
    ax[1].set_xlabel("v47_tab 誤差(ft)"); ax[1].set_ylabel("v69_tab 誤差(ft)")
    ax[1].set_title(f"誤差散布(相関 {pear_err:.3f})"); ax[1].grid(alpha=0.3)
    fig.tight_layout(); png = P / "v69/artifacts_v69/corr_tabular_v47_v69.png"; fig.savefig(png, dpi=110)
    print(f"[png] {png}")
except Exception as ex:
    print(f"[png skip] {type(ex).__name__}: {ex}")
