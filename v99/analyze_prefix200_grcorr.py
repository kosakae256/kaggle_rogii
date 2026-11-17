# -*- coding: utf-8 -*-
"""事前区間の"縦区間"を除いた版: 推論開始MDから 200MD 手前までの window で
horizontal GR vs typewell GR の相関(grcorr_tw)を生CSVから再計算し、full-prefix版と比較。
その window版 grcorr が eval精度(井戸別 tw RMSE / oracle) を予測できるか検証。GPU不要。
出力: v98/prefix200_grcorr.png + コンソール。"""
import os, sys, io, glob
try:
    sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
except Exception:
    pass
from pathlib import Path
import numpy as np, pandas as pd, pyarrow.parquet as pq
from scipy.stats import spearmanr, pearsonr

TR = "rogii-wellbore-geology-prediction/train"
PQ = "v97/artifacts_v97/data/train_combo.parquet"
BACK_MD = float(os.environ.get("BACK_MD", "200"))
SM = 5   # GR 軽い平滑


def grcorr(hw_gr, tw_tvt, tw_gr, tvtin):
    """window内: horizontal GR と typewell GR(TVT_input位置で補間) の相関。"""
    m = np.isfinite(hw_gr) & np.isfinite(tvtin)
    if m.sum() < 8:
        return np.nan
    ref = np.interp(tvtin[m], tw_tvt, tw_gr)
    a, b = hw_gr[m], ref
    if np.std(a) < 1e-6 or np.std(b) < 1e-6:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


# ---- 井戸別 eval精度(parquet: tw=pf_ancc_delta* 平均, oracle=各表現min) ----
names = pq.read_schema(PQ).names
DELTA = {
    "tw":         [c for c in names if c == "pf_ancc_delta" or (c.startswith("pf_ancc_delta_") and c.split("_")[-1].isdigit())],
    "self":       [c for c in names if c.startswith("pf_self_delta_") and not c.startswith("pf_self_graft")],
    "nbr":        [c for c in names if c.startswith("pf_nbr_delta_")],
    "nbr_gr5":    [c for c in names if c.startswith("pf_nbr_gr5_delta_")],
    "self_graft": [c for c in names if c.startswith("pf_self_graft_delta_")],
}
allcols = ["well", "target"] + sum(DELTA.values(), [])
df = pq.read_table(PQ, columns=[c for c in dict.fromkeys(allcols) if c in names]).to_pandas()
g = df["well"].to_numpy(); y = df["target"].to_numpy(np.float64)
rmse = {}
for w, ix in ((w, np.where(g == w)[0]) for w in pd.unique(g)):
    yy = y[ix]; m = np.isfinite(yy)
    if m.sum() < 20:
        continue
    rr = {}
    for r, dc in DELTA.items():
        if dc:
            pred = df.iloc[ix][dc].to_numpy(np.float64).mean(1)
            e = pred[m] - yy[m]; rr[r] = float(np.sqrt(np.nanmean(e ** 2)))
    rmse[w] = rr
print(f"eval RMSE 集計井={len(rmse)}", flush=True)

# ---- 生CSVから grcorr_tw を full-prefix と 200MD-window で計算 ----
rows = []
wells = sorted(w.split("__")[0] for w in os.listdir(TR) if w.endswith("__horizontal_well.csv"))
for wi, wn in enumerate(wells):
    if wn not in rmse:
        continue
    try:
        hw = pd.read_csv(os.path.join(TR, f"{wn}__horizontal_well.csv"), usecols=["MD", "GR", "TVT_input"])
        tw = pd.read_csv(os.path.join(TR, f"{wn}__typewell.csv"), usecols=["TVT", "GR"]).sort_values("TVT")
    except Exception:
        continue
    md = hw["MD"].to_numpy(float); gr = pd.Series(hw["GR"].to_numpy(float)).rolling(SM, center=True, min_periods=1).mean().to_numpy()
    ti = hw["TVT_input"].to_numpy(float); kn = np.isfinite(ti)
    if kn.sum() < 10 or (~kn).sum() < 5:
        continue
    ev0 = int(np.where(~kn)[0].min()); ev_md = md[ev0]                      # 推論開始MD
    tw_tvt = tw["TVT"].to_numpy(float); tw_gr = tw["GR"].to_numpy(float)
    full = kn.copy()                                                        # 事前区間 全体(縦込み)
    win = kn & (md >= ev_md - BACK_MD)                                      # 直近200MD(水平部)
    rows.append(dict(well=wn,
                     gc_full=grcorr(gr[full], tw_tvt, tw_gr, ti[full]),
                     gc_win=grcorr(gr[win], tw_tvt, tw_gr, ti[win]),
                     nwin=int(win.sum()),
                     rmse_tw=rmse[wn].get("tw", np.nan),
                     rmse_oracle=min(rmse[wn].values()) if rmse[wn] else np.nan,
                     rmse_mean=float(np.mean(list(rmse[wn].values()))) if rmse[wn] else np.nan))
    if (wi + 1) % 200 == 0:
        print(f"  {wi+1}/{len(wells)}", flush=True)
W = pd.DataFrame(rows)
print(f"\n有効井={len(W)}  grcorr平均: full={W['gc_full'].mean():.3f}  200MD窓={W['gc_win'].mean():.3f}  "
      f"(200MD窓 中央値={W['gc_win'].median():.3f} 標準偏差={W['gc_win'].std():.3f})", flush=True)
print(f"200MD窓の平均点数={W['nwin'].mean():.0f}", flush=True)

print("\n=== grcorr が eval精度を予測するか(井戸横断 Spearman, 負=高相関→低誤差) ===", flush=True)
for gcol, gname in [("gc_full", "full-prefix(縦込)"), ("gc_win", f"直近{int(BACK_MD)}MD(水平)")]:
    for ycol, yname in [("rmse_tw", "tw RMSE"), ("rmse_oracle", "oracle RMSE"), ("rmse_mean", "平均RMSE")]:
        x = W[gcol].to_numpy(); yv = W[ycol].to_numpy(); m = np.isfinite(x) & np.isfinite(yv)
        sp = spearmanr(x[m], yv[m])[0]; pr = pearsonr(x[m], yv[m])[0]
        print(f"  {gname:20s} vs {yname:12s}: Spearman={sp:+.3f} Pearson={pr:+.3f} (n={m.sum()})", flush=True)

# 200MD窓 grcorr 五分位 → 平均 tw RMSE
print(f"\n=== 直近{int(BACK_MD)}MD grcorr 五分位 → 平均 tw RMSE(単調低下なら予測力) ===", flush=True)
x = W["gc_win"].to_numpy(); yv = W["rmse_tw"].to_numpy(); m = np.isfinite(x) & np.isfinite(yv)
xb, yb = x[m], yv[m]; q = np.quantile(xb, [0, .2, .4, .6, .8, 1.0])
for i in range(5):
    s = (xb >= q[i]) & (xb <= q[i + 1]); print(f"  Q{i+1} grcorr[{q[i]:+.2f},{q[i+1]:+.2f}]: 平均twRMSE={yb[s].mean():.3f} (n={s.sum()})", flush=True)

try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "Meiryo"
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    ax[0].hist(W["gc_full"].dropna(), bins=40, alpha=0.6, label="full(縦込)")
    ax[0].hist(W["gc_win"].dropna(), bins=40, alpha=0.6, label=f"直近{int(BACK_MD)}MD")
    ax[0].set_title("grcorr分布(縦除くと散らばる?)"); ax[0].set_xlabel("grcorr_tw"); ax[0].legend()
    for gcol, lab in [("gc_full", "full"), ("gc_win", f"{int(BACK_MD)}MD")]:
        x = W[gcol]; yv = W["rmse_tw"]; m = np.isfinite(x) & np.isfinite(yv)
        ax[1].scatter(x[m], yv[m], s=7, alpha=0.35, label=lab)
    ax[1].set_ylim(0, 25); ax[1].set_xlabel("grcorr_tw"); ax[1].set_ylabel("tw eval RMSE"); ax[1].set_title("grcorr vs eval精度"); ax[1].legend()
    q = np.quantile(xb, np.linspace(0, 1, 6)); mids = [(q[i]+q[i+1])/2 for i in range(5)]; means = [yb[(xb>=q[i])&(xb<=q[i+1])].mean() for i in range(5)]
    ax[2].plot(mids, means, "o-"); ax[2].set_xlabel(f"直近{int(BACK_MD)}MD grcorr 五分位"); ax[2].set_ylabel("平均 tw RMSE"); ax[2].set_title("水平prefix grcorr↑で精度↑か")
    fig.tight_layout(); fig.savefig("v98/prefix200_grcorr.png", dpi=110); print("\nsaved v98/prefix200_grcorr.png", flush=True)
except Exception as e:
    print("plot skip:", e, flush=True)
