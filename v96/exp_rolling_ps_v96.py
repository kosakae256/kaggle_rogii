# -*- coding: utf-8 -*-
"""v96 実験1: Rolling pseudo-PS の基盤検証(CPU)。
核心: prefixを疑似PS分割し、既知TVTで「GR照合 vs 安全外挿」をbacktest。
そのbacktest結果が suffix の難度(blend RMSE)を予測できるか＝本番前に井戸別confidenceを得られるか。
できれば「どの井戸でGRを信じるか」を prefix から井戸別決定=Rolling pseudo-PS の土台。真TVTは評価/prefix既知のみ。
出力: v96/exp_rolling_ps.txt"""
import os, sys, io, time
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
from pathlib import Path
import numpy as np, pandas as pd, pyarrow.parquet as pq
from scipy.optimize import nnls
from scipy.ndimage import gaussian_filter1d
TR = "rogii-wellbore-geology-prediction/train"; B = "v95/artifacts_v95b/"
Path("v96").mkdir(exist_ok=True)
CALFRAC = 0.6           # prefix の前60%=較正(self-template+dip)、後40%=backtest
DR = 30.0; delta = np.arange(-DR, DR + 1e-6, 1.0); D = len(delta); TSIG = 2.0

def build_grid(tvt, gr, step=1.0):
    m = np.isfinite(tvt) & np.isfinite(gr); tvt, gr = tvt[m], gr[m]
    if len(tvt) < 5: return None
    edges = np.arange(tvt.min(), tvt.max() + step, step)
    if len(edges) < 3: return None
    idx = np.clip(np.digitize(tvt, edges) - 1, 0, len(edges) - 2)
    gg = np.array([np.median(gr[idx == b]) if (idx == b).any() else np.nan for b in range(len(edges) - 1)])
    ax = (edges[:-1] + edges[1:]) / 2; ok = np.isfinite(gg)
    return (ax[ok], gg[ok]) if ok.sum() >= 5 else None

def gridmatch(obs, ax, gg, anc):
    T = len(obs); pos = anc[:, None] + delta[None, :]
    ref = np.interp(pos.ravel(), ax, gg).reshape(T, D)
    o = (obs - np.nanmean(obs)) / (np.nanstd(obs) + 1e-6); rz = (ref - np.nanmean(ref)) / (np.nanstd(ref) + 1e-6)
    E = np.clip(np.exp(-0.5 * np.clip(((o[:, None] - rz)) ** 2, 0, 60)), 1e-12, None)
    f = E[0] / E[0].sum(); fwd = [f]
    for i in range(1, T):
        f = gaussian_filter1d(f, TSIG, mode="nearest") * E[i]; s = f.sum(); f = f / s if s > 1e-300 else np.full(D, 1.0 / D); fwd.append(f)
    b = np.ones(D) / D; bwd = [None] * T; bwd[T - 1] = b
    for i in range(T - 2, -1, -1):
        b = gaussian_filter1d(b * E[i + 1], TSIG, mode="nearest"); s = b.sum(); b = b / s if s > 1e-300 else np.full(D, 1.0 / D); bwd[i] = b
    post = np.array(fwd) * np.array(bwd); post /= post.sum(1, keepdims=True) + 1e-300
    return anc + (post * delta[None, :]).sum(1)

# blend(suffix) per well
tb = pq.read_table(B + "data/train.parquet", columns=["well", "target", "last_known_tvt"]).to_pandas()
g = tb["well"].astype(str).to_numpy(); yv = tb["target"].to_numpy(float); lk = tb["last_known_tvt"].to_numpy(float)
d1 = np.load(B.replace("v95b", "v95") + "l1_oof.npz"); gbm = np.mean([d1[k] for k in d1.files if k not in ("y", "well")], 0)
tcn = np.mean([v for k, v in np.load(B + "nn_oof_combo.npz").items() if k.startswith("mdn2")], 0)
l5 = np.load(B + "l5_ext_oof.npz")["l5_mean"].astype(float)
Mm = np.stack([gbm, tcn, l5], 1); mf = np.isfinite(Mm).all(1) & np.isfinite(yv); wj, _ = nnls(Mm[mf], yv[mf]); wj /= wj.sum(); blend = Mm @ wj
rows_of = {w: np.where(g == w)[0] for w in pd.unique(g)}
suf_rmse = {w: float(np.sqrt(np.nanmean((blend[r] - yv[r]) ** 2))) for w, r in rows_of.items()}

wells = sorted(w.split("__")[0] for w in os.listdir(TR) if w.endswith("__horizontal_well.csv"))
rows = []; t0 = time.time()
for wi, wn in enumerate(wells):
    if wn not in rows_of: continue
    try: hw = pd.read_csv(os.path.join(TR, f"{wn}__horizontal_well.csv"))
    except Exception: continue
    kn = hw["TVT_input"].notna().to_numpy(); ki = np.where(kn)[0]
    if len(ki) < 60: continue
    md = hw["MD"].to_numpy(float); z = hw["Z"].to_numpy(float); gr = hw["GR"].to_numpy(float); ti = hw["TVT_input"].to_numpy(float)
    ki = ki[np.argsort(md[ki])]                                  # prefix行 MD順
    nc = int(len(ki) * CALFRAC); cal = ki[:nc]; bt = ki[nc:]
    if len(bt) < 20 or len(cal) < 30: continue
    # 較正: self-template + prefix dip(change slope)
    tmpl = build_grid(ti[cal], gr[cal])
    if tmpl is None: continue
    ml = md[cal] - md[cal][-1]; ch = (ti[cal] - ti[cal[-1]]) + (z[cal] - z[cal[-1]])   # 地質change
    dip = float(np.clip(np.polyfit(ml, ch, 1)[0], -0.055, 0.055)) if len(cal) >= 5 else 0.0
    # backtest 区間: 安全外挿 anchor + GR照合
    md_b = md[bt] - md[cal][-1]; dZ = z[bt] - z[cal[-1]]
    anc = ti[cal[-1]] + dip * md_b - dZ                          # 安全外挿(GRなし)
    obs_b = pd.Series(gr[bt]).interpolate(limit_direction="both").fillna(np.nanmean(gr)).to_numpy()
    matched = gridmatch(obs_b, tmpl[0], tmpl[1], anc)
    true_b = ti[bt]
    bt_safe = float(np.sqrt(np.mean((anc - true_b) ** 2)))
    bt_gr = float(np.sqrt(np.mean((matched - true_b) ** 2)))
    rows.append(dict(well=wn, bt_safe=bt_safe, bt_gr=bt_gr, gr_helps=bt_safe - bt_gr,
                     suf=suf_rmse[wn], n_bt=len(bt)))
    if (wi + 1) % 100 == 0: print(f"  {wi+1}/{len(wells)} ({time.time()-t0:.0f}s)", flush=True)
W = pd.DataFrame(rows)
out = ["=== v96実験1: Rolling pseudo-PS 基盤検証 ==="]
out.append(f"  対象{len(W)}井(prefix十分). backtest中央{W.n_bt.median():.0f}行")
out.append(f"  backtest: 安全外挿RMSE中央={W.bt_safe.median():.2f} / GR照合RMSE中央={W.bt_gr.median():.2f}")
out.append(f"  GRがbacktestで安全外挿に勝つ井: {(W.gr_helps>0).mean()*100:.0f}%")
# ★核心: prefix backtest(bt_gr, gr_helps)が suffix難度(suf)を予測するか
for c in ["bt_gr", "bt_safe", "gr_helps"]:
    m = np.isfinite(W[c]) & np.isfinite(W.suf); rho = np.corrcoef(W[c][m], W.suf[m])[0, 1]
    out.append(f"  ρ({c}, suffix難度) = {rho:+.3f}")
# 硬い井(suf上位20%)を backtest で予測できるか
hard = W.suf >= W.suf.quantile(0.8)
from sklearn.metrics import roc_auc_score
try: auc = roc_auc_score(hard.astype(int), W.bt_gr)
except Exception: auc = np.nan
out.append(f"  ★ハード井(suf上位20%)を bt_gr で検出 AUC = {auc:.3f}  (>0.7なら本番前に井戸別confidence可能)")
out.append("  → ρ/AUCが高ければ Rolling pseudo-PS で本番前に井戸別に'GRを信じるか'を決定でき、適応の土台成立。")
txt = "\n".join(out); print("\n" + txt, flush=True); open("v96/exp_rolling_ps.txt", "w", encoding="utf-8").write(txt)
