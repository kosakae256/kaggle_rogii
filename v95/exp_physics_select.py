# -*- coding: utf-8 -*-
"""夜間実験E(CPU): v91物理候補(oracle~3.5)を「今のv95特徴で選ぶ」ランキング問題に。
各候補の特徴=GR一致/dip/blend乖離/PF乖離/plausibility → 候補誤差を学習回帰(well-CV)→最小予測誤差候補を選択。
blendを文脈に含むので最低でもblend復元可。それを超えれば物理選択に価値。真TVTはlabel/評価のみ。
env NW(井数) N(候補) K SEED。出力: v95/artifacts_v95b/exp_physics_select.txt"""
import os, sys, io, time
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
import numpy as np, pandas as pd, pyarrow.parquet as pq
from scipy.optimize import nnls
from sklearn.model_selection import GroupKFold
import lightgbm as lgb
TR = "rogii-wellbore-geology-prediction/train"; B = "v95/artifacts_v95b/"
NW = int(os.environ.get("NW", "150")); N = int(os.environ.get("N", "600")); K = 4
SLOPE = 55 / 1000.0; CONT = 4.0; DEV = 25 / 1000.0; DIPSP = 20 / 1000.0
rng = np.random.default_rng(0)
# ---- v95 blend(絶対TVT) を parquet から, well行順 ----
tb = pq.read_table(B + "data/train.parquet", columns=["well", "target", "last_known_tvt",
        "pf_ancc_pstd" if "pf_ancc_pstd" in set(pq.read_schema(B+'data/train.parquet').names) else "pf_agree_std_1"]).to_pandas()
sch = set(pq.read_schema(B + "data/train.parquet").names)
g = tb["well"].astype(str).to_numpy(); yv = tb["target"].to_numpy(float); lk = tb["last_known_tvt"].to_numpy(float)
d1 = np.load(B.replace("v95b", "v95") + "l1_oof.npz"); gbm = np.mean([d1[k] for k in d1.files if k not in ("y", "well")], 0)
tcn = np.mean([v for k, v in np.load(B + "nn_oof_combo.npz").items() if k.startswith("mdn2")], 0)
l5 = np.load(B + "l5_ext_oof.npz")["l5_mean"].astype(float)
Mm = np.stack([gbm, tcn, l5], 1); mf = np.isfinite(Mm).all(1) & np.isfinite(yv); wj, _ = nnls(Mm[mf], yv[mf]); wj /= wj.sum()
blend_tvt = lk + Mm @ wj
rows_of = {}
for w in pd.unique(g): rows_of[w] = np.where(g == w)[0]

def gen(md, z, tvt_last, z_last, t_min, t_max, pmd_rel, pchange):
    L = len(md); md_rel = md - md[0]; dZ = z - z_last
    pdip = float(np.clip(np.polyfit(pmd_rel, pchange, 1)[0], -SLOPE, SLOPE)) if len(pmd_rel) >= 5 and pmd_rel[-1] - pmd_rel[0] > 1 else 0.0
    xk = np.linspace(0.0, md_rel[-1] + 1e-6, K); seg = np.diff(xk)
    dip_n = pdip + rng.uniform(-DIPSP, DIPSP, size=(N, 1))
    dincr = rng.normal(0.0, DEV, size=(N, K - 1)) * seg[None, :]
    dyk = np.concatenate([np.zeros((N, 1)), np.cumsum(dincr, 1)], 1)   # (N,K)
    ix = np.clip(np.searchsorted(xk, md_rel) - 1, 0, K - 2)           # (L,) 区分線形をベクトル化
    wgt = (md_rel - xk[ix]) / np.maximum(xk[ix + 1] - xk[ix], 1e-9)
    dev = dyk[:, ix] * (1 - wgt)[None, :] + dyk[:, ix + 1] * wgt[None, :]  # (N,L)
    cand = tvt_last + dip_n * md_rel[None, :] + dev - dZ[None, :]
    ok = ((cand >= t_min - 5) & (cand <= t_max + 5)).all(1)
    nc = min(30, L); ok &= (np.abs(cand[:, :nc] - tvt_last) < CONT).all(1)
    return (cand[ok] if ok.any() else None), pdip

wells = sorted(w.split("__")[0] for w in os.listdir(TR) if w.endswith("__horizontal_well.csv"))[:NW]
ROWS = []; t0 = time.time()
for wi, wn in enumerate(wells):
    try:
        hw = pd.read_csv(os.path.join(TR, f"{wn}__horizontal_well.csv")); tw = pd.read_csv(os.path.join(TR, f"{wn}__typewell.csv"))
    except Exception: continue
    if wn not in rows_of: continue
    kn = hw["TVT_input"].notna().to_numpy()
    if kn.sum() < 10 or (~kn).sum() < 30: continue
    hps = int(np.where(kn)[0].max()); ev = slice(hps + 1, len(hw))
    z = hw["Z"].to_numpy(float); md = hw["MD"].to_numpy(float); tvt = hw["TVT"].to_numpy(float)
    ti = hw["TVT_input"].to_numpy(float); gr = hw["GR"].to_numpy(float)
    t_tvt = tw["TVT"].to_numpy(float); t_gr = tw["GR"].to_numpy(float)
    m = np.isfinite(t_tvt) & np.isfinite(t_gr); t_tvt, t_gr = t_tvt[m], t_gr[m]
    if len(t_tvt) < 8: continue
    o = np.argsort(t_tvt); t_tvt, t_gr = t_tvt[o], t_gr[o]
    tvt_last = float(ti[hps]); z_last = float(z[hps])
    ki = np.where(kn)[0]; ki = ki[ki >= max(0, hps - 80)]
    pmd = md[ki] - md[hps]; pch = (ti[ki] - tvt_last) + (z[ki] - z_last)
    r = rows_of[wn]; L = len(hw.iloc[ev])
    if len(r) != L: continue                                  # parquet行と一致必須
    cand, pdip = gen(md[ev], z[ev], tvt_last, z_last, float(t_tvt.min()), float(t_tvt.max()), pmd, pch)
    if cand is None: continue
    tr = tvt[ev]; bl = blend_tvt[r]; zev = z[ev]; mdrel = md[ev] - md[ev][0]
    obs = pd.Series(gr[ev]).interpolate(limit_direction="both").fillna(float(np.nanmean(t_gr))).rolling(5, center=True, min_periods=1).mean().to_numpy()
    grp = np.interp(cand, t_tvt, t_gr)
    gr_score = np.mean((grp - obs[None, :]) ** 2, 1)          # GR残差(小=良)
    dip = (cand[:, -1] - cand[:, 0]) / max(mdrel[-1], 1)
    dev_bl = cand - bl[None, :]; dev_bl_m = np.mean(dev_bl, 1); dev_bl_s = np.std(dev_bl, 1)
    rmse_bl = np.sqrt(np.mean(dev_bl ** 2, 1))               # ★blend近接度(RMSE)
    cand_rough = np.mean(np.abs(np.diff(cand, 2, axis=1)), 1)
    cerr = np.sqrt(np.mean((cand - tr[None, :]) ** 2, 1))     # ★label=候補の真誤差
    for j in range(len(cand)):
        ROWS.append((wn, gr_score[j], dip[j], dev_bl_m[j], dev_bl_s[j], rmse_bl[j], cand_rough[j],
                     float(np.mean(cand[j])), float(np.std(cand[j])), pdip, cerr[j], j))
    if (wi + 1) % 50 == 0: print(f"  gen {wi+1}/{len(wells)} ({time.time()-t0:.0f}s)", flush=True)

F = pd.DataFrame(ROWS, columns=["well", "gr_score", "dip", "dev_bl_m", "dev_bl_s", "rmse_bl", "rough",
                                "cand_mean", "cand_std", "pdip", "cerr", "cidx"])
print(f"候補行={len(F)} 井={F['well'].nunique()}", flush=True)
FEATS = ["gr_score", "dip", "dev_bl_m", "dev_bl_s", "rmse_bl", "rough", "cand_mean", "cand_std", "pdip"]
X = F[FEATS].to_numpy(np.float32); yv2 = F["cerr"].to_numpy(); grp = F["well"].to_numpy()
pred = np.zeros(len(F))
for tr_, va in GroupKFold(5).split(X, yv2, grp):
    m = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.04, num_leaves=31, min_child_samples=50,
                          subsample=0.8, colsample_bytree=0.7, device="cpu", n_jobs=-1, verbose=-1)
    m.fit(X[tr_], yv2[tr_]); pred[va] = m.predict(X[va])
F["pred"] = pred
# 各井: 選択方式ごとに代表候補→pooled RMSE(候補のcerrを重み=行数で近似: cerrは既にwell内RMSE)
# pooled近似: 各wellの選択候補のcerr を well行数で加重
wl = {w: len(rows_of[w]) for w in F["well"].unique()}
def pooled(colsel):
    num = 0.0; den = 0
    for w, sub in F.groupby("well"):
        j = colsel(sub); e = sub["cerr"].to_numpy()[j]; n = wl[w]
        num += (e ** 2) * n; den += n
    return float(np.sqrt(num / den))
r_orac = pooled(lambda s: int(np.argmin(s["cerr"].to_numpy())))
r_gr = pooled(lambda s: int(np.argmin(s["gr_score"].to_numpy())))
r_learn = pooled(lambda s: int(np.argmin(s["pred"].to_numpy())))
r_nearbl = pooled(lambda s: int(np.argmin(np.abs(s["dev_bl_m"].to_numpy()))))
# blend自体のpooled(この井集合)
be = []
for w in F["well"].unique():
    r = rows_of[w]; be.append(np.sqrt(np.mean((blend_tvt[r] - (lk[r] + yv[r])) ** 2)) ** 2 * wl[w])
r_blend = float(np.sqrt(sum(be) / sum(wl.values())))
out = ["=== 実験E: 物理候補を今のv95特徴で選択 (ユーザー案) ===",
 f"  井={F['well'].nunique()} 候補/井中央={int(F.groupby('well').size().median())}",
 f"  blend(この井集合)        = {r_blend:.4f}",
 f"  ★oracle(候補内最良)      = {r_orac:.4f}  (答えは候補内=確認)",
 f"  GRマッチ選択(v91)        = {r_gr:.4f}",
 f"  blend最近傍候補          = {r_nearbl:.4f}  (blend復元の下限)",
 f"  ★★学習選択(今の特徴)     = {r_learn:.4f}  (blend {r_learn-r_blend:+.4f} / oracle {r_learn-r_orac:+.4f})",
 f"  → 学習選択がblendを超えれば物理選択に価値。blend近傍止まりなら選択の壁(方向不能)再確認。"]
txt = "\n".join(out); print("\n" + txt, flush=True); open(B + "exp_physics_select.txt", "w", encoding="utf-8").write(txt)
