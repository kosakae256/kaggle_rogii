# -*- coding: utf-8 -*-
"""物理候補(GRフリー, 区分線形dip, prefix-dip中心)を各井で生成し、代表軌跡(mean/med/q25/q75)+広がり(std)を
combo parquet の eval行に整列して delta列で保存。L5候補トークンに足す用。CPU(GPU不要)。
出力: v97/artifacts_v97/phys_features.npz (行=combo parquet行, keys=phys_mean_d/phys_med_d/phys_q25_d/phys_q75_d/phys_std_d)"""
import os, sys, io, time
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
import numpy as np, pandas as pd, pyarrow.parquet as pq
TR = "rogii-wellbore-geology-prediction/train"
PQC = "v97/artifacts_v97/data/train_combo.parquet"
N = 3000; K = 4; SEEDS = [0, 1, 2]
SLOPE = 55 / 1000.0; CONT = 4.0; DEV = 25 / 1000.0; DIPSP = 20 / 1000.0


def gen(md, z, tvt_last, z_last, t_min, t_max, pmd, pch, rng):   # snap_leg_v95 の物理候補生成(GRフリー)
    L = len(md); md_rel = md - md[0]; dZ = z - z_last
    pdip = float(np.clip(np.polyfit(pmd, pch, 1)[0], -SLOPE, SLOPE)) if len(pmd) >= 5 and pmd[-1] - pmd[0] > 1 else 0.0
    xk = np.linspace(0.0, md_rel[-1] + 1e-6, K); seg = np.diff(xk)
    dip_n = pdip + rng.uniform(-DIPSP, DIPSP, size=(N, 1))
    dyk = np.concatenate([np.zeros((N, 1)), np.cumsum(rng.normal(0, DEV, (N, K - 1)) * seg[None, :], 1)], 1)
    ix = np.clip(np.searchsorted(xk, md_rel) - 1, 0, K - 2); wg = (md_rel - xk[ix]) / np.maximum(xk[ix + 1] - xk[ix], 1e-9)
    dev = dyk[:, ix] * (1 - wg)[None, :] + dyk[:, ix + 1] * wg[None, :]
    cand = tvt_last + dip_n * md_rel[None, :] + dev - dZ[None, :]
    ok = ((cand >= t_min - 5) & (cand <= t_max + 5)).all(1); nc = min(30, L); ok &= (np.abs(cand[:, :nc] - tvt_last) < CONT).all(1)
    return cand[ok] if ok.any() else None


tb = pq.read_table(PQC, columns=["well", "last_known_tvt"]).to_pandas()
g = tb["well"].astype(str).to_numpy(); lk = tb["last_known_tvt"].to_numpy(float)
rows_of = {w: np.where(g == w)[0] for w in pd.unique(g)}
Nrow = len(g)
out = {k: np.full(Nrow, np.nan, np.float32) for k in ["phys_mean_d", "phys_med_d", "phys_q25_d", "phys_q75_d", "phys_std_d"]}
wells = sorted(w.split("__")[0] for w in os.listdir(TR) if w.endswith("__horizontal_well.csv"))
t0 = time.time(); nused = 0
for wi, wn in enumerate(wells):
    if wn not in rows_of: continue
    try:
        hw = pd.read_csv(os.path.join(TR, f"{wn}__horizontal_well.csv")); tw = pd.read_csv(os.path.join(TR, f"{wn}__typewell.csv"))
    except Exception: continue
    kn = hw["TVT_input"].notna().to_numpy()
    if kn.sum() < 10 or (~kn).sum() < 10: continue
    hps = int(np.where(kn)[0].max()); ev = slice(hps + 1, len(hw))
    z = hw["Z"].to_numpy(float); md = hw["MD"].to_numpy(float); ti = hw["TVT_input"].to_numpy(float)
    tt = tw["TVT"].to_numpy(float); m = np.isfinite(tt); t_min, t_max = float(tt[m].min()), float(tt[m].max())
    tvt_last = float(ti[hps]); z_last = float(z[hps])
    ki = np.where(kn)[0]; ki = ki[ki >= max(0, hps - 80)]; pmd = md[ki] - md[hps]; pch = (ti[ki] - tvt_last) + (z[ki] - z_last)
    r = rows_of[wn]; L = len(hw.iloc[ev])
    if len(r) != L: continue
    cands = []
    for sd in SEEDS:
        c = gen(md[ev], z[ev], tvt_last, z_last, t_min, t_max, pmd, pch, np.random.default_rng(sd))
        if c is not None: cands.append(c)
    if not cands: continue
    C = np.concatenate(cands, 0)                     # (Ncand, L) 絶対TVT
    lkr = lk[r]                                        # last_known (L,)
    out["phys_mean_d"][r] = (C.mean(0) - lkr).astype(np.float32)
    out["phys_med_d"][r] = (np.median(C, 0) - lkr).astype(np.float32)
    out["phys_q25_d"][r] = (np.quantile(C, 0.25, 0) - lkr).astype(np.float32)
    out["phys_q75_d"][r] = (np.quantile(C, 0.75, 0) - lkr).astype(np.float32)
    out["phys_std_d"][r] = C.std(0).astype(np.float32)
    nused += 1
    if (wi + 1) % 100 == 0: print(f"  {wi+1}/{len(wells)} used={nused} ({time.time()-t0:.0f}s)", flush=True)
np.savez(os.path.join("v97/artifacts_v97", "phys_features.npz"), **out)
cov = np.isfinite(out["phys_mean_d"]).mean()
print(f"[phys] 保存 {nused}井 被覆={cov*100:.0f}%  oracle参考: 物理候補は個別GRフリー", flush=True)
# 参考: phys_med 単体の精度(delta vs target)
y = pq.read_table(PQC, columns=["target"]).to_pandas()["target"].to_numpy(float)
for k in ["phys_mean_d", "phys_med_d"]:
    e = out[k] - y; e = e[np.isfinite(e)]; print(f"  {k} 単体RMSE={np.sqrt(np.mean(e**2)):.3f}", flush=True)
