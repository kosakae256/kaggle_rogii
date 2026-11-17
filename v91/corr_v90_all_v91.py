# -*- coding: utf-8 -*-
"""全773井: 物理候補選択(r0錨のみ / r0錨+GR) vs v90本番。U-Net distはf0のみなので全井ではU-Net抜き。
   目的: 易しい155井でなく全井で、物理正則化がPF暴走を救い v90と脱相関するか。"""
import os, sys, io, pickle, time
try: sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
except Exception: pass
import numpy as np, pandas as pd, pyarrow.parquet as pq
from pathlib import Path
PROJ = Path(r"C:\Users\kosaka256\Documents\rogii_claude")
TR = PROJ / "rogii-wellbore-geology-prediction" / "train"
AV = PROJ / "v89" / "artifacts_v89"; A90 = PROJ / "v90" / "artifacts_v90"
N = 5000; K = 4; SLOPE = 55/1000.; DEV = 25/1000.; DIPSP = 20/1000.; CONT = 4.0
W_GR = float(os.environ.get("W_GR", "0.2")); W_NN = 0.62
rng = np.random.default_rng(0)


def gen_cands(md, z, tvt_last, z_last, t_min, t_max, pmd, pch):
    L = len(md); mr = md - md[0]; dZ = z - z_last
    pdip = float(np.clip(np.polyfit(pmd, pch, 1)[0], -SLOPE, SLOPE)) if (len(pmd) >= 5 and (pmd[-1]-pmd[0]) > 1) else 0.0
    xk = np.linspace(0.0, mr[-1]+1e-6, K); seg = np.diff(xk)
    dip_n = pdip + rng.uniform(-DIPSP, DIPSP, size=(N, 1))
    dyk = np.concatenate([np.zeros((N, 1)), np.cumsum(rng.normal(0, DEV, (N, K-1))*seg[None, :], axis=1)], axis=1)
    dev = np.empty((N, L))
    for i in range(N): dev[i] = np.interp(mr, xk, dyk[i])
    cand = tvt_last + dip_n*mr[None, :] + dev - dZ[None, :]
    ok = ((cand >= t_min-5) & (cand <= t_max+5)).all(axis=1)
    nc = min(30, L); ok &= (np.abs(cand[:, :nc]-tvt_last) < CONT).all(axis=1)
    return cand[ok] if ok.any() else None


print("loading v90 ...", flush=True)
tb = pq.read_table(A90 / "data" / "train.parquet", columns=["well", "target"])
well = tb.column("well").to_numpy(zero_copy_only=False); y = tb.column("target").to_numpy(zero_copy_only=False).astype(np.float64)
gbm = np.load(A90 / "gbm_v41_oof.npy").astype(np.float64)[:len(y)]
nnz = np.load(A90 / "nn_oof_combo3.npz"); nn = np.mean([nnz[k] for k in nnz.files], axis=0)[:len(y)]
v90err_all = ((1 - W_NN) * gbm + W_NN * nn) - y
idx_by_well = {}
for i, w in enumerate(well): idx_by_well.setdefault(w, []).append(i)
PF = dict(np.load(AV / "pf_center_v89.npz"))
zn = lambda v: (v - v.mean()) / (v.std() + 1e-9)
E_snap, E_gr, E_v90, E_r0, nmis, t0 = [], [], [], [], 0, time.time()
wells = [w for w in PF if w in idx_by_well]
for c, wn in enumerate(wells):
    try:
        hw = pd.read_csv(TR / f"{wn}__horizontal_well.csv"); tw = pd.read_csv(TR / f"{wn}__typewell.csv")
    except Exception: continue
    kn = hw["TVT_input"].notna().to_numpy()
    if kn.sum() < 10 or (~kn).sum() < 30: continue
    hps = int(np.where(kn)[0].max()); ev = slice(hps+1, len(hw))
    z = hw["Z"].to_numpy(float); md = hw["MD"].to_numpy(float); tvt = hw["TVT"].to_numpy(float)
    ti = hw["TVT_input"].to_numpy(float); gr = hw["GR"].to_numpy(float)
    t_tvt = tw["TVT"].to_numpy(float); t_gr = tw["GR"].to_numpy(float)
    m = np.isfinite(t_tvt) & np.isfinite(t_gr); t_tvt, t_gr = t_tvt[m], t_gr[m]
    if len(t_tvt) < 8: continue
    o = np.argsort(t_tvt); t_tvt, t_gr = t_tvt[o], t_gr[o]
    tvt_last = float(ti[hps]); z_last = float(z[hps])
    ki = np.where(kn)[0]; ki = ki[ki >= max(0, hps-80)]
    cand = gen_cands(md[ev], z[ev], tvt_last, z_last, float(t_tvt.min()), float(t_tvt.max()),
                     md[ki]-md[hps], (ti[ki]-tvt_last)+(z[ki]-z_last))
    if cand is None: continue
    true = tvt[ev]; fin = np.isfinite(true); L = len(true)
    obs = pd.Series(gr[ev]).interpolate(limit_direction="both").fillna(float(np.nanmean(t_gr))).rolling(5, center=True, min_periods=1).mean().to_numpy()
    gscore = np.mean((np.interp(cand, t_tvt, t_gr)-obs[None, :])**2, axis=1)
    r0 = np.asarray(PF[wn], float)[:L]
    r0d = np.sqrt(np.mean((cand-r0[None, :])**2, axis=1))
    s_r = -zn(r0d)
    snap = cand[int(np.argmax(s_r))]
    grsel = cand[int(np.argmax(s_r + W_GR*(-zn(gscore))))]
    v90e = v90err_all[np.array(idx_by_well[wn])]
    n = min(L, len(v90e)); f = fin[:n]
    if abs(L - len(v90e)) > 3: nmis += 1
    E_snap.append((snap[:n]-true[:n])[f]); E_gr.append((grsel[:n]-true[:n])[f])
    E_r0.append((r0[:n]-true[:n])[f]); E_v90.append(v90e[:n][f])
    if (c+1) % 150 == 0: print(f"  {c+1}/{len(wells)} ({time.time()-t0:.0f}s)", flush=True)

es = np.concatenate(E_snap); eg = np.concatenate(E_gr); er0 = np.concatenate(E_r0); ev = np.concatenate(E_v90)
rmse = lambda e: float(np.sqrt(np.mean(e**2)))
print(f"\n[全{len(E_snap)}井 行対応 (長さ不一致>3井={nmis})]")
print(f"  v90本番          = {rmse(ev):.3f}ft")
print(f"  素PF r0          = {rmse(er0):.3f}ft")
print(f"  物理スナップ(r0錨のみ)   = {rmse(es):.3f}ft   ρ(v90)={np.corrcoef(ev, es)[0,1]:.3f}")
print(f"  物理 r0錨+GR(W={W_GR})   = {rmse(eg):.3f}ft   ρ(v90)={np.corrcoef(ev, eg)[0,1]:.3f}")
print(f"  ρ(v90, 素PF)={np.corrcoef(ev, er0)[0,1]:.3f}")
print("  --- blend v90×物理スナップ ---")
for w in [0.1, 0.2, 0.3]:
    print(f"    w={w}: {rmse((1-w)*ev + w*es):.3f}")
