# -*- coding: utf-8 -*-
"""物理候補選択(r0錨+U-Net) と 素PF(r0) の誤差相関 + blend効果を測る。fold0 155井。"""
import os, sys, io, pickle
try: sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
except Exception: pass
import numpy as np, pandas as pd
from pathlib import Path
PROJ = Path(r"C:\Users\kosaka256\Documents\rogii_claude")
TR = PROJ / "rogii-wellbore-geology-prediction" / "train"
A = PROJ / "v64" / "artifacts_v64"; AV = PROJ / "v89" / "artifacts_v89"
N = 5000; K = 4; SLOPE = 55/1000.; DEV = 25/1000.; DIPSP = 20/1000.; CONT = 4.0
W_EV = 0.2; NB = 90; ND = 181; STEP = 0.5
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


DIST = dict(np.load(A / "v89_dist_f0_dist.npz"))
EVD = pickle.loads(open(AV / "learned_boot.pkl", "rb").read())
PF = dict(np.load(AV / "pf_center_v89.npz"))
zn = lambda v: (v - v.mean()) / (v.std() + 1e-9)
E_r0, E_ref, E_u, per, peru = [], [], [], [], []
for wn in [w for w in DIST if w in EVD and w in PF]:
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
    st = np.asarray(EVD[wn]["st"], float); bid = np.asarray(EVD[wn]["bid"], int); P = np.asarray(DIST[wn], float)
    Ln = min(L, len(st), len(bid)); delta = cand[:, :Ln]-st[None, :Ln]
    j = np.round(delta/STEP+NB); jf = np.clip(j.astype(int), 0, ND-1)
    uscore = np.where((j >= 0) & (j < ND), np.log(P[bid[None, :Ln], jf]+1e-6), -12.0).sum(axis=1)
    obs = pd.Series(gr[ev]).interpolate(limit_direction="both").fillna(float(np.nanmean(t_gr))).rolling(5, center=True, min_periods=1).mean().to_numpy()
    gscore = np.mean((np.interp(cand, t_tvt, t_gr)-obs[None, :])**2, axis=1)
    r0 = np.asarray(PF[wn], float)[:L]
    r0d = np.sqrt(np.mean((cand-r0[None, :])**2, axis=1))
    ref = cand[int(np.argmax(-zn(r0d) + W_EV*(-zn(gscore) + zn(uscore))))]  # r0錨+証拠
    uonly = cand[int(np.argmax(zn(uscore)))]                                 # U-Net単独(r0不使用)
    er0 = (r0-true)[fin]; eref = (ref-true)[fin]; eu = (uonly-true)[fin]
    E_r0.append(er0); E_ref.append(eref); E_u.append(eu)
    per.append(np.corrcoef(er0, eref)[0, 1]); peru.append(np.corrcoef(er0, eu)[0, 1])

er0 = np.concatenate(E_r0); eref = np.concatenate(E_ref); eu = np.concatenate(E_u)
rmse = lambda e: float(np.sqrt(np.mean(e**2)))
pc = np.array(per); pu = np.array(peru)
print(f"--- A) r0錨+U-Net証拠 (7.49) ---")
print(f"  ρ(r0, 物理選択)= 全行{np.corrcoef(er0, eref)[0,1]:.3f} / 井戸別中央{np.median(pc):.3f}")
print(f"  blend w=0.5 → {rmse(0.5*er0 + 0.5*eref):.3f}")
print(f"--- B) U-Net単独選択 (r0不使用, 脱相関のはず) ---")
print(f"  単体 = {rmse(eu):.3f}")
print(f"  ρ(r0, U-Net単独)= 全行{np.corrcoef(er0, eu)[0,1]:.3f} / 井戸別中央{np.median(pu):.3f}")
for w in [0.2, 0.3, 0.5]:
    print(f"  blend w_uonly={w} → {rmse((1-w)*er0 + w*eu):.3f}")
