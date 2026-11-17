# -*- coding: utf-8 -*-
"""v91×v89 approach2 可視化: W_EV掃引曲線 + 井戸別RMSE(r0 vs 物理錨精緻化)ECDF + 例軌跡。"""
import os, sys, io, pickle
try: sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
except Exception: pass
import numpy as np, pandas as pd
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Meiryo"
PROJ = Path(r"C:\Users\kosaka256\Documents\rogii_claude")
TR = PROJ / "rogii-wellbore-geology-prediction" / "train"
A = PROJ / "v64" / "artifacts_v64"; AV = PROJ / "v89" / "artifacts_v89"
N = 5000; K = 4; SLOPE = 55/1000.; DEV = 25/1000.; DIPSP = 20/1000.; CONT = 4.0
W_EV = 0.2; NB = 90; ND = 181; STEP = 0.5
rng = np.random.default_rng(0)


def gen_cands(md, z, tvt_last, z_last, t_min, t_max, pmd_rel, pchange):
    L = len(md); md_rel = md - md[0]; dZ = z - z_last
    pdip = float(np.clip(np.polyfit(pmd_rel, pchange, 1)[0], -SLOPE, SLOPE)) if (len(pmd_rel) >= 5 and (pmd_rel[-1]-pmd_rel[0]) > 1) else 0.0
    xk = np.linspace(0.0, md_rel[-1]+1e-6, K); seg = np.diff(xk)
    dip_n = pdip + rng.uniform(-DIPSP, DIPSP, size=(N, 1))
    dyk = np.concatenate([np.zeros((N, 1)), np.cumsum(rng.normal(0, DEV, (N, K-1))*seg[None, :], axis=1)], axis=1)
    dev = np.empty((N, L))
    for i in range(N): dev[i] = np.interp(md_rel, xk, dyk[i])
    cand = tvt_last + dip_n*md_rel[None, :] + dev - dZ[None, :]
    ok = ((cand >= t_min-5) & (cand <= t_max+5)).all(axis=1)
    nc = min(30, L); ok &= (np.abs(cand[:, :nc]-tvt_last) < CONT).all(axis=1)
    if not ok.any(): return None
    return cand[ok]


DIST = dict(np.load(A / "v89_dist_f0_dist.npz"))
EVD = pickle.loads(open(AV / "learned_boot.pkl", "rb").read())
PF = dict(np.load(AV / "pf_center_v89.npz"))
wells = [w for w in DIST if w in EVD and w in PF]
zn = lambda v: (v - v.mean()) / (v.std() + 1e-9)
rows = []  # (wn, rmse_r0, rmse_ref, r0traj, reftraj, true, md)
for wn in wells:
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
    pmd = md[ki]-md[hps]; pch = (ti[ki]-tvt_last)+(z[ki]-z_last)
    cand = gen_cands(md[ev], z[ev], tvt_last, z_last, float(t_tvt.min()), float(t_tvt.max()), pmd, pch)
    if cand is None: continue
    true = tvt[ev]; fin = np.isfinite(true); L = len(true)
    st = np.asarray(EVD[wn]["st"], float); bid = np.asarray(EVD[wn]["bid"], int)
    P = np.asarray(DIST[wn], float); Ln = min(L, len(st), len(bid))
    delta = cand[:, :Ln]-st[None, :Ln]; jf = np.clip(np.round(delta/STEP+NB).astype(int), 0, ND-1)
    inb = (np.round(delta/STEP+NB) >= 0) & (np.round(delta/STEP+NB) < ND)
    logp = np.where(inb, np.log(P[bid[None, :Ln], jf]+1e-6), -12.0); uscore = logp.sum(axis=1)
    obs = pd.Series(gr[ev]).interpolate(limit_direction="both").fillna(float(np.nanmean(t_gr))).rolling(5, center=True, min_periods=1).mean().to_numpy()
    gpred = np.interp(cand, t_tvt, t_gr); gscore = np.mean((gpred-obs[None, :])**2, axis=1)
    r0 = np.asarray(PF[wn], float)[:L]
    r0d = np.sqrt(np.mean((cand-r0[None, :])**2, axis=1))
    comb = -zn(r0d) + W_EV*(-zn(gscore) + zn(uscore))
    ref = cand[int(np.argmax(comb))]
    rr = lambda t: float(np.sqrt(np.mean(((t-true)[fin])**2)))
    rows.append((wn, rr(r0), rr(ref), r0, ref, true, md[ev]))

r0r = np.array([x[1] for x in rows]); refr = np.array([x[2] for x in rows])
print(f"{len(rows)}井  井平均RMSE  r0={r0r.mean():.3f}  ref={refr.mean():.3f}  改善井={int((refr<r0r-0.05).sum())}  悪化井={int((refr>r0r+0.05).sum())}")

fig, ax = plt.subplots(1, 3, figsize=(16, 5))
# (1) W_EV掃引
we = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0]; ov = [8.032, 7.803, 7.493, 7.740, 8.218, 8.617, 9.528]  # 0.0=r0スナップ7.95近似; overall
ax[0].plot(we, ov, "o-", color="#1f77b4"); ax[0].axhline(8.032, ls="--", c="gray", label="r0素 8.03")
ax[0].scatter([0.2], [7.493], s=120, c="red", zorder=5, label="最適 W_EV=0.2")
ax[0].set_xlabel("W_EV (証拠の重み)"); ax[0].set_ylabel("overall RMSE (ft)"); ax[0].set_title("W_EV掃引 (物理錨+GR/U-Net証拠)"); ax[0].legend()
# (2) 井戸別RMSE ECDF
for arr, lab, c in [(r0r, "r0素", "gray"), (refr, "物理錨+証拠(W_EV=0.2)", "red")]:
    s = np.sort(arr); ax[1].plot(s, np.linspace(0, 1, len(s)), label=f"{lab} (中央{np.median(arr):.2f})", color=c)
ax[1].set_xlabel("井戸別 RMSE (ft)"); ax[1].set_ylabel("累積割合"); ax[1].set_xlim(0, 25); ax[1].set_title("井戸別RMSE ECDF"); ax[1].legend()
# (3) 改善が大きい例1井
gains = r0r - refr; idx = int(np.argmax(gains)); wn, _, _, r0, ref, true, mde = rows[idx]
ax[2].plot(mde, true, "k-", lw=2, label="真値TVT")
ax[2].plot(mde, r0[:len(true)], color="gray", label=f"r0素 ({r0r[idx]:.1f}ft)")
ax[2].plot(mde, ref[:len(true)], color="red", label=f"物理錨+証拠 ({refr[idx]:.1f}ft)")
ax[2].set_xlabel("MD (ft)"); ax[2].set_ylabel("TVT (ft)"); ax[2].invert_yaxis(); ax[2].set_title(f"例: {wn[:14]} (改善{gains[idx]:.1f}ft)"); ax[2].legend()
plt.tight_layout(); out = PROJ / "v91" / "x_unet_refine_v91.png"; plt.savefig(out, dpi=110)
print("saved", out)
