# -*- coding: utf-8 -*-
"""v90本番(GBM+TCN blend)の誤差 と 物理候補選択の誤差 の相関 + blend効果。fold0 155井で行対応。
   v90 OOFは残差空間(err = pred_res - target)。物理はTVT空間だが lk相殺で同一誤差。行はwell内MD昇順で対応。"""
import os, sys, io, pickle
try: sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
except Exception: pass
import numpy as np, pandas as pd, pyarrow.parquet as pq
from pathlib import Path
PROJ = Path(r"C:\Users\kosaka256\Documents\rogii_claude")
TR = PROJ / "rogii-wellbore-geology-prediction" / "train"
A = PROJ / "v64" / "artifacts_v64"; AV = PROJ / "v89" / "artifacts_v89"; A90 = PROJ / "v90" / "artifacts_v90"
N = 5000; K = 4; SLOPE = 55/1000.; DEV = 25/1000.; DIPSP = 20/1000.; CONT = 4.0
W_EV = 0.2; NB = 90; ND = 181; STEP = 0.5; W_NN = 0.62
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


# --- v90 OOF(残差) → well別の誤差列 ---
print("loading v90 parquet well/target ...", flush=True)
tb = pq.read_table(A90 / "data" / "train.parquet", columns=["well", "target"])
well = tb.column("well").to_numpy(zero_copy_only=False); y = tb.column("target").to_numpy(zero_copy_only=False).astype(np.float64)
gbm = np.load(A90 / "gbm_v41_oof.npy").astype(np.float64)[:len(y)]
nnz = np.load(A90 / "nn_oof_combo3.npz"); nn = np.mean([nnz[k] for k in nnz.files], axis=0)[:len(y)]
v90res = (1 - W_NN) * gbm + W_NN * nn
v90err_all = v90res - y                          # 残差空間の誤差 = TVT空間の誤差
# well別の行区間(連続ブロック前提)
order = np.argsort(well, kind="stable")  # 安定ソートで well毎に連続化しつつ元順維持は別途
idx_by_well = {}
for i, w in enumerate(well):
    idx_by_well.setdefault(w, []).append(i)

DIST = dict(np.load(A / "v89_dist_f0_dist.npz"))
EVD = pickle.loads(open(AV / "learned_boot.pkl", "rb").read())
PF = dict(np.load(AV / "pf_center_v89.npz"))
zn = lambda v: (v - v.mean()) / (v.std() + 1e-9)
E_ref, E_v90, E_r0, E_uo, nmis = [], [], [], [], 0
for wn in [w for w in DIST if w in EVD and w in PF and w in idx_by_well]:
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
    ref = cand[int(np.argmax(-zn(r0d) + W_EV*(-zn(gscore) + zn(uscore))))]
    uonly = cand[int(np.argmax(zn(uscore)))]                           # U-Net単独(脱相関側)
    eref = (ref-true)[fin]; er0 = (r0-true)[fin]; euo = (uonly-true)[fin]
    v90e = v90err_all[np.array(idx_by_well[wn])]                       # v90誤差(この坑井の全parquet行)
    n = min(len(eref), len(v90e))
    if abs(len(eref) - len(v90e)) > 3: nmis += 1
    E_ref.append(eref[:n]); E_v90.append(v90e[:n]); E_r0.append(er0[:n]); E_uo.append(euo[:n])

eref = np.concatenate(E_ref); ev90 = np.concatenate(E_v90); er0 = np.concatenate(E_r0); euo = np.concatenate(E_uo)
rmse = lambda e: float(np.sqrt(np.mean(e**2)))
print(f"\n[fold0 {len(E_ref)}井, 行対応 (長さ不一致>3の井={nmis})]")
print(f"  v90本番(GBM+TCN) 単体   = {rmse(ev90):.3f}ft")
print(f"  物理選択(r0錨+U-Net)     = {rmse(eref):.3f}ft")
print(f"  素PF r0                 = {rmse(er0):.3f}ft")
print(f"  ρ(v90誤差, 物理選択誤差) = {np.corrcoef(ev90, eref)[0,1]:.3f}")
print(f"  ρ(v90誤差, 素PF誤差)     = {np.corrcoef(ev90, er0)[0,1]:.3f}")
print("  --- blend v90×物理選択 ---")
for w in [0.15, 0.25, 0.35, 0.5]:
    print(f"    w_phys={w}: {rmse((1-w)*ev90 + w*eref):.3f}")
print("  --- 脱相関側: U-Net単独(11ft級)を本番にblend ---")
print(f"    U-Net単独 単体={rmse(euo):.3f}  ρ(v90,U単独)={np.corrcoef(ev90, euo)[0,1]:.3f}")
for w in [0.1, 0.15, 0.2]:
    print(f"    w_uonly={w}: {rmse((1-w)*ev90 + w*euo):.3f}")
