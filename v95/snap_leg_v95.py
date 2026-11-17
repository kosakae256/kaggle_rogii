# -*- coding: utf-8 -*-
"""v91 snap物理脚を v95 に適用(research=hard-well転移成功例, 未回収)。
物理候補(prefix-dip, N, K4, 多seed)を pf_center_v89(r0)にsnap→多seed平均=snapms。
v95 blend(5.987)へ固定weight blend(v91転移安全recipe)。真TVTは評価のみ。
出力: v95/artifacts_v95b/snap_leg_v95.npz + テキスト"""
import os, sys, io, time
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
import numpy as np, pandas as pd, pyarrow.parquet as pq
from scipy.optimize import nnls
TR = "rogii-wellbore-geology-prediction/train"; B = "v95/artifacts_v95b/"
N = int(os.environ.get("N", "3000")); K = 4; SEEDS = [0, 1, 2]
SLOPE = 55 / 1000.0; CONT = 4.0; DEV = 25 / 1000.0; DIPSP = 20 / 1000.0
pfc = np.load("v89/artifacts_v89/pf_center_v89.npz", allow_pickle=True)
tb = pq.read_table(B + "data/train.parquet", columns=["well", "target", "last_known_tvt"]).to_pandas()
g = tb["well"].astype(str).to_numpy(); yv = tb["target"].to_numpy(float); lk = tb["last_known_tvt"].to_numpy(float)
d1 = np.load(B.replace("v95b", "v95") + "l1_oof.npz"); gbm = np.mean([d1[k] for k in d1.files if k not in ("y", "well")], 0)
tcn = np.mean([v for k, v in np.load(B + "nn_oof_combo.npz").items() if k.startswith("mdn2")], 0)
l5 = np.load(B + "l5_ext_oof.npz")["l5_mean"].astype(float)
Mm = np.stack([gbm, tcn, l5], 1); mf = np.isfinite(Mm).all(1) & np.isfinite(yv); wj, _ = nnls(Mm[mf], yv[mf]); wj /= wj.sum()
blend = Mm @ wj                                    # 残差
rows_of = {w: np.where(g == w)[0] for w in pd.unique(g)}

def gen(md, z, tvt_last, z_last, t_min, t_max, pmd, pch, rng):
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

def R(p): m = np.isfinite(p) & np.isfinite(yv); return float(np.sqrt(np.mean((p[m] - yv[m]) ** 2)))
wells = sorted(w.split("__")[0] for w in os.listdir(TR) if w.endswith("__horizontal_well.csv"))
snapms = np.full(len(g), np.nan); t0 = time.time(); nused = 0
for wi, wn in enumerate(wells):
    if wn not in rows_of or wn not in pfc.files: continue
    try: hw = pd.read_csv(os.path.join(TR, f"{wn}__horizontal_well.csv")); tw = pd.read_csv(os.path.join(TR, f"{wn}__typewell.csv"))
    except Exception: continue
    kn = hw["TVT_input"].notna().to_numpy()
    if kn.sum() < 10 or (~kn).sum() < 30: continue
    hps = int(np.where(kn)[0].max()); ev = slice(hps + 1, len(hw))
    z = hw["Z"].to_numpy(float); md = hw["MD"].to_numpy(float); ti = hw["TVT_input"].to_numpy(float)
    tt = tw["TVT"].to_numpy(float); m = np.isfinite(tt); t_min, t_max = float(tt[m].min()), float(tt[m].max())
    tvt_last = float(ti[hps]); z_last = float(z[hps])
    ki = np.where(kn)[0]; ki = ki[ki >= max(0, hps - 80)]; pmd = md[ki] - md[hps]; pch = (ti[ki] - tvt_last) + (z[ki] - z_last)
    r = rows_of[wn]; L = len(hw.iloc[ev])
    if len(r) != L: continue
    r0 = np.asarray(pfc[wn], float)
    if len(r0) != L: r0 = r0[:L] if len(r0) > L else np.pad(r0, (0, L - len(r0)), mode="edge")
    snaps = []
    for sd in SEEDS:
        cand = gen(md[ev], z[ev], tvt_last, z_last, t_min, t_max, pmd, pch, np.random.default_rng(sd))
        if cand is None: continue
        r0d = np.sqrt(np.mean((cand - r0[None, :]) ** 2, 1))   # r0(pf_center)への近さ
        snaps.append(cand[int(np.argmin(r0d))])
    if not snaps: continue
    sm = np.mean(snaps, 0); snapms[r] = sm - lk[r]              # 残差化
    nused += 1
    if (wi + 1) % 100 == 0: print(f"  {wi+1}/{len(wells)} ({time.time()-t0:.0f}s)", flush=True)
m = np.isfinite(snapms) & np.isfinite(blend) & np.isfinite(yv)
rho = float(np.corrcoef(snapms[m] - yv[m], blend[m] - yv[m])[0, 1])
out = ["=== v91 snap物理脚 → v95 blend評価 ==="]
out.append(f"  snap単体 RMSE={R(snapms):.4f}  blend={R(blend):.4f}  被覆={np.isfinite(snapms).mean()*100:.0f}%")
out.append(f"  ★ρ(snap誤差, blend誤差)={rho:.3f}  (v91は v90で0.616)")
for fw in [0.05, 0.10, 0.15]:
    p = blend.copy(); p[m] = (1 - fw) * blend[m] + fw * snapms[m]
    out.append(f"  ★固定weight snap={fw}: overall={R(p):.4f} ({R(p)-R(blend):+.4f})")
# nnls 4脚
M2 = np.stack([gbm, tcn, l5, snapms], 1); m2 = np.isfinite(M2).all(1) & np.isfinite(yv); w2, _ = nnls(M2[m2], yv[m2]); w2 /= w2.sum()
oof2 = np.full(len(g), np.nan); oof2[m2] = M2[m2] @ w2
out.append(f"  nnls 4脚(gbm,tcn,l5,snap)={R(oof2):.4f} 重み{tuple(round(float(x),3) for x in w2)}")
txt = "\n".join(out); print("\n" + txt, flush=True); open(B + "snap_leg_v95.txt", "w", encoding="utf-8").write(txt)
np.savez(B + "snap_leg_v95.npz", snapms=snapms.astype(np.float32))
