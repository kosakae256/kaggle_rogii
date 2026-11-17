# -*- coding: utf-8 -*-
"""実験J(CPU): バッチPF = PFの理屈(重み付き滑らか軌跡分布+posterior-mean hedge)を
逐次でなく v91物理候補(低自由度・oracle3.3)の"全区間GR尤度 softmax 重み付き平均"で実現。
v91はargmax選択(22で死)しか試してない。PFの本体=posterior-mean hedgeを物理候補でやる。
真TVTは評価のみ。env NW N β sweep。出力: exp_batch_pf_v95.txt"""
import os, sys, io, time
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
import numpy as np, pandas as pd, pyarrow.parquet as pq
from scipy.optimize import nnls
TR = "rogii-wellbore-geology-prediction/train"; B = "v95/artifacts_v95b/"
NW = int(os.environ.get("NW", "300")); N = int(os.environ.get("N", "3000")); K = 4
SLOPE = 55 / 1000.0; CONT = 4.0; DEV = 25 / 1000.0; DIPSP = 20 / 1000.0
rng = np.random.default_rng(0)
tb = pq.read_table(B + "data/train.parquet", columns=["well", "target", "last_known_tvt"]).to_pandas()
g = tb["well"].astype(str).to_numpy(); yv = tb["target"].to_numpy(float); lk = tb["last_known_tvt"].to_numpy(float)
d1 = np.load(B.replace("v95b", "v95") + "l1_oof.npz"); gbm = np.mean([d1[k] for k in d1.files if k not in ("y", "well")], 0)
tcn = np.mean([v for k, v in np.load(B + "nn_oof_combo.npz").items() if k.startswith("mdn2")], 0)
l5 = np.load(B + "l5_ext_oof.npz")["l5_mean"].astype(float)
Mm = np.stack([gbm, tcn, l5], 1); mf = np.isfinite(Mm).all(1) & np.isfinite(yv); wj, _ = nnls(Mm[mf], yv[mf]); wj /= wj.sum()
blend_tvt = lk + Mm @ wj
rows_of = {w: np.where(g == w)[0] for w in pd.unique(g)}

def gen(md, z, tvt_last, z_last, t_min, t_max, pmd, pch):
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

def R(err): e = np.concatenate(err); return float(np.sqrt(np.mean(e ** 2)))
wells = sorted(w.split("__")[0] for w in os.listdir(TR) if w.endswith("__horizontal_well.csv"))[:NW]
BETAS = [0.5, 1.0, 2.0, 4.0, 8.0]
err_bl = []; err_or = []; err_arg = []; err_w = {b: [] for b in BETAS}; preds_best = {}
t0 = time.time(); nused = 0
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
    t_tvt = tw["TVT"].to_numpy(float); t_gr = tw["GR"].to_numpy(float); m = np.isfinite(t_tvt) & np.isfinite(t_gr)
    t_tvt, t_gr = t_tvt[m], t_gr[m]
    if len(t_tvt) < 8: continue
    o = np.argsort(t_tvt); t_tvt, t_gr = t_tvt[o], t_gr[o]
    tvt_last = float(ti[hps]); z_last = float(z[hps])
    ki = np.where(kn)[0]; ki = ki[ki >= max(0, hps - 80)]
    pmd = md[ki] - md[hps]; pch = (ti[ki] - tvt_last) + (z[ki] - z_last)
    r = rows_of[wn]; L = len(hw.iloc[ev])
    if len(r) != L: continue
    cand = gen(md[ev], z[ev], tvt_last, z_last, float(t_tvt.min()), float(t_tvt.max()), pmd, pch)
    if cand is None: continue
    tr = tvt[ev]; bl = blend_tvt[r]
    obs = pd.Series(gr[ev]).interpolate(limit_direction="both").fillna(np.nanmean(t_gr)).to_numpy()
    grp = np.interp(cand, t_tvt, t_gr)                         # (Nc,L) typewell GR at candidate
    # GR z正規化(較正吸収)して尤度 = 全区間残差
    oz = (obs - obs.mean()) / (obs.std() + 1e-6)
    gz = (grp - grp.mean(1, keepdims=True)) / (grp.std(1, keepdims=True) + 1e-6)
    ll = -0.5 * np.mean((gz - oz[None, :]) ** 2, 1)           # per-candidate 平均対数尤度(z空間)
    err_bl.append(bl - tr); err_or.append(cand[np.argmin(np.sqrt(np.mean((cand - tr[None, :]) ** 2, 1)))] - tr)
    err_arg.append(cand[np.argmax(ll)] - tr)                  # argmax(v91=22相当)
    for b in BETAS:
        w = np.exp(b * (ll - ll.max())); w /= w.sum()
        pm = (w[:, None] * cand).sum(0)                        # ★posterior-mean hedge(PFの本体)
        err_w[b].append(pm - tr)
        if b == 2.0: preds_best[wn] = (blend_tvt[r] * 0 + pm).astype(np.float32)  # save β2 for blend
    nused += 1
    if (wi + 1) % 50 == 0: print(f"  {wi+1}/{len(wells)} ({time.time()-t0:.0f}s)", flush=True)
out = ["=== 実験J: バッチPF (物理候補 + GR尤度softmax posterior-mean) ==="]
out.append(f"  井={nused} N={N}")
out.append(f"  blend            = {R(err_bl):.4f}")
out.append(f"  物理oracle       = {R(err_or):.4f}")
out.append(f"  argmax選択(v91)  = {R(err_arg):.4f}")
for b in BETAS:
    out.append(f"  ★posterior-mean β={b:<4} = {R(err_w[b]):.4f}")
out.append("  → posterior-mean(hedge)がargmaxを大きく下回れば、PFの理屈が物理候補で生きる。blend近傍なら脱相関脚候補。")
txt = "\n".join(out); print("\n" + txt, flush=True)
open(B + "exp_batch_pf_v95.txt", "w", encoding="utf-8").write(txt)
np.savez(B + "batch_pf_v95.npz", **preds_best)
