# -*- coding: utf-8 -*-
"""v95 多表現グリッドベイズフィルタ(非PF) — tw+self+nbr+nbr_gr5 emission 積合成(生格子)。
★元のgridfilter_v95(tw生格子=10.24)と同じ生格子方式で全表現を作る(build_smoother_inputsのフレーム不使用)。
真δは全表現合意/decoyはtwのみ→積で真δ鋭化(v68 GF-MR思想)。物理錨grfree中心。
env: NW REPS DR TSIG GS。出力: v95/artifacts_v95b/gf_mr_v95.npz
"""
import os, sys, io, time, pickle, glob
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
from pathlib import Path
import numpy as np, pandas as pd
from scipy.ndimage import gaussian_filter1d
sys.path.insert(0, str(Path(r"C:\Users\kosaka256\Documents\rogii_claude") / "v95"))
import imputers_v95 as imp
TR = Path("rogii-wellbore-geology-prediction/train"); ART = Path("v95/artifacts_v95")
NW = int(os.environ.get("NW", "0")) or None
REPS = [r for r in os.environ.get("REPS", "tw,self,nbr,nbr_gr5").split(",") if r]
DR = float(os.environ.get("DR", "40")); DSTEP = float(os.environ.get("DSTEP", "1.0"))
TSIG = float(os.environ.get("TSIG", "2.0")); GS = float(os.environ.get("GS", "1.0"))
delta = np.arange(-DR, DR + 1e-6, DSTEP); D = len(delta)
anchor_d = pickle.loads((ART / "grfree_anchor_train.pkl").read_bytes())

def build_grid(tvt, gr, step=1.0):
    m = np.isfinite(tvt) & np.isfinite(gr); tvt, gr = tvt[m], gr[m]
    if len(tvt) < 5: return None
    edges = np.arange(tvt.min(), tvt.max() + step, step)
    if len(edges) < 3: return None
    idx = np.clip(np.digitize(tvt, edges) - 1, 0, len(edges) - 2)
    gg = np.full(len(edges) - 1, np.nan)
    for b in range(len(edges) - 1):
        v = gr[idx == b]
        if len(v): gg[b] = np.median(v)
    ax = (edges[:-1] + edges[1:]) / 2; ok = np.isfinite(gg)
    return (ax[ok], gg[ok]) if ok.sum() >= 5 else None

def rep_emission(obs, ax, gg, anc):
    T = len(obs); pos = anc[:, None] + delta[None, :]
    ref = np.interp(pos.ravel(), ax, gg).reshape(T, D)
    o = (obs - np.nanmean(obs)) / (np.nanstd(obs) + 1e-6)
    rz = (ref - np.nanmean(ref)) / (np.nanstd(ref) + 1e-6)
    dd = (o[:, None] - rz) / GS
    return np.exp(-0.5 * np.clip(dd * dd, 0, 60))

def main():
    imp.build_imputers()
    wells = sorted({os.path.basename(q).split("__")[0] for q in glob.glob(str(TR / "*__horizontal_well.csv"))})
    if NW: wells = wells[:NW]
    t0 = time.time(); preds = {}; E_all = []; nused = 0
    for wi, wn in enumerate(wells):
        try:
            hw = pd.read_csv(TR / f"{wn}__horizontal_well.csv"); tw = pd.read_csv(TR / f"{wn}__typewell.csv")
        except Exception: continue
        if "TVT" not in hw.columns: continue
        kn = hw["TVT_input"].notna().to_numpy()
        if kn.sum() < 10 or (~kn).sum() < 30: continue
        hps = int(np.where(kn)[0].max()); ev = slice(hps + 1, len(hw))
        sp = anchor_d.get(wn)
        if sp is None: continue
        anc = np.asarray(sp["tvt"], float)
        obs = hw["GR"].to_numpy(float)[ev]; true = hw["TVT"].to_numpy(float)[ev]
        if len(anc) != len(obs): continue
        obs = pd.Series(obs).interpolate(limit_direction="both").fillna(float(np.nanmean(hw["GR"]))).to_numpy()
        T = len(obs)
        cx, cy = float(hw["X"].mean()), float(hw["Y"].mean()); cz = float(hw["Z"].mean())
        grids = {}
        # tw
        grids["tw"] = build_grid(tw["TVT"].to_numpy(float), tw["GR"].to_numpy(float))
        # self = prefix (TVT_input, GR)
        knp = hw[hw["TVT_input"].notna()]
        grids["self"] = build_grid(knp["TVT_input"].to_numpy(float), knp["GR"].to_numpy(float))
        # nbr = 距離近傍 concat
        rw = imp.neighbors_of(cx, cy, exclude_wid=wn)
        if len(rw) >= imp.NEED_REFS:
            gg = [imp.ref_grtvt(r) for r, _ in rw]
            grids["nbr"] = build_grid(np.concatenate([t for _, t in gg]), np.concatenate([g for g, _ in gg]))
        else: grids["nbr"] = None
        # nbr_gr5 = GR類似近傍 concat
        ng = imp.neighbors_gr_of(knp["TVT_input"].to_numpy(float), knp["GR"].to_numpy(float), cx, cy, cz, exclude_wid=wn)
        if ng:
            gg5 = [imp.ref_grtvt(w) for w, _, _, _ in ng]
            grids["nbr_gr5"] = build_grid(np.concatenate([t for _, t in gg5]), np.concatenate([g for g, _ in gg5]))
        else: grids["nbr_gr5"] = None
        REP_W = float(os.environ.get("REP_W", "1.0"))     # 非tw表現の重み(down-weight)
        logE = np.zeros((T, D)); nrep = 0
        for r in REPS:
            gd = grids.get(r)
            if gd is None: continue
            E = rep_emission(obs, gd[0], gd[1], anc); w = 1.0 if r == "tw" else REP_W
            logE += w * np.log(np.clip(E, 1e-6, None)); nrep += 1
        if nrep == 0: continue
        asig = np.clip(np.asarray(sp["sig"], float), 1e-3, None)   # ★錨prior(物理項): δを0(錨)へ引く
        logE = logE + (-0.5 * (delta[None, :] / asig[:, None]) ** 2)
        E = np.exp(logE - logE.max(1, keepdims=True)); E = np.clip(E, 1e-12, None)
        fwd = np.empty((T, D)); f = E[0] / E[0].sum(); fwd[0] = f
        for i in range(1, T):
            f = gaussian_filter1d(f, TSIG, mode="nearest") * E[i]; s = f.sum(); f = f / s if s > 1e-300 else np.full(D, 1.0 / D); fwd[i] = f
        bwd = np.empty((T, D)); b = np.ones(D) / D; bwd[T - 1] = b
        for i in range(T - 2, -1, -1):
            b = gaussian_filter1d(b * E[i + 1], TSIG, mode="nearest"); s = b.sum(); b = b / s if s > 1e-300 else np.full(D, 1.0 / D); bwd[i] = b
        post = fwd * bwd; post /= post.sum(1, keepdims=True) + 1e-300
        pred = anc + (post * delta[None, :]).sum(1)
        e = pred - true; E_all.append(e[np.isfinite(e)]); preds[wn] = pred.astype(np.float32); nused += 1
        if (wi + 1) % 50 == 0: print(f"  {wi+1}/{len(wells)} ({time.time()-t0:.0f}s)", flush=True)
    e = np.concatenate(E_all); rmse = float(np.sqrt(np.mean(e ** 2)))
    print(f"\n=== 多表現グリッドフィルタ v95(生格子) [REPS={REPS}] ===")
    print(f"  井={nused} RMSE={rmse:.4f}  (単表現tw=10.24, PF=8.6目標)  |e|中央={np.median(np.abs(e)):.2f}")
    np.savez("v95/artifacts_v95b/gf_mr_v95.npz", **preds)

if __name__ == "__main__":
    main()
