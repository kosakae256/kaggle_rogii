# -*- coding: utf-8 -*-
"""
v85 統合 — 多表現PFアンサンブル(全773井, struct不使用)
=====================================================
各井戸でPF(pf_tuning struct-free, r0)を複数のGR表現で回し、行別誤差を集約:
  raw / logamp / affine / bandcap  (antieq/lowpassは死のため除外)
出力: 全773井 pooled overall RMSE(各表現単体) + 相関行列 + 閉形式多脚blend(Σw=1)。
= ユーザー最終要件「全井戸の結果」。行別誤差をpklキャッシュして再集計可。
使い方: NW=100000 REPS=raw,logamp,affine,bandcap python v85/ensemble_v85.py
"""
import os, io, sys, json, time, glob, pickle, itertools
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
from pathlib import Path
import numpy as np, pandas as pd

P0 = Path(r"C:\Users\kosaka256\Documents\rogii_claude")
DATA = P0 / "rogii-wellbore-geology-prediction"
ART = P0 / "v85" / "artifacts_v85"; ART.mkdir(parents=True, exist_ok=True)
NB = P0 / "pf_tuning" / "pf_tuning_pf01.ipynb"
NW = int(os.environ.get("NW", "100000"))
SEED = int(os.environ.get("SEED", "1"))
REPS = os.environ.get("REPS", "raw,logamp,affine,bandcap").split(",")
MAR = int(os.environ.get("MAR", "13")); LOG_G0 = float(os.environ.get("LOG_G0", "40.0"))
AMIN, AMAX = 0.3, 3.0

_nb = json.loads(NB.read_text(encoding="utf-8"))
_ns = dict(np=np, pd=pd, json=json, hashlib=__import__("hashlib"))
from dataclasses import dataclass, asdict, field
_ns.update(dataclass=dataclass, asdict=asdict, field=field)
exec("".join(_nb["cells"][5]["source"]), _ns)
run_pf = _ns["run_pf_lik_ensemble_param"]
R0 = dict(n_particles=400, n_seeds=8, likelihood_scale=10.536319667317654,
    init_pos_std=0.9331541486020234, init_rate_std=0.00047753793163756304,
    mom=0.9997911117425017, vn=0.0006297672017716754, pn=0.005014009140727263,
    rp=0.7627309890952053, rr=0.00045663430474523125, resamp=0.032049849143995945,
    gr_sig_min=11.865793233968713, gr_sig_max=44.98927964612316, gr_sig_def=119.25397464636981,
    gr_sig_mult=3.058318309943017, tvt_clip_margin=150.56111180234504, rate_clip=2.0,
    gr_power=1.7826896851395453, hgr_smooth_r=0, tw_gr_smooth_r=0, jump_prob=0.002, jump_std=5.0)


def _ma(x, r):
    return pd.Series(x).rolling(2 * r + 1, center=True, min_periods=1).mean().to_numpy() if r > 1 else np.asarray(x, float)


def transform(hw, tw, rep):
    hg = hw["GR"].astype(float).interpolate(limit_direction="both").to_numpy()
    tg = tw["GR"].astype(float).to_numpy()
    hw2 = hw.copy(); tw2 = tw.copy()
    if rep == "raw":
        return hw2, tw2
    if rep == "logamp":
        f = lambda g: np.log1p(np.clip(g, 0, None) / LOG_G0) * LOG_G0
        hw2["GR"] = f(hg); tw2["GR"] = f(tg); return hw2, tw2
    if rep == "bandcap":
        hw2["GR"] = hg + _ma(hg, MAR); tw2["GR"] = tg + _ma(tg, MAR); return hw2, tw2
    if rep == "affine":
        kn = hw["TVT_input"].notna().to_numpy()
        ttv = tw["TVT"].to_numpy(float); tgr = tw["GR"].to_numpy(float)
        m = np.isfinite(ttv) & np.isfinite(tgr)
        if m.sum() >= 8 and kn.sum() >= 100:
            o = np.argsort(ttv[m]); ttv2, tgr2 = ttv[m][o], tgr[m][o]
            x = np.interp(hw["TVT_input"].to_numpy(float)[kn], ttv2, tgr2); y = hg[kn]
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() >= 100:
                x, y = x[ok], y[ok]; c = np.polyfit(x, y, 1)
                for _ in range(3):
                    r = y - np.polyval(c, x); s = np.median(np.abs(r)) * 1.4826 + 1e-6
                    c = np.polyfit(x, y, 1, w=1.0 / (1.0 + (r / (2 * s)) ** 2))
                a, b = float(c[0]), float(c[1])
                if np.isfinite(a) and AMIN < a < AMAX:
                    hw2["GR"] = (hg - b) / a
        return hw2, tw2
    raise ValueError(rep)


def errs_for(wid):
    hw = pd.read_csv(DATA / "train" / f"{wid}__horizontal_well.csv").reset_index(drop=True)
    tw = pd.read_csv(DATA / "train" / f"{wid}__typewell.csv").reset_index(drop=True)
    hw["__row_id"] = np.arange(len(hw))
    ev = hw["TVT_input"].isna().to_numpy() & hw["TVT"].notna().to_numpy()
    if ev.sum() == 0: return None
    tvt = hw["TVT"].to_numpy(float)
    out = {}
    for rep in REPS:
        hr, tr = transform(hw, tw, rep)
        pred = run_pf(hr, tr, params=R0, return_full=True)
        e = (pred - tvt)[ev]
        out[rep] = e.astype(np.float32)
    return out


def main():
    wids = sorted({os.path.basename(p).split("__")[0] for p in glob.glob(str(DATA / "train" / "*__horizontal_well.csv"))})
    rng = np.random.default_rng(SEED)
    if NW < len(wids):
        wids = list(rng.choice(wids, size=NW, replace=False))
    t0 = time.time(); E = {r: [] for r in REPS}; nok = 0
    for i, w in enumerate(wids):
        try:
            o = errs_for(w)
        except Exception as ex:
            print(f"  {w[:8]}: ERROR {type(ex).__name__} {ex}"); continue
        if o is None: continue
        m = np.ones(len(next(iter(o.values()))), bool)
        for v in o.values(): m &= np.isfinite(v)
        for r in REPS: E[r].append(o[r][m])
        nok += 1
        if (i + 1) % 50 == 0:
            cur = {r: float(np.sqrt(np.mean(np.concatenate(E[r]) ** 2))) for r in REPS}
            print(f"  {i+1}/{len(wids)} " + " ".join(f"{r}={cur[r]:.3f}" for r in REPS) + f" ({time.time()-t0:.0f}s)", flush=True)
    Em = {r: np.concatenate(E[r]) for r in REPS}
    n = min(len(v) for v in Em.values())
    M = np.stack([Em[r][:n] for r in REPS])
    (ART / "ensemble_errs.pkl").write_bytes(pickle.dumps({r: Em[r] for r in REPS}, protocol=4))
    print(f"\n=== v85 多表現PFアンサンブル ({nok}井, 全行プール) ===")
    for r in REPS:
        print(f"  {r:8s} 単体 = {np.sqrt(np.mean(Em[r]**2)):.4f}")
    C = np.corrcoef(M)
    print("  相関ρ:"); [print("    ", REPS[i], np.round(C[i], 3)) for i in range(len(REPS))]
    G = M @ M.T / M.shape[1]
    def solve(idx):
        Gs = G[np.ix_(idx, idx)]; k = len(idx)
        A = np.zeros((k + 1, k + 1)); A[:k, :k] = Gs; A[:k, k] = 1; A[k, :k] = 1
        b = np.zeros(k + 1); b[k] = 1; w = np.linalg.solve(A, b)[:k]
        return float(np.sqrt(w @ Gs @ w)), w
    best = {}
    for k in range(2, len(REPS) + 1):
        for idx in itertools.combinations(range(len(REPS)), k):
            if 0 not in idx: continue                    # rawを必ず含む
            r, w = solve(list(idx))
            if k not in best or r < best[k][0]: best[k] = (r, idx, w)
    base = np.sqrt(np.mean(Em["raw"] ** 2))
    print(f"  ★多脚blend(raw基準{base:.4f}):")
    for k in sorted(best):
        r, idx, w = best[k]
        print(f"    {k}脚 {r:.4f} (−{base-r:.3f}) {[REPS[i] for i in idx]} w={[round(float(x),2) for x in w]}")


if __name__ == "__main__":
    main()
