# -*- coding: utf-8 -*-
"""
v91 物理事前分布 — 地質面(区分線形)をサンプリングして候補TVTを生成。他井戸/GR不使用。
================================================================================
物理: TVT = 地質面 − Z + b。 candidate_tvt(md) = tvt_last + change(md) − (Z(md)−Z_last)。
change = 区分線形(K制御点, 各区間の傾斜をサンプル, change(0)=0で最終既知に錨)。
拘束: typewell TVT範囲[t_min,t_max]内 / 最初~30点は最終既知の±CONT ft(連続)。
oracle = 真値に最も近い候補のRMSE(全行プール)。目標~4.0。
env: NW(井戸数) / N(粒子=1500) / K(制御点=6) / SLOPE(傾斜range ft/1000ft=55) / CONT(連続拘束ft=4) / SEED
"""
import os, sys, io, time
try: sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
except Exception: pass
import numpy as np, pandas as pd
TR = "rogii-wellbore-geology-prediction/train"
NW = int(os.environ.get("NW", "0")) or None
N = int(os.environ.get("N", "1500")); K = int(os.environ.get("K", "6"))
SLOPE = float(os.environ.get("SLOPE", "55")) / 1000.0    # 傾斜range ft/ft
CONT = float(os.environ.get("CONT", "4.0"))
SEED = int(os.environ.get("SEED", "0"))
rng = np.random.default_rng(SEED)


DEV = float(os.environ.get("DEV", "25")) / 1000.0     # 偏差傾斜σ ft/ft(prefix dipからのずれ)
DIPSP = float(os.environ.get("DIPSP", "20")) / 1000.0  # eval全体dipのprefixからのspread ft/ft


def gen_oracle(md, z, tvt_last, z_last, t_min, t_max, true_tvt, pmd_rel, pchange, grsel=None):
    """★prefix地質dip中心の候補生成 → oracle/GR選択/base。change = (prefix_dip±spread)×md + 滑らか偏差。"""
    L = len(md); md_rel = md - md[0]; dZ = z - z_last
    # prefix地質dip(既知区間の change slope をrobust fit)
    if len(pmd_rel) >= 5 and (pmd_rel[-1] - pmd_rel[0]) > 1:
        pdip = float(np.polyfit(pmd_rel, pchange, 1)[0])
    else:
        pdip = 0.0
    pdip = float(np.clip(pdip, -SLOPE, SLOPE))
    xk = np.linspace(0.0, md_rel[-1] + 1e-6, K); seg = np.diff(xk)
    # 各粒子: base dip(prefix±spread) + 制御点偏差(滑らか)
    dip_n = pdip + rng.uniform(-DIPSP, DIPSP, size=(N, 1))          # (N,1) eval全体dip
    dslope = rng.normal(0.0, DEV, size=(N, K - 1))                  # (N,K-1) 偏差傾斜
    dincr = dslope * seg[None, :]
    dyk = np.concatenate([np.zeros((N, 1)), np.cumsum(dincr, axis=1)], axis=1)  # (N,K) 偏差@制御点
    dev = np.empty((N, L), np.float64)
    for i in range(N):
        dev[i] = np.interp(md_rel, xk, dyk[i])
    change = dip_n * md_rel[None, :] + dev                          # (N,L) 地質change
    cand = tvt_last + change - dZ[None, :]
    ok = ((cand >= t_min - 5) & (cand <= t_max + 5)).all(axis=1)
    ncont = min(30, L)
    ok &= (np.abs(cand[:, :ncont] - tvt_last) < CONT).all(axis=1)
    if not ok.any(): return None
    cand = cand[ok]; e = cand - true_tvt[None, :]
    rmse = np.sqrt(np.mean(e ** 2, axis=1)); bi = int(np.argmin(rmse))
    out = dict(oracle=cand[bi], nc=int(ok.sum()))
    # ★現実的選択1: GRマッチ(typewell GRを候補TVTで内挿し観測GRと比較, 純幾何生成のGRは選択のみ)
    if grsel is not None:
        tw_tvt, tw_gr, obs_gr = grsel
        gr_pred = np.interp(cand, tw_tvt, tw_gr)                    # (Nok,L)
        gscore = np.mean((gr_pred - obs_gr[None, :]) ** 2, axis=1)  # GR残差(小=良)
        out["gr"] = cand[int(np.argmin(gscore))]
    # ★現実的選択2: 無選択=prefix dip外挿(偏差0, 中心候補)
    base = tvt_last + pdip * md_rel - dZ
    out["base"] = base
    return out


def main():
    t0 = time.time()
    wells = sorted(w.split("__")[0] for w in os.listdir(TR) if w.endswith("__horizontal_well.csv"))
    if NW: wells = wells[:NW]
    eo = []; eg = []; eb = []; ncands = []; nused = 0; nskip = 0
    for wi, wn in enumerate(wells):
        try:
            hw = pd.read_csv(os.path.join(TR, f"{wn}__horizontal_well.csv"))
            tw = pd.read_csv(os.path.join(TR, f"{wn}__typewell.csv"))
        except Exception:
            continue
        kn = hw["TVT_input"].notna().to_numpy()
        if kn.sum() < 10 or (~kn).sum() < 30: continue
        hps = int(np.where(kn)[0].max())
        ev = slice(hps + 1, len(hw))
        z = hw["Z"].to_numpy(float); md = hw["MD"].to_numpy(float); tvt = hw["TVT"].to_numpy(float)
        ti = hw["TVT_input"].to_numpy(float); gr = hw["GR"].to_numpy(float)
        t_tvt = tw["TVT"].to_numpy(float); t_gr = tw["GR"].to_numpy(float)
        m = np.isfinite(t_tvt) & np.isfinite(t_gr); t_tvt, t_gr = t_tvt[m], t_gr[m]
        if len(t_tvt) < 8: continue
        order = np.argsort(t_tvt); t_tvt, t_gr = t_tvt[order], t_gr[order]
        t_min, t_max = float(t_tvt.min()), float(t_tvt.max())
        tvt_last = float(ti[hps]); z_last = float(z[hps])
        kn_idx = np.where(kn)[0]; kn_idx = kn_idx[kn_idx >= max(0, hps - 80)]
        pmd_rel = md[kn_idx] - md[hps]
        pchange = (ti[kn_idx] - tvt_last) + (z[kn_idx] - z_last)
        obs_gr = pd.Series(gr[ev]).interpolate(limit_direction="both").fillna(float(np.nanmean(t_gr))).rolling(5, center=True, min_periods=1).mean().to_numpy()
        r = gen_oracle(md[ev], z[ev], tvt_last, z_last, t_min, t_max, tvt[ev], pmd_rel, pchange,
                       grsel=(t_tvt, t_gr, obs_gr))
        if r is None: nskip += 1; continue
        tr = tvt[ev]; fin = np.isfinite(tr)
        eo.append((r["oracle"] - tr)[fin]); eb.append((r["base"] - tr)[fin])
        if "gr" in r: eg.append((r["gr"] - tr)[fin])
        ncands.append(r["nc"]); nused += 1
        if (wi + 1) % 100 == 0: print(f"  {wi+1}/{len(wells)} ({time.time()-t0:.0f}s)", flush=True)
    rmse = lambda L: float(np.sqrt(np.mean(np.concatenate(L) ** 2)))
    print(f"\n[v91 physics prior] N={N} K={K} SLOPE={SLOPE*1000:.0f}/1000 DEV={DEV*1000:.0f} DIPSP={DIPSP*1000:.0f} ({nused}井,skip={nskip})")
    print(f"  ★oracle(真値で最良選択=天井)  = {rmse(eo):.4f}ft")
    print(f"  現実的: GRマッチ選択          = {rmse(eg):.4f}ft  ← 実際に近い(選択がGR頼み)")
    print(f"  無選択: prefix dip外挿(base)  = {rmse(eb):.4f}ft  ← 候補生成の中心の実力")
    print(f"  拘束通過候補数 中央{int(np.median(ncands))}/{N}")


if __name__ == "__main__":
    main()
