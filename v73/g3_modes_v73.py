# -*- coding: utf-8 -*-
"""
v73 段g-3 — seedモード分解 + oracle測定(+学習ゲートの前段診断)
================================================================
32seedの軌跡は双峰坑井で別モードに割れる。現行合成=全seedのll-softmax(=posterior mean)。
v34(LB実証: 学習soft posterior-mean +0.039)のチェーン版を作る前に、賞金の上限を測る:
  - モード分解(truth不使用): per-seed offset o_s=median(ps_s−comb) を1Dソートし、gap>GAPで分割
  - mode-oracle: 正解モード(モード内posterior-meanのRMSE最小)を選べた場合の overall
  - mass-pick: 最大質量モードを選ぶ(非学習の基準)
oracle−現行 が大きければ学習ゲート(v34式)を実装する価値がある。
使い方: DUMPTAG=champ2 python v73/g3_modes_v73.py
"""
import os, io, sys, pickle
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
from pathlib import Path
import numpy as np

ART = Path(r"C:\Users\kosaka256\Documents\rogii_claude\v73\artifacts_v73")
TAG = os.environ.get("DUMPTAG", "champ2")
GAP = float(os.environ.get("GAP", "6.0"))
LS = 10.536319667317654
D = pickle.loads((ART / f"seed_dump_{TAG}.pkl").read_bytes())
wells = sorted(D.keys())
print(f"dump={TAG} {len(wells)}井  GAP={GAP}ft", flush=True)


def wsoft(ll):
    m = np.isfinite(ll)
    mx = np.nanmax(ll[m]) if m.any() else 0.0
    lw = np.where(m, ll - mx, -np.inf)
    ww = np.exp(lw / LS); s = ww.sum()
    return ww / s if s > 1e-300 else np.full(len(ll), 1.0 / len(ll))


se_base = se_orc = se_mass = 0.0; n_tot = 0
nmodes_all = []; corr_mass = []; orc_gain_wells = []
for w in wells:
    d = D[w]; err = d["err"].astype(np.float32)     # (S,T) = ps − true
    ll = d["ll"]; S, T = err.shape
    ww = wsoft(ll)
    ec = (ww[:, None] * err).sum(0)                 # 現行合成の誤差
    fin = np.isfinite(ec)
    n = fin.sum()
    if n == 0: continue
    # --- モード分解(truth不使用: ps_s−comb = err_s−ec) ---
    off = np.nanmedian(err - ec[None, :], axis=1)   # per-seed offset
    order = np.argsort(off)
    gaps = np.diff(off[order])
    cuts = np.where(gaps > GAP)[0]
    mode_id = np.zeros(S, int)
    mid = 0
    for j, si in enumerate(order):
        if j > 0 and (j - 1) in set(cuts): mid += 1
        mode_id[si] = mid
    M = mid + 1
    nmodes_all.append(M)
    # モードごとの posterior mean(モード内ll-softmax) と質量
    best_rmse = None; best_e = None; mass_best_e = None; mass_max = -1
    mass_of = np.zeros(M)
    for m_ in range(M):
        sel = mode_id == m_
        mass = ww[sel].sum(); mass_of[m_] = mass
        wwm = ww[sel] / max(ww[sel].sum(), 1e-12)
        em = (wwm[:, None] * err[sel]).sum(0)
        r = np.sqrt(np.nanmean(em[fin] ** 2))
        if best_rmse is None or r < best_rmse:
            best_rmse = r; best_e = em
        if mass > mass_max:
            mass_max = mass; mass_best_e = em
    corr_is_massmax = np.allclose(best_e, mass_best_e)
    corr_mass.append(corr_is_massmax if M > 1 else np.nan)
    se_base += np.nansum(ec[fin] ** 2)
    se_orc += np.nansum(best_e[fin] ** 2)
    se_mass += np.nansum(mass_best_e[fin] ** 2)
    n_tot += n
    if M > 1:
        r0 = np.sqrt(np.nanmean(ec[fin] ** 2)); r1 = best_rmse
        orc_gain_wells.append((w, r0, r1, M, mass_of.max()))

nm = np.array(nmodes_all)
print(f"\nモード数分布: 1つ={np.mean(nm==1)*100:.0f}% / 2つ={np.mean(nm==2)*100:.0f}% / 3+={np.mean(nm>=3)*100:.0f}%")
cm = np.array([c for c in corr_mass if c == c], dtype=float)
print(f"多モード坑井で「正解モード=最大質量」の率: {np.nanmean(cm)*100:.0f}%")
print(f"\n=== overall pooled RMSE ===")
print(f"  現行合成(全seed posterior-mean) : {np.sqrt(se_base/n_tot):.4f}")
print(f"  mass-pick(最大質量モード)        : {np.sqrt(se_mass/n_tot):.4f}")
print(f"  ★mode-oracle(正解モード)        : {np.sqrt(se_orc/n_tot):.4f}   ← 学習ゲートの上限")
og = sorted(orc_gain_wells, key=lambda t: t[1] - t[2], reverse=True)[:10]
print("\n多モード坑井の oracle 改善上位10:")
for w, r0, r1, M, mm in og:
    print(f"  {w}: {r0:6.2f} → {r1:6.2f}  (モード{M}, 最大質量{mm:.2f})")
