# -*- coding: utf-8 -*-
"""
v91 × v89 — 物理候補(幾何拘束)を U-Net(v89)の学習emission δ分布で採点して選択。
================================================================================
raw GRマッチ選択は22ft(サイクル縮退で破綻)。U-Netの学習emission(ゴースト抑制, 双峰でtop2に真枝69.6%)で
物理候補を採点すれば、幾何拘束×学習GRで選択が改善するか? fold0 val井(v89_dist_f0)で検証。
比較: oracle(真値最良) / U-Net採点選択 / GRマッチ選択 / base。
"""
import os, sys, io, time, pickle
try: sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
except Exception: pass
import numpy as np, pandas as pd
from pathlib import Path
PROJ = Path(r"C:\Users\kosaka256\Documents\rogii_claude")
TR = PROJ / "rogii-wellbore-geology-prediction" / "train"
A = PROJ / "v64" / "artifacts_v64"; AV = PROJ / "v89" / "artifacts_v89"
N = int(os.environ.get("N", "5000")); K = int(os.environ.get("K", "4"))
SLOPE = float(os.environ.get("SLOPE", "55")) / 1000.0
DEV = float(os.environ.get("DEV", "25")) / 1000.0; DIPSP = float(os.environ.get("DIPSP", "20")) / 1000.0
W_EV = float(os.environ.get("W_EV", "0.3"))
CONT = float(os.environ.get("CONT", "4.0")); NW = int(os.environ.get("NW", "0")) or None
NB = 90; ND = 181; STEP = 0.5
rng = np.random.default_rng(0)


def gen_cands(md, z, tvt_last, z_last, t_min, t_max, pmd_rel, pchange):
    """物理候補(N,L) を生成(v91と同一: prefix dip中心)。返り値 cand と md_rel。"""
    L = len(md); md_rel = md - md[0]; dZ = z - z_last
    if len(pmd_rel) >= 5 and (pmd_rel[-1] - pmd_rel[0]) > 1:
        pdip = float(np.clip(np.polyfit(pmd_rel, pchange, 1)[0], -SLOPE, SLOPE))
    else:
        pdip = 0.0
    xk = np.linspace(0.0, md_rel[-1] + 1e-6, K); seg = np.diff(xk)
    dip_n = pdip + rng.uniform(-DIPSP, DIPSP, size=(N, 1))
    dyk = np.concatenate([np.zeros((N, 1)), np.cumsum(rng.normal(0, DEV, (N, K - 1)) * seg[None, :], axis=1)], axis=1)
    dev = np.empty((N, L));
    for i in range(N): dev[i] = np.interp(md_rel, xk, dyk[i])
    cand = tvt_last + dip_n * md_rel[None, :] + dev - dZ[None, :]
    ok = ((cand >= t_min - 5) & (cand <= t_max + 5)).all(axis=1)
    nc = min(30, L); ok &= (np.abs(cand[:, :nc] - tvt_last) < CONT).all(axis=1)
    if not ok.any(): return None, None, pdip
    return cand[ok], md_rel, pdip


def main():
    t0 = time.time()
    DIST = dict(np.load(A / "v89_dist_f0_dist.npz"))          # U-Net δ分布(fold0 val, per bin)
    EVD = pickle.loads(open(AV / "learned_boot.pkl", "rb").read())  # st,bid,y_ev
    PF = dict(np.load(AV / "pf_center_v89.npz"))              # ★r0 PF軌跡(物理平滑化の対象)
    wells = [w for w in DIST if w in EVD]
    if NW: wells = wells[:NW]
    eo = []; eu = []; eg = []; ec = []; er0 = []; eps_ = []; ea = []; eag = []; eau = []; nused = 0
    for wi, wn in enumerate(wells):
        try:
            hw = pd.read_csv(TR / f"{wn}__horizontal_well.csv"); tw = pd.read_csv(TR / f"{wn}__typewell.csv")
        except Exception:
            continue
        kn = hw["TVT_input"].notna().to_numpy()
        if kn.sum() < 10 or (~kn).sum() < 30: continue
        hps = int(np.where(kn)[0].max()); ev = slice(hps + 1, len(hw))
        z = hw["Z"].to_numpy(float); md = hw["MD"].to_numpy(float); tvt = hw["TVT"].to_numpy(float)
        ti = hw["TVT_input"].to_numpy(float); gr = hw["GR"].to_numpy(float)
        t_tvt = tw["TVT"].to_numpy(float); t_gr = tw["GR"].to_numpy(float)
        m = np.isfinite(t_tvt) & np.isfinite(t_gr); t_tvt, t_gr = t_tvt[m], t_gr[m]
        if len(t_tvt) < 8: continue
        o = np.argsort(t_tvt); t_tvt, t_gr = t_tvt[o], t_gr[o]
        tvt_last = float(ti[hps]); z_last = float(z[hps])
        ki = np.where(kn)[0]; ki = ki[ki >= max(0, hps - 80)]
        pmd_rel = md[ki] - md[hps]; pchange = (ti[ki] - tvt_last) + (z[ki] - z_last)
        cand, md_rel, pdip = gen_cands(md[ev], z[ev], tvt_last, z_last, float(t_tvt.min()), float(t_tvt.max()), pmd_rel, pchange)
        if cand is None: continue
        true = tvt[ev]; fin = np.isfinite(true); L = len(true)
        # --- U-Net δ分布で採点 ---
        st = np.asarray(EVD[wn]["st"], float); bid = np.asarray(EVD[wn]["bid"], int)
        P = np.asarray(DIST[wn], float)                          # (nb,ND)
        Ln = min(L, len(st), len(bid))
        c = cand[:, :Ln]; stc = st[:Ln]; bidc = bid[:Ln]
        delta = c - stc[None, :]                                 # (Ncand,Ln) δ
        jf = np.round(delta / STEP + NB).astype(int)
        inb = (jf >= 0) & (jf < ND)
        jf = np.clip(jf, 0, ND - 1)
        logp = np.log(P[bidc[None, :], jf] + 1e-6)               # (Ncand,Ln)
        logp = np.where(inb, logp, -12.0)
        uscore = logp.sum(axis=1)                                 # U-Net尤度(大=良)
        u_best = cand[int(np.argmax(uscore))]
        # --- GRマッチ選択 ---
        obs = pd.Series(gr[ev]).interpolate(limit_direction="both").fillna(float(np.nanmean(t_gr))).rolling(5, center=True, min_periods=1).mean().to_numpy()
        gpred = np.interp(cand, t_tvt, t_gr); gscore = np.mean((gpred - obs[None, :]) ** 2, axis=1)
        g_best = cand[int(np.argmin(gscore))]
        # --- oracle ---
        o_best = cand[int(np.argmin(np.sqrt(np.mean((cand - true[None, :]) ** 2, axis=1))))]
        eo.append((o_best - true)[fin]); eu.append((u_best - true)[fin]); eg.append((g_best - true)[fin])
        # 井内z正規化(候補次元)。GR=小良→符号反転して"大良"に統一, U-Net logp=大良。
        zn = lambda v: (v - v.mean()) / (v.std() + 1e-9)
        s_g = -zn(gscore); s_u = zn(uscore)
        # --- 複合採点(GR+U-Net) ---
        ec.append((cand[int(np.argmax(s_g + s_u))] - true)[fin])
        # --- ★物理正則化: r0(PF, 現状最良8.03)を錨に GR+U-Net で精緻化 ---
        if wn in PF:
            r0 = np.asarray(PF[wn], float)[:L]; lr = len(r0)
            finr = np.isfinite(true[:lr])
            er0.append((r0 - true[:lr])[finr])
            r0d = np.sqrt(np.mean((cand[:, :lr] - r0[None, :]) ** 2, axis=1))
            s_r = -zn(r0d)                                    # r0近い=大良
            ps = cand[int(np.argmax(s_r))]                    # r0スナップ(=物理平滑化)
            eps_.append((ps - true)[fin])
            # 錨s_r をベースに GR+U-Net を弱く足す(W_EV掃引): 物理平滑を証拠でずらす
            comb = s_r + W_EV * (s_g + s_u)
            ea.append((cand[int(np.argmax(comb))] - true)[fin])
            eag.append((cand[int(np.argmax(s_r + W_EV * s_g))] - true)[fin])   # 錨+GRのみ
            eau.append((cand[int(np.argmax(s_r + W_EV * s_u))] - true)[fin])   # 錨+U-Netのみ
        nused += 1
        if (wi + 1) % 30 == 0: print(f"  {wi+1}/{len(wells)} ({time.time()-t0:.0f}s)", flush=True)
    r = lambda L: float(np.sqrt(np.mean(np.concatenate(L) ** 2)))
    print(f"\n[v91×v89 物理候補×採点] N={N} K={K} W_EV={W_EV} ({nused}井 fold0)")
    print(f"  oracle(真値最良)          = {r(eo):.4f}ft")
    print(f"  U-Net δ分布 単独          = {r(eu):.4f}ft")
    print(f"  GRマッチ 単独             = {r(eg):.4f}ft")
    print(f"  GR+U-Net 複合            = {r(ec):.4f}ft")
    print(f"  --- 物理正則化(r0錨) ---")
    print(f"  r0素                      = {r(er0):.4f}ft")
    print(f"  r0スナップ(物理平滑)       = {r(eps_):.4f}ft")
    print(f"  ★r0錨+証拠(W_EV={W_EV})     = {r(ea):.4f}ft  ← 錨を証拠で精緻化して改善するか")
    print(f"    (錨+GRのみ={r(eag):.4f} / 錨+U-Netのみ={r(eau):.4f})")


if __name__ == "__main__":
    main()
