# -*- coding: utf-8 -*-
"""
v89 (位置,レート)グリッド・ベイズ・フィルタ — PF/struct/nbr 不使用で自己相似を越える。
================================================================================================
v48格子(位置のみ,forward)はstruct無しだと自己相似で潰れる(structが枝を破っていた)。
→ **レート状態(dTVT/dMD=dip)を持たせる**。レートの慣性で真の降下経路を辿れる(PFが8.7を出す原理)。
状態=(δ=TVT−lk, rate)。遷移: δ←δ+rate·dMD(+pos noise), rate←rate(+rate noise)。emission=GR×typewell照合。
forward-backward で posterior平均δ → 予測=lk+δ。**粒子でなくグリッド=PFでない**。struct/nbr一切不使用。
使い方: NW=200 python v89/grid_fb_v89.py   (env: SIGP, SIGR, DGRID, RGRID)
"""
import os, io, sys, time
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
from pathlib import Path
import numpy as np
PROJECT = Path(r"C:\Users\kosaka256\Documents\rogii_claude"); sys.path.insert(0, str(PROJECT / "v65"))
os.environ["V65_TWSRC"] = "tw"
import pf_engine_v65 as base
import warmup_v65 as wu

NW = int(os.environ.get("NW", "200")) or None
DMAX = float(os.environ.get("DGRID", "60")); DSTP = 0.5
RMAX = float(os.environ.get("RGRID", "0.05")); RSTP = float(os.environ.get("RSTP", "0.0025"))
SIGP = float(os.environ.get("SIGP", "1.5"))     # 位置遷移ノイズ(ft/step)
SIGR = float(os.environ.get("SIGR", "0.004"))   # レート遷移ノイズ
STRIDE = int(os.environ.get("STRIDE", "3"))     # eval行間引き(速度)
DG = np.arange(-DMAX, DMAX + 1e-9, DSTP); ND = len(DG)
RG = np.arange(-RMAX, RMAX + 1e-9, RSTP); NR = len(RG)
P = base.full_params(dict(wu.R0_PARAMS)); P["use_anchor"] = False
GP = float(P["gr_power"])


def _gauss_kernel(sig, step, n):
    r = max(1, int(round(3 * sig / step))); x = np.arange(-r, r + 1) * step
    k = np.exp(-0.5 * (x / sig) ** 2); return k / k.sum()


KR = _gauss_kernel(SIGR, RSTP, NR)              # レート拡散カーネル
KP = _gauss_kernel(SIGP, DSTP, ND)              # 位置拡散カーネル


def filt_well(c):
    tw_tvt = np.asarray(c["tw_tvt"], float); tw_gr0 = base._smooth(c["tw_gr"], c["tw_fb"], P["tw_gr_smooth_r"]).astype(float)
    _o = np.argsort(tw_tvt); tw_tvt = tw_tvt[_o]; tw_gr = tw_gr0[_o]   # ★np.interp用に昇順ソート
    gr_full = base._smooth(c["gr_filled"], c["tw_fb"], P["hgr_smooth_r"]).astype(float)
    kpos = c["kpos"]; ktvt = c["kn_tvtin"]
    if len(kpos) >= 20:
        resid = gr_full[kpos] - np.interp(ktvt, tw_tvt, tw_gr); gs = float(np.nanstd(resid))
        gs = P["gr_sig_def"] if not np.isfinite(gs) or gs <= 0 else gs
    else:
        gs = P["gr_sig_def"]
    gs = float(np.clip(gs * P["gr_sig_mult"], P["gr_sig_min"], max(P["gr_sig_max"], P["gr_sig_min"] + 1e-6)))
    lk = float(ktvt[-1])
    md = c["md"].astype(float); obs = gr_full[c["ev_pos"]].astype(float)
    T = len(md); sel = np.arange(0, T, STRIDE)
    mds = md[sel]; obss = obs[sel]
    # emission L[t, δ] = GR照合(自己相似で多峰)
    eg = np.interp(lk + DG, tw_tvt, tw_gr)                        # (ND,)
    d = np.abs((obss[:, None] - eg[None, :]) / gs) ** GP
    L = np.where(d < 600, np.exp(-0.5 * d), 0.0); L = np.clip(L, 1e-30, None)
    # forward
    a = np.zeros((len(sel), NR, ND))
    al = np.exp(-0.5*(RG[:,None]/0.015)**2)*np.ones((1,ND))   # 初期rate≈0近傍
    al /= al.sum()
    al = al * L[0][None, :]; al /= al.sum(); a[0] = al
    shifts = RG                                                   # rate(ft/ft)×dMD → δシフト量(ft)
    for t in range(1, len(sel)):
        dm = max(mds[t] - mds[t - 1], 1.0)
        pr = np.apply_along_axis(lambda v: np.convolve(v, KR, "same"), 0, a[t - 1])   # rate拡散(rate軸)
        # δシフト by rate·dm(各rate行を線形補間シフト)
        nx = np.empty_like(pr)
        for ri in range(NR):
            sft = shifts[ri] * dm                                # ft
            nx[ri] = np.interp(DG - sft, DG, pr[ri], left=0.0, right=0.0)
        # 位置拡散
        nx = np.apply_along_axis(lambda v: np.convolve(v, KP, "same"), 1, nx)
        nx = nx * L[t][None, :]; s = nx.sum(); a[t] = nx / s if s > 1e-30 else nx + 1.0 / (NR * ND)
    # backward smoother(簡易: rate周辺化した位置marginalのFB)
    md_delta = np.gradient(mds)
    b = np.ones((len(sel), NR, ND));
    for t in range(len(sel) - 2, -1, -1):
        dm = max(mds[t + 1] - mds[t], 1.0)
        bt = b[t + 1] * a[t + 1]
        # 逆遷移(近似: 同カーネルで戻す)
        bt = np.apply_along_axis(lambda v: np.convolve(v, KR, "same"), 0, bt)
        nx = np.empty_like(bt)
        for ri in range(NR):
            nx[ri] = np.interp(DG + shifts[ri] * dm, DG, bt[ri], left=0.0, right=0.0)
        nx = np.apply_along_axis(lambda v: np.convolve(v, KP, "same"), 1, nx)
        s = nx.sum(); b[t] = nx / s if s > 1e-30 else nx
    FB = os.environ.get("FB","0")=="1"
    post = (a * b) if FB else a
    postd = post.sum(1)
    postd = postd / postd.sum(1, keepdims=True).clip(1e-30)
    dhat = (postd * DG[None, :]).sum(1)                          # E[δ]
    # 全eval行へ補間
    full = np.interp(md, mds, lk + dhat)
    return full


def main():
    wells = base.list_wells()
    if NW: wells = wells[:NW]
    t0 = time.time(); es = []; done = 0
    for w in wells:
        c = base.build_cache(w)
        if c is None or c["tvt_true"] is None: continue
        try:
            pred = filt_well(c)
        except Exception as ex:
            continue
        t = np.asarray(c["tvt_true"], float); n = min(len(pred), len(t))
        e = pred[:n] - t[:n]; es.append(e[np.isfinite(e)]); done += 1
        if done % 50 == 0: print(f"  {done}井 ({time.time()-t0:.0f}s)", flush=True)
    e = np.concatenate(es)
    print(f"\n[gridFB] SIGP={SIGP} SIGR={SIGR} STRIDE={STRIDE} DGRID±{DMAX} RGRID±{RMAX}")
    print(f"  {done}井 overall RMSE = {np.sqrt(np.mean(e**2)):.4f}  (PF8.72/平坦15.9/U-Net12.9と比較, <10目標)  {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
