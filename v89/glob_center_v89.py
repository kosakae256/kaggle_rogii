# -*- coding: utf-8 -*-
"""
v89 グローバルGR整列 中心 — own-well情報だけで作る非逐次の中心trajectory(PF/struct/nbr不使用)。
============================================================================================
局所GR照合は自己相似で両義だが、**坑井全体でGR相関を最大化するdip/offset**は真の値を選ぶ
(長い坑井では真dipだけが全体整合を保つ)。既知帯lkでアンカー(b≈0)。
中心 = lk + a*·(MD−MD_lk) + b*(小)。これをU-Netのδ枠中心にすれば flat(15.9)より良い出発点。
まず中心単体のRMSE(vs 平坦15.9)を測る。良ければ pf中心の代わりに使ってU-Net学習。
出力: v89/artifacts_v89/glob_center.npz {well: ev行 中心TVT}
使い方: NW=200 python v89/glob_center_v89.py  (SAVE=1 で全井dump)
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
SAVE = os.environ.get("SAVE", "0") == "1"
AGRID = np.arange(-0.05, 0.05 + 1e-9, 0.002)     # dip探索
BGRID = np.arange(-14, 14 + 1e-9, 1.0)           # offset探索(±1cycle弱, アンカー付近)
NSEG = int(os.environ.get("NSEG", "2"))          # 区分dip(1=単一, 2=heel/toe)
P = base.full_params(dict(wu.R0_PARAMS)); P["use_anchor"] = False


def glob_center(c):
    tw_tvt = np.asarray(c["tw_tvt"], float); tw_gr = base._smooth(c["tw_gr"], c["tw_fb"], P["tw_gr_smooth_r"]).astype(float)
    o = np.argsort(tw_tvt); tw_tvt, tw_gr = tw_tvt[o], tw_gr[o]
    gr_full = base._smooth(c["gr_filled"], c["tw_fb"], P["hgr_smooth_r"]).astype(float)
    lk = float(c["kn_tvtin"][-1]); md = c["md"].astype(float)
    obs = gr_full[c["ev_pos"]].astype(float)
    md0 = md - md[0]
    ss = np.linspace(0, len(md), NSEG + 1).astype(int)
    center = np.full(len(md), lk)
    prev_end = lk; prev_md = 0.0
    for si in range(NSEG):
        sl = slice(ss[si], ss[si + 1])
        m = md0[sl]; ob = obs[sl]
        base_tvt = prev_end + 0.0 * (m - prev_md)     # 前区間の終端から連続
        best = (-1e9, 0.0, 0.0)
        for a in AGRID:
            for b in BGRID:
                hyp = prev_end + a * (m - prev_md) + (b if si == 0 else 0.0)  # 1区間目のみoffset探索(アンカー)
                eg = np.interp(hyp, tw_tvt, tw_gr)
                v = np.isfinite(eg) & np.isfinite(ob)
                if v.sum() < 20: continue
                sc = float(np.corrcoef(ob[v], eg[v])[0, 1])   # 全体相関を最大化
                if np.isfinite(sc) and sc > best[0]: best = (sc, a, b)
        _, a, b = best
        center[sl] = prev_end + a * (m - prev_md) + (b if si == 0 else 0.0)
        prev_end = center[sl][-1]; prev_md = m[-1]
    return center


def main():
    wells = base.list_wells()
    if NW: wells = wells[:NW]
    es = []; efl = []; out = {}; done = 0; t0 = time.time()
    for w in wells:
        c = base.build_cache(w)
        if c is None or c["tvt_true"] is None: continue
        try:
            ctr = glob_center(c)
        except Exception:
            continue
        t = np.asarray(c["tvt_true"], float); n = min(len(ctr), len(t))
        e = ctr[:n] - t[:n]; es.append(e[np.isfinite(e)])
        lk = float(c["kn_tvtin"][-1]); efl.append((lk - t[:n])[np.isfinite(t[:n])])
        out[w] = ctr.astype(np.float32); done += 1
        if done % 50 == 0: print(f"  {done}井 ({time.time()-t0:.0f}s)", flush=True)
    e = np.concatenate(es); ef = np.concatenate(efl)
    print(f"\n[glob-center] NSEG={NSEG} {done}井")
    print(f"  グローバル整列中心 RMSE = {np.sqrt(np.mean(e**2)):.4f}  (平坦線={np.sqrt(np.mean(ef**2)):.4f} / PF8.72)")
    if SAVE:
        np.savez_compressed(PROJECT / "v89" / "artifacts_v89" / "glob_center.npz", **out)
        print(f"  saved glob_center.npz ({len(out)}井)")


if __name__ == "__main__":
    main()
