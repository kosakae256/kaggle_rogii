# -*- coding: utf-8 -*-
"""
v89 PFフリー画像ビルダー — PF/struct/他井戸(nbr)を一切使わない非逐次localizer用。
==========================================================================================
使える情報: GR / typewell / self(自prefix) / 自井戸の物理(Z, MD, lk錨, dip). 他井戸(nbr)・PF・struct禁止.
δ枠中心 = 平坦線(lk水平コピー, 純幾何・非逐次). 真δ=真−lk が ±45ft窓内98%.
学習target dtrue = median(y − lk) = 真のTVTオフセット.

チャネル(REP env で選択, 既定=all):
  GR×typewell:  c_d16 窓GR差/σ, c_ncc16/64/128/256, c_grad 勾配NCC,
                c_aff ★affine較正した絶対GRレベルマッチ(NCCが捨てる反エイリアス信号)
  GR×self:      c_self16/64 NCC(自prefix疑似tw), c_selfcov 被覆
  物理(自井戸):  c_lk lk錨stripe(δ0), c_mds MDランプ, c_z Z相対stripe
出力: v89/artifacts_v89/{OUT}.pkl (v64トレーナ互換)
使い方: REP=all OUT=pf_free.pkl python v89/build_pf_free_v89.py  (NWELLS=でスモーク)
"""
import os, sys, time, pickle
import numpy as np, pandas as pd
P0 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(P0, "v64"))
import build_images_v64 as bi
smooth, grid_from_pairs, winsum, ncc_img = bi.smooth, bi.grid_from_pairs, bi.winsum, bi.ncc_img
OFF, NB, ND, STEP, STR, W16, W64, SIG_LK = bi.OFF, bi.NB, bi.ND, bi.STEP, bi.STR, bi.W16, bi.W64, bi.SIG_LK
ART = os.path.join(P0, "v89", "artifacts_v89"); os.makedirs(ART, exist_ok=True)
SRC = os.environ.get("SRC", os.path.join(P0, "rogii-wellbore-geology-prediction", "train"))
OUT = os.path.join(ART, os.environ.get("OUT", "pf_free.pkl"))
REP = os.environ.get("REP", "all")
CENTER_NPZ = os.environ.get("CENTER_NPZ", "")
_CTR = {}
if CENTER_NPZ:
    for _f in CENTER_NPZ.split(","):
        _z = np.load(_f)
        for _k in _z.files: _CTR[_k] = _z[_k]
    print(f"[v89] bootstrap center: {len(_CTR)}井")


def main():
    t0 = time.time()
    files = sorted(f.split("__")[0] for f in os.listdir(SRC) if f.endswith("__horizontal_well.csv"))
    if os.environ.get("NWELLS"): files = files[:int(os.environ["NWELLS"])]
    OUTD = {}
    for wi, wn in enumerate(files):
        hw = pd.read_csv(os.path.join(SRC, f"{wn}__horizontal_well.csv"))
        tw = pd.read_csv(os.path.join(SRC, f"{wn}__typewell.csv")).sort_values("TVT")
        t_tvt = tw["TVT"].to_numpy(float); t_gr = tw["GR"].to_numpy(float)
        fin = np.isfinite(t_tvt) & np.isfinite(t_gr); t_tvt, t_gr = t_tvt[fin], t_gr[fin]
        if len(t_tvt) < 50: continue
        kn = hw["TVT_input"].notna().to_numpy(); ev = ~kn
        if kn.sum() < 10 or ev.sum() < 100: continue
        fb = float(np.nanmean(t_gr))
        gr_all = smooth(hw["GR"].to_numpy(float), fb, 2)
        g = gr_all[ev]; y = hw["TVT"].to_numpy(float)[ev]
        z_all = hw["Z"].to_numpy(float); md_all = hw["MD"].to_numpy(float)
        kn_tvt = hw["TVT_input"].to_numpy(float)[kn]
        lk = float(kn_tvt[-1])
        if _CTR and wn in _CTR:
            stv = np.asarray(_CTR[wn], np.float64)
            if len(stv) < len(g): stv = np.pad(stv, (0, len(g)-len(stv)), mode="edge")
        else:
            stv = np.full(len(g), lk, np.float64)             # ★平坦線中心(PF/struct/nbr不使用)
        T = min(len(g), len(stv)); g = g[:T]; yv = y[:T]; stv = stv[:T]
        z_ev = z_all[ev][:T]
        # σ較正(自prefix)
        tws = smooth(t_gr, fb, 2)
        resid = gr_all[kn] - np.interp(kn_tvt, t_tvt, tws)
        sigw = float(np.nanstd(resid)); sigw = 15.0 if not np.isfinite(sigw) or sigw < 3 else max(sigw, 5.0)
        pos = stv[:, None] + OFF[None, :]
        gmin, gmax = pos.min() - 1, pos.max() + 1
        Ttw = np.interp(pos, t_tvt, tws)
        ch = []; names = []
        def add(c, nm): ch.append(np.asarray(c, np.float32)); names.append(nm)
        want = lambda k: (REP == "all" or k in REP)
        # --- GR × typewell ---
        if want("d16"):
            d16, _ = winsum((g[:, None] - Ttw), W16); add((d16 / W16) / sigw, "d16")
        if want("ncc"):
            for WL in (W16, W64, 128, 256):
                c, _ = ncc_img(g, Ttw, WL); add(c, f"ncc{WL}")
        if want("grad"):
            c, _ = ncc_img(np.gradient(g), np.gradient(Ttw, axis=0), W16); add(c, "grad")
        if want("aff"):
            # ★affine較正: well prefix GR ≈ α·tw(TVT_input)+β を robust fit → tw を well GRスケールへ。
            #   その上で **絶対値マッチ**(NCCと違いレベルを保持=サイクル区別の反エイリアス信号)。
            twk = np.interp(kn_tvt, t_tvt, tws); v = np.isfinite(twk) & np.isfinite(gr_all[kn])
            if v.sum() >= 20 and np.std(twk[v]) > 1e-6:
                a, b = np.polyfit(twk[v], gr_all[kn][v], 1); a = float(np.clip(a, 0.3, 3.0))
            else:
                a, b = 1.0, 0.0
            Taff = a * Ttw + b
            resid_a = gr_all[kn] - (a * twk + b); sa = float(np.nanstd(resid_a[np.isfinite(resid_a)]))
            sa = sigw if not np.isfinite(sa) or sa < 3 else max(sa, 5.0)
            add(np.exp(-0.5 * ((g[:, None] - Taff) / sa) ** 2), "aff")     # 絶対レベルマッチ[0,1]
        # --- GR × self(自prefix) ---
        if want("self"):
            gs_, vs_ = grid_from_pairs(kn_tvt, gr_all[kn], gmin, gmax)
            Tself = np.interp(pos, gs_, np.where(np.isfinite(vs_), vs_, np.interp(gs_, t_tvt, tws)))
            Aself = np.interp(pos, gs_, np.isfinite(vs_).astype(float))
            c, _ = ncc_img(g, Tself, W16); add(c, "self16")
            c, _ = ncc_img(g, Tself, W64); add(c, "self64")
            a5s, _ = winsum(Aself, W16); add(a5s / W16, "selfcov")
        # --- 物理(自井戸) ---
        nb_ = (T + STR - 1) // STR
        sel = np.clip(np.arange(nb_) * STR + STR // 2, 0, T - 1)
        if want("lk"):
            add(np.repeat(np.exp(-0.5 * (OFF / SIG_LK) ** 2)[None, :], T, 0), "lk")   # δ0錨(lk)
        if want("mds"):
            md_lk = float(md_all[kn][-1]) if kn.sum() else float(md_all[0])
            mds_row = np.clip((md_all[ev][:T] - md_lk) / 1000.0, 0, 3.0) / 3.0
            add(np.repeat(mds_row[:, None], ND, 1), "mds")
        if want("z"):
            z_lk = float(z_all[kn][-1]) if kn.sum() else float(z_all[0])
            zr = np.clip((z_ev - z_lk) / 50.0, -3, 3) / 3.0
            add(np.repeat(zr[:, None], ND, 1), "z")
        EV = np.stack([c[sel] for c in ch], 0).astype(np.float16)
        bid = np.minimum(np.arange(T) // STR, nb_ - 1)
        dtrue = np.full(nb_, np.nan); wgt = np.zeros(nb_); dely = yv - stv
        for b in range(nb_):
            m = bid == b
            if m.sum(): dtrue[b] = float(np.nanmedian(dely[m])); wgt[b] = float(np.isfinite(dely[m]).sum())
        OUTD[wn] = dict(EV=EV, dtrue=dtrue.astype(np.float32), wgt=wgt.astype(np.float32),
                        st=stv.astype(np.float32), bid=bid.astype(np.int32), y_ev=yv.astype(np.float32))
        if (wi + 1) % 100 == 0: print(f"  {wi+1}/{len(files)} ({time.time()-t0:.0f}s)", flush=True)
    pickle.dump(OUTD, open(OUT, "wb"), protocol=4)
    din = np.median([np.mean(np.abs(d["dtrue"][np.isfinite(d["dtrue"])]) <= 45) for d in OUTD.values()])
    print(f"saved {os.path.basename(OUT)}: {len(OUTD)}井 {len(names)}ch [{','.join(names)}] "
          f"({os.path.getsize(OUT)/(1<<20):.0f}MB, {time.time()-t0:.0f}s) 真δ窓内中央{din:.0%}")


if __name__ == "__main__":
    main()
