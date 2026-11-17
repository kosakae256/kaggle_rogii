# -*- coding: utf-8 -*-
# v64 — PF-free 2D UNet 用 GR照合像ビルダー(Tucker 3位再現路線 + 物理prior)
# =========================================================================
# v61 build_images_v61.py の6ch(全てPF不使用のGR照合)に、物理priorチャネル3枚を追加した9ch版:
#   c0 窓平均GR差/σ (W16)          c1 NCC vs typewell (W16)     c2 NCC vs typewell (W64)
#   c3 NCC vs self疑似tw (W16)     c4 NCC vs nbr疑似tw (W16)    c5 self/nbr被覆率
#   c6 struct信頼ストライプ exp(−½(δ/σ_struct)²)   … v38 struct prior の行別σ(物理prior)
#   c7 lk錨ストライプ exp(−½((δ−(lk−st))/3)²)      … 錨位置(δ枠)
#   c8 mdsランプ clip(mds/1000,0,3)/3               … 錨からの距離文脈(δ方向に一様)
# δ枠の中心 st = v51 SIMD の struct 軌跡(=中心自体が物理prior)。PF出力は一切使わない。
# -------------------------------------------------------------------------
# env: NWELLS(スモーク) / OUT(既定 imagesP_v64.pkl) / SRC(train dir; pseudo は pseudo_wells_* dir)
#      STMAP(pseudo用: id→donor の manifest.json パス。st/struct/nbr座標を donor から流用)
# 出力: v64/artifacts_v64/{OUT} {well: {EV:(9,nb,181)fp16, dtrue, wgt, st, bid, y_ev}}(v60トレーナ互換)
# 実行例:
#   real  : .venv/Scripts/python.exe v64/build_images_v64.py
#   pseudo: SRC=v63/artifacts_v63/pseudo_wells_p1 STMAP=v63/artifacts_v63/pseudo_wells_p1/manifest.json \
#           OUT=imagesP_v64_pseudo_p1.pkl .venv/Scripts/python.exe v64/build_images_v64.py
import os, sys, json, time, pickle
import numpy as np
import pandas as pd

P0 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(P0, "v51"))
ART = os.path.join(P0, "v64", "artifacts_v64"); os.makedirs(ART, exist_ok=True)
SRC = os.environ.get("SRC", os.path.join(P0, "rogii-wellbore-geology-prediction", "train"))
OUT = os.path.join(ART, os.environ.get("OUT", "imagesP_v64.pkl"))
STMAP = os.environ.get("STMAP", "")
NB = 90; ND = 181; STEP = 0.5; STR = 16
OFF = (np.arange(ND) - NB) * STEP
W16, W64 = 16, 64
SIG_LK = 3.0          # lk錨ストライプ幅(ft)
SIG_ST_FB = 6.0       # structσのフォールバック(ft)

def smooth(v, fb, r=2):
    s = pd.Series(v, dtype="float64").interpolate(limit_direction="both").fillna(fb)
    return s.rolling(2 * r + 1, center=True, min_periods=1).mean().to_numpy()

def grid_from_pairs(tvt, gr, gmin, gmax):
    """(TVT,GR)ペア→0.5ft格子。値なし=NaN"""
    g = np.arange(gmin, gmax + STEP, STEP)
    out = np.full(len(g), np.nan)
    idx = np.round((tvt - gmin) / STEP).astype(int)
    ok = (idx >= 0) & (idx < len(g)) & np.isfinite(gr) & np.isfinite(tvt)
    if ok.sum():
        ssum = np.zeros(len(g)); scnt = np.zeros(len(g))
        np.add.at(ssum, idx[ok], gr[ok]); np.add.at(scnt, idx[ok], 1)
        m = scnt > 0
        out[m] = ssum[m] / scnt[m]
    return g, out

def winsum(A, W):
    c = np.cumsum(np.concatenate([np.zeros((1,) + A.shape[1:]), A], 0), 0)
    lo = np.clip(np.arange(len(A)) - W // 2, 0, len(A))
    hi = np.clip(np.arange(len(A)) + W - W // 2, 0, len(A))
    return c[hi] - c[lo], (hi - lo).astype(np.float64)

def ncc_img(g, T, W):
    V = np.isfinite(T)
    Tz = np.where(V, T, 0.0)
    gs = g[:, None] * V
    n, _ = winsum(V.astype(np.float64), W)
    sg, _ = winsum(gs, W); st_, _ = winsum(Tz, W)
    sgg, _ = winsum((g[:, None] ** 2) * V, W); stt, _ = winsum(Tz ** 2, W)
    sgt, _ = winsum(g[:, None] * Tz, W)
    n = np.maximum(n, 1)
    mg = sg / n; mt = st_ / n
    vg = np.maximum(sgg / n - mg ** 2, 1e-6); vt = np.maximum(stt / n - mt ** 2, 1e-6)
    ncc = (sgt / n - mg * mt) / np.sqrt(vg * vt)
    return np.clip(ncc, -1, 1), (n / W)

def main():
    import nnpf_v51 as npf
    t0 = time.time()
    STRUCT = pickle.loads(open(os.path.join(P0, "v38", "artifacts_v38", "struct_prior_train.pkl"), "rb").read())
    stmap = {}
    if STMAP:
        man = json.loads(open(STMAP, encoding="utf-8").read())
        stmap = {r["id"]: r["donor"] for r in man["wells"]}
    ST_SIDE = {}
    _stp = os.path.join(SRC, "pseudo_st.npz")
    if os.path.exists(_stp):
        _z = np.load(_stp)
        ST_SIDE = {k: _z[k] for k in _z.files}
        print(f"[st] sidecar pseudo_st.npz: {sum(1 for k in ST_SIDE if not k.endswith('__sig'))} wells")
    NBR_SIDE = {}
    _nbp = os.path.join(SRC, "pseudo_nbr.npz")
    if os.path.exists(_nbp):
        _z = np.load(_nbp)
        NBR_SIDE = {k: _z[k] for k in _z.files}
        print(f"[nbr] sidecar pseudo_nbr.npz: {sum(1 for k in NBR_SIDE if not k.endswith('__gr'))} wells")
    files = sorted(f.split("__")[0] for f in os.listdir(SRC) if f.endswith("__horizontal_well.csv"))
    if os.environ.get("NWELLS"): files = files[:int(os.environ["NWELLS"])]
    # nbrプール(実trainから。pseudoもdonor座標近傍なので実プールでOK。donorは除外=リーク防止)
    cache = pickle.loads(open(os.path.join(P0, "v50", "artifacts_v50", "cacheA_v50.pkl"), "rb").read())
    realw = sorted(cache.keys())
    XY = np.array([cache[w]["xy"] for w in realw])
    KN = {w: ((cache[w]["y"][(~cache[w]["ev"]) & np.isfinite(cache[w]["y"])]).astype(float),
              (cache[w]["graw"][(~cache[w]["ev"]) & np.isfinite(cache[w]["y"])]).astype(float)) for w in realw}
    from scipy.spatial import cKDTree
    tree = cKDTree(XY)
    OUTD = {}
    for wi, wn in enumerate(files):
        hw = pd.read_csv(os.path.join(SRC, f"{wn}__horizontal_well.csv"))
        tw = pd.read_csv(os.path.join(SRC, f"{wn}__typewell.csv")).sort_values("TVT")
        t_tvt = tw["TVT"].to_numpy(float); t_gr = tw["GR"].to_numpy(float)
        fin = np.isfinite(t_tvt) & np.isfinite(t_gr); t_tvt, t_gr = t_tvt[fin], t_gr[fin]
        if len(t_tvt) < 50: continue
        kn = hw["TVT_input"].notna().to_numpy(); ev = ~kn
        if kn.sum() < 10 or ev.sum() < 100: continue
        donor = stmap.get(wn, wn)
        sig_side = None
        if wn in ST_SIDE:
            st = np.asarray(ST_SIDE[wn], np.float64)          # 生成器の疑似struct(真値近傍に較正済)
            sig_side = ST_SIDE.get(wn + "__sig")
        else:
            sd = npf.SIMD.get(donor)
            if sd is None: continue
            st = np.asarray(sd["st"], np.float64)
        fb = float(np.nanmean(t_gr))
        gr_all = smooth(hw["GR"].to_numpy(float), fb, 2)
        g = gr_all[ev]
        y = hw["TVT"].to_numpy(float)[ev]
        T = min(len(g), len(st)); g = g[:T]; yv = y[:T]; stv = st[:T]
        if STMAP and wn not in ST_SIDE:
            # pseudo(sidecar無し=v63系): surface型は donor st から数十ft ずれ得る(±45窓から漏れる)。
            # 井戸定数シフトで粗く再センタリングし、±10ft ジッタで「structは粗い」状況は残す。
            import zlib
            _rw_ = np.random.default_rng(zlib.crc32(wn.encode()) & 0xFFFFFFFF)
            _shift = float(np.nanmedian(yv - stv)) + float(_rw_.uniform(-10, 10))
            if np.isfinite(_shift):
                stv = stv + _shift
        # σ較正(prefix)
        kn_tvt = hw["TVT_input"].to_numpy(float)[kn]
        resid = gr_all[kn] - np.interp(kn_tvt, t_tvt, smooth(t_gr, fb, 2))
        sigw = float(np.nanstd(resid)); sigw = 15.0 if not np.isfinite(sigw) or sigw < 3 else max(sigw, 5.0)
        # プロファイル行列 (T,181)
        pos = stv[:, None] + OFF[None, :]
        tws = smooth(t_gr, fb, 2)
        Ttw = np.interp(pos, t_tvt, tws)
        gmin, gmax = pos.min() - 1, pos.max() + 1
        # self疑似
        gs_, vs_ = grid_from_pairs(kn_tvt, gr_all[kn], gmin, gmax)
        Tself = np.interp(pos, gs_, np.where(np.isfinite(vs_), vs_, np.interp(gs_, t_tvt, tws)))
        Aself = np.interp(pos, gs_, np.isfinite(vs_).astype(float))
        # nbr疑似(donor座標の近傍、donor自身は除外)
        if donor in cache:
            _, nidx = tree.query(cache[donor]["xy"], k=9)
            js = [realw[j] for j in np.atleast_1d(nidx) if realw[j] != donor][:8]
        else:
            js = []
        if wn in NBR_SIDE:
            # q2: 合成近傍プール(synthcol等、c4縮退の解消)
            gn_, vn_ = grid_from_pairs(np.asarray(NBR_SIDE[wn], float), np.asarray(NBR_SIDE[wn + "__gr"], float), gmin, gmax)
            Tnbr = np.interp(pos, gn_, np.where(np.isfinite(vn_), vn_, np.interp(gn_, t_tvt, tws)))
            Anbr = np.interp(pos, gn_, np.isfinite(vn_).astype(float))
        elif js:
            ny = np.concatenate([KN[j][0] for j in js]); ng = np.concatenate([KN[j][1] for j in js])
            gn_, vn_ = grid_from_pairs(ny, ng, gmin, gmax)
            Tnbr = np.interp(pos, gn_, np.where(np.isfinite(vn_), vn_, np.interp(gn_, t_tvt, tws)))
            Anbr = np.interp(pos, gn_, np.isfinite(vn_).astype(float))
        else:
            Tnbr = Ttw; Anbr = np.zeros_like(Ttw)
        # ---- チャネル計算(行→bin間引き) ----
        d16, _ = winsum((g[:, None] - Ttw), W16)
        c0 = (d16 / W16) / sigw
        c1, _ = ncc_img(g, Ttw, W16)
        c2, _ = ncc_img(g, Ttw, W64)
        c3, _ = ncc_img(g, Tself, W16)
        c4, _ = ncc_img(g, Tnbr, W16)
        a5s, _ = winsum(Aself, W16); a5n, _ = winsum(Anbr, W16)
        c5 = 0.5 * (a5s / W16) + 0.5 * (a5n / W16)
        nb_ = (T + STR - 1) // STR
        sel = np.arange(nb_) * STR + STR // 2
        sel = np.clip(sel, 0, T - 1)
        # ---- 物理priorチャネル(v64追加) ----
        if sig_side is not None and len(sig_side) >= T:
            sig_row = np.clip(np.asarray(sig_side, np.float64)[:T], 1.0, 45.0)
        else:
            sp = STRUCT.get(donor)
            if sp is not None and len(np.asarray(sp["sig"])) >= T:
                sig_row = np.clip(np.asarray(sp["sig"], np.float64)[:T], 1.0, 45.0)
            else:
                sig_row = np.full(T, SIG_ST_FB)
        c6_b = np.exp(-0.5 * (OFF[None, :] / sig_row[sel][:, None]) ** 2)          # (nb,181)
        lk = float(kn_tvt[-1])
        zl_b = lk - stv[sel]                                                        # 錨のδ位置
        c7_b = np.exp(-0.5 * ((OFF[None, :] - zl_b[:, None]) / SIG_LK) ** 2)
        md_all = hw["MD"].to_numpy(float)
        md_lk = float(md_all[kn][-1]) if kn.sum() else float(md_all[0])
        mds_row = np.maximum(md_all[ev][:T] - md_lk, 0.0)
        c8_b = np.repeat((np.clip(mds_row[sel] / 1000.0, 0, 3.0) / 3.0)[:, None], ND, 1)
        EV = np.concatenate([np.stack([c[sel] for c in (c0, c1, c2, c3, c4, c5)], 0),
                             np.stack([c6_b, c7_b, c8_b], 0)], 0).astype(np.float16)
        bid = np.minimum(np.arange(T) // STR, nb_ - 1)
        dtrue = np.full(nb_, np.nan)
        wgt = np.zeros(nb_)
        dely = yv - stv
        for b in range(nb_):
            m = bid == b
            if m.sum():
                dtrue[b] = float(np.nanmedian(dely[m])); wgt[b] = float(np.isfinite(dely[m]).sum())
        OUTD[wn] = dict(EV=EV, dtrue=dtrue.astype(np.float32), wgt=wgt.astype(np.float32),
                        st=stv.astype(np.float32), bid=bid.astype(np.int32), y_ev=yv.astype(np.float32))
        if (wi + 1) % 100 == 0: print(f"  {wi+1}/{len(files)} ({time.time()-t0:.0f}s)", flush=True)
    pickle.dump(OUTD, open(OUT, "wb"), protocol=4)
    sz = os.path.getsize(OUT) / (1 << 20)
    print(f"saved {os.path.basename(OUT)}: {len(OUTD)} wells 9ch ({sz:.0f}MB, {time.time()-t0:.0f}s)")

if __name__ == "__main__":
    main()
