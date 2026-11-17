# v61 P0 — GR照合像ビルダー(docs/v61.md)
#   出力: imagesR_v61.pkl {well: {EV:(6,nb,181)fp16, dtrue, wgt, st, bid, y_ev}}(v60トレーナ互換)
#   env: NWELLS(スモーク), OUT(既定 imagesR_v61.pkl), SRC(train dir; 合成用に差し替え可),
#        STMAP(合成用: synth→donor の st 流用 manifest.json パス)
import os, sys, json, time, pickle
import numpy as np
import pandas as pd

P0 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(P0, "v51"))
ART = os.path.join(P0, "v61", "artifacts_v61"); os.makedirs(ART, exist_ok=True)
SRC = os.environ.get("SRC", os.path.join(P0, "rogii-wellbore-geology-prediction", "train"))
OUT = os.path.join(ART, os.environ.get("OUT", "imagesR_v61.pkl"))
STMAP = os.environ.get("STMAP", "")
NB = 90; ND = 181; STEP = 0.5; STR = 16
OFF = (np.arange(ND) - NB) * STEP
W16, W64 = 16, 64

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
    """行方向の移動和(centered, same length)。A (T,) or (T,D)"""
    c = np.cumsum(np.concatenate([np.zeros((1,) + A.shape[1:]), A], 0), 0)
    lo = np.clip(np.arange(len(A)) - W // 2, 0, len(A))
    hi = np.clip(np.arange(len(A)) + W - W // 2, 0, len(A))
    return c[hi] - c[lo], (hi - lo).astype(np.float64)

def ncc_img(g, T, W):
    """g (T,), T(prof) (T,D) → NCC (T,D)(窓W, NaN対応: 有効率も返す)"""
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
    stmap = {}
    if STMAP:
        man = json.loads(open(STMAP).read())
        stmap = {r["id"]: r["donor"] for r in man["wells"]}
    files = sorted(f.split("__")[0] for f in os.listdir(SRC) if f.endswith("__horizontal_well.csv"))
    if os.environ.get("NWELLS"): files = files[:int(os.environ["NWELLS"])]
    # nbrプール(実trainから。合成もdonor座標なので実プールでOK)
    TRR = os.path.join(P0, "rogii-wellbore-geology-prediction", "train")
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
        sd = npf.SIMD.get(donor)
        if sd is None: continue
        st = np.asarray(sd["st"], np.float64)
        fb = float(np.nanmean(t_gr))
        gr_all = smooth(hw["GR"].to_numpy(float), fb, 2)
        g = gr_all[ev]
        y = hw["TVT"].to_numpy(float)[ev]
        T = min(len(g), len(st)); g = g[:T]; yv = y[:T]; stv = st[:T]
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
        # nbr疑似(donor座標の近傍)
        if donor in cache:
            _, nidx = tree.query(cache[donor]["xy"], k=9)
            js = [realw[j] for j in np.atleast_1d(nidx) if realw[j] != donor][:8]
        else:
            js = []
        if js:
            ny = np.concatenate([KN[j][0] for j in js]); ng = np.concatenate([KN[j][1] for j in js])
            gn_, vn_ = grid_from_pairs(ny, ng, gmin, gmax)
            Tnbr = np.interp(pos, gn_, np.where(np.isfinite(vn_), vn_, np.interp(gn_, t_tvt, tws)))
            Anbr = np.interp(pos, gn_, np.isfinite(vn_).astype(float))
        else:
            Tnbr = Ttw; Anbr = np.zeros_like(Ttw)
        # チャネル計算(行→bin間引き)
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
        EV = np.stack([c[sel] for c in (c0, c1, c2, c3, c4, c5)], 0).astype(np.float16)
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
    print(f"saved {os.path.basename(OUT)}: {len(OUTD)} wells ({sz:.0f}MB, {time.time()-t0:.0f}s)")

if __name__ == "__main__":
    main()
