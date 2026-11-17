# -*- coding: utf-8 -*-
# v64b — struct-free GR照合像ビルダ(lk中心・±60ft・8ch)
# =========================================================================
# v64(st中心9ch)からの変更(ユーザー方針 2026-07-17「structなしで」):
#   ① δ窓の中心 = last_known_tvt(lk)固定 — train/test で品質差ゼロの唯一の錨。struct 依存を完全排除。
#   ② 窓 ±60ft(241bin, 0.5ft)— lk からの drift 尾をカバー(実データ |truth−lk| p95≈23ft)。
#   ③ チャネル 8ch: c0-c5 = v64 と同じ純GR照合6ch / c6 = dead-reckoning ストライプ
#      (自分の prefix dip の外挿線、σは mds で拡大 — 井戸内情報のみ=合法) / c7 = mds ランプ。
#      struct σストライプ・lk ストライプは廃止(lk は中心そのもの)。
#   ④ pseudo: st 捏造(sidecar)不要 — lk は生成真値の最終既知値で実井戸と同義。事前学習の座標系ギャップ=0。
# -------------------------------------------------------------------------
# env: NWELLS / OUT(既定 imagesB_v64.pkl) / SRC / STMAP(pseudo: donorのnbrプール用) / NBIN(120)
# 出力: {well: {EV:(8,nb,ND)fp16, dtrue, wgt, st(=lk定数列), bid, y_ev}}(v60系トレーナ互換, NBIN env)
import os, sys, json, time, pickle
import numpy as np
import pandas as pd

P0 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART = os.path.join(P0, "v64", "artifacts_v64"); os.makedirs(ART, exist_ok=True)
SRC = os.environ.get("SRC", os.path.join(P0, "rogii-wellbore-geology-prediction", "train"))
OUT = os.path.join(ART, os.environ.get("OUT", "imagesB_v64.pkl"))
STMAP = os.environ.get("STMAP", "")
NB = int(os.environ.get("NBIN", "120")); ND = 2 * NB + 1; STEP = 0.5; STR = 16
OFF = (np.arange(ND) - NB) * STEP
W16, W64 = 16, 64

def smooth(v, fb, r=2):
    s = pd.Series(v, dtype="float64").interpolate(limit_direction="both").fillna(fb)
    return s.rolling(2 * r + 1, center=True, min_periods=1).mean().to_numpy()

def grid_from_pairs(tvt, gr, gmin, gmax):
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
    V = np.isfinite(T); Tz = np.where(V, T, 0.0); gs = g[:, None] * V
    n, _ = winsum(V.astype(np.float64), W)
    sg, _ = winsum(gs, W); st_, _ = winsum(Tz, W)
    sgg, _ = winsum((g[:, None] ** 2) * V, W); stt, _ = winsum(Tz ** 2, W)
    sgt, _ = winsum(g[:, None] * Tz, W)
    n = np.maximum(n, 1)
    mg = sg / n; mt = st_ / n
    vg = np.maximum(sgg / n - mg ** 2, 1e-6); vt = np.maximum(stt / n - mt ** 2, 1e-6)
    return np.clip((sgt / n - mg * mt) / np.sqrt(vg * vt), -1, 1), (n / W)

def main():
    t0 = time.time()
    stmap = {}
    if STMAP:
        man = json.loads(open(STMAP, encoding="utf-8").read())
        stmap = {r["id"]: r["donor"] for r in man["wells"]}
    NBR_SIDE = {}
    _nbp = os.path.join(SRC, "pseudo_nbr.npz")
    if os.path.exists(_nbp):
        _z = np.load(_nbp)
        NBR_SIDE = {k: _z[k] for k in _z.files}
        print(f"[nbr] sidecar pseudo_nbr.npz: {sum(1 for k in NBR_SIDE if not k.endswith('__gr'))} wells")
    files = sorted(f.split("__")[0] for f in os.listdir(SRC) if f.endswith("__horizontal_well.csv"))
    if os.environ.get("NWELLS"): files = files[:int(os.environ["NWELLS"])]
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
        fb = float(np.nanmean(t_gr))
        gr_all = smooth(hw["GR"].to_numpy(float), fb, 2)
        g = gr_all[ev]
        y = hw["TVT"].to_numpy(float)[ev]
        T = len(g); yv = y
        kn_tvt = hw["TVT_input"].to_numpy(float)[kn]
        lk = float(kn_tvt[-1])
        stv = np.full(T, lk, np.float64)                     # ★中心 = lk 固定(struct不使用)
        # dead-reckoning: prefix末尾30行の dip(dTVT/dMD)を外挿
        md_all = hw["MD"].to_numpy(float)
        md_kn = md_all[kn]; md_lk = float(md_kn[-1])
        tl_t = kn_tvt[-30:]; tl_m = md_kn[-30:]
        dm = np.diff(tl_m); dt = np.diff(tl_t); mok = dm > 0
        ir = float(np.median(dt[mok] / dm[mok])) if mok.sum() >= 3 else 0.0
        ir = float(np.clip(ir, -0.05, 0.05))
        mds_row = np.maximum(md_all[ev] - md_lk, 0.0)
        # σ較正(prefix)
        resid = gr_all[kn] - np.interp(kn_tvt, t_tvt, smooth(t_gr, fb, 2))
        sigw = float(np.nanstd(resid)); sigw = 15.0 if not np.isfinite(sigw) or sigw < 3 else max(sigw, 5.0)
        pos = stv[:, None] + OFF[None, :]
        tws = smooth(t_gr, fb, 2)
        Ttw = np.interp(pos, t_tvt, tws)
        gmin, gmax = pos.min() - 1, pos.max() + 1
        gs_, vs_ = grid_from_pairs(kn_tvt, gr_all[kn], gmin, gmax)
        Tself = np.interp(pos, gs_, np.where(np.isfinite(vs_), vs_, np.interp(gs_, t_tvt, tws)))
        Aself = np.interp(pos, gs_, np.isfinite(vs_).astype(float))
        if wn in NBR_SIDE:
            gn_, vn_ = grid_from_pairs(np.asarray(NBR_SIDE[wn], float), np.asarray(NBR_SIDE[wn + "__gr"], float), gmin, gmax)
            Tnbr = np.interp(pos, gn_, np.where(np.isfinite(vn_), vn_, np.interp(gn_, t_tvt, tws)))
            Anbr = np.interp(pos, gn_, np.isfinite(vn_).astype(float))
        else:
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
        d16, _ = winsum((g[:, None] - Ttw), W16)
        c0 = (d16 / W16) / sigw
        c1, _ = ncc_img(g, Ttw, W16)
        c2, _ = ncc_img(g, Ttw, W64)
        c3, _ = ncc_img(g, Tself, W16)
        c4, _ = ncc_img(g, Tnbr, W16)
        a5s, _ = winsum(Aself, W16); a5n, _ = winsum(Anbr, W16)
        c5 = 0.5 * (a5s / W16) + 0.5 * (a5n / W16)
        nb_ = (T + STR - 1) // STR
        sel = np.clip(np.arange(nb_) * STR + STR // 2, 0, T - 1)
        # c6: dead-reckoning ストライプ(δ = ir×mds, σ = 3 + 0.004×mds を 3..25 にクリップ)
        dr_b = np.clip(ir * mds_row[sel], -NB * STEP, NB * STEP)
        sg_b = np.clip(3.0 + 0.004 * mds_row[sel], 3.0, 25.0)
        c6_b = np.exp(-0.5 * ((OFF[None, :] - dr_b[:, None]) / sg_b[:, None]) ** 2)
        # c7: mds ランプ
        c7_b = np.repeat((np.clip(mds_row[sel] / 1000.0, 0, 3.0) / 3.0)[:, None], ND, 1)
        EV = np.concatenate([np.stack([c[sel] for c in (c0, c1, c2, c3, c4, c5)], 0),
                             np.stack([c6_b, c7_b], 0)], 0).astype(np.float16)
        bid = np.minimum(np.arange(T) // STR, nb_ - 1)
        dtrue = np.full(nb_, np.nan); wgt = np.zeros(nb_)
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
    print(f"saved {os.path.basename(OUT)}: {len(OUTD)} wells 8ch ND={ND} ({sz:.0f}MB, {time.time()-t0:.0f}s)")

if __name__ == "__main__":
    main()
