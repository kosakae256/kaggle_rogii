# -*- coding: utf-8 -*-
"""
v89 — struct-free 2D UNet 用 GR照合像ビルダー。
=====================================================================================
v64 との違い(全て struct を外す):
  - δ枠の中心 st = **tw-PF軌跡(v88/base, struct錨なし)**(v64 は struct軌跡だった)。
    → 真のδ(=PFの枝誤差)は ±1-2サイクル(~±32ft)内=±45ft窓に収まる。U-Netは非逐次文脈で枝を訂正。
  - 9ch は全て struct-free:
     c0 窓平均GR差/σ(W16)  c1 NCC vs typewell(W16)  c2 NCC vs typewell(W64)
     c3 NCC vs self疑似tw(W16)  c4 NCC vs nbr疑似tw(W16)  c5 self/nbr被覆率
     c6 PF信頼ストライプ exp(−½(δ/σ_pf)²)   … PF中心=δ0 が尤もらしいという弱prior(struct不使用)
     c7 lk錨ストライプ exp(−½((δ−(lk−st))/3)²)   … last_known_tvt(struct不使用)
     c8 mdsランプ                                  … 錨からの距離(struct不使用)
  - 学習target dtrue = median(y − st) = 真−PF = PFが外した枝オフセット。
出力: v89/artifacts_v89/images_v89.pkl {well:{EV(9,nb,181)fp16, dtrue, wgt, st, bid, y_ev}}(v64トレーナ互換)
使い方: NWELLS=8 python v89/build_images_v89.py  /  本番 NW=0
"""
import os, sys, time, pickle
import numpy as np, pandas as pd
P0 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(P0, "v64"))
import build_images_v64 as bi   # smooth/grid_from_pairs/winsum/ncc_img/OFF/NB/ND/STEP/STR/W16/W64/SIG_LK 流用

ART = os.path.join(P0, "v89", "artifacts_v89"); os.makedirs(ART, exist_ok=True)
SRC = os.environ.get("SRC", os.path.join(P0, "rogii-wellbore-geology-prediction", "train"))
OUT = os.path.join(ART, os.environ.get("OUT", "images_v89.pkl"))
OFF, ND, STR, W16, W64, SIG_LK = bi.OFF, bi.ND, bi.STR, bi.W16, bi.W64, bi.SIG_LK
smooth, grid_from_pairs, winsum, ncc_img = bi.smooth, bi.grid_from_pairs, bi.winsum, bi.ncc_img
SIG_PF = float(os.environ.get("SIG_PF", "8.0"))
SIG_GUIDE = float(os.environ.get("SIG_GUIDE", "4.0"))
CENTER = os.environ.get("CENTER", "pf")   # pf=tw-PF中心(逐次) / line=prefix傾き外挿(非逐次・PF非依存のablation)
_BANKS = None
if "banks" in os.environ.get("REP", ""):
    _bp = os.path.join(ART, "pf_banks_v89.npz")
    _BANKS = dict(np.load(_bp)) if os.path.exists(_bp) else {}
    print(f"[v89] PF banks guide: {len(_BANKS)}エントリ")


def _dip_line(hw, kn, ev, n):
    """prefix既知帯の傾き(dTVT_input/dMD)を eval帯に外挿した純幾何dip線(PF不使用・非逐次)。"""
    md = hw["MD"].to_numpy(float)
    ti = hw["TVT_input"].to_numpy(float)
    mk = md[kn]; tk = ti[kn]
    lk = float(tk[-1]); mlk = float(mk[-1])
    # ★平坦線(lk水平コピー)= 純幾何・PFフリー・非逐次。RMSE≈15.9(直線ベースライン)、真δ±45窓内98%。
    #   prefix急dipの外挿はランナウェイ(70ft)するので使わない。U-Netがここからどれだけ登れるかを見る。
    return np.full(n, lk, np.float64)


def main():
    t0 = time.time()
    PFC = dict(np.load(os.path.join(ART, "pf_center_v89.npz")))   # well -> ev行 PF TVT
    print(f"[v89] PF center: {len(PFC)}井", flush=True)
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
        if wn not in PFC: continue
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
        if CENTER == "line":
            stv = _dip_line(hw, kn, ev, len(g))                    # ★PFフリー: prefix傾き外挿(非逐次・純幾何)
        else:
            stv = np.asarray(PFC[wn], np.float64)                  # struct-free center = PF
        T = min(len(g), len(stv)); g = g[:T]; yv = y[:T]; stv = stv[:T]
        if not np.isfinite(stv).all():
            stv = pd.Series(stv).interpolate(limit_direction="both").fillna(float(np.nanmedian(yv))).to_numpy()
        kn_tvt = hw["TVT_input"].to_numpy(float)[kn]
        resid = gr_all[kn] - np.interp(kn_tvt, t_tvt, smooth(t_gr, fb, 2))
        sigw = float(np.nanstd(resid)); sigw = 15.0 if not np.isfinite(sigw) or sigw < 3 else max(sigw, 5.0)
        pos = stv[:, None] + OFF[None, :]
        tws = smooth(t_gr, fb, 2)
        Ttw = np.interp(pos, t_tvt, tws)
        gmin, gmax = pos.min() - 1, pos.max() + 1
        gs_, vs_ = grid_from_pairs(kn_tvt, gr_all[kn], gmin, gmax)
        Tself = np.interp(pos, gs_, np.where(np.isfinite(vs_), vs_, np.interp(gs_, t_tvt, tws)))
        Aself = np.interp(pos, gs_, np.isfinite(vs_).astype(float))
        _, nidx = tree.query(cache[wn]["xy"], k=9) if wn in cache else (None, [])
        js = [realw[j] for j in np.atleast_1d(nidx) if realw[j] != wn][:8] if wn in cache else []
        if js:
            ny = np.concatenate([KN[j][0] for j in js]); ng = np.concatenate([KN[j][1] for j in js])
            gn_, vn_ = grid_from_pairs(ny, ng, gmin, gmax)
            Tnbr = np.interp(pos, gn_, np.where(np.isfinite(vn_), vn_, np.interp(gn_, t_tvt, tws)))
            Anbr = np.interp(pos, gn_, np.isfinite(vn_).astype(float))
        else:
            Tnbr = Ttw; Anbr = np.zeros_like(Ttw)
        d16, _ = winsum((g[:, None] - Ttw), W16); c0 = (d16 / W16) / sigw
        c1, _ = ncc_img(g, Ttw, W16); c2, _ = ncc_img(g, Ttw, W64)
        c3, _ = ncc_img(g, Tself, W16); c4, _ = ncc_img(g, Tnbr, W16)
        a5s, _ = winsum(Aself, W16); a5n, _ = winsum(Anbr, W16)
        c5 = 0.5 * (a5s / W16) + 0.5 * (a5n / W16)
        # ★GR表現/窓長の追加チャネル(REP env)。自己相似(サイクル両義)を長窓NCC・勾配で破る狙い。
        EXTRA = []
        REP = os.environ.get("REP", "")
        if "long" in REP:      # 長窓NCC(vs typewell): 波形シーケンスを長く取り真の整列だけ残す
            for WL in (128, 256):
                ci, _ = ncc_img(g, Ttw, WL); EXTRA.append(ci)
        if "grad" in REP:      # GR勾配(形): オフセットdrift不変、遷移を強調
            gg2 = np.gradient(g); Tg = np.gradient(Ttw, axis=0)
            cg, _ = ncc_img(gg2, Tg, W16); EXTRA.append(cg)
        if "selflong" in REP:  # self/nbr の長窓NCC
            cs, _ = ncc_img(g, Tself, W64); cn, _ = ncc_img(g, Tnbr, W64)
            EXTRA += [cs, cn]
        if "logamp" in REP:    # ★v85脱相関表現: 対数振幅圧縮したGRでNCC(生と別の枝構造を見る)
            g0 = 40.0
            gL = np.log1p(np.maximum(g, 0) / g0) * g0
            TtwL = np.log1p(np.maximum(Ttw, 0) / g0) * g0
            cl16, _ = ncc_img(gL, TtwL, W16); cl64, _ = ncc_img(gL, TtwL, W64)
            EXTRA += [cl16, cl64]
        if "banks" in REP and _BANKS:   # ★複数PF(p_1..p_5)の投票をδ方向のガウス縞で(全解像度)
            for lab in ("p_1", "p_2", "p_3", "p_4", "p_5"):
                key = f"{wn}__{lab}"
                if key not in _BANKS: continue
                bpf = np.asarray(_BANKS[key], float)[:T]
                if len(bpf) < T: bpf = np.pad(bpf, (0, T - len(bpf)), mode="edge")
                doff = bpf - stv                                   # 各行のδ位置(bank − center)
                stripe = np.exp(-0.5 * ((OFF[None, :] - doff[:, None]) / SIG_GUIDE) ** 2)  # (T,181)
                EXTRA.append(stripe.astype(np.float32))
        nb_ = (T + STR - 1) // STR
        sel = np.clip(np.arange(nb_) * STR + STR // 2, 0, T - 1)
        # struct-free prior チャネル
        c6_b = np.repeat(np.exp(-0.5 * (OFF / SIG_PF) ** 2)[None, :], nb_, 0)       # PF信頼(δ0中心)
        lk = float(kn_tvt[-1]); zl_b = lk - stv[sel]
        c7_b = np.exp(-0.5 * ((OFF[None, :] - zl_b[:, None]) / SIG_LK) ** 2)
        md_all = hw["MD"].to_numpy(float)
        md_lk = float(md_all[kn][-1]) if kn.sum() else float(md_all[0])
        mds_row = np.maximum(md_all[ev][:T] - md_lk, 0.0)
        c8_b = np.repeat((np.clip(mds_row[sel] / 1000.0, 0, 3.0) / 3.0)[:, None], ND, 1)
        base_ch = np.stack([c[sel] for c in (c0, c1, c2, c3, c4, c5)], 0)
        prior_ch = np.stack([c6_b, c7_b, c8_b], 0)
        parts = [base_ch] + ([np.stack([c[sel] for c in EXTRA], 0)] if EXTRA else []) + [prior_ch]
        EV = np.concatenate(parts, 0).astype(np.float16)
        bid = np.minimum(np.arange(T) // STR, nb_ - 1)
        dtrue = np.full(nb_, np.nan); wgt = np.zeros(nb_); dely = yv - stv
        for b in range(nb_):
            m = bid == b
            if m.sum():
                dtrue[b] = float(np.nanmedian(dely[m])); wgt[b] = float(np.isfinite(dely[m]).sum())
        OUTD[wn] = dict(EV=EV, dtrue=dtrue.astype(np.float32), wgt=wgt.astype(np.float32),
                        st=stv.astype(np.float32), bid=bid.astype(np.int32), y_ev=yv.astype(np.float32))
        if (wi + 1) % 100 == 0: print(f"  {wi+1}/{len(files)} ({time.time()-t0:.0f}s)", flush=True)
    pickle.dump(OUTD, open(OUT, "wb"), protocol=4)
    # 診断: 真δが窓内に入る割合(=学習可能行)
    din = []
    for w, d in OUTD.items():
        dt = d["dtrue"][np.isfinite(d["dtrue"])]
        din.append(np.mean(np.abs(dt) <= 45.0))
    print(f"saved {os.path.basename(OUT)}: {len(OUTD)}井 9ch struct-free "
          f"({os.path.getsize(OUT)/(1<<20):.0f}MB, {time.time()-t0:.0f}s)")
    print(f"  真δ(=真−PF)が±45ft窓内の割合: 中央 {np.median(din)*100:.1f}% (学習可能行)")


if __name__ == "__main__":
    main()
