# -*- coding: utf-8 -*-
# v64 pseudo 生成器 v2 — レシピ式・大量産・layered warp(「作り方で結果が変わる」を試すための基盤)
# =========================================================================
# ユーザー指摘(2026-07-16): v63 生成器は不自然(滑らかな低周波warp+donor軌跡固定)。
# 改善: ①層単位(ブロック状)の歪み=layered warp ②合成軌跡(新規井戸形状) ③typewell劣化
#       ④断層ジャンプ ⑤GR欠測合成 ⑥疑似struct(st/sig)を生成器が出力(sidecar npz)
# -------------------------------------------------------------------------
# レシピ(env RECIPE="synthcol:0.45,trajmix:0.25,surface:0.15,reskin:0.15"):
#   reskin  : donor軌跡・真値継承、GR=layered warp した donor typewell から再合成(v55系の自然化版)
#   surface : donor軌跡+3D面(ANCC面+共有残差場)から真値再構成(v63系)、GR=layered warp
#   trajmix : 地質=donor typewell(真の柱状図扱い)+ **合成軌跡**。渡す typewell は劣化版
#   synthcol: 地質=typewell 1〜2本の接ぎ合わせ+layered warp(完全合成柱状図)+ 合成軌跡。donor無し
# 共通ノブ: LTH(平均層厚ft) THICK_L(層logサイズσ) AMP_L OFF_L / TW_STRETCH TW_DROP TW_NOISE(typewell劣化)
#           FAULT_RATE FAULT_FT / GAP_RATE / NOISE_LVLS,NOISE_MIX(v63と同じ3水準) / ST_ERR ST_DATUM(疑似struct品質)
# 出力: v64/artifacts_v64/pseudo_wells_{TAG}/{id}__horizontal_well.csv / __typewell.csv
#       + manifest.json + pseudo_st.npz({id}=st, {id}__sig=sig; eval行整列)
# 実行: .venv/Scripts/python.exe v64/make_pseudo_v64.py   (env: TAG=q1 NSYN=2000 ...)
import os, sys, json, time
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, binary_dilation

sys.stdout = open(1, "w", encoding="utf-8", closefd=False)
P0 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN = os.path.join(P0, "rogii-wellbore-geology-prediction", "train")
TAG = os.environ.get("TAG", "q1")
OUT = os.path.join(P0, "v64", "artifacts_v64", f"pseudo_wells_{TAG}")
os.makedirs(OUT, exist_ok=True)

NSYN = int(os.environ.get("NSYN", "2000"))
RECIPE = os.environ.get("RECIPE", "synthcol:0.45,trajmix:0.25,surface:0.15,reskin:0.15")
SEED = int(os.environ.get("SEED", "0"))
# --- layered warp(層単位の歪み) ---
LTH = float(os.environ.get("LTH", "25"))         # 平均層厚 ft(Poisson境界)
THICK_L = float(os.environ.get("THICK_L", "0.12"))  # 層log厚みσ
AMP_L = float(os.environ.get("AMP_L", "0.12"))      # 層ゲインσ
OFF_L = float(os.environ.get("OFF_L", "5.0"))       # 層オフセットσ(API)
# --- typewell 劣化(参照は地質と少し違う) ---
TW_STRETCH = float(os.environ.get("TW_STRETCH", "0.05"))
TW_DROP = float(os.environ.get("TW_DROP", "0.05"))  # 層dropout率
TW_NOISE = float(os.environ.get("TW_NOISE", "2.0"))
# --- 断層/欠測/ノイズ ---
FAULT_RATE = float(os.environ.get("FAULT_RATE", "0.25"))   # 本/井戸(Poisson)
FAULT_FT = float(os.environ.get("FAULT_FT", "10"))         # ジャンプ幅上限(±U(3,FAULT_FT))
GAP_RATE = float(os.environ.get("GAP_RATE", "0.015"))      # GR欠測ブロック開始率/行(合成軌跡用)
NOISE_LVLS = [float(x) for x in os.environ.get("NOISE_LVLS", "1.5,6.0,11.0").split(",")]
NOISE_MIX = [float(x) for x in os.environ.get("NOISE_MIX", "0.25,0.5,0.25").split(",")]
RES = int(os.environ.get("RES", "2"))
# --- 疑似struct(st)品質 ---
ST_ERR = float(os.environ.get("ST_ERR", "6.0"))     # st誤差場σ(ft)※ST_MIX指定時は不使用
ST_CLEN = float(os.environ.get("ST_CLEN", "900"))   # st誤差場相関長(ft)
ST_DATUM = float(os.environ.get("ST_DATUM", "10"))  # 井戸レベルdatumオフセット±(ft)
# q2: 井戸ごとにσを混合抽選(易/中/hard井戸の再現)。空文字なら旧動作(ST_ERR固定)
ST_MIX = os.environ.get("ST_MIX", "")               # 例 "4:0.5,8:0.35,16:0.15"
# q2: 「typewellのwarpでは説明できない」GR区間(facies混入)の割合
MISMATCH_P = float(os.environ.get("MISMATCH_P", "0.0"))
# q2: synthcol に合成近傍プールを生成(c4縮退の解消)
SYN_NBR = os.environ.get("SYN_NBR", "0") == "1"
# q3: GRテクスチャ現実化 — 白色→AR(1)相関ノイズ(rho)、スパイク床(rate/行)
AR_RHO = float(os.environ.get("AR_RHO", "0"))
SPIKE_RATE = float(os.environ.get("SPIKE_RATE", "0"))
# q4: MISMATCH の混入源 — tw=別typewell補間(q2既定) / real=実坑井GR断片スプライス(本物のテクスチャ)
MISMATCH_MODE = os.environ.get("MISMATCH_MODE", "tw")
MIX_W_LO = float(os.environ.get("MIX_W_LO", "0.35")); MIX_W_HI = float(os.environ.get("MIX_W_HI", "0.6"))
# q7: 合成軌跡のdrift較正(実データ: 井戸p95中央値18.7ft に合わせる)
UND_LO = float(os.environ.get("UND_LO", "2.0")); UND_HI = float(os.environ.get("UND_HI", "9.0"))
SLOPE_MAX = float(os.environ.get("SLOPE_MAX", "0.0015"))
# --- surfaceレシピ(v63と同じ面モデル) ---
CELL = float(os.environ.get("CELL_FT", "250"))
RES_FT = float(os.environ.get("RES_FT", "5.0")); RES_CLEN = float(os.environ.get("RES_CLEN", "900"))
JIT_FT = float(os.environ.get("JIT_FT", "1500"))
FORMS = ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]


def gfield(rng, n, step_ft, clen, sigma):
    """長さnの相関長clen(ft)ガウス場(σ=sigma)。step_ft=1行あたりft。"""
    if sigma <= 0: return np.zeros(n)
    r = max(int(round(clen / max(step_ft, 1e-6))), 1)
    w = rng.standard_normal(n + 2 * r)
    k = np.exp(-0.5 * (np.arange(-3 * r, 3 * r + 1) / r) ** 2)
    f = np.convolve(w, k / np.sqrt((k ** 2).sum()), mode="same")[r:r + n]
    return sigma * f


def layer_bounds(rng, s_lo, s_hi, mean_th):
    """[s_lo,s_hi] を平均層厚 mean_th で分割する境界列。"""
    bs = [s_lo]
    while bs[-1] < s_hi:
        bs.append(bs[-1] + max(rng.exponential(mean_th), 3.0))
    bs[-1] = s_hi
    return np.array(bs)


def layered_warp(rng, s_grid, thick_l=THICK_L, amp_l=AMP_L, off_l=OFF_L, mean_th=LTH):
    """層単位のブロック状歪み: 単調warp g(s)(層厚伸縮の積分)+層別ゲインa(s)/オフセットb(s)。"""
    bs = layer_bounds(rng, s_grid[0], s_grid[-1], mean_th)
    nl = len(bs) - 1
    lth = np.exp(rng.normal(0, thick_l, nl))            # 層ごとの厚み倍率
    la = 1.0 + rng.normal(0, amp_l, nl)
    lb = rng.normal(0, off_l, nl)
    li = np.clip(np.searchsorted(bs, s_grid, side="right") - 1, 0, nl - 1)
    st_ = np.median(s_grid)
    dg = lth[li]                                         # 局所伸縮率
    g = np.cumsum(dg) * np.median(np.diff(s_grid))
    g = g - np.interp(st_, s_grid, g) + st_              # 井戸帯でゼロ中心の単調warp
    return np.maximum.accumulate(g), la[li], lb[li]


def degrade_typewell(rng, t_tvt, t_gr):
    """消費者に渡す typewell = 地質列の劣化版(層stretch+層dropout+ノイズ)。"""
    g, la, lb = layered_warp(rng, t_tvt, TW_STRETCH, 0.05, 2.0, LTH * 1.5)
    gr = la * np.interp(g, t_tvt, t_gr) + lb + rng.standard_normal(len(t_tvt)) * TW_NOISE
    bs = layer_bounds(rng, t_tvt[0], t_tvt[-1], LTH * 1.5)
    keep = np.ones(len(t_tvt), bool)
    for j in range(len(bs) - 1):
        if rng.random() < TW_DROP:
            keep &= ~((t_tvt >= bs[j]) & (t_tvt < bs[j + 1]))
    if keep.sum() < 60: keep[:] = True
    return t_tvt[keep], gr[keep]


def synth_trajectory(rng, col_lo, col_hi):
    """合成軌跡: landing(prefix, TVT_input既知)+lateral(うねり+断層)。1ft刻みMD。"""
    Lp = int(rng.uniform(300, 800))                     # prefix行数
    Ll = int(rng.uniform(3500, 11000))                  # lateral行数
    n = Lp + Ll
    s0 = rng.uniform(col_lo + 25, col_hi - 25)          # lateral目標深度(列座標)
    # landing: 列上部から s0 へ滑らかに降下
    top = max(col_lo + 5, s0 - rng.uniform(80, 250))
    tpre = np.linspace(0, 1, Lp)
    s_pre = top + (s0 - top) * (0.5 - 0.5 * np.cos(np.pi * tpre))
    # lateral: うねり+緩傾斜+断層
    und = gfield(rng, Ll, 1.0, rng.uniform(700, 2500), rng.uniform(UND_LO, UND_HI))
    slope = rng.uniform(-SLOPE_MAX, SLOPE_MAX) * np.arange(Ll)
    s_lat = s0 + und - und[0] + slope
    nf = rng.poisson(FAULT_RATE)
    for _ in range(nf):
        j = int(rng.uniform(0.15, 0.95) * Ll)
        s_lat[j:] += rng.uniform(3, FAULT_FT) * (1 if rng.random() < 0.5 else -1)
    s = np.concatenate([s_pre, s_lat])
    s = np.clip(s, col_lo + 3, col_hi - 3)
    md = 11000.0 + np.arange(n) * 1.0
    kn = np.zeros(n, bool); kn[:Lp] = True
    return md, s, kn


def render_gr(rng, s, t_tvt, t_gr, sigma_noise, gaps=True, kn=None):
    """地質列(t_tvt,t_gr)を layered warp した「真の地質」から GR(md) を描画。"""
    g, la, lb = layered_warp(rng, t_tvt)
    Ggrid = la * np.interp(g, t_tvt, t_gr) + lb
    n = len(s)
    if AR_RHO > 0:   # q3: AR(1)相関ノイズ(分散をsigma_noise²に正規化)
        from scipy.signal import lfilter
        e = lfilter([1.0], [1.0, -AR_RHO], rng.standard_normal(n)) * np.sqrt(1 - AR_RHO ** 2)
    else:
        e = rng.standard_normal(n)
    gr = np.interp(s, t_tvt, Ggrid) + e * sigma_noise
    if SPIKE_RATE > 0:   # q3: スパイク床(薄い高GR/低GR層の模擬)
        for j in np.where(rng.random(n) < SPIKE_RATE)[0]:
            L = int(rng.uniform(1, 5))
            gr[j:j + L] += rng.uniform(15, 60) * (1 if rng.random() < 0.7 else -1)
    if RES > 0:
        gr = pd.Series(gr).rolling(2 * RES + 1, center=True, min_periods=1).mean().to_numpy()
    gr = np.array(gr, copy=True)   # rolling().to_numpy() は読み取り専用のことがある(CoW)
    if gaps:
        i = 0
        while i < len(gr):
            if rng.random() < GAP_RATE:
                L = int(rng.uniform(5, 40)); gr[i:i + L] = np.nan; i += L
            i += 1
        if kn is not None: gr[np.where(kn)[0][:8]] = np.nan if False else gr[np.where(kn)[0][:8]]
    return gr


_ST_MIX = None
if ST_MIX:
    _ST_MIX = ([float(t.split(":")[0]) for t in ST_MIX.split(",")],
               np.array([float(t.split(":")[1]) for t in ST_MIX.split(",")]))
    _ST_MIX = (_ST_MIX[0], _ST_MIX[1] / _ST_MIX[1].sum())

def make_st(rng, truth, kn):
    """疑似struct: 真値+datumオフセット+相関誤差場(eval行のみ返す)。sig=誤差水準の較正値。"""
    n = len(truth)
    if _ST_MIX is not None:   # q2: 井戸ごとにσ抽選(hard井戸=大誤差の再現)
        s_err = float(_ST_MIX[0][int(rng.choice(len(_ST_MIX[0]), p=_ST_MIX[1]))]) * rng.uniform(0.8, 1.2)
    else:
        s_err = ST_ERR
    err = gfield(rng, n, 1.0, ST_CLEN, s_err) + rng.uniform(-ST_DATUM, ST_DATUM)
    st_full = truth + err
    ev = ~kn
    st = st_full[ev]
    sig = np.clip(np.abs(pd.Series(err[ev]).rolling(301, center=True, min_periods=1).std().to_numpy())
                  + 0.6 * s_err, 2.0, 30.0)
    return st.astype(np.float32), sig.astype(np.float32)


# ---------- v63 と同じ 3D 面モデル(surface レシピ用) ----------
def build_surfaces(wells):
    pts = []
    for w in wells:
        try:
            h = pd.read_csv(os.path.join(TRAIN, f"{w}__horizontal_well.csv"), usecols=["X", "Y"] + FORMS)
        except Exception:
            continue
        pts.append(h.iloc[::8].to_numpy(float))
    P = np.concatenate(pts, 0)
    x0, x1 = np.nanmin(P[:, 0]), np.nanmax(P[:, 0]); y0, y1 = np.nanmin(P[:, 1]), np.nanmax(P[:, 1])
    nx = int((x1 - x0) / CELL) + 3; ny = int((y1 - y0) / CELL) + 3
    ix = np.clip(((P[:, 0] - x0) / CELL).astype(int) + 1, 0, nx - 1)
    iy = np.clip(((P[:, 1] - y0) / CELL).astype(int) + 1, 0, ny - 1)
    cnt = np.zeros((ny, nx)); np.add.at(cnt, (iy, ix), 1.0)
    v = P[:, 2]; m = np.isfinite(v)   # ANCCのみで十分(真値はANCC恒等式)
    ssum = np.zeros((ny, nx)); scnt = np.zeros((ny, nx))
    np.add.at(ssum, (iy[m], ix[m]), v[m]); np.add.at(scnt, (iy[m], ix[m]), 1.0)
    g = np.where(scnt > 0, ssum / np.maximum(scnt, 1), np.nan)
    filled = g.copy()
    for _ in range(400):
        nanm = ~np.isfinite(filled)
        if not nanm.any(): break
        vz = np.where(np.isfinite(filled), filled, 0.0); vm = np.isfinite(filled).astype(float)
        sm_v = gaussian_filter(vz, 1.2); sm_m = gaussian_filter(vm, 1.2)
        cand = np.where(sm_m > 1e-6, sm_v / np.maximum(sm_m, 1e-6), np.nan)
        filled[nanm] = cand[nanm]
    S = gaussian_filter(np.where(np.isfinite(filled), filled, 0.0), 1.0)
    hull = binary_dilation(cnt > 0, iterations=4)
    return S, hull, (x0, y0, CELL, nx, ny)


def sampler(geo):
    x0, y0, cell, nx, ny = geo
    def samp(grid, x, y):
        fx = (np.asarray(x, float) - x0) / cell + 1.0
        fy = (np.asarray(y, float) - y0) / cell + 1.0
        i0 = np.floor(fx).astype(int); j0 = np.floor(fy).astype(int)
        ok = (i0 >= 0) & (i0 < nx - 1) & (j0 >= 0) & (j0 < ny - 1)
        i0c = np.clip(i0, 0, nx - 2); j0c = np.clip(j0, 0, ny - 2)
        tx = fx - i0c; ty = fy - j0c
        v = (grid[j0c, i0c] * (1 - tx) * (1 - ty) + grid[j0c, i0c + 1] * tx * (1 - ty)
             + grid[j0c + 1, i0c] * (1 - tx) * ty + grid[j0c + 1, i0c + 1] * tx * ty)
        return np.where(ok, v, np.nan)
    def inhull(h, x, y):
        i = np.clip(((np.asarray(x, float) - x0) / cell).astype(int) + 1, 0, nx - 1)
        j = np.clip(((np.asarray(y, float) - y0) / cell).astype(int) + 1, 0, ny - 1)
        return h[j, i]
    return samp, inhull


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    rmix = []
    for tok in RECIPE.split(","):
        k, v = tok.split(":"); rmix.append((k.strip(), float(v)))
    rp = np.array([v for _, v in rmix]); rp = rp / rp.sum()
    wells = sorted(set(f.split("__")[0] for f in os.listdir(TRAIN) if f.endswith("__horizontal_well.csv")))
    need_surface = any(k == "surface" for k, _ in rmix)
    if need_surface:
        S_A, hull, geo = build_surfaces(wells)
        samp, inhull = sampler(geo)
        Rf = gaussian_filter(rng.standard_normal(S_A.shape), max(RES_CLEN / CELL, 0.8))
        Rf = Rf / max(Rf.std(), 1e-9) * RES_FT
    _RGR = {"pool": None}
    def real_gr_segment(rng, L):
        """実坑井GRのランダム断片(q4 MISMATCH_MODE=real 用)。初回に200本キャッシュ。"""
        if _RGR["pool"] is None:
            arrs = []
            for w_ in [wells[int(rng.integers(len(wells)))] for _ in range(200)]:
                try:
                    g_ = pd.read_csv(os.path.join(TRAIN, f"{w_}__horizontal_well.csv"), usecols=["GR"])["GR"].to_numpy(float)
                    if np.isfinite(g_).sum() > 500: arrs.append(g_[np.isfinite(g_)])
                except Exception:
                    pass
            _RGR["pool"] = arrs
        arrs = _RGR["pool"]
        if not arrs: return None
        a = arrs[int(rng.integers(len(arrs)))]
        if len(a) <= L + 1: return None
        j = int(rng.integers(len(a) - L))
        return a[j:j + L].copy()

    TWC = {}
    def tw_of(w):
        if w not in TWC:
            t = pd.read_csv(os.path.join(TRAIN, f"{w}__typewell.csv")).sort_values("TVT")
            tt = t["TVT"].to_numpy(float); tg = t["GR"].to_numpy(float)
            fin = np.isfinite(tt) & np.isfinite(tg)
            TWC[w] = (tt[fin], tg[fin])
        return TWC[w]
    made = []; ST_SIDE = {}; NBR_SIDE = {}; tries = 0
    rej = {"precond": 0, "cover": 0, "gr": 0, "traj": 0}
    while len(made) < NSYN and tries < NSYN * 6:
        tries += 1
        recipe = rmix[int(rng.choice(len(rmix), p=rp))][0]
        wn = wells[int(rng.integers(len(wells)))]
        try:
            hw = pd.read_csv(os.path.join(TRAIN, f"{wn}__horizontal_well.csv"))
        except Exception:
            continue
        t_tvt, t_gr = tw_of(wn)
        if len(t_tvt) < 120: rej["precond"] += 1; continue
        cat = int(rng.choice(len(NOISE_LVLS), p=np.array(NOISE_MIX) / sum(NOISE_MIX)))
        sigma = NOISE_LVLS[cat]
        donor = wn
        if recipe in ("reskin", "surface"):
            if "TVT" not in hw.columns or hw["TVT_input"].notna().sum() < 10 or hw["TVT_input"].isna().sum() < 100:
                rej["precond"] += 1; continue
            truth = hw["TVT"].to_numpy(float)
            if not np.isfinite(truth).all(): rej["precond"] += 1; continue
            kn = hw["TVT_input"].notna().to_numpy()
            md = hw["MD"].to_numpy(float)
            if recipe == "surface":
                X = hw["X"].to_numpy(float); Y = hw["Y"].to_numpy(float); Z = hw["Z"].to_numpy(float)
                d = (rng.uniform(-JIT_FT, JIT_FT), rng.uniform(-JIT_FT, JIT_FT))
                if not inhull(hull, X + d[0], Y + d[1]).all(): d = (0.0, 0.0)
                anc = samp(S_A, X + d[0], Y + d[1]) + samp(Rf, X + d[0], Y + d[1])
                if not np.isfinite(anc).all(): rej["cover"] += 1; continue
                b_dat = float(np.nanmedian(truth - (hw["ANCC"].to_numpy(float) - Z)))
                truth = (anc - Z) + b_dat
                if np.mean((truth >= t_tvt.min() - 20) & (truth <= t_tvt.max() + 20)) < 0.95:
                    rej["cover"] += 1; continue
            gr = render_gr(rng, truth, t_tvt, t_gr, sigma, gaps=False)
            gr = np.where(np.isfinite(hw["GR"].to_numpy(float)), gr, np.nan)   # donorの欠測継承
            out = hw.copy()
            out["TVT"] = truth; out["GR"] = gr
            out["TVT_input"] = np.where(kn, truth, np.nan)
            tw_out_t, tw_out_g = t_tvt, t_gr                                    # 参照twはdonorのまま(warpが地質差を担う)
        else:
            # 合成軌跡系(trajmix / synthcol)
            if recipe == "synthcol":
                w2 = wells[int(rng.integers(len(wells)))]
                t2t, t2g = tw_of(w2)
                if len(t2t) < 120: rej["precond"] += 1; continue
                cut = rng.uniform(0.35, 0.65)
                k1 = int(len(t_tvt) * cut); k2 = int(len(t2t) * cut)
                gr2s = t2g[k2:] + (t_gr[k1 - 1] - t2g[k2]) * np.exp(-np.arange(len(t2g) - k2) / 15.0)
                col_t = np.concatenate([t_tvt[:k1], t_tvt[k1 - 1] + np.cumsum(np.full(len(t2t) - k2, np.median(np.diff(t2t))))])
                col_g = np.concatenate([t_gr[:k1], gr2s])
                donor = ""                                                       # 近傍プール無効(実地質でない)
            else:
                col_t, col_g = t_tvt, t_gr
            md, s, kn = synth_trajectory(rng, col_t.min(), col_t.max())
            if kn.sum() < 10 or (~kn).sum() < 100: rej["traj"] += 1; continue
            truth = s
            gr = render_gr(rng, truth, col_t, col_g, sigma, gaps=True)
            twd_t, twd_g = degrade_typewell(rng, col_t, col_g)                   # 参照は劣化版
            n = len(md)
            out = pd.DataFrame(dict(MD=md, X=np.full(n, 0.0), Y=np.full(n, 0.0), Z=-truth,
                                    TVT=truth, GR=gr, TVT_input=np.where(kn, truth, np.nan)))
            if recipe == "trajmix":
                out["X"] = float(hw["X"].mean()); out["Y"] = float(hw["Y"].mean())
            tw_out_t, tw_out_g = twd_t, twd_g
        if not np.isfinite(np.nan_to_num(gr, nan=0.0)).all(): rej["gr"] += 1; continue
        # q2: facies混入 = typewell の warp では説明できない GR 区間(別柱状図とのブレンド)
        # q4: MISMATCH_MODE=real なら実坑井GR断片(本物のテクスチャ)を局所統計に合わせてスプライス
        if MISMATCH_P > 0:
            trv = out["TVT"].to_numpy(float)
            grv = np.array(out["GR"].to_numpy(float), copy=True)   # CoW対策
            knv_ = out["TVT_input"].notna().to_numpy()
            lat = np.where(~knv_)[0]
            alt = None
            if MISMATCH_MODE != "real":
                w2m = wells[int(rng.integers(len(wells)))]
                a_t, a_g = tw_of(w2m)
                if len(a_t) >= 120:
                    alt = np.interp(np.clip(trv, a_t.min(), a_t.max()), a_t, a_g)
            if MISMATCH_MODE == "real" or alt is not None:
                tot = int(MISMATCH_P * len(lat)); done = 0
                while done < tot and len(lat) > 50:
                    L = int(rng.uniform(120, 600)); j0 = int(rng.uniform(lat[0], max(lat[-1] - L, lat[0] + 1)))
                    mixw = rng.uniform(MIX_W_LO, MIX_W_HI)
                    if MISMATCH_MODE == "real":
                        seg = real_gr_segment(rng, L)
                        if seg is None: break
                        loc = grv[j0:j0 + L]
                        m_ = np.isfinite(loc)
                        if m_.sum() > 20 and np.isfinite(seg[:m_.size]).sum() > 20:
                            mu_l, sd_l = float(np.nanmean(loc)), float(np.nanstd(loc)) + 1e-6
                            mu_s, sd_s = float(np.nanmean(seg)), float(np.nanstd(seg)) + 1e-6
                            segz = (seg[:len(loc)] - mu_s) / sd_s * sd_l + mu_l   # 局所統計へ整合
                            grv[j0:j0 + L] = (1 - mixw) * loc + mixw * np.nan_to_num(segz, nan=mu_l)
                    else:
                        grv[j0:j0 + L] = (1 - mixw) * grv[j0:j0 + L] + mixw * alt[j0:j0 + L]
                    done += L
                out["GR"] = grv
        sid = f"qs{len(made):05d}_{recipe[:4]}"
        knv = out["TVT_input"].notna().to_numpy()
        st, sg = make_st(rng, out["TVT"].to_numpy(float), knv)
        ST_SIDE[sid] = st; ST_SIDE[sid + "__sig"] = sg
        # q2: synthcol の合成近傍プール(c4縮退解消): 同じ柱状図から別warpの(TVT,GR)点群を3本分
        if SYN_NBR and recipe == "synthcol":
            nb_t = []; nb_g = []
            for _ in range(3):
                lo = rng.uniform(col_t.min(), max(col_t.min() + 1, col_t.max() - 150))
                hi = min(lo + rng.uniform(120, 300), col_t.max())
                tv_ = np.sort(rng.uniform(lo, hi, int(rng.uniform(300, 700))))
                g_, la_, lb_ = layered_warp(rng, col_t)
                Gg_ = la_ * np.interp(g_, col_t, col_g) + lb_
                nb_t.append(tv_); nb_g.append(np.interp(tv_, col_t, Gg_) + rng.standard_normal(len(tv_)) * 5.0)
            NBR_SIDE[sid] = np.concatenate(nb_t).astype(np.float32)
            NBR_SIDE[sid + "__gr"] = np.concatenate(nb_g).astype(np.float32)
        out.to_csv(os.path.join(OUT, f"{sid}__horizontal_well.csv"), index=False)
        pd.DataFrame(dict(TVT=tw_out_t, GR=tw_out_g, Geology="")).to_csv(
            os.path.join(OUT, f"{sid}__typewell.csv"), index=False)
        made.append(dict(id=sid, donor=donor, mode=recipe, noise_cat=["clean", "real", "high"][cat], noise_sigma=sigma))
        if len(made) % 500 == 0:
            print(f"  {len(made)}/{NSYN} ({time.time()-t0:.0f}s)", flush=True)
    np.savez_compressed(os.path.join(OUT, "pseudo_st.npz"), **ST_SIDE)
    if NBR_SIDE:
        np.savez_compressed(os.path.join(OUT, "pseudo_nbr.npz"), **NBR_SIDE)
    json.dump(dict(tag=TAG, nsyn=len(made), recipe=RECIPE, seed=SEED,
                   knobs=dict(lth=LTH, thick_l=THICK_L, amp_l=AMP_L, off_l=OFF_L,
                              tw_stretch=TW_STRETCH, tw_drop=TW_DROP, tw_noise=TW_NOISE,
                              fault_rate=FAULT_RATE, fault_ft=FAULT_FT, gap_rate=GAP_RATE,
                              noise_lvls=NOISE_LVLS, noise_mix=NOISE_MIX,
                              st_err=ST_ERR, st_clen=ST_CLEN, st_datum=ST_DATUM,
                              st_mix=ST_MIX, mismatch_p=MISMATCH_P, syn_nbr=SYN_NBR),
                   wells=made),
              open(os.path.join(OUT, "manifest.json"), "w"), indent=1)
    cnt = {}
    for m in made: cnt[m["mode"]] = cnt.get(m["mode"], 0) + 1
    print(f"pseudo_{TAG}: {len(made)} wells {cnt} 棄却={rej} -> {OUT} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
