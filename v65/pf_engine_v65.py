"""
v65 PFエンジン — v38コアの拡張パラメータ版(Fable最終便)
=====================================================
v38 iterative_pf_boost の GPU物理PF を、錨(struct)を P パラメータ化し、
誤差解剖の知見に対応する新ノブを追加したもの:
  use_anchor / anchor_mult(錨 on/off・強度を trial 単位で探索可能)
  anchor_ramp_md(錨の遅延起動: 錨近傍の引きずり対策=v41教訓)
  lk_floor(尤度の一様混合床: 洗掘/スパイク/facies不整合で正解粒子が死ぬ問題のロバスト化)
  gr_debias(prefix残差中央値でGRオフセット補正=井戸レベル工具バイアス)
  rate_target(momentum減衰先 0/初期dip: dip過外挿vs平均回帰)
  grid_step(typewell格子 0.2/0.35/0.5: v49「粗が良い」の本線導入)
wellmean錨モードは削除(v32: 井戸定数シフトは非転移)。
(以下は v38 原文)
v38 構造事前分布つき 物理PF(v19コア + 絶対TVTソフト錨 log-prior)
==================================================================
v19/iterative_pf_boost_v19.py のGPU物理PFに、GRフリーの構造モデル(K16 dip場)が出す
絶対TVT予測を「ソフト錨(対数prior)」として重み更新へ1項掛ける:

    lka = exp(-0.5 * ((TVT_particle - struct_tvt) / (ANCHOR_MULT * sig_struct))**2)
    w  *= lk(GR尤度) * lkp(dip率prior=死) * lka(構造錨)

要点(ユーザー方針):
  - 錨の強さは Optuna で振らない。sig_struct は構造モデルの実誤差(ドナー距離較正, struct_prior_v38.py)。
    priorの分散として物理的に決まる量なので、Optunaが強度を極限まで下げる問題が起きない。
  - ANCHOR_MULT だけを定数(既定1.0)/狭グリッドで扱う。既定はPF側パラメータのみ探索。
  - use_phys(dip率prior)は v19同様 False(死んでいる)。錨だけが本物の物理項。

依存: 先に  python v38/struct_prior_v38.py  で struct_prior_train.pkl を作っておく。
"""
import os, time, json, glob, math, warnings, pickle
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import optuna

warnings.filterwarnings("ignore")
np.seterr(all="ignore")

# ============================== CONFIG ==============================
PROJECT   = Path(r"C:\Users\kosaka256\Documents\rogii_claude")
DATA_DIR  = PROJECT / "rogii-wellbore-geology-prediction" / "train"
OUT_DIR   = PROJECT / "v65" / "artifacts_v65"; OUT_DIR.mkdir(parents=True, exist_ok=True)
V38_ART   = PROJECT / "v38" / "artifacts_v38"
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE     = torch.float32

USE_PHYS    = False       # dip率prior(v19で死んでいる)。錨とは別物、既定OFF。
USE_ANCHOR  = True        # 構造ソフト錨。False で v19 と等価(比較用)。
# ★prior汎用化: 環境変数で prior と倍率を切替(例: set V38_PRIOR=formation & set V38_MULT=30)
#   struct(既定)=K16 dip場 / make_priors_v38.py が作る selfdip / formation / dense など
PRIOR_NAME  = os.environ.get("V38_PRIOR", "v66_leanv1cumul")   # v66改善版struct を既定(v38→v66置換, PF A/B −0.226)。旧v38は V38_PRIOR=struct
ANCHOR_MULT = float(os.environ.get("V38_MULT", "20"))
# ★錨の累積方式: row=毎行独立掛け(現行, multで温度補正) / wellmean=telescoping井戸レベル錨(⑤)
#   wellmean: 粒子ごとの偏差ランニング平均 D_t に w *= N(D_t;0,σ_b)/N(D_{t-1};0,σ_b) を掛ける。
#   累積証拠は常に exp(-0.5 D_T²/σ_b²) の1回分=水増しゼロ。multノブ不要(σ_b=坑井平均σ×BMULT)。
ANCHOR_MODE = os.environ.get("V38_AMODE", "row")           # "row" | "wellmean"
BIAS_MULT   = float(os.environ.get("V38_BMULT", "1.0"))    # wellmean の σ_b 倍率(既定1=較正値そのまま)
STRUCT_PKL  = (V38_ART / "struct_prior_train.pkl" if PRIOR_NAME == "struct"
               else V38_ART / "priors" / f"prior_{PRIOR_NAME}_train.pkl")
print(f"[v65] PRIOR={PRIOR_NAME}  pkl={STRUCT_PKL.name}(錨のon/off/mult/rampはPパラメータ)")

SEED0        = 99999
STORAGE      = f"sqlite:///{(OUT_DIR / 'optuna.db').as_posix()}"

optuna.logging.set_verbosity(optuna.logging.WARNING)
CACHE = {}

# 構造prior(well -> {"tvt":arr,"sig":arr}, ev行順)を読み込む
_STRUCT = pickle.loads(STRUCT_PKL.read_bytes()) if STRUCT_PKL.exists() else {}
if not _STRUCT:
    print(f"WARNING: {STRUCT_PKL} が無い。struct_prior_v38.py を先に実行。錨は全wellでOFFになる。")


# ----------------------- 物理PF(v19 + 錨) -----------------------
def list_wells():
    return sorted({Path(p).stem.replace("__horizontal_well", "")
                   for p in glob.glob(str(DATA_DIR / "*__horizontal_well.csv"))})

def build_cache(wid):
    hw = pd.read_csv(DATA_DIR / f"{wid}__horizontal_well.csv")
    tw = pd.read_csv(DATA_DIR / f"{wid}__typewell.csv").sort_values("TVT")
    if "TVT" not in hw.columns:
        return None
    kn = hw[hw["TVT_input"].notna()]; ev = hw[hw["TVT_input"].isna()]
    if len(ev) == 0 or len(kn) < 10:
        return None
    tw_tvt = tw["TVT"].to_numpy(float); tw_gr = tw["GR"].to_numpy(float)
    if len(tw_tvt) < 3:
        return None
    tw_fb = float(np.nanmean(tw_gr)) if np.isfinite(np.nanmean(tw_gr)) else 0.
    gr_filled = hw["GR"].astype(float).interpolate(limit_direction="both").fillna(tw_fb).to_numpy(float)
    ev_pos = hw.index.get_indexer(ev.index); kpos = hw.index.get_indexer(kn.index)
    # ★v86: GR表現変換フック(env V86_REP。既定raw=無変更でv65挙動を保つ)。hw/twのGRを同一変換。
    _rep = os.environ.get("V86_REP", "raw")
    if _rep != "raw":
        _g0 = float(os.environ.get("V86_LOG_G0", "40.0"))
        if _rep == "logamp":
            f = lambda g: np.log1p(np.clip(g, 0, None) / _g0) * _g0
            gr_filled = f(gr_filled); tw_gr = f(tw_gr); tw_fb = float(np.nanmean(tw_gr))
        elif _rep == "affine":
            _kt = kn["TVT_input"].to_numpy(float); _kg = gr_filled[kpos]
            _mm = np.isfinite(tw_tvt) & np.isfinite(tw_gr)
            if _mm.sum() >= 8 and len(_kt) >= 100:
                _o = np.argsort(tw_tvt[_mm]); _tt, _tg = tw_tvt[_mm][_o], tw_gr[_mm][_o]
                _x = np.interp(_kt, _tt, _tg); _ok = np.isfinite(_x) & np.isfinite(_kg)
                if _ok.sum() >= 100:
                    _c = np.polyfit(_x[_ok], _kg[_ok], 1)
                    for _ in range(3):
                        _r = _kg[_ok] - np.polyval(_c, _x[_ok]); _s = np.median(np.abs(_r)) * 1.4826 + 1e-6
                        _c = np.polyfit(_x[_ok], _kg[_ok], 1, w=1.0 / (1.0 + (_r / (2 * _s)) ** 2))
                    _a, _b = float(_c[0]), float(_c[1])
                    if np.isfinite(_a) and 0.3 < _a < 3.0:
                        gr_filled = (gr_filled - _b) / _a; tw_fb = float(np.nanmean(tw_gr))
    ls = float(kn["TVT_input"].iloc[-1] + kn["Z"].iloc[-1])
    tail = kn.tail(30); dt = np.diff(tail["TVT_input"].values); dz = np.diff(tail["Z"].values)
    dm = np.diff(tail["MD"].values); m = dm > 0
    ir = float(np.median((dt + dz)[m] / dm[m])) if m.sum() >= 3 else 0.
    tv = ev["TVT"].values.astype(np.float64) if "TVT" in ev.columns else None
    nev = len(ev)
    # --- 構造錨(ev行順に整列)。cacheは常に錨データを保持し、ON/OFFは _pf_core の USE_ANCHOR で切替 ---
    sp = _STRUCT.get(wid)
    if sp is not None and len(sp["tvt"]) == nev:
        anc = np.asarray(sp["tvt"], np.float64); ancs = np.asarray(sp["sig"], np.float64)
    else:
        anc = np.zeros(nev, np.float64); ancs = np.full(nev, 1e9, np.float64)   # 構造prior欠損→この井戸は錨不能
    nbr_pairs = None
    if "nbr" in os.environ.get("V65_TWSRC", "tw"):
        cx, cy = float(hw["X"].mean()), float(hw["Y"].mean())
        nbr_pairs = _nbr_pairs_of(wid, cx, cy)
    return dict(md=ev["MD"].values.astype(np.float64), z=ev["Z"].values.astype(np.float64),
                gr_filled=gr_filled, ev_pos=ev_pos, kpos=kpos,
                kn_tvtin=kn["TVT_input"].values.astype(float),
                tw_tvt=tw_tvt, tw_gr=tw_gr, tw_fb=tw_fb, ls=ls, ir=ir, tvt_true=tv,
                anc=anc, ancs=ancs, nbr_pairs=nbr_pairs)

# ---- self/neighbor 疑似typewell(v28/v47 と同仕様) ----
MAX_DIST = 1500.0; K_MAX = 3; NEED_REFS = 2
_TRC = None; _REFC = {}

def _train_centroids():
    """train全坑井のXY重心(初回のみCSV走査、以後 artifacts にキャッシュ)。"""
    global _TRC
    if _TRC is None:
        fp = OUT_DIR / "train_centroids.json"
        if fp.exists():
            d = json.loads(fp.read_text())
            _TRC = (d["wids"], np.array(d["xy"]))
        else:
            wids = list_wells(); xy = []
            for w in wids:
                h = pd.read_csv(DATA_DIR / f"{w}__horizontal_well.csv", usecols=["X", "Y"])
                xy.append((float(h["X"].mean()), float(h["Y"].mean())))
            _TRC = (wids, np.array(xy))
            fp.write_text(json.dumps(dict(wids=wids, xy=[list(p) for p in xy])))
    return _TRC

def _ref_pairs(wid):
    """参照坑井の既知prefix (TVT_input, GR)(raw GR。平滑は make_inputs で hgr_smooth_r)。"""
    if wid not in _REFC:
        h = pd.read_csv(DATA_DIR / f"{wid}__horizontal_well.csv", usecols=["GR", "TVT_input"])
        m = h["TVT_input"].notna().to_numpy() & np.isfinite(h["GR"].to_numpy(float))
        _REFC[wid] = (h["TVT_input"].to_numpy(float)[m], h["GR"].to_numpy(float)[m])
    return _REFC[wid]

def _nbr_pairs_of(wid, cx, cy):
    """≤MAX_DISTの近傍(自分除く)最大K_MAX本の(TVT,GR)連結。NEED_REFS未満はNone(=twのまま)。"""
    wids, xy = _train_centroids()
    d = np.hypot(xy[:, 0] - cx, xy[:, 1] - cy)
    refs = []
    for j in np.argsort(d):
        if wids[j] == wid: continue
        if d[j] > MAX_DIST: break
        refs.append(wids[j])
        if len(refs) >= K_MAX: break
    if len(refs) < NEED_REFS: return None
    ts = []; gs_ = []
    for r in refs:
        t_, g_ = _ref_pairs(r); ts.append(t_); gs_.append(g_)
    return (np.concatenate(ts), np.concatenate(gs_))

def _bin_pairs(tvt, gr, gmin, step, G):
    """(TVT,GR)点群 → 格子bin中央値(grid_step共通)。無データbin=NaN。"""
    out = np.full(G, np.nan)
    idx = np.round((np.asarray(tvt, float) - gmin) / step).astype(int)
    ok = (idx >= 0) & (idx < G) & np.isfinite(gr) & np.isfinite(tvt)
    if ok.sum():
        srt = pd.Series(np.asarray(gr, float)[ok]).groupby(idx[ok]).median()
        out[srt.index.to_numpy()] = srt.to_numpy()
    return out

def _smooth(vals, fb, r):
    r = int(r); s = pd.Series(vals, dtype="float32").interpolate(limit_direction="both").fillna(float(fb))
    return s.to_numpy(np.float32) if r <= 0 else s.rolling(2 * r + 1, center=True, min_periods=1).mean().to_numpy(np.float32)

def _grid(tw_tvt, tw_gr, step=0.2):
    a, b = float(tw_tvt.min()), float(tw_tvt.max()); g = np.arange(a, b + step, step)
    return np.interp(g, tw_tvt, tw_gr).astype(np.float64), float(a), float(step)

def make_inputs(c, P):
    tw_gr_pf = _smooth(c["tw_gr"], c["tw_fb"], P["tw_gr_smooth_r"]).astype(np.float64)
    gg, gmin, gst = _grid(c["tw_tvt"], tw_gr_pf, step=float(P.get("grid_step", 0.2)))
    gr_sm = _smooth(c["gr_filled"], c["tw_fb"], P["hgr_smooth_r"])
    bias = 0.0
    if len(c["kpos"]) < 20:
        gs = P["gr_sig_def"]
    else:
        resid = gr_sm[c["kpos"]] - np.interp(c["kn_tvtin"], c["tw_tvt"], tw_gr_pf)
        gs = float(np.nanstd(resid)); gs = P["gr_sig_def"] if (not np.isfinite(gs) or gs <= 0) else gs
        if int(P.get("gr_debias", 0)):
            b_ = float(np.nanmedian(resid))
            if np.isfinite(b_): bias = b_       # 井戸レベルGRオフセット(工具較正差)を尤度前に補正
    gs = float(np.clip(gs * P["gr_sig_mult"], P["gr_sig_min"], max(P["gr_sig_max"], P["gr_sig_min"] + 1e-6)))
    if bias != 0.0:
        gr_sm = gr_sm - bias
    # ---- self/neighbor 疑似typewell混合(σ較正後。grid_step は全格子共通) ----
    src = str(P.get("tw_source", "tw"))
    if src != "tw":
        gg = gg.copy()
        if "nbr" in src and c.get("nbr_pairs") is not None:
            nt, ng = c["nbr_pairs"]
            ngs = _smooth(ng, float(np.nanmean(ng)), P["hgr_smooth_r"]).astype(np.float64)
            gh = _bin_pairs(nt, ngs, gmin, gst, len(gg))
            have = np.isfinite(gh); w = float(P.get("nbr_mix_w", 1.0))
            gg[have] = w * gh[have] + (1 - w) * gg[have]
        if "self" in src:
            gh = _bin_pairs(c["kn_tvtin"], gr_sm[c["kpos"]].astype(np.float64), gmin, gst, len(gg))
            have = np.isfinite(gh); w = float(P.get("self_mix_w", 1.0))
            gg[have] = w * gh[have] + (1 - w) * gg[have]
    return dict(md=c["md"], z=c["z"], gr=gr_sm[c["ev_pos"]].astype(np.float64),
                gg=gg, gmin=gmin, gst=gst, gs=gs, ls=c["ls"], ir=c["ir"], N=int(P["n_particles"]),
                anc=c["anc"], ancs=c["ancs"])

def _pf_core(md, z, gr, valid, grid, glen, vmin, step, gs, ls, ir, anc, ancs, P, N, device, gen, dtype=DTYPE):
    B, T = md.shape
    ALPHA = P["mom"]; RN = P["vn"]; PN = P["pn"]; IR = P["init_rate_std"]; IS = P["init_pos_std"]
    RP = P["rp"]; RR = P["rr"]; RESAMP = P["resamp"]; RATE_CLIP = P["rate_clip"]
    GR_POWER = P["gr_power"]; JUMP_PROB = P["jump_prob"]; JUMP_STD = P["jump_std"]; CLIP = P["tvt_clip_margin"]
    PHYS_SIG = P["phys_sig"]; USE_P = P["use_phys"]
    USE_A = bool(P.get("use_anchor", True)); AMUL = float(P.get("anchor_mult", 20.0))
    RAMP = float(P.get("anchor_ramp_md", 0.0))          # >0: 錨をmdsでランプイン(近錨域の引きずり防止)
    LKF = float(P.get("lk_floor", 0.0))                  # >0: 尤度の一様混合床(外れGRロバスト化)
    RT_IR = (str(P.get("rate_target", "zero")) == "ir")  # momentum減衰先: 0 or 初期dip
    def rn(): return torch.randn((B, N), generator=gen, device=device, dtype=dtype)
    pos = ls[:, None] + IS * rn(); rate = ir[:, None] + IR * rn()
    w = torch.full((B, N), 1.0 / N, device=device, dtype=dtype)
    log_lik = torch.zeros(B, device=device, dtype=torch.float64)
    pts = torch.zeros((B, T), device=device, dtype=dtype)
    tvt_lo = vmin - CLIP; tvt_hi = vmin + (glen.to(dtype) - 1) * step + CLIP
    glast = torch.gather(grid, 1, (glen - 1).clamp_min(0).unsqueeze(1)); g0 = grid[:, 0:1]
    pm = md[:, 0] - 1.0; arN = torch.arange(N, device=device, dtype=dtype)
    zero = torch.zeros((), device=device, dtype=dtype); dipp = ir[:, None]
    for i in range(T):
        act = valid[:, i]; cur = md[:, i]; dm = (cur - pm).clamp_min(1.0)
        rate_n = ALPHA * rate + ((1.0 - ALPHA) * dipp if RT_IR else 0.0) + RN * rn()
        if RATE_CLIP > 0: rate_n = rate_n.clamp(-RATE_CLIP, RATE_CLIP)
        pos_n = pos + rate_n * dm[:, None] + PN * rn()
        if JUMP_PROB > 0 and JUMP_STD > 0:
            jm = torch.rand((B, N), generator=gen, device=device, dtype=dtype) < JUMP_PROB
            pos_n = pos_n + torch.where(jm, JUMP_STD * rn(), zero)
        zi = z[:, i][:, None]; tvt = (pos_n - zi).clamp(tvt_lo[:, None], tvt_hi[:, None]); pos_n = tvt + zi
        am = act[:, None]; pos = torch.where(am, pos_n, pos); rate = torch.where(am, rate_n, rate)
        gri = gr[:, i]; obs = act & ~torch.isnan(gri)
        v = pos - zi; ii = (v - vmin[:, None]) / step[:, None]; i0 = torch.floor(ii).long()
        below = i0 < 0; above = i0 >= (glen - 1)[:, None]
        i0c = i0.clamp_min(0); i0c = torch.minimum(i0c, (glen - 2).clamp_min(0)[:, None])
        t = ii - i0c.to(dtype); gA = torch.gather(grid, 1, i0c); gB = torch.gather(grid, 1, i0c + 1)
        eg = gA * (1 - t) + gB * t; eg = torch.where(below, g0, eg); eg = torch.where(above, glast, eg)
        d = ((gri[:, None] - eg) / gs[:, None]).abs(); dp = d ** GR_POWER
        lk = torch.where(dp < 600.0, torch.exp(-0.5 * dp), zero)
        if LKF > 0:
            lk = (1.0 - LKF) * lk + LKF                  # 一様混合床: 外れGRで正解粒子が即死しない
        lk = lk.clamp_min(1e-300)
        avg = (w * lk).sum(1)
        log_lik = log_lik + torch.where(obs, torch.log(avg.clamp_min(1e-300).double()),
                                        torch.zeros((), device=device, dtype=torch.float64))
        if USE_P:
            dphys = (rate - dipp) / PHYS_SIG; lkp = torch.exp(-0.5 * dphys * dphys).clamp_min(1e-300)
        else:
            lkp = torch.ones_like(w)
        if USE_A:                                   # ★ row錨(毎行独立掛け, mult=温度補正, ramp=遅延起動)
            asig = (ancs[:, i][:, None] * AMUL).clamp_min(1e-6)
            da = (v - anc[:, i][:, None]) / asig    # v = pos - zi = 粒子のTVT
            da2 = da * da
            if RAMP > 0:
                tf = ((cur - md[:, 0]).clamp_min(0.0) / RAMP).clamp(0.0, 1.0)[:, None]
                da2 = da2 * tf                       # 温度でランプイン(近錨域は錨を弱く)
            lka = torch.where(da2 < 1200.0, torch.exp(-0.5 * da2), zero).clamp_min(1e-300)
        else:
            lka = torch.ones_like(w)
        w_new = torch.where(obs[:, None], w * lk * lkp * lka, w * lkp * lka)
        ws = w_new.sum(1, keepdim=True)
        w = torch.where(ws > 0, w_new / ws.clamp_min(1e-300), torch.full_like(w, 1.0 / N))
        ne = (w * w).sum(1); need = act & ((1.0 / ne) < (RESAMP * N))
        if bool(need.any()):
            cumw = torch.cumsum(w, 1); u0 = torch.rand((B, 1), generator=gen, device=device, dtype=dtype) * (1.0 / N)
            u = u0 + arN[None, :] / N; idx = torch.searchsorted(cumw, u, right=False).clamp(0, N - 1)
            pos_rs = torch.gather(pos, 1, idx) + RP * rn(); rate_rs = torch.gather(rate, 1, idx) + RR * rn()
            if RATE_CLIP > 0: rate_rs = rate_rs.clamp(-RATE_CLIP, RATE_CLIP)
            nm = need[:, None]; pos = torch.where(nm, pos_rs, pos); rate = torch.where(nm, rate_rs, rate)
            w = torch.where(nm, torch.full_like(w, 1.0 / N), w)
        pts[:, i] = (w * (pos - zi)).sum(1); pm = torch.where(act, cur, pm)
    return pts, log_lik

def _pad(inps, device, dtype=DTYPE):
    W = len(inps); Tmax = max(len(x["md"]) for x in inps); Gmax = max(len(x["gg"]) for x in inps)
    md = torch.zeros((W, Tmax), dtype=dtype); z = torch.zeros((W, Tmax), dtype=dtype)
    gr = torch.full((W, Tmax), float("nan"), dtype=dtype); valid = torch.zeros((W, Tmax), dtype=torch.bool)
    grid = torch.zeros((W, Gmax), dtype=dtype); glen = torch.zeros(W, dtype=torch.long)
    vmin = torch.zeros(W, dtype=dtype); step = torch.zeros(W, dtype=dtype); gs = torch.zeros(W, dtype=dtype)
    ls = torch.zeros(W, dtype=dtype); ir = torch.zeros(W, dtype=dtype)
    anc = torch.zeros((W, Tmax), dtype=dtype); ancs = torch.full((W, Tmax), 1e9, dtype=dtype)
    for b, x in enumerate(inps):
        Tn = len(x["md"]); G = len(x["gg"])
        md[b, :Tn] = torch.from_numpy(x["md"].astype("float32"))
        if Tn < Tmax: md[b, Tn:] = md[b, Tn - 1]
        z[b, :Tn] = torch.from_numpy(x["z"].astype("float32"))
        gr[b, :Tn] = torch.from_numpy(x["gr"].astype("float32")); valid[b, :Tn] = True
        grid[b, :G] = torch.from_numpy(x["gg"].astype("float32"))
        if G < Gmax: grid[b, G:] = grid[b, G - 1]
        glen[b] = G; vmin[b] = x["gmin"]; step[b] = x["gst"]; gs[b] = x["gs"]; ls[b] = x["ls"]; ir[b] = x["ir"]
        anc[b, :Tn] = torch.from_numpy(x["anc"].astype("float32"))
        if Tn < Tmax: anc[b, Tn:] = anc[b, Tn - 1]
        ancs[b, :Tn] = torch.from_numpy(x["ancs"].astype("float32"))
    to = lambda tt: tt.to(device)
    return dict(md=to(md), z=to(z), gr=to(gr), valid=to(valid), grid=to(grid), glen=to(glen),
                vmin=to(vmin), step=to(step), gs=to(gs), ls=to(ls), ir=to(ir), anc=to(anc), ancs=to(ancs))

def run_forward_ens(inps, P, device, seed, n_seeds):
    N = int(P["n_particles"]); S = n_seeds; W = len(inps); pad = _pad(inps, device)
    rep = lambda tt: tt.repeat_interleave(S, 0); gen = torch.Generator(device=device); gen.manual_seed(seed)
    pts, ll = _pf_core(rep(pad["md"]), rep(pad["z"]), rep(pad["gr"]), rep(pad["valid"]), rep(pad["grid"]),
                       pad["glen"].repeat_interleave(S), pad["vmin"].repeat_interleave(S), pad["step"].repeat_interleave(S),
                       pad["gs"].repeat_interleave(S), pad["ls"].repeat_interleave(S), pad["ir"].repeat_interleave(S),
                       rep(pad["anc"]), rep(pad["ancs"]), P, N, device, gen)
    pts = pts.view(W, S, -1).double().cpu().numpy(); ll = ll.view(W, S).cpu().numpy()
    outs = []
    for w in range(W):
        T = len(inps[w]["md"]); pr = pts[w, :, :T]
        llw = ll[w]
        if np.isfinite(llw).any():
            mx = np.nanmax(llw[np.isfinite(llw)])
            lk = np.where(np.isfinite(llw), llw - mx, -np.inf)
            ww = np.exp(lk / max(P["likelihood_scale"], 1e-6))
            s = ww.sum()
            ww = ww / s if s > 1e-300 else np.full(S, 1.0 / S)
        else:
            ww = np.full(S, 1.0 / S)
        outs.append((ww[:, None] * pr).sum(0))
    return outs


# ----------------------- 評価(対象well指定) -----------------------
def predict_errs(P, subset, seed, n_seeds, chunk):
    res = {}
    for c0 in range(0, len(subset), chunk):
        sub = subset[c0:c0 + chunk]
        inps = [make_inputs(CACHE[w], P) for w in sub]
        preds = run_forward_ens(inps, P, DEVICE, seed, n_seeds)
        for k, w in enumerate(sub):
            res[w] = preds[k] - CACHE[w]["tvt_true"]
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    return res

def overall_rmse_on(P, subset, seed, n_seeds, chunk):
    errs = predict_errs(P, subset, seed, n_seeds, chunk)
    e = np.concatenate([errs[w] for w in subset]); e = e[np.isfinite(e)]
    if e.size == 0 or not np.isfinite(e).all():
        return 1e3
    return float(np.sqrt(np.mean(e ** 2)))

def perwell_rmse(P, subset, seed, n_seeds, chunk):
    errs = predict_errs(P, subset, seed, n_seeds, chunk)
    out = {}
    for w, e in errs.items():
        e = e[np.isfinite(e)]
        out[w] = float(np.sqrt(np.mean(e ** 2))) if e.size else 1e3
    return out


# ----------------------- Optuna 探索空間(v19と同一。anchor_sigは足さない) -----------------------
def suggest_params(trial):
    return dict(
        n_particles      = trial.suggest_categorical("n_particles", [400, 500, 600, 700, 800]),
        likelihood_scale = trial.suggest_float("likelihood_scale", 6.0, 60.0, log=True),
        init_pos_std     = trial.suggest_float("init_pos_std", 0.1, 4.0, log=True),
        init_rate_std    = trial.suggest_float("init_rate_std", 0.0003, 0.03, log=True),
        mom              = trial.suggest_float("mom", 0.990, 0.99998),
        vn               = trial.suggest_float("vn", 0.0003, 0.012, log=True),
        pn               = trial.suggest_float("pn", 0.002, 0.06, log=True),
        rp               = trial.suggest_float("rp", 0.1, 1.5, log=True),
        rr               = trial.suggest_float("rr", 0.00005, 0.004, log=True),
        resamp           = trial.suggest_float("resamp", 0.03, 0.6, log=True),
        gr_sig_min       = trial.suggest_float("gr_sig_min", 4.0, 12.0),
        gr_sig_max       = trial.suggest_float("gr_sig_max", 22.0, 60.0),
        gr_sig_def       = trial.suggest_float("gr_sig_def", 70.0, 160.0),
        gr_sig_mult      = trial.suggest_float("gr_sig_mult", 1.6, 4.0),
        tvt_clip_margin  = trial.suggest_float("tvt_clip_margin", 50.0, 200.0),
        rate_clip        = trial.suggest_categorical("rate_clip", [0.0, 1.5, 2.0, 2.5, 3.0, 5.0]),
        gr_power         = trial.suggest_float("gr_power", 1.4, 3.2),
        hgr_smooth_r     = trial.suggest_categorical("hgr_smooth_r", [0, 1, 2, 3, 4, 6]),
        tw_gr_smooth_r   = trial.suggest_categorical("tw_gr_smooth_r", [0, 1, 2]),
        jump_prob        = trial.suggest_categorical("jump_prob", [0.0, 0.002, 0.005, 0.01, 0.02]),
        jump_std         = trial.suggest_categorical("jump_std", [5.0, 10.0, 15.0, 20.0]),
        phys_sig         = trial.suggest_float("phys_sig", 0.003, 5.0, log=True),
        use_phys         = USE_PHYS,
        # ---- v65 拡張ノブ ----
        use_anchor       = trial.suggest_categorical("use_anchor", [True, False]),
        anchor_mult      = trial.suggest_float("anchor_mult", 3.0, 80.0, log=True),
        anchor_ramp_md   = trial.suggest_categorical("anchor_ramp_md", [0.0, 300.0, 600.0, 1200.0, 2400.0]),
        lk_floor         = trial.suggest_float("lk_floor", 1e-5, 0.05, log=True),
        gr_debias        = trial.suggest_categorical("gr_debias", [0, 1]),
        rate_target      = trial.suggest_categorical("rate_target", ["zero", "ir"]),
        grid_step        = trial.suggest_categorical("grid_step", [0.2, 0.35, 0.5]),
        # ---- self/neighbor(source は env V65_TWSRC で study 単位に固定。混合比のみ探索) ----
        **({"self_mix_w": trial.suggest_float("self_mix_w", 0.1, 1.0)}
           if "self" in os.environ.get("V65_TWSRC", "tw") else {}),
        **({"nbr_mix_w": trial.suggest_float("nbr_mix_w", 0.1, 1.0)}
           if "nbr" in os.environ.get("V65_TWSRC", "tw") else {}),
    )

_V65_DEFAULTS = dict(use_phys=False, phys_sig=1.0, use_anchor=True, anchor_mult=20.0, anchor_ramp_md=0.0,
                     lk_floor=0.0, gr_debias=0, rate_target="zero", grid_step=0.2,
                     tw_source="tw", self_mix_w=1.0, nbr_mix_w=1.0)

def full_params(best_params):
    return {**_V65_DEFAULTS, **best_params, "use_phys": USE_PHYS,
            "tw_source": os.environ.get("V65_TWSRC", "tw")}
