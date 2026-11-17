# -*- coding: utf-8 -*-
"""
v93 PFエンジン — GRフリー錨(v91 OOF, LB10.877)を anc/ancs に使う位置錨PF。
================================================================================
v65 pf_engine の構造を理解して書き直したクリーン版(import/コピペせず、不要な
v86 GR表現フック・wellmean錨モードは削除して最適化)。人間可読を優先。

錨の効かせ方(位置錨, v65と同じ数理):
    asig = clamp_min(ancs * anchor_mult, 1e-6)
    da   = (particle_TVT - anc) / asig
    lka  = exp(-0.5 * da^2)          # anchor_ramp_md>0 なら近錨域を温度でランプイン
    w   *= lk(GR尤度) * lka(錨)      # dip率prior(use_phys)は死んでいるので既定OFF

錨 prior(anc=目標TVT, ancs=不確実性σ)は GRフリー錨 pkl から読む:
    v93/artifacts_v93/grfree_anchor_train.pkl = {wid: {"tvt":(ev行,), "sig":(ev行,)}}
STEP1でmult20(=v66場用較正)は10ft精度のGRフリーに強すぎと判明 → anchor_mult 等を
optuna(tune_v93.py)で GRフリー錨用に再探索する。
"""
import os, glob, pickle
from pathlib import Path
import numpy as np
import pandas as pd
import torch

PROJECT   = Path(os.environ.get("V93_ROOT", r"C:\Users\kosaka256\Documents\rogii_claude"))
DATA_DIR  = PROJECT / "rogii-wellbore-geology-prediction" / "train"
ART       = PROJECT / "v94" / "artifacts_v94"; ART.mkdir(parents=True, exist_ok=True)  # ★v94独立(v93非参照)
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE     = torch.float32

# 錨 prior: GRフリー錨(v93/artifacts_v93 に格納済み)。env で差替可(将来 test 用など)
ANCHOR_PKL = Path(os.environ.get("V93_ANCHOR_PKL", str(ART / "grfree_anchor_train.pkl")))
_ANCHOR = pickle.loads(ANCHOR_PKL.read_bytes()) if ANCHOR_PKL.exists() else {}
if not _ANCHOR:
    print(f"WARNING: {ANCHOR_PKL} が無い。全wellで錨OFF(ancs=1e9)になる。")
else:
    print(f"[v93] 錨prior={ANCHOR_PKL.name}  {len(_ANCHOR)}井 (anchor_mult/ramp/on-off は P パラメータ)")

CACHE = {}   # wid -> build_cache 結果


def list_wells():
    return sorted({Path(p).stem.replace("__horizontal_well", "")
                   for p in glob.glob(str(DATA_DIR / "*__horizontal_well.csv"))})


# ---------------- 前処理: 1井 → PF入力キャッシュ ----------------
def _gauss1d(a, sigma):
    """自己完結ガウシアン平滑(scipy非依存, reflect端処理)。sigma[サンプル]。"""
    a = np.asarray(a, float)
    r = int(round(4 * sigma))
    if sigma <= 0 or r < 1:
        return a
    r = min(r, len(a) - 1)
    if r < 1:
        return a
    x = np.arange(-r, r + 1); k = np.exp(-0.5 * (x / sigma) ** 2); k = k / k.sum()
    return np.convolve(np.pad(a, r, mode="reflect"), k, mode="valid")


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
    tw_fb = float(np.nanmean(tw_gr)) if np.isfinite(np.nanmean(tw_gr)) else 0.0
    gr_filled = hw["GR"].astype(float).interpolate(limit_direction="both").fillna(tw_fb).to_numpy(float)
    if int(os.environ.get("GR_DERIV", "0")):
        # ★dGR: typewellはTVT空間で微分、horizontalは行(MD)方向で微分。両方GRライクなスケールへ正規化。
        dtw = np.gradient(tw_gr, tw_tvt); s = float(np.nanstd(dtw))
        tw_gr = (dtw / (s + 1e-9) * 20.0) if s > 0 else dtw
        dh = np.gradient(gr_filled); s2 = float(np.nanstd(dh))
        gr_filled = (dh / (s2 + 1e-9) * 20.0) if s2 > 0 else dh
        tw_fb = 0.0
    elif int(os.environ.get("GR_BANDPASS", "0")):
        # ★band-pass: gaussian(σ_lo)−gaussian(σ_hi) で中スケール(細ベッド)帯だけ残す。GRライクなスケールへ正規化。
        # σは物理ft指定→系列ごとの間隔でサンプルへ変換(typewell=TVT間隔, horizontal=MD間隔)。
        slo_ft = float(os.environ.get("BP_SLO_FT", "2")); shi_ft = float(os.environ.get("BP_SHI_FT", "32"))

        def _bp(a, spacing):
            a = np.asarray(a, float); m = np.isfinite(a)
            if m.sum() < 3:
                return np.zeros_like(a)
            if not m.all():                        # NaNは線形補間(gaussianがNaN伝播しないよう)
                a = np.interp(np.arange(len(a)), np.where(m)[0], a[m])
            sp = spacing if (np.isfinite(spacing) and spacing > 1e-6) else 1.0
            slo = max(slo_ft / sp, 0.6); shi = max(shi_ft / sp, slo * 2.0)   # ft→サンプル
            d = _gauss1d(a, slo) - _gauss1d(a, shi); sd = float(np.std(d))
            return (d / (sd + 1e-9) * 20.0) if sd > 0 else d
        tw_sp = float(np.median(np.abs(np.diff(tw_tvt)))) if len(tw_tvt) > 1 else 1.0
        md_all = hw["MD"].to_numpy(float)
        md_sp = float(np.median(np.abs(np.diff(md_all)))) if len(md_all) > 1 else 1.0
        tw_gr = _bp(tw_gr, tw_sp); gr_filled = _bp(gr_filled, md_sp); tw_fb = 0.0
    ev_pos = hw.index.get_indexer(ev.index); kpos = hw.index.get_indexer(kn.index)
    ls = float(kn["TVT_input"].iloc[-1] + kn["Z"].iloc[-1])         # 最終既知の地質面レベル
    tail = kn.tail(30); dt = np.diff(tail["TVT_input"].values); dz = np.diff(tail["Z"].values)
    dm = np.diff(tail["MD"].values); m = dm > 0
    ir = float(np.median((dt + dz)[m] / dm[m])) if m.sum() >= 3 else 0.0   # prefix dip率
    nev = len(ev)
    # 錨(ev行順に整列)。欠損はこの井を錨不能(ancs=1e9)にする
    sp = _ANCHOR.get(wid)
    if sp is not None and len(sp["tvt"]) == nev:
        anc = np.asarray(sp["tvt"], np.float64); ancs = np.asarray(sp["sig"], np.float64)
    else:
        anc = np.zeros(nev, np.float64); ancs = np.full(nev, 1e9, np.float64)
    return dict(
        md=ev["MD"].to_numpy(np.float64), z=ev["Z"].to_numpy(np.float64),
        gr_filled=gr_filled, ev_pos=ev_pos, kpos=kpos,
        kn_tvtin=kn["TVT_input"].to_numpy(np.float64),
        tw_tvt=tw_tvt.astype(np.float64), tw_gr=tw_gr.astype(np.float64), tw_fb=tw_fb,
        ls=ls, ir=ir, anc=anc, ancs=ancs,
        tvt_true=ev["TVT"].to_numpy(np.float64) if "TVT" in ev.columns else None)


# ---------------- 小物 ----------------
def _smooth(vals, fb, r):
    r = int(r)
    if r <= 0:
        return np.asarray(vals, float)
    s = pd.Series(vals).rolling(2 * r + 1, center=True, min_periods=1).mean().to_numpy()
    return np.where(np.isfinite(s), s, fb)

def _grid(tw_tvt, tw_gr, step=0.2):
    m = np.isfinite(tw_tvt) & np.isfinite(tw_gr)
    tt, tg = tw_tvt[m], tw_gr[m]
    o = np.argsort(tt); tt, tg = tt[o], tg[o]
    vmin = float(tt[0]); vmax = float(tt[-1])
    n = max(int(np.ceil((vmax - vmin) / step)) + 1, 2)
    xs = vmin + np.arange(n) * step
    gg = np.interp(xs, tt, tg)
    return gg.astype(np.float64), vmin, float(step)


# ---------------- PF入力(尤度σ/格子/self-nbr) ----------------
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
            if np.isfinite(b_): bias = b_          # 井戸レベルGRオフセット補正
    gs = float(np.clip(gs * P["gr_sig_mult"], P["gr_sig_min"], max(P["gr_sig_max"], P["gr_sig_min"] + 1e-6)))
    gs = gs * float(P.get("gs_scale", 1.0))        # ★post-clip の gs 倍率(GR尤度を緩める。tuner で 1.0-1.5)
    if bias != 0.0:
        gr_sm = gr_sm - bias
    return dict(md=c["md"], z=c["z"], gr=gr_sm[c["ev_pos"]].astype(np.float64),
                gg=gg, gmin=gmin, gst=gst, gs=gs, ls=c["ls"], ir=c["ir"],
                N=int(P["n_particles"]), anc=c["anc"], ancs=c["ancs"])


# ---------------- GPU PF 本体 ----------------
def _tw_interp(vv, grid, glen, vmin, step, g0, glast, dtype):
    """粒子TVT vv(B,N) を typewell格子(grid)で線形補間→typewell GR(B,N)。範囲外は端値。"""
    ii = (vv - vmin[:, None]) / step[:, None]; i0 = torch.floor(ii).long()
    below = i0 < 0; above = i0 >= (glen - 1)[:, None]
    i0c = i0.clamp_min(0); i0c = torch.minimum(i0c, (glen - 2).clamp_min(0)[:, None])
    t = ii - i0c.to(dtype); gA = torch.gather(grid, 1, i0c); gB = torch.gather(grid, 1, i0c + 1)
    eg = gA * (1 - t) + gB * t; eg = torch.where(below, g0, eg); eg = torch.where(above, glast, eg)
    return eg


def _pf_core(md, z, gr, valid, grid, glen, vmin, step, gs, ls, ir, anc, ancs, P, N, device, gen, dtype=DTYPE):
    B, T = md.shape
    ALPHA = P["mom"]; RN = P["vn"]; PN = P["pn"]; IR = P["init_rate_std"]; IS = P["init_pos_std"]
    RP = P["rp"]; RR = P["rr"]; RESAMP = P["resamp"]; RATE_CLIP = P["rate_clip"]
    GR_POWER = P["gr_power"]; JUMP_PROB = P["jump_prob"]; JUMP_STD = P["jump_std"]; CLIP = P["tvt_clip_margin"]
    USE_A = bool(P.get("use_anchor", True)); AMUL = float(P.get("anchor_mult", 20.0))
    RAMP = float(P.get("anchor_ramp_md", 0.0)); LKF = float(P.get("lk_floor", 0.0))
    RT_IR = (str(P.get("rate_target", "zero")) == "ir")
    # ★shape項: 各粒子のrateで窓[i-K,i+K]のTVTを軌跡外挿し typewell窓と相関(真TVT不使用)。
    #   相関はレベル不変=振幅(点尤度)と直交し形状のみを効かせる。shape=0で完全に無効(baseline一致)。
    SHAPE = int(P.get("shape", 0)); SHK = int(P.get("shape_k", 4)); SHB = float(P.get("shape_beta", 4.0))
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
            lk = (1.0 - LKF) * lk + LKF
        lk = lk.clamp_min(1e-300)
        if SHAPE:                                    # ★窓相関(形状)項: rateで軌跡外挿→typewell窓と相関
            wn = float(2 * SHK + 1)
            sh = torch.zeros_like(cur); sh2 = torch.zeros_like(cur)       # HG(B,) 積算
            st = torch.zeros_like(w); st2 = torch.zeros_like(w); sht = torch.zeros_like(w)  # TW(B,N) 積算
            hg_ok = torch.ones_like(cur, dtype=torch.bool)
            for j in range(-SHK, SHK + 1):
                ij = min(max(i + j, 0), T - 1)
                hgj = gr[:, ij]; hg_ok = hg_ok & torch.isfinite(hgj); hgj = torch.nan_to_num(hgj, nan=0.0)
                dmd = (md[:, ij] - cur)[:, None]; dzj = (z[:, ij] - z[:, i])[:, None]
                tvt_j = (v + rate * dmd - dzj).clamp(tvt_lo[:, None], tvt_hi[:, None])   # 粒子軌跡のTVT(真TVT不使用)
                twj = _tw_interp(tvt_j, grid, glen, vmin, step, g0, glast, dtype)         # typewell窓(B,N)
                sh = sh + hgj; sh2 = sh2 + hgj * hgj
                st = st + twj; st2 = st2 + twj * twj; sht = sht + hgj[:, None] * twj
            covht = wn * sht - sh[:, None] * st
            varh = (wn * sh2 - sh * sh).clamp_min(0.0); vart = (wn * st2 - st * st).clamp_min(0.0)
            corr = (covht / (varh[:, None].sqrt() * vart.sqrt()).clamp_min(1e-6)).clamp(-1.0, 1.0)
            shp = torch.exp(SHB * (corr - 1.0))       # corr=1で1, 低いほど減衰
            ok = (obs & hg_ok & (varh > 1e-6))[:, None]
            lk = torch.where(ok, (lk * shp).clamp_min(1e-300), lk)
        avg = (w * lk).sum(1)
        log_lik = log_lik + torch.where(obs, torch.log(avg.clamp_min(1e-300).double()),
                                        torch.zeros((), device=device, dtype=torch.float64))
        if USE_A:                                    # 位置錨(毎行独立, mult=温度, ramp=遅延起動)
            asig = (ancs[:, i][:, None] * AMUL).clamp_min(1e-6)
            da = (v - anc[:, i][:, None]) / asig     # v = 粒子のTVT
            da2 = da * da
            if RAMP > 0:
                tf = ((cur - md[:, 0]).clamp_min(0.0) / RAMP).clamp(0.0, 1.0)[:, None]
                da2 = da2 * tf
            lka = torch.where(da2 < 1200.0, torch.exp(-0.5 * da2), zero).clamp_min(1e-300)
        else:
            lka = torch.ones_like(w)
        w_new = torch.where(obs[:, None], w * lk * lka, w * lka)
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
        Tn = len(x["md"]); Gn = len(x["gg"])
        md[b, :Tn] = torch.from_numpy(x["md"].astype("float32")); z[b, :Tn] = torch.from_numpy(x["z"].astype("float32"))
        gr[b, :Tn] = torch.from_numpy(x["gr"].astype("float32")); valid[b, :Tn] = True
        grid[b, :Gn] = torch.from_numpy(x["gg"].astype("float32")); glen[b] = Gn
        vmin[b] = x["gmin"]; step[b] = x["gst"]; gs[b] = x["gs"]; ls[b] = x["ls"]; ir[b] = x["ir"]
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
        T = len(inps[w]["md"]); pr = pts[w, :, :T]; llw = ll[w]
        if np.isfinite(llw).any():
            mx = np.nanmax(llw[np.isfinite(llw)]); lk = np.where(np.isfinite(llw), llw - mx, -np.inf)
            ww = np.exp(lk / max(P["likelihood_scale"], 1e-6)); s = ww.sum()
            ww = ww / s if s > 1e-300 else np.full(S, 1.0 / S)
        else:
            ww = np.full(S, 1.0 / S)
        outs.append((ww[:, None] * pr).sum(0))
    return outs


# ---------------- 評価 ----------------
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
    return float(np.sqrt(np.mean(e ** 2))) if e.size else 1e3


# ---------------- Optuna 探索空間 / デフォルト ----------------
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
        # ---- 錨(GRフリー)ノブ: STEP1でmult20は強すぎ→下限も探れるよう据置(3-80 log) ----
        use_anchor       = trial.suggest_categorical("use_anchor", [True, False]),
        anchor_mult      = trial.suggest_float("anchor_mult", 3.0, 80.0, log=True),
        anchor_ramp_md   = trial.suggest_categorical("anchor_ramp_md", [0.0, 300.0, 600.0, 1200.0, 2400.0]),
        lk_floor         = trial.suggest_float("lk_floor", 1e-5, 0.05, log=True),
        gr_debias        = trial.suggest_categorical("gr_debias", [0, 1]),
        rate_target      = trial.suggest_categorical("rate_target", ["zero", "ir"]),
        grid_step        = trial.suggest_categorical("grid_step", [0.2, 0.35, 0.5]),
    )

_DEFAULTS = dict(use_anchor=True, anchor_mult=20.0, anchor_ramp_md=0.0, lk_floor=0.0,
                 gr_debias=0, rate_target="zero", grid_step=0.2)

def full_params(best_params):
    return {**_DEFAULTS, **best_params}
