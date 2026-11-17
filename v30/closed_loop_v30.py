# -*- coding: utf-8 -*-
"""
v30 closed-loop: GBM×NN ブレンド(OOF)を prior に、r0 固定ラグ smoother を再走行(pseudo-satellite)
====================================================================================
狙い(docs/beyond_tabular_proposals.md ②):
  現ベスト点推定(v28 GBM × v29 NN blend, OOF apply_pp ≈6.87)を prior として
  smooth PF(r0, lag192 = LB6.786 実証エンジン)の forward 重みにアンカー項
      w *= exp(-0.5 * ((tvt粒子 - tvt_prior_t) / PRIOR_SIG)^2)
  を掛けて再走行し、**最終出力を prior 誘導 smoother の軌跡**にする。
  prior が大域配置(datum/高さ)を縛り、PF は GR整合+dip連続の局所補正に専念する分業。

リーク無し: prior は train では全て OOF(gbm_oof.npy + nn_oof_{RUN}.npz, fold は井戸単位)。
seed加重の log_lik は GR のみ(prior は尤度に混ぜない = v25 pseudosat 設計を踏襲)。

評価: overall = 全行プール RMSE(smoothed / forward / β·smoother+(1-β)·prior の粗いβ)。
  基準 = prior 自身の RMSE(≈ results.json の blend_pp)。どの σ でも下回らなければ撤退。
  PRIOR_SIGS に None(=priorなし素smoother) を含めてベースラインも同時取得。

出力: v30/artifacts_v30/closed_loop_r0.json(σごと成績, 再開可) + preds_sig*.npz(全行予測)
実行: rogii_claude/.venv/Scripts/python.exe v30/closed_loop_v30.py   (実行はユーザー)
"""

import glob, json, math, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")
np.seterr(all="ignore")

# ============================== CONFIG ==============================
PROJECT  = Path(r"C:\Users\kosaka256\Documents\rogii_claude")
DATA_DIR = PROJECT / "rogii-wellbore-geology-prediction" / "train"
V28_ART  = PROJECT / "v28" / "artifacts_v28"
V29_ART  = PROJECT / "v29" / "artifacts_v29"
V29B_ART = PROJECT / "v29" / "artifacts_v29b"
OUT_DIR  = PROJECT / "v30" / "artifacts_v30"; OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "closed_loop_r0.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.float32

SEED    = 0
N_SEED  = 32            # r0_seed32 regime(チューニング=推論一致則)
CHUNK   = 100           # 井戸バッチ(OOM調整)
SMOOTH_LAG = 192        # LB6.786 実証の固定ラグ

# prior の帯域スイープ(ft)。None = prior なし(素の smoother ベースライン)
PRIOR_SIGS = [None, 12.0, 192.0, 96.0, 48.0, 24.0]
BETAS = np.arange(0.0, 1.01, 0.1)   # out = (1-β)·prior + β·smoother の粗い走査

# prior に使う NN RUN(artifacts_v29b/nn_oof_{RUN}.npz + results.json の blend_w)。先頭から見つかった物を使う
NN_RUN_CANDIDATES = ["full", "combo"]

# --- 固定 PF = r0_seed32(v19/artifacts_v19_32seeds/pf_bank.json[0], v25 smoother と同一) ---
R0_PARAMS = {
    "n_particles": 400, "likelihood_scale": 10.536319667317654,
    "init_pos_std": 0.9331541486020234, "init_rate_std": 0.00047753793163756304,
    "mom": 0.9997911117425017, "vn": 0.0006297672017716754, "pn": 0.005014009140727263,
    "rp": 0.7627309890952053, "rr": 0.00045663430474523125, "resamp": 0.032049849143995945,
    "gr_sig_min": 11.865793233968713, "gr_sig_max": 44.98927964612316, "gr_sig_def": 119.25397464636981,
    "gr_sig_mult": 3.058318309943017, "tvt_clip_margin": 150.56111180234504, "rate_clip": 2.0,
    "gr_power": 1.7826896851395453, "hgr_smooth_r": 0, "tw_gr_smooth_r": 0,
    "jump_prob": 0.002, "jump_std": 5.0, "phys_sig": 0.013666550145707291, "use_phys": False,
}
# ===================================================================


# ----------------------- データ(v25 tune_smoother と同一, 自己完結) -----------------------
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
    ls = float(kn["TVT_input"].iloc[-1] + kn["Z"].iloc[-1])
    tail = kn.tail(30); dt = np.diff(tail["TVT_input"].values); dz = np.diff(tail["Z"].values)
    dm = np.diff(tail["MD"].values); m = dm > 0
    ir = float(np.median((dt + dz)[m] / dm[m])) if m.sum() >= 3 else 0.
    tv = ev["TVT"].values.astype(np.float64) if "TVT" in ev.columns else None
    return dict(md=ev["MD"].values.astype(np.float64), z=ev["Z"].values.astype(np.float64),
                gr_filled=gr_filled, ev_pos=ev_pos, kpos=kpos,
                kn_tvtin=kn["TVT_input"].values.astype(float),
                tw_tvt=tw_tvt, tw_gr=tw_gr, tw_fb=tw_fb, ls=ls, ir=ir, tvt_true=tv)

def _smooth(vals, fb, r):
    r = int(r); s = pd.Series(vals, dtype="float32").interpolate(limit_direction="both").fillna(float(fb))
    return s.to_numpy(np.float32) if r <= 0 else s.rolling(2 * r + 1, center=True, min_periods=1).mean().to_numpy(np.float32)

def _grid(tw_tvt, tw_gr, step=0.2):
    a, b = float(tw_tvt.min()), float(tw_tvt.max()); g = np.arange(a, b + step, step)
    return np.interp(g, tw_tvt, tw_gr).astype(np.float64), float(a), float(step)

def make_inputs(c, P, prior):
    tw_gr_pf = _smooth(c["tw_gr"], c["tw_fb"], P["tw_gr_smooth_r"]).astype(np.float64)
    gg, gmin, gst = _grid(c["tw_tvt"], tw_gr_pf)
    gr_sm = _smooth(c["gr_filled"], c["tw_fb"], P["hgr_smooth_r"])
    if len(c["kpos"]) < 20:
        gs = P["gr_sig_def"]
    else:
        resid = gr_sm[c["kpos"]] - np.interp(c["kn_tvtin"], c["tw_tvt"], tw_gr_pf)
        gs = float(np.nanstd(resid)); gs = P["gr_sig_def"] if (not np.isfinite(gs) or gs <= 0) else gs
    gs = float(np.clip(gs * P["gr_sig_mult"], P["gr_sig_min"], max(P["gr_sig_max"], P["gr_sig_min"] + 1e-6)))
    return dict(md=c["md"], z=c["z"], gr=gr_sm[c["ev_pos"]].astype(np.float64),
                gg=gg, gmin=gmin, gst=gst, gs=gs, ls=c["ls"], ir=c["ir"],
                prior=prior, N=int(P["n_particles"]))


# ----------------------- prior(OOF ブレンド)構築 -----------------------
def build_priors(cache):
    """v28 parquet 行順 + OOF から井戸ごとの tvt_prior(eval行, MD昇順)を作る。"""
    import pyarrow.parquet as pq
    tbl = pq.read_table(V28_ART / "data" / "train.parquet",
                        columns=["well", "target", "last_known_tvt", "md_since", "pf_ancc"])
    g   = tbl.column("well").to_pandas().to_numpy()
    y   = tbl.column("target").to_numpy(zero_copy_only=False).astype(np.float64)
    lk  = tbl.column("last_known_tvt").to_numpy(zero_copy_only=False).astype(np.float64)
    mds = tbl.column("md_since").to_numpy(zero_copy_only=False).astype(np.float64)
    pfa = tbl.column("pf_ancc").to_numpy(zero_copy_only=False).astype(np.float64)

    gbm = np.load(V29_ART / "gbm_oof.npy")
    run = None
    for r in NN_RUN_CANDIDATES:
        if (V29B_ART / f"nn_oof_{r}.npz").exists():
            run = r; break
    if run is None:
        raise FileNotFoundError(f"nn_oof_{{{'/'.join(NN_RUN_CANDIDATES)}}}.npz が artifacts_v29b に無い")
    nnz = np.load(V29B_ART / f"nn_oof_{run}.npz")
    nn_ens = np.mean([nnz[k] for k in nnz.files], axis=0)
    res_info = json.loads((V29B_ART / "results.json").read_text())[run]
    w_nn = float(res_info["blend_w"])
    res_b = (1.0 - w_nn) * gbm + w_nn * nn_ens

    pp = (json.loads((V28_ART / "pp_params.json").read_text())
          if (V28_ART / "pp_params.json").exists() else {"alpha": 1.0, "tau": 85, "w_pf": 0.09})
    w_pf = float(pp.get("w_pf", 0.09)); tau = pp.get("tau", 85); alpha = float(pp.get("alpha", 1.0))
    d = res_b * (1 - w_pf) + (pfa - lk) * w_pf
    if tau: d = d * (1.0 - np.exp(-np.maximum(mds, 0.0) / float(tau)))
    tvt_prior = lk + d * alpha

    priors = {}; skipped = []
    for wid in pd.unique(g):
        rows = np.where(g == wid)[0]
        o = rows[np.argsort(mds[rows], kind="stable")]
        c = cache.get(wid)
        if c is None or len(o) != len(c["md"]):
            skipped.append(wid); continue
        priors[wid] = tvt_prior[o]
    # 整合チェック: prior 自身の overall(≈ results.json の blend_pp のはず)
    e = np.concatenate([priors[w] - cache[w]["tvt_true"] for w in priors])
    e = e[np.isfinite(e)]
    prior_rmse = float(np.sqrt(np.mean(e ** 2)))
    print(f"prior: run={run} w_nn={w_nn:.2f} wells={len(priors)} skipped={len(skipped)}"
          f" | prior overall={prior_rmse:.4f} (results.json blend_pp={res_info['blend_pp']:.4f})")
    return priors, dict(run=run, w_nn=w_nn, prior_rmse=prior_rmse, n_wells=len(priors),
                        skipped=len(skipped))


# ----------------------- PF + 固定ラグ smoother(v25 と同一。prior 項のみ追加) -----------------------
def _pf_smoother_core(md, z, gr, valid, grid, glen, vmin, step, gs, ls, ir, prior, prior_sig,
                      P, N, device, gen, dtype=DTYPE):
    B, T = md.shape
    ALPHA = P["mom"]; RN = P["vn"]; PN = P["pn"]; IR = P["init_rate_std"]; IS = P["init_pos_std"]
    RP = P["rp"]; RR = P["rr"]; RESAMP = P["resamp"]; RATE_CLIP = P["rate_clip"]
    GR_POWER = P["gr_power"]; JUMP_PROB = P["jump_prob"]; JUMP_STD = P["jump_std"]; CLIP = P["tvt_clip_margin"]
    PHYS_SIG = P["phys_sig"]; USE_P = P["use_phys"]
    L = int(P.get("smooth_lag", 32)); L = max(0, min(L, T))
    use_prior = prior_sig is not None and math.isfinite(prior_sig)

    def rn(): return torch.randn((B, N), generator=gen, device=device, dtype=dtype)
    pos = ls[:, None] + IS * rn(); rate = ir[:, None] + IR * rn()
    w = torch.full((B, N), 1.0 / N, device=device, dtype=dtype)
    log_lik = torch.zeros(B, device=device, dtype=torch.float64)
    pts_f = torch.zeros((B, T), device=device, dtype=dtype)
    pts_s = torch.zeros((B, T), device=device, dtype=dtype)
    tvt_lo = vmin - CLIP; tvt_hi = vmin + (glen.to(dtype) - 1) * step + CLIP
    glast = torch.gather(grid, 1, (glen - 1).clamp_min(0).unsqueeze(1)); g0 = grid[:, 0:1]
    pm = md[:, 0] - 1.0; arN = torch.arange(N, device=device, dtype=dtype)
    zero = torch.zeros((), device=device, dtype=dtype); dipp = ir[:, None]

    use_sm = L > 0
    if use_sm:
        buf = torch.full((B, L, N), float("nan"), device=device, dtype=dtype)
        ptr = 0

    for i in range(T):
        act = valid[:, i]; cur = md[:, i]; dm = (cur - pm).clamp_min(1.0)
        rate_n = ALPHA * rate + RN * rn()
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
        lk = torch.where(dp < 600.0, torch.exp(-0.5 * dp), zero).clamp_min(1e-300)
        avg = (w * lk).sum(1)
        log_lik = log_lik + torch.where(obs, torch.log(avg.clamp_min(1e-300).double()),
                                        torch.zeros((), device=device, dtype=torch.float64))
        if USE_P:
            dphys = (rate - dipp) / PHYS_SIG; lkp = torch.exp(-0.5 * dphys * dphys).clamp_min(1e-300)
        else:
            lkp = torch.ones_like(w)
        w_new = torch.where(obs[:, None], w * lk * lkp, w * lkp)
        # --- prior アンカー項(v30 追加): GRとは独立に毎有効ステップで乗算。log_lik には混ぜない ---
        if use_prior:
            pri = prior[:, i][:, None]
            okp = am & torch.isfinite(pri)
            dpr = (v - pri) / prior_sig
            lk_pr = torch.where(okp, torch.exp(-0.5 * dpr * dpr).clamp_min(1e-30), torch.ones_like(w))
            w_new = w_new * lk_pr
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
            if use_sm:
                buf_g = torch.gather(buf, 2, idx[:, None, :].expand(B, L, N))
                buf = torch.where(need[:, None, None], buf_g, buf)

        pts_f[:, i] = (w * (pos - zi)).sum(1)
        if use_sm:
            if i >= L:
                old = buf[:, ptr, :]
                pts_s[:, i - L] = (w * old).sum(1) - z[:, i - L]
            buf[:, ptr, :] = pos
            ptr = (ptr + 1) % L
        pm = torch.where(act, cur, pm)

    if use_sm:
        for j in range(max(0, T - L), T):
            pts_s[:, j] = (w * buf[:, j % L, :]).sum(1) - z[:, j]
    else:
        pts_s = pts_f
    return pts_f, pts_s, log_lik


def _pad(inps, device, dtype=DTYPE):
    W = len(inps); Tmax = max(len(x["md"]) for x in inps); Gmax = max(len(x["gg"]) for x in inps)
    md = torch.zeros((W, Tmax), dtype=dtype); z = torch.zeros((W, Tmax), dtype=dtype)
    gr = torch.full((W, Tmax), float("nan"), dtype=dtype); valid = torch.zeros((W, Tmax), dtype=torch.bool)
    grid = torch.zeros((W, Gmax), dtype=dtype); glen = torch.zeros(W, dtype=torch.long)
    vmin = torch.zeros(W, dtype=dtype); step = torch.zeros(W, dtype=dtype); gs = torch.zeros(W, dtype=dtype)
    ls = torch.zeros(W, dtype=dtype); ir = torch.zeros(W, dtype=dtype)
    pri = torch.full((W, Tmax), float("nan"), dtype=dtype)
    for b, x in enumerate(inps):
        Tn = len(x["md"]); G = len(x["gg"])
        md[b, :Tn] = torch.from_numpy(x["md"].astype("float32"))
        if Tn < Tmax: md[b, Tn:] = md[b, Tn - 1]
        z[b, :Tn] = torch.from_numpy(x["z"].astype("float32"))
        gr[b, :Tn] = torch.from_numpy(x["gr"].astype("float32")); valid[b, :Tn] = True
        grid[b, :G] = torch.from_numpy(x["gg"].astype("float32"))
        if G < Gmax: grid[b, G:] = grid[b, G - 1]
        glen[b] = G; vmin[b] = x["gmin"]; step[b] = x["gst"]; gs[b] = x["gs"]; ls[b] = x["ls"]; ir[b] = x["ir"]
        if x["prior"] is not None:
            pri[b, :Tn] = torch.from_numpy(x["prior"].astype("float32"))
    to = lambda tt: tt.to(device)
    return dict(md=to(md), z=to(z), gr=to(gr), valid=to(valid), grid=to(grid), glen=to(glen),
                vmin=to(vmin), step=to(step), gs=to(gs), ls=to(ls), ir=to(ir), prior=to(pri))


def run_smoother_ens(inps, P, prior_sig, device, seed, n_seeds):
    N = int(P["n_particles"]); S = n_seeds; W = len(inps); pad = _pad(inps, device)
    rep = lambda tt: tt.repeat_interleave(S, 0); gen = torch.Generator(device=device); gen.manual_seed(seed)
    pf, ps, ll = _pf_smoother_core(
        rep(pad["md"]), rep(pad["z"]), rep(pad["gr"]), rep(pad["valid"]), rep(pad["grid"]),
        pad["glen"].repeat_interleave(S), pad["vmin"].repeat_interleave(S), pad["step"].repeat_interleave(S),
        pad["gs"].repeat_interleave(S), pad["ls"].repeat_interleave(S), pad["ir"].repeat_interleave(S),
        rep(pad["prior"]), prior_sig, P, N, device, gen)
    pf = pf.view(W, S, -1).double().cpu().numpy(); ps = ps.view(W, S, -1).double().cpu().numpy()
    ll = ll.view(W, S).cpu().numpy()
    out_f = []; out_s = []
    for w in range(W):
        T = len(inps[w]["md"]); llw = ll[w]
        if np.isfinite(llw).any():
            mx = np.nanmax(llw[np.isfinite(llw)])
            lk = np.where(np.isfinite(llw), llw - mx, -np.inf)
            ww = np.exp(lk / max(P["likelihood_scale"], 1e-6)); s = ww.sum()
            ww = ww / s if s > 1e-300 else np.full(S, 1.0 / S)
        else:
            ww = np.full(S, 1.0 / S)
        out_f.append((ww[:, None] * pf[w, :, :T]).sum(0))
        out_s.append((ww[:, None] * ps[w, :, :T]).sum(0))
    return out_f, out_s


# ----------------------- 評価 -----------------------
def overall(err_list):
    e = np.concatenate(err_list); e = e[np.isfinite(e)]
    return float(np.sqrt(np.mean(e ** 2))) if e.size else 1e3

def wells_over(errs, thr=10.0):
    return int(sum(1 for e in errs if np.sqrt(np.nanmean(e ** 2)) > thr))


def main():
    t00 = time.time()
    wells = list_wells()
    print(f"wells: {len(wells)} | device={DEVICE} | n_seed={N_SEED} | lag={SMOOTH_LAG}")
    cache = {}
    for wid in wells:
        c = build_cache(wid)
        if c is not None and c["tvt_true"] is not None:
            cache[wid] = c
    print(f"cache: {len(cache)} wells ({time.time()-t00:.0f}s)")

    priors, prior_meta = build_priors(cache)
    wl = [w for w in wells if w in priors]          # prior が作れた井戸のみで比較(全構成同一集合)
    tvt_true = {w: cache[w]["tvt_true"] for w in wl}

    results = json.loads(OUT_PATH.read_text()) if OUT_PATH.exists() else {}
    results["_meta"] = dict(prior=prior_meta, n_seed=N_SEED, smooth_lag=SMOOTH_LAG,
                            wells=len(wl), r0="v19 32seeds round0")

    for sig in PRIOR_SIGS:
        key = "none" if sig is None else f"{sig:g}"
        npz_fp = OUT_DIR / f"preds_sig{key}.npz"
        if key in results and npz_fp.exists():
            print(f"[skip] sig={key} (済)"); continue
        t0 = time.time()
        P = {**R0_PARAMS, "smooth_lag": SMOOTH_LAG}
        pred_s = {}; pred_f = {}
        for c0 in range(0, len(wl), CHUNK):
            sub = wl[c0:c0 + CHUNK]
            inps = [make_inputs(cache[w], P, priors[w] if sig is not None else None) for w in sub]
            pf, ps = run_smoother_ens(inps, P, sig, DEVICE, SEED, N_SEED)
            for k, w in enumerate(sub):
                pred_f[w] = pf[k]; pred_s[w] = ps[k]
            print(f"  sig={key} {min(c0+CHUNK,len(wl))}/{len(wl)} ({time.time()-t0:.0f}s)", flush=True)
        es = [pred_s[w] - tvt_true[w] for w in wl]
        ef = [pred_f[w] - tvt_true[w] for w in wl]
        # β走査: out = (1-β)·prior + β·smoother
        betas = {}
        for b in BETAS:
            eb = [(1 - b) * priors[w] + b * pred_s[w] - tvt_true[w] for w in wl]
            betas[f"{b:.1f}"] = round(overall(eb), 4)
        bb = min(betas, key=betas.get)
        results[key] = dict(rmse_smoothed=round(overall(es), 4), rmse_forward=round(overall(ef), 4),
                            over10_smoothed=wells_over(es), best_beta=float(bb),
                            rmse_best_beta=betas[bb], betas=betas, sec=round(time.time() - t0))
        np.savez_compressed(npz_fp, wells=np.array(wl),
                            **{f"s_{w}": pred_s[w] for w in wl})
        OUT_PATH.write_text(json.dumps(results, indent=1))
        print(f"[sig={key}] smoothed={results[key]['rmse_smoothed']:.4f} "
              f"fwd={results[key]['rmse_forward']:.4f} >10={results[key]['over10_smoothed']} "
              f"| bestβ={bb}:{betas[bb]:.4f} | prior={prior_meta['prior_rmse']:.4f} "
              f"({results[key]['sec']}s)", flush=True)

    print("\n=== まとめ(基準: prior overall = 現ベストblend) ===")
    print(f"prior({prior_meta['run']}, w_nn={prior_meta['w_nn']:.2f}) = {prior_meta['prior_rmse']:.4f}")
    for sig in PRIOR_SIGS:
        key = "none" if sig is None else f"{sig:g}"
        r = results.get(key)
        if r:
            print(f"sig={key:>5s}: smoothed={r['rmse_smoothed']:.4f} bestβ={r['best_beta']}"
                  f"→{r['rmse_best_beta']:.4f} >10={r['over10_smoothed']}")
    print("判定: どの σ でも bestβ が 0.0(=prior単独)なら閉ループは prior に勝てず撤退。")


if __name__ == "__main__":
    main()
