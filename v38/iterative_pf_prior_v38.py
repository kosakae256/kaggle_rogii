"""
v38p priorの「入れ方」を変えた物理PF(v38コアの注入方式バリエーション)
=====================================================================
iterative_pf_boost_v38.py(row=毎行重み乗算, mult=温度補正)に対し、
**priorをどこに注入するか**を変えた4方式を V38_PMODE で切替:

  row   : 従来と同一(毎行重み乗算, σ×MULT)。本ファイルの基準・検算用。
  gate  : ★GR曖昧ゲート錨。行ごとに GR尤度の判別度 amb = mean(lk)/max(lk) を測り、
          lka^amb を掛ける。GRが判別的(尤度が尖る)→amb→0=錨消滅 /
          GRが曖昧・欠損(尤度フラット)→amb→1=錨フル。
          = 「GRが暴走したときだけ引き戻す」を行単位で自動化。MULTは小さめ(5〜20)推奨。
  trob  : Student-t錨(重い裾, 自由度 V38_TNU)。ガウスの二乗ペナルティは
          structが外れる坑井で引きずり続ける → t裾は「大きく外れたらGRに任せる」。
  guide : ダイナミクス注入(OU緩和)。重みには一切触れず、伝播時に
          pos += β·(anc − tvt),  β = (dm/GUIDE_L) · S0²/(S0²+σ²)  (≤0.25/step)
          σ較正で信頼低下すると自動で弱まる。重み枯渇ゼロ・証拠水増しゼロ。
  jump  : ジャンプ再生成注入。jump の飛び先を「現在地+JUMP_STDノイズ」から
          「prior分布 N(anc, σ) からのドロー」に変更。裏バンドルに嵌った粒子群を
          prior枝へテレポートさせ、生存判定はGR尤度に委ねる(flip回収の王道)。
          jump_prob(Optuna探索対象)が再生成率。prior欠損行は従来ノイズジャンプ。

v38原則の維持: 錨の強さはOptunaで振らない。σは較正済LOO誤差、外側定数は
MULT / TNU / GUIDE_L のみ(環境変数)。suggest_params は v19/v38 と完全同一。

実行例:
  $env:V38_PMODE="gate";  $env:V38_MULT="10"; python v38/tune_prior_v38.py
  $env:V38_PMODE="trob";  $env:V38_MULT="10"; $env:V38_TNU="3";   python v38/tune_prior_v38.py
  $env:V38_PMODE="guide"; $env:V38_GLEN="1500";                    python v38/tune_prior_v38.py
  $env:V38_PMODE="jump";                                           python v38/tune_prior_v38.py
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
OUT_DIR   = PROJECT / "v38" / "artifacts_v38"; OUT_DIR.mkdir(parents=True, exist_ok=True)
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE     = torch.float32

USE_PHYS    = False       # dip率prior(v19で死んでいる)。既定OFF。
USE_ANCHOR  = True        # False で全注入OFF(=v19と等価。比較用)。
PRIOR_NAME  = os.environ.get("V38_PRIOR", "struct")
PMODE       = os.environ.get("V38_PMODE", "gate")            # row|gate|trob|guide|jump
ANCHOR_MULT = float(os.environ.get("V38_MULT", "10"))        # row/gate/trob の σ倍率
TNU         = float(os.environ.get("V38_TNU", "3.0"))        # trob: Student-t 自由度
GUIDE_L     = float(os.environ.get("V38_GLEN", "1500"))      # guide: 緩和長(ft MD)
S0_SIG      = float(os.environ.get("V38_S0", "8.0"))         # guide: 基準σ(較正σの中央値≈8ft)
assert PMODE in ("row", "gate", "trob", "guide", "jump"), PMODE
STRUCT_PKL  = (OUT_DIR / "struct_prior_train.pkl" if PRIOR_NAME == "struct"
               else OUT_DIR / "priors" / f"prior_{PRIOR_NAME}_train.pkl")
print(f"[v38p] PRIOR={PRIOR_NAME}  PMODE={PMODE}  MULT={ANCHOR_MULT:g}  TNU={TNU:g}"
      f"  GLEN={GUIDE_L:g}  S0={S0_SIG:g}  pkl={STRUCT_PKL.name}")

SEED0        = 99999
STORAGE      = f"sqlite:///{(OUT_DIR / 'optuna.db').as_posix()}"

optuna.logging.set_verbosity(optuna.logging.WARNING)
CACHE = {}

# 構造prior(well -> {"tvt":arr,"sig":arr}, ev行順)を読み込む
_STRUCT = pickle.loads(STRUCT_PKL.read_bytes()) if STRUCT_PKL.exists() else {}
if not _STRUCT:
    print(f"WARNING: {STRUCT_PKL} が無い。struct_prior_v38.py を先に実行。錨は全wellでOFFになる。")


# ----------------------- 物理PF(v19 + prior注入バリエーション) -----------------------
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
    nev = len(ev)
    sp = _STRUCT.get(wid)
    if sp is not None and len(sp["tvt"]) == nev:
        anc = np.asarray(sp["tvt"], np.float64); ancs = np.asarray(sp["sig"], np.float64)
    else:
        anc = np.zeros(nev, np.float64); ancs = np.full(nev, 1e9, np.float64)   # prior欠損→この井戸は錨不能
    return dict(md=ev["MD"].values.astype(np.float64), z=ev["Z"].values.astype(np.float64),
                gr_filled=gr_filled, ev_pos=ev_pos, kpos=kpos,
                kn_tvtin=kn["TVT_input"].values.astype(float),
                tw_tvt=tw_tvt, tw_gr=tw_gr, tw_fb=tw_fb, ls=ls, ir=ir, tvt_true=tv,
                anc=anc, ancs=ancs)

def _smooth(vals, fb, r):
    r = int(r); s = pd.Series(vals, dtype="float32").interpolate(limit_direction="both").fillna(float(fb))
    return s.to_numpy(np.float32) if r <= 0 else s.rolling(2 * r + 1, center=True, min_periods=1).mean().to_numpy(np.float32)

def _grid(tw_tvt, tw_gr, step=0.2):
    a, b = float(tw_tvt.min()), float(tw_tvt.max()); g = np.arange(a, b + step, step)
    return np.interp(g, tw_tvt, tw_gr).astype(np.float64), float(a), float(step)

def make_inputs(c, P):
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
                gg=gg, gmin=gmin, gst=gst, gs=gs, ls=c["ls"], ir=c["ir"], N=int(P["n_particles"]),
                anc=c["anc"], ancs=c["ancs"])

def _pf_core(md, z, gr, valid, grid, glen, vmin, step, gs, ls, ir, anc, ancs, P, N, device, gen, dtype=DTYPE):
    B, T = md.shape
    ALPHA = P["mom"]; RN = P["vn"]; PN = P["pn"]; IR = P["init_rate_std"]; IS = P["init_pos_std"]
    RP = P["rp"]; RR = P["rr"]; RESAMP = P["resamp"]; RATE_CLIP = P["rate_clip"]
    GR_POWER = P["gr_power"]; JUMP_PROB = P["jump_prob"]; JUMP_STD = P["jump_std"]; CLIP = P["tvt_clip_margin"]
    PHYS_SIG = P["phys_sig"]; USE_P = P["use_phys"]; AMUL = float(ANCHOR_MULT)
    MODE = PMODE if USE_ANCHOR else "off"                     # ★注入方式(OFF=v19相当)
    def rn(): return torch.randn((B, N), generator=gen, device=device, dtype=dtype)
    pos = ls[:, None] + IS * rn(); rate = ir[:, None] + IR * rn()
    w = torch.full((B, N), 1.0 / N, device=device, dtype=dtype)
    log_lik = torch.zeros(B, device=device, dtype=torch.float64)
    pts = torch.zeros((B, T), device=device, dtype=dtype)
    tvt_lo = vmin - CLIP; tvt_hi = vmin + (glen.to(dtype) - 1) * step + CLIP
    glast = torch.gather(grid, 1, (glen - 1).clamp_min(0).unsqueeze(1)); g0 = grid[:, 0:1]
    pm = md[:, 0] - 1.0; arN = torch.arange(N, device=device, dtype=dtype)
    zero = torch.zeros((), device=device, dtype=dtype); dipp = ir[:, None]
    S02 = S0_SIG * S0_SIG
    for i in range(T):
        act = valid[:, i]; cur = md[:, i]; dm = (cur - pm).clamp_min(1.0)
        zi = z[:, i][:, None]
        anc_i = anc[:, i][:, None]; sig_i = ancs[:, i][:, None]     # prior(TVT/σ, 較正済)
        rate_n = ALPHA * rate + RN * rn()
        if RATE_CLIP > 0: rate_n = rate_n.clamp(-RATE_CLIP, RATE_CLIP)
        pos_n = pos + rate_n * dm[:, None] + PN * rn()
        if MODE == "jump":
            # ★ジャンプ再生成注入: 飛び先= prior分布 N(anc, σ) のドロー(prior欠損行は従来ノイズ)
            if JUMP_PROB > 0:
                jm = torch.rand((B, N), generator=gen, device=device, dtype=dtype) < JUMP_PROB
                okp = sig_i < 1e8
                tgt = anc_i + sig_i * rn() + zi                       # pos空間(TVT+Z)へ
                pos_prior = torch.where(okp, tgt, pos_n + JUMP_STD * rn())
                pos_n = torch.where(jm, pos_prior, pos_n)
        else:
            if JUMP_PROB > 0 and JUMP_STD > 0:
                jm = torch.rand((B, N), generator=gen, device=device, dtype=dtype) < JUMP_PROB
                pos_n = pos_n + torch.where(jm, JUMP_STD * rn(), zero)
        if MODE == "guide":
            # ★ダイナミクス注入(OU緩和): β=(dm/L)·S0²/(S0²+σ²)。σ大(信頼低/欠損)→β→0
            beta = ((dm[:, None] / GUIDE_L) * (S02 / (S02 + sig_i * sig_i))).clamp(0.0, 0.25)
            pos_n = pos_n + beta * (anc_i - (pos_n - zi))
        tvt = (pos_n - zi).clamp(tvt_lo[:, None], tvt_hi[:, None]); pos_n = tvt + zi
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
        if MODE == "row":                            # 従来: 毎行重み乗算(mult=温度補正)
            asig = (sig_i * AMUL).clamp_min(1e-6)
            da = (v - anc_i) / asig
            lka = torch.where(da * da < 1200.0, torch.exp(-0.5 * da * da), zero).clamp_min(1e-300)
        elif MODE == "trob":                         # ★Student-t錨(重い裾): 大外れはGRに任せる
            asig = (sig_i * AMUL).clamp_min(1e-6)
            da2 = ((v - anc_i) / asig) ** 2
            lka = (1.0 + da2 / TNU).pow(-0.5 * (TNU + 1.0)).clamp_min(1e-300)
        elif MODE == "gate":                         # ★GR曖昧ゲート錨: 判別的な行では錨が消える
            asig = (sig_i * AMUL).clamp_min(1e-6)
            da = (v - anc_i) / asig
            lka0 = torch.where(da * da < 1200.0, torch.exp(-0.5 * da * da), zero).clamp_min(1e-300)
            lmax = lk.max(1, keepdim=True).values
            amb = (lk.mean(1, keepdim=True) / lmax.clamp_min(1e-300)).clamp(0.0, 1.0)
            # 尤度フラット/GR欠損 → amb=1(錨フル)。尤度が尖る=GRが位置を判別 → amb→0(錨消滅)
            lka = lka0.pow(amb)
        else:                                        # guide / jump / off: 重み錨なし
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


# ----------------------- Optuna 探索空間(v19/v38と完全同一。prior強度は足さない) -----------------------
def suggest_params(trial):
    return dict(
        n_particles      = trial.suggest_categorical("n_particles", [400, 500, 600, 800]),
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
        tw_gr_smooth_r   = trial.suggest_categorical("tw_gr_smooth_r", [0]),
        jump_prob        = trial.suggest_categorical("jump_prob", [0.0, 0.002, 0.005, 0.01, 0.02]),
        jump_std         = trial.suggest_categorical("jump_std", [5.0, 10.0, 15.0, 20.0]),
        phys_sig         = trial.suggest_float("phys_sig", 0.003, 5.0, log=True),
        use_phys         = USE_PHYS,
    )

def full_params(best_params):
    return {**best_params, "use_phys": USE_PHYS}
