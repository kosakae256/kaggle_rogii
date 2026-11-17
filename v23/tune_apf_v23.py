"""
v23 Auxiliary Particle Filter (APF) カテゴリ別チューナー(6分割・各カテゴリ独立・自動・再開可能)
================================================================================================
やること:
  v22 と同じ枠組み(selector の 6カテゴリ code 0..5 = 評価長2bin × z分散3bin ごとに well を分け、
  各カテゴリで Optuna により overall RMSE 最小の PFパラメータを探索)で、
  **エンジンだけ通常の物理PF → Auxiliary Particle Filter (APF) に差し替えた**もの。

APF とは(Pitt & Shephard 1999):
  通常の bootstrap PF は「伝播 → 尤度で重み付け → (ESS低下時)リサンプリング」。
  APF は **リサンプリングの前に「次の観測を先読み」**する:
    1. 各粒子の決定論的予測点 μ^i = E[x_t | x_{t-1}^i]  (= pos + mom*rate*dMD、過程ノイズ無し)
    2. 一段目重み g^i ∝ w^i · p(y_t | μ^i)        ← 観測を覗いた重み
    3. g で祖先 index を系統リサンプリング         ← 「次に尤度が高くなる粒子」を先に増やす
    4. 祖先からノイズ伝播  x_t^i ~ p(x_t | x_{t-1}^{a_i})
    5. 二段目重み w_t^i ∝ p(y_t | x_t^i) / p(y_t | μ^{a_i})   ← 先読み分を補正(不偏)
  → 尤度が鋭い(GRがよく効く)坑井で**粒子枯渇を抑え**、少ない粒子で事後を保つのが狙い。
  読み出しは通常PFと同じ事後平均 (w·pos).sum。シード尤度加重アンサンブルもそのまま。

パラメータ空間は v22(=v19)と完全に同一にしてある:
  - 比較・ウォームスタート(v22 の best をそのまま APF に入れて出発点にできる)のため。
  - ただし **`resamp` は APF では未使用**(APFは毎観測ステップで先読みリサンプリングする)。
    `rp`/`rr` は「リサンプリングした祖先へのジッタ(正則化)」として伝播前に適用する。

要件(v22 と同じ):
  - 残差ブースティング無し。各カテゴリ独立。
  - 途中で止めても再開可: Optuna は SQLite(trial 単位)、カテゴリ進捗は state_apf.json。
  - 1回起動すれば 6カテゴリを順に自動で回る。

出力: v23/artifacts_v23/per_category_apf.json  (code -> 最良パラメータ)
実行: rogii_claude/.venv/Scripts/python.exe v23/tune_apf_v23.py
※ build_cache/well_code/カテゴリ定義/IO は v22 と同一。エンジン(_apf_core)だけ APF。
※ grid チューナー(tune_grid_filter_v23.py)とは別の optuna_apf.db / state_apf.json を使うので衝突しない。
"""

import os, time, json, glob, math, warnings
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
OUT_DIR   = PROJECT / "v23" / "artifacts_v23"; OUT_DIR.mkdir(parents=True, exist_ok=True)
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE     = torch.float32

SEARCH_SEEDS = 32         # 探索(Optuna objective)用シード数
EVAL_SEEDS   = 32         # カテゴリ best の記録用(本番一致)
SEED0        = 0
SEARCH_CHUNK = 600        # 探索GPUバッチwell数(APFは一時テンソルが通常PFより多い→v22の773より控えめ)
EVAL_CHUNK   = 350        # 評価分割
N_WELLS      = None       # None=全well, 整数で先頭N(デバッグ)

N_TRIALS     = 250        # 1カテゴリあたりの Optuna 探索回数
USE_PHYS     = True

# ---- カテゴリ定義(7.191 selector と一致, v22 と同一)----
N_EVAL_THRESHOLD  = 4840.0
Z_SPAN_THRESHOLDS = (136.73000000000016, 185.5133333333342)
N_CATS = 6

STORAGE      = f"sqlite:///{(OUT_DIR / 'optuna_apf.db').as_posix()}"
STATE_PATH   = OUT_DIR / "state_apf.json"
PER_CAT_PATH = OUT_DIR / "per_category_apf.json"
CUR_PATH     = OUT_DIR / "current_cat_apf.json"
# ===================================================================

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ----------------------- データ(v22と同一) -----------------------
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
    z_ev = ev["Z"].values.astype(np.float64)
    n_eval = int(len(z_ev))
    z_span = float(np.nanmax(z_ev) - np.nanmin(z_ev)) if len(z_ev) else 0.0
    return dict(md=ev["MD"].values.astype(np.float64), z=z_ev,
                gr_filled=gr_filled, ev_pos=ev_pos, kpos=kpos,
                kn_tvtin=kn["TVT_input"].values.astype(float),
                tw_tvt=tw_tvt, tw_gr=tw_gr, tw_fb=tw_fb, ls=ls, ir=ir, tvt_true=tv,
                n_eval=n_eval, z_span=z_span)

def well_code(c):
    n_bin = int(c["n_eval"] > N_EVAL_THRESHOLD)
    z_bin = int(np.searchsorted(Z_SPAN_THRESHOLDS, c["z_span"], side="right"))
    return n_bin + 2 * z_bin

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
                gg=gg, gmin=gmin, gst=gst, gs=gs, ls=c["ls"], ir=c["ir"], N=int(P["n_particles"]))


# ----------------------- Auxiliary Particle Filter GPU -----------------------
def _apf_core(md, z, gr, valid, grid, glen, vmin, step, gs, ls, ir, P, N, device, gen, dtype=DTYPE):
    """
    Auxiliary PF。状態 = (pos=geo_z=TVT+Z, rate=dip)。観測 = GR。
    各観測ステップで:
      予測点μで先読み → 一段目重み g=w·p(y|μ) で祖先を系統リサンプリング
      → 祖先(+rp/rrジッタ)からノイズ伝播 → 二段目重み w∝p(y|x)/p(y|μ_anc)·phys。
    観測の無い act ステップは通常通り「ノイズ伝播のみ(リサンプリング無し)+phys重み」。
    返り値: pts(各ステップの事後平均 TVT), log_lik(APF不偏予測尤度の対数和; emissionのみ=v19と整合)。
    """
    B, T = md.shape
    ALPHA = P["mom"]; RN = P["vn"]; PN = P["pn"]; IR = P["init_rate_std"]; IS = P["init_pos_std"]
    RP = P["rp"]; RR = P["rr"]; RATE_CLIP = P["rate_clip"]
    GR_POWER = P["gr_power"]; JUMP_PROB = P["jump_prob"]; JUMP_STD = P["jump_std"]; CLIP = P["tvt_clip_margin"]
    PHYS_SIG = P["phys_sig"]; USE_P = P["use_phys"]
    LA = float(P.get("apf_lookahead", 1.0))   # 先読み強度 β: <1=穏やか, 1=標準APF, >1=攻撃的先読み
    MU_DRIFT = float(P.get("apf_mu_drift", 1.0))   # 先読み予測点の外挿量 κ: <1=控えめ, 1=素直に外挿, >1=強め外挿
    # 注: P["resamp"] は APF では未使用(毎観測ステップで先読みリサンプリングするため)。

    def rn(): return torch.randn((B, N), generator=gen, device=device, dtype=dtype)
    zero = torch.zeros((), device=device, dtype=dtype)
    tvt_lo = vmin - CLIP; tvt_hi = vmin + (glen.to(dtype) - 1) * step + CLIP
    glast = torch.gather(grid, 1, (glen - 1).clamp_min(0).unsqueeze(1)); g0 = grid[:, 0:1]
    arN = torch.arange(N, device=device, dtype=dtype)
    dipp = ir[:, None]

    def emission(pos_val, zi, gri):
        # pos_val:(B,N) geo位置 → TVT=pos-zi を typewellグリッドで線形補間し GR尤度 lk:(B,N)
        v = pos_val - zi; ii = (v - vmin[:, None]) / step[:, None]; i0 = torch.floor(ii).long()
        below = i0 < 0; above = i0 >= (glen - 1)[:, None]
        i0c = i0.clamp_min(0); i0c = torch.minimum(i0c, (glen - 2).clamp_min(0)[:, None])
        t = ii - i0c.to(dtype); gA = torch.gather(grid, 1, i0c); gB = torch.gather(grid, 1, i0c + 1)
        eg = gA * (1 - t) + gB * t; eg = torch.where(below, g0, eg); eg = torch.where(above, glast, eg)
        d = ((gri[:, None] - eg) / gs[:, None]).abs(); dp = d ** GR_POWER
        return torch.where(dp < 600.0, torch.exp(-0.5 * dp), zero).clamp_min(1e-300)

    def clip_tvt(pos_n, zi):
        tvt = (pos_n - zi).clamp(tvt_lo[:, None], tvt_hi[:, None]); return tvt + zi

    pos = ls[:, None] + IS * rn(); rate = ir[:, None] + IR * rn()
    w = torch.full((B, N), 1.0 / N, device=device, dtype=dtype)
    log_lik = torch.zeros(B, device=device, dtype=torch.float64)
    pts = torch.zeros((B, T), device=device, dtype=dtype)
    pm = md[:, 0] - 1.0

    for i in range(T):
        act = valid[:, i]; cur = md[:, i]; dm = (cur - pm).clamp_min(1.0)
        zi = z[:, i][:, None]; gri = gr[:, i]; obs = act & ~torch.isnan(gri)
        obsm = obs[:, None]; am = act[:, None]

        # ---- (A) 通常伝播パス(リサンプリング無し): 観測の無い act ステップ用 ----
        rate_pl = ALPHA * rate + RN * rn()
        if RATE_CLIP > 0: rate_pl = rate_pl.clamp(-RATE_CLIP, RATE_CLIP)
        pos_pl = pos + rate_pl * dm[:, None] + PN * rn()
        if JUMP_PROB > 0 and JUMP_STD > 0:
            jm = torch.rand((B, N), generator=gen, device=device, dtype=dtype) < JUMP_PROB
            pos_pl = pos_pl + torch.where(jm, JUMP_STD * rn(), zero)
        pos_pl = clip_tvt(pos_pl, zi)
        if USE_P:
            dphys_pl = (rate_pl - dipp) / PHYS_SIG; lkp_pl = torch.exp(-0.5 * dphys_pl * dphys_pl).clamp_min(1e-300)
        else:
            lkp_pl = torch.ones_like(w)
        w_pl = w * lkp_pl   # 観測なし → 尤度はかけず phys のみ(重みは保持)

        # ---- (B) APF パス(観測あり): 先読み → 祖先リサンプリング → 伝播 → 二段目重み ----
        # 1) 決定論的予測点 μ(過程ノイズ・ジャンプ無し)。MU_DRIFT で外挿量をスケール
        rate_pred = ALPHA * rate
        if RATE_CLIP > 0: rate_pred = rate_pred.clamp(-RATE_CLIP, RATE_CLIP)
        pos_pred = clip_tvt(pos + MU_DRIFT * rate_pred * dm[:, None], zi)
        lk_mu = emission(pos_pred, zi, gri)                 # p(y_t | μ^i)
        # 2) 一段目重み g = w·p(y|μ)^β(β=先読み強度)、正規化前の総和 gsum は APF予測尤度の第1因子
        lk_mu_la = lk_mu if LA == 1.0 else lk_mu.pow(LA)
        g = w * lk_mu_la
        gsum = g.sum(1, keepdim=True)                        # (B,1)
        g = g / gsum.clamp_min(1e-300)
        # 3) g で系統リサンプリング(祖先 index)
        cumg = torch.cumsum(g, 1)
        u0 = torch.rand((B, 1), generator=gen, device=device, dtype=dtype) * (1.0 / N)
        u = u0 + arN[None, :] / N
        idx = torch.searchsorted(cumg, u, right=False).clamp(0, N - 1)
        pos_a = torch.gather(pos, 1, idx); rate_a = torch.gather(rate, 1, idx)
        lk_mu_a = torch.gather(lk_mu_la, 1, idx).clamp_min(1e-12)   # 祖先の先読み尤度 p(y|μ^{a})^β
        # rp/rr ジッタ(正則化)を祖先に付与
        pos_a = pos_a + RP * rn(); rate_a = rate_a + RR * rn()
        if RATE_CLIP > 0: rate_a = rate_a.clamp(-RATE_CLIP, RATE_CLIP)
        # 4) 祖先からノイズ伝播
        rate_ap = ALPHA * rate_a + RN * rn()
        if RATE_CLIP > 0: rate_ap = rate_ap.clamp(-RATE_CLIP, RATE_CLIP)
        pos_ap = pos_a + rate_ap * dm[:, None] + PN * rn()
        if JUMP_PROB > 0 and JUMP_STD > 0:
            jm2 = torch.rand((B, N), generator=gen, device=device, dtype=dtype) < JUMP_PROB
            pos_ap = pos_ap + torch.where(jm2, JUMP_STD * rn(), zero)
        pos_ap = clip_tvt(pos_ap, zi)
        # 5) 二段目重み w ∝ p(y|x)/p(y|μ_anc) · phys
        lk_ap = emission(pos_ap, zi, gri)                    # p(y_t | x_t^i)
        w_e = (lk_ap / lk_mu_a).clamp(max=1e8)               # emission 二段目(不偏補正)
        if USE_P:
            dphys_ap = (rate_ap - dipp) / PHYS_SIG; lkp_ap = torch.exp(-0.5 * dphys_ap * dphys_ap).clamp_min(1e-300)
        else:
            lkp_ap = torch.ones_like(w)
        w_apf = (w_e * lkp_ap).clamp_min(1e-300)             # 正規化前の APF 重み(リサンプ後なので一様起点)

        # ---- 予測尤度(seed アンサンブル用, emission のみ=v19整合): gsum × mean(w_e) ----
        pl_pred = gsum[:, 0] * w_e.mean(1)
        log_lik = log_lik + torch.where(obs, torch.log(pl_pred.clamp_min(1e-300).double()),
                                        torch.zeros((), device=device, dtype=torch.float64))

        # ---- パス選択(obsなら APF、obsなしなら通常伝播)。act の行だけ更新 ----
        pos_new = torch.where(obsm, pos_ap, pos_pl)
        rate_new = torch.where(obsm, rate_ap, rate_pl)
        w_un = torch.where(obsm, w_apf, w_pl)
        ws = w_un.sum(1, keepdim=True)
        w_norm = torch.where(ws > 0, w_un / ws.clamp_min(1e-300), torch.full_like(w_un, 1.0 / N))
        pos = torch.where(am, pos_new, pos)
        rate = torch.where(am, rate_new, rate)
        w = torch.where(am, w_norm, w)

        pts[:, i] = (w * (pos - zi)).sum(1)
        pm = torch.where(act, cur, pm)
    return pts, log_lik


def _pad(inps, device, dtype=DTYPE):
    W = len(inps); Tmax = max(len(x["md"]) for x in inps); Gmax = max(len(x["gg"]) for x in inps)
    md = torch.zeros((W, Tmax), dtype=dtype); z = torch.zeros((W, Tmax), dtype=dtype)
    gr = torch.full((W, Tmax), float("nan"), dtype=dtype); valid = torch.zeros((W, Tmax), dtype=torch.bool)
    grid = torch.zeros((W, Gmax), dtype=dtype); glen = torch.zeros(W, dtype=torch.long)
    vmin = torch.zeros(W, dtype=dtype); step = torch.zeros(W, dtype=dtype); gs = torch.zeros(W, dtype=dtype)
    ls = torch.zeros(W, dtype=dtype); ir = torch.zeros(W, dtype=dtype)
    for b, x in enumerate(inps):
        Tn = len(x["md"]); G = len(x["gg"])
        md[b, :Tn] = torch.from_numpy(x["md"].astype("float32"))
        if Tn < Tmax: md[b, Tn:] = md[b, Tn - 1]
        z[b, :Tn] = torch.from_numpy(x["z"].astype("float32"))
        gr[b, :Tn] = torch.from_numpy(x["gr"].astype("float32")); valid[b, :Tn] = True
        grid[b, :G] = torch.from_numpy(x["gg"].astype("float32"))
        if G < Gmax: grid[b, G:] = grid[b, G - 1]
        glen[b] = G; vmin[b] = x["gmin"]; step[b] = x["gst"]; gs[b] = x["gs"]; ls[b] = x["ls"]; ir[b] = x["ir"]
    to = lambda tt: tt.to(device)
    return dict(md=to(md), z=to(z), gr=to(gr), valid=to(valid), grid=to(grid), glen=to(glen),
                vmin=to(vmin), step=to(step), gs=to(gs), ls=to(ls), ir=to(ir))

def run_forward_ens(inps, P, device, seed, n_seeds):
    N = int(P["n_particles"]); S = n_seeds; W = len(inps); pad = _pad(inps, device)
    rep = lambda tt: tt.repeat_interleave(S, 0); gen = torch.Generator(device=device); gen.manual_seed(seed)
    pts, ll = _apf_core(rep(pad["md"]), rep(pad["z"]), rep(pad["gr"]), rep(pad["valid"]), rep(pad["grid"]),
                        pad["glen"].repeat_interleave(S), pad["vmin"].repeat_interleave(S), pad["step"].repeat_interleave(S),
                        pad["gs"].repeat_interleave(S), pad["ls"].repeat_interleave(S), pad["ir"].repeat_interleave(S),
                        P, N, device, gen)
    pts = pts.view(W, S, -1).double().cpu().numpy(); ll = ll.view(W, S).cpu().numpy()
    outs = []
    for w in range(W):
        T = len(inps[w]["md"]); pr = pts[w, :, :T]; llw = ll[w]
        if np.isfinite(llw).any():
            mx = np.nanmax(llw[np.isfinite(llw)])
            lk = np.where(np.isfinite(llw), llw - mx, -np.inf)
            ww = np.exp(lk / max(P["likelihood_scale"], 1e-6)); s = ww.sum()
            ww = ww / s if s > 1e-300 else np.full(S, 1.0 / S)
        else:
            ww = np.full(S, 1.0 / S)
        outs.append((ww[:, None] * pr).sum(0))
    return outs


# ----------------------- 評価 -----------------------
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


# ----------------------- Optuna 探索空間(v22=v19 と同一) -----------------------
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
        resamp           = trial.suggest_float("resamp", 0.03, 0.6, log=True),   # APFでは未使用(空間維持のため残す)
        apf_lookahead    = trial.suggest_float("apf_lookahead", 0.5, 2.0),       # 先読み強度 β: <1=穏やか, 1=標準APF, >1=攻撃的
        apf_mu_drift     = trial.suggest_float("apf_mu_drift", 0.5, 2.0),        # 先読み予測点の外挿量 κ: <1=控えめ, 1=素直, >1=強め
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


# ----------------------- 状態(再開用) -----------------------
def load_state():
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {"done_codes": [], "results": {}}

def save_state(state):
    STATE_PATH.write_text(json.dumps(state, indent=2))
    PER_CAT_PATH.write_text(json.dumps(dict(
        algorithm="auxiliary_particle_filter", n_eval_threshold=N_EVAL_THRESHOLD,
        z_span_thresholds=list(Z_SPAN_THRESHOLDS), eval_seeds=EVAL_SEEDS, use_phys=USE_PHYS,
        categories={str(code): state["results"][str(code)] for code in range(N_CATS)
                    if str(code) in state["results"]},
    ), indent=2))


# ----------------------- カテゴリ実行 -----------------------
def run_category(code, wells):
    seed = SEED0 + 100 * code
    study = optuna.create_study(direction="minimize", study_name=f"cat{code}",
                                storage=STORAGE, load_if_exists=True,
                                sampler=optuna.samplers.TPESampler(seed=seed))
    done = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)
    remaining = max(0, N_TRIALS - done)
    print(f"[cat {code}] wells={len(wells)} seed={seed} optuna {done}/{N_TRIALS} done, run {remaining} more", flush=True)

    def objective(trial):
        P = suggest_params(trial)
        try:
            rmse = overall_rmse_on(P, wells, seed, SEARCH_SEEDS, SEARCH_CHUNK)
        except Exception:
            return 1e3
        return rmse if math.isfinite(rmse) else 1e3

    def cb(study, trial):
        try:
            bv = study.best_value
        except Exception:
            return
        nd = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)
        CUR_PATH.write_text(json.dumps(dict(code=code, seed=seed, trials_done=nd, n_wells=len(wells),
                                            best_value=bv, best_params=full_params(study.best_params)), indent=2))
        tv = trial.value if trial.value is not None else float("nan")
        print(f"   [cat{code}] {nd}/{N_TRIALS} this={tv:.4f} best={bv:.4f}", flush=True)

    if remaining > 0:
        study.optimize(objective, n_trials=remaining, callbacks=[cb])
    bp = full_params(study.best_params)
    overall_cat = overall_rmse_on(bp, wells, seed, EVAL_SEEDS, EVAL_CHUNK)
    print(f"[cat {code}] best(search)={study.best_value:.4f}  overall(EVAL_SEEDS)={overall_cat:.4f}", flush=True)
    return dict(code=code, seed=seed, n_wells=len(wells), best_value_search=float(study.best_value),
                overall_on_cat=float(overall_cat), best_params=bp)


def main():
    global CACHE
    if DEVICE != "cuda":
        print("WARNING: CUDA not available, will be slow.")
    print(f"device={DEVICE}  storage={STORAGE}")

    wells = list_wells()
    if N_WELLS:
        wells = wells[:N_WELLS]
    print(f"loading {len(wells)} wells ...")
    CACHE = {}; t0 = time.time()
    for i, w in enumerate(wells):
        c = build_cache(w)
        if c is not None and c["tvt_true"] is not None:
            CACHE[w] = c
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(wells)} ({time.time()-t0:.0f}s)")
    all_wells = [w for w in wells if w in CACHE]

    cats = {code: [] for code in range(N_CATS)}
    for w in all_wells:
        cats[well_code(CACHE[w])].append(w)
    print(f"usable wells: {len(all_wells)}")
    print("カテゴリ別 well 数:", {code: len(cats[code]) for code in range(N_CATS)})

    state = load_state()
    print(f"resume: done codes = {state['done_codes']}")

    for code in range(N_CATS):
        if code in state["done_codes"]:
            continue
        if len(cats[code]) == 0:
            print(f"[cat {code}] well 0 件 → スキップ")
            state["done_codes"].append(code); save_state(state); continue
        rec = run_category(code, cats[code])
        state["results"][str(code)] = rec
        state["done_codes"].append(code)
        save_state(state)
        print(f"==> cat {code} saved. overall_on_cat={rec['overall_on_cat']:.4f}\n")

    print("ALL CATEGORIES DONE.")
    for code in range(N_CATS):
        r = state["results"].get(str(code))
        if r: print(f"  cat {code}: wells={r['n_wells']:4d}  overall_on_cat={r['overall_on_cat']:.4f}")
    print(f"6つの APF パラメータ -> {PER_CAT_PATH}")


if __name__ == "__main__":
    main()
