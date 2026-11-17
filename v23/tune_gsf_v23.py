# -*- coding: utf-8 -*-
"""
v23 Gaussian Sum Filter (GSF) 反復ブースティング・チューナー(v19構造・自動・再開可能)
=====================================================================================
構造は v19/iterative_pf_boost_v19.py と同じ「残差(ワースト坑井)ブースティングでバンク生成」:
  Round 0    : GSF を Optuna で N_TRIALS 探索(全well対象, overall RMSE最小化) → 最良パラメータ。
               その best で全wellを推論 → per-well RMSE のワースト WORST_K well を次ラウンド対象に。
  Round r>=1 : 直前ラウンドのワースト WORST_K well を対象に、もう一度 Optuna で N_TRIALS 探索。
  これを N_ROUNDS 回くり返し、**複数の GSF パラメータ(=各ラウンドの最良)=GSFバンク** を得る。
  (カテゴリ分割は廃止。全坑井プールで v19 と同じ形。)

GSF エンジン(_gsf_core, W×K バッチ・決定論):
  事後を K個のガウス成分(pos=geo_z=TVT+Z, rate=dip の2D)の混合で表す。
  - predict: 線形ガウス遷移 rate'=α·rate, pos'=pos+rate'·dm、P'=F P Fᵀ+Q。
  - update : 観測 h(pos)=typewell GR@(pos−z) をグリッド局所傾き s で EKF線形化、ゲイン×ekf_gain で平均をGRへ。
             重みは PF流 exp(-0.5(|y|/gs)^gr_power)(+任意 phys)。
  - 多峰維持: 初期に K成分を TVT方向へ init_span で散らす + ESS低下時に成分リサンプリング(固定シードのジッタ)。
  - 読み出し: TVT=Σ w_k^read_temp·(pos_k−z)(read_temp<1=hedge)。
  - 決定論(シード不要)。ラウンドごとに jitter seed = SEED0+1000*r を変え、バンクに多様性を持たせる(v19の seed+1000 相当)。

出力: v23/artifacts_v23/gsf_bank.json  (各ラウンドの best params)
実行: rogii_claude/.venv/Scripts/python.exe v23/tune_gsf_v23.py
※ GSFは小バッチ×逐次ループで GPU起動律速 → FORCE_CPU=True(CPUの方が速い)。
※ Claude は実行しない(ユーザーが実行)。
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
# GSFは小バッチ(B×K)・逐次Tループで GPU起動律速。小さいので CPU の方が速い → 既定 CPU。
FORCE_CPU = True
DEVICE    = "cpu" if FORCE_CPU else ("cuda" if torch.cuda.is_available() else "cpu")
DTYPE     = torch.float32

WELL_CHUNK = 800         # GPU/CPUバッチ坑井数(GSFはシード不要=軽い。全well一括でOK)
N_WELLS    = None        # None=全well, 整数で先頭N(デバッグ)
SEED0      = 0

N_TRIALS   = 200         # 1ラウンドあたりの Optuna 探索回数
N_ROUNDS   = 50          # ラウンド数(=得る GSF の本数)
WORST_K    = 200         # 次ラウンドの対象=ワースト何well
USE_PHYS   = True

STORAGE    = f"sqlite:///{(OUT_DIR / 'optuna_gsf.db').as_posix()}"
STATE_PATH = OUT_DIR / "state_gsf.json"
BANK_PATH  = OUT_DIR / "gsf_bank.json"
CUR_PATH   = OUT_DIR / "current_round_gsf.json"
# ===================================================================

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ----------------------- データ(v19と同一) -----------------------
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
                gg=gg, gmin=gmin, gst=gst, gs=gs, ls=c["ls"], ir=c["ir"])


# ----------------------- Gaussian Sum Filter GPU/CPU(W×K バッチ・決定論) -----------------------
def _gsf_core(md, z, gr, valid, grid, glen, vmin, step, gs, ls, ir, P, K, device, gen, dtype=DTYPE):
    B, T = md.shape
    ALPHA = P["mom"]; QP = P["pn"]; QR = P["vn"]; IS = P["init_pos_std"]; IR = P["init_rate_std"]
    SPAN = P["init_span"]; EKF = P["ekf_gain"]; GR_POWER = P["gr_power"]; CLIP = P["tvt_clip_margin"]
    RESAMP = P["resamp"]; JP = P["jitter_pos"]; JR = P["jitter_rate"]; RTEMP = P["read_temp"]
    PHYS_SIG = P["phys_sig"]; USE_P = P["use_phys"]

    arK = torch.arange(K, device=device, dtype=dtype)
    zero = torch.zeros((), device=device, dtype=dtype)
    tvt_lo = vmin - CLIP; tvt_hi = vmin + (glen.to(dtype) - 1) * step + CLIP
    glast = torch.gather(grid, 1, (glen - 1).clamp_min(0).unsqueeze(1)); g0 = grid[:, 0:1]
    dipp = ir[:, None]

    def grid_hs(v):
        ii = (v - vmin[:, None]) / step[:, None]; i0 = torch.floor(ii).long()
        below = i0 < 0; above = i0 >= (glen - 1)[:, None]
        i0c = i0.clamp_min(0); i0c = torch.minimum(i0c, (glen - 2).clamp_min(0)[:, None])
        fr = ii - i0c.to(dtype); gA = torch.gather(grid, 1, i0c); gB = torch.gather(grid, 1, i0c + 1)
        h = gA * (1 - fr) + gB * fr; s = (gB - gA) / step[:, None]
        h = torch.where(below, g0, h); h = torch.where(above, glast, h)
        s = torch.where(below | above, zero, s)
        return h, s

    off = torch.linspace(-1.0, 1.0, K, device=device, dtype=dtype)[None, :] * SPAN
    mpos = ls[:, None] + off
    mrate = ir[:, None].expand(B, K).clone()
    Ppp = torch.full((B, K), IS * IS, device=device, dtype=dtype)
    Prr = torch.full((B, K), IR * IR, device=device, dtype=dtype)
    Ppr = torch.zeros((B, K), device=device, dtype=dtype)
    w = torch.full((B, K), 1.0 / K, device=device, dtype=dtype)
    pred = torch.zeros((B, T), device=device, dtype=dtype)
    pm = md[:, 0] - 1.0

    for i in range(T):
        act = valid[:, i]; cur = md[:, i]; dm = (cur - pm).clamp_min(1.0)[:, None]
        zi = z[:, i][:, None]; gri = gr[:, i]; obs = act & ~torch.isnan(gri); am = act[:, None]

        mrate_p = ALPHA * mrate
        mpos_p = mpos + mrate_p * dm
        Ppp_p = Ppp + 2 * dm * Ppr + dm * dm * Prr + QP * QP
        Ppr_p = ALPHA * (Ppr + dm * Prr)
        Prr_p = ALPHA * ALPHA * Prr + QR * QR
        tvt_p = (mpos_p - zi).clamp(tvt_lo[:, None], tvt_hi[:, None]); mpos_p = tvt_p + zi

        h, s = grid_hs(mpos_p - zi)
        y = gri[:, None] - h
        S = s * s * Ppp_p + gs[:, None] * gs[:, None]
        Kp = EKF * (Ppp_p * s) / S.clamp_min(1e-12)
        Kr = EKF * (Ppr_p * s) / S.clamp_min(1e-12)
        mpos_u = mpos_p + Kp * y; mrate_u = mrate_p + Kr * y
        u = 1.0 - Kp * s; vv = -Kr * s
        Ppp_u = (u * Ppp_p).clamp_min(1e-9)
        Ppr_u = 0.5 * (u * Ppr_p + vv * Ppp_p + Ppr_p)
        Prr_u = (vv * Ppr_p + Prr_p).clamp_min(1e-9)
        tvt_u = (mpos_u - zi).clamp(tvt_lo[:, None], tvt_hi[:, None]); mpos_u = tvt_u + zi

        ob = obs[:, None]
        mpos_n = torch.where(ob, mpos_u, mpos_p); mrate_n = torch.where(ob, mrate_u, mrate_p)
        Ppp_n = torch.where(ob, Ppp_u, Ppp_p); Ppr_n = torch.where(ob, Ppr_u, Ppr_p); Prr_n = torch.where(ob, Prr_u, Prr_p)
        mpos = torch.where(am, mpos_n, mpos); mrate = torch.where(am, mrate_n, mrate)
        Ppp = torch.where(am, Ppp_n, Ppp); Ppr = torch.where(am, Ppr_n, Ppr); Prr = torch.where(am, Prr_n, Prr)

        d = (y / gs[:, None]).abs(); dp = d ** GR_POWER
        lk = torch.where(dp < 600.0, torch.exp(-0.5 * dp), zero).clamp_min(1e-300)
        if USE_P:
            dphys = (mrate - dipp) / PHYS_SIG; lkp = torch.exp(-0.5 * dphys * dphys).clamp_min(1e-300)
        else:
            lkp = torch.ones_like(w)
        w_new = torch.where(ob, w * lk * lkp, w * lkp)
        ws = w_new.sum(1, keepdim=True)
        w_norm = torch.where(ws > 0, w_new / ws.clamp_min(1e-300), torch.full_like(w, 1.0 / K))
        w = torch.where(am, w_norm, w)

        # 成分リサンプリング(毎ステップ実行しマスク適用、host同期回避)
        ne = (w * w).sum(1); need = act & ((1.0 / ne) < (RESAMP * K))
        cumw = torch.cumsum(w, 1)
        uu = (0.5 / K) + arK[None, :] / K
        idx = torch.searchsorted(cumw, uu.expand(B, K).contiguous(), right=False).clamp(0, K - 1)
        mpos_r = torch.gather(mpos, 1, idx) + JP * torch.randn((B, K), generator=gen, device=device, dtype=dtype)
        mrate_r = torch.gather(mrate, 1, idx) + JR * torch.randn((B, K), generator=gen, device=device, dtype=dtype)
        Ppp_r = torch.gather(Ppp, 1, idx); Ppr_r = torch.gather(Ppr, 1, idx); Prr_r = torch.gather(Prr, 1, idx)
        nm = need[:, None]
        mpos = torch.where(nm, mpos_r, mpos); mrate = torch.where(nm, mrate_r, mrate)
        Ppp = torch.where(nm, Ppp_r, Ppp); Ppr = torch.where(nm, Ppr_r, Ppr); Prr = torch.where(nm, Prr_r, Prr)
        w = torch.where(nm, torch.full_like(w, 1.0 / K), w)

        wt = w.clamp_min(1e-30) ** RTEMP; wt = wt / wt.sum(1, keepdim=True).clamp_min(1e-30)
        pred[:, i] = (wt * (mpos - zi)).sum(1)
        pm = torch.where(act, cur, pm)
    return pred


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

def run_forward(inps, P, device, jit_seed):
    K = int(P["n_components"]); W = len(inps); pad = _pad(inps, device)
    gen = torch.Generator(device=device); gen.manual_seed(int(jit_seed))
    pred = _gsf_core(pad["md"], pad["z"], pad["gr"], pad["valid"], pad["grid"], pad["glen"],
                     pad["vmin"], pad["step"], pad["gs"], pad["ls"], pad["ir"], P, K, device, gen)
    pred = pred.double().cpu().numpy()
    return [pred[w, :len(inps[w]["md"])] for w in range(W)]


# ----------------------- 評価(対象well指定, 決定論=シード不要) -----------------------
def predict_errs(P, subset, jit_seed, chunk):
    res = {}
    for c0 in range(0, len(subset), chunk):
        sub = subset[c0:c0 + chunk]
        inps = [make_inputs(CACHE[w], P) for w in sub]
        preds = run_forward(inps, P, DEVICE, jit_seed)
        for k, w in enumerate(sub):
            res[w] = preds[k] - CACHE[w]["tvt_true"]
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    return res

def overall_rmse_on(P, subset, jit_seed, chunk):
    errs = predict_errs(P, subset, jit_seed, chunk)
    e = np.concatenate([errs[w] for w in subset]); e = e[np.isfinite(e)]
    if e.size == 0 or not np.isfinite(e).all():
        return 1e3
    return float(np.sqrt(np.mean(e ** 2)))

def perwell_rmse(P, subset, jit_seed, chunk):
    errs = predict_errs(P, subset, jit_seed, chunk)
    out = {}
    for w, e in errs.items():
        e = e[np.isfinite(e)]
        out[w] = float(np.sqrt(np.mean(e ** 2))) if e.size else 1e3
    return out


# ----------------------- Optuna 探索空間(GSF) -----------------------
def suggest_params(trial):
    return dict(
        n_components   = trial.suggest_categorical("n_components", [16, 32, 64, 96]),
        mom            = trial.suggest_float("mom", 0.990, 0.99998),
        init_pos_std   = trial.suggest_float("init_pos_std", 0.1, 5.0, log=True),
        init_rate_std  = trial.suggest_float("init_rate_std", 0.0003, 0.03, log=True),
        init_span      = trial.suggest_float("init_span", 2.0, 120.0, log=True),
        pn             = trial.suggest_float("pn", 0.002, 0.06, log=True),
        vn             = trial.suggest_float("vn", 0.0003, 0.012, log=True),
        ekf_gain       = trial.suggest_float("ekf_gain", 0.0, 1.0),
        gr_sig_min     = trial.suggest_float("gr_sig_min", 4.0, 12.0),
        gr_sig_max     = trial.suggest_float("gr_sig_max", 22.0, 60.0),
        gr_sig_def     = trial.suggest_float("gr_sig_def", 70.0, 160.0),
        gr_sig_mult    = trial.suggest_float("gr_sig_mult", 1.6, 4.0),
        gr_power       = trial.suggest_float("gr_power", 1.4, 3.2),
        hgr_smooth_r   = trial.suggest_categorical("hgr_smooth_r", [0, 1, 2, 3, 4, 6]),
        tw_gr_smooth_r = trial.suggest_categorical("tw_gr_smooth_r", [0]),
        tvt_clip_margin= trial.suggest_float("tvt_clip_margin", 50.0, 200.0),
        resamp         = trial.suggest_float("resamp", 0.1, 0.8, log=True),
        jitter_pos     = trial.suggest_float("jitter_pos", 0.05, 1.5, log=True),
        jitter_rate    = trial.suggest_float("jitter_rate", 0.00005, 0.004, log=True),
        read_temp      = trial.suggest_float("read_temp", 0.3, 2.5, log=True),
        phys_sig       = trial.suggest_float("phys_sig", 0.003, 5.0, log=True),
        use_phys       = USE_PHYS,
    )

def full_params(best_params):
    return {**best_params, "use_phys": USE_PHYS}


# ----------------------- 状態(再開用) -----------------------
def load_state():
    # 旧カテゴリ版の state(done_codes/results)が残っていても新形式(rounds)として扱う
    if STATE_PATH.exists():
        d = json.loads(STATE_PATH.read_text())
        if isinstance(d, dict) and "rounds" in d:
            return d
    return {"rounds": []}

def save_state(state):
    STATE_PATH.write_text(json.dumps(state, indent=2))
    BANK_PATH.write_text(json.dumps(
        [{"round": r["round"], "jit_seed": r.get("jit_seed"), "overall_on_all": r["overall_on_all"], "params": r["best_params"]}
         for r in state["rounds"]], indent=2))


# ----------------------- ラウンド実行(v19構造) -----------------------
def run_round(r, target_wells):
    jit_seed = SEED0 + 1000 * r   # ラウンドごとにジッタseedを+1000(バンク多様性)
    study = optuna.create_study(direction="minimize", study_name=f"round{r}",
                                storage=STORAGE, load_if_exists=True,
                                sampler=optuna.samplers.TPESampler(seed=jit_seed))
    done = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)
    remaining = max(0, N_TRIALS - done)
    print(f"[round {r}] target={len(target_wells)} wells, jit_seed={jit_seed}, optuna {done}/{N_TRIALS} done, run {remaining} more", flush=True)

    def objective(trial):
        P = suggest_params(trial)
        try:
            rmse = overall_rmse_on(P, target_wells, jit_seed, WELL_CHUNK)
        except Exception:
            return 1e3
        return rmse if math.isfinite(rmse) else 1e3

    def cb(study, trial):
        try:
            bv = study.best_value; bp_cur = full_params(study.best_params)
        except Exception:
            return
        nd = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)
        CUR_PATH.write_text(json.dumps(dict(round=r, jit_seed=jit_seed, trials_done=nd,
                                            target_n=len(target_wells),
                                            best_value=bv, best_params=bp_cur), indent=2))
        tv = trial.value if trial.value is not None else float("nan")
        print(f"   [r{r}] {nd}/{N_TRIALS}  this={tv:.4f}  best={bv:.4f}", flush=True)

    if remaining > 0:
        study.optimize(objective, n_trials=remaining, callbacks=[cb])
    bp = full_params(study.best_params)
    print(f"[round {r}] best on target = {study.best_value:.4f}", flush=True)
    # best で全wellを推論 → per-well RMSE → ワースト WORST_K
    pw = perwell_rmse(bp, ALL_WELLS, jit_seed, WELL_CHUNK)
    e_all = predict_errs(bp, ALL_WELLS, jit_seed, WELL_CHUNK)
    ea = np.concatenate([e_all[w] for w in ALL_WELLS]); ea = ea[np.isfinite(ea)]
    overall_all = float(np.sqrt(np.mean(ea ** 2)))
    n_over10 = int(sum(v > 10 for v in pw.values()))
    worst = sorted(ALL_WELLS, key=lambda w: pw[w], reverse=True)[:WORST_K]
    print(f"[round {r}] overall(all)={overall_all:.4f}  RMSE>10 wells={n_over10}", flush=True)
    return bp, float(study.best_value), overall_all, pw, worst, n_over10


def main():
    global CACHE, ALL_WELLS
    if DEVICE != "cuda":
        print(f"device={DEVICE} (GSFは小バッチ×逐次のためCPU既定)")
    print(f"storage={STORAGE}")

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
    ALL_WELLS = [w for w in wells if w in CACHE]
    print(f"usable wells: {len(ALL_WELLS)}")

    state = load_state()
    completed = state["rounds"]
    print(f"resume: {len(completed)} rounds already completed")

    for r in range(N_ROUNDS):
        if r < len(completed):
            continue
        target = ALL_WELLS if r == 0 else completed[r - 1]["worst"]
        target = [w for w in target if w in CACHE]
        bp, val, overall_all, pw, worst, n_over10 = run_round(r, target)
        rec = dict(round=r, jit_seed=SEED0 + 1000 * r, target_n=len(target), best_params=bp,
                   best_value_on_target=val, overall_on_all=overall_all,
                   n_over10=n_over10, worst=worst, perwell_rmse=pw)
        completed.append(rec); state["rounds"] = completed; save_state(state)
        print(f"==> round {r} saved. overall(all)={overall_all:.4f}  >10={n_over10}\n")

    print("ALL ROUNDS DONE.")
    print("各ラウンド最良(overall on all):")
    for r in completed:
        print(f"  round {r['round']}: target={r['target_n']:4d}  overall_all={r['overall_on_all']:.4f}  >10={r['n_over10']}")
    print(f"GSFバンク -> {BANK_PATH}")


if __name__ == "__main__":
    main()
