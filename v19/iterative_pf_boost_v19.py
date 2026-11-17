"""
v19 反復ブースティング PFチューナー(物理モデル版・自動・再開可能)
=================================================================

やること(ユーザー指示):
  Round 0: 物理PFを Optuna で N_TRIALS 回探索(全well対象, overall RMSE最小化) → 最良パラメータを使う。
  Round r>=1: 直前ラウンドの最良パラメータで全wellを推論 → per-well RMSE のワースト WORST_K well を対象に、
              もう一度 Optuna で N_TRIALS 探索 → その(対象wellでの)最良を採用。
  これを N_ROUNDS 回くり返し、**複数のPFパラメータ(=各ラウンドの最良)** を得る。

特徴(要件):
  - 各ステップの最良パラメータを記録(state.json / pf_bank.json)。
  - 途中で止めても再開できる:
      * Optuna は SQLite ストレージ(trial 単位で再開)。
      * ラウンド進捗は state.json(完了ラウンドはスキップ)。
  - 1回起動すれば最後まで自動で回る(寝てる間用)。

実行: rogii_claude/.venv/Scripts/python.exe v19/iterative_pf_boost_v19.py
※ ベースは pf_tuning/pf_tuning_gpu_phys.ipynb の物理PF。GPU(ancc, n_seeds=8)。
"""

import os, time, json, glob, math, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import optuna

warnings.filterwarnings("ignore")     # 極端パラメータ由来の RuntimeWarning(overflow/invalid)を抑制
np.seterr(all="ignore")

# ============================== CONFIG ==============================
PROJECT   = Path(r"C:\Users\kosaka256\Documents\rogii_claude")
DATA_DIR  = PROJECT / "rogii-wellbore-geology-prediction" / "train"
OUT_DIR   = PROJECT / "v19" / "artifacts_v19_aug"; OUT_DIR.mkdir(parents=True, exist_ok=True)
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE     = torch.float32

SEARCH_SEEDS = 128         # 探索(Optuna objective)用のシード数(速さ優先)
EVAL_SEEDS   = 128        # 選定(ワースト400)・記録best・全well推論用(本番一致, 1ラウンド1回なので安い)
SEED0      = 9999           # trial間の公平比較用に固定
SEARCH_CHUNK = 120       # 探索(8seed): GPUバッチに載せるwell数(大きめOK)
EVAL_CHUNK   = 120       # 評価(32seed): 全well同時はOOMするので分割
N_WELLS    = None        # None=全well, 整数で先頭N(デバッグ)

N_TRIALS   = 300         # 1ラウンドあたりの Optuna 探索回数
N_ROUNDS   = 50          # ラウンド数(=得るPFの本数)
WORST_K    = 400         # 次ラウンドの対象=ワースト何well

STORAGE    = f"sqlite:///{(OUT_DIR / 'optuna.db').as_posix()}"
STATE_PATH = OUT_DIR / "state.json"
BANK_PATH  = OUT_DIR / "pf_bank.json"
CUR_PATH   = OUT_DIR / "current_round.json"   # 実行中ラウンドの best を逐次記録
USE_PHYS   = False
# ===================================================================

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ----------------------- 物理PF(pf_tuning_gpu_phys より) -----------------------
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
    ir = float(np.median((dt + dz)[m] / dm[m])) if m.sum() >= 3 else 0.   # dip_prior
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
                gg=gg, gmin=gmin, gst=gst, gs=gs, ls=c["ls"], ir=c["ir"], N=int(P["n_particles"]))

def _pf_core(md, z, gr, valid, grid, glen, vmin, step, gs, ls, ir, P, N, device, gen, dtype=DTYPE):
    B, T = md.shape
    ALPHA = P["mom"]; RN = P["vn"]; PN = P["pn"]; IR = P["init_rate_std"]; IS = P["init_pos_std"]
    RP = P["rp"]; RR = P["rr"]; RESAMP = P["resamp"]; RATE_CLIP = P["rate_clip"]
    GR_POWER = P["gr_power"]; JUMP_PROB = P["jump_prob"]; JUMP_STD = P["jump_std"]; CLIP = P["tvt_clip_margin"]
    PHYS_SIG = P["phys_sig"]; USE_P = P["use_phys"]
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
    pts, ll = _pf_core(rep(pad["md"]), rep(pad["z"]), rep(pad["gr"]), rep(pad["valid"]), rep(pad["grid"]),
                       pad["glen"].repeat_interleave(S), pad["vmin"].repeat_interleave(S), pad["step"].repeat_interleave(S),
                       pad["gs"].repeat_interleave(S), pad["ls"].repeat_interleave(S), pad["ir"].repeat_interleave(S),
                       P, N, device, gen)
    pts = pts.view(W, S, -1).double().cpu().numpy(); ll = ll.view(W, S).cpu().numpy()
    outs = []
    for w in range(W):
        T = len(inps[w]["md"]); pr = pts[w, :, :T]
        llw = ll[w]
        if np.isfinite(llw).any():                       # 全非有限なら一様重み(極端パラメータ対策)
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
    """subset の各 well の (pred - true) 残差を返す dict。seed=PFシード, n_seeds=アンサンブル本数, chunk=GPUバッチwell数。"""
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


# ----------------------- Optuna 探索空間(広いが安定: 病的な極端域は削除) -----------------------
# ※ 旧レンジの trial が optuna.db にあると分布不一致でエラー → db を消してから回す。
# ※ NaN になる極端パラメータは objective が 1e3 で弾くが、無駄trialを減らすため裾は除去済み。
def suggest_params(trial):
    return dict(
        n_particles      = trial.suggest_categorical("n_particles", [400, 500, 600, 800]),
        likelihood_scale = trial.suggest_float("likelihood_scale", 6.0, 60.0, log=True),
        init_pos_std     = trial.suggest_float("init_pos_std", 0.1, 4.0, log=True),      # 急シフト被覆(常識的範囲)
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


# ----------------------- 状態(再開用) -----------------------
def load_state():
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {"rounds": []}

def save_state(state):
    STATE_PATH.write_text(json.dumps(state, indent=2))
    BANK_PATH.write_text(json.dumps(
        [{"round": r["round"], "seed": r.get("seed"), "overall_on_all": r["overall_on_all"], "params": r["best_params"]}
         for r in state["rounds"]], indent=2))


# ----------------------- ラウンド実行 -----------------------
def run_round(r, target_wells):
    seed = SEED0 + 1000 * r   # ラウンドごとに PFシードを +1000(シード多様性)
    study = optuna.create_study(direction="minimize", study_name=f"round{r}",
                                storage=STORAGE, load_if_exists=True,
                                sampler=optuna.samplers.TPESampler(seed=seed))
    done = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)
    remaining = max(0, N_TRIALS - done)
    print(f"[round {r}] target={len(target_wells)} wells, seed={seed}, optuna {done}/{N_TRIALS} done, run {remaining} more", flush=True)

    def objective(trial):
        P = suggest_params(trial)
        try:
            rmse = overall_rmse_on(P, target_wells, seed, SEARCH_SEEDS, SEARCH_CHUNK)
        except Exception:
            return 1e3
        return rmse if math.isfinite(rmse) else 1e3

    def cb(study, trial):
        # 実行中も逐次 best を artifacts_v19 に記録(クラッシュ/覗き見対応)+ 毎trial表示
        try:
            bv = study.best_value; bp_cur = full_params(study.best_params)
        except Exception:
            return
        nd = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)
        CUR_PATH.write_text(json.dumps(dict(round=r, seed=seed, trials_done=nd,
                                            target_n=len(target_wells),
                                            best_value=bv, best_params=bp_cur), indent=2))
        tv = trial.value if trial.value is not None else float("nan")
        print(f"   [r{r}] {nd}/{N_TRIALS}  this={tv:.4f}  best={bv:.4f}", flush=True)

    if remaining > 0:
        study.optimize(objective, n_trials=remaining, callbacks=[cb])
    bp = full_params(study.best_params)
    print(f"[round {r}] best on target = {study.best_value:.4f}", flush=True)
    # この best を使って全wellを推論(本番EVAL_SEEDS, 同じラウンドseed) → per-well RMSE → ワースト WORST_K
    pw = perwell_rmse(bp, ALL_WELLS, seed, EVAL_SEEDS, EVAL_CHUNK)
    e_all = predict_errs(bp, ALL_WELLS, seed, EVAL_SEEDS, EVAL_CHUNK)
    ea = np.concatenate([e_all[w] for w in ALL_WELLS]); ea = ea[np.isfinite(ea)]
    overall_all = float(np.sqrt(np.mean(ea ** 2)))
    n_over10 = int(sum(v > 10 for v in pw.values()))
    worst = sorted(ALL_WELLS, key=lambda w: pw[w], reverse=True)[:WORST_K]
    print(f"[round {r}] overall(all)={overall_all:.4f}  RMSE>10 wells={n_over10}", flush=True)
    return bp, float(study.best_value), overall_all, pw, worst, n_over10


def main():
    global CACHE, ALL_WELLS
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
    ALL_WELLS = [w for w in wells if w in CACHE]
    print(f"usable wells: {len(ALL_WELLS)}")

    state = load_state()
    completed = state["rounds"]
    print(f"resume: {len(completed)} rounds already completed")

    for r in range(N_ROUNDS):
        if r < len(completed):
            continue                                          # 完了済みはスキップ
        target = ALL_WELLS if r == 0 else completed[r - 1]["worst"]
        target = [w for w in target if w in CACHE]            # 念のため
        bp, val, overall_all, pw, worst, n_over10 = run_round(r, target)
        rec = dict(round=r, seed=SEED0 + 1000 * r, target_n=len(target), best_params=bp,
                   best_value_on_target=val, overall_on_all=overall_all,
                   n_over10=n_over10, worst=worst, perwell_rmse=pw)
        completed.append(rec); state["rounds"] = completed; save_state(state)
        print(f"==> round {r} saved. overall(all)={overall_all:.4f}  >10={n_over10}\n")

    print("ALL ROUNDS DONE.")
    print("各ラウンド最良(overall on all):")
    for r in completed:
        print(f"  round {r['round']}: target={r['target_n']:4d}  overall_all={r['overall_on_all']:.4f}  >10={r['n_over10']}")
    print(f"PFバンク -> {BANK_PATH}")


if __name__ == "__main__":
    main()
