# -*- coding: utf-8 -*-
"""
v65 PFチューナー — 拡張パラメータ空間の Optuna 探索(tune_anchor_v38 ベース, iteration無し)
=====================================================================================
- エンジン: v65/pf_engine_v65.py(v38コア + use_anchor/anchor_mult/anchor_ramp_md/
  lk_floor/gr_debias/rate_target/grid_step を trial 単位で探索)
- 目的関数: 全坑井 overall pooled RMSE(Kaggle指標と同一)
- 報告規律: ベストは探索seedと**別seed(+1000)の両方**で評価して表示(v38教訓:
  探索seedの数字は~0.6下駄が乗り得る。採用判断・docs記載は別seed値で行うこと)
- 実行 : rogii_claude/.venv/Scripts/python.exe v65/tune_pf_v65.py
  env  : STUDY(v65_pf_{prior}_{tw}) N_TRIALS(300) SEARCH_SEEDS(32) EVAL_SEEDS(32) N_WELLS(0=全)
         SEARCH_CHUNK(240) EVAL_CHUNK(120) V38_PRIOR(v66_leanv1cumul=改善版struct既定。旧v38は struct)
  ※anchor_ramp_md/lk_floor/gr_debias/rate_target は探索対象外(中立固定, 2026-07-19)
  再開 : optuna.db(v65/artifacts_v65)に貯まるので同じコマンドで続行。
  途中経過: {STUDY}_current.json が毎trial更新(best_params は full_params 済み=そのまま実行可)
"""
import os, json, math
from pathlib import Path
import numpy as np
import optuna
import sys
sys.stdout = open(1, "w", encoding="utf-8", closefd=False)
sys.path.insert(0, str(Path(__file__).parent))
import pf_engine_v65 as pf
import warmup_v65 as wu

# ============================== CONFIG(v86: GR表現別PF最適化, 錨/struct/dec不使用) ==============================
# ★v86: GR表現ごとにPFを最適化。raw=r0_seeds32そのまま/logamp/affine=各表現前提でOptuna。
#   物理錨(use_anchor)・struct・phys・gr_debias・rate_target・dec は全て不使用(中立固定・探索から除外)。
#   初期値=r0_seed32(warmup_v65.R0_PARAMS)をenqueue。n_seeds=32。
REP          = os.environ.get("REP", "logamp")          # raw | logamp | affine
os.environ["V86_REP"] = REP                              # エンジンbuild_cacheのGR変換に伝播
os.environ.setdefault("V86_LOG_G0", "40.0")             # logampのg0(固定)
TW_SOURCE    = "tw"                                       # ★v86: self/nbr不使用(生波形=twのみ)
os.environ["V65_TWSRC"] = TW_SOURCE
STUDY_NAME   = os.environ.get("STUDY", f"{REP}_seeds32_0")
N_TRIALS     = int(os.environ.get("N_TRIALS", "300"))
SEARCH_SEEDS = int(os.environ.get("SEARCH_SEEDS", "32"))
EVAL_SEEDS   = int(os.environ.get("EVAL_SEEDS", "32"))
SEARCH_CHUNK = int(os.environ.get("SEARCH_CHUNK", "120"))
EVAL_CHUNK   = int(os.environ.get("EVAL_CHUNK", "120"))
N_WELLS      = int(os.environ.get("N_WELLS", "120")) or None
SEED_SEARCH  = 20250707
SEED_ALT     = SEED_SEARCH + 1000       # 別seed報告用
HARDCORE_CSV = pf.PROJECT / "v32" / "artifacts_v32" / "hardcore_diag.csv"

# ★v86 探索空間: 錨/struct/phys/gr_debias/rate_target/grid_step を除外(中立固定)。GR系+動力学のみ探索。
def suggest_params_v86(trial):
    return dict(
        n_particles      = trial.suggest_categorical("n_particles", [400, 500, 600, 700, 800]),
        likelihood_scale = trial.suggest_float("likelihood_scale", 6.0, 40.0, log=True),
        init_pos_std     = trial.suggest_float("init_pos_std", 0.15, 2.0, log=True),
        init_rate_std    = trial.suggest_float("init_rate_std", 0.0003, 0.02, log=True),
        mom              = trial.suggest_float("mom", 0.995, 0.99998),
        vn               = trial.suggest_float("vn", 0.0003, 0.010, log=True),
        pn               = trial.suggest_float("pn", 0.002, 0.03, log=True),
        rp               = trial.suggest_float("rp", 0.1, 1.2, log=True),
        rr               = trial.suggest_float("rr", 0.00005, 0.003, log=True),
        resamp           = trial.suggest_float("resamp", 0.03, 0.4, log=True),
        gr_sig_min       = trial.suggest_float("gr_sig_min", 4.0, 14.0),
        gr_sig_max       = trial.suggest_float("gr_sig_max", 22.0, 55.0),
        gr_sig_def       = trial.suggest_float("gr_sig_def", 70.0, 155.0),
        gr_sig_mult      = trial.suggest_float("gr_sig_mult", 1.6, 3.8),
        tvt_clip_margin  = trial.suggest_float("tvt_clip_margin", 60.0, 180.0),
        rate_clip        = trial.suggest_categorical("rate_clip", [0.0, 1.5, 2.0, 2.5, 3.0]),
        gr_power         = trial.suggest_float("gr_power", 1.4, 3.0),
        hgr_smooth_r     = trial.suggest_categorical("hgr_smooth_r", [0, 1, 2, 3]),
        tw_gr_smooth_r   = trial.suggest_categorical("tw_gr_smooth_r", [0, 1]),
        jump_prob        = trial.suggest_categorical("jump_prob", [0.0, 0.002, 0.005, 0.01]),
        jump_std         = trial.suggest_categorical("jump_std", [5.0, 10.0, 15.0]),
        # ---- 中立固定(探索しない): 錨/struct/phys/debias/rate/grid ----
        use_phys=False, phys_sig=1.0, use_anchor=False, anchor_mult=20.0, anchor_ramp_md=0.0,
        lk_floor=0.0, gr_debias=0, rate_target="zero", grid_step=0.2,
    )
pf.suggest_params = suggest_params_v86                    # ★エンジンのsuggestを差し替え

# ★中立固定値(探索対象外)。study.best_paramsはsuggestした値しか持たないため、
#   full_params()するとエンジン既定(use_anchor=True等)が埋まってstructが混入する。
#   報告・保存の直前にこれで上書きし、探索時と同じ struct-free 条件に揃える。
FIXED_NEUTRAL = dict(use_phys=False, phys_sig=1.0, use_anchor=False, anchor_mult=20.0,
                     anchor_ramp_md=0.0, lk_floor=0.0, gr_debias=0, rate_target="zero", grid_step=0.2)
# ===================================================================

OUT      = pf.PROJECT / "v86" / "artifacts_v86"; OUT.mkdir(parents=True, exist_ok=True)
STORAGE  = f"sqlite:///{(OUT / 'optuna.db').as_posix()}"
OUT_BEST = OUT / f"{STUDY_NAME}_best.json"
CUR_PATH = OUT / f"{STUDY_NAME}_current.json"


def load_hardcore():
    try:
        import pandas as pd
        d = pd.read_csv(HARDCORE_CSV)
        return set(d[d["hard"] == True].iloc[:, 0].astype(str))
    except Exception:
        return set()


def main():
    if pf.DEVICE != "cuda":
        print("WARNING: CUDA 無し。遅いです。")
    if not pf._STRUCT:
        print("WARNING: 構造prior無し。use_anchor=True の trial は錨OFF相当で走る(ancs=1e9)。")

    wells = pf.list_wells()
    if N_WELLS:
        wells = wells[:N_WELLS]
    cache = {}
    n_anch = 0
    for w in wells:
        c = pf.build_cache(w)
        if c is not None and c["tvt_true"] is not None:
            cache[w] = c
            if np.isfinite(c["ancs"]).all() and (c["ancs"] < 1e8).any():
                n_anch += 1
    pf.CACHE = cache
    ALL = [w for w in wells if w in cache]
    n_nbr = sum(1 for w in ALL if pf.CACHE[w].get("nbr_pairs") is not None)
    print(f"usable wells: {len(ALL)}  (構造錨あり {n_anch} / nbrプールあり {n_nbr})  tw_source={TW_SOURCE}")

    # ★v86 raw: 最適化せず r0_seeds32 をそのまま評価・保存
    if REP == "raw":
        Praw = pf.full_params(dict(wu.R0_PARAMS)); Praw["use_anchor"] = False
        o = pf.overall_rmse_on(Praw, ALL, SEED_ALT, EVAL_SEEDS, EVAL_CHUNK)
        (OUT / "raw_seeds32_0.json").write_text(json.dumps({**Praw, "_rep": "raw", "overall_alt": o}, indent=2))
        print(f"[v86] raw=r0_seeds32そのまま保存。全{len(ALL)}井 overall(別seed)={o:.4f}")
        return

    study = optuna.create_study(direction="minimize", study_name=STUDY_NAME, storage=STORAGE,
                                load_if_exists=True, sampler=optuna.samplers.TPESampler(seed=SEED_SEARCH))
    done = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)
    print(f"optuna {done}/{N_TRIALS} done")
    # ---- warm-up: r0_seed32 を初期trialに(錨/struct無し。空study時。FORCE_WARMUP=1で既存にも) ----
    _sp = wu.parse_engine_space()
    _wq = {k: v for k, v in wu.snap_to_space(wu.R0_PARAMS, _sp).items()
           if k not in ("self_mix_w", "nbr_mix_w", "use_anchor", "anchor_mult", "anchor_ramp_md",
                        "lk_floor", "gr_debias", "rate_target", "grid_step", "phys_sig", "use_phys")}
    # ★完了trialが0なら必ずwarmup(stale dbの未完了trialに邪魔されないよう done で判定)
    if done == 0 or os.environ.get("FORCE_WARMUP", "0") == "1":
        study.enqueue_trial(_wq, skip_if_exists=True)
        print(f"[v86 warmup] r0_seed32(錨off) をenqueue REP={REP}", flush=True)
    else:
        print(f"[v86] 既存study(done={done})を継続。warmupスキップ(FORCE_WARMUP=1で強制)", flush=True)

    def objective(trial):
        P = pf.full_params(pf.suggest_params(trial))
        try:
            r = pf.overall_rmse_on(P, ALL, SEED_SEARCH, SEARCH_SEEDS, SEARCH_CHUNK)
        except Exception:
            return 1e3
        return r if math.isfinite(r) else 1e3

    def cb(study, trial):
        try:
            bv = study.best_value; bp = {**pf.full_params(study.best_params), **FIXED_NEUTRAL}
        except Exception:
            return
        nd = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)
        CUR_PATH.write_text(json.dumps(dict(study=STUDY_NAME, tw_source=TW_SOURCE, trials_done=nd,
                                            best_value=bv, best_params=bp), indent=2))
        tv = trial.value if trial.value is not None else float("nan")
        print(f"   [v65] {nd}/{N_TRIALS}  this={tv:.4f}  best={bv:.4f}", flush=True)

    if done < N_TRIALS:
        study.optimize(objective, n_trials=N_TRIALS - done, callbacks=[cb])
    bp = {**pf.full_params(study.best_params), **FIXED_NEUTRAL}   # ★struct-freeに揃える(use_anchor=False)
    print(f"\n[v65] best overall(探索seed, in-sample) = {study.best_value:.4f}")
    print("best_params:", json.dumps({k: (round(v, 6) if isinstance(v, float) else v) for k, v in bp.items()}))

    # ---- 報告: 探索seed / 別seed × overall(全坑井)。採用判断は別seedで ----
    def rep(tag, P, seed):
        e_all = pf.predict_errs(P, ALL, seed, EVAL_SEEDS, EVAL_CHUNK)
        ov = np.concatenate([e_all[w] for w in ALL]); ov = ov[np.isfinite(ov)]
        o = float(np.sqrt(np.mean(ov ** 2)))
        print(f"  [{tag}] overall={o:.4f}")
        return o

    print(f"\n=== v86 {REP}_seeds32_0 best評価(採用判断は別seed値で) ===")
    on_ov = rep(f"seed{SEED_SEARCH}(探索)", bp, SEED_SEARCH)
    alt_ov = rep(f"seed{SEED_ALT}(別seed)", bp, SEED_ALT)
    save = OUT / f"{REP}_seeds32_0.json"
    save.write_text(json.dumps({**bp, "_rep": REP, "_log_g0": float(os.environ["V86_LOG_G0"]),
        "study": STUDY_NAME, "best_value_search": float(study.best_value),
        "overall_search_seed": on_ov, "overall_alt_seed": alt_ov}, indent=2))
    print(f"\nsaved -> {save}")


if __name__ == "__main__":
    main()
