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

# ============================== CONFIG ==============================
TW_SOURCE    = os.environ.get("TW_SOURCE", "tw")      # tw | self | nbr | self+nbr(study単位で固定)
os.environ["V65_TWSRC"] = TW_SOURCE                      # エンジンのcache/suggestに伝播
# ★study名に prior 名を含める = 別prior(v38 struct / v66_*)の trial が optuna.db で混ざらないように
STUDY_NAME   = os.environ.get("STUDY", f"v65_pf_{pf.PRIOR_NAME}_{TW_SOURCE.replace(chr(43), chr(95))}")
N_TRIALS     = int(os.environ.get("N_TRIALS", "300"))
SEARCH_SEEDS = int(os.environ.get("SEARCH_SEEDS", "32"))
EVAL_SEEDS   = int(os.environ.get("EVAL_SEEDS", "32"))
SEARCH_CHUNK = int(os.environ.get("SEARCH_CHUNK", "240"))
EVAL_CHUNK   = int(os.environ.get("EVAL_CHUNK", "120"))
N_WELLS      = int(os.environ.get("N_WELLS", "0")) or None
SEED_SEARCH  = 20250707
SEED_ALT     = SEED_SEARCH + 1000       # 別seed報告用
HARDCORE_CSV = pf.PROJECT / "v32" / "artifacts_v32" / "hardcore_diag.csv"
# ===================================================================

OUT      = pf.OUT_DIR
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

    study = optuna.create_study(direction="minimize", study_name=STUDY_NAME, storage=STORAGE,
                                load_if_exists=True, sampler=optuna.samplers.TPESampler(seed=SEED_SEARCH))
    done = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)
    print(f"optuna {done}/{N_TRIALS} done")
    # ---- warm-up: r0_seed32 を初期trialに(空study時。FORCE_WARMUP=1で既存studyにも) ----
    _wq = dict(wu.snap_to_space(wu.R0_PARAMS, wu.parse_engine_space()))
    if "self" not in TW_SOURCE: _wq.pop("self_mix_w", None)
    if "nbr" not in TW_SOURCE: _wq.pop("nbr_mix_w", None)
    _wq2 = dict(wu.snap_to_space(wu.ST20_PARAMS, wu.parse_engine_space()))
    if "self" not in TW_SOURCE: _wq2.pop("self_mix_w", None)
    if "nbr" not in TW_SOURCE: _wq2.pop("nbr_mix_w", None)
    if len(study.trials) == 0 or os.environ.get("FORCE_WARMUP", "0") == "1":
        study.enqueue_trial(_wq, skip_if_exists=True)
        study.enqueue_trial(_wq2, skip_if_exists=True)
        print(f"[warmup] r0_seed32 + struct20(LB7.881) をenqueue", flush=True)

    def objective(trial):
        P = pf.full_params(pf.suggest_params(trial))
        try:
            r = pf.overall_rmse_on(P, ALL, SEED_SEARCH, SEARCH_SEEDS, SEARCH_CHUNK)
        except Exception:
            return 1e3
        return r if math.isfinite(r) else 1e3

    def cb(study, trial):
        try:
            bv = study.best_value; bp = pf.full_params(study.best_params)
        except Exception:
            return
        nd = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)
        CUR_PATH.write_text(json.dumps(dict(study=STUDY_NAME, tw_source=TW_SOURCE, trials_done=nd,
                                            best_value=bv, best_params=bp), indent=2))
        tv = trial.value if trial.value is not None else float("nan")
        print(f"   [v65] {nd}/{N_TRIALS}  this={tv:.4f}  best={bv:.4f}", flush=True)

    if done < N_TRIALS:
        study.optimize(objective, n_trials=N_TRIALS - done, callbacks=[cb])
    bp = pf.full_params(study.best_params)
    print(f"\n[v65] best overall(探索seed, in-sample) = {study.best_value:.4f}")
    print("best_params:", json.dumps({k: (round(v, 6) if isinstance(v, float) else v) for k, v in bp.items()}))

    # ---- 報告: 探索seed / 別seed × overall / 硬い核。錨採否は best の use_anchor 反転でも測る ----
    hard = load_hardcore() & set(ALL)
    hard = [w for w in ALL if w in hard]

    def rep(tag, P, seed):
        e_all = pf.predict_errs(P, ALL, seed, EVAL_SEEDS, EVAL_CHUNK)
        ov = np.concatenate([e_all[w] for w in ALL]); ov = ov[np.isfinite(ov)]
        hv = np.concatenate([e_all[w] for w in hard]) if hard else np.array([])
        hv = hv[np.isfinite(hv)]
        o = float(np.sqrt(np.mean(ov ** 2))); h = float(np.sqrt(np.mean(hv ** 2))) if hv.size else None
        print(f"  [{tag}] overall={o:.4f}  hardcore({len(hard)})={h if h else float('nan'):.4f}")
        return o, h

    print("\n=== best params の評価(採用判断は別seed値で) ===")
    on_ov, on_hd = rep(f"seed{SEED_SEARCH}(探索)", bp, SEED_SEARCH)
    alt_ov, alt_hd = rep(f"seed{SEED_ALT}(別seed)", bp, SEED_ALT)
    bp_flip = dict(bp); bp_flip["use_anchor"] = not bool(bp.get("use_anchor", True))
    fl_ov, fl_hd = rep(f"use_anchor反転({bp_flip['use_anchor']})", bp_flip, SEED_ALT)

    OUT_BEST.write_text(json.dumps(dict(
        study=STUDY_NAME, tw_source=TW_SOURCE, best_value_search=float(study.best_value),
        overall_search_seed=on_ov, overall_alt_seed=alt_ov,
        hardcore_search_seed=on_hd, hardcore_alt_seed=alt_hd,
        overall_anchor_flipped_alt=fl_ov, hardcore_anchor_flipped_alt=fl_hd,
        params=bp), indent=2))
    print(f"\nsaved -> {OUT_BEST}")


if __name__ == "__main__":
    main()
