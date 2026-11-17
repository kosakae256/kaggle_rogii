# -*- coding: utf-8 -*-
"""
v65 PF チューナー(間引き/MD粗化版) — DEC=10 固定で decimated PF を Optuna 探索
=====================================================================================
既存 tune_pf_v65.py の decimation 版。エンジン(pf_engine_v65)は無改造で、pf_dec_v65 経由で
MD=DEC 集約 → 全解像度に補間して採点(本物の MD-1 メトリク)。**別 study・別 db(optuna_dec.db)**なので
current v65 のチューニングに一切影響しない。
- 目的: 「間引きで RMSE が current v65(MD-1)を維持/改善するか」= decimation 固有効果(overconfidence/速度)の実測。
- 報告規律: 探索seed / 別seed(+1000) / use_anchor反転 の3点(採用判断は別seedで, [[v38-struct-prior-pf]])。
実行 : rogii_claude/.venv/Scripts/python.exe v65/tune_pf_dec_v65.py
env  : DEC(10) AGG(mean) TW_SOURCE(tw) STUDY(自動=v65dec{DEC}_{prior}_{tw}) N_TRIALS(300)
       SEARCH_SEEDS(32) EVAL_SEEDS(32) N_WELLS(0=全) SEARCH_CHUNK(240) EVAL_CHUNK(120) V38_PRIOR(既定=v66_leanv1cumul)
       ※anchor_ramp_md/lk_floor/gr_debias/rate_target は探索対象外(engine 側で中立固定, 既存と同じ)
"""
import os, json, math
from pathlib import Path
import numpy as np
import optuna
import sys
sys.stdout = open(1, "w", encoding="utf-8", closefd=False)
sys.path.insert(0, str(Path(__file__).parent))
import pf_engine_v65 as pf
import pf_dec_v65 as dec
import warmup_v65 as wu

# ============================== CONFIG ==============================
TW_SOURCE    = os.environ.get("TW_SOURCE", "tw")
os.environ["V65_TWSRC"] = TW_SOURCE
STUDY_NAME   = os.environ.get("STUDY", f"v65dec_no_st_tw_3_{dec.DEC}_{pf.PRIOR_NAME}_{TW_SOURCE.replace(chr(43), chr(95))}")
N_TRIALS     = int(os.environ.get("N_TRIALS", "300"))
SEARCH_SEEDS = int(os.environ.get("SEARCH_SEEDS", "32"))
EVAL_SEEDS   = int(os.environ.get("EVAL_SEEDS", "32"))
SEARCH_CHUNK = int(os.environ.get("SEARCH_CHUNK", "240"))
EVAL_CHUNK   = int(os.environ.get("EVAL_CHUNK", "120"))
N_WELLS      = int(os.environ.get("N_WELLS", "0")) or None
SEED_SEARCH  = 20250707
SEED_ALT     = SEED_SEARCH + 1000
HARDCORE_CSV = pf.PROJECT / "v32" / "artifacts_v32" / "hardcore_diag.csv"
# ===================================================================

OUT      = pf.OUT_DIR
STORAGE  = f"sqlite:///{(OUT / 'optuna_dec.db').as_posix()}"     # ★別db=current v65に非干渉
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
        print("WARNING: 構造prior無し。use_anchor=True の trial は錨OFF相当で走る。")
    print(f"[dec] DEC={dec.DEC} AGG={dec.AGG}  PRIOR={pf.PRIOR_NAME}  study={STUDY_NAME}")

    wells = pf.list_wells()
    if N_WELLS:
        wells = wells[:N_WELLS]
    cache = {}; n_anch = 0
    for w in wells:
        c = pf.build_cache(w)
        if c is not None and c["tvt_true"] is not None:
            cache[w] = c
            if np.isfinite(c["ancs"]).all() and (c["ancs"] < 1e8).any():
                n_anch += 1
    pf.CACHE = cache
    ALL = [w for w in wells if w in cache]
    print(f"usable wells: {len(ALL)}  (構造錨あり {n_anch})  tw_source={TW_SOURCE}  → MD{dec.DEC}集約でPF")

    study = optuna.create_study(direction="minimize", study_name=STUDY_NAME, storage=STORAGE,
                                load_if_exists=True, sampler=optuna.samplers.TPESampler(seed=SEED_SEARCH))
    done = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)
    print(f"optuna {done}/{N_TRIALS} done  -> {STUDY_NAME}")
    # warm-up: r0_seed32 + struct20(空study時 or FORCE_WARMUP=1)
    _wq = dict(wu.snap_to_space(wu.R0_PARAMS, wu.parse_engine_space()))
    _wq2 = dict(wu.snap_to_space(wu.ST20_PARAMS, wu.parse_engine_space()))
    for q in (_wq, _wq2):
        if "self" not in TW_SOURCE: q.pop("self_mix_w", None)
        if "nbr" not in TW_SOURCE: q.pop("nbr_mix_w", None)
    if len(study.trials) == 0 or os.environ.get("FORCE_WARMUP", "0") == "1":
        study.enqueue_trial(_wq, skip_if_exists=True); study.enqueue_trial(_wq2, skip_if_exists=True)
        print("[warmup] r0_seed32 + struct20 をenqueue", flush=True)

    def objective(trial):
        P = pf.full_params(pf.suggest_params(trial))



        P["use_anchor"] = False


        
        try:
            r = dec.overall_rmse_on_dec(P, ALL, SEED_SEARCH, SEARCH_SEEDS, SEARCH_CHUNK)
        except Exception:
            return 1e3
        return r if math.isfinite(r) else 1e3

    def cb(study, trial):
        try:
            bv = study.best_value; bp = pf.full_params(study.best_params)
        except Exception:
            return
        nd = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)
        CUR_PATH.write_text(json.dumps(dict(study=STUDY_NAME, dec=dec.DEC, agg=dec.AGG, tw_source=TW_SOURCE,
                                            trials_done=nd, best_value=bv, best_params=bp), indent=2))
        tv = trial.value if trial.value is not None else float("nan")
        print(f"   [dec{dec.DEC}] {nd}/{N_TRIALS}  this={tv:.4f}  best={bv:.4f}", flush=True)

    if done < N_TRIALS:
        study.optimize(objective, n_trials=N_TRIALS - done, callbacks=[cb])
    bp = pf.full_params(study.best_params)
    print(f"\n[dec{dec.DEC}] best overall(探索seed, in-sample, 全解像度) = {study.best_value:.4f}")
    print("best_params:", json.dumps({k: (round(v, 6) if isinstance(v, float) else v) for k, v in bp.items()}))

    hard = load_hardcore() & set(ALL); hard = [w for w in ALL if w in hard]

    def rep(tag, P, seed):
        e_all = dec.predict_errs_dec(P, ALL, seed, EVAL_SEEDS, EVAL_CHUNK)
        ov = np.concatenate([e_all[w] for w in ALL]); ov = ov[np.isfinite(ov)]
        hv = np.concatenate([e_all[w] for w in hard]) if hard else np.array([]); hv = hv[np.isfinite(hv)]
        o = float(np.sqrt(np.mean(ov ** 2))); h = float(np.sqrt(np.mean(hv ** 2))) if hv.size else None
        print(f"  [{tag}] overall={o:.4f}  hardcore({len(hard)})={h if h else float('nan'):.4f}")
        return o, h

    print("\n=== best params の評価(全解像度メトリク=current v65 と比較可。採用判断は別seedで) ===")
    on_ov, on_hd = rep(f"seed{SEED_SEARCH}(探索)", bp, SEED_SEARCH)
    alt_ov, alt_hd = rep(f"seed{SEED_ALT}(別seed)", bp, SEED_ALT)
    bp_flip = dict(bp); bp_flip["use_anchor"] = not bool(bp.get("use_anchor", True))
    fl_ov, fl_hd = rep(f"use_anchor反転({bp_flip['use_anchor']})", bp_flip, SEED_ALT)

    OUT_BEST.write_text(json.dumps(dict(
        study=STUDY_NAME, dec=dec.DEC, agg=dec.AGG, tw_source=TW_SOURCE,
        best_value_search=float(study.best_value),
        overall_search_seed=on_ov, overall_alt_seed=alt_ov,
        hardcore_search_seed=on_hd, hardcore_alt_seed=alt_hd,
        overall_anchor_flipped_alt=fl_ov, hardcore_anchor_flipped_alt=fl_hd, params=bp), indent=2))
    print(f"\nsaved -> {OUT_BEST}")


if __name__ == "__main__":
    main()
