# -*- coding: utf-8 -*-
"""
v65 合成尤度PFチューナー — tune_pf_v65 のコピー。エンジンだけ pf_engine_multi_v65 に差し替え。
=====================================================================================
- tw/self/nbr の emission尤度を"積"で合成する PF(pf_engine_multi_v65)を Optuna 最適化。
  self_mix_w/nbr_mix_w は「emission指数β(信頼度)」として振られる(suggest_params は本体を再利用)。
- **TW_SOURCE 既定=tw+self+nbr**(tw だけなら合成の意味なし=本体 tune_pf_v65 と等価)。
- 別 study名(v65m_pf_...)+別db(optuna_multi.db) → 既存 blend系 study と混ざらない。
- 報告規律は本体と同じ: ベストは探索seed/別seed(+1000)両方で評価(採用判断は別seed値)。
実行 : rogii_claude/.venv/Scripts/python.exe v65/tune_pf_multi_v65.py
  env : TW_SOURCE(tw+self+nbr) STUDY N_TRIALS(300) SEARCH_SEEDS(32) EVAL_SEEDS(32)
        SEARCH_CHUNK(240) EVAL_CHUNK(120) V38_PRIOR(v66_leanv1cumul) N_WELLS(0=全)
  再開: optuna_multi.db に貯まるので同じコマンドで続行。{STUDY}_current.json が毎trial更新。
"""
import os, json, math
from pathlib import Path
import numpy as np
import optuna
import sys
sys.stdout = open(1, "w", encoding="utf-8", closefd=False)
sys.path.insert(0, str(Path(__file__).parent))
import pf_engine_multi_v65 as pf
import warmup_v65 as wu

# ============================== CONFIG ==============================
TW_SOURCE    = os.environ.get("TW_SOURCE", "tw+self+nbr")   # ★合成尤度は self/nbr 込みが既定
os.environ["V65_TWSRC"] = TW_SOURCE                          # エンジンの build_cache(nbr) / suggest に伝播
STUDY_NAME   = os.environ.get("STUDY", f"v65m_pf_{pf.PRIOR_NAME}_{TW_SOURCE.replace(chr(43), chr(95))}")
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
STORAGE  = f"sqlite:///{(OUT / 'optuna_multi.db').as_posix()}"   # ★blend系と別db
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
    print(f"[合成尤度] 参照 = tw + " + " + ".join([s for s in ("self", "nbr") if s in TW_SOURCE])
          + f"  (nbrプール無い {len(ALL)-n_nbr} 坑井は nbr尤度=中立)")

    study = optuna.create_study(direction="minimize", study_name=STUDY_NAME, storage=STORAGE,
                                load_if_exists=True, sampler=optuna.samplers.TPESampler(seed=SEED_SEARCH))
    done = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)
    print(f"optuna {done}/{N_TRIALS} done  -> {STUDY_NAME}")
    # ---- warm-up: r0_seed32 / struct20 を初期trialに(空study時。FORCE_WARMUP=1で既存studyにも) ----
    _wq = dict(wu.snap_to_space(wu.R0_PARAMS, wu.parse_engine_space()))
    if "self" not in TW_SOURCE: _wq.pop("self_mix_w", None)
    if "nbr" not in TW_SOURCE: _wq.pop("nbr_mix_w", None)
    _wq2 = dict(wu.snap_to_space(wu.ST20_PARAMS, wu.parse_engine_space()))
    if "self" not in TW_SOURCE: _wq2.pop("self_mix_w", None)
    if "nbr" not in TW_SOURCE: _wq2.pop("nbr_mix_w", None)
    if len(study.trials) == 0 or os.environ.get("FORCE_WARMUP", "0") == "1":
        study.enqueue_trial(_wq, skip_if_exists=True)
        study.enqueue_trial(_wq2, skip_if_exists=True)
        print(f"[warmup] r0_seed32 + struct20 をenqueue", flush=True)

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
        print(f"   [v65m] {nd}/{N_TRIALS}  this={tv:.4f}  best={bv:.4f}", flush=True)

    if done < N_TRIALS:
        study.optimize(objective, n_trials=N_TRIALS - done, callbacks=[cb])
    bp = pf.full_params(study.best_params)
    print(f"\n[v65m] best overall(探索seed, in-sample) = {study.best_value:.4f}")
    print("best_params:", json.dumps({k: (round(v, 6) if isinstance(v, float) else v) for k, v in bp.items()}))

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
