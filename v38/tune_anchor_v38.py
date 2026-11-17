"""
v38 構造錨つきPFのチューナー
============================
全wellの overall RMSE を最小化するように PFパラメータを Optuna 探索する。
錨(構造prior)は ON 固定・強度は sig_struct(較正済) × ANCHOR_MULT(定数)。
→ Optunaは尤度/ダイナミクスだけを調整し、錨の強さには触れない(強度が潰れる問題を回避)。

比較のため、最良パラメータで
  (a) 錨ON / (b) 錨OFF を、overall と 硬い核(v32 hardcore_diag)で両方測って表示する。
錨の価値は tail に集中するので、overall だけでなく硬い核RMSEで効きを確認すること。

実行 : rogii_claude/.venv/Scripts/python.exe v38/tune_anchor_v38.py
再開 : optuna.db(v38/artifacts_v38) に貯まるので同じコマンドで続行。
mult : ANCHOR_MULT を変えて数点(例 0, 1.0, 1.5)回すのは v38/iterative_pf_boost_v38.py の定数を編集。
       ANCHOR_MULT=0 は錨OFFと等価(比較の下限)。
"""
import os, sys, json, time, math
from pathlib import Path
import numpy as np
import optuna
import iterative_pf_boost_v38 as pf

# ============================== CONFIG ==============================
N_TRIALS     = 300
SEARCH_SEEDS = 32
EVAL_SEEDS   = 32
SEARCH_CHUNK = 240
EVAL_CHUNK   = 120
SEED_SEARCH  = 20250707
_base = (f"v38_anchor_mult{pf.ANCHOR_MULT:g}" if pf.PRIOR_NAME == "struct"
         else f"v38_{pf.PRIOR_NAME}_mult{pf.ANCHOR_MULT:g}")          # priorごとに別study(旧名と互換)
STUDY_NAME = _base if pf.ANCHOR_MODE == "row" else f"v38_{pf.PRIOR_NAME}_wm{pf.BIAS_MULT:g}"   # ⑤=wellmean
N_WELLS      = None          # None=全well, 整数で先頭N(デバッグ)
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
        print("ERROR: 構造prior未生成。先に  python v38/struct_prior_v38.py  を実行。"); return

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
    print(f"usable wells: {len(ALL)}  (構造錨あり {n_anch})")
    print(f"ANCHOR: USE={pf.USE_ANCHOR}  MULT={pf.ANCHOR_MULT}  (sig=較正済 struct LOO誤差)")

    study = optuna.create_study(direction="minimize", study_name=STUDY_NAME, storage=STORAGE,
                                load_if_exists=True, sampler=optuna.samplers.TPESampler(seed=SEED_SEARCH))
    done = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)
    print(f"optuna {done}/{N_TRIALS} done")

    def objective(trial):
        P = pf.suggest_params(trial)
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
        CUR_PATH.write_text(json.dumps(dict(study=STUDY_NAME, trials_done=nd,
                                            best_value=bv, best_params=bp), indent=2))
        tv = trial.value if trial.value is not None else float("nan")
        print(f"   [v38] {nd}/{N_TRIALS}  this={tv:.4f}  best={bv:.4f}", flush=True)

    if done < N_TRIALS:
        study.optimize(objective, n_trials=N_TRIALS - done, callbacks=[cb])
    bp = pf.full_params(study.best_params)
    print(f"\n[v38] best overall(anchor ON) on all wells = {study.best_value:.4f}")

    # ---- 錨 ON/OFF × overall/硬い核 で効きを可視化 ----
    hard = load_hardcore() & set(ALL)
    hard = [w for w in ALL if w in hard]
    def rep(tag):
        e_all = pf.predict_errs(bp, ALL, SEED_SEARCH, EVAL_SEEDS, EVAL_CHUNK)
        ov = np.concatenate([e_all[w] for w in ALL]); ov = ov[np.isfinite(ov)]
        hv = np.concatenate([e_all[w] for w in hard]) if hard else np.array([np.nan])
        hv = hv[np.isfinite(hv)]
        print(f"  [{tag}] overall={np.sqrt(np.mean(ov**2)):.4f}  "
              f"hardcore({len(hard)})={np.sqrt(np.mean(hv**2)) if hv.size else float('nan'):.4f}")
        return float(np.sqrt(np.mean(ov**2))), float(np.sqrt(np.mean(hv**2))) if hv.size else None
    print("\n=== best params: 錨 ON vs OFF ===")
    pf.USE_ANCHOR = True;  on_ov, on_hd = rep("anchor ON ")
    pf.USE_ANCHOR = False; of_ov, of_hd = rep("anchor OFF")
    pf.USE_ANCHOR = True

    OUT_BEST.write_text(json.dumps(dict(study=STUDY_NAME, anchor_mult=pf.ANCHOR_MULT,
                                        best_value_on_all=float(study.best_value),
                                        overall_on=on_ov, overall_off=of_ov,
                                        hardcore_on=on_hd, hardcore_off=of_hd,
                                        params=bp), indent=2))
    print(f"\nsaved -> {OUT_BEST}")


if __name__ == "__main__":
    main()
