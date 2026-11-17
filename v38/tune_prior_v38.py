"""
v38p prior注入方式バリエーションのチューナー
=============================================
iterative_pf_prior_v38.py の PMODE(row/gate/trob/guide/jump)ごとに
PFパラメータを Optuna 探索する。方式ごとに study 別管理。

- 注入の強さはOptunaで振らない(v38原則)。外側定数=V38_MULT/V38_TNU/V38_GLEN。
- ★warm-start: study が空なら現最良(v38_anchor_mult20_current.json の best_params)を
  初期トライアルとして enqueue(注入方式が変わってもPF側の良い領域は近いはず)。
- 最良パラメータで 注入ON/OFF × overall/硬い核 を測って表示(従来と同じ)。
- ★報告・比較は必ず別seed(チューニングseed値は+0.2程度甘く出る。docs/v38.md プロトコル)。

実行例:
  $env:V38_PMODE="gate";  $env:V38_MULT="10"; python v38/tune_prior_v38.py
  $env:V38_PMODE="trob";  $env:V38_MULT="10"; python v38/tune_prior_v38.py
  $env:V38_PMODE="guide"; $env:V38_GLEN="1500"; python v38/tune_prior_v38.py
  $env:V38_PMODE="jump";  python v38/tune_prior_v38.py
再開: optuna.db に貯まるので同じコマンドで続行。
"""
import os, sys, json, time, math
from pathlib import Path
import numpy as np
import optuna
import iterative_pf_prior_v38 as pf

# ============================== CONFIG ==============================
N_TRIALS     = 300
SEARCH_SEEDS = 32
EVAL_SEEDS   = 32
SEARCH_CHUNK = 240
EVAL_CHUNK   = 120
SEED_SEARCH  = 20250707
_tag = {"row":   f"row_m{pf.ANCHOR_MULT:g}",
        "gate":  f"gate_m{pf.ANCHOR_MULT:g}",
        "trob":  f"trob_m{pf.ANCHOR_MULT:g}_nu{pf.TNU:g}",
        "guide": f"guide_L{pf.GUIDE_L:g}",
        "jump":  "jump"}[pf.PMODE]
STUDY_NAME = f"v38p_{pf.PRIOR_NAME}_{_tag}"
N_WELLS      = None          # None=全well, 整数で先頭N(デバッグ)
HARDCORE_CSV = pf.PROJECT / "v32" / "artifacts_v32" / "hardcore_diag.csv"
WARM_JSON    = pf.OUT_DIR / "v38_anchor_mult20_current.json"   # ★現最良(struct row mult30)
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
    print(f"PRIOR注入: PMODE={pf.PMODE}  MULT={pf.ANCHOR_MULT}  TNU={pf.TNU}  GLEN={pf.GUIDE_L}"
          f"  (sig=較正済 LOO誤差)  study={STUDY_NAME}")

    study = optuna.create_study(direction="minimize", study_name=STUDY_NAME, storage=STORAGE,
                                load_if_exists=True, sampler=optuna.samplers.TPESampler(seed=SEED_SEARCH))
    done = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)
    # ★warm-start: 空studyなら現最良paramsを初期トライアルへ(use_physは suggest 対象外なので除去)
    if len(study.trials) == 0 and WARM_JSON.exists():
        wp = json.loads(WARM_JSON.read_text())["best_params"]
        wp = {k: v for k, v in wp.items() if k != "use_phys"}
        study.enqueue_trial(wp)
        print(f"warm-start: enqueue {WARM_JSON.name} の best_params")
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
        print(f"   [v38p] {nd}/{N_TRIALS}  this={tv:.4f}  best={bv:.4f}", flush=True)

    if done < N_TRIALS:
        study.optimize(objective, n_trials=N_TRIALS - done, callbacks=[cb])
    bp = pf.full_params(study.best_params)
    print(f"\n[v38p:{pf.PMODE}] best overall(注入ON) on all wells = {study.best_value:.4f}")

    # ---- 注入 ON/OFF × overall/硬い核 で効きを可視化 ----
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
    print("\n=== best params: 注入 ON vs OFF ===")
    pf.USE_ANCHOR = True;  on_ov, on_hd = rep(f"{pf.PMODE} ON ")
    pf.USE_ANCHOR = False; of_ov, of_hd = rep(f"{pf.PMODE} OFF")
    pf.USE_ANCHOR = True

    OUT_BEST.write_text(json.dumps(dict(study=STUDY_NAME, pmode=pf.PMODE,
                                        anchor_mult=pf.ANCHOR_MULT, tnu=pf.TNU, guide_L=pf.GUIDE_L,
                                        best_value_on_all=float(study.best_value),
                                        overall_on=on_ov, overall_off=of_ov,
                                        hardcore_on=on_hd, hardcore_off=of_hd,
                                        params=bp), indent=2))
    print(f"\nsaved -> {OUT_BEST}")


if __name__ == "__main__":
    main()
