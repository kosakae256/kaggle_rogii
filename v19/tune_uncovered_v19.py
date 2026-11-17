"""
v19 未カバー坑井ねらい撃ちチューナー
=====================================
「**ここまでの全PF(バンク + 任意でv9)のいずれでも RMSE>THRESH な坑井**(=バンクの oracle 残差)」
だけを対象に Optuna でPFパラメータ探索する。boosting の「直前PFのワースト200」と違い、
**バンク全体がカバーできない硬い核**を直接狙う(set-cover的)。

- PFコア/探索空間/評価は iterative_pf_boost_v19 を再利用。
- 各バンクPFの per-well RMSE を計算(キャッシュ, 再開可) → oracle(各wellの最小) → 対象=oracle>THRESH。
- 対象wellの overall RMSE を最小化(探索SEARCH_SEEDS)。最良を保存。
- 追加後に「>10 が何本減るか/対象を何本カバーできたか」を表示。

実行: rogii_claude/.venv/Scripts/python.exe v19/tune_uncovered_v19.py
"""

import os, sys, json, time, pickle, math
from pathlib import Path
import numpy as np
import optuna
import iterative_pf_boost_v19 as pf   # PFコア/評価/探索空間を再利用

# ============================== CONFIG ==============================
INCLUDE_V9   = True          # v9 pf_1/2/3(物理OFF)もバンクに含めて oracle を取る
THRESH       = 10.0          # oracle>THRESH の well を「未カバー」として対象に
USE_WORST_K  = None          # 整数を入れると oracle最悪K well を対象(THRESHより優先)
N_TRIALS     = 300
SEARCH_SEEDS = 32             # 探索(objective)
EVAL_SEEDS   = 32            # バンクper-well評価 & 追加後評価(本番一致)
SEARCH_CHUNK = 400
EVAL_CHUNK   = 400
SEED_SEARCH  = 12345         # この探索のPF/サンプラーseed(既存ラウンドと別)
STUDY_NAME   = "uncovered0"  # 連番にすれば複数本ねらい撃ち可

OUT        = pf.OUT_DIR
STORAGE    = f"sqlite:///{(OUT / 'optuna.db').as_posix()}"
BANK_PATH  = OUT / "pf_bank.json"
BANK_CACHE = OUT / "uncovered_bank_cache.pkl"   # バンク各PFの per-well RMSE
CUR_PATH   = OUT / "uncovered_current.json"
OUT_BEST   = OUT / f"{STUDY_NAME}_best.json"
# ===================================================================

optuna.logging.set_verbosity(optuna.logging.WARNING)


def bank_items():
    """(label, seed, raw_params) のリスト = v19バンク + (任意で)v9 pf_1/2/3(物理OFF)。"""
    items = []
    bank = json.loads(BANK_PATH.read_text())
    for e in bank:
        items.append((f"r{e['round']}", e.get("seed", 0), e["params"]))
    if INCLUDE_V9:
        sys.path.append(str(pf.PROJECT / "v9"))
        from pf_gpu_run import PF_PARAM_SETS_RAW
        for name, raw in PF_PARAM_SETS_RAW.items():
            r = dict(raw); r["use_phys"] = False; r.setdefault("phys_sig", 1.0)
            items.append((f"v9_{name}", 0, r))
    return items


def main():
    if pf.DEVICE != "cuda":
        print("WARNING: CUDA 無し。遅いです。")
    # 全wellのキャッシュを pf に渡す(pf.predict系は pf.CACHE を参照)
    wells = pf.list_wells()
    cache = {}
    for w in wells:
        c = pf.build_cache(w)
        if c is not None and c["tvt_true"] is not None:
            cache[w] = c
    pf.CACHE = cache
    ALL = [w for w in wells if w in cache]
    print(f"usable wells: {len(ALL)}")

    # ---- バンク各PFの per-well RMSE(キャッシュ, 再開可) ----
    store = pickle.loads(BANK_CACHE.read_bytes()) if BANK_CACHE.exists() else {}
    items = bank_items()
    print(f"バンク {len(items)} 本 (INCLUDE_V9={INCLUDE_V9})")
    rmat = []
    for label, seed, params in items:
        key = f"{label}_n{EVAL_SEEDS}"
        if key not in store:
            t0 = time.time()
            store[key] = pf.perwell_rmse(params, ALL, seed, EVAL_SEEDS, EVAL_CHUNK)
            BANK_CACHE.write_bytes(pickle.dumps(store))
            print(f"  bank {key}: done ({time.time()-t0:.0f}s)", flush=True)
        rmat.append([store[key][w] for w in ALL])
    rmat = np.array(rmat)                 # [nPF, nW]
    oracle = rmat.min(0)                  # 各wellのバンクoracle

    # ---- 未カバー対象 ----
    if USE_WORST_K:
        idx = np.argsort(oracle)[::-1][:USE_WORST_K]
    else:
        idx = np.where(oracle > THRESH)[0]
    target = [ALL[int(i)] for i in idx]
    print(f"\n対象(いずれのPFでもカバー不可) = {len(target)} well "
          f"({'worstK='+str(USE_WORST_K) if USE_WORST_K else 'oracle>'+str(THRESH)})")
    if not target:
        print("対象が0。THRESHを下げるか USE_WORST_K を使ってください。"); return
    ot = oracle[idx]
    print("対象の oracle rmse 分位:", {f"p{q}": round(float(np.percentile(ot, q)), 1) for q in [50, 90, 100]})

    # ---- Optuna: 対象wellの overall RMSE 最小化 ----
    study = optuna.create_study(direction="minimize", study_name=STUDY_NAME, storage=STORAGE,
                                load_if_exists=True, sampler=optuna.samplers.TPESampler(seed=SEED_SEARCH))
    done = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)
    print(f"optuna {done}/{N_TRIALS} done")

    def objective(trial):
        P = pf.suggest_params(trial)
        try:
            r = pf.overall_rmse_on(P, target, SEED_SEARCH, SEARCH_SEEDS, SEARCH_CHUNK)
        except Exception:
            return 1e3
        return r if math.isfinite(r) else 1e3

    def cb(study, trial):
        try:
            bv = study.best_value; bp = pf.full_params(study.best_params)
        except Exception:
            return
        nd = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)
        CUR_PATH.write_text(json.dumps(dict(study=STUDY_NAME, target_n=len(target),
                                            trials_done=nd, best_value=bv, best_params=bp), indent=2))
        tv = trial.value if trial.value is not None else float("nan")
        print(f"   [uncov] {nd}/{N_TRIALS}  this={tv:.4f}  best={bv:.4f}", flush=True)

    if done < N_TRIALS:
        study.optimize(objective, n_trials=N_TRIALS - done, callbacks=[cb])
    bp = pf.full_params(study.best_params)
    print(f"\n[uncov] best on target = {study.best_value:.4f}")

    # ---- 追加後の効果(本番EVAL_SEEDSで全well推論) ----
    pw_new = pf.perwell_rmse(bp, ALL, SEED_SEARCH, EVAL_SEEDS, EVAL_CHUNK)
    new_row = np.array([pw_new[w] for w in ALL])
    new_oracle = np.minimum(oracle, new_row)
    before10 = int((oracle >= 10).sum()); after10 = int((new_oracle >= 10).sum())
    covered = int((new_row[idx] < 10).sum())   # 対象のうち新PFが<10にできた数
    print(f"対象 {len(target)} のうち 新PFが rmse<10 にできた: {covered} 本")
    print(f"バンク全体 >10: {before10} -> {after10}  (この新PFを足すと {before10-after10} 本カバー)")
    print(f"新PF単体の対象内 overall(EVAL)= {float(np.sqrt(np.mean(new_row[idx]**2))):.3f}")

    OUT_BEST.write_text(json.dumps(dict(label=STUDY_NAME, seed=SEED_SEARCH, target_n=len(target),
                                        best_value_on_target=float(study.best_value),
                                        covered_target=covered, bank_over10_before=before10,
                                        bank_over10_after=after10, params=bp), indent=2))
    print(f"\nsaved -> {OUT_BEST}")
    print("採用するなら pf_bank.json に1エントリ追記(round=次番号, seed=SEED_SEARCH, params=これ)するか、別バンクとして管理。")


if __name__ == "__main__":
    main()
