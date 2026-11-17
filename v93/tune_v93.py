# -*- coding: utf-8 -*-
"""
v93 PFチューナ — GRフリー錨(位置錨)PF を optuna で最適化。
================================================================================
★全井戸(773)に対して1つの study で最適化する(5分割CVではない)。
  objective = pf_engine_v93.overall_rmse_on(P, ALL_WELLS)  ← 全井 overall RMSE を最小化。
  warmup: struct20(v93/artifacts_v93/struct20_init_params.json)を初期trialに enqueue。
          anchor_mult 含め全paramを GRフリー錨用に再探索(STEP1でmult20は強すぎと判明)。
出力: v93/artifacts_v93/v93_all_current.json / v93_all_best.json、study=optuna_v93.db(v93_all)。
env: N_TRIALS(200) / N_SEEDS(32) / CHUNK(60) / SUBSET(0=全井, N=先頭N井で軽く試す)
使い方: N_TRIALS=200 .venv/Scripts/python.exe v93/tune_v93.py
        SUBSET=60 N_TRIALS=30 .venv/Scripts/python.exe v93/tune_v93.py   # 軽くスモーク
"""
import os, sys, io, json
try: sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
except Exception: pass
import optuna
from pathlib import Path
import pf_engine_v93 as pf   # ★v93自身のengine(他バージョンimportしない)

optuna.logging.set_verbosity(optuna.logging.WARNING)
ART = pf.ART
N_TRIALS = int(os.environ.get("N_TRIALS", "200"))
N_SEEDS  = int(os.environ.get("N_SEEDS", "32"))
CHUNK    = int(os.environ.get("CHUNK", "60"))
SUBSET   = int(os.environ.get("SUBSET", "0"))          # >0: 先頭N井だけで軽く試す
STORAGE  = f"sqlite:///{(ART / 'optuna_v93.db').as_posix()}"

# ---- 全井キャッシュ ----
WELLS = [w for w in pf.list_wells() if pf.CACHE.setdefault(w, pf.build_cache(w)) is not None]
WELLS = [w for w in WELLS if pf.CACHE[w] is not None]
if SUBSET > 0:
    WELLS = WELLS[:SUBSET]
print(f"[v93 tune] 全井最適化  対象井={len(WELLS)}  N_TRIALS={N_TRIALS} N_SEEDS={N_SEEDS}", flush=True)

# ---- struct20 初期値(探索空間のキーだけ抽出して enqueue) ----
_ST20 = json.loads((ART / "struct20_init_params.json").read_text())
SPACE_KEYS = ["n_particles","likelihood_scale","init_pos_std","init_rate_std","mom","vn","pn","rp","rr",
              "resamp","gr_sig_min","gr_sig_max","gr_sig_def","gr_sig_mult","tvt_clip_margin","rate_clip",
              "gr_power","hgr_smooth_r","tw_gr_smooth_r","jump_prob","jump_std",
              "use_anchor","anchor_mult","anchor_ramp_md","lk_floor","gr_debias","rate_target","grid_step"]
def st20_enqueue():
    return {k: _ST20[k] for k in SPACE_KEYS if k in _ST20}

# ---- objective: 全井 overall RMSE ----
def objective(trial):
    P = pf.full_params(pf.suggest_params(trial))
    return pf.overall_rmse_on(P, WELLS, seed=99999 + trial.number, n_seeds=N_SEEDS, chunk=CHUNK)

if __name__ == "__main__":
    study = optuna.create_study(direction="minimize", study_name="v93_all", storage=STORAGE,
                                load_if_exists=True, sampler=optuna.samplers.TPESampler(seed=0))
    if len(study.trials) == 0 or os.environ.get("FORCE_WARMUP", "0") == "1":
        try:
            study.enqueue_trial(st20_enqueue(), skip_if_exists=True)
            print("  struct20 を初期trialに enqueue", flush=True)
        except Exception as e:
            print(f"  warmup enqueue失敗(無視): {e}", flush=True)
    done0 = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)
    for _ in range(max(0, N_TRIALS - done0)):
        study.optimize(objective, n_trials=1, catch=(Exception,))
        b = study.best_trial
        (ART / "v93_all_current.json").write_text(json.dumps(dict(
            n_wells=len(WELLS), n_trials=len(study.trials),
            best_value=b.value, best_params=b.params), indent=1))
        print(f"  trial {len(study.trials)}: best RMSE={study.best_value:.4f} "
              f"(anchor_mult={study.best_params.get('anchor_mult'):.2f} use_anchor={study.best_params.get('use_anchor')})",
              flush=True)
    (ART / "v93_all_best.json").write_text(json.dumps(dict(
        n_wells=len(WELLS), best_value=study.best_value,
        best_params=pf.full_params(study.best_params)), indent=1))
    print(f"\n[v93 all-well] 全{len(WELLS)}井 best RMSE = {study.best_value:.4f}  (trials {len(study.trials)})", flush=True)
    print(f"  → v93/artifacts_v93/v93_all_best.json", flush=True)
