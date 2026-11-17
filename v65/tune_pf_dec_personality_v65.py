# -*- coding: utf-8 -*-
"""
v65 性格別PFチューナー(間引き/MD粗化版) — 性格リストを一つずつ順番に回す(寝てる間に全性格)
=====================================================================================
tune_pf_personality_v65 の PERSONAS/BASE(単一ソース)を再利用し、目的関数を decimation
(pf_dec_v65.overall_rmse_on_dec, MD=DEC集約→全解像度に補間して採点)に差し替えた版。
既存 tune_pf_personality_v65 / pf_engine_v65 は無改造。**別 study・別 db(optuna_dec.db)**。

★複数性格を PERSONAS(カンマ区切り)で指定 → **1つずつ順番に**フル探索(前が終わってから次)。
  各性格は optuna で再開可能(途中で止めても続きから)。study 名 = v65pdec{DEC}_{性格}_{prior}[_{tw}]。

実行:
  # 全8性格を順番に(既定=全部)。寝てる間に一晩でどうぞ。
  N_TRIALS=200 .venv/Scripts/python.exe v65/tune_pf_dec_personality_v65.py
  # 一部だけ:
  PERSONAS=smooth_tracker,robust_skeptic,dip_holder .venv/Scripts/python.exe v65/tune_pf_dec_personality_v65.py
env: DEC(10) AGG(mean) TW_SOURCE(tw) N_TRIALS(200) SEARCH_SEEDS(32) EVAL_SEEDS(32)
     SEARCH_CHUNK(240) EVAL_CHUNK(120) N_WELLS(0=全) V38_PRIOR(既定=v66_leanv1cumul)
     PERSONAS(既定=全性格) FORCE_WARMUP(0)
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
from tune_pf_personality_v65 import PERSONAS, BASE          # ★性格定義は単一ソース(データのみ利用)

# ============================== CONFIG ==============================
TW_SOURCE    = os.environ.get("TW_SOURCE", "tw")
os.environ["V65_TWSRC"] = TW_SOURCE
N_TRIALS     = int(os.environ.get("N_TRIALS", "300"))
SEARCH_SEEDS = int(os.environ.get("SEARCH_SEEDS", "32"))
EVAL_SEEDS   = int(os.environ.get("EVAL_SEEDS", "32"))
SEARCH_CHUNK = int(os.environ.get("SEARCH_CHUNK", "240"))
EVAL_CHUNK   = int(os.environ.get("EVAL_CHUNK", "120"))
N_WELLS      = int(os.environ.get("N_WELLS", "0")) or None
SEED_SEARCH  = 20250707
SEED_ALT     = SEED_SEARCH + 1000
HARDCORE_CSV = pf.PROJECT / "v32" / "artifacts_v32" / "hardcore_diag.csv"
DEFAULT_ORDER = ["smooth_tracker", "jumpy_explorer", "anchor_faithful", "anchor_loose",
                 "sharp_matcher", "robust_skeptic", "dip_holder", "wide_net"]
PERSONAS_TO_RUN = [p.strip() for p in os.environ.get(
    "PERSONAS", ",".join([p for p in DEFAULT_ORDER if p in PERSONAS])).split(",") if p.strip()]
# ===================================================================

OUT     = pf.OUT_DIR
STORAGE = f"sqlite:///{(OUT / 'optuna_dec.db').as_posix()}"     # 別db=current v65 非干渉


def load_hardcore():
    try:
        import pandas as pd
        d = pd.read_csv(HARDCORE_CSV)
        return set(d[d["hard"] == True].iloc[:, 0].astype(str))
    except Exception:
        return set()


def build_space(persona):
    space = {**BASE, **persona["overrides"]}
    if not persona["use_anchor"]:
        space.pop("anchor_mult", None)          # 錨なし性格は錨系を探索しない
    return space


def suggest(trial, space, persona):
    P = {}
    for k, spec in space.items():
        if spec[0] == "pin":
            P[k] = spec[1]
        elif spec[0] == "cat":
            P[k] = trial.suggest_categorical(k, spec[1])
        else:
            P[k] = trial.suggest_float(k, spec[1], spec[2], log=spec[3])
    if "self" in TW_SOURCE:
        P["self_mix_w"] = trial.suggest_float("self_mix_w", 0.1, 1.0)
    if "nbr" in TW_SOURCE:
        P["nbr_mix_w"] = trial.suggest_float("nbr_mix_w", 0.1, 1.0)
    P["use_anchor"] = bool(persona["use_anchor"])   # 宣言制
    P["use_phys"] = False
    return pf.full_params(P)


def run_persona(pkey, ALL, hard):
    persona = PERSONAS[pkey]
    space = build_space(persona)
    pinned = {k: v[1] for k, v in space.items() if v[0] == "pin"}
    sname = f"v65pdec{dec.DEC}_{pkey}_{pf.PRIOR_NAME}" + ("" if TW_SOURCE == "tw" else f"_{TW_SOURCE.replace('+', '_')}")
    cur = OUT / f"{sname}_current.json"; best = OUT / f"{sname}_best.json"
    print(f"\n{'='*66}\n[性格 {pkey}] use_anchor={persona['use_anchor']} DEC={dec.DEC} study={sname}")
    print(f"  制約: " + json.dumps({k: (v[1] if v[0] != 'f' else [v[1], v[2]]) for k, v in persona['overrides'].items()}, ensure_ascii=False), flush=True)

    study = optuna.create_study(direction="minimize", study_name=sname, storage=STORAGE,
                                load_if_exists=True, sampler=optuna.samplers.TPESampler(seed=SEED_SEARCH))
    done = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)
    # warm-up: r0 を性格制約へスナップ(空study or FORCE_WARMUP)
    _spec = {k: v for k, v in space.items() if v[0] != "pin"}
    if "self" in TW_SOURCE: _spec["self_mix_w"] = ("f", 0.1, 1.0, False)
    if "nbr" in TW_SOURCE: _spec["nbr_mix_w"] = ("f", 0.1, 1.0, False)
    if len(study.trials) == 0 or os.environ.get("FORCE_WARMUP", "0") == "1":
        study.enqueue_trial(wu.snap_to_space(wu.R0_PARAMS, _spec), skip_if_exists=True)

    def full_of(bp_):
        P = pf.full_params({**pinned, **bp_}); P["use_anchor"] = bool(persona["use_anchor"]); return P

    def objective(trial):
        P = suggest(trial, space, persona)
        try:
            r = dec.overall_rmse_on_dec(P, ALL, SEED_SEARCH, SEARCH_SEEDS, SEARCH_CHUNK)
        except Exception:
            return 1e3
        return r if math.isfinite(r) else 1e3

    def cb(study, trial):
        try:
            bv = study.best_value; bp = full_of(study.best_params)
        except Exception:
            return
        nd = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)
        cur.write_text(json.dumps(dict(study=sname, personality=pkey, dec=dec.DEC, tw_source=TW_SOURCE,
                                       use_anchor=persona["use_anchor"], trials_done=nd,
                                       best_value=bv, best_params=bp), indent=2))
        tv = trial.value if trial.value is not None else float("nan")
        print(f"   [{pkey} dec{dec.DEC}] {nd}/{N_TRIALS}  this={tv:.4f}  best={bv:.4f}", flush=True)

    if done < N_TRIALS:
        study.optimize(objective, n_trials=N_TRIALS - done, callbacks=[cb])
    bp = full_of(study.best_params)
    # 別seed評価(採用判断は別seedで)
    def rep(tag, P, seed):
        e = dec.predict_errs_dec(P, ALL, seed, EVAL_SEEDS, EVAL_CHUNK)
        ov = np.concatenate([e[w] for w in ALL]); ov = ov[np.isfinite(ov)]
        hv = np.concatenate([e[w] for w in hard]) if hard else np.array([]); hv = hv[np.isfinite(hv)]
        o = float(np.sqrt(np.mean(ov ** 2))); h = float(np.sqrt(np.mean(hv ** 2))) if hv.size else None
        print(f"    [{tag}] overall={o:.4f} hardcore({len(hard)})={h if h else float('nan'):.4f}")
        return o, h
    print(f"[性格 {pkey}] 完了。探索best={study.best_value:.4f}")
    on, onh = rep("探索seed", bp, SEED_SEARCH); alt, alth = rep("別seed", bp, SEED_ALT)
    best.write_text(json.dumps(dict(study=sname, personality=pkey, dec=dec.DEC, tw_source=TW_SOURCE,
                                    use_anchor=persona["use_anchor"], best_value_search=float(study.best_value),
                                    overall_search_seed=on, overall_alt_seed=alt,
                                    hardcore_search_seed=onh, hardcore_alt_seed=alth, params=bp), indent=2))
    print(f"  saved -> {best.name}", flush=True)
    return alt


def main():
    if pf.DEVICE != "cuda":
        print("WARNING: CUDA 無し。遅いです。")
    bad = [p for p in PERSONAS_TO_RUN if p not in PERSONAS]
    if bad:
        print(f"★不明な性格 {bad}。選べるのは {list(PERSONAS)}"); return
    print(f"[dec性格] DEC={dec.DEC} AGG={dec.AGG} prior={pf.PRIOR_NAME}  N_TRIALS={N_TRIALS}/性格")
    print(f"[dec性格] 順番に回す {len(PERSONAS_TO_RUN)}性格: {PERSONAS_TO_RUN}")

    wells = pf.list_wells()
    if N_WELLS:
        wells = wells[:N_WELLS]
    cache = {}
    for w in wells:
        c = pf.build_cache(w)
        if c is not None and c["tvt_true"] is not None:
            cache[w] = c
    pf.CACHE = cache
    ALL = [w for w in wells if w in cache]
    hard = [w for w in ALL if w in (load_hardcore() & set(ALL))]
    print(f"usable {len(ALL)} 坑井(hardcore {len(hard)})  → MD{dec.DEC}集約でPF\n")

    results = {}
    for i, pkey in enumerate(PERSONAS_TO_RUN, 1):
        print(f"\n########## [{i}/{len(PERSONAS_TO_RUN)}] {pkey} ##########", flush=True)
        try:
            results[pkey] = run_persona(pkey, ALL, hard)
        except Exception as e:
            print(f"★[{pkey}] 失敗: {type(e).__name__}: {e}(次の性格へ)", flush=True)
    print(f"\n{'='*66}\n[dec性格] 全{len(PERSONAS_TO_RUN)}性格 完了。別seed overall 一覧:")
    for k, v in sorted(results.items(), key=lambda kv: kv[1]):
        print(f"  {k:16s} {v:.4f}")


if __name__ == "__main__":
    main()
