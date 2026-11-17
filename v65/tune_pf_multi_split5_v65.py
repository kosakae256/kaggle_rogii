# -*- coding: utf-8 -*-
"""
v65 合成尤度PFチューナー(split5版) — tw/self/nbr を"積"で合成する PF を、train5分割・分割ごと最適化
=====================================================================================
発端(ユーザー 2026-07-16): tw/self/nbr を組み合わせる合成尤度PF(pf_engine_multi_v65)を調整したい。
  ただし **全773で調整すると LB-CV 転移が起きない → split5(分割ごと最適化)で調整する**(転移が出やすい経験則)。
= tune_pf_split5_v65 の split5機構 × tune_pf_multi_v65 の合成尤度エンジン。

方針:
  - train 全773坑井を **ランダムに N_SPLITS(=5) 分割**(シード SPLIT_SEED で再現可能)。
  - **各分割を別々に最適化**(= N_SPLITS 個の Optuna study。各 study はその分割の overall RMSE を最小化)。
  - **エンジン=合成尤度**: 各行 emission を tw/self/nbr の積 Π_s exp(-0.5·β_s·((GR-eg_s)/gs)^p)。
    self_mix_w/nbr_mix_w は「emission指数β(信頼度)」として探索(pf_engine_v65.suggest_params 流用)。
    近傍<2本(MAX_DIST=1500)の坑井は nbr尤度=中立(=1)。GR照合のみ=struct任意で転移安全。
  - **錨は ANCHOR で切替**: off(既定=合成尤度は元々転移安全なので錨なし) / on(v66錨 強制) / search(探索)。
  - **間引きなし(MD-1 = pf.overall_rmse_on)**。報告規律は tune_pf_v65 と同じ(採用判断は別seed +1000 で)。

出力:
  - 分割定義: artifacts_v65/split5_s{SPLIT_SEED}.json  (well -> group。split5_v65 と同一シードなら同一分割=共有)
  - 各分割: v65msplit5_s{SPLIT_SEED}_g{g}_{prior}_{tw}[_anc]_current.json / _best.json
  別db(optuna_multi_split5.db)なので既存 study に非干渉。
実行:
  # 全5分割を順番に(既定=tw+self+nbr, 錨なし):
  N_TRIALS=200 .venv/Scripts/python.exe v65/tune_pf_multi_split5_v65.py
  # 一部の分割だけ(★env名は SPLIT_GROUPS):
  SPLIT_GROUPS=0,1 .venv/Scripts/python.exe v65/tune_pf_multi_split5_v65.py
env: TW_SOURCE(tw+self+nbr) ANCHOR(off|on|search) SPLIT_SEED(43) N_SPLITS(5) SPLIT_GROUPS(既定=全) N_TRIALS(200)
     V38_PRIOR(v66_leanv1cumul=v66錨) SEARCH_SEEDS(32) EVAL_SEEDS(32) SEARCH_CHUNK(240) EVAL_CHUNK(120) N_WELLS(0=全) FULL_EVAL(1)
"""
import os, json, math
from pathlib import Path
import numpy as np
import optuna
import sys
sys.stdout = open(1, "w", encoding="utf-8", closefd=False)
sys.path.insert(0, str(Path(__file__).parent))
import pf_engine_multi_v65 as pf        # ★合成尤度エンジン(tw/self/nbr を積で合成)
import warmup_v65 as wu

# ============================== CONFIG ==============================
TW_SOURCE    = os.environ.get("TW_SOURCE", "tw+self+nbr")    # ★合成尤度は self/nbr 込みが既定
os.environ["V65_TWSRC"] = TW_SOURCE                           # build_cache(nbr)/suggest に伝播
ANCHOR_MODE  = os.environ.get("ANCHOR", "off").lower()       # ★off(既定=錨なし=合成尤度は転移安全) | on | search
assert ANCHOR_MODE in ("off", "on", "search"), "ANCHOR は off|on|search"
SPLIT_SEED   = int(os.environ.get("SPLIT_SEED", "4322"))       # ★分割の乱数シード(固定=再現可能。split5_v65 と共有可)
N_SPLITS     = int(os.environ.get("N_SPLITS", "5"))
N_TRIALS     = int(os.environ.get("N_TRIALS", "200"))
SEARCH_SEEDS = int(os.environ.get("SEARCH_SEEDS", "32"))
EVAL_SEEDS   = int(os.environ.get("EVAL_SEEDS", "32"))
SEARCH_CHUNK = int(os.environ.get("SEARCH_CHUNK", "240"))
EVAL_CHUNK   = int(os.environ.get("EVAL_CHUNK", "120"))
N_WELLS      = int(os.environ.get("N_WELLS", "0")) or None
FULL_EVAL    = os.environ.get("FULL_EVAL", "1") == "1"       # 各分割bestを全773でも評価(参考, 遅い)
SEED_SEARCH  = 20250715
SEED_ALT     = SEED_SEARCH + 1000
GROUPS_TO_RUN = [int(x) for x in os.environ.get("SPLIT_GROUPS", ",".join(map(str, range(N_SPLITS)))).split(",") if x.strip() != ""]   # ★GROUPSはbash予約なのでSPLIT_GROUPS
# ===================================================================

OUT     = pf.OUT_DIR
STORAGE = f"sqlite:///{(OUT / 'optuna_multi_split5.db').as_posix()}"    # ★別db=既存 study 非干渉
_TWTAG  = "" if TW_SOURCE == "tw" else f"_{TW_SOURCE.replace('+', '_')}"
_ANCTAG = {"off": "", "on": "_anc", "search": "_ancsrch"}[ANCHOR_MODE]


def apply_anchor(P):
    """ANCHOR_MODE に従い use_anchor を設定。off=強制False / on=強制True / search=suggest_paramsの選択を尊重。"""
    if ANCHOR_MODE == "off":   P["use_anchor"] = False
    elif ANCHOR_MODE == "on":  P["use_anchor"] = True
    P["use_phys"] = False
    return P


def suggest_P(trial):
    return apply_anchor(pf.full_params(pf.suggest_params(trial)))


def make_split(ALL):
    """ALL(全坑井)を SPLIT_SEED でランダムに N_SPLITS 分割(再現可能。split5_v65 と同一シード=同一分割)。"""
    rng = np.random.RandomState(SPLIT_SEED)
    idx = rng.permutation(len(ALL))
    parts = np.array_split(idx, N_SPLITS)
    groups = [[ALL[int(i)] for i in part] for part in parts]
    assign = {}
    for g, ws in enumerate(groups):
        for w in ws:
            assign[w] = g
    (OUT / f"split5_s{SPLIT_SEED}.json").write_text(json.dumps(dict(
        split_seed=SPLIT_SEED, n_splits=N_SPLITS, sizes=[len(g) for g in groups], assign=assign), indent=1))
    return groups


def run_group(g, wells_g, hard):
    sname = f"v65msplit5_s{SPLIT_SEED}_g{g}_{pf.PRIOR_NAME}{_TWTAG}{_ANCTAG}"
    cur = OUT / f"{sname}_current.json"; best = OUT / f"{sname}_best.json"
    print(f"\n{'='*66}\n[合成split5 {g}/{N_SPLITS-1}] wells={len(wells_g)}  study={sname}  (錨={ANCHOR_MODE} src={TW_SOURCE} / MD-1)")

    study = optuna.create_study(direction="minimize", study_name=sname, storage=STORAGE,
                                load_if_exists=True, sampler=optuna.samplers.TPESampler(seed=SEED_SEARCH))
    done = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)
    print(f"  optuna {done}/{N_TRIALS} done")
    # warm-up: r0_seed32 を初期trialに(self/nbr の混合比βは snap で既定)
    _wq = dict(wu.snap_to_space(wu.R0_PARAMS, wu.parse_engine_space()))
    if "self" not in TW_SOURCE: _wq.pop("self_mix_w", None)
    if "nbr" not in TW_SOURCE: _wq.pop("nbr_mix_w", None)
    if len(study.trials) == 0 or os.environ.get("FORCE_WARMUP", "0") == "1":
        study.enqueue_trial(_wq, skip_if_exists=True)
        print("  [warmup] r0_seed32をenqueue", flush=True)
        if ANCHOR_MODE != "off":
            _wq2 = dict(wu.snap_to_space(wu.ST20_PARAMS, wu.parse_engine_space()))
            if "self" not in TW_SOURCE: _wq2.pop("self_mix_w", None)
            if "nbr" not in TW_SOURCE: _wq2.pop("nbr_mix_w", None)
            study.enqueue_trial(_wq2, skip_if_exists=True)
            print("  [warmup] struct20(v66錨あり)をenqueue", flush=True)

    def objective(trial):
        P = suggest_P(trial)
        try:
            r = pf.overall_rmse_on(P, wells_g, SEED_SEARCH, SEARCH_SEEDS, SEARCH_CHUNK)   # ★その分割の overall(合成尤度, MD-1)
        except Exception:
            return 1e3
        return r if math.isfinite(r) else 1e3

    def full_of(bp_):
        return apply_anchor(pf.full_params(bp_))

    def cb(study, trial):
        try:
            bv = study.best_value; bp = full_of(study.best_params)
        except Exception:
            return
        nd = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)
        cur.write_text(json.dumps(dict(study=sname, split_seed=SPLIT_SEED, group=g, n_wells=len(wells_g),
                                       tw_source=TW_SOURCE, use_anchor=bool(bp.get("use_anchor", False)), trials_done=nd,
                                       best_value=bv, best_params=bp), indent=2))
        tv = trial.value if trial.value is not None else float("nan")
        print(f"   [g{g}] {nd}/{N_TRIALS}  this={tv:.4f}  best={bv:.4f}", flush=True)

    if done < N_TRIALS:
        study.optimize(objective, n_trials=N_TRIALS - done, callbacks=[cb])
    bp = full_of(study.best_params)
    print(f"[合成split5 {g}] 探索best(in-sample, 探索seed) = {study.best_value:.4f}")

    def rep(tag, wells, seed):
        e = pf.predict_errs(bp, wells, seed, EVAL_SEEDS, EVAL_CHUNK)
        ov = np.concatenate([e[w] for w in wells]); ov = ov[np.isfinite(ov)]
        return float(np.sqrt(np.mean(ov ** 2))) if ov.size else float("nan")

    ig_s = rep(f"分割{g}探索seed", wells_g, SEED_SEARCH)
    ig_a = rep(f"分割{g}別seed", wells_g, SEED_ALT)
    print(f"  [分割内 overall] 探索seed={ig_s:.4f}  別seed={ig_a:.4f}  (採用判断は別seed)")
    full_a = rep("全773別seed", ALL_GLOBAL, SEED_ALT) if FULL_EVAL else None
    if FULL_EVAL:
        print(f"  [参考: 全773 overall 別seed] = {full_a:.4f}")

    best.write_text(json.dumps(dict(study=sname, split_seed=SPLIT_SEED, group=g, n_wells=len(wells_g),
                                    tw_source=TW_SOURCE, use_anchor=bool(bp.get("use_anchor", False)),
                                    best_value_search=float(study.best_value),
                                    in_group_search_seed=ig_s, in_group_alt_seed=ig_a,
                                    full773_alt_seed=full_a, params=bp), indent=2))
    print(f"  saved -> {best.name}", flush=True)
    return ig_a


ALL_GLOBAL = []


def main():
    global ALL_GLOBAL
    if pf.DEVICE != "cuda":
        print("WARNING: CUDA 無し。遅いです。")
    bad = [g for g in GROUPS_TO_RUN if g < 0 or g >= N_SPLITS]
    if bad:
        print(f"★不正な分割番号 {bad}(0..{N_SPLITS-1})"); return

    wells = pf.list_wells()
    if N_WELLS:
        wells = wells[:N_WELLS]
    cache = {}
    n_nbr = 0
    for w in wells:
        c = pf.build_cache(w)
        if c is not None and c["tvt_true"] is not None:
            cache[w] = c
            if c.get("nbr_pairs") is not None:
                n_nbr += 1
    pf.CACHE = cache
    ALL = [w for w in wells if w in cache]
    ALL_GLOBAL = ALL
    if ANCHOR_MODE != "off" and not pf._STRUCT:
        print(f"★WARNING: ANCHOR={ANCHOR_MODE} だが構造prior({pf.PRIOR_NAME})が空 → 錨が効かない(ancs=1e9)。")
    print(f"[合成split5] usable {len(ALL)} 坑井 (nbrプールあり {n_nbr})  prior={pf.PRIOR_NAME}  src={TW_SOURCE}  錨={ANCHOR_MODE}")
    print(f"[合成split5] 参照 = tw + " + " + ".join([s for s in ("self", "nbr") if s in TW_SOURCE])
          + f"  (nbr無い {len(ALL)-n_nbr} 坑井は nbr尤度=中立)")
    print(f"[合成split5] SPLIT_SEED={SPLIT_SEED} N_SPLITS={N_SPLITS}  → 各分割を別studyで最適化(合成尤度/MD-1)")

    groups = make_split(ALL)
    print(f"[合成split5] 分割サイズ: {[len(g) for g in groups]}  (saved split5_s{SPLIT_SEED}.json)")
    print(f"[合成split5] 回す分割: {GROUPS_TO_RUN}\n")

    hard = set()
    results = {}
    for i, g in enumerate(GROUPS_TO_RUN, 1):
        print(f"\n########## [{i}/{len(GROUPS_TO_RUN)}] 分割 {g} ##########", flush=True)
        try:
            results[g] = run_group(g, groups[g], hard)
        except Exception as e:
            print(f"★[分割{g}] 失敗: {type(e).__name__}: {e}(次へ)", flush=True)
    print(f"\n{'='*66}\n[合成split5] 完了。各分割 別seed overall:")
    for g in sorted(results):
        print(f"  分割{g}: {results[g]:.4f}")


if __name__ == "__main__":
    main()
