# -*- coding: utf-8 -*-
"""
v65 性格別PFチューナー — 探索空間を明示的に制限した「性格違いPF」の量産用
==========================================================================
- **PERSONALITY(下のCONFIG)を変えるだけで、探索範囲の制限とファイル名が切り替わる。**
- 性格は「範囲の狭め」と「ピン留め(探索対象から除外)」で定義する。
- 錨(struct)は**性格ごとに宣言制**: use_anchor=True の性格では必ず錨が使われる(有無は探索しない)。
- tw/self/neighbor は性格の対象外(従来どおり env TW_SOURCE で study 単位指定、混合比のみ探索)。
- **anchor_ramp_md / lk_floor / gr_debias / rate_target は探索しない**(ユーザー指示 2026-07-19、中立固定)。
  → 旧 anchor_late(ramp)・dip_keeper(rate_target)は固定パラメータ非依存の anchor_loose・dip_holder へ転用。
- 出力: v65/artifacts_v65/v65p_{性格}[_{tw_source}]_current.json / _best.json(personality を明記)

実行 : rogii_claude/.venv/Scripts/python.exe v65/tune_pf_personality_v65.py
env  : N_TRIALS(200) SEARCH_SEEDS(32) EVAL_SEEDS(32) N_WELLS(0=全) SEARCH_CHUNK(240) EVAL_CHUNK(120)
       TW_SOURCE(tw) V38_PRIOR(v66_leanv1cumul=改善版struct既定)
"""
import os, json, math
from pathlib import Path
import numpy as np
import optuna
import sys
sys.stdout = open(1, "w", encoding="utf-8", closefd=False)
sys.path.insert(0, str(Path(__file__).parent))

# ============================== CONFIG ==============================
PERSONALITY = "smooth_tracker"   # ★ここを変える(下の PERSONAS の8種)。env PERSONALITY でも上書き可(テスト用)
PERSONALITY = __import__("os").environ.get("PERSONALITY", PERSONALITY)
# ===================================================================

TW_SOURCE = os.environ.get("TW_SOURCE", "tw")
os.environ["V65_TWSRC"] = TW_SOURCE
import pf_engine_v65 as pf
import warmup_v65 as wu

# ---- 基本探索空間(pf_engine_v65.suggest_params と同一の範囲を宣言的に) ----
#   ("f", lo, hi, log) = suggest_float / ("cat", [..]) = suggest_categorical / ("pin", v) = 固定(探索しない)
BASE = {
    "n_particles":     ("cat", [400, 500, 600, 700, 800]),
    "likelihood_scale": ("f", 6.0, 60.0, True),
    "init_pos_std":    ("f", 0.1, 4.0, True),
    "init_rate_std":   ("f", 0.0003, 0.03, True),
    "mom":             ("f", 0.990, 0.99998, False),
    "vn":              ("f", 0.0003, 0.012, True),
    "pn":              ("f", 0.002, 0.06, True),
    "rp":              ("f", 0.1, 1.5, True),
    "rr":              ("f", 0.00005, 0.004, True),
    "resamp":          ("f", 0.03, 0.6, True),
    "gr_sig_min":      ("f", 4.0, 12.0, False),
    "gr_sig_max":      ("f", 22.0, 60.0, False),
    "gr_sig_def":      ("f", 70.0, 160.0, False),
    "gr_sig_mult":     ("f", 1.6, 4.0, False),
    "tvt_clip_margin": ("f", 50.0, 200.0, False),
    "rate_clip":       ("cat", [0.0, 1.5, 2.0, 2.5, 3.0, 5.0]),
    "gr_power":        ("f", 1.4, 3.2, False),
    "hgr_smooth_r":    ("cat", [0, 1, 2, 3, 4, 6]),
    "tw_gr_smooth_r":  ("cat", [0, 1, 2]),
    "jump_prob":       ("cat", [0.0, 0.002, 0.005, 0.01, 0.02]),
    "jump_std":        ("cat", [5.0, 10.0, 15.0, 20.0]),
    "phys_sig":        ("f", 0.003, 5.0, True),
    # anchor_ramp_md / lk_floor / gr_debias / rate_target は探索しない(ユーザー指示 2026-07-19)。
    #   full_params の中立デフォルト(ramp=0 / lk_floor=0 / gr_debias=0 / rate_target="zero")に固定。
    #   性格の identity にもこれらは使わない(旧 anchor_late=ramp / dip_keeper=rate_target は転用済)。
    "grid_step":       ("cat", [0.1, 0.15, 0.2, 0.35, 0.5]),
    # 錨系は use_anchor=True の性格でのみ探索される
    "anchor_mult":     ("f", 3.0, 80.0, True),
}

# ---- 8性格(anchor は宣言制: use_anchor は探索されず必ずこの値) ----
PERSONAS = {
    "smooth_tracker": dict(   # 滑らか追従: jump禁止・高momentum・低ノイズ。モード跳びしない保守派
        use_anchor=False,
        overrides={"jump_prob": ("pin", 0.0), "mom": ("f", 0.997, 0.99998, False),
                   "vn": ("f", 0.0003, 0.004, True), "pn": ("f", 0.002, 0.02, True),
                   "rate_clip": ("cat", [1.5, 2.0, 2.5])}),
    "jumpy_explorer": dict(   # 双峰脱出: jump必須(0除外)・低momentum・高位置ノイズ
        use_anchor=False,
        overrides={"jump_prob": ("cat", [0.005, 0.01, 0.02, 0.05]),
                   "jump_std": ("cat", [10.0, 15.0, 20.0, 30.0]),
                   "mom": ("f", 0.990, 0.999, False), "pn": ("f", 0.01, 0.06, True)}),
    "anchor_faithful": dict(  # struct信者: 錨必須・強め(mult小=強)・常時ON
        use_anchor=True,
        overrides={"anchor_mult": ("f", 5.0, 25.0, True)}),
    "anchor_loose": dict(     # 弱錨: 錨必須だが広く弱い(mult大=弱)。structを緩い誘導に留める
        use_anchor=True,      #   (旧 anchor_late=ramp遅延起動は ramp 固定化で不可 → 錨強度違いへ転用)
        overrides={"anchor_mult": ("f", 25.0, 80.0, True)}),
    "sharp_matcher": dict(    # 先鋭照合: 高power・低σ倍率・細格子固定。GR細部にコミット
        use_anchor=False,
        overrides={"gr_power": ("f", 2.4, 3.5, False), "gr_sig_mult": ("f", 1.2, 2.2, False),
                   "hgr_smooth_r": ("cat", [0, 1]), "grid_step": ("pin", 0.2)}),
    "robust_skeptic": dict(   # 鈍感ロバスト: 低power・高σ倍率・強平滑・粗格子。外れGRに引きずられない
        use_anchor=False,     #   (旧 lk_floor 床は固定化で不可 → power/σ倍率/平滑/格子で鈍感性を出す)
        overrides={"gr_power": ("f", 1.2, 2.0, False),
                   "gr_sig_mult": ("f", 2.5, 5.0, False), "hgr_smooth_r": ("cat", [2, 3, 4, 6]),
                   "grid_step": ("cat", [0.35, 0.5])}),
    "dip_holder": dict(       # dip持続: 超高momentum+低rateノイズ+rate拘束で初期dipを長く保持
        use_anchor=False,     #   (旧 dip_keeper=rate_target"ir" は固定化で不可 → 高momで rate減衰を遅くし近似)
        overrides={"mom": ("f", 0.9995, 0.99998, False),
                   "init_rate_std": ("f", 0.0003, 0.004, True), "vn": ("f", 0.0003, 0.003, True),
                   "rate_clip": ("cat", [1.5, 2.0, 3.0])}),
    "wide_net": dict(         # 大群広網: 粒子多・広初期分散・低頻度リサンプル。事後分布を厚く持つ
        use_anchor=False,
        overrides={"n_particles": ("cat", [1200, 1600, 2400]),
                   "init_pos_std": ("f", 2.0, 8.0, True),
                   "resamp": ("f", 0.02, 0.12, True), "rp": ("f", 0.3, 1.5, True)}),
}

assert PERSONALITY in PERSONAS, f"PERSONALITY は {list(PERSONAS)} から選ぶ"
PERSONA = PERSONAS[PERSONALITY]
SPACE = {**BASE, **PERSONA["overrides"]}
if not PERSONA["use_anchor"]:
    SPACE.pop("anchor_mult", None)   # 錨なし性格では錨系を探索しない(anchor_ramp_md は既に探索対象外)

N_TRIALS     = int(os.environ.get("N_TRIALS", "200"))
SEARCH_SEEDS = int(os.environ.get("SEARCH_SEEDS", "32"))
EVAL_SEEDS   = int(os.environ.get("EVAL_SEEDS", "32"))
SEARCH_CHUNK = int(os.environ.get("SEARCH_CHUNK", "240"))
EVAL_CHUNK   = int(os.environ.get("EVAL_CHUNK", "120"))
N_WELLS      = int(os.environ.get("N_WELLS", "0")) or None
SEED_SEARCH  = 20250707
SEED_ALT     = SEED_SEARCH + 1000
HARDCORE_CSV = pf.PROJECT / "v32" / "artifacts_v32" / "hardcore_diag.csv"

# ★study名に prior 名を含める(別prior=v38 struct / v66_* の trial が optuna.db で混ざらないように)
STUDY_NAME = f"v65p_{PERSONALITY}_{pf.PRIOR_NAME}" + ("" if TW_SOURCE == "tw" else f"_{TW_SOURCE.replace('+', '_')}")
OUT      = pf.OUT_DIR
STORAGE  = f"sqlite:///{(OUT / 'optuna.db').as_posix()}"
OUT_BEST = OUT / f"{STUDY_NAME}_best.json"
CUR_PATH = OUT / f"{STUDY_NAME}_current.json"

PINNED = {k: v[1] for k, v in SPACE.items() if v[0] == "pin"}


def suggest(trial):
    P = {}
    for k, spec in SPACE.items():
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
    P["use_anchor"] = bool(PERSONA["use_anchor"])   # 宣言制: 探索しない
    P["use_phys"] = False
    return pf.full_params(P)


def load_hardcore():
    try:
        import pandas as pd
        d = pd.read_csv(HARDCORE_CSV)
        return set(d[d["hard"] == True].iloc[:, 0].astype(str))
    except Exception:
        return set()


def main():
    print(f"[v65p] 性格={PERSONALITY}(use_anchor={PERSONA['use_anchor']} 固定) tw_source={TW_SOURCE}")
    print(f"       制約: " + json.dumps(
        {k: (v[1] if v[0] == 'pin' else (v[1] if v[0] == 'cat' else [v[1], v[2]])) for k, v in PERSONA["overrides"].items()},
        ensure_ascii=False))
    if pf.DEVICE != "cuda":
        print("WARNING: CUDA 無し。遅いです。")

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
    print(f"usable wells: {len(ALL)}")

    study = optuna.create_study(direction="minimize", study_name=STUDY_NAME, storage=STORAGE,
                                load_if_exists=True, sampler=optuna.samplers.TPESampler(seed=SEED_SEARCH))
    done = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)
    print(f"optuna {done}/{N_TRIALS} done  -> {STUDY_NAME}")
    # ---- warm-up: r0_seed32 を性格の制約内にスナップして初期trialに(空study時/FORCE_WARMUP=1) ----
    _spec = {k: v for k, v in SPACE.items() if v[0] != "pin"}
    if "self" in TW_SOURCE: _spec["self_mix_w"] = ("f", 0.1, 1.0, False)
    if "nbr" in TW_SOURCE: _spec["nbr_mix_w"] = ("f", 0.1, 1.0, False)
    _wq = wu.snap_to_space(wu.R0_PARAMS, _spec)
    if len(study.trials) == 0 or os.environ.get("FORCE_WARMUP", "0") == "1":
        study.enqueue_trial(_wq, skip_if_exists=True)
        print(f"[warmup] r0_seed32(性格制約へスナップ)をenqueue({len(_wq)}キー)", flush=True)

    def objective(trial):
        P = suggest(trial)
        try:
            r = pf.overall_rmse_on(P, ALL, SEED_SEARCH, SEARCH_SEEDS, SEARCH_CHUNK)
        except Exception:
            return 1e3
        return r if math.isfinite(r) else 1e3

    def full_of(best_params):
        P = pf.full_params({**PINNED, **best_params})
        P["use_anchor"] = bool(PERSONA["use_anchor"])
        return P

    def cb(study, trial):
        try:
            bv = study.best_value; bp = full_of(study.best_params)
        except Exception:
            return
        nd = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)
        CUR_PATH.write_text(json.dumps(dict(study=STUDY_NAME, personality=PERSONALITY,
                                            tw_source=TW_SOURCE, use_anchor=PERSONA["use_anchor"],
                                            trials_done=nd, best_value=bv, best_params=bp), indent=2))
        tv = trial.value if trial.value is not None else float("nan")
        print(f"   [{PERSONALITY}] {nd}/{N_TRIALS}  this={tv:.4f}  best={bv:.4f}", flush=True)

    if done < N_TRIALS:
        study.optimize(objective, n_trials=N_TRIALS - done, callbacks=[cb])
    bp = full_of(study.best_params)
    print(f"\n[v65p:{PERSONALITY}] best overall(探索seed, in-sample) = {study.best_value:.4f}")
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

    OUT_BEST.write_text(json.dumps(dict(
        study=STUDY_NAME, personality=PERSONALITY, tw_source=TW_SOURCE,
        use_anchor=PERSONA["use_anchor"],
        constraints={k: str(v) for k, v in PERSONA["overrides"].items()},
        best_value_search=float(study.best_value),
        overall_search_seed=on_ov, overall_alt_seed=alt_ov,
        hardcore_search_seed=on_hd, hardcore_alt_seed=alt_hd,
        params=bp), indent=2))
    print(f"\nsaved -> {OUT_BEST}")


if __name__ == "__main__":
    main()
