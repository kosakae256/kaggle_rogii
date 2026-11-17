# -*- coding: utf-8 -*-
"""v94 band-pass "別枠特徴" チューナ(ユーザー指示 2026-07-30)。

背景:
  band-pass PF は単体RMSE≈15と弱い(帯域通過でGR絶対レベルを失う→絶対datum無し)が、
  ★per-well oracle=5.86 / 137井で最良PF(振幅が外す枝をband-passが当てる) = 振幅とは別の判別情報を持つ。
  線形blend利得は-0.01で死ぬ(単体15が高すぎる)。→ ★「振幅GRバンク(pf_1/2/3)とは別枠の特徴」として使う。
  選択器/GBMが per-well/per-row で band-pass予測を利用すれば oracle 5.86 を取りに行ける。

設計:
  - 表現 = GR_BANDPASS_TVT (錨配置 + band-pass, band=2-32ft 既定)。→ os.environ で engine に指示。
  - use_anchor=off (脱相関の源。bp-PF1=0.333 を維持。錨をonにすると絶対がPF2と共通化し脱相関が減る)。
  - ★相関制約なし(別枠特徴なので corr≤0.7 は課さない)。目的 = 自身の overall RMSE 最小。
    ただし振幅バンク pf_1/2/3 との誤差相関は毎trial計算して報告(参考)。
  - 600井, trials=200, n_seeds=32, gs_scale 1.0-2.0。
  - ★振幅バンクと名前衝突させない: v94_bpfeat.json / bpfeat_errs.pkl / optuna_v94_bpfeat.db。
    which_idx() の pf_*_errs.pkl カウントにも入らない(=振幅#4以降に影響しない)。

実行: rogii_claude/.venv/Scripts/python.exe v94/tune_v94_bpfeat.py
出力(v94/artifacts_v94): v94_bpfeat.json(採用params+相関) / bpfeat_errs.pkl(全井誤差) / bpfeat_seed.txt
"""
import os, sys, io, json, glob, pickle
# ★engine import 前に band-pass 表現を有効化(build_cache が読む)
os.environ["GR_BANDPASS_TVT"] = "1"
os.environ.setdefault("BP_SLO_FT", "2")
os.environ.setdefault("BP_SHI_FT", "32")
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
from pathlib import Path
import numpy as np
import optuna
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pf_engine_v94_bp as pf

optuna.logging.set_verbosity(optuna.logging.WARNING)
ART = pf.ART
N_TRIALS = int(os.environ.get("N_TRIALS", "200"))
N_SEEDS  = int(os.environ.get("N_SEEDS", "32"))
CHUNK    = int(os.environ.get("CHUNK", "300"))
N_WELLS  = int(os.environ.get("N_WELLS", "600"))
GS_LO, GS_HI = float(os.environ.get("GS_LO", "1.0")), float(os.environ.get("GS_HI", "2.0"))
USE_ANCHOR = int(os.environ.get("BPFEAT_ANCHOR", "0"))   # 既定=off(脱相関維持)

STORAGE = f"sqlite:///{(ART / 'optuna_v94_bpfeat.db').as_posix()}"
STUDY_NAME = "v94_bpfeat"
OUT_JSON = ART / "v94_bpfeat.json"
ERRS_PKL = ART / "bpfeat_errs.pkl"
SEED_FILE = ART / "bpfeat_seed.txt"
if SEED_FILE.exists():
    SEED = int(SEED_FILE.read_text().strip())
else:
    SEED = int.from_bytes(os.urandom(4), "little")
    SEED_FILE.write_text(str(SEED))


def select_wells(n):
    cand = list(pf.list_wells())
    rng = np.random.RandomState(SEED); rng.shuffle(cand)
    out = []
    for w in cand:
        if pf.CACHE.setdefault(w, pf.build_cache(w)) is not None:
            out.append(w)
        if len(out) >= n:
            break
    return sorted(out)


def all_valid_wells():
    return sorted(w for w in pf.list_wells() if pf.CACHE.setdefault(w, pf.build_cache(w)) is not None)


def load_amp_bank_errs(wells):
    """参考: 振幅バンク pf_1/2/3 の保存済み全井誤差を現wellsに絞る(相関報告用, 制約には使わない)。"""
    priors = []
    for p in sorted(glob.glob(str(ART / "pf_*_errs.pkl"))):
        j = os.path.basename(p)
        d = pickle.loads(Path(p).read_bytes())
        priors.append((j, {w: np.asarray(d[w], np.float64) for w in wells if w in d}))
    return priors


def pooled_corr(errs_cand, prior_map, wells):
    ca, cb = [], []
    for w in wells:
        if w not in prior_map:
            continue
        a = np.asarray(errs_cand[w], np.float64); b = prior_map[w]
        m = min(len(a), len(b))
        if m == 0:
            continue
        aa, bb = a[:m], b[:m]; msk = np.isfinite(aa) & np.isfinite(bb)
        if msk.any():
            ca.append(aa[msk]); cb.append(bb[msk])
    if not ca:
        return 0.0
    A = np.concatenate(ca); B = np.concatenate(cb)
    if A.size > 10 and np.std(A) > 1e-9 and np.std(B) > 1e-9:
        return float(np.corrcoef(A, B)[0, 1])
    return 0.0


WELLS = select_wells(N_WELLS)
AMP = load_amp_bank_errs(WELLS)
print(f"[bpfeat] band=[{os.environ['BP_SLO_FT']},{os.environ['BP_SHI_FT']}]ft use_anchor={bool(USE_ANCHOR)} "
      f"seed={SEED} 対象井={len(WELLS)} 振幅バンク数={len(AMP)} N_TRIALS={N_TRIALS} N_SEEDS={N_SEEDS} "
      f"gs=[{GS_LO},{GS_HI}] ★相関制約なし(目的=自身RMSE)", flush=True)


def objective(trial):
    params = pf.suggest_params(trial)
    params["gs_scale"] = trial.suggest_float("gs_scale", GS_LO, GS_HI)
    params["use_anchor"] = bool(USE_ANCHOR)
    P = pf.full_params(params)
    errs = pf.predict_errs(P, WELLS, seed=99999 + trial.number, n_seeds=N_SEEDS, chunk=CHUNK)
    e = np.concatenate([errs[w] for w in WELLS]); e = e[np.isfinite(e)]
    rmse = float(np.sqrt(np.mean(e ** 2))) if e.size else 1e3
    corrs = [(j, pooled_corr(errs, pm, WELLS)) for j, pm in AMP]
    trial.set_user_attr("rmse", rmse)
    trial.set_user_attr("corrs", [[j, round(c, 3)] for j, c in corrs])
    return rmse                                   # ★別枠特徴: 相関制約なし, 自身RMSE最小


def best_trial(study):
    comp = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
            and t.user_attrs.get("rmse") is not None]
    return min(comp, key=lambda t: t.user_attrs["rmse"]) if comp else None


def save_best(bt):
    bp = pf.full_params({**bt.params}); bp["use_anchor"] = bool(USE_ANCHOR)
    OUT_JSON.write_text(json.dumps(dict(
        kind="bandpass_feature", band=[os.environ["BP_SLO_FT"], os.environ["BP_SHI_FT"]],
        use_anchor=bool(USE_ANCHOR), seed=SEED, rmse=bt.user_attrs["rmse"],
        corrs_vs_amp=bt.user_attrs["corrs"], n_wells=len(WELLS), best_params=bp), indent=1), encoding="utf-8")
    return bp


def main():
    study = optuna.create_study(direction="minimize", study_name=STUDY_NAME, storage=STORAGE,
                                load_if_exists=True, sampler=optuna.samplers.TPESampler(seed=SEED))
    wj = os.environ.get("WARMUP_JSON")
    if wj and Path(wj).exists() and len(study.trials) == 0:
        study.enqueue_trial(json.load(open(wj, encoding="utf-8")))
        print(f"[bpfeat] warmup enqueue: {os.path.basename(wj)}", flush=True)
    for _ in range(N_TRIALS):
        study.optimize(objective, n_trials=1, catch=(Exception,))
        bt = best_trial(study)
        if bt is not None:
            save_best(bt)
            print(f"  trial {len(study.trials)}: best RMSE={bt.user_attrs['rmse']:.4f} "
                  f"corrs_vs_amp={bt.user_attrs['corrs']} gs={bt.params.get('gs_scale'):.3f}", flush=True)
    bt = best_trial(study)
    if bt is None:
        print("[bpfeat] 完了trialなし。", flush=True); return
    bp = save_best(bt)
    print(f"\n[bpfeat] ★採用: RMSE={bt.user_attrs['rmse']:.4f} corrs_vs_amp={bt.user_attrs['corrs']} "
          f"-> {OUT_JSON.name}", flush=True)
    # ★採用band-pass特徴を全773井で走らせ誤差保存(選択器/GBM/oracle測定の入力)
    print(f"[bpfeat] 採用を全井で走らせ誤差保存中({ERRS_PKL.name})...", flush=True)
    ALLW = all_valid_wells()
    errs_all = pf.predict_errs(bp, ALLW, seed=777, n_seeds=N_SEEDS, chunk=CHUNK)
    ERRS_PKL.write_bytes(pickle.dumps({w: np.asarray(errs_all[w], np.float32) for w in ALLW}))
    ea = np.concatenate([errs_all[w] for w in ALLW]); ea = ea[np.isfinite(ea)]
    print(f"[bpfeat] 全井誤差保存完了 {len(ALLW)}井 全井RMSE={np.sqrt(np.mean(ea**2)):.4f} -> {ERRS_PKL.name}", flush=True)


if __name__ == "__main__":
    main()
