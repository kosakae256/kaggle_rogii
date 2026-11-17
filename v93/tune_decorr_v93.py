# -*- coding: utf-8 -*-
"""v93 PF 脱相関チューナ — 既存6PFと誤差相関が低い(=別の特徴を持つ)PFを探す。

方針(ユーザー指示 2026-07-29):
  - PF にとって重要なのは「他の PF の推論結果と別の特徴を持つ」= 相関を低くすること。
  - 既存 v93 の 6 PF の推論結果を train.parquet の pf_ancc_1..6 から取得し、その誤差
    (pf_ancc_i - true_tvt, true=target+last_known_tvt)を基準にする。
  - 候補 PF の誤差 vs 各既存 PF 誤差の |相関| が【すべて ≤ CORR_MAX(0.7)】である制約を課し、
    その制約内で overall RMSE が最小のものを採用する。
  - 調整は SEED ベースでランダムに選んだ 160 井のみ(全井不要)。
  - 物理 prior(GRフリー錨)を使う/使わないは use_anchor(既存 suggest_params の categorical[True,False])で選択。
    env PHYS=on/off/both(既定both)で固定も可。
  - gs(GRノイズスケール)を gs_scale 1.0-1.5 で optuna 調整(make_inputs で post-clip 乗算)。

出力: v93/artifacts_v93/v93_decorr_best.json、study=optuna_v93_decorr.db(v93_decorr)
env: N_TRIALS(300) N_SEEDS(16) CHUNK(60) N_WELLS(160) SEL_SEED(0) CORR_MAX(0.7) PHYS(both)
実行: rogii_claude/.venv/Scripts/python.exe v93/tune_decorr_v93.py
"""
import os, sys, io, json
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
from pathlib import Path
import numpy as np
import optuna
import pyarrow.parquet as pq
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pf_engine_v93 as pf

optuna.logging.set_verbosity(optuna.logging.WARNING)
PROJ = Path(r"C:\Users\kosaka256\Documents\rogii_claude")
ART = PROJ / "v93" / "artifacts_v93"
TRAIN_PQ = ART / "data" / "train.parquet"

N_TRIALS = int(os.environ.get("N_TRIALS", "300"))
N_SEEDS  = int(os.environ.get("N_SEEDS", "32"))
CHUNK    = int(os.environ.get("CHUNK", "160"))
N_WELLS  = int(os.environ.get("N_WELLS", "160"))
SEL_SEED = int(os.environ.get("SEL_SEED", "93738"))
CORR_MAX = float(os.environ.get("CORR_MAX", "0.7"))
PHYS     = os.environ.get("PHYS", "both").lower()   # both / on / off
_PS = "" if PHYS == "both" else f"_{PHYS}"           # 物理prior 別に出力/studyを分ける
STORAGE  = f"sqlite:///{(ART / f'optuna_v93_decorr{_PS}.db').as_posix()}"
OUT_JSON = ART / f"v93_decorr_best{_PS}.json"
STUDY_NAME = f"v93_decorr{_PS}"
BANKS = [f"pf_ancc_{k}" for k in range(1, 7)]        # 既存 v93 の 6 PF


def eff_seed():
    # ★anchor(PHYS=on)は井戸選択シードを変える(=別の160井で調整)。ユーザー指示 2026-07-29。
    return SEL_SEED + (100000 if PHYS == "on" else 0)


def select_wells():
    """SEEDベースでランダムに N_WELLS 井(build_cache 成功井)を選ぶ。PHYS=on は別シード=別井戸。"""
    cand = list(pf.list_wells())
    rng = np.random.RandomState(eff_seed()); rng.shuffle(cand)
    out = []
    for w in cand:
        if pf.CACHE.setdefault(w, pf.build_cache(w)) is not None:
            out.append(w)
        if len(out) >= N_WELLS:
            break
    return sorted(out)


def load_existing_errs(wells):
    """train.parquet の pf_ancc_1..6 誤差(= pf_ancc_i - (target+lk))を選択井ぶん、井ごとに取得。"""
    tb = pq.read_table(TRAIN_PQ, columns=["well", "target", "last_known_tvt"] + BANKS)
    wcol = tb.column("well").to_pandas().astype(str).to_numpy()
    tgt = tb.column("target").to_numpy(zero_copy_only=False).astype(np.float64)
    lk = tb.column("last_known_tvt").to_numpy(zero_copy_only=False).astype(np.float64)
    true = tgt + lk
    wset = set(wells)
    # 井 -> 行 index(parquet 上は連続ブロック。順序は eval-MD 順で engine と一致する前提)
    idx_of = {}
    for w in wells:
        idx_of[w] = np.where(wcol == w)[0]
    EXIST = {}
    for bi, bc in enumerate(BANKS):
        v = tb.column(bc).to_numpy(zero_copy_only=False).astype(np.float64)
        err = v - true
        EXIST[bi] = {w: err[idx_of[w]] for w in wells}
    del tb
    return EXIST


def pooled_corr(errs_cand, EXIST_bi, wells):
    """候補誤差 vs 1バンク既存誤差の プール相関(井ごとに min長で整列して連結)。"""
    ca, cb = [], []
    for w in wells:
        a = np.asarray(errs_cand[w], np.float64); b = np.asarray(EXIST_bi[w], np.float64)
        m = min(len(a), len(b))
        if m == 0:
            continue
        aa, bb = a[:m], b[:m]
        msk = np.isfinite(aa) & np.isfinite(bb)
        if msk.any():
            ca.append(aa[msk]); cb.append(bb[msk])
    if not ca:
        return 0.0
    A = np.concatenate(ca); B = np.concatenate(cb)
    if A.size > 10 and np.std(A) > 1e-9 and np.std(B) > 1e-9:
        return float(np.corrcoef(A, B)[0, 1])
    return 0.0


WELLS = select_wells()
print(f"[decorr tune] 対象井={len(WELLS)} (seed={eff_seed()} [SEL_SEED={SEL_SEED},PHYS={PHYS}]) N_TRIALS={N_TRIALS} N_SEEDS={N_SEEDS} "
      f"CORR_MAX={CORR_MAX} PHYS={PHYS}", flush=True)
EXIST = load_existing_errs(WELLS)
# 井ごとの行数整合チェック(engine eval行 vs parquet行)
c0 = pf.CACHE[WELLS[0]]; n_eng = len(c0["md"]); n_pq = len(EXIST[0][WELLS[0]])
print(f"[decorr tune] 既存PF誤差ロード完了。整合チェック(先頭井): engine eval={n_eng} vs parquet={n_pq} "
      f"{'一致' if n_eng == n_pq else '★不一致(min長整列で対処)'}", flush=True)


def objective(trial):
    params = pf.suggest_params(trial)
    params["gs_scale"] = trial.suggest_float("gs_scale", 1.0, 2.0)   # ★gs 1.0-2.0(上限1.5→2.0拡張, 2026-07-29)
    if PHYS == "on":
        params["use_anchor"] = True
    elif PHYS == "off":
        params["use_anchor"] = False
    P = pf.full_params(params)
    errs = pf.predict_errs(P, WELLS, seed=99999 + trial.number, n_seeds=N_SEEDS, chunk=CHUNK)
    e = np.concatenate([errs[w] for w in WELLS]); e = e[np.isfinite(e)]
    rmse = float(np.sqrt(np.mean(e ** 2))) if e.size else 1e3
    corrs = [pooled_corr(errs, EXIST[bi], WELLS) for bi in range(len(BANKS))]
    max_corr = float(max(corrs))     # ★符号付き(PF結果は実質正相関のみ。負相関はまず起きない)
    trial.set_user_attr("rmse", rmse)
    trial.set_user_attr("max_corr", max_corr)
    trial.set_user_attr("corrs", [round(c, 4) for c in corrs])
    # ソフト罰則: corr>CORR_MAX を大きく罰し, feasible 域で RMSE 最小へ誘導。採用は最後に feasible から選ぶ。
    return rmse + 1000.0 * max(0.0, max_corr - CORR_MAX)


def feasible_best(study):
    feas = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
            and t.user_attrs.get("max_corr", 9.0) <= CORR_MAX]
    return min(feas, key=lambda t: t.user_attrs["rmse"]) if feas else None


def save_best(bt):
    bp = pf.full_params({**bt.params})
    if PHYS == "on":                  # ★PHYS 強制を best_params にも反映(suggest値でなく実効値)
        bp["use_anchor"] = True
    elif PHYS == "off":
        bp["use_anchor"] = False
    OUT_JSON.write_text(json.dumps(dict(
        rmse=bt.user_attrs["rmse"], max_corr=bt.user_attrs["max_corr"],
        corrs=bt.user_attrs["corrs"], n_wells=len(WELLS), sel_seed=SEL_SEED, corr_max=CORR_MAX,
        phys=PHYS, best_params=bp), indent=1), encoding="utf-8")


def main():
    study = optuna.create_study(direction="minimize", study_name=STUDY_NAME, storage=STORAGE,
                                load_if_exists=True, sampler=optuna.samplers.TPESampler(seed=0))
    for _ in range(N_TRIALS):
        study.optimize(objective, n_trials=1, catch=(Exception,))
        bt = feasible_best(study)
        if bt is not None:
            save_best(bt)
            print(f"  trial {len(study.trials)}: feasible best RMSE={bt.user_attrs['rmse']:.4f} "
                  f"maxcorr={bt.user_attrs['max_corr']:.3f} corrs={bt.user_attrs['corrs']} "
                  f"use_anchor={bt.params.get('use_anchor')} gs_scale={bt.params.get('gs_scale'):.3f}", flush=True)
        else:
            lt = study.trials[-1]
            print(f"  trial {len(study.trials)}: feasible(全corr≤{CORR_MAX})まだ無し "
                  f"(last rmse={lt.user_attrs.get('rmse')} maxcorr={lt.user_attrs.get('max_corr')})", flush=True)
    bt = feasible_best(study)
    if bt is not None:
        print(f"\n[decorr] ★feasible best: RMSE={bt.user_attrs['rmse']:.4f} "
              f"maxcorr={bt.user_attrs['max_corr']:.3f} corrs={bt.user_attrs['corrs']}")
        print(f"  use_anchor={bt.params.get('use_anchor')} gs_scale={bt.params.get('gs_scale'):.3f}")
        print(f"  -> {OUT_JSON.name} 保存済み (PHYS={PHYS})")
    else:
        print(f"\n[decorr] feasible(全|corr|≤{CORR_MAX})な試行なし。CORR_MAX を上げるか N_TRIALS を増やす。")


if __name__ == "__main__":
    main()
