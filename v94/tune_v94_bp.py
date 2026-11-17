# -*- coding: utf-8 -*-
"""v94 逐次脱相関PFチューナ — ★完全に作り直す新しいPFバンクをゼロから逐次構築。

設計(ユーザー指示 2026-07-29):
  - forward PF(pf_engine_v94, 非smoothでOK)を optuna で調整。★v93非参照の独立環境。
  - well = ランダムに選んだ 600 井。シード値は毎回変える(PF idx ごとに別シード)。
  - PF#1(最初) = 比較対象なし(相関制約なし)。純粋に overall RMSE 最小を探す。
  - PF#2 以降 = これまで採用した v94 PF 全部との誤差相関 |corr| ≤ 0.7 を制約に追加。
  - ★比較対象はPFが増えるごとに増える。→ 各PF採用後にそのPFの【全773井の誤差】を保存(pf_{i}_errs.pkl)。
    次のPFは保存済み誤差を現在の600井にインデックスして相関を測る(priorを毎回再走させない)。
  - 物理anchor: v94 の順序 = off, on, off, on, ... (PF idx 奇数=off / 偶数=on)。
  - trials=200, n_seeds=32固定, gs_scale 1.0-2.0。
  - CANCELLED: v93 の anchor付き調整(旧予定)は破棄=このv94逐次に置換。

出力(全て v94/artifacts_v94):
  v94_pf_{idx}.json(採用params+相関) / pf_{idx}_errs.pkl(全井誤差) / seed_{idx}.txt(使用シード) / optuna db
IDX は既存 v94_pf_*.json 数+1 で自動。env IDX で明示指定も可。
実行: rogii_claude/.venv/Scripts/python.exe v94/tune_v94.py
"""
import os, sys, io, json, glob, pickle
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
CORR_MAX = float(os.environ.get("CORR_MAX", "0.7"))
GS_LO, GS_HI = float(os.environ.get("GS_LO", "1.0")), float(os.environ.get("GS_HI", "2.0"))


def which_idx():
    if os.environ.get("IDX"):
        return int(os.environ["IDX"])
    # ★完了マーカー = pf_{i}_errs.pkl(全井保存済み)。json は毎trial保存で未完でも存在するため使わない。
    return len(glob.glob(str(ART / "pf_*_errs.pkl"))) + 1


IDX = which_idx()
PHYS = "off" if (IDX % 2 == 1) else "on"      # ★順序: 1=off,2=on,3=off,4=on,...
SEED_FILE = ART / f"seed_{IDX}.txt"
if SEED_FILE.exists():
    SEED = int(SEED_FILE.read_text().strip())
else:
    SEED = int.from_bytes(os.urandom(4), "little")   # ★毎回変わるシード(生成後は保存=resume安定)
    SEED_FILE.write_text(str(SEED))
STORAGE = f"sqlite:///{(ART / f'optuna_v94_pf{IDX}.db').as_posix()}"
STUDY_NAME = f"v94_pf{IDX}"
OUT_JSON = ART / f"v94_pf_{IDX}.json"
ERRS_PKL = ART / f"pf_{IDX}_errs.pkl"


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
    out = []
    for w in pf.list_wells():
        if pf.CACHE.setdefault(w, pf.build_cache(w)) is not None:
            out.append(w)
    return sorted(out)


def load_prior_errs(wells):
    """採用済み v94 PF(1..IDX-1)の【保存済み全井誤差】を現在の wells に絞って返す。"""
    priors = []
    for j in range(1, IDX):
        p = ART / f"pf_{j}_errs.pkl"
        if not p.exists():
            print(f"[v94 pf{IDX}] ★prior pf_{j}_errs.pkl が無い(PF#{j}未完)。相関比較から除外", flush=True)
            continue
        d = pickle.loads(p.read_bytes())
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
PRIORS = load_prior_errs(WELLS)
print(f"[v94 pf{IDX}] PHYS={PHYS} seed={SEED} 対象井={len(WELLS)} prior数={len(PRIORS)} "
      f"N_TRIALS={N_TRIALS} N_SEEDS={N_SEEDS} CORR_MAX={CORR_MAX} gs=[{GS_LO},{GS_HI}] "
      f"{'★PF#1=相関制約なし' if IDX == 1 else ''}", flush=True)


def _pooled_matrix(errs_cand, priors, wells):
    """[prior_1..k-1, candidate] を井集合 wells でプールした誤差行列 (N,k)。"""
    cols = []
    for w in wells:
        arrs = [pm[w] for j, pm in priors if w in pm]
        if len(arrs) != len(priors):
            continue
        arrs.append(np.asarray(errs_cand[w], np.float64))
        m = min(len(a) for a in arrs)
        st = np.stack([np.asarray(a[:m], np.float64) for a in arrs], 1)
        msk = np.isfinite(st).all(1)
        if msk.any():
            cols.append(st[msk])
    return np.concatenate(cols, 0) if cols else None


def _minvar_weights(E):
    C = E.T @ E / len(E) + 1e-6 * np.eye(E.shape[1])
    one = np.ones(E.shape[1]); ci = np.linalg.inv(C)
    return ci @ one / (one @ ci @ one)


def blend_rmse_heldout(errs_cand, priors, wells):
    """★held-out(交差検証) min-var ブレンドRMSE。600井を2分割し「片方で重みfit→もう片方で評価」を
       両方向平均。ノイズ利用(弱くて偽独立なゴミPF)は分割を跨いで転移しないので弾かれる。"""
    ws = sorted(wells); A = ws[0::2]; B = ws[1::2]
    outs = []
    for fit, ev in ((A, B), (B, A)):
        Ef = _pooled_matrix(errs_cand, priors, fit)
        Ee = _pooled_matrix(errs_cand, priors, ev)
        if Ef is None or Ee is None or len(Ef) < 10 or len(Ee) < 10:
            return 1e3
        try:
            w = _minvar_weights(Ef)
            outs.append(float(np.sqrt(np.mean((Ee @ w) ** 2))))   # fit重みをheld-out井で評価
        except Exception:
            return 1e3
    return float(np.mean(outs))


def objective(trial):
    params = pf.suggest_params(trial)
    params["gs_scale"] = trial.suggest_float("gs_scale", GS_LO, GS_HI)
    params["use_anchor"] = (PHYS == "on")
    P = pf.full_params(params)
    errs = pf.predict_errs(P, WELLS, seed=99999 + trial.number, n_seeds=N_SEEDS, chunk=CHUNK)
    e = np.concatenate([errs[w] for w in WELLS]); e = e[np.isfinite(e)]
    rmse = float(np.sqrt(np.mean(e ** 2))) if e.size else 1e3
    corrs = [(j, pooled_corr(errs, pm, WELLS)) for j, pm in PRIORS]
    max_corr = float(max((c for _, c in corrs), default=0.0))
    trial.set_user_attr("rmse", rmse)
    trial.set_user_attr("max_corr", max_corr)
    trial.set_user_attr("corrs", [[j, round(c, 4)] for j, c in corrs])
    # ★corr≤CORR_MAX 制約下で単体RMSE最小(blend目的は弱いPFに収束するため撤回, 2026-07-30)
    return rmse + 1000.0 * max(0.0, max_corr - CORR_MAX)


def feasible_best(study):
    feas = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
            and t.user_attrs.get("max_corr", 9.0) <= CORR_MAX]
    return min(feas, key=lambda t: t.user_attrs["rmse"]) if feas else None


def save_best(bt):
    bp = pf.full_params({**bt.params}); bp["use_anchor"] = (PHYS == "on")
    OUT_JSON.write_text(json.dumps(dict(
        idx=IDX, phys=PHYS, seed=SEED, rmse=bt.user_attrs["rmse"],
        max_corr=bt.user_attrs["max_corr"], corrs=bt.user_attrs["corrs"],
        n_wells=len(WELLS), corr_max=CORR_MAX, best_params=bp), indent=1), encoding="utf-8")
    return bp


def main():
    study = optuna.create_study(direction="minimize", study_name=STUDY_NAME, storage=STORAGE,
                                load_if_exists=True, sampler=optuna.samplers.TPESampler(seed=SEED))
    # ★warmup: 指定paramsを初期trialとしてenqueue(良い初期値からTPE探索を始める)
    wj = os.environ.get("WARMUP_JSON")
    if wj and Path(wj).exists() and len(study.trials) == 0:
        wp = json.load(open(wj, encoding="utf-8"))
        study.enqueue_trial(wp)
        print(f"[v94 pf{IDX}] warmup enqueue: {os.path.basename(wj)} ({len(wp)}params)", flush=True)
    for _ in range(N_TRIALS):
        study.optimize(objective, n_trials=1, catch=(Exception,))
        bt = feasible_best(study)
        if bt is not None:
            save_best(bt)
            print(f"  trial {len(study.trials)}: best RMSE={bt.user_attrs['rmse']:.4f} "
                  f"maxcorr={bt.user_attrs['max_corr']:.3f} corrs={bt.user_attrs['corrs']} "
                  f"gs={bt.params.get('gs_scale'):.3f}", flush=True)
        else:
            lt = study.trials[-1]
            print(f"  trial {len(study.trials)}: feasible(≤{CORR_MAX})まだ無し "
                  f"(last rmse={lt.user_attrs.get('rmse')} maxcorr={lt.user_attrs.get('max_corr')})", flush=True)
    bt = feasible_best(study)
    if bt is None:
        print(f"\n[v94 pf{IDX}] feasible な試行なし。CORR_MAX を上げるか N_TRIALS 増。", flush=True)
        return
    bp = save_best(bt)
    print(f"\n[v94 pf{IDX}] ★採用: RMSE={bt.user_attrs['rmse']:.4f} maxcorr={bt.user_attrs['max_corr']:.3f} "
          f"PHYS={PHYS} -> {OUT_JSON.name}", flush=True)
    # ★採用PFを全773井で走らせ、誤差を保存(次PFの相関比較の対象=増えていく)
    print(f"[v94 pf{IDX}] 採用PFを全井で走らせ誤差を保存中(pf_{IDX}_errs.pkl)...", flush=True)
    ALLW = all_valid_wells()
    errs_all = pf.predict_errs(bp, ALLW, seed=777, n_seeds=N_SEEDS, chunk=CHUNK)
    ERRS_PKL.write_bytes(pickle.dumps({w: np.asarray(errs_all[w], np.float32) for w in ALLW}))
    ea = np.concatenate([errs_all[w] for w in ALLW]); ea = ea[np.isfinite(ea)]
    print(f"[v94 pf{IDX}] 全井誤差保存完了 {len(ALLW)}井 全井RMSE={np.sqrt(np.mean(ea**2)):.4f} -> {ERRS_PKL.name}", flush=True)


if __name__ == "__main__":
    main()
