"""
v19 PFバンクの oracle / カバレッジ確認
=====================================

pf_bank.json の各PFパラメータで全wellを推論し、
  - 各PF単体の overall RMSE
  - **oracle**(各wellで一番良いPFを選んだ場合) の overall RMSE
  - RMSE 閾値ごとのカバレッジ(oracleで rmse<3/5/7/10 の well数, >10=未カバー)
  - 貪欲にPFを足したときのカバレッジ曲線(何本で頭打ちか / 全部カバーできるか)
を出す。「複数PFで全wellをカバーできているか？」を確認するためのもの。

PFコアは iterative_pf_boost_v19.py を再利用。重いので PFごとにキャッシュ(再開可)。
実行: rogii_claude/.venv/Scripts/python.exe v19/check_bank_oracle.py
"""

import json, pickle, time
from pathlib import Path
import numpy as np
import iterative_pf_boost_re_v19 as pf   # PFコア(build_cache, make_inputs, run_forward_ens, DEVICE 等)

# ============================== CONFIG ==============================
BANK_PATH    = pf.OUT_DIR / "pf_bank.json"
CACHE_PATH   = pf.OUT_DIR / "bank_eval_cache.pkl"   # PFごとの per-well RMSE/残差(再開用)
N_SEEDS_EVAL = 32           # 推論シード数(本番一致)。速く概算したいなら 8。
CHUNK        = 200          # GPUバッチwell数(OOMなら下げる)
THRESH       = [3, 5, 7, 10]
N_WELLS      = None         # None=全well, 整数で先頭N(デバッグ)
INCLUDE_V9   = True         # v9のPFパラメータ(pf_1/2/3, 物理OFF)もバンクに加えて比較
# ===================================================================


def v9_entries():
    """v9 の PF_PARAM_SETS_RAW を v19 _pf_core 互換に変換(物理OFF)。"""
    import sys
    sys.path.append(str(pf.PROJECT / "v9"))
    from pf_gpu_run import PF_PARAM_SETS_RAW
    out = []
    for name, raw in PF_PARAM_SETS_RAW.items():
        P = {k: v for k, v in raw.items() if k != "n_seeds"}
        P["use_phys"] = False; P["phys_sig"] = 1.0   # v9は物理項なし
        out.append((f"v9_{name}_n{N_SEEDS_EVAL}", 0, P))
    return out


def predict_rmse_resid(P, seed, wells, cache):
    """1つのPFで wells を推論 → {wid: rmse}, {wid: residual(finite)} を返す。"""
    rmse, resid = {}, {}
    for c0 in range(0, len(wells), CHUNK):
        sub = wells[c0:c0 + CHUNK]
        inps = [pf.make_inputs(cache[w], P) for w in sub]
        preds = pf.run_forward_ens(inps, P, pf.DEVICE, seed, N_SEEDS_EVAL)
        for k, w in enumerate(sub):
            e = preds[k] - cache[w]["tvt_true"]; e = e[np.isfinite(e)]
            resid[w] = e
            rmse[w] = float(np.sqrt(np.mean(e ** 2))) if e.size else 1e3
    return rmse, resid


def main():
    bank = json.loads(BANK_PATH.read_text())
    if not bank:
        print("pf_bank.json が空です。先に v19 を回してください。"); return
    print(f"PFバンク: {len(bank)} 本")

    wells = pf.list_wells()
    if N_WELLS:
        wells = wells[:N_WELLS]
    cache = {}
    for w in wells:
        c = pf.build_cache(w)
        if c is not None and c["tvt_true"] is not None:
            cache[w] = c
    WELLS = [w for w in wells if w in cache]
    print(f"usable wells: {len(WELLS)}")

    # ---- 各PFで推論(キャッシュ再開) ----
    store = {}
    if CACHE_PATH.exists():
        store = pickle.loads(CACHE_PATH.read_bytes())
        print(f"cache: {len(store)} PF 済み")
    # 実験対象 = v19バンク(+ 任意で v9 pf_1/2/3)
    entries = [(f"r{e['round']}_s{e.get('seed', 0)}_n{N_SEEDS_EVAL}", e.get("seed", 0), e["params"]) for e in bank]
    if INCLUDE_V9:
        entries += v9_entries()
    print(f"実験対象 PF: {len(entries)} 本" + (f" (v19 {len(bank)} + v9 {len(entries)-len(bank)})" if INCLUDE_V9 else ""))

    RMSE, RESID, labels = [], [], []
    for label, seed, params in entries:
        if label not in store:
            t0 = time.time()
            rm, rs = predict_rmse_resid(params, seed, WELLS, cache)
            store[label] = dict(rmse=rm, resid=rs)
            CACHE_PATH.write_bytes(pickle.dumps(store))
            print(f"  {label}: done ({time.time()-t0:.0f}s)", flush=True)
        RMSE.append(store[label]["rmse"]); RESID.append(store[label]["resid"]); labels.append(label)
    nP = len(RMSE)

    # ---- 各PF単体 overall ----
    def pooled(resid_dict):
        e = np.concatenate([resid_dict[w] for w in WELLS]); return float(np.sqrt(np.mean(e ** 2)))
    print("\n-- 各PF単体 overall RMSE --")
    for i, lab in enumerate(labels):
        n10 = sum(RMSE[i][w] > 10 for w in WELLS)
        print(f"  {lab:16s} overall={pooled(RESID[i]):.3f}  >10={n10}")

    # ---- oracle(各wellで最良PF) ----
    rmat = np.array([[RMSE[i][w] for w in WELLS] for i in range(nP)])   # [nP, nW]
    best_pf = rmat.argmin(0)                                            # 各wellの最良PF
    orac_rmse = rmat.min(0)                                             # 各wellの oracle rmse
    orac_resid = [RESID[best_pf[j]][w] for j, w in enumerate(WELLS)]
    orac_overall = float(np.sqrt(np.mean(np.concatenate(orac_resid) ** 2)))
    print(f"\n=== ORACLE (各wellで最良PF) overall RMSE = {orac_overall:.3f} ===")

    # ---- 閾値カバレッジ ----
    print("-- oracle per-well rmse のカバレッジ --")
    prev = 0
    for t in THRESH:
        n = int((orac_rmse < t).sum())
        print(f"  rmse < {t:2d}: {n:4d}/{len(WELLS)} ({100*n/len(WELLS):.1f}%)  [区間{prev}-{t}: {n - int((orac_rmse<prev).sum()) if prev else n}]")
        prev = t
    n_over = int((orac_rmse >= 10).sum())
    print(f"  rmse >=10 (未カバー): {n_over} 本")
    if n_over:
        order = np.argsort(orac_rmse)[::-1]
        worst = [(WELLS[j], round(float(orac_rmse[j]), 1)) for j in order[:20]]
        print("   未カバー上位:", worst)

    # ---- 単純アンサンブル(選択なし: 全PFを平均/中央値) ----
    def ens_resid(idxs, agg):
        out = []
        for w in WELLS:
            arrs = [RESID[i][w] for i in idxs]
            L = min(len(a) for a in arrs)
            S = np.stack([a[:L] for a in arrs], 0)
            out.append(S.mean(0) if agg == "mean" else np.median(S, 0))
        return out

    def report_ens(name, resid_list):
        ov = float(np.sqrt(np.mean(np.concatenate(resid_list) ** 2)))
        pw = np.array([float(np.sqrt(np.mean(r ** 2))) if r.size else 1e3 for r in resid_list])
        cov = "  ".join(f"<{t}={int((pw < t).sum())}({100*(pw<t).mean():.0f}%)" for t in THRESH)
        print(f"  {name:18s} overall={ov:.3f}  >10={int((pw >= 10).sum())}   {cov}")

    print("\n-- 単純アンサンブル(選択なし, 全PF) --")
    allidx = list(range(nP))
    report_ens("mean", ens_resid(allidx, "mean"))
    report_ens("median", ens_resid(allidx, "median"))
    print(f"  (参考) oracle overall = {orac_overall:.3f}  / 最良単体 = {min(pooled(RESID[i]) for i in range(nP)):.3f}")

    # ---- 貪欲カバレッジ曲線(何本で頭打ち/全カバーか) ----
    print("\n-- 貪欲にPFを追加したときの推移(>10未カバー数 と oracle overall) --")
    cur = np.full(len(WELLS), np.inf); chosen = []; remaining = set(range(nP))
    for step in range(nP):
        best_i, best_unc = None, None
        for i in remaining:
            new = np.minimum(cur, rmat[i])
            unc = int((new >= 10).sum())
            if best_unc is None or unc < best_unc:
                best_unc, best_i = unc, i
        cur = np.minimum(cur, rmat[best_i]); chosen.append(best_i); remaining.discard(best_i)
        # chosen サブセットでの pooled oracle
        bsub = np.array([[RMSE[i][w] for w in WELLS] for i in chosen]).argmin(0)
        sub_resid = [RESID[chosen[bsub[j]]][w] for j, w in enumerate(WELLS)]
        sub_overall = float(np.sqrt(np.mean(np.concatenate(sub_resid) ** 2)))
        print(f"  +{labels[best_i]:16s} ({step+1}本): >10={best_unc:4d}  oracle_overall={sub_overall:.3f}", flush=True)
        if best_unc == 0:
            print("  → 全wellが rmse<10 でカバーされた。");
    print("\n結論の見方: >10 が 0 になれば壊滅は全カバー。頭打ちなら、その本数以降は別系統(機構違いPF/別生成器)が必要。")


if __name__ == "__main__":
    main()
