# -*- coding: utf-8 -*-
"""tune_decorr の現在 best(PHYS=off)を全773井で走らせ、overall RMSE と既存6PFとの相関を見る。
   tuner 並走前提: GPU 軽量(n_seeds=16, chunk=40)。tuner は止めない。
実行: rogii_claude/.venv/Scripts/python.exe v93/run_best_allwells.py"""
import os, sys, io, json
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
from pathlib import Path
import numpy as np, pyarrow.parquet as pq
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pf_engine_v93 as pf

ART = Path(r"C:\Users\kosaka256\Documents\rogii_claude") / "v93" / "artifacts_v93"
BEST = json.load(open(ART / "v93_decorr_best_off.json", encoding="utf-8"))
P = dict(BEST["best_params"])
P["use_anchor"] = False   # ★PHYS=off の実効値(ラベルbug対策)
N_SEEDS = int(os.environ.get("NS", "32")); CHUNK = 60   # ★n_seeds=32(本番同等)。tuner停止中はfull GPU
BANKS = [f"pf_ancc_{k}" for k in range(1, 7)]
NAMES = ["pf_1", "pf_2", "pf_3", "r0_s32", "r1_s32", "pfA"]
print(f"[best allwells] best(160井) RMSE={BEST['rmse']:.3f} max_corr={BEST['max_corr']:.3f} "
      f"use_anchor=False gs_scale={P.get('gs_scale'):.3f}  n_seeds={N_SEEDS}", flush=True)

# 全有効井
wells = [w for w in pf.list_wells() if pf.CACHE.setdefault(w, pf.build_cache(w)) is not None]
print(f"[best allwells] 有効井={len(wells)} 予測開始(GPU軽量, tuner並走)...", flush=True)
errs = pf.predict_errs(P, wells, seed=777, n_seeds=N_SEEDS, chunk=CHUNK)   # {well: pred-true}
e = np.concatenate([errs[w] for w in wells]); e = e[np.isfinite(e)]
rmse_all = float(np.sqrt(np.mean(e ** 2)))
print(f"\n★ 候補PF(調整中best) 全{len(wells)}井 overall RMSE = {rmse_all:.4f}", flush=True)

# 既存6PF誤差(全井, parquet)+ 候補との相関
tb = pq.read_table(ART / "data" / "train.parquet", columns=["well", "target", "last_known_tvt"] + BANKS)
wcol = tb.column("well").to_pandas().astype(str).to_numpy()
tgt = tb.column("target").to_numpy(zero_copy_only=False).astype(float)
lk = tb.column("last_known_tvt").to_numpy(zero_copy_only=False).astype(float)
true = tgt + lk
idx_of = {w: np.where(wcol == w)[0] for w in wells}
EX = {bi: tb.column(bc).to_numpy(zero_copy_only=False).astype(float) - true for bi, bc in enumerate(BANKS)}


def pooled_corr(bi):
    ca, cb = [], []
    for w in wells:
        a = np.asarray(errs[w], float); b = EX[bi][idx_of[w]]
        m = min(len(a), len(b))
        if m == 0:
            continue
        aa, bb = a[:m], b[:m]; msk = np.isfinite(aa) & np.isfinite(bb)
        if msk.any():
            ca.append(aa[msk]); cb.append(bb[msk])
    A = np.concatenate(ca); B = np.concatenate(cb)
    return float(np.corrcoef(A, B)[0, 1]) if A.size > 10 else 0.0


corrs = [pooled_corr(bi) for bi in range(6)]
print("\n=== 候補PF vs 既存6PF 相関(全井, 誤差ベース) ===")
for n, c in zip(NAMES, corrs):
    print(f"  {n:>7s}: {c:+.3f}")
print(f"  max_corr(全井)={max(corrs):.3f}  (160井では {BEST['max_corr']:.3f})")
print(f"  ※ 制約 corr≤0.7 を全井でも満たすか: {'○' if max(corrs)<=0.7 else '×(160井最適が全井では超過)'}")
