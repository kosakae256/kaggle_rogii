# -*- coding: utf-8 -*-
"""
v51 — NN-emission PF バンク群の相関 + 単純アンサンブル評価
============================================================
各バンク(best W)の行別エラーから、行レベル相関行列と等重み平均アンサンブルを測る。
参照: v38(seed777, 素の錨PF)/ v50 StageB combo も混ぜた場合。
実行: python v51/ens_nnpf_v51.py
"""
import json
import numpy as np
from pathlib import Path
P0 = Path(r"C:\Users\kosaka256\Documents\rogii_claude")
ART51 = P0 / "v51" / "artifacts_v51"

BANKS = {"r0": "nnpf_errs_r0_seed32_w0.02.npz", "pfA": "nnpf_errs_pfA_w0.01.npz",
         "pf_1": "nnpf_errs_pf_1_w0.02.npz", "pf_2": "nnpf_errs_pf_2_w0.02.npz",
         "pf_3": "nnpf_errs_pf_3_w0.02.npz", "r1": "nnpf_errs_r1_seed32_w0.02.npz",
         "r4": "nnpf_errs_r4_seed32_w0.02.npz"}
E = {k: dict(np.load(ART51 / v)) for k, v in BANKS.items()}
E["v38"] = dict(np.load(P0 / "v38" / "artifacts_v38" / "v38_mult30_errs_seed777.npz"))
E["v50B"] = dict(np.load(P0 / "v50" / "artifacts_v50" / "v50_stageBcombo_errs.npz"))
names = list(E.keys())
wells = sorted(set.intersection(*[set(E[n]) for n in names]))
print(f"共通坑井: {len(wells)}")

# 行アライン(井戸ごと最短長+全系有限)
ALN = {}
for w in wells:
    n0 = min(len(E[n][w]) for n in names)
    arrs = {n: E[n][w][:n0].astype(np.float64) for n in names}
    m = np.ones(n0, bool)
    for n in names: m &= np.isfinite(arrs[n])
    if m.sum() > 0:
        ALN[w] = {n: arrs[n][m] for n in names}
CAT = {n: np.concatenate([ALN[w][n] for w in ALN]) for n in names}
print("行数:", len(CAT[names[0]]))

def rmse(e): return float(np.sqrt(np.mean(e ** 2)))
print("\n=== 単体(整列行, 検算) ===")
for n in names: print(f"  {n:5s}: {rmse(CAT[n]):.4f}")

print("\n=== 行レベル誤差相関 ===")
hdr = "      " + " ".join(f"{n:>6s}" for n in names)
print(hdr)
C = np.zeros((len(names), len(names)))
for i, a in enumerate(names):
    row = f"{a:5s} "
    for j, b in enumerate(names):
        C[i, j] = np.corrcoef(CAT[a], CAT[b])[0, 1]
        row += f" {C[i, j]:6.3f}"
    print(row)

def ens(sub, weights=None):
    ws = np.ones(len(sub)) / len(sub) if weights is None else np.array(weights, float)
    ws = ws / ws.sum()
    e = sum(w * CAT[n] for w, n in zip(ws, sub))
    return rmse(e)

print("\n=== 等重み単純アンサンブル ===")
combos = [
    ("NN7本全部", ["r0", "pfA", "pf_1", "pf_2", "pf_3", "r1", "r4"]),
    ("NN上位3(r0,pfA,pf_3)", ["r0", "pfA", "pf_3"]),
    ("NN上位4(+pf_1)", ["r0", "pfA", "pf_3", "pf_1"]),
    ("r0+pfA", ["r0", "pfA"]),
    ("r0+pfA+v50B", ["r0", "pfA", "v50B"]),
    ("r0+pfA+v38", ["r0", "pfA", "v38"]),
    ("NN上位3+v50B+v38", ["r0", "pfA", "pf_3", "v50B", "v38"]),
    ("NN7+v50B+v38", ["r0", "pfA", "pf_1", "pf_2", "pf_3", "r1", "r4", "v50B", "v38"]),
]
for tag, sub in combos:
    print(f"  {tag:24s}: {ens(sub):.4f}")

# r0×pfA の重み走査
ws = np.arange(0, 1.001, 0.05)
cur = [rmse((1 - x) * CAT["r0"] + x * CAT["pfA"]) for x in ws]
bi = int(np.argmin(cur))
print(f"\n  r0×pfA 重み走査: best w_pfA={ws[bi]:.2f} -> {cur[bi]:.4f}")
json.dump(dict(corr={a: {b: float(C[i, j]) for j, b in enumerate(names)} for i, a in enumerate(names)},
               ens={t: ens(s) for t, s in combos}),
          open(ART51 / "ens_nnpf_results.json", "w"), indent=1)
print("saved ens_nnpf_results.json")
