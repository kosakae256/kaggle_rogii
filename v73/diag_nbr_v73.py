# -*- coding: utf-8 -*-
"""C4 nbr診断: nbr被覆のある坑井だけを対象に、tw単独 vs tw+nbr の per-well RMSE を直接比較。
   悪化が「nbrあり坑井に限定して大きい」なら実装でなく証拠(データム/GR較正ずれ)の問題。
   使い方: NW=200 NSEED=8 python v73/diag_nbr_v73.py"""
import os, io, sys
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
os.environ.setdefault("TWSRC", "tw+nbr")     # importでnbr_pairs構築を有効化
import numpy as np
import nnpf_v73 as M

NSEED = int(os.environ.get("NSEED", "8"))
DATA, SIMD = M.load_data()
wns = [d[0] for d in DATA]; inps = [d[1] for d in DATA]; tvs = [d[2] for d in DATA]
sims_w = [SIMD[w] for w in wns] if SIMD is not None else [None] * len(wns)
has_nbr = [bool(x["ref_masks"][1].sum() > 0) for x in inps]   # 参照2番目(nbr)のmaskが立つ坑井
print(f"wells={len(DATA)}  nbr被覆あり={sum(has_nbr)}", flush=True)

# tw のみ(refsのnbr層βを0に=中立化して同一seedで比較)
import copy
inps_tw = []
for x in inps:
    y = dict(x); y["ref_grids"] = x["ref_grids"][:1]; y["ref_masks"] = x["ref_masks"][:1]
    y["ref_beta"] = x["ref_beta"][:1]
    inps_tw.append(y)

res = {}
for tag, ii in [("tw", inps_tw), ("tw+nbr", inps)]:
    pf, ps, ll, _ = M.run_pf(ii, sims_w, M.R0, M.SEED0, NSEED, M.CHUNK, 0.02)
    per = {}
    for b, wn in enumerate(wns):
        comb, _ = M.seed_combine(ps[b], ll[b], float(M.R0["likelihood_scale"]))
        e = comb[:len(tvs[b])] - tvs[b][:len(comb)]; e = e[np.isfinite(e)]
        per[wn] = float(np.sqrt(np.mean(e ** 2))) if e.size else np.nan
    res[tag] = per
    print(f"[{tag}] done", flush=True)

print("\n=== nbr被覆あり坑井の per-well RMSE (tw → tw+nbr) ===")
rows = [(w, res["tw"][w], res["tw+nbr"][w]) for w, h in zip(wns, has_nbr) if h]
rows.sort(key=lambda r: r[2] - r[1])
for w, a, b in rows:
    print(f"  {w}: {a:7.2f} → {b:7.2f}  ({b-a:+7.2f})")
imp = sum(1 for _, a, b in rows if b < a - 0.05); wor = sum(1 for _, a, b in rows if b > a + 0.05)
print(f"\n改善 {imp} / 悪化 {wor} / 同等 {len(rows)-imp-wor}  (全{len(rows)}本)")
print("nbr無し坑井の一致確認(差の最大):",
      max(abs(res["tw"][w] - res["tw+nbr"][w]) for w, h in zip(wns, has_nbr) if not h))
