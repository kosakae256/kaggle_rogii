# -*- coding: utf-8 -*-
"""
v77 統合blend評価 — artifacts_v77 の全メソッドpkl + 素PF を貪欲前向き選択で合成
使い方: python v77/grand_blend_v77.py p5_out_dip03_60w p6_out_lp13_60w p12_out_aw0.03g13_60w ...
        (引数なし=既定リスト)。全pklに共通の井戸だけで評価。
"""
import io, sys, pickle, itertools
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
from pathlib import Path
import numpy as np

P0 = Path(r"C:\Users\kosaka256\Documents\rogii_claude")
ART = P0 / "v77" / "artifacts_v77"
LS = 10.536319667317654
names = sys.argv[1:] or ["p5_out_dip03_60w", "p6_out_lp13_60w", "p12_out_aw0.03g13_60w", "p11_out_aw0.3g1_60w"]
anc = pickle.loads((ART / "k16_anchor_train.pkl").read_bytes())
bare = pickle.loads((P0 / "v73" / "artifacts_v73" / "seed_dump_bare0.pkl").read_bytes())
Ms = {}
for n in names:
    fp = ART / f"{n}.pkl"
    if not fp.exists():
        print(f"[skip] {n} なし"); continue
    Ms[n] = pickle.loads(fp.read_bytes())
common = set(bare) & set(anc)
for n in Ms: common &= set(Ms[n])
common = sorted(common)
print(f"共通 {len(common)}井  members: PF, 錨, {list(Ms)}")

errs = {"PF": [], "錨": []}
for n in Ms: errs[n] = []
for wid in common:
    d = bare[wid]; ll = d["ll"].astype(np.float64)
    ww = np.exp((ll - ll.max()) / LS); ww /= ww.sum()
    epf = (ww[:, None] * d["err"].astype(np.float64)).sum(0)
    tv = d["true"]
    T = min(len(epf), len(tv), len(anc[wid]["tvt"]), *[len(np.asarray(Ms[n][wid]["pred"] if "pred" in Ms[n][wid] else Ms[n][wid]["soft"])) for n in Ms])
    legs = {"PF": epf[:T], "錨": anc[wid]["tvt"][:T] - tv[:T]}
    for n in Ms:
        o = Ms[n][wid]
        pr = np.asarray(o["pred"] if "pred" in o else o["soft"], dtype=np.float64)
        legs[n] = pr[:T] - tv[:T]
    m = np.ones(T, bool)
    for v in legs.values(): m &= np.isfinite(v)
    for k, v in legs.items(): errs[k].append(v[m])
E = {k: np.concatenate(v) for k, v in errs.items()}
for k in E: print(f"  {k}: {np.sqrt(np.mean(E[k]**2)):.4f}")
keys = list(E)
M = np.corrcoef(np.stack([E[k] for k in keys]))
print("ρ:"); [print(" ", keys[i], np.round(M[i], 3)) for i in range(len(keys))]

# 貪欲前向き選択(重みは都度11点grid精緻化)
def fit_w(sel):
    Xs = [E[k] for k in sel]
    best = (1e9, None)
    grid = np.linspace(0, 1, 21)
    for w in itertools.product(grid, repeat=len(Xs) - 1):
        if sum(w) > 1: continue
        wl = list(w) + [1 - sum(w)]
        r = float(np.sqrt(np.mean(sum(wi * x for wi, x in zip(wl, Xs)) ** 2)))
        if r < best[0]: best = (r, wl)
    return best

sel = ["PF"]; cur = np.sqrt(np.mean(E["PF"] ** 2))
while True:
    cands = [(fit_w(sel + [k]), k) for k in keys if k not in sel]
    (r, wl), k = min(cands)
    if r > cur - 1e-3: break
    sel.append(k); cur = r
    print(f"+{k}: {r:.4f}  w={dict(zip(sel, [round(float(x),2) for x in wl]))}")
print(f"\n最終: {cur:.4f}  legs={sel}")
