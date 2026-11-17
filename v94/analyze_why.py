# -*- coding: utf-8 -*-
"""なぜ nbr_gr5 > tw か切り分け(2026-07-30)。同一100井・同一PF(off)で参照変種を比較:
  tw / nbr_gr1(1井) / nbr_gr5(5井) / nbr_rand5(ランダム5井) / nbr_gr5_pure(tw土台なし純5井)
  → デノイズ(1vs5) / GR類似選択(rand vs gr) / ハイブリッド(pure vs hybrid) の寄与を分離。
出力: why_refs.txt"""
import os, sys, io, json, glob
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
import numpy as np, pandas as pd
from scipy.stats import binned_statistic
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pf_engine_v94 as pf
ART = pf.ART; NS = 32; CHUNK = 200
P = json.load(open(ART / "v94_pf_1.json", encoding="utf-8"))["best_params"]

def cc(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    return np.corrcoef(a[m], b[m])[0, 1] if m.sum() > 20 and np.std(a[m]) > 1e-9 and np.std(b[m]) > 1e-9 else np.nan

Wd = {}
for f in sorted(glob.glob(str(pf.DATA_DIR / "*__horizontal_well.csv"))):
    w = os.path.basename(f).split("__")[0]
    try: d = pd.read_csv(f, usecols=["X", "Y", "GR", "TVT_input", "TVT"])
    except Exception: continue
    tv = d["TVT"].to_numpy(float)
    if not np.isfinite(tv).any(): continue
    o = np.argsort(tv)
    Wd[w] = dict(gr=d["GR"].to_numpy(float), ti=d["TVT_input"].to_numpy(float), tv=tv, o=o)
wids = list(Wd); widx = {w: i for i, w in enumerate(wids)}
_valid = sorted(w for w in pf.list_wells() if pf.CACHE.setdefault(w, pf.build_cache(w)) is not None and w in widx)
NW = int(os.environ.get("N_WELLS", "100"))
if NW < len(_valid):
    rng = np.random.RandomState(0); ALLW = sorted(rng.choice(_valid, NW, replace=False).tolist())
else:
    ALLW = _valid
print(f"[why] 対象={len(ALLW)}井", flush=True)

def hybrid(c, src_tvt, src_gr):
    tt = c["tw_tvt"]; tg = c["tw_gr"]; grid = np.arange(float(tt.min()), float(tt.max()), 0.2)
    if len(grid) < 12: return tt.copy(), tg.copy()
    base = np.interp(grid, tt, tg); m = np.isfinite(src_tvt) & np.isfinite(src_gr)
    if m.sum() >= 20:
        edges = np.concatenate([grid, [grid[-1] + 0.2]])
        med, _, _ = binned_statistic(src_tvt[m], src_gr[m], statistic="median", bins=edges)
        h = np.isfinite(med); base[h] = med[h]
    return grid, base

def pure(src_tvt, src_gr):
    m = np.isfinite(src_tvt) & np.isfinite(src_gr)
    if m.sum() < 30: return None
    lo, hi = np.percentile(src_tvt[m], 1), np.percentile(src_tvt[m], 99); grid = np.arange(lo, hi, 0.2)
    if len(grid) < 12: return None
    edges = np.concatenate([grid, [grid[-1] + 0.2]])
    med, _, _ = binned_statistic(src_tvt[m], src_gr[m], statistic="median", bins=edges)
    h = np.isfinite(med)
    return (grid[h], med[h]) if h.sum() > 10 else None

def concat_src(sel):
    return (np.concatenate([Wd[wids[j]]["tv"] for j in sel]), np.concatenate([Wd[wids[j]]["gr"] for j in sel]))

REF = {k: {} for k in ["tw", "nbr_gr1", "nbr_gr5", "nbr_rand5", "nbr_gr5_pure"]}
rr = np.random.RandomState(7)
for w in ALLW:
    c = pf.CACHE[w]; i = widx[w]; T = Wd[w]; pm = np.isfinite(T["ti"])
    REF["tw"][w] = (c["tw_tvt"].copy(), c["tw_gr"].copy())
    sims = np.full(len(wids), -9.0)
    if pm.sum() >= 40:
        tip = T["ti"][pm]; grp = T["gr"][pm]
        if len(tip) > 80:
            st_ = len(tip) // 80; tip = tip[::st_]; grp = grp[::st_]
        for j in range(len(wids)):
            if j == i: continue
            N = Wd[wids[j]]; s = cc(grp, np.interp(tip, N["tv"][N["o"]], N["gr"][N["o"]])); sims[j] = s if np.isfinite(s) else -9
    order = list(np.argsort(-sims))
    g1, g5 = order[:1], order[:5]
    rand5 = [j for j in rr.choice(len(wids), 8, replace=False) if j != i][:5]
    st, sg = concat_src(g1); REF["nbr_gr1"][w] = hybrid(c, st, sg)
    st, sg = concat_src(g5); REF["nbr_gr5"][w] = hybrid(c, st, sg)
    st, sg = concat_src(rand5); REF["nbr_rand5"][w] = hybrid(c, st, sg)
    st, sg = concat_src(g5); pr = pure(st, sg); REF["nbr_gr5_pure"][w] = pr if pr else REF["tw"][w]

def predict(refmap, seed):
    res = {}
    for c0 in range(0, len(ALLW), CHUNK):
        sub = ALLW[c0:c0 + CHUNK]; inps = []
        for w in sub:
            c = dict(pf.CACHE[w]); t, g = refmap[w]
            c["tw_tvt"] = np.asarray(t, np.float64); c["tw_gr"] = np.asarray(g, np.float64); c["tw_fb"] = float(np.nanmedian(g))
            inps.append(pf.make_inputs(c, P))
        preds = pf.run_forward_ens(inps, P, pf.DEVICE, seed, NS)
        for k, w in enumerate(sub): res[w] = preds[k] - pf.CACHE[w]["tvt_true"]
    return res

def rmse(res):
    e = np.concatenate([np.asarray(res[w], float) for w in ALLW]); e = e[np.isfinite(e)]; return np.sqrt(np.mean(e ** 2))

out = ["=== なぜ nbr_gr5>tw か 切り分け(100井, off, 別シード) ==="]
# ★サブセット易しさの物差し: pf_1_errs(773調整@777) を同100井にpool
import pickle
try:
    e1 = pickle.loads((ART / "pf_1_errs.pkl").read_bytes())
    ee = np.concatenate([np.asarray(e1[w], float) for w in ALLW if w in e1]); ee = ee[np.isfinite(ee)]
    out.append(f"  [物差し] #1 off(773調整)@777 を同100井にpool = {np.sqrt(np.mean(ee**2)):.4f}  (cf 773全体=9.46 → 差=サブセット易しさ)")
except Exception as ex:
    out.append(f"  [物差し] pf_1_errs読込失敗: {ex}")
for nm, sd in [("tw", 501), ("nbr_gr1", 502), ("nbr_gr5", 503), ("nbr_rand5", 504), ("nbr_gr5_pure", 505)]:
    r = rmse(predict(REF[nm], sd)); out.append(f"  {nm:14s} = {r:.4f}")
out.append("  解釈: gr1 vs gr5=デノイズ / rand5 vs gr5=GR類似選択 / gr5_pure vs gr5=ハイブリッド寄与")
txt = "\n".join(out); print("\n" + txt, flush=True)
(ART / "why_refs.txt").write_text(txt, encoding="utf-8")
