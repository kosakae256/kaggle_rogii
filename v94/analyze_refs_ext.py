# -*- coding: utf-8 -*-
"""拡張参照実験(2026-07-30): 既存4参照 + 新規(self_graft/nbr_gr5_bp/nbr_gr5_deriv) を
同一単体PF(off)・別シードで走らせ、相関/min-var blend/oracle を測る。
狙い: ①GR類似選択の"空間"を変える(raw/band-pass/微分)と直交近傍が得られるか
      ②self graft外挿(v49)がself被覆を広げ照合器を強くするか。
真TVTは評価のみ・全て test適用可(選択=targetのprefixのみ, 参照=train井の既知ラベル)。
出力: refs_ext_analysis.txt / refs_ext_analysis.png
env: N_WELLS(既定10000=全井) / K(近傍数, 既定5) / NS(既定32) / CHUNK(既定200)
"""
import os, sys, io, json, glob
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Meiryo"
from scipy.stats import binned_statistic
from scipy.ndimage import gaussian_filter1d
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pf_engine_v94 as pf

ART = pf.ART; NS = int(os.environ.get("NS", "32")); CHUNK = int(os.environ.get("CHUNK", "200"))
K = int(os.environ.get("K", "5"))
P = json.load(open(ART / "v94_pf_1.json", encoding="utf-8"))["best_params"]

def cc(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    return np.corrcoef(a[m], b[m])[0, 1] if m.sum() > 15 and np.std(a[m]) > 1e-9 and np.std(b[m]) > 1e-9 else np.nan

# 類似度空間の変換(共通TVT格子, 最終軸に沿って適用=ベクトル化)
def tf_row(arr, space):
    if space == "raw": return arr
    if space == "bp": return gaussian_filter1d(arr, 4, axis=-1) - gaussian_filter1d(arr, 64, axis=-1)
    if space == "deriv": return np.gradient(gaussian_filter1d(arr, 3, axis=-1), axis=-1)
    return arr
SPACES = ["raw", "bp", "deriv"]

print("[ext] CSV読込...", flush=True)
Wd = {}
for f in sorted(glob.glob(str(pf.DATA_DIR / "*__horizontal_well.csv"))):
    w = os.path.basename(f).split("__")[0]
    try: d = pd.read_csv(f, usecols=["X", "Y", "GR", "TVT_input", "TVT"])
    except Exception: continue
    tv = d["TVT"].to_numpy(float)
    if not np.isfinite(tv).any(): continue
    o = np.argsort(tv)
    Wd[w] = dict(cx=np.nanmean(d["X"]), cy=np.nanmean(d["Y"]), gr=d["GR"].to_numpy(float),
                 ti=d["TVT_input"].to_numpy(float), tv=tv[o], grs=d["GR"].to_numpy(float)[o])
wids = list(Wd); widx = {w: i for i, w in enumerate(wids)}
CX = np.array([Wd[w]["cx"] for w in wids]); CY = np.array([Wd[w]["cy"] for w in wids])
_valid = sorted(w for w in pf.list_wells() if pf.CACHE.setdefault(w, pf.build_cache(w)) is not None and w in widx)
NW = int(os.environ.get("N_WELLS", "10000"))
ALLW = _valid if NW >= len(_valid) else sorted(np.random.RandomState(0).choice(_valid, NW, replace=False).tolist())
print(f"[ext] 対象={len(ALLW)}井 K={K} 空間={SPACES}", flush=True)

# ★共通TVT格子に各井の生GRを事前ビン(GR類似のベクトル化用)
GLO = np.floor(min(Wd[w]["tv"][0] for w in wids)); GHI = np.ceil(max(Wd[w]["tv"][-1] for w in wids))
G = np.arange(GLO, GHI, 0.5)
RAWGRID = np.empty((len(wids), len(G)), np.float64)      # 平坦外挿(np.interp既定)=元の挙動に忠実
COVER = np.zeros((len(wids), len(G)), bool)
for idx, w in enumerate(wids):
    N = Wd[w]; RAWGRID[idx] = np.interp(G, N["tv"], N["grs"])
    COVER[idx] = (G >= N["tv"][0]) & (G <= N["tv"][-1])
print(f"[ext] 共通格子 {len(G)}点 事前ビン完了", flush=True)

def hybrid(c, src_tvt, src_gr):
    tt = c["tw_tvt"]; tg = c["tw_gr"]; grid = np.arange(float(tt.min()), float(tt.max()), 0.2)
    if len(grid) < 12: return tt.copy(), tg.copy()
    base = np.interp(grid, tt, tg); m = np.isfinite(src_tvt) & np.isfinite(src_gr)
    if m.sum() >= 20:
        edges = np.concatenate([grid, [grid[-1] + 0.2]])
        med, _, _ = binned_statistic(src_tvt[m], src_gr[m], statistic="median", bins=edges)
        h = np.isfinite(med); base[h] = med[h]
    return grid, base

def self_graft(c, T):
    """v49流: self(prefix)をtw格子に置換 + prefix範囲外は tw を self較正(a*tw+b)して外挿。"""
    tt = c["tw_tvt"]; tg = c["tw_gr"]; grid = np.arange(float(tt.min()), float(tt.max()), 0.2)
    if len(grid) < 12: return tt.copy(), tg.copy()
    base = np.interp(grid, tt, tg); pm = np.isfinite(T["ti"])
    if pm.sum() < 30: return grid, base
    edges = np.concatenate([grid, [grid[-1] + 0.2]])
    smed, _, _ = binned_statistic(T["ti"][pm], T["gr"][pm], statistic="median", bins=edges)
    have = np.isfinite(smed)
    if have.sum() < 8: return grid, base
    # 重なりで tw→self 較正 (self ≈ a*tw + b)
    A = np.c_[base[have], np.ones(have.sum())]
    try:
        coef, *_ = np.linalg.lstsq(A, smed[have], rcond=None); a, b = coef
    except Exception:
        a, b = 1.0, 0.0
    out = base.copy(); out[have] = smed[have]                    # prefix範囲=self実測
    if 0.2 <= a <= 5.0:
        out[~have] = a * base[~have] + b                          # 範囲外=較正したtwで外挿
    return grid, out

def nbr_by_space(c, T, i, space):
    """元analyze_refs4に忠実なループ選択(raw=6.93実績)+空間変換。prefix範囲の一様格子で相関。"""
    pm = np.isfinite(T["ti"])
    if pm.sum() < 40: return c["tw_tvt"].copy(), c["tw_gr"].copy()
    tip = T["ti"][pm]; grp = T["gr"][pm]; so = np.argsort(tip); tip = tip[so]; grp = grp[so]
    lo, hi = float(tip[0]), float(tip[-1]); ug = np.arange(lo, hi, 0.5)
    if len(ug) < 12: return c["tw_tvt"].copy(), c["tw_gr"].copy()
    tgt = tf_row(np.interp(ug, tip, grp)[None, :], space)[0]; tc = tgt - tgt.mean(); tn = np.linalg.norm(tc) + 1e-9
    sims = np.full(len(wids), -9.0)
    for j in range(len(wids)):
        if j == i: continue
        N = Wd[wids[j]]
        if N["tv"][0] > hi - 5 or N["tv"][-1] < lo + 5: continue          # 被覆不足を除外
        cg = tf_row(np.interp(ug, N["tv"], N["grs"])[None, :], space)[0]; cn = np.linalg.norm(cg - cg.mean())
        if cn < 1e-6: continue
        sims[j] = float((cg - cg.mean()) @ tc) / (cn * tn)
    top = list(np.argsort(-sims)[:K])
    st = np.concatenate([Wd[wids[j]]["tv"] for j in top]); sg = np.concatenate([Wd[wids[j]]["grs"] for j in top])
    return hybrid(c, st, sg)

print("[ext] 参照構築...", flush=True)
REF = {k: {} for k in ["tw", "self", "self_graft", "nbr_gr5_raw", "nbr_gr5_bp", "nbr_gr5_deriv"]}
for w in ALLW:
    c = pf.CACHE[w]; i = widx[w]; T = Wd[w]
    REF["tw"][w] = (c["tw_tvt"].copy(), c["tw_gr"].copy())
    REF["self"][w] = hybrid(c, T["ti"], T["gr"])
    REF["self_graft"][w] = self_graft(c, T)
    REF["nbr_gr5_raw"][w] = nbr_by_space(c, T, i, "raw")
    REF["nbr_gr5_bp"][w] = nbr_by_space(c, T, i, "bp")
    REF["nbr_gr5_deriv"][w] = nbr_by_space(c, T, i, "deriv")

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

names = list(REF); seeds = [601, 602, 603, 604, 605, 606]
E = {}
for nm, sd in zip(names, seeds):
    print(f"[ext] PF走行: {nm}@{sd} ...", flush=True); E[nm] = predict(REF[nm], sd)

cols = {nm: [] for nm in names}; per_well = []
for w in ALLW:
    arrs = [np.asarray(E[nm][w], np.float64) for nm in names]
    m = min(len(a) for a in arrs); msk = np.all([np.isfinite(a[:m]) for a in arrs], 0)
    if msk.sum() == 0: continue
    cut = [a[:m][msk] for a in arrs]
    for nm, a in zip(names, cut): cols[nm].append(a)
    per_well.append(cut)
M = np.stack([np.concatenate(cols[nm]) for nm in names], 1)
r = [float(np.sqrt(np.mean(M[:, i] ** 2))) for i in range(len(names))]
rho = np.corrcoef(M.T)
def minvar(cov):
    one = np.ones(cov.shape[0]); ci = np.linalg.inv(cov + 1e-6 * np.eye(len(cov)))
    w = ci @ one / (one @ ci @ one); return w, float(np.sqrt(max(0, w @ cov @ w)))
w_all, r_all = minvar(M.T @ M / len(M))
orow = []; owell = []; sel = [0] * len(names)
for cut in per_well:
    A = np.stack(cut, 1); orow.append(A[np.arange(len(A)), np.argmin(np.abs(A), 1)])
    rk = [np.sqrt(np.mean(a ** 2)) for a in cut]; bi = int(np.argmin(rk)); sel[bi] += 1; owell.append(cut[bi])
r_row = float(np.sqrt(np.mean(np.concatenate(orow) ** 2))); r_well = float(np.sqrt(np.mean(np.concatenate(owell) ** 2)))

L = [f"=== 拡張参照({len(names)}本) 単体PF(off, 別シード) 相関/blend/oracle  ({len(ALLW)}井) ==="]
L.append("  単体RMSE: " + "  ".join(f"{nm}={r[i]:.3f}" for i, nm in enumerate(names)))
L.append("  相関ρ 全行列:")
L.append("    " + " " * 14 + "".join(f"{n[:9]:>10s}" for n in names))
for i, nm in enumerate(names):
    L.append(f"    {nm:14s}" + "".join(f"{rho[i,j]:>10.3f}" for j in range(len(names))))
L.append(f"  全{len(names)}本 min-var blend = {r_all:.4f}")
L.append(f"  per-well oracle = {r_well:.4f}  選択={dict(zip(names,sel))}")
L.append(f"  per-row  oracle = {r_row:.4f}")
txt = "\n".join(L); print("\n" + txt, flush=True)
(ART / "refs_ext_analysis.txt").write_text(txt, encoding="utf-8")

fig, ax = plt.subplots(1, 2, figsize=(15, 6))
im = ax[0].imshow(rho, vmin=0, vmax=1, cmap="RdYlBu_r")
ax[0].set_xticks(range(len(names))); ax[0].set_yticks(range(len(names)))
ax[0].set_xticklabels(names, rotation=40, ha="right"); ax[0].set_yticklabels(names)
for i in range(len(names)):
    for j in range(len(names)): ax[0].text(j, i, f"{rho[i,j]:.2f}", ha="center", va="center", fontsize=8)
ax[0].set_title("拡張参照 誤差相関 ρ"); fig.colorbar(im, ax=ax[0], fraction=.046)
bn = names + ["blend", "well-orac", "row-orac"]; bv = r + [r_all, r_well, r_row]
ax[1].bar(range(len(bn)), bv, color=["C0"] * len(names) + ["gray", "k", "darkred"])
ax[1].set_xticks(range(len(bn))); ax[1].set_xticklabels(bn, rotation=40, ha="right")
for i, v in enumerate(bv): ax[1].text(i, v + 0.05, f"{v:.2f}", ha="center", fontsize=8)
ax[1].set_ylabel("overall RMSE [ft]"); ax[1].set_title("単体 vs blend vs oracle"); ax[1].grid(axis="y", alpha=.3)
plt.tight_layout(); plt.savefig(ART / "refs_ext_analysis.png", dpi=110); print("saved refs_ext_analysis.png", flush=True)
