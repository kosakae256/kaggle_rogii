# -*- coding: utf-8 -*-
"""4参照(tw/self/neighbor距離/nbr_gr5=GR類似5井)を同一単体PFパラメータで走らせ、
相関・min-varブレンド・oracle を比較(ユーザー指示 2026-07-30)。
参照は tw格子上に self/nbr の median を substitute(v93流, MIX=1.0=データ有る所は置換)。
★4参照は別シードで走らせ genuine相関。PFパラメータ = v94_pf_1.json(off, 錨なし=参照純比較)。
出力: refs4_analysis.txt / refs4_analysis.png"""
import os, sys, io, json, glob, pickle
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Meiryo"
from scipy.stats import binned_statistic
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pf_engine_v94 as pf

ART = pf.ART
NS = int(os.environ.get("NS", "32")); CHUNK = int(os.environ.get("CHUNK", "200"))
P = json.load(open(ART / "v94_pf_1.json", encoding="utf-8"))["best_params"]

def cc(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    return np.corrcoef(a[m], b[m])[0, 1] if m.sum() > 20 and np.std(a[m]) > 1e-9 and np.std(b[m]) > 1e-9 else np.nan

# --- 全坑井の (X,Y重心, GR, TVT_input, TVT) を読み込み(近傍材料) ---
print("[refs4] CSV読込...", flush=True)
Wd = {}
for f in sorted(glob.glob(str(pf.DATA_DIR / "*__horizontal_well.csv"))):
    w = os.path.basename(f).split("__")[0]
    try: d = pd.read_csv(f, usecols=["X", "Y", "GR", "TVT_input", "TVT"])
    except Exception: continue
    tv = d["TVT"].to_numpy(float)
    if not np.isfinite(tv).any(): continue
    o = np.argsort(tv)
    Wd[w] = dict(cx=np.nanmean(d["X"]), cy=np.nanmean(d["Y"]), gr=d["GR"].to_numpy(float),
                 ti=d["TVT_input"].to_numpy(float), tv=tv, o=o)
wids = list(Wd); CX = np.array([Wd[w]["cx"] for w in wids]); CY = np.array([Wd[w]["cy"] for w in wids])
widx = {w: i for i, w in enumerate(wids)}

_valid = sorted(w for w in pf.list_wells() if pf.CACHE.setdefault(w, pf.build_cache(w)) is not None and w in widx)
NW = int(os.environ.get("N_WELLS", "100"))
if NW < len(_valid):
    rng = np.random.RandomState(int(os.environ.get("WSEED", "0")))
    ALLW = sorted(rng.choice(_valid, NW, replace=False).tolist())
else:
    ALLW = _valid
print(f"[refs4] 対象井={len(ALLW)} (ランダム{NW})", flush=True)

def sub_on_twgrid(c, src_tvt, src_gr):
    """tw格子(0.2ft)上に、src(median)をデータ有る所だけ置換した参照(tvt,gr)を返す。"""
    tt = c["tw_tvt"]; tg = c["tw_gr"]; lo, hi = float(tt.min()), float(tt.max())
    grid = np.arange(lo, hi, 0.2)
    if len(grid) < 12: return tt.copy(), tg.copy()
    base = np.interp(grid, tt, tg)
    m = np.isfinite(src_tvt) & np.isfinite(src_gr)
    if m.sum() >= 20:
        edges = np.concatenate([grid, [grid[-1] + 0.2]])
        med, _, _ = binned_statistic(src_tvt[m], src_gr[m], statistic="median", bins=edges)
        have = np.isfinite(med)
        base[have] = med[have]
    return grid, base

# --- 4参照を各井で構築 ---
print("[refs4] 参照構築(self/neighbor/nbr_gr5)...", flush=True)
REF = {k: {} for k in ["tw", "self", "nbr", "nbr_gr5"]}
for w in ALLW:
    c = pf.CACHE[w]; i = widx[w]; T = Wd[w]
    REF["tw"][w] = (c["tw_tvt"].copy(), c["tw_gr"].copy())
    # self: 自分のprefix (gr@kpos, TVT_input)
    REF["self"][w] = sub_on_twgrid(c, T["ti"], T["gr"])
    # neighbor(距離): 1500ft内 最近傍3(自分除外)
    d = np.hypot(CX - T["cx"], CY - T["cy"]); d[i] = 1e18
    near = [j for j in np.argsort(d) if d[j] < 1500][:3]
    if near:
        sg = np.concatenate([Wd[wids[j]]["gr"] for j in near]); st = np.concatenate([Wd[wids[j]]["tv"] for j in near])
        REF["nbr"][w] = sub_on_twgrid(c, st, sg)
    else:
        REF["nbr"][w] = REF["tw"][w]
    # nbr_gr5: 全坑井から prefix GR類似 top5
    pm = np.isfinite(T["ti"])
    if pm.sum() >= 40:
        sims = np.full(len(wids), -9.0)
        tip = T["ti"][pm]; grp = T["gr"][pm]
        if len(tip) > 80:                     # ★prefix間引き(GR類似計算の高速化, 全井O(N^2)対策)
            st_ = len(tip) // 80; tip = tip[::st_]; grp = grp[::st_]
        for j in range(len(wids)):
            if j == i: continue
            N = Wd[wids[j]]; pg = np.interp(tip, N["tv"][N["o"]], N["gr"][N["o"]])
            s = cc(grp, pg); sims[j] = s if np.isfinite(s) else -9
        top5 = list(np.argsort(-sims)[:5])
        sg = np.concatenate([Wd[wids[j]]["gr"] for j in top5]); st = np.concatenate([Wd[wids[j]]["tv"] for j in top5])
        REF["nbr_gr5"][w] = sub_on_twgrid(c, st, sg)
    else:
        REF["nbr_gr5"][w] = REF["tw"][w]

def predict_with_ref(refmap, seed):
    res = {}
    for c0 in range(0, len(ALLW), CHUNK):
        sub = ALLW[c0:c0 + CHUNK]; inps = []
        for w in sub:
            c = dict(pf.CACHE[w]); t, g = refmap[w]
            c["tw_tvt"] = np.asarray(t, np.float64); c["tw_gr"] = np.asarray(g, np.float64)
            c["tw_fb"] = float(np.nanmedian(g))
            inps.append(pf.make_inputs(c, P))
        preds = pf.run_forward_ens(inps, P, pf.DEVICE, seed, NS)
        for k, w in enumerate(sub):
            res[w] = preds[k] - pf.CACHE[w]["tvt_true"]
    return res

names = ["tw", "self", "nbr", "nbr_gr5"]; seeds = [401, 402, 403, 404]
E = {}
for nm, sd in zip(names, seeds):
    print(f"[refs4] PF走行: {nm}@{sd} ...", flush=True)
    E[nm] = predict_with_ref(REF[nm], sd)

# --- プール解析 ---
cols = {nm: [] for nm in names}; per_well = []
for w in ALLW:
    arrs = [np.asarray(E[nm][w], np.float64) for nm in names]
    m = min(len(a) for a in arrs); msk = np.all([np.isfinite(a[:m]) for a in arrs], 0)
    if msk.sum() == 0: continue
    cut = [a[:m][msk] for a in arrs]
    for nm, a in zip(names, cut): cols[nm].append(a)
    per_well.append(cut)
M = np.stack([np.concatenate(cols[nm]) for nm in names], 1)
r = [float(np.sqrt(np.mean(M[:, i] ** 2))) for i in range(4)]
rho = np.corrcoef(M.T)

def minvar(cov):
    one = np.ones(cov.shape[0]); ci = np.linalg.inv(cov + 1e-6 * np.eye(len(cov)))
    w = ci @ one / (one @ ci @ one); return w, float(np.sqrt(max(0, w @ cov @ w)))
C4 = M.T @ M / len(M); w4, r4 = minvar(C4)
# oracle
orow = []; owell = []; sel = [0, 0, 0, 0]
for cut in per_well:
    A = np.stack(cut, 1)
    orow.append(A[np.arange(len(A)), np.argmin(np.abs(A), 1)])
    rk = [np.sqrt(np.mean(a ** 2)) for a in cut]; bi = int(np.argmin(rk)); sel[bi] += 1; owell.append(cut[bi])
r_row = float(np.sqrt(np.mean(np.concatenate(orow) ** 2)))
r_well = float(np.sqrt(np.mean(np.concatenate(owell) ** 2)))

L = ["=== 4参照 単体PF(v94_pf_1 off, 別シード) 相関/ブレンド/oracle ==="]
L.append(f"  単体RMSE: " + "  ".join(f"{nm}={r[i]:.4f}" for i, nm in enumerate(names)))
L.append("  相関ρ:")
for i in range(4):
    L.append("    " + "  ".join(f"{names[i]}-{names[j]}={rho[i,j]:.3f}" for j in range(4) if j > i))
L.append(f"  4本 min-var blend = {r4:.4f}  重み={tuple(round(x,2) for x in w4)}")
L.append(f"  per-well oracle = {r_well:.4f}  (最良選択 {dict(zip(names,sel))})")
L.append(f"  per-row  oracle = {r_row:.4f}")
txt = "\n".join(L); print("\n" + txt, flush=True)
(ART / "refs4_analysis.txt").write_text(txt, encoding="utf-8")

fig, ax = plt.subplots(1, 2, figsize=(13, 5))
im = ax[0].imshow(rho, vmin=0, vmax=1, cmap="RdYlBu_r")
ax[0].set_xticks(range(4)); ax[0].set_yticks(range(4)); ax[0].set_xticklabels(names); ax[0].set_yticklabels(names)
for i in range(4):
    for j in range(4): ax[0].text(j, i, f"{rho[i,j]:.2f}", ha="center", va="center", fontsize=11)
ax[0].set_title("4参照 誤差相関 ρ (別シード)"); fig.colorbar(im, ax=ax[0], fraction=.046)
bn = names + ["4本blend", "well-oracle", "row-oracle"]; bv = r + [r4, r_well, r_row]
ax[1].bar(bn, bv, color=["C0", "C1", "C2", "C3", "gray", "k", "darkred"])
for i, v in enumerate(bv): ax[1].text(i, v + 0.1, f"{v:.2f}", ha="center", fontsize=9)
ax[1].set_ylabel("overall RMSE [ft]"); ax[1].set_title("4参照 単体 vs blend vs oracle"); ax[1].grid(axis="y", alpha=.3)
plt.setp(ax[1].get_xticklabels(), rotation=25, ha="right")
plt.tight_layout(); plt.savefig(ART / "refs4_analysis.png", dpi=110); print("saved refs4_analysis.png", flush=True)
