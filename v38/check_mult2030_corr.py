# -*- coding: utf-8 -*-
# v38 anchor_mult20 vs mult30 の相関調査(ユーザー依頼 2026-07-15)
#   mult30 = 既存ダンプ v38_mult30_errs.npz(全773坑井)
#   mult20 = best_params(v38_anchor_mult20_current.json)で predict_errs をサブセット実行
#   env: SUBSTEP(default3=1/3サブセット) NSEEDS(32) CHUNK(40)
import os, sys, json, time
os.environ["V38_MULT"] = "20"          # import前に設定(pf.ANCHOR_MULT に効く)
os.environ["V38_PRIOR"] = "struct"       # ★struct錨に固定(デフォルトはholdに変わっている)
import numpy as np

sys.stdout = open(1, "w", encoding="utf-8", closefd=False)
P0 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(P0, "v38"))
import iterative_pf_boost_v38 as pf

SUBSTEP = int(os.environ.get("SUBSTEP", "3"))
SEED = int(os.environ.get("SEED", "20250707"))   # mult20評価seed(探索seed=20250707。別seedで下駄チェック)
NSEEDS = int(os.environ.get("NSEEDS", "32"))
CHUNK = int(os.environ.get("CHUNK", "40"))
ART = os.path.join(P0, "v38", "artifacts_v38")
bp = json.loads(open(os.path.join(ART, "v38_anchor_mult20_current.json"), encoding="utf-8").read())["best_params"]
z30 = np.load(os.path.join(ART, "v38_mult30_errs.npz"))

wells = [w for w in pf.list_wells() if w in set(z30.files)][::SUBSTEP]
ok = []
for w in wells:
    c = pf.build_cache(w)
    if c is not None and c["tvt_true"] is not None:
        pf.CACHE[w] = c; ok.append(w)
print(f"[corr] mult20を実行: wells={len(ok)}(1/{SUBSTEP}サブセット) seeds={NSEEDS} chunk={CHUNK} device={pf.DEVICE} MULT={pf.ANCHOR_MULT}")
t0 = time.time()
print(f"      seed={SEED}")
errs20 = pf.predict_errs(bp, ok, SEED, NSEEDS, CHUNK)
print(f"      done {time.time()-t0:.0f}s")

rows20 = []; rows30 = []; wr = []
for w in ok:
    e20 = np.asarray(errs20[w], float); e30 = np.asarray(z30[w], float)
    n = min(len(e20), len(e30))
    m = np.isfinite(e20[:n]) & np.isfinite(e30[:n])
    if m.sum() < 10: continue
    rows20.append(e20[:n][m]); rows30.append(e30[:n][m])
    wr.append((w, float(np.sqrt(np.mean(e20[:n][m] ** 2))), float(np.sqrt(np.mean(e30[:n][m] ** 2)))))
E20 = np.concatenate(rows20); E30 = np.concatenate(rows30)
r20 = np.array([x[1] for x in wr]); r30 = np.array([x[2] for x in wr])

def rmse(e): return float(np.sqrt(np.mean(e ** 2)))
print(f"\n=== v38 錨mult20 vs mult30(同一坑井 {len(wr)}本, 行 {len(E20):,}) ===")
print(f"  overall RMSE : mult20 {rmse(E20):.3f} / mult30 {rmse(E30):.3f}")
print(f"  行レベル誤差相関   corr(e20,e30) = {np.corrcoef(E20, E30)[0,1]:.4f}")
print(f"  坑井RMSE相関       pearson {np.corrcoef(r20, r30)[0,1]:.4f} / spearman "
      f"{np.corrcoef(np.argsort(np.argsort(r20)), np.argsort(np.argsort(r30)))[0,1]:.4f}")
print(f"  予測平均ブレンド    rmse((e20+e30)/2) = {rmse((E20+E30)/2):.3f}")
print(f"  坑井別min oracle   {float(np.sqrt(np.sum(np.minimum(r20,r30)**2 * np.array([len(x) for x in rows20])) / sum(len(x) for x in rows20))):.3f}")
d = r20 - r30
big = sorted(zip([x[0] for x in wr], r20, r30), key=lambda t: -(abs(t[1]-t[2])))[:8]
print("  乖離が大きい坑井(RMSE20 vs RMSE30):")
for w, a, b in big: print(f"    {w}: {a:7.2f} vs {b:7.2f}")

# ---- 参考: mult30 同一設定・別seed(seed777ダンプ)の複製相関 = corrの物差し ----
_p777 = os.path.join(ART, "v38_mult30_errs_seed777.npz")
if os.path.exists(_p777):
    z777 = np.load(_p777)
    _a = []; _b = []
    for _w, _e30x in zip([x[0] for x in wr], rows30):
        if _w not in set(z777.files): continue
        _e7 = np.asarray(z777[_w], float); _n = min(len(_e7), len(_e30x))
        _m = np.isfinite(_e7[:_n]) & np.isfinite(_e30x[:_n])
        _a.append(_e30x[:_n][_m]); _b.append(_e7[:_n][_m])
    if _a:
        _A = np.concatenate(_a); _B = np.concatenate(_b)
        print(f"  [物差し] mult30 vs mult30(seed777): 行corr = {np.corrcoef(_A,_B)[0,1]:.4f}"
              f" / RMSE {rmse(_A):.3f} vs {rmse(_B):.3f}")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Meiryo"
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].scatter(r30, r20, s=10, alpha=0.5)
lim = [0.3, max(r20.max(), r30.max()) * 1.1]
ax[0].plot(lim, lim, "k--", lw=0.8); ax[0].set_xscale("log"); ax[0].set_yscale("log")
ax[0].set_xlabel("mult30 坑井RMSE"); ax[0].set_ylabel("mult20 坑井RMSE")
ax[0].set_title(f"坑井RMSE比較(corr={np.corrcoef(r20,r30)[0,1]:.3f})"); ax[0].grid(alpha=0.3)
ss = np.random.default_rng(0).choice(len(E20), size=min(60000, len(E20)), replace=False)
ax[1].scatter(E30[ss], E20[ss], s=2, alpha=0.15)
ax[1].set_xlabel("mult30 行誤差"); ax[1].set_ylabel("mult20 行誤差")
ax[1].set_title(f"行誤差(corr={np.corrcoef(E20,E30)[0,1]:.3f})"); ax[1].grid(alpha=0.3)
ax[1].set_xlim(-40, 40); ax[1].set_ylim(-40, 40)
plt.tight_layout(); plt.savefig(os.path.join(P0, "v38", f"fig_mult2030_corr_s{SEED}.png"), dpi=120)
print("saved v38/fig_mult2030_corr.png")
