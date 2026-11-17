# -*- coding: utf-8 -*-
"""PF#1(off,調整済み) と PF#2(on=物理錨版, 作成中の現best) の全井相関＋ブレンドを実測。
   PF#2は v94_pf_2.json の現在best paramsで全井走行(seed=777でPF#1 errs.pklと揃える)。
   tuner非停止。GPU共存のため chunk小さめ。出力: blend_pf1_pf2.txt / .png"""
import os, sys, io, json, pickle
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pf_engine_v94 as pf
CHUNK = int(os.environ.get("CHUNK", "30")); NS = int(os.environ.get("NS", "32"))
SEED2 = int(os.environ.get("SEED2", "314159"))   # ★PF#2は別シード(PF#1 errs.pkl=777 と揃えない=seedノイズ共有を排除)
ART = pf.ART

e1d = pickle.loads((ART / "pf_1_errs.pkl").read_bytes())          # PF#1 全井誤差(seed777)
P2 = json.load(open(ART / "v94_pf_2.json", encoding="utf-8"))
bp2 = P2["best_params"]
print(f"[blend] PF#2現best: RMSE(600井)={P2['rmse']:.4f} corr(vsPF#1)={P2['max_corr']:.3f} "
      f"phys={P2['phys']} use_anchor={bp2.get('use_anchor')} gs={bp2.get('gs_scale'):.3f}", flush=True)

ALLW = [w for w in pf.list_wells() if pf.CACHE.setdefault(w, pf.build_cache(w)) is not None]
ALLW = sorted(set(ALLW) & set(e1d.keys()))
if os.environ.get("NW"):   # ★軽量版: 井をサブセットに(tuner競合下で高速に blend推定)
    import numpy as _np
    _r = _np.random.RandomState(2024); _r.shuffle(ALLW); ALLW = sorted(ALLW[:int(os.environ["NW"])])
print(f"[blend] PF#2を全{len(ALLW)}井で走行中(★別シード seed={SEED2}≠PF#1の777, n_seeds={NS}, chunk={CHUNK})...", flush=True)
e2d = pf.predict_errs(bp2, ALLW, seed=SEED2, n_seeds=NS, chunk=CHUNK)

# 井ごと min長整列で連結
E1, E2 = [], []
for w in ALLW:
    a = np.asarray(e1d[w], np.float64); b = np.asarray(e2d[w], np.float64)
    m = min(len(a), len(b)); msk = np.isfinite(a[:m]) & np.isfinite(b[:m])
    E1.append(a[:m][msk]); E2.append(b[:m][msk])
E1 = np.concatenate(E1); E2 = np.concatenate(E2)
r1 = np.sqrt(np.mean(E1 ** 2)); r2 = np.sqrt(np.mean(E2 ** 2))
rho = float(np.corrcoef(E1, E2)[0, 1])

# ブレンド(pred空間 = err+true だが、誤差の線形結合でRMSE評価と等価: blend_err = w·e1+(1-w)·e2)
ws = np.linspace(0, 1, 51)
rb = [np.sqrt(np.mean((w * E1 + (1 - w) * E2) ** 2)) for w in ws]
wbest = ws[int(np.argmin(rb))]; rbest = min(rb)
# 解析最適重み(分散最小): w* = (σ2² - ρσ1σ2)/(σ1²+σ2²-2ρσ1σ2)
s1, s2 = E1.std(), E2.std()
wstar = (s2 ** 2 - rho * s1 * s2) / (s1 ** 2 + s2 ** 2 - 2 * rho * s1 * s2)

out = []
out.append("=== PF#1(off) × PF#2(on=物理錨,現best) 全井ブレンド (★別シード: PF#1=777/PF#2=%d) ===" % SEED2)
out.append(f"  PF#1 全井RMSE = {r1:.4f}")
out.append(f"  PF#2 全井RMSE = {r2:.4f}  (現best, 調整未完)")
out.append(f"  ★誤差相関 rho = {rho:.4f}")
out.append(f"  等重み(0.5) blend RMSE = {np.sqrt(np.mean((0.5*E1+0.5*E2)**2)):.4f}")
out.append(f"  ★最適重み blend: w(PF#1)={wbest:.2f} -> RMSE={rbest:.4f}  (解析w*={wstar:.3f})")
out.append(f"  改善: 良い方(min {min(r1,r2):.4f}) から {rbest-min(r1,r2):+.4f}")
txt = "\n".join(out); print("\n" + txt, flush=True)
(ART / "blend_pf1_pf2.txt").write_text(txt, encoding="utf-8")

try:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "Meiryo"
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot(ws, rb, "-o", ms=3); ax[0].axvline(wbest, color="r", ls="--", label=f"最適 w={wbest:.2f}")
    ax[0].axhline(r1, color="g", ls=":", label=f"PF#1 {r1:.3f}"); ax[0].axhline(r2, color="orange", ls=":", label=f"PF#2 {r2:.3f}")
    ax[0].set_xlabel("w (PF#1の重み)"); ax[0].set_ylabel("blend RMSE"); ax[0].set_title(f"PF#1×PF#2 ブレンド (rho={rho:.3f})"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
    idx = np.random.RandomState(0).choice(len(E1), min(4000, len(E1)), replace=False)
    ax[1].scatter(E1[idx], E2[idx], s=4, alpha=.3); ax[1].set_xlabel("PF#1 誤差"); ax[1].set_ylabel("PF#2 誤差")
    ax[1].set_title(f"誤差散布 rho={rho:.3f}"); ax[1].grid(alpha=.3); ax[1].axhline(0, color="k", lw=.5); ax[1].axvline(0, color="k", lw=.5)
    fig.tight_layout(); fig.savefig(ART / "blend_pf1_pf2.png", dpi=110)
    print(f"\nsaved {ART/'blend_pf1_pf2.png'}", flush=True)
except Exception as e:
    print("plot skip:", e, flush=True)
