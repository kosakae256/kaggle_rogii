# -*- coding: utf-8 -*-
"""v30 3者ブレンド走査: out = wp·prior + wn·素smoother(sig=none) + wc·閉ループ(sig=SIG_CL)
単純グリッド(0.05刻み, wp+wn+wc=1)で overall / wells>10 を出す。npzから数秒。
実行: rogii_claude/.venv/Scripts/python.exe v30/blend3_v30.py
"""
import json
from pathlib import Path
import numpy as np

import closed_loop_v30 as cl

SIG_CL = "96"     # 閉ループ側に使う σ キー(preds_sig{SIG_CL}.npz)。64/128を回したら差し替えて再実行
STEP = 0.05

def main():
    wells = cl.list_wells()
    cache = {}
    for wid in wells:
        c = cl.build_cache(wid)
        if c is not None and c["tvt_true"] is not None:
            cache[wid] = c
    priors, meta = cl.build_priors(cache)
    wl = [w for w in wells if w in priors]
    zn = np.load(cl.OUT_DIR / "preds_signone.npz", allow_pickle=True)
    zc = np.load(cl.OUT_DIR / f"preds_sig{SIG_CL}.npz", allow_pickle=True)

    P = np.concatenate([priors[w] for w in wl])
    Sm = np.concatenate([zn[f"s_{w}"] for w in wl])
    Cl = np.concatenate([zc[f"s_{w}"] for w in wl])
    T = np.concatenate([cache[w]["tvt_true"] for w in wl])
    fin = np.isfinite(P) & np.isfinite(Sm) & np.isfinite(Cl) & np.isfinite(T)
    lens = [len(priors[w]) for w in wl]
    edges = np.cumsum([0] + lens)

    def overall(pred):
        e = (pred - T)[fin]; return float(np.sqrt(np.mean(e ** 2)))
    def over10(pred):
        n = 0
        for i in range(len(wl)):
            e = (pred - T)[edges[i]:edges[i + 1]]
            e = e[np.isfinite(e)]
            if e.size and np.sqrt(np.mean(e ** 2)) > 10: n += 1
        return n

    print(f"prior={overall(P):.4f}  smoother(none)={overall(Sm):.4f}  cl{SIG_CL}={overall(Cl):.4f}")
    best = None
    for wn in np.arange(0.0, 1.0 + 1e-9, STEP):
        for wc in np.arange(0.0, 1.0 - wn + 1e-9, STEP):
            wp = 1.0 - wn - wc
            sc = overall(wp * P + wn * Sm + wc * Cl)
            if best is None or sc < best[0]:
                best = (sc, wp, wn, wc)
    sc, wp, wn, wc = best
    pred = wp * P + wn * Sm + wc * Cl
    print(f"★3者best: prior={wp:.2f} smoother={wn:.2f} cl{SIG_CL}={wc:.2f}"
          f" → overall={sc:.4f}  >10={over10(pred)}")
    # 参考: 2者(prior×各)のbest
    for name, X in [("smoother", Sm), (f"cl{SIG_CL}", Cl)]:
        b2 = min((overall((1 - b) * P + b * X), float(b)) for b in np.arange(0, 1.001, 0.05))
        print(f"  2者 prior×{name}: β={b2[1]:.2f} → {b2[0]:.4f}")

if __name__ == "__main__":
    main()
