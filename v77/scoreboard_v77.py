# -*- coding: utf-8 -*-
"""
v77 共通スコアボード — メソッド出力 vs 素PF(bare0) の standalone/ρ/blend判定
使い方: python v77/scoreboard_v77.py v77/artifacts_v77/p6_out_60w.pkl [pred|soft]
pklは {wid: {pred|soft(np.f32 T,), ...}} 形式(TVT予測)。真値・素PFは seed_dump_bare0 から。
"""
import io, sys, pickle
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
from pathlib import Path
import numpy as np

P0 = Path(r"C:\Users\kosaka256\Documents\rogii_claude")
LS = 10.536319667317654
fp = Path(sys.argv[1]); key = sys.argv[2] if len(sys.argv) > 2 else None
outs = pickle.loads(fp.read_bytes())
bare = pickle.loads((P0 / "v73" / "artifacts_v73" / "seed_dump_bare0.pkl").read_bytes())

e_m, e_p = [], []
for wid, o in outs.items():
    if wid not in bare: continue
    d = bare[wid]
    ll = d["ll"].astype(np.float64)
    ww = np.exp((ll - ll.max()) / LS); ww /= ww.sum()
    epf = (ww[:, None] * d["err"].astype(np.float64)).sum(0)      # 素PF誤差(T,)
    tv = d["true"]
    k = key or ("pred" if "pred" in o else "soft")
    pr = np.asarray(o[k], dtype=np.float64)
    T = min(len(pr), len(tv), len(epf))
    em = pr[:T] - tv[:T]
    ep = epf[:T]
    m = np.isfinite(em) & np.isfinite(ep)
    e_m.append(em[m]); e_p.append(ep[m])
em = np.concatenate(e_m); ep = np.concatenate(e_p)
s_m = np.sqrt(np.mean(em ** 2)); s_p = np.sqrt(np.mean(ep ** 2))
rho = float(np.corrcoef(em, ep)[0, 1])
print(f"対象 {len(e_m)}井 / {len(em)}行   [{fp.name}]")
print(f"  method standalone = {s_m:.4f}   素PF = {s_p:.4f}   ρ = {rho:.3f}")
print(f"  blend成立条件 ρ<σ小/σ大 = {min(s_m,s_p)/max(s_m,s_p):.3f} → {'成立' if rho < min(s_m,s_p)/max(s_m,s_p) else '不成立(w=0)'}")
ws = np.linspace(0, 1, 101)
best = min(((float(np.sqrt(np.mean(((1 - w) * ep + w * em) ** 2))), float(w)) for w in ws))
print(f"  最適blend: w_method={best[1]:.2f} → {best[0]:.4f}  (素PF比 {best[0]-s_p:+.4f})")
