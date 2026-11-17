# -*- coding: utf-8 -*-
"""
v73 段g — seed合成のオフライン掃引(nnpf MODE=dump の出力を消費)
================================================================
現行: ww ∝ exp((ll−max)/τ), τ=likelihood_scale=10.54(素のr0時代にOptunaが決めた値)。
チェーン(a+b)が変わった今、τ最適が動いている可能性が高い。PF再走行なしで掃引:
  1) τグリッド  2) top-k(ll上位kのみで合成; k=1=ベスト単独)  3) 一様平均
使い方: DUMPTAG=champ python v73/g_sweep_v73.py
"""
import os, io, sys, pickle
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
from pathlib import Path
import numpy as np

ART = Path(r"C:\Users\kosaka256\Documents\rogii_claude\v73\artifacts_v73")
TAG = os.environ.get("DUMPTAG", "champ")
D = pickle.loads((ART / f"seed_dump_{TAG}.pkl").read_bytes())
wells = sorted(D.keys())
print(f"dump={TAG}  {len(wells)}井", flush=True)


def overall(fn):
    es = []
    for w in wells:
        d = D[w]; ps = d["ps"].astype(np.float32); ll = d["ll"]; tv = d["true"]
        comb = fn(ps, ll)
        e = comb[:len(tv)] - tv; es.append(e[np.isfinite(e)])
    e = np.concatenate(es)
    return float(np.sqrt(np.mean(e ** 2)))


def soft(ps, ll, tau):
    m = np.isfinite(ll)
    if not m.any(): return ps.mean(0)
    mx = np.nanmax(ll[m])
    lw = np.where(m, ll - mx, -np.inf)
    ww = np.exp(lw / max(tau, 1e-6)); s = ww.sum()
    ww = ww / s if s > 1e-300 else np.full(len(ll), 1.0 / len(ll))
    return (ww[:, None] * ps).sum(0)


def topk(ps, ll, k, tau=None):
    idx = np.argsort(np.where(np.isfinite(ll), ll, -np.inf))[::-1][:k]
    if tau is None:
        return ps[idx].mean(0)
    return soft(ps[idx], ll[idx], tau)


print("\n=== τ(温度)掃引 [現行 τ=10.54] ===")
for tau in (2, 5, 8, 10.54, 15, 25, 50, 1e9):
    lbl = "uniform" if tau >= 1e8 else f"τ={tau:g}"
    print(f"  {lbl:>9}: {overall(lambda ps, ll, t=tau: soft(ps, ll, t)):.4f}", flush=True)
print("\n=== top-k(ll上位のみ, 一様平均) ===")
for k in (1, 2, 4, 8, 16, 24):
    print(f"  k={k:>2}: {overall(lambda ps, ll, kk=k: topk(ps, ll, kk)):.4f}", flush=True)
print("\n=== top-k × τ ===")
for k, tau in ((8, 5), (8, 10.54), (16, 5), (16, 10.54), (24, 10.54)):
    print(f"  k={k:>2} τ={tau:g}: {overall(lambda ps, ll, kk=k, t=tau: topk(ps, ll, kk, t)):.4f}", flush=True)
