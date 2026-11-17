# -*- coding: utf-8 -*-
"""
v89 — 血管(vessel)表現の視覚探索 v2。各MD列で正規化して尾根を全域で可視化。
================================================================================
前版の反省: prefix帯が高コントラストでvmaxを支配し eval帯が潰れて見えなかった。
→ **各MD列(bin)ごとに [0,1] 正規化**(その位置での相対的な合致度)。これで自己相似の
   平行バンド(デコイ)と真の血管が全域で見える。形(局所z-score)表現 × tw/self/nbr。
使い方: python v89/render_vessel_v89.py
"""
import os, io, sys
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams["font.family"] = "Meiryo"

PROJECT = Path(r"C:\Users\kosaka256\Documents\rogii_claude")
sys.path.insert(0, str(PROJECT / "v65")); sys.path.insert(0, str(PROJECT / "v88"))
os.environ["V65_TWSRC"] = "tw+self+nbr"
import pf_engine_v65 as base
import pf_engine_v88 as v88
import warmup_v65 as wu

P = v88.full_params(dict(wu.R0_PARAMS)); P["use_anchor"] = False; P["n_seeds"] = 32
P["tw_source"] = "tw"; P["v88_src"] = "tw+self+nbr"; P["v88_self_fill"] = 20.0
PFC = dict(np.load(PROJECT / "v89" / "artifacts_v89" / "pf_center_v89.npz"))
NCOL = 360


def zscore(v, r):
    s = pd.Series(np.asarray(v, float))
    m = s.rolling(2 * r + 1, center=True, min_periods=1).mean()
    sd = s.rolling(2 * r + 1, center=True, min_periods=1).std().fillna(1.0).clip(lower=1e-3)
    return ((s - m) / sd).to_numpy()


def col_image(obs_b, ref_z, mask, tvt_g, ycol, sig_z=1.0):
    """各列(MD bin) obs_b vs ref_z(y)。列ごと[0,1]正規化。"""
    rv = np.interp(ycol, tvt_g, ref_z); cv = np.interp(ycol, tvt_g, mask)
    S = np.exp(-0.5 * ((obs_b[:, None] - rv[None, :]) / sig_z) ** 2)   # (col,y)
    S[:, cv < 0.5] = np.nan
    mn = np.nanmin(S, 1, keepdims=True); mx = np.nanmax(S, 1, keepdims=True)
    return (S - mn) / np.maximum(mx - mn, 1e-9)


def main():
    wells = base.list_wells()
    cache = {}
    for w in wells[:300]:
        c = base.build_cache(w)
        if c is not None and c["tvt_true"] is not None and w in PFC: cache[w] = c
    errs = {}
    for w, c in cache.items():
        p = PFC[w][:len(c["tvt_true"])]; t = np.asarray(c["tvt_true"], float)[:len(p)]
        e = (p - t); e = e[np.isfinite(e)]; errs[w] = float(np.sqrt(np.mean(e**2))) if len(e) else 1e9
    order = sorted(cache, key=lambda w: errs[w])
    picks = [order[3], order[len(order)//2], order[-8], order[-3]]; tags = ["easy", "mid", "hard", "hard2"]
    reps = ["tw形", "self形", "nbr形", "tw+self+nbr形(平均)"]
    fig, axes = plt.subplots(len(picks), len(reps), figsize=(4.2*len(reps), 3.3*len(picks)))
    for r, w in enumerate(picks):
        c = cache[w]; x = v88.make_inputs(c, P)
        gmin, gst = x["gmin"], x["gst"]; G = x["ch_ref"].shape[1]; tvt_full = gmin + gst*np.arange(G)
        t = np.asarray(c["tvt_true"], float); pf = PFC[w][:len(t)]
        obs = np.asarray(x["ch_obs"][0], float); n = min(len(obs), len(t), len(pf))
        obs, t, pf = obs[:n], t[:n], pf[:n]
        obs_z = zscore(obs, 12)
        # MD bin
        edges = np.linspace(0, n, NCOL + 1).astype(int)
        ob_b = np.array([np.nanmean(obs_z[a:b]) if b > a else np.nan for a, b in zip(edges[:-1], edges[1:])])
        t_b = np.array([np.nanmedian(t[a:b]) if b > a else np.nan for a, b in zip(edges[:-1], edges[1:])])
        pf_b = np.array([np.nanmedian(pf[a:b]) if b > a else np.nan for a, b in zip(edges[:-1], edges[1:])])
        y0, y1 = np.nanmin(t)-35, np.nanmax(t)+35; ycol = np.arange(y0, y1, 0.5)
        imgs = {}
        for si, nm in enumerate(["tw", "self", "nbr"]):
            if si < x["ch_ref"].shape[0]:
                rz = zscore(np.asarray(x["ch_ref"][si], float), 12)
                imgs[nm] = col_image(ob_b, rz, np.asarray(x["ch_mask"][si], float), tvt_full, ycol)
            else:
                imgs[nm] = np.full((NCOL, len(ycol)), np.nan)
        comb = np.nanmean(np.stack([imgs["tw"], imgs["self"], imgs["nbr"]]), 0)
        show = {"tw形": imgs["tw"], "self形": imgs["self"], "nbr形": imgs["nbr"], "tw+self+nbr形(平均)": comb}
        xc = (edges[:-1] + edges[1:]) / 2
        for cc, nm in enumerate(reps):
            ax = axes[r, cc]
            ax.imshow(show[nm].T, aspect="auto", origin="lower", extent=[0, n, y0, y1], cmap="magma", vmin=0, vmax=1)
            ax.plot(xc, t_b, "c-", lw=1.0); ax.plot(xc, pf_b, "w--", lw=0.7, alpha=0.8)
            if r == 0: ax.set_title(nm, fontsize=10)
            if cc == 0: ax.set_ylabel(f"{tags[r]} PF={errs[w]:.1f}\nTVT", fontsize=8)
            ax.tick_params(labelsize=6)
    plt.suptitle("v89 血管表現(形×列正規化): 真=シアン, PF=白破線。真の血管が連結尾根で見えるか/デコイ平行バンドと分離できるか", fontsize=11)
    plt.tight_layout()
    outp = PROJECT / "v89" / "artifacts_v89" / "vessel.png"
    plt.savefig(outp, dpi=85); print(f"saved {outp}")
    for w, tg in zip(picks, tags): print(f"  {tg}: {w} obs_finite={np.isfinite(cache[w]['tvt_true']).mean():.2f} PF={errs[w]:.2f}")


if __name__ == "__main__":
    main()
