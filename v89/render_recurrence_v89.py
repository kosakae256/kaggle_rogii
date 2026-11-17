# -*- coding: utf-8 -*-
"""
v89 — GR表現の視覚探索(ユーザーsteer「人が視覚追跡できる血管修復的表現」)。
=================================================================================
現行のδヒートマップでなく、MD×TVT の相似行列(リカレンス/ドットプロット)を描く。
真の経路 TVT_true(MD) は連結した"血管(尾根)"、デコイ(±サイクル)は平行バンド。
狙い: **多ソース(tw/self/nbr)の積**でデコイが抑制され、真の血管が浮くかを目で確かめる。
出力: v89/artifacts_v89/recurrence_*.png (Readして評価)
使い方: python v89/render_recurrence_v89.py
"""
import os, io, sys
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
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


def _zscore(v, r):
    import pandas as pd
    s = pd.Series(v, dtype="float64")
    m = s.rolling(2*r+1, center=True, min_periods=1).mean()
    sd = s.rolling(2*r+1, center=True, min_periods=1).std().fillna(1.0).clip(lower=1e-3)
    return ((s - m) / sd).to_numpy()

def sim_image(obs, ref, mask, sig, tvt_g, ycol, mode="raw", rz=12):
    """R[x=MD, y=TVT]。mode=raw:絶対GR差 / shape:局所z-score(形)差。mask未被覆はNaN。"""
    cv = np.interp(ycol, tvt_g, mask)
    if mode == "shape":
        # 参照は自分の格子(TVT)でz-score、観測はMDでz-score。形(wiggle)を比べる。
        ref_z_grid = _zscore(ref, rz)
        rv = np.interp(ycol, tvt_g, ref_z_grid)
        ov = _zscore(obs, rz)
        d = (ov[:, None] - rv[None, :])              # z空間の差(σ=1相当)
        S = np.exp(-0.5 * (d / 1.0) ** 2)
    else:
        rv = np.interp(ycol, tvt_g, ref)
        d = (obs[:, None] - rv[None, :]) / sig
        S = np.exp(-0.5 * d * d)
    S[:, cv < 0.5] = np.nan
    return S


def main():
    wells = base.list_wells()
    cache = {}
    for w in wells[:400]:
        c = base.build_cache(w)
        if c is not None and c["tvt_true"] is not None and w in PFC:
            cache[w] = c
    # PF誤差で easy/hard を選ぶ
    errs = {}
    for w, c in cache.items():
        p = PFC[w][:len(c["tvt_true"])]; t = np.asarray(c["tvt_true"], float)[:len(p)]
        e = p - t; e = e[np.isfinite(e)]
        errs[w] = float(np.sqrt(np.mean(e ** 2))) if len(e) else 1e9
    order = sorted(cache, key=lambda w: errs[w])
    picks = [order[3], order[len(order)//2], order[-8], order[-3]]   # easy, mid, hard, hard
    tags = ["easy", "mid", "hard", "hard2"]

    reps = ["tw生値", "tw形", "self形", "nbr形", "tw×self×nbr形"]
    fig, axes = plt.subplots(len(picks), len(reps), figsize=(4 * len(reps), 3.2 * len(picks)))
    for r, w in enumerate(picks):
        c = cache[w]; x = v88.make_inputs(c, P)
        gmin, gst = x["gmin"], x["gst"]; G = x["ch_ref"].shape[1]
        tvt_full = gmin + gst * np.arange(G)
        t = np.asarray(c["tvt_true"], float); pf = PFC[w][:len(t)]
        obs = np.asarray(x["ch_obs"][0], float)
        n = min(len(obs), len(t), len(pf))
        obs, t, pf = obs[:n], t[:n], pf[:n]
        # y範囲は真値±60ft(見やすさ)
        y0, y1 = np.nanmin(t) - 40, np.nanmax(t) + 40
        ycol = np.arange(y0, y1, 0.5)
        xsel = np.linspace(0, n - 1, min(n, 500)).astype(int)
        def si_of(si, mode):
            if si < x["ch_ref"].shape[0]:
                return sim_image(obs[xsel], np.asarray(x["ch_ref"][si], float),
                                 np.asarray(x["ch_mask"][si], float), float(x["ch_sig"][si]), tvt_full, ycol, mode)
            return np.full((len(xsel), len(ycol)), np.nan)
        tws_sh = si_of(0, "shape"); slf_sh = si_of(1, "shape"); nbr_sh = si_of(2, "shape")
        def prod(imgs):
            acc = np.ones((len(xsel), len(ycol)))
            for im in imgs: acc = acc * np.where(np.isfinite(im), im, 1.0)
            return acc
        img = {"tw生値": si_of(0, "raw"), "tw形": tws_sh, "self形": slf_sh, "nbr形": nbr_sh,
               "tw×self×nbr形": prod([tws_sh, slf_sh, nbr_sh])}
        for cc, nm in enumerate(reps):
            ax = axes[r, cc] if len(picks) > 1 else axes[cc]
            S = img[nm]
            ax.imshow(S.T, aspect="auto", origin="lower", extent=[0, n, y0, y1],
                      cmap="magma", vmin=0, vmax=np.nanpercentile(S, 99) if np.isfinite(S).any() else 1)
            ax.plot(xsel / len(xsel) * n * (n - 1) / max(n - 1, 1), t[xsel], "c-", lw=0.8, label="true")
            ax.plot(np.arange(n), t, "c-", lw=0.7)
            ax.plot(np.arange(n), pf, "w--", lw=0.6, alpha=0.7)
            if r == 0: ax.set_title(nm, fontsize=10)
            if cc == 0: ax.set_ylabel(f"{tags[r]}\nPF_rmse={errs[w]:.1f}\nTVT", fontsize=8)
            ax.tick_params(labelsize=6)
    plt.suptitle("v89 GR相似行列(リカレンス): 真経路=シアン, PF=白破線。tw/self/nbrの積でデコイ抑制されるか", fontsize=11)
    plt.tight_layout()
    outp = PROJECT / "v89" / "artifacts_v89" / "recurrence_shape.png"
    plt.savefig(outp, dpi=85); print(f"saved {outp}")
    for w, tg in zip(picks, tags): print(f"  {tg}: {w} PF_rmse={errs[w]:.2f}")


if __name__ == "__main__":
    main()
