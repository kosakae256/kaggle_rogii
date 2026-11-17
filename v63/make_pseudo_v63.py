# -*- coding: utf-8 -*-
# v63 pseudo(合成井戸)生成器 — train の 3D 面マッピングを利用
# =========================================================================
# 2モード:
#   surface: train全坑井の (X,Y)→各フォーメーション面(ANCC..BUDA)を格子ラスタ化した「3D面モデル」から
#            新しい位置(donor軌跡+XYジッタ)の地層トップを再構成し、TVT真値を v56 の恒等式
#            TVT = (Z − ANCC(x,y)) + b で作る。面には**共有の独立残差場 R(x,y)**(相関長RES_CLEN,
#            振幅RES_FT)を加算 = 「近傍が知らない短波長起伏」を再現 → nbr特徴の情報量が現実的になる。
#   reskin : v55 方式。donor の軌跡・TVT真値をそのまま継承し GR だけ再合成(完全に正直・リーク無し)。
# GRノイズ: 「完璧か/ノイズありか不明」→ 混合。clean/real/high の3水準を NOISE_MIX の比率で
#           井戸ごとに抽選し、manifest に記録(後で水準別にablation可能)。
# neighbor規律: create_v63.py が manifest の donor を近傍プールから除外する。
#           surface型の真値は「面の平滑成分(近傍由来)+R(近傍が知らない)」なので nbr は
#           現実同様「役立つが完全ではない」情報になる。
# fold規律: 学習側で pseudo は donor の fold に帰属させる(donorがvalのfoldでは使わない)。
# -------------------------------------------------------------------------
# env: TAG(p1) NSYN(200) MODE(mix|surface|reskin) SURF_FRAC(0.7) PLACE(jitter|donor) JIT_FT(1500)
#      CELL_FT(250) RES_FT(5.0) RES_CLEN(900) NOISE_LVLS("1.5,6.0,11.0") NOISE_MIX("0.25,0.5,0.25")
#      THICK(0.05) WARPFT(2.0) AMP(0.10) OFFGR(6.0) CLEN(80) RES(2) SEED(0)
# 出力: v63/artifacts_v63/pseudo_wells_{TAG}/{id}__horizontal_well.csv / {id}__typewell.csv
#       + manifest.json + example_*.png
# 実行: .venv/Scripts/python.exe v63/make_pseudo_v63.py
import os, sys, json, time
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, binary_dilation

sys.stdout = open(1, "w", encoding="utf-8", closefd=False)
P0 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN = os.path.join(P0, "rogii-wellbore-geology-prediction", "train")
TAG = os.environ.get("TAG", "p1")
OUT = os.path.join(P0, "v63", "artifacts_v63", f"pseudo_wells_{TAG}")
os.makedirs(OUT, exist_ok=True)

NSYN = int(os.environ.get("NSYN", "300"))
MODE = os.environ.get("MODE", "mix")                 # mix | surface | reskin
SURF_FRAC = float(os.environ.get("SURF_FRAC", "0.7"))
PLACE = os.environ.get("PLACE", "jitter")            # jitter | donor
JIT_FT = float(os.environ.get("JIT_FT", "1500"))
CELL = float(os.environ.get("CELL_FT", "250"))
RES_FT = float(os.environ.get("RES_FT", "5.0"))      # 面の独立残差(近傍が知らない起伏)σ
RES_CLEN = float(os.environ.get("RES_CLEN", "900"))  # 残差の相関長 ft
NOISE_LVLS = [float(x) for x in os.environ.get("NOISE_LVLS", "1.5,6.0,11.0").split(",")]
NOISE_MIX = [float(x) for x in os.environ.get("NOISE_MIX", "0.25,0.5,0.25").split(",")]
THICK = float(os.environ.get("THICK", "0.05"))       # GR warp: 厚み伸縮(log勾配σ)
WARPFT = float(os.environ.get("WARPFT", "2.0"))      # GR warp: ドリフト振幅 ft
AMP = float(os.environ.get("AMP", "0.10"))           # GR ゲイン±
OFFGR = float(os.environ.get("OFFGR", "6.0"))        # GR オフセット(API)
CLEN = float(os.environ.get("CLEN", "80.0"))         # GR warp の相関長(地質座標 ft)
RES = int(os.environ.get("RES", "2"))                # 分解能平滑半径(行)
SEED = int(os.environ.get("SEED", "0"))
FORMS = ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]


def lowfreq(rng, s_grid, clen, sigma):
    """s_grid(ft)上の相関長clenの滑らかなガウス場(σ=sigma)。v55と同一。"""
    if sigma <= 0: return np.zeros_like(s_grid)
    st = np.nanmedian(np.diff(s_grid)) if len(s_grid) > 1 else 0.5
    st = 0.5 if not np.isfinite(st) or st <= 0 else st
    n = len(s_grid); r = max(int(round(clen / st)), 1)
    w = rng.standard_normal(n + 2 * r)
    k = np.exp(-0.5 * (np.arange(-3 * r, 3 * r + 1) / r) ** 2)
    f = np.convolve(w, k / np.sqrt((k ** 2).sum()), mode="same")[r:r + n]
    return sigma * f


# ==================== 3D 面モデル(格子ラスタ) ====================
def build_surfaces():
    wells = sorted(set(f.split("__")[0] for f in os.listdir(TRAIN) if f.endswith("__horizontal_well.csv")))
    pts = []
    for w in wells:
        try:
            h = pd.read_csv(os.path.join(TRAIN, f"{w}__horizontal_well.csv"), usecols=["X", "Y"] + FORMS)
        except Exception:
            continue
        pts.append(h.iloc[::8].to_numpy(float))
    P = np.concatenate(pts, 0)
    x0, x1 = np.nanmin(P[:, 0]), np.nanmax(P[:, 0]); y0, y1 = np.nanmin(P[:, 1]), np.nanmax(P[:, 1])
    nx = int((x1 - x0) / CELL) + 3; ny = int((y1 - y0) / CELL) + 3
    ix = np.clip(((P[:, 0] - x0) / CELL).astype(int) + 1, 0, nx - 1)
    iy = np.clip(((P[:, 1] - y0) / CELL).astype(int) + 1, 0, ny - 1)
    S = {}; cnt = np.zeros((ny, nx))
    np.add.at(cnt, (iy, ix), 1.0)
    for j, f in enumerate(FORMS):
        v = P[:, 2 + j]; m = np.isfinite(v)
        ssum = np.zeros((ny, nx)); scnt = np.zeros((ny, nx))
        np.add.at(ssum, (iy[m], ix[m]), v[m]); np.add.at(scnt, (iy[m], ix[m]), 1.0)
        g = np.where(scnt > 0, ssum / np.maximum(scnt, 1), np.nan)
        # 穴埋め: 近傍平均の反復拡張 → その後ガウス平滑(有効マスク重み)
        filled = g.copy()
        for _ in range(400):
            nanm = ~np.isfinite(filled)
            if not nanm.any(): break
            vz = np.where(np.isfinite(filled), filled, 0.0); vm = np.isfinite(filled).astype(float)
            sm_v = gaussian_filter(vz, 1.2); sm_m = gaussian_filter(vm, 1.2)
            cand = np.where(sm_m > 1e-6, sm_v / np.maximum(sm_m, 1e-6), np.nan)
            filled[nanm] = cand[nanm]
        vz = np.where(np.isfinite(filled), filled, 0.0)
        S[f] = gaussian_filter(vz, 1.0)
        S[f + "_obs"] = scnt > 0
    hull = binary_dilation(cnt > 0, iterations=4)   # データ近傍のみ配置可
    print(f"[surfaces] grid {ny}x{nx} cell={CELL}ft wells={len(wells)} pts={len(P)}"
          f" ANCC観測セル={int(S['ANCC_obs'].sum())}")
    return S, hull, (x0, y0, CELL, nx, ny), wells


def make_resfield(rng, shape):
    """共有の独立残差場 R(x,y): 相関長RES_CLEN・σ=RES_FT。地質の「近傍が知らない起伏」。"""
    sig_c = max(RES_CLEN / CELL, 0.8)
    f = gaussian_filter(rng.standard_normal(shape), sig_c)
    sd = f.std()
    return f / max(sd, 1e-9) * RES_FT


def sampler(geo):
    x0, y0, cell, nx, ny = geo
    def samp(grid, x, y):
        fx = (np.asarray(x, float) - x0) / cell + 1.0
        fy = (np.asarray(y, float) - y0) / cell + 1.0
        i0 = np.floor(fx).astype(int); j0 = np.floor(fy).astype(int)
        ok = (i0 >= 0) & (i0 < nx - 1) & (j0 >= 0) & (j0 < ny - 1)
        i0c = np.clip(i0, 0, nx - 2); j0c = np.clip(j0, 0, ny - 2)
        tx = fx - i0c; ty = fy - j0c
        v = (grid[j0c, i0c] * (1 - tx) * (1 - ty) + grid[j0c, i0c + 1] * tx * (1 - ty)
             + grid[j0c + 1, i0c] * (1 - tx) * ty + grid[j0c + 1, i0c + 1] * tx * ty)
        return np.where(ok, v, np.nan)
    def inhull(hullgrid, x, y):
        i = np.clip(((np.asarray(x, float) - x0) / cell).astype(int) + 1, 0, nx - 1)
        j = np.clip(((np.asarray(y, float) - y0) / cell).astype(int) + 1, 0, ny - 1)
        return hullgrid[j, i]
    return samp, inhull


# ==================== GR 合成(v55 コア: 真値TVT列に対して) ====================
def synth_gr(rng, truth_tvt, tw, sigma_noise):
    twv = tw.sort_values("TVT")
    t_tvt = twv["TVT"].to_numpy(float); t_gr = twv["GR"].to_numpy(float)
    fin = np.isfinite(t_tvt) & np.isfinite(t_gr)
    t_tvt, t_gr = t_tvt[fin], t_gr[fin]
    if len(t_tvt) < 50: return None
    g0, g1 = t_tvt.min() - 30, t_tvt.max() + 30
    sg = np.arange(g0, g1, 0.5)
    stretch = lowfreq(rng, sg, CLEN, THICK)
    warp_cum = np.cumsum(stretch) * 0.5
    warp_cum -= np.interp(np.median(truth_tvt), sg, warp_cum)
    drift = lowfreq(rng, sg, CLEN * 2, WARPFT)
    gmap = np.maximum.accumulate(sg + np.clip(warp_cum + drift, -12, 12))
    a = 1.0 + lowfreq(rng, sg, CLEN, AMP)
    b = lowfreq(rng, sg, CLEN, OFFGR)
    Ggrid = a * np.interp(gmap, t_tvt, t_gr) + b
    gr = np.interp(truth_tvt, sg, Ggrid) + rng.standard_normal(len(truth_tvt)) * sigma_noise
    if RES > 0:
        gr = pd.Series(gr).rolling(2 * RES + 1, center=True, min_periods=1).mean().to_numpy()
    return gr


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    S, hull, geo, wells = build_surfaces()
    samp, inhull = sampler(geo)
    Rf = make_resfield(rng, S["ANCC"].shape)
    made = []; tries = 0
    rej = {"precond": 0, "ancc_nan": 0, "cover": 0, "gr": 0}
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "Meiryo"
    n_ex = 0
    while len(made) < NSYN and tries < NSYN * 8:
        tries += 1
        wn = wells[int(rng.integers(len(wells)))]
        try:
            hw = pd.read_csv(os.path.join(TRAIN, f"{wn}__horizontal_well.csv"))
            tw = pd.read_csv(os.path.join(TRAIN, f"{wn}__typewell.csv"))
        except Exception:
            continue
        if "TVT" not in hw.columns or hw["TVT_input"].notna().sum() < 10 or hw["TVT_input"].isna().sum() < 100:
            rej["precond"] += 1; continue
        if not np.isfinite(hw["TVT"].to_numpy(float)).all():
            rej["precond"] += 1; continue
        mode = ("surface" if (MODE == "surface" or (MODE == "mix" and rng.random() < SURF_FRAC))
                else "reskin") if MODE != "reskin" else "reskin"
        X = hw["X"].to_numpy(float); Y = hw["Y"].to_numpy(float); Z = hw["Z"].to_numpy(float)
        cat = int(rng.choice(len(NOISE_LVLS), p=np.array(NOISE_MIX) / sum(NOISE_MIX)))
        sigma = NOISE_LVLS[cat]
        out = hw.copy(); dxy = (0.0, 0.0)
        if mode == "surface":
            # --- 配置(XYジッタ, データ近傍hull内) ---
            for _ in range(25):
                d = (rng.uniform(-JIT_FT, JIT_FT), rng.uniform(-JIT_FT, JIT_FT)) if PLACE == "jitter" else (0., 0.)
                if inhull(hull, X + d[0], Y + d[1]).all():
                    dxy = d; break
            xs, ys = X + dxy[0], Y + dxy[1]
            tops = {f: samp(S[f], xs, ys) + samp(Rf, xs, ys) for f in FORMS}
            if not np.isfinite(tops["ANCC"]).all():
                rej["ancc_nan"] += 1; continue
            # v56恒等式: TVT = (ANCC − Z) + b(実測で行間std≈0.01ft の厳密恒等式)
            b_dat = float(np.nanmedian(hw["TVT"].to_numpy(float) - (hw["ANCC"].to_numpy(float) - Z)))
            truth = (tops["ANCC"] - Z) + b_dat
            # typewell被覆ガード(PF格子外に出る合成は棄却)
            twlo, twhi = tw["TVT"].min() - 20, tw["TVT"].max() + 20
            if np.mean((truth >= twlo) & (truth <= twhi)) < 0.95:
                rej["cover"] += 1; continue
            for f in FORMS: out[f] = tops[f]
            out["TVT"] = truth
            out["TVT_input"] = np.where(hw["TVT_input"].notna(), truth, np.nan)
            gr = synth_gr(rng, truth, tw, sigma)
        else:
            truth = hw["TVT"].to_numpy(float)
            gr = synth_gr(rng, truth, tw, sigma)
        if gr is None or not np.isfinite(gr).all():
            rej["gr"] += 1; continue
        gr = np.where(np.isfinite(hw["GR"].to_numpy(float)), gr, np.nan)   # donorのGR欠測パターンを継承
        out["GR"] = gr
        sid = f"ps{len(made):04d}_{wn[:8]}"
        out.to_csv(os.path.join(OUT, f"{sid}__horizontal_well.csv"), index=False)
        tw.to_csv(os.path.join(OUT, f"{sid}__typewell.csv"), index=False)
        made.append(dict(id=sid, donor=wn, mode=mode, dxy=[round(dxy[0], 1), round(dxy[1], 1)],
                         noise_cat=["clean", "real", "high"][cat], noise_sigma=sigma))
        if n_ex < 2:                                             # 例図
            fig, ax = plt.subplots(2, 1, figsize=(13, 6), sharex=True)
            md = hw["MD"].to_numpy(float)
            ax[0].plot(md, hw["GR"], lw=0.5, alpha=0.8, label="実GR(donor)")
            ax[0].plot(md, gr, lw=0.5, alpha=0.8, label=f"合成GR(σ={sigma})")
            ax[0].legend(); ax[0].set_ylabel("GR")
            ax[0].set_title(f"{sid} mode={mode} donor={wn} dxy={dxy} noise={made[-1]['noise_cat']}")
            ax[1].plot(md, hw["TVT"], lw=0.8, label="donor TVT")
            ax[1].plot(md, out["TVT"], lw=0.8, label="pseudo TVT(真値)")
            ax[1].legend(); ax[1].set_ylabel("TVT"); ax[1].set_xlabel("MD")
            plt.tight_layout(); plt.savefig(os.path.join(OUT, f"example_{n_ex}.png"), dpi=110); plt.close()
            n_ex += 1
    json.dump(dict(tag=TAG, nsyn=len(made), mode=MODE, surf_frac=SURF_FRAC, place=PLACE, jit_ft=JIT_FT,
                   cell_ft=CELL, res_ft=RES_FT, res_clen=RES_CLEN,
                   noise_lvls=NOISE_LVLS, noise_mix=NOISE_MIX,
                   thick=THICK, warpft=WARPFT, amp=AMP, offgr=OFFGR, clen=CLEN, res=RES, seed=SEED,
                   wells=made),
              open(os.path.join(OUT, "manifest.json"), "w"), indent=1)
    nsurf = sum(1 for m in made if m["mode"] == "surface")
    print(f"pseudo_{TAG}: {len(made)} wells (surface {nsurf} / reskin {len(made)-nsurf})"
          f" 棄却={rej} -> {OUT} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
