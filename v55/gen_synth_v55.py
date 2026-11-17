# v55 合成井戸ジェネレータ v1(geometry re-skin 方式)
#   実ドナー井戸の軌跡・ラベル(MD,X,Y,Z,TVT,TVT_input マスク)をそのまま継承し、
#   GR だけを「歪めた地質」から再合成する:
#     G(s) = a(s)·T_GR(g(s)) + b(s)   (T=ドナーtypewell, g=厚み伸縮+ドリフトの単調warp)
#     gr(md) = G(TVT(md)) + 白色ノイズ → 分解能平滑
#   与える参照 typewell は無警告の T そのもの → 「参照は地質と少し違う」という実問題を再現。
#   ラベルは donor の TVT を真とする(warpは横方向変動=解くべき頑健性の対象)。
# ノブ(env): NSYN(合成井戸数), THICK(厚み伸縮σ, 対数勾配), WARPFT(ドリフト振幅ft),
#            AMP(ゲイン±), OFFGR(オフセットGR), NOISE(白色σGR), RES(平滑半径行), CLEN(相関長ft), SEED
# 出力: v55/artifacts_v55/synth_{TAG}/{id}__horizontal_well.csv / {id}__typewell.csv + manifest.json + 例図PNG
import os, sys, json, time
import numpy as np
import pandas as pd

P0 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN = os.path.join(P0, "rogii-wellbore-geology-prediction", "train")
TAG = os.environ.get("TAG", "v1")
OUT = os.path.join(P0, "v55", "artifacts_v55", f"synth_{TAG}")
os.makedirs(OUT, exist_ok=True)

NSYN = int(os.environ.get("NSYN", "100"))
THICK = float(os.environ.get("THICK", "0.05"))      # 厚み伸縮(log勾配σ)
WARPFT = float(os.environ.get("WARPFT", "2.0"))     # warpドリフト振幅(ft)
AMP = float(os.environ.get("AMP", "0.10"))          # GRゲイン±
OFFGR = float(os.environ.get("OFFGR", "6.0"))       # GRオフセット(API)
NOISE = float(os.environ.get("NOISE", "6.0"))       # 白色ノイズσ(API)
RES = int(os.environ.get("RES", "2"))               # 分解能平滑半径(行)
CLEN = float(os.environ.get("CLEN", "80.0"))        # 低周波相関長(ft)
SEED = int(os.environ.get("SEED", "0"))

def lowfreq(rng, s_grid, clen, sigma):
    """s_grid(ft)上の相関長clenの滑らかなガウス場(σ=sigma)。"""
    if sigma <= 0: return np.zeros_like(s_grid)
    st = np.nanmedian(np.diff(s_grid)) if len(s_grid) > 1 else 0.5
    st = 0.5 if not np.isfinite(st) or st <= 0 else st
    n = len(s_grid); r = max(int(round(clen / st)), 1)
    w = rng.standard_normal(n + 2 * r)
    k = np.exp(-0.5 * (np.arange(-3 * r, 3 * r + 1) / r) ** 2)
    f = np.convolve(w, k / np.sqrt((k ** 2).sum()), mode="same")[r:r + n]
    return sigma * f

def synth_one(rng, wn, hw, tw):
    twv = tw.sort_values("TVT")
    t_tvt = twv["TVT"].to_numpy(float); t_gr = twv["GR"].to_numpy(float)
    fin = np.isfinite(t_tvt) & np.isfinite(t_gr)
    t_tvt, t_gr = t_tvt[fin], t_gr[fin]
    if len(t_tvt) < 50: return None
    s = hw["TVT"].to_numpy(float)
    if not np.isfinite(s).all(): return None
    # 地質座標グリッド(typewell範囲+余白)
    g0, g1 = t_tvt.min() - 30, t_tvt.max() + 30
    sg = np.arange(g0, g1, 0.5)
    # 単調warp: g(s) = s + drift(s) + ∫厚み伸縮
    stretch = lowfreq(rng, sg, CLEN, THICK)                     # log伸縮勾配
    warp_cum = np.cumsum(stretch) * 0.5                         # ftに積分
    warp_cum -= np.interp(np.median(s), sg, warp_cum)           # 井戸帯でゼロ中心
    drift = lowfreq(rng, sg, CLEN * 2, WARPFT)
    gmap = sg + np.clip(warp_cum + drift, -12, 12)
    # 単調性を保証
    gmap = np.maximum.accumulate(gmap)
    a = 1.0 + lowfreq(rng, sg, CLEN, AMP)
    b = lowfreq(rng, sg, CLEN, OFFGR)
    Ggrid = a * np.interp(gmap, t_tvt, t_gr) + b                # G(s) on sg
    gr_syn = np.interp(s, sg, Ggrid) + rng.standard_normal(len(s)) * NOISE
    if RES > 0:
        gr_syn = pd.Series(gr_syn).rolling(2 * RES + 1, center=True, min_periods=1).mean().to_numpy()
    return gr_syn

def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    wells = sorted(set(f.split("__")[0] for f in os.listdir(TRAIN) if f.endswith("__horizontal_well.csv")))
    made = []; tries = 0
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "Meiryo"
    while len(made) < NSYN and tries < NSYN * 4:
        tries += 1
        wn = wells[int(rng.integers(len(wells)))]
        hw = pd.read_csv(os.path.join(TRAIN, f"{wn}__horizontal_well.csv"))
        tw = pd.read_csv(os.path.join(TRAIN, f"{wn}__typewell.csv"))
        if "TVT" not in hw.columns or hw["TVT_input"].notna().sum() < 10 or hw["TVT_input"].isna().sum() < 100:
            continue
        gr = synth_one(rng, wn, hw, tw)
        if gr is None: continue
        sid = f"sy{len(made):04d}_{wn[:8]}"
        out = hw.copy(); out["GR"] = gr
        out.to_csv(os.path.join(OUT, f"{sid}__horizontal_well.csv"), index=False)
        tw.to_csv(os.path.join(OUT, f"{sid}__typewell.csv"), index=False)
        made.append(dict(id=sid, donor=wn))
        if len(made) == 1:                                       # 例図
            fig, ax = plt.subplots(2, 1, figsize=(13, 6), sharex=True)
            md = hw["MD"].to_numpy(float)
            ax[0].plot(md, hw["GR"], lw=0.6, label="実GR(ドナー)")
            ax[0].plot(md, gr, lw=0.6, label="合成GR")
            ax[0].legend(); ax[0].set_ylabel("GR")
            ax[0].set_title(f"合成例 {sid} (donor={wn}, THICK={THICK} WARP={WARPFT}ft AMP={AMP} NOISE={NOISE})")
            ax[1].plot(md, hw["TVT"], lw=0.8, color="k"); ax[1].set_ylabel("TVT(ラベル=継承)"); ax[1].set_xlabel("MD")
            plt.tight_layout(); plt.savefig(os.path.join(OUT, "example.png"), dpi=110); plt.close()
    json.dump(dict(tag=TAG, n=len(made), knobs=dict(THICK=THICK, WARPFT=WARPFT, AMP=AMP, OFFGR=OFFGR,
                                                     NOISE=NOISE, RES=RES, CLEN=CLEN, SEED=SEED),
                   wells=made), open(os.path.join(OUT, "manifest.json"), "w"), indent=1)
    print(f"synth_{TAG}: {len(made)} wells -> {OUT} ({time.time()-t0:.0f}s)")

if __name__ == "__main__":
    main()
