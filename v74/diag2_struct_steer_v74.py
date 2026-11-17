# v74 診断2 — struct場(GRフリー低周波の現状最強~9.7)× 操舵高周波 κ·HP_W(Z) の合成プローブ
#   pred_g(W,κ) = LP_W(g_struct) + κ·HP_W(Z),  TVT̂ = pred_g − Z
#   LP は既知帯の真g(合法)と評価帯の g_struct を連結してから取る(境界アーティファクト回避)。
#   併せて struct 誤差の LP/HP 分解(structが高周波をどう扱っているかの実測)。
# 入力: v38/artifacts_v38/priors/prior_v66_leanv1cumul_train.pkl (per-well: tvt,sig @評価行全行)
# 出力: v74/artifacts_v74/diag2_struct_steer.{json,png}
# env: NW(0=全)
import os, json, time, pickle
import numpy as np
import pandas as pd

P0 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN = os.path.join(P0, "rogii-wellbore-geology-prediction", "train")
PKL = os.path.join(P0, "v38", "artifacts_v38", "priors", "prior_v66_leanv1cumul_train.pkl")
ART = os.path.join(P0, "v74", "artifacts_v74"); os.makedirs(ART, exist_ok=True)
NW = int(os.environ.get("NW", "0"))
WINS = [301, 1001, 2001, 4001]
KS = np.round(np.arange(0.0, 1.41, 0.05), 2)

def lowpass(x, win):
    return pd.Series(x).rolling(win, center=True, min_periods=1).mean().to_numpy()

def main():
    t0 = time.time()
    with open(PKL, "rb") as f:
        struct = pickle.load(f)
    wells = sorted(set(f.split("__")[0] for f in os.listdir(TRAIN)
                       if f.endswith("__horizontal_well.csv")))
    if NW:
        wells = wells[:NW]
    ES = []                      # struct as-is 誤差
    pool = {W: dict(r=[], zh=[], sh=[], glo_err=[], ghi_err=[]) for W in WINS}
    nskip = 0
    for wi, w in enumerate(wells):
        sp = struct.get(w)
        if sp is None:
            nskip += 1; continue
        hw = pd.read_csv(os.path.join(TRAIN, f"{w}__horizontal_well.csv"),
                         usecols=["MD", "Z", "TVT", "TVT_input"])
        z = hw["Z"].to_numpy(float)
        tvt = hw["TVT"].to_numpy(float); tin = hw["TVT_input"].to_numpy(float)
        kn = np.isfinite(tin) & np.isfinite(tvt)
        evm = ~np.isfinite(tin)
        if len(np.asarray(sp["tvt"])) != int(evm.sum()):
            nskip += 1; continue
        ev = evm & np.isfinite(tvt)
        if kn.sum() < 100 or ev.sum() < 50:
            nskip += 1; continue
        g = tvt + z
        # struct trace を全行配列に展開(評価行のみ値, 既知行は真g)
        s_tvt_full = np.full(len(hw), np.nan)
        s_tvt_full[evm] = np.asarray(sp["tvt"], float)
        g_s = np.where(evm, s_tvt_full + z, g)      # 既知帯=真g(合法), 評価帯=struct
        ok = np.isfinite(g_s)
        # 前後埋め(TVT欠損行など)
        idx = np.arange(len(g_s))
        g_s = np.interp(idx, idx[ok], g_s[ok])
        ei = np.where(ev)[0]
        ES.append((g_s - g)[ei])
        for W in WINS:
            zh = z - lowpass(z, W)
            gl_s = lowpass(g_s, W)
            r = (gl_s - g)[ei]                       # 低周波のみ予測の誤差成分
            pool[W]["r"].append(r)
            pool[W]["zh"].append(zh[ei])
            pool[W]["sh"].append((g_s - gl_s)[ei])   # struct自身の高周波帯

            pool[W]["glo_err"].append((gl_s - lowpass(g, W))[ei])   # struct低周波誤差
            pool[W]["ghi_err"].append((g - lowpass(g, W))[ei])      # 真の高周波(=κ=0で残る)
        if (wi + 1) % 150 == 0:
            print(f"  {wi+1}/{len(wells)}  ({time.time()-t0:.0f}s)", flush=True)
    es = np.concatenate(ES)

    def rmse(e):
        return float(np.sqrt(np.mean(e ** 2)))
    # struct誤差の正体分解: 井戸定数datum(oracle除去) vs 残りのin-wellドリフト、md_since帯別
    es_dm = np.concatenate([e - np.median(e) for e in ES])
    msb = np.concatenate([np.arange(len(e), dtype=float) for e in ES])   # 評価開始からの行数≈ft
    bands = [(0, 1000), (1000, 2000), (2000, 4000), (4000, 8000), (8000, 10 ** 9)]
    band_rmse = [rmse(es[(msb >= a) & (msb < b)]) for a, b in bands]
    band_rmse_dm = [rmse(es_dm[(msb >= a) & (msb < b)]) for a, b in bands]
    res = dict(n_wells=len(ES), n_rows=int(len(es)), nskip=nskip,
               rmse_struct=rmse(es), rmse_struct_demean=rmse(es_dm),
               bands=[list(b) for b in bands], band_rmse=band_rmse,
               band_rmse_demean=band_rmse_dm,
               ks=list(map(float, KS)), wins=WINS)
    print(f"wells={len(ES)} (skip {nskip})  rows={len(es)}  struct as-is = {res['rmse_struct']:.4f}",
          flush=True)
    print(f"  井戸定数datum除去(oracle) = {res['rmse_struct_demean']:.4f} "
          f"(datum寄与 = {np.sqrt(max(res['rmse_struct']**2 - res['rmse_struct_demean']**2, 0)):.3f})",
          flush=True)
    print("  md_since帯RMSE      :", " ".join(f"{v:6.3f}" for v in band_rmse), flush=True)
    print("  md_since帯RMSE(demean):", " ".join(f"{v:6.3f}" for v in band_rmse_dm), flush=True)
    curves = {}
    rows = []
    for W in WINS:
        r = np.concatenate(pool[W]["r"]); zh = np.concatenate(pool[W]["zh"])
        lo = np.concatenate(pool[W]["glo_err"]); hi = np.concatenate(pool[W]["ghi_err"])
        cur = [rmse(r + k * zh) for k in KS]
        i = int(np.argmin(cur))
        # 2パラメータ族: err = r + α·sh + κ·zh を最小二乗(α=struct帯のshrink, κ=操舵追加)
        sh = np.concatenate(pool[W]["sh"])
        A = np.c_[sh, zh]
        coef, *_ = np.linalg.lstsq(A, -r, rcond=None)
        e2 = rmse(r + A @ coef)
        rows.append(dict(W=W, struct_lo=rmse(lo), true_hi=rmse(hi),
                         k_best=float(KS[i]), best=cur[i], k0=cur[0],
                         k1=cur[int(np.where(KS == 1.0)[0][0])],
                         alpha=float(coef[0]), kappa2=float(coef[1]), best2=e2))
        curves[W] = list(map(float, cur))
        print(f"W={W:5d}  struct低周波誤差={rows[-1]['struct_lo']:6.3f}  真高周波={rows[-1]['true_hi']:6.3f}  "
              f"κ=0:{cur[0]:6.3f}  κ=1:{rows[-1]['k1']:6.3f}  κ*={KS[i]:.2f}:{cur[i]:6.3f}  "
              f"[α={coef[0]:.2f},κ={coef[1]:.2f}]:{e2:6.3f}", flush=True)
    res["scales"] = rows; res["curves"] = curves
    with open(os.path.join(ART, "diag2_struct_steer.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "Meiryo"
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for W in WINS:
        ax[0].plot(KS, curves[W], marker="o", ms=2.5, label=f"W={W}")
    ax[0].axhline(res["rmse_struct"], color="gray", ls="--", lw=0.8,
                  label=f"struct as-is: {res['rmse_struct']:.3f}")
    ax[0].set_title("LP_W(struct g) + κ·HP_W(Z) の overall RMSE")
    ax[0].set_xlabel("κ"); ax[0].set_ylabel("overall RMSE (ft)")
    ax[0].legend(); ax[0].grid(alpha=0.3)
    Ws = [r["W"] for r in rows]
    ax[1].plot(Ws, [r["struct_lo"] for r in rows], marker="o", label="struct 低周波誤差")
    ax[1].plot(Ws, [r["true_hi"] for r in rows], marker="s", label="真の高周波振幅(κ=0で残る)")
    ax[1].plot(Ws, [r["best"] for r in rows], marker="^", label="合成 best")
    ax[1].set_xscale("log"); ax[1].set_xlabel("W (行≈ft)"); ax[1].set_ylabel("RMSE (ft)")
    ax[1].set_title("スケール別分解: ボトルネックはどこか")
    ax[1].legend(); ax[1].grid(alpha=0.3)
    fig.suptitle("v74 診断2 — struct × 操舵高周波の合成 (GRフリー)", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(ART, "diag2_struct_steer.png"), dpi=130)
    print("saved:", os.path.join(ART, "diag2_struct_steer.png"))

if __name__ == "__main__":
    main()
