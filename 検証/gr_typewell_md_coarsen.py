# -*- coding: utf-8 -*-
"""
GR × typewell 相関の MD 粗化依存(間引き vs bin平均)
=====================================================================
gr_typewell_correlation.py と同一定義:
  obs = 水平井 観測GR(eval行) / exp = typewell GR を真TVTで補間(=期待GR)。
  生 median Pearson r ≈ 0.665(error_anatomy §3.5)。
本スクリプトは MD を s=1..10 に粗くしたとき相関がどう動くかを2方式で測る:
  - 間引き(subsample): obs[::s] vs exp[::s](点は減るがノイズそのまま)
  - bin平均(binmean) : s行mean(obs) vs s行mean(exp)(=情報圧縮の本体=denoise)
判定: binmean で median r が上がる → 低周波信号を残しノイズを落とせている = MD粗化が照合に有効。
      間引きで横ばい/微減・binmeanで上昇 なら「効くのは"平均"であって"間引き"ではない」。
実行: .venv/Scripts/python.exe 検証/gr_typewell_md_coarsen.py   (GPU不使用)
"""
import glob, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.stdout = open(1, "w", encoding="utf-8", closefd=False)

PROJECT = Path(r"C:\Users\kosaka256\Documents\rogii_claude")
DATA_DIR = PROJECT / "rogii-wellbore-geology-prediction" / "train"
OUT = PROJECT / "検証"
SVALS = list(range(1, 11))          # MD 粗化率 1..10
MIN_ROWS = 60


def pearson(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b); a, b = a[m], b[m]
    if len(a) < 10 or a.std() < 1e-9 or b.std() < 1e-9:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def binmean(x, s):
    n = (len(x) // s) * s
    if n < s: return x[:0]
    return x[:n].reshape(-1, s).mean(1)


def ma(x, w):
    """中央移動平均(窓w、全行保持=解像度そのまま=既存パイプラインの GR 平滑相当)。"""
    if w <= 1: return np.asarray(x, float)
    return pd.Series(x, dtype="float64").rolling(int(w), center=True, min_periods=1).mean().to_numpy()


def process_well(wid):
    try:
        hw = pd.read_csv(DATA_DIR / f"{wid}__horizontal_well.csv", usecols=["GR", "TVT", "TVT_input", "MD"])
        tw = pd.read_csv(DATA_DIR / f"{wid}__typewell.csv", usecols=["TVT", "GR"]).sort_values("TVT")
    except Exception:
        return None
    if "TVT" not in hw.columns: return None
    ev = hw[hw["TVT_input"].isna()].sort_values("MD")
    if len(ev) < MIN_ROWS: return None
    tvt = ev["TVT"].to_numpy(float); obs = ev["GR"].to_numpy(float)
    tw_tvt = tw["TVT"].to_numpy(float); tw_gr = tw["GR"].to_numpy(float)
    ok = np.isfinite(tvt) & np.isfinite(obs)
    if ok.sum() < MIN_ROWS or len(tw_tvt) < 3: return None
    tvt, obs = tvt[ok], obs[ok]
    exp = np.interp(tvt, tw_tvt, tw_gr)
    return obs, exp


def main():
    wells = sorted({Path(p).stem.replace("__horizontal_well", "")
                    for p in glob.glob(str(DATA_DIR / "*__horizontal_well.csv"))})
    data = [r for r in (process_well(w) for w in wells) if r is not None]
    print(f"対象坑井: {len(data)} / {len(wells)}")

    sub_med, bin_med, sub_imp, bin_imp = [], [], [], []
    raw = np.array([pearson(o, e) for o, e in data])
    raw_med = np.nanmedian(raw)
    print(f"\n生(s=1) median Pearson r = {raw_med:.4f}  (error_anatomy 実測 ≈0.665 と整合確認)\n")
    print(f"{'s(MD)':>5} | {'間引き med r':>11} | {'bin平均 med r':>12} | {'binΔ vs生':>10} | {'bin改善坑井%':>10}")
    print("-" * 62)
    for s in SVALS:
        rs_sub = np.array([pearson(o[::s], e[::s]) for o, e in data])
        rs_bin = np.array([pearson(binmean(o, s), binmean(e, s)) for o, e in data])
        sm, bm = np.nanmedian(rs_sub), np.nanmedian(rs_bin)
        fin = np.isfinite(rs_bin) & np.isfinite(raw)
        bimp = float(np.mean(rs_bin[fin] > raw[fin]) * 100)
        sfin = np.isfinite(rs_sub) & np.isfinite(raw)
        simp = float(np.mean(rs_sub[sfin] > raw[sfin]) * 100)
        sub_med.append(sm); bin_med.append(bm); sub_imp.append(simp); bin_imp.append(bimp)
        print(f"{s:>5} | {sm:11.4f} | {bm:12.4f} | {bm-raw_med:+10.4f} | {bimp:9.1f}")

    # --- 水平GRを MA(全行保持=解像度そのまま=既存パイプライン相当)した場合 ---
    print("\n--- 水平GRを MA(全行保持=既存の GR 平滑相当)。bin平均(=MA+間引き, 同窓)と比較 ---")
    print(f"{'MA窓':>5} | {'MA(obs)vs exp':>13} | {'MA(obs)vsMA(exp)':>16} | {'bin平均(同s)':>12} | {'生比(MA両方)':>11}")
    print("-" * 68)
    for W in (3, 5, 7):
        r_obs = np.nanmedian([pearson(ma(o, W), e) for o, e in data])
        r_both = np.nanmedian([pearson(ma(o, W), ma(e, W)) for o, e in data])
        r_bin = bin_med[W - 1]
        print(f"{W:>5} | {r_obs:13.4f} | {r_both:16.4f} | {r_bin:12.4f} | {r_both-raw_med:+11.4f}")
    print("  → MA(両方) ≈ bin平均 なら『相関の利得は denoise で、既存MAで既に取れている』"
          "(MD粗化の相関的な新規性は無い=価値は target denoise/ステップ数/速度の別軸)。")

    print("\n--- 解釈 ---")
    best_bin = SVALS[int(np.nanargmax(bin_med))]
    print(f"  bin平均のピーク: s={best_bin}  (median r={np.nanmax(bin_med):.4f}, 生比 {np.nanmax(bin_med)-raw_med:+.4f})")
    print("  ・bin平均で r が上がる → MD粗化(denoise)は照合に有効 = 配線する価値あり。")
    print("  ・間引きが横ばいで bin平均だけ上がる → 効くのは平均であって間引きではない(観測の bin平均が本体)。")
    print("  ※ 真TVTを使った相関の上限診断。r 上昇=尤度の谷が深くなる示唆(PF/LSTM 双方に効く)。")

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        plt.rcParams["font.family"] = "Meiryo"
        fig, ax = plt.subplots(1, 2, figsize=(13, 5))
        ax[0].plot(SVALS, sub_med, "o-", label="間引き subsample")
        ax[0].plot(SVALS, bin_med, "s-", label="bin平均 binmean")
        ax[0].axhline(raw_med, color="k", ls="--", lw=1, label=f"生 r={raw_med:.3f}")
        ax[0].set_xlabel("MD 粗化率 s"); ax[0].set_ylabel("坑井別 Pearson r 中央値")
        ax[0].set_title("GR × typewell期待GR 相関 vs MD粗化"); ax[0].legend(); ax[0].grid(alpha=0.3)
        ax[1].plot(SVALS, bin_imp, "s-", label="bin平均 改善坑井%")
        ax[1].plot(SVALS, sub_imp, "o-", label="間引き 改善坑井%")
        ax[1].axhline(50, color="k", ls="--", lw=1)
        ax[1].set_xlabel("MD 粗化率 s"); ax[1].set_ylabel("生(s=1)より相関UPした坑井 %")
        ax[1].set_title("坑井のうち相関が改善した割合"); ax[1].legend(); ax[1].grid(alpha=0.3)
        fig.tight_layout(); p = OUT / "gr_typewell_md_coarsen.png"; fig.savefig(p, dpi=120); plt.close(fig)
        print(f"\n[png] {p}")
    except Exception as e:
        print(f"[png] スキップ({e})")


if __name__ == "__main__":
    main()
