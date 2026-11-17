# -*- coding: utf-8 -*-
"""
v38(最初のstruct)と v66(改善版)の相関診断
=====================================================================
2つの構造priorがどれだけ似ているか/違うかを多層で測る:
  A 予測(構造の形): 坑井内 tvt 軌跡の相関(anchor不変=within-well Pearson)
  B 誤差(間違い方): err=tvt−true の相関(高=冗長/低=多様=ブレンド価値)
  C 信頼度 σ      : per-row σ の相関(v66.1 累積σ が v38 σ とどれだけ別物か)
  D 坑井別RMSE    : どの坑井を難しいと見るかの一致(Pearson/Spearman)
出力: v66/artifacts_v66/corr_v38_v66.png + サマリ。read-only。
実行: .venv/Scripts/python.exe v66/corr_v38_v66.py
env : V66(v66_leanv1cumul 改善版prior名) V38(struct=v38既定)
"""
import sys, os, glob, pickle
from pathlib import Path
import numpy as np, pandas as pd

sys.stdout = open(1, "w", encoding="utf-8", closefd=False)
PROJECT = Path(r"C:\Users\kosaka256\Documents\rogii_claude")
OUT = PROJECT / "v66" / "artifacts_v66"
V38_ART = PROJECT / "v38" / "artifacts_v38"
TRAIN_DIR = PROJECT / "rogii-wellbore-geology-prediction" / "train"
V66_NAME = os.environ.get("V66", "v66_leanv1cumul")


def load_prior(name):
    if name == "v38" or name == "struct":
        return pickle.loads((V38_ART / "struct_prior_train.pkl").read_bytes())
    for p in [V38_ART / "priors" / f"prior_{name}_train.pkl", OUT / f"prior_{name}_train.pkl"]:
        if p.exists():
            return pickle.loads(p.read_bytes())
    raise FileNotFoundError(name)


def true_tail(fp):
    w = pd.read_csv(fp, usecols=["TVT", "TVT_input"])
    ti = w["TVT_input"].to_numpy(float); tvt = w["TVT"].to_numpy(float)
    fin = np.where(np.isfinite(ti))[0]
    if len(fin) == 0:
        return None
    s = int(fin.max())
    return tvt[s + 1:]


def pear(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3 or np.std(a[m]) < 1e-9 or np.std(b[m]) < 1e-9:
        return np.nan
    return float(np.corrcoef(a[m], b[m])[0, 1])


def main():
    p38 = load_prior("v38"); p66 = load_prior(V66_NAME)
    print(f"[prior] v38 ({len(p38)}坑井) vs {V66_NAME} ({len(p66)}坑井)")

    tvt_corr_well, err_corr_well, sig_corr_well = [], [], []
    rmse38, rmse66, dtvt_well = {}, {}, {}
    e38_pool, e66_pool, s38_pool, s66_pool, r38_pool, r66_pool = [], [], [], [], [], []
    for fp in sorted(glob.glob(str(TRAIN_DIR / "*__horizontal_well.csv"))):
        nm = os.path.basename(fp).split("__")[0]
        if nm not in p38 or nm not in p66:
            continue
        true = true_tail(fp)
        if true is None:
            continue
        t38 = np.asarray(p38[nm]["tvt"], float); t66 = np.asarray(p66[nm]["tvt"], float)
        s38 = np.asarray(p38[nm].get("sig", np.full(len(t38), np.nan)), float)
        s66 = np.asarray(p66[nm].get("sig", np.full(len(t66), np.nan)), float)
        L = min(len(t38), len(t66), len(true))
        if L < 10 or len(t38) != len(t66) or len(t38) != len(true):
            continue
        t38, t66, s38, s66, true = t38[:L], t66[:L], s38[:L], s66[:L], true[:L]
        e38 = t38 - true; e66 = t66 - true
        fin = np.isfinite(e38) & np.isfinite(e66)
        if fin.sum() < 10:
            continue
        # A 予測: within-well 軌跡相関(anchor不変)
        tvt_corr_well.append(pear(t38[fin], t66[fin]))
        # B 誤差
        err_corr_well.append(pear(e38[fin], e66[fin]))
        # C σ
        sig_corr_well.append(pear(s38[fin], s66[fin]))
        rmse38[nm] = float(np.sqrt(np.mean(e38[fin] ** 2)))
        rmse66[nm] = float(np.sqrt(np.mean(e66[fin] ** 2)))
        dtvt_well[nm] = float(np.mean(np.abs(t38[fin] - t66[fin])))
        e38_pool.append(e38[fin]); e66_pool.append(e66[fin])
        s38_pool.append(s38[fin]); s66_pool.append(s66[fin])
        r38_pool.append((t38 - t38[0])[fin]); r66_pool.append((t66 - t66[0])[fin])

    e38p = np.concatenate(e38_pool); e66p = np.concatenate(e66_pool)
    s38p = np.concatenate(s38_pool); s66p = np.concatenate(s66_pool)
    r38p = np.concatenate(r38_pool); r66p = np.concatenate(r66_pool)
    n = len(rmse38); common = list(rmse38)
    a = np.array([rmse38[w] for w in common]); b = np.array([rmse66[w] for w in common])
    dt = np.array([dtvt_well[w] for w in common])

    def sp(x, y):
        from scipy.stats import spearmanr
        return float(spearmanr(x, y).statistic)

    print(f"\n=== 相関(v38 ↔ {V66_NAME}, 共通 {n} 坑井)===")
    print(f"[A 予測 tvt]  坑井内軌跡 Pearson 中央 {np.nanmedian(tvt_corr_well):.4f}"
          f"  平均 {np.nanmean(tvt_corr_well):.4f}   (relief 相対軌跡 pooled {pear(r38p, r66p):.4f})")
    print(f"[B 誤差 err]  pooled Pearson {pear(e38p, e66p):.4f}   坑井内 中央 {np.nanmedian(err_corr_well):.4f}"
          f"   ← 高=間違い方が冗長/低=多様(ブレンド価値)")
    print(f"[C 信頼度 σ]  pooled Pearson {pear(s38p, s66p):.4f}   坑井内 中央 {np.nanmedian(sig_corr_well):.4f}"
          f"   ← v66.1累積σが v38 σ とどれだけ別物か")
    print(f"[D 坑井RMSE]  Pearson {pear(a, b):.4f}   Spearman {sp(a, b):.4f}   ← 難坑井の見立ての一致")
    print(f"\n[予測の動き] 坑井別 平均|Δtvt| 中央 {np.median(dt):.2f}ft  p90 {np.quantile(dt,0.9):.2f}ft  最大 {dt.max():.1f}ft")
    print(f"[RMSE]       v38 pooled {np.sqrt(np.mean(e38p**2)):.4f}  {V66_NAME} {np.sqrt(np.mean(e66p**2)):.4f}")
    big = dt > 3.0
    print(f"[大きく動いた坑井] |Δtvt|平均>3ft: {big.sum()}/{n} 本({big.mean():.1%})")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams["font.family"] = "Meiryo"
        fig, ax = plt.subplots(2, 2, figsize=(13, 11))
        ax[0, 0].scatter(a, b, s=8, alpha=0.4)
        lim = [min(a.min(), b.min()) * 0.8, max(a.max(), b.max()) * 1.2]
        ax[0, 0].plot(lim, lim, "k--", lw=1); ax[0, 0].set_xscale("log"); ax[0, 0].set_yscale("log")
        ax[0, 0].set_xlabel("v38 坑井RMSE"); ax[0, 0].set_ylabel(f"{V66_NAME} 坑井RMSE")
        ax[0, 0].set_title(f"D 坑井別RMSE(Pearson {pear(a,b):.3f}/Spearman {sp(a,b):.3f})")
        idx = np.linspace(0, len(e38p) - 1, min(len(e38p), 40000)).astype(int)
        ax[0, 1].scatter(e38p[idx], e66p[idx], s=3, alpha=0.2)
        el = [np.percentile(e38p, 0.5), np.percentile(e38p, 99.5)]
        ax[0, 1].plot(el, el, "k--", lw=1)
        ax[0, 1].set_xlabel("v38 誤差 [ft]"); ax[0, 1].set_ylabel(f"{V66_NAME} 誤差 [ft]")
        ax[0, 1].set_title(f"B 行別誤差(pooled Pearson {pear(e38p,e66p):.3f})")
        ax[1, 0].scatter(s38p[idx], s66p[idx], s=3, alpha=0.2)
        ax[1, 0].set_xlabel("v38 σ [ft]"); ax[1, 0].set_ylabel(f"{V66_NAME} σ [ft]")
        ax[1, 0].set_title(f"C 行別σ(pooled Pearson {pear(s38p,s66p):.3f})")
        ax[1, 1].hist(np.array(tvt_corr_well), bins=60, alpha=0.7, label="A 予測tvt")
        ax[1, 1].hist(np.array(err_corr_well), bins=60, alpha=0.5, label="B 誤差")
        ax[1, 1].set_xlabel("坑井内 Pearson"); ax[1, 1].set_ylabel("坑井数"); ax[1, 1].legend()
        ax[1, 1].set_title("坑井内相関の分布")
        fig.tight_layout(); p = OUT / "corr_v38_v66.png"; fig.savefig(p, dpi=120); plt.close(fig)
        print(f"\n[png] {p}")
    except Exception as e:
        print(f"[png] スキップ({e})")


if __name__ == "__main__":
    main()
