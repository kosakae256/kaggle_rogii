# -*- coding: utf-8 -*-
"""
v28 tabular 2版(PF=r0/r1/r4 vs pf_1/2/3)の OOF 相関 & 50:50 アンサンブル調査
=====================================================================
- 学習済みモデルは再利用(v28/artifacts_v28_4=r014, v28/artifacts_v28=pf123)。
- fold は決定的 GroupKFold(5, by well)で両版一致 → OOF は行整列(id/well/target 一致を確認済)。
- OOF = 各版 9モデル(lgb/cb/xgb ×3)の fold-holdout 予測の平均。target=残差(TVT−lk)。
- pp = 各版の pf_ancc で apply_pp(w_pf/tau/alpha は各版 pp_params.json)。
- ★GPU不使用: CUDA_VISIBLE_DEVICES="" で隠し、xgbは device=cpu 強制(predictはCPU)。
実行: .venv/Scripts/python.exe v28/corr_pf_versions_v28.py
出力: v28/artifacts_v28/corr_pf_versions.png + oof を scratchpad に保存
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""      # ★GPU隠す(GBM predict をCPUに)
import sys, json, joblib, gc
from pathlib import Path
import numpy as np, pandas as pd
sys.stdout = open(1, "w", encoding="utf-8", closefd=False)
from sklearn.model_selection import GroupKFold
from sklearn.metrics import root_mean_squared_error as rmse

PROJ = Path(r"C:\Users\kosaka256\Documents\rogii_claude")
SCR = Path(r"C:\Users\KOSAKA~1\AppData\Local\Temp\claude\c--Users-kosaka256-Documents-rogii-claude\1c67796e-d953-4c10-8874-a500ac704725\scratchpad")
# ★各版の学習 ROW_FRAC を明示(r014=artifacts_v28_4 は 0.5 学習=ユーザー確認)。
#   frac<1 は学習時 df.iloc[::step] 後に GroupKFold されているので、同じサブサンプルで坑井→fold を復元する。
DIRS = {"r014": (PROJ / "v28" / "artifacts_v28_4", 0.5), "pf123": (PROJ / "v28" / "artifacts_v28", 1.0)}
VARIANTS = [f"{a}-{i}" for a in ("lightgbm", "catboost", "xgboost") for i in (1, 2, 3)]
N = 5


def build_full_fold(g, frac):
    """学習時と同じ(サブサンプル後)GroupKFold の坑井→fold を復元し、全行に展開。
       GroupKFold は坑井丸ごと holdout なので、fold{k} は割当坑井の全行に対し正当な OOF。"""
    n = len(g)
    if frac < 0.6:
        step = max(1, int(round(1.0 / frac)))
        gsub = g[np.arange(0, n, step)]                  # df.iloc[::step] と同一
    else:
        gsub = g
    w2f = {}
    for k, (_tr, va) in enumerate(GroupKFold(N).split(np.zeros((len(gsub), 1)), np.zeros(len(gsub)), groups=gsub)):
        for w in np.unique(gsub[va]):
            w2f[w] = k
    return np.array([w2f.get(w, 0) for w in g], dtype=np.int64)


def oof_of(tag, art, frac):
    feats = json.loads((art / "features.json").read_text())
    pp = json.loads((art / "pp_params.json").read_text())
    cols = list(dict.fromkeys(feats + ["well", "target", "last_known_tvt", "md_since", "pf_ancc"]))
    df = pd.read_parquet(art / "data" / "train.parquet", columns=cols)
    y = df["target"].to_numpy(np.float32); g = df["well"].to_numpy()
    lk = df["last_known_tvt"].to_numpy(np.float64); mds = df["md_since"].to_numpy(np.float64)
    pf_abs = df["pf_ancc"].to_numpy(np.float64)
    X = df[feats]                                  # 既に float32(dtype確認済)
    del df; gc.collect()
    full_fold = build_full_fold(g, frac)           # ★frac に合わせた正しい per-row holdout fold
    rows_by_fold = [np.where(full_fold == k)[0] for k in range(N)]
    print(f"  [{tag}] frac={frac} fold行数: {[len(r) for r in rows_by_fold]}")
    oof_sum = np.zeros(len(y), np.float64); nmod = 0
    for name in VARIANTS:
        d = art / "models" / name
        if not d.exists():
            print(f"  [{tag}] {name} 無し→スキップ"); continue
        acc = np.zeros(len(y), np.float32)
        for k in range(N):
            rk = rows_by_fold[k]
            if len(rk) == 0: continue
            m = joblib.load(d / f"fold{k}.pkl")
            try: m.set_params(device="cpu")
            except Exception: pass
            if name.startswith("xgboost"):
                try: m.get_booster().set_param({"device": "cpu"})
                except Exception: pass
            acc[rk] = m.predict(X.iloc[rk]).astype(np.float32)
            del m; gc.collect()
        oof_sum += acc; nmod += 1
        print(f"  [{tag}] {name} OOF rmse(res)={rmse(y, acc):.4f}", flush=True)
    ens = (oof_sum / max(nmod, 1)).astype(np.float64)
    del X; gc.collect()
    # apply_pp(submit準拠)
    wpf = float(pp.get("w_pf", 0.09)); tau = pp.get("tau", 85); al = float(pp.get("alpha", 1.0))
    pf_res = pf_abs - lk
    d_ = ens * (1 - wpf) + pf_res * wpf
    if tau: d_ = d_ * (1.0 - np.exp(-np.maximum(mds, 0.0) / float(tau)))
    final = d_ * al
    print(f"  [{tag}] === raw {rmse(y, ens):.4f} / pp {rmse(y, final):.4f}  (nmod={nmod}, pp={pp}) ===", flush=True)
    return dict(y=y, g=g, ens=ens, final=final, rmse_raw=float(rmse(y, ens)), rmse_pp=float(rmse(y, final)))


def pear(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def main():
    R = {}
    for tag, (art, frac) in DIRS.items():
        print(f"\n===== {tag} ({art.name}, ROW_FRAC={frac}) OOF 再現 =====")
        R[tag] = oof_of(tag, art, frac)
        np.save(SCR / f"v28oof_{tag}.npy", np.column_stack([R[tag]["ens"], R[tag]["final"]]))
    A, B = R["r014"], R["pf123"]
    assert np.array_equal(A["g"], B["g"]) and np.allclose(A["y"], B["y"]), "行非整列"
    y = A["y"].astype(np.float64); g = A["g"]
    eA, eB, fA, fB = A["ens"], B["ens"], A["final"], B["final"]

    print("\n" + "=" * 64)
    print(f"[各版 OOF overall RMSE(残差=Kaggle指標)]")
    print(f"  r014 : raw {A['rmse_raw']:.4f} / pp {A['rmse_pp']:.4f}")
    print(f"  pf123: raw {B['rmse_raw']:.4f} / pp {B['rmse_pp']:.4f}")
    print(f"\n[相関(r014 ↔ pf123)]")
    print(f"  予測(raw GBM残差)  Pearson {pear(eA, eB):.4f}")
    print(f"  誤差(pred−true)     Pearson {pear(eA - y, eB - y):.4f}   ← 低いほどブレンド価値")
    print(f"  最終(pp後)          Pearson {pear(fA, fB):.4f}")
    # 50:50
    ens50 = 0.5 * (eA + eB); fin50 = 0.5 * (fA + fB)
    print(f"\n[50:50 アンサンブル overall RMSE]")
    print(f"  raw  50:50 : {rmse(y, ens50):.4f}   (各版 raw {A['rmse_raw']:.4f}/{B['rmse_raw']:.4f}, 最良比 {rmse(y,ens50)-min(A['rmse_raw'],B['rmse_raw']):+.4f})")
    print(f"  pp   50:50 : {rmse(y, fin50):.4f}   (各版 pp  {A['rmse_pp']:.4f}/{B['rmse_pp']:.4f}, 最良比 {rmse(y,fin50)-min(A['rmse_pp'],B['rmse_pp']):+.4f})")
    # 最適重み(pp)も参考に
    ws = np.linspace(0, 1, 21); best = min(ws, key=lambda w: rmse(y, w * fA + (1 - w) * fB))
    print(f"  pp 最適重み探索: w_r014={best:.2f} で RMSE {rmse(y, best*fA+(1-best)*fB):.4f}")
    # 坑井別RMSE相関
    import pandas as pd
    dfw = pd.DataFrame({"g": g, "eA2": (eA - y) ** 2, "eB2": (eB - y) ** 2})
    wr = dfw.groupby("g").mean()
    rA = np.sqrt(wr["eA2"].to_numpy()); rB = np.sqrt(wr["eB2"].to_numpy())
    from scipy.stats import spearmanr
    print(f"\n[坑井別RMSE] Pearson {pear(rA, rB):.4f} / Spearman {float(spearmanr(rA, rB).statistic):.4f}  ({len(rA)}坑井)")

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        plt.rcParams["font.family"] = "Meiryo"
        idx = np.linspace(0, len(y) - 1, min(len(y), 40000)).astype(int)
        fig, ax = plt.subplots(1, 3, figsize=(16, 5))
        ax[0].scatter((eA - y)[idx], (eB - y)[idx], s=2, alpha=0.15)
        lim = [np.percentile(eA - y, 0.5), np.percentile(eA - y, 99.5)]; ax[0].plot(lim, lim, "k--", lw=1)
        ax[0].set_xlabel("r014 誤差"); ax[0].set_ylabel("pf123 誤差")
        ax[0].set_title(f"行別誤差(Pearson {pear(eA-y, eB-y):.3f})")
        ax[1].scatter(rA, rB, s=8, alpha=0.4)
        lm = [min(rA.min(), rB.min()) * 0.8, max(rA.max(), rB.max()) * 1.2]
        ax[1].plot(lm, lm, "k--", lw=1); ax[1].set_xscale("log"); ax[1].set_yscale("log")
        ax[1].set_xlabel("r014 坑井RMSE"); ax[1].set_ylabel("pf123 坑井RMSE")
        ax[1].set_title(f"坑井別RMSE(Spearman {float(spearmanr(rA, rB).statistic):.3f})")
        ax[2].plot(ws, [rmse(y, w * fA + (1 - w) * fB) for w in ws], "o-")
        ax[2].set_xlabel("r014 の重み w"); ax[2].set_ylabel("ブレンド RMSE(pp)")
        ax[2].set_title(f"ブレンド重みスイープ(50:50={rmse(y,fin50):.4f})")
        fig.tight_layout(); p = PROJ / "v28" / "artifacts_v28" / "corr_pf_versions.png"
        fig.savefig(p, dpi=120); plt.close(fig); print(f"\n[png] {p}")
    except Exception as e:
        print(f"[png] スキップ({e})")


if __name__ == "__main__":
    main()
