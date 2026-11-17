# -*- coding: utf-8 -*-
"""
MD-1(current v65 best)× MD-10(間引き版 best)の相関調査 = ブレンド多様性の判定。
=====================================================================
両者を同一 prior(現行既定)・同一 wells・同一seedで走らせ、行別誤差(pred−tvt_true)の相関を測る。
低相関 = MD-1/MD-10 は多様(ブレンド価値あり)/高相関 = 冗長。50:50 ブレンド RMSE も出す。
実行: .venv/Scripts/python.exe v65/corr_md1_md10.py
env : SEED(20251707) SEEDS(32) CHUNK(120) N_WELLS(0=全)
      P1(v65_pf_tw_current.json) P10(v65dec10_v66_leanv1cumul_tw_current.json)
"""
import os, sys, json
os.environ.setdefault("DEC", "10")
from pathlib import Path
import numpy as np
sys.stdout = open(1, "w", encoding="utf-8", closefd=False)
sys.path.insert(0, str(Path(__file__).parent))
import pf_engine_v65 as pf
import pf_dec_v65 as dec

ART = pf.OUT_DIR
SEED = int(os.environ.get("SEED", "20251707"))
SEEDS = int(os.environ.get("SEEDS", "32"))
CHUNK = int(os.environ.get("CHUNK", "120"))
N_WELLS = int(os.environ.get("N_WELLS", "0")) or None
P1_JSON = os.environ.get("P1", str(ART / "v65_pf_tw_current.json"))
P10_JSON = os.environ.get("P10", str(ART / "v65dec10_v66_leanv1cumul_tw_current.json"))


def load_params(fp):
    d = json.load(open(fp, encoding="utf-8"))
    p = d.get("best_params", d.get("params", d))
    return pf.full_params(p)


def pear(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3: return np.nan
    return float(np.corrcoef(a[m], b[m])[0, 1])


def rmse(e):
    e = e[np.isfinite(e)]
    return float(np.sqrt(np.mean(e ** 2))) if e.size else float("nan")


def main():
    P1 = load_params(P1_JSON); P10 = load_params(P10_JSON)
    print(f"[prior] {pf.PRIOR_NAME}(両者共通)  seed={SEED}x{SEEDS}")
    print(f"[P1 MD-1 ] mult={P1.get('anchor_mult')} anchor={P1.get('use_anchor')} np={P1.get('n_particles')} grid={P1.get('grid_step')}")
    print(f"[P10 MD-10] mult={P10.get('anchor_mult')} anchor={P10.get('use_anchor')} np={P10.get('n_particles')} grid={P10.get('grid_step')}")
    wells = pf.list_wells()
    if N_WELLS: wells = wells[:N_WELLS]
    cache = {}
    for w in wells:
        c = pf.build_cache(w)
        if c is not None and c["tvt_true"] is not None: cache[w] = c
    pf.CACHE = cache
    ALL = [w for w in wells if w in cache]
    print(f"usable {len(ALL)} 坑井\n")

    e1 = pf.predict_errs(P1, ALL, SEED, SEEDS, CHUNK)            # MD-1(非間引き)
    e10 = dec.predict_errs_dec(P10, ALL, SEED, SEEDS, CHUNK)     # MD-10(全解像度に補間済)

    # 行別プール(整列: 同一 tvt_true 行)
    a = np.concatenate([e1[w] for w in ALL]); b = np.concatenate([e10[w] for w in ALL])
    L = min(len(a), len(b))
    # 各坑井 行数一致のはず(同 eval行)。念のため per-well で連結し直す
    aa, bb = [], []
    for w in ALL:
        n = min(len(e1[w]), len(e10[w]))
        aa.append(e1[w][:n]); bb.append(e10[w][:n])
    a = np.concatenate(aa); b = np.concatenate(bb)

    r1, r10 = rmse(a), rmse(b)
    print(f"[overall RMSE]  MD-1 = {r1:.4f}   MD-10 = {r10:.4f}")
    print(f"[誤差相関]      pooled Pearson = {pear(a, b):.4f}  ← 低いほどブレンド多様性")
    # 坑井別 RMSE 相関
    rw1 = np.array([rmse(e1[w]) for w in ALL]); rw10 = np.array([rmse(e10[w]) for w in ALL])
    from scipy.stats import spearmanr
    print(f"[坑井別RMSE]    Pearson {pear(rw1, rw10):.4f}  Spearman {float(spearmanr(rw1, rw10).statistic):.4f}")
    # 50:50 ブレンド(pred平均=誤差平均)
    blend = 0.5 * (a + b)
    print(f"\n[50:50 ブレンド RMSE] = {rmse(blend):.4f}  (最良単体 {min(r1, r10):.4f} 比 {rmse(blend)-min(r1, r10):+.4f})")
    ws = np.linspace(0, 1, 21)
    best = min(ws, key=lambda w: rmse(w * a + (1 - w) * b))
    print(f"[最適重み] w_MD1={best:.2f} で RMSE {rmse(best * a + (1 - best) * b):.4f}")

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        plt.rcParams["font.family"] = "Meiryo"
        fig, ax = plt.subplots(1, 3, figsize=(16, 5))
        idx = np.linspace(0, len(a) - 1, min(len(a), 40000)).astype(int)
        ax[0].scatter(a[idx], b[idx], s=2, alpha=0.15)
        lim = [np.percentile(a, 0.5), np.percentile(a, 99.5)]; ax[0].plot(lim, lim, "k--", lw=1)
        ax[0].set_xlabel("MD-1 誤差"); ax[0].set_ylabel("MD-10 誤差")
        ax[0].set_title(f"行別誤差(Pearson {pear(a, b):.3f})")
        ax[1].scatter(rw1, rw10, s=8, alpha=0.4)
        lm = [min(rw1.min(), rw10.min()) * 0.8, max(rw1.max(), rw10.max()) * 1.2]
        ax[1].plot(lm, lm, "k--", lw=1); ax[1].set_xscale("log"); ax[1].set_yscale("log")
        ax[1].set_xlabel("MD-1 坑井RMSE"); ax[1].set_ylabel("MD-10 坑井RMSE")
        ax[1].set_title(f"坑井別RMSE(Spearman {float(spearmanr(rw1, rw10).statistic):.3f})")
        ax[2].plot(ws, [rmse(w * a + (1 - w) * b) for w in ws], "o-")
        ax[2].set_xlabel("MD-1 の重み w"); ax[2].set_ylabel("ブレンド RMSE")
        ax[2].set_title(f"重みスイープ(50:50={rmse(blend):.4f})")
        fig.tight_layout(); p = ART / "corr_md1_md10.png"; fig.savefig(p, dpi=120); plt.close(fig)
        print(f"\n[png] {p}")
    except Exception as e:
        print(f"[png] スキップ({e})")


if __name__ == "__main__":
    main()
