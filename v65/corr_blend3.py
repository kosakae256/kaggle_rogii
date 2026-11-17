# -*- coding: utf-8 -*-
"""
3本ブレンド調査: struct20(MD-1) + dec10(MD-10) + r0(MD-1, anchor無し) の相関行列・最適重み。
=====================================================================
同一 prior(v66_leanv1cumul)・同一 wells・同一seedで各々走らせ、行別誤差 e_i=pred_i−true を得る。
blend予測=Σw_i pred_i, Σw_i=1 なら誤差=Σw_i e_i → RMSE(w)=√(wᵀ G w)(G=行別誤差の Gram/共分散)。
→ 相関行列 / 個別RMSE / 等重み(1/3) / 最適重み(非負simplex探索 & sum1解析解) / 既存2本(struct20+dec10 50:50) を比較。
実行: .venv/Scripts/python.exe v65/corr_blend3.py
env : SEED(20251707) SEEDS(32) CHUNK(120) N_WELLS(0=全)
      PS20 / PDEC / PR0 (各param json パス)
"""
import os, sys, json, itertools
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
PS20 = os.environ.get("PS20", str(ART / "struct20_params.json"))
PDEC = os.environ.get("PDEC", str(ART / "v65dec10_v66_leanv1cumul_tw_current.json"))
PR0 = os.environ.get("PR0", str(ART / "r0_params.json"))
NAMES = ["struct20", "dec10", "r0"]


def load_params(fp):
    d = json.load(open(fp, encoding="utf-8"))
    return pf.full_params(d.get("best_params", d.get("params", d)))


def rmse(e):
    e = e[np.isfinite(e)]
    return float(np.sqrt(np.mean(e ** 2))) if e.size else float("nan")


def main():
    Ps20 = load_params(PS20); Pdec = load_params(PDEC); Pr0 = load_params(PR0)
    print(f"[prior] {pf.PRIOR_NAME}  seed={SEED}x{SEEDS}")
    print(f"[struct20 MD-1 ] mult={Ps20.get('anchor_mult')} anchor={Ps20.get('use_anchor')} np={Ps20.get('n_particles')} grid={Ps20.get('grid_step')}")
    print(f"[dec10    MD-10] mult={Pdec.get('anchor_mult'):.1f} anchor={Pdec.get('use_anchor')} np={Pdec.get('n_particles')} grid={Pdec.get('grid_step')}")
    print(f"[r0       MD-1 ] mult={Pr0.get('anchor_mult')} anchor={Pr0.get('use_anchor')} np={Pr0.get('n_particles')} grid={Pr0.get('grid_step')}")

    wells = pf.list_wells()
    if N_WELLS: wells = wells[:N_WELLS]
    cache = {}
    for w in wells:
        c = pf.build_cache(w)
        if c is not None and c.get("tvt_true") is not None: cache[w] = c
    pf.CACHE = cache
    ALL = [w for w in wells if w in cache]
    print(f"usable {len(ALL)} 坑井\n")

    e_s20 = pf.predict_errs(Ps20, ALL, SEED, SEEDS, CHUNK)          # MD-1
    e_dec = dec.predict_errs_dec(Pdec, ALL, SEED, SEEDS, CHUNK)     # MD-10(全解像度補間済)
    e_r0 = pf.predict_errs(Pr0, ALL, SEED, SEEDS, CHUNK)           # MD-1

    # 行別誤差を連結(坑井ごとに3系列の長さを揃える=同一eval行に整列)
    cols = [[], [], []]
    for w in ALL:
        m = min(len(e_s20[w]), len(e_dec[w]), len(e_r0[w]))
        cols[0].append(np.asarray(e_s20[w])[:m]); cols[1].append(np.asarray(e_dec[w])[:m]); cols[2].append(np.asarray(e_r0[w])[:m])
    E = np.stack([np.concatenate(cols[0]), np.concatenate(cols[1]), np.concatenate(cols[2])], 1)  # (n,3)
    fin = np.isfinite(E).all(1); Ef = E[fin]
    n = len(Ef)
    print(f"共通有限行 {n}\n")

    # 個別RMSE(共通マスク)
    print("[個別 overall RMSE(共通行)]")
    for i, nm in enumerate(NAMES):
        print(f"  {nm:9s} {rmse(Ef[:, i]):.4f}")

    # 相関行列
    C = np.corrcoef(Ef.T)
    print("\n[誤差相関行列 Pearson]")
    print("            " + "  ".join(f"{nm:>9s}" for nm in NAMES))
    for i, nm in enumerate(NAMES):
        print(f"  {nm:9s} " + "  ".join(f"{C[i,j]:9.3f}" for j in range(3)))

    # Gram(RMSE(w)=sqrt(wᵀGw))
    G = (Ef.T @ Ef) / n

    def rw(w):
        w = np.asarray(w, float); return float(np.sqrt(max(w @ G @ w, 0.0)))

    # 等重み
    w_eq = np.ones(3) / 3
    # sum1 解析解(最小分散, 負許容)
    Gi = np.linalg.inv(G + 1e-9 * np.eye(3)); one = np.ones(3)
    w_an = Gi @ one / (one @ Gi @ one)
    # 非負 simplex グリッド探索(step 0.05)
    best = None
    for a in range(0, 21):
        for b in range(0, 21 - a):
            c = 20 - a - b
            w = np.array([a, b, c]) / 20.0
            r = rw(w)
            if best is None or r < best[0]: best = (r, w)
    print("\n[ブレンドRMSE]")
    print(f"  等重み(1/3,1/3,1/3)         = {rw(w_eq):.4f}")
    print(f"  最適(sum1解析, 負許容) w={np.round(w_an,3)} = {rw(w_an):.4f}")
    print(f"  最適(非負simplex step0.05) w={np.round(best[1],3)} = {best[0]:.4f}  [順:{NAMES}]")
    # 参考: 2本(struct20+dec10 50:50 / 最適2本)
    w_pair = np.array([0.5, 0.5, 0.0])
    print(f"\n[参考] struct20+dec10 50:50(r0無し) = {rw(w_pair):.4f}")
    for i, j in itertools.combinations(range(3), 2):
        # 2本最適(非負)
        bb = None
        for a in range(0, 21):
            w = np.zeros(3); w[i] = a / 20.0; w[j] = 1 - a / 20.0
            r = rw(w)
            if bb is None or r < bb[0]: bb = (r, w[i])
        print(f"  {NAMES[i]}+{NAMES[j]} 最適 w_{NAMES[i]}={bb[1]:.2f} = {bb[0]:.4f}")

    # --- グラフ ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams["font.family"] = "Meiryo"
        fig, ax = plt.subplots(1, 2, figsize=(13, 5))
        # 左: RMSE バー
        labels = NAMES + ["等重み3", "最適3\n非負", "s20+dec10\n50:50"]
        vals = [rmse(Ef[:, 0]), rmse(Ef[:, 1]), rmse(Ef[:, 2]), rw(w_eq), best[0], rw(w_pair)]
        colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B3", "#CCB974", "#999999"]
        bars = ax[0].bar(labels, vals, color=colors)
        for bmk, v in zip(bars, vals): ax[0].text(bmk.get_x() + bmk.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)
        ax[0].set_ylabel("overall RMSE (train, 共通行)"); ax[0].set_ylim(min(vals) - 0.3, max(vals) + 0.3)
        ax[0].set_title(f"3本ブレンド RMSE  最適重み w={np.round(best[1],2)} [{','.join(NAMES)}]")
        ax[0].grid(axis="y", alpha=0.3)
        # 右: 相関ヒートマップ
        im = ax[1].imshow(C, vmin=0.5, vmax=1.0, cmap="RdYlGn_r")
        ax[1].set_xticks(range(3)); ax[1].set_yticks(range(3))
        ax[1].set_xticklabels(NAMES); ax[1].set_yticklabels(NAMES)
        for i in range(3):
            for j in range(3): ax[1].text(j, i, f"{C[i,j]:.3f}", ha="center", va="center", fontsize=11)
        ax[1].set_title("誤差相関(低=直交=多様)"); fig.colorbar(im, ax=ax[1], fraction=0.046)
        fig.tight_layout()
        png = ART / "corr_blend3.png"; fig.savefig(png, dpi=110)
        print(f"\n[png] {png}")
    except Exception as ex:
        print(f"[png skip] {type(ex).__name__}: {ex}")


if __name__ == "__main__":
    main()
