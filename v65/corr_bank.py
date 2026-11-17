# -*- coding: utf-8 -*-
"""
バンク相関: 任意のPF構成群(MD-1/MD-10 混在可)の行別誤差相関行列・個別RMSE・対ターゲット50:50。
=====================================================================
SETS = [(name, param_json, is_dec)] を同一 prior/wells/seed で走らせ、誤差 e=pred−true を集める。
ターゲット(SETS[0])が各メンバーとどれだけ直交するか(=バンク多様性)を判定する。
既定: anchor_faithful(MD-10) を struct20 / dec10 / r0 と比較(ユーザー 2026-07-15)。
実行: .venv/Scripts/python.exe v65/corr_bank.py
env : SEED(20251707) SEEDS(32) CHUNK(120) N_WELLS(0=全)
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

# (表示名, param json, is_dec[MD-10か])  ← SETS[0]=ターゲット
SETS = [
    ("anchor_faithful", ART / "v65pdec10_anchor_faithful_v66_leanv1cumul_current.json", True),
    ("struct20",        ART / "struct20_params.json",                                    False),
    ("dec10",           ART / "v65dec10_v66_leanv1cumul_tw_current.json",                True),
    ("r0",              ART / "r0_params.json",                                          False),
    ("sharp_matcher",   ART / "v65pdec10_sharp_matcher_v66_leanv1cumul_current.json",    True),   # MD-10, anchor無
    ("smooth_tracker",  ART / "v65p_smooth_tracker_v66_leanv1cumul_current.json",        False),  # MD-1, anchor無
]


def load_params(fp):
    d = json.load(open(fp, encoding="utf-8"))
    return pf.full_params(d.get("best_params", d.get("params", d)))


def rmse(e):
    e = e[np.isfinite(e)]
    return float(np.sqrt(np.mean(e ** 2))) if e.size else float("nan")


def main():
    names = [s[0] for s in SETS]
    params = [load_params(s[1]) for s in SETS]
    isdec = [s[2] for s in SETS]
    print(f"[prior] {pf.PRIOR_NAME}  seed={SEED}x{SEEDS}")
    for nm, P, d in zip(names, params, isdec):
        print(f"  {nm:16s} {'MD-10' if d else 'MD-1 '} anchor={P.get('use_anchor')} mult={P.get('anchor_mult'):.1f} grid={P.get('grid_step')} np={P.get('n_particles')}")

    wells = pf.list_wells()
    if N_WELLS: wells = wells[:N_WELLS]
    cache = {}
    for w in wells:
        c = pf.build_cache(w)
        if c is not None and c.get("tvt_true") is not None: cache[w] = c
    pf.CACHE = cache
    ALL = [w for w in wells if w in cache]
    print(f"usable {len(ALL)} 坑井\n")

    errs = []
    for nm, P, d in zip(names, params, isdec):
        e = dec.predict_errs_dec(P, ALL, SEED, SEEDS, CHUNK) if d else pf.predict_errs(P, ALL, SEED, SEEDS, CHUNK)
        errs.append(e)
        print(f"  ran {nm}", flush=True)

    # 坑井ごとに全系列の長さを揃えて連結
    K = len(SETS); cols = [[] for _ in range(K)]
    for w in ALL:
        m = min(len(errs[k][w]) for k in range(K))
        for k in range(K): cols[k].append(np.asarray(errs[k][w])[:m])
    E = np.stack([np.concatenate(cols[k]) for k in range(K)], 1)   # (n,K)
    fin = np.isfinite(E).all(1); Ef = E[fin]; n = len(Ef)
    print(f"\n共通有限行 {n}")

    print("\n[個別 overall RMSE(共通行)]")
    for k, nm in enumerate(names): print(f"  {nm:16s} {rmse(Ef[:, k]):.4f}")

    C = np.corrcoef(Ef.T)
    print("\n[誤差相関行列 Pearson]")
    print("                  " + "  ".join(f"{nm[:8]:>8s}" for nm in names))
    for i, nm in enumerate(names):
        print(f"  {nm:16s} " + "  ".join(f"{C[i,j]:8.3f}" for j in range(K)))

    # Gram で対ターゲット50:50 + 最適2本重み
    G = (Ef.T @ Ef) / n
    def rw(w): w = np.asarray(w, float); return float(np.sqrt(max(w @ G @ w, 0.0)))
    tgt = 0
    print(f"\n[ターゲット={names[tgt]} 対 各メンバー]  (相関 / 50:50ブレンド / 最適2本)")
    for k in range(K):
        if k == tgt: continue
        w50 = np.zeros(K); w50[tgt] = w50[k] = 0.5
        bb = None
        for a in range(0, 21):
            w = np.zeros(K); w[tgt] = a / 20.0; w[k] = 1 - a / 20.0
            r = rw(w)
            if bb is None or r < bb[0]: bb = (r, w[tgt])
        print(f"  vs {names[k]:16s} corr={C[tgt,k]:.3f}  50:50={rw(w50):.4f}  最適 w_{names[tgt]}={bb[1]:.2f}={bb[0]:.4f}")

    # グラフ
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams["font.family"] = "Meiryo"
        fig, ax = plt.subplots(1, 2, figsize=(13, 5))
        vals = [rmse(Ef[:, k]) for k in range(K)]
        bars = ax[0].bar(names, vals, color=["#C44E52", "#4C72B0", "#55A868", "#8172B3", "#CCB974", "#64B5CD"][:K])
        ax[0].tick_params(axis="x", rotation=30)
        for bmk, v in zip(bars, vals): ax[0].text(bmk.get_x()+bmk.get_width()/2, v+0.02, f"{v:.3f}", ha="center", fontsize=9)
        ax[0].set_ylabel("overall RMSE(train,共通行)"); ax[0].set_ylim(min(vals)-0.3, max(vals)+0.3)
        ax[0].set_title(f"個別RMSE(target={names[0]})"); ax[0].grid(axis="y", alpha=0.3)
        im = ax[1].imshow(C, vmin=0.5, vmax=1.0, cmap="RdYlGn_r")
        ax[1].set_xticks(range(K)); ax[1].set_yticks(range(K))
        ax[1].set_xticklabels(names, rotation=30, ha="right"); ax[1].set_yticklabels(names)
        for i in range(K):
            for j in range(K): ax[1].text(j, i, f"{C[i,j]:.3f}", ha="center", va="center", fontsize=10)
        ax[1].set_title("誤差相関(低=直交=多様)"); fig.colorbar(im, ax=ax[1], fraction=0.046)
        fig.tight_layout(); png = ART / "corr_bank_anchor_faithful.png"; fig.savefig(png, dpi=110)
        print(f"\n[png] {png}")
    except Exception as ex:
        print(f"[png skip] {type(ex).__name__}: {ex}")


if __name__ == "__main__":
    main()
