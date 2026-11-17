# -*- coding: utf-8 -*-
"""
v66 [2-1] クロスウェル・ペア勾配の前提テスト = ANCC datum 整合性の診断
=====================================================================
仮説: 現行の場は各坑井が「自分の進行方向の傾き」しか語れない(平行掘りfieldでは
走向勾配が原理的に見えない = 896d15b9 型の根本原因)。train は ANCC が全行既知なので、
近接2坑井の ΔANCC/距離 = ペア方位の方向微分が取れ、平行lateralの横断ペアが走向勾配を直接与える。

ただし TVT=ANCC−Z+b_well(error_anatomy §1)= ANCC に per-well offset b がある可能性。
b があると cross-well ΔANCC は (b_i−b_j) で汚染 → ペア勾配が使えない。ここでそれを測る。

測ること:
  Q1 datum整合: 近接クロス点(<Rc ft)の ΔANCC 分布。中央 |ΔANCC| が dip×距離 で説明できる範囲か。
  Q2 offset性: ペア差が「per-well定数 o_i−o_j」で説明できるか(=除去可能)。三角整合の残差で判定。
  Q3 走向勾配の可視性: 局所クロスウェル平面フィット ANCC≈a+gx·x+gy·y の走向成分(θ0直交)が
     単一坑井 dip と比べて有意/一貫か。ここが「新情報」の実体。
出力: v66/artifacts_v66/crosswell_diag.png + 標準出力サマリ。read-only(場は書かない)。
実行: .venv/Scripts/python.exe v66/diag_crosswell_v66.py
env : RC(120 近接判定ft) RMAX(500 局所平面の近傍ft) NPAIR_MAX(4000) MINLAT(150)
"""
import sys, os, glob
from pathlib import Path
import numpy as np, pandas as pd

sys.stdout = open(1, "w", encoding="utf-8", closefd=False)
PROJECT = Path(r"C:\Users\kosaka256\Documents\rogii_claude")
OUT = PROJECT / "v66" / "artifacts_v66"; OUT.mkdir(parents=True, exist_ok=True)
TRAIN_DIR = PROJECT / "rogii-wellbore-geology-prediction" / "train"
THETA0 = 118.4
_C0, _S0 = np.cos(np.radians(THETA0)), np.sin(np.radians(THETA0))
RC = float(os.environ.get("RC", "120"))
RMAX = float(os.environ.get("RMAX", "500"))
NPAIR_MAX = int(os.environ.get("NPAIR_MAX", "4000"))
MINLAT = int(os.environ.get("MINLAT", "150"))


def load_laterals():
    """各train坑井の lateral 区間(既知末尾以降)の X,Y,ANCC を集める。"""
    wells = []
    for fp in sorted(glob.glob(str(TRAIN_DIR / "*__horizontal_well.csv"))):
        nm = os.path.basename(fp).split("__")[0]
        try:
            w = pd.read_csv(fp, usecols=["X", "Y", "Z", "ANCC", "TVT_input"])
        except Exception:
            continue
        ti = w["TVT_input"].to_numpy(float)
        anc = w["ANCC"].to_numpy(float)
        fin = np.where(np.isfinite(ti))[0]
        if len(fin) == 0:
            continue
        s = int(fin.max())
        X = w["X"].to_numpy(float)[s:]; Y = w["Y"].to_numpy(float)[s:]; A = anc[s:]
        m = np.isfinite(X) & np.isfinite(Y) & np.isfinite(A)
        if m.sum() < MINLAT:
            continue
        wells.append(dict(nm=nm, X=X[m], Y=Y[m], A=A[m], wi=len(wells)))
    return wells


def main():
    from scipy.spatial import cKDTree
    wells = load_laterals()
    allX = np.concatenate([w["X"] for w in wells])
    allY = np.concatenate([w["Y"] for w in wells])
    allA = np.concatenate([w["A"] for w in wells])
    allW = np.concatenate([np.full(len(w["X"]), w["wi"]) for w in wells])
    print(f"[data] ANCC 有効 lateral 坑井 {len(wells)}  総点 {len(allX)}")
    tree = cKDTree(np.column_stack([allX, allY]))

    # ---- Q1: 近接クロス点の ΔANCC(< RC ft, 別坑井)----
    dpair, ancdiff, wpair = [], [], []
    rng_pts = np.linspace(0, len(allX) - 1, min(len(allX), 60000)).astype(int)
    for i in rng_pts:
        idx = tree.query_ball_point([allX[i], allY[i]], RC)
        for j in idx:
            if allW[j] <= allW[i]:      # 別坑井・重複回避(i<j well)
                continue
            d = np.hypot(allX[j] - allX[i], allY[j] - allY[i])
            if d < 1.0:
                continue
            dpair.append(d); ancdiff.append(allA[j] - allA[i])
            wpair.append((int(allW[i]), int(allW[j])))
    dpair = np.array(dpair); ancdiff = np.array(ancdiff)
    print(f"\n[Q1 datum整合] 近接クロス点ペア {len(dpair)}(<{RC:.0f}ft)")
    if len(dpair):
        print(f"  |ΔANCC|  中央 {np.median(np.abs(ancdiff)):.2f}  p90 {np.quantile(np.abs(ancdiff),0.9):.2f}  最大 {np.abs(ancdiff).max():.1f} ft")
        print(f"  ΔANCC/距離 中央 {np.median(np.abs(ancdiff)/np.maximum(dpair,1)):.4f} ft/ft(参考: 単一坑井 dip κ≈0.1〜0.8)")

    # ---- Q2: ペア差は per-well 定数 offset で説明できるか ----
    #   各坑井ペア(i,j)で ΔANCC の中央値 m_ij を取り、o_i−o_j = m_ij を最小二乗で解く。
    #   残差(定数で説明できない分)が小さいほど「offsetは per-well 定数=除去可能」。
    from collections import defaultdict
    byp = defaultdict(list)
    for (a, b), dv in zip(wpair, ancdiff):
        byp[(a, b)].append(dv)
    pkeys = [k for k, v in byp.items() if len(v) >= 3]
    if len(pkeys) >= 10:
        widx = sorted(set([k[0] for k in pkeys] + [k[1] for k in pkeys]))
        wmap = {w: i for i, w in enumerate(widx)}
        rows, yv = [], []
        for (a, b) in pkeys:
            r = np.zeros(len(widx)); r[wmap[a]] = 1; r[wmap[b]] = -1
            rows.append(r); yv.append(np.median(byp[(a, b)]))
        Rm = np.array(rows); yv = np.array(yv)
        Rm = np.column_stack([Rm, np.ones(len(Rm))])       # ゲージ自由度を吸収する定数列
        sol = np.linalg.lstsq(Rm, yv, rcond=None)[0]
        pred = Rm @ sol
        resid = yv - pred
        raw_rms = float(np.sqrt(np.mean(yv ** 2)))
        res_rms = float(np.sqrt(np.mean(resid ** 2)))
        print(f"\n[Q2 offset性] 坑井ペア {len(pkeys)}  坑井 {len(widx)}")
        print(f"  ペア差 ΔANCC 中央値の RMS = {raw_rms:.2f} ft")
        print(f"  per-well 定数で説明後の残差 RMS = {res_rms:.2f} ft  (小さいほど offset は定数=除去可能)")
        print(f"  → offset の坑井間 std ≈ {np.std(sol[:-1]):.2f} ft")
        if res_rms < 0.5 * raw_rms:
            print("  判定: ★ペア差の過半が per-well 定数 offset。除去すれば勾配抽出が有望"
                  "(ただし平行lateralでは offset と走向勾配が交絡=要 anchor 縛り)")
        else:
            print("  判定: 定数offsetで説明できない残差が大。ANCCは素直な共通datumでない → ペア勾配は要注意")

    # ---- Q3: 局所クロスウェル平面フィットで走向勾配を回収 ----
    #   各サンプル中心の RMAX 近傍(別坑井を必ず含む)で ANCC≈a+gx·x+gy·y を頑健フィット。
    #   走向成分 g_perp = g·(−S0,C0)、傾斜成分 g_par = g·(C0,S0)。
    centers = np.linspace(0, len(allX) - 1, min(len(allX), NPAIR_MAX)).astype(int)
    gpar, gperp, nwell_local, planeres = [], [], [], []
    for i in centers:
        idx = tree.query_ball_point([allX[i], allY[i]], RMAX)
        if len(idx) < 12:
            continue
        idx = np.array(idx)
        ws = allW[idx]
        if len(np.unique(ws)) < 2:      # クロスウェルでない(走向見えない)→除外
            continue
        x = allX[idx] - allX[i]; y = allY[idx] - allY[i]; a = allA[idx] - allA[i]
        Xd = np.column_stack([np.ones(len(x)), x, y])
        sol, *_ = np.linalg.lstsq(Xd, a, rcond=None)
        gx, gy = sol[1], sol[2]
        res = a - Xd @ sol
        gpar.append(gx * _C0 + gy * _S0)
        gperp.append(-gx * _S0 + gy * _C0)
        nwell_local.append(len(np.unique(ws)))
        planeres.append(np.sqrt(np.mean(res ** 2)))
    gpar = np.array(gpar); gperp = np.array(gperp); planeres = np.array(planeres)
    print(f"\n[Q3 走向勾配の可視性] クロスウェル局所平面 {len(gpar)}(近傍≥2坑井, R={RMAX:.0f}ft)")
    if len(gpar):
        print(f"  傾斜成分 |g_par|  中央 {np.median(np.abs(gpar)):.4f} ft/ft")
        print(f"  走向成分 |g_perp| 中央 {np.median(np.abs(gperp)):.4f} ft/ft  ← これが単一坑井で見えない新情報")
        print(f"  走向/傾斜 比 中央 {np.median(np.abs(gperp))/max(np.median(np.abs(gpar)),1e-6):.2f}")
        print(f"  平面フィット残差 中央 {np.median(planeres):.2f} ft(小=面が局所平面で近似でき勾配が信頼できる)")
        strong = np.abs(gperp) > 0.02
        print(f"  |g_perp|>0.02 の割合 {strong.mean():.1%}(走向勾配が実在に効く領域の広さ)")

    # ---- PNG ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams["font.family"] = "Meiryo"
        fig, ax = plt.subplots(1, 3, figsize=(16, 5))
        if len(dpair):
            ax[0].scatter(dpair, ancdiff, s=4, alpha=0.3)
            ax[0].set_xlabel("クロス点 距離 [ft]"); ax[0].set_ylabel("ΔANCC [ft]")
            ax[0].set_title(f"Q1 datum整合(中央|ΔANCC| {np.median(np.abs(ancdiff)):.2f}ft)")
            ax[0].axhline(0, color="k", lw=0.5)
        if len(gpar):
            ax[1].hist(gperp, bins=80, alpha=0.7, label="g_perp(走向)")
            ax[1].hist(gpar, bins=80, alpha=0.5, label="g_par(傾斜)")
            ax[1].set_xlabel("局所勾配 [ft/ft]"); ax[1].set_ylabel("頻度"); ax[1].legend()
            ax[1].set_title("Q3 勾配成分分布"); ax[1].set_xlim(-0.15, 0.15)
            ax[2].scatter(gpar, gperp, s=4, alpha=0.3)
            ax[2].set_xlabel("g_par(傾斜)"); ax[2].set_ylabel("g_perp(走向)")
            ax[2].set_title("傾斜 vs 走向 勾配"); ax[2].axhline(0, color="k", lw=0.5); ax[2].axvline(0, color="k", lw=0.5)
            ax[2].set_xlim(-0.15, 0.15); ax[2].set_ylim(-0.15, 0.15)
        fig.tight_layout(); p = OUT / "crosswell_diag.png"; fig.savefig(p, dpi=120); plt.close(fig)
        print(f"\n[png] {p}")
    except Exception as e:
        print(f"[png] スキップ({e})")


if __name__ == "__main__":
    main()
