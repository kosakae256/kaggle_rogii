# -*- coding: utf-8 -*-
"""
v69_2 cat6 ヒューリスティック(beam/hold含有)再最適化 — 新PF4バンク版
=====================================================================
やりたいこと(ユーザー 2026-07-15): v27 の cat6 後処理の **beam/hold 含有** を、art_v69_2 の
**新PF 4バンク(pf_ancc_1..4)それぞれ**について、**6分割(長さ n_eval × 地形分散 z_span)ごと**に再最適化。
仮説「長いほど hold 大きい方がいい」を検証。pp は対象外(ユーザー指示)。

cat6 の定義(v27 `submit_selfxynn_heuristic_v27.ipynb` / `tuned-heuristic-only-cat6.ipynb` と同一):
  code = n_bin(長さ: n_eval>N_THR で0/1) が上位, z_bin(地形分散: searchsorted(Z_THR, z_span) で0/1/2) が下位
  → code = 3*n_bin + z_bin ∈ 0..5   (codes 3,4,5 = 長い群)
含有ブレンド(v27 と同一式):
  beam: pred = (1-bw)·pred + bw·(last_known + beam_δ)      [bw = BEAM_W[code]]
  hold: pred = (1-hwt)·pred + hwt·last_known               [hwt = HOLD_W[code]]  ← アンカーへ引き戻す
評価 = overall pooled RMSE(全eval行, Kaggle指標)。true_TVT = last_known + target(=残差)。

データ:
  parquet: v69/artifacts_v69_2/data/train.parquet(well/target/last_known_tvt/pf_ancc_1..4/beam_mean_d)
  z_span/n_eval: 生CSV(rogii-wellbore-geology-prediction/train/{well}__horizontal_well.csv)の eval区間(TVT_input NaN)の
                 Z幅と行数から計算 → art_v69_2/wellcat_v69_2.json にキャッシュ。
出力: art_v69_2/beamhold_cat6_v69_2.json(バンク別 BEAM_W/HOLD_W と RMSE) + beamhold_cat6_v69_2.png(グラフ)。
実行: .venv/Scripts/python.exe v69/tune_beamhold_cat6_v69_2.py
env: BEAM_COL(beam_mean_d) BANKS(1,2,3,4) BW_MAX(0.5) HW_MAX(0.5) STEP(0.05)
     N_THR(4840) Z_THR(136.73,185.51)  RECOMPUTE_THR(0=既存閾値, 1=このデータの中央値/3分位で再導出)
"""
import os, sys, json, glob
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
sys.stdout = open(1, "w", encoding="utf-8", closefd=False)

PROJ = Path(r"C:\Users\kosaka256\Documents\rogii_claude")
ART = PROJ / "v69" / "artifacts_v69"
PQ = ART / "data" / "train.parquet"
DATA_DIR = PROJ / "rogii-wellbore-geology-prediction" / "train"
BEAM_COL = os.environ.get("BEAM_COL", "beam_mean_d")
BANKS = [int(x) for x in os.environ.get("BANKS", "1,2,3,4").split(",")]
BW_MAX = float(os.environ.get("BW_MAX", "0.5")); HW_MAX = float(os.environ.get("HW_MAX", "0.5"))
STEP = float(os.environ.get("STEP", "0.05"))
N_THR = float(os.environ.get("N_THR", "4840"))
Z_THR = tuple(float(x) for x in os.environ.get("Z_THR", "136.73,185.51").split(","))
RECOMPUTE_THR = os.environ.get("RECOMPUTE_THR", "0") == "1"
BW_GRID = np.round(np.arange(0.0, BW_MAX + 1e-9, STEP), 4)
HW_GRID = np.round(np.arange(0.0, HW_MAX + 1e-9, STEP), 4)


def well_cats():
    """各坑井の n_eval(eval行数) と z_span(eval区間Z幅) を生CSVから計算(json キャッシュ)。"""
    fp = ART / "wellcat_v69_2.json"
    if fp.exists():
        return {k: tuple(v) for k, v in json.loads(fp.read_text()).items()}
    out = {}
    files = sorted(glob.glob(str(DATA_DIR / "*__horizontal_well.csv")))
    for i, p in enumerate(files):
        wid = Path(p).stem.replace("__horizontal_well", "")
        df = pd.read_csv(p, usecols=["Z", "TVT_input"])
        ev = df[df["TVT_input"].isna()]
        z = ev["Z"].to_numpy(float); z = z[np.isfinite(z)]
        out[wid] = (int(len(ev)), float(z.max() - z.min()) if z.size else 0.0)
        if (i + 1) % 100 == 0: print(f"  z_span計算 {i+1}/{len(files)}", flush=True)
    fp.write_text(json.dumps(out))
    return out


def code_of(n_eval, z_span, n_thr, z_thr):
    n_bin = int(n_eval > n_thr)
    z_bin = int(np.searchsorted(z_thr, z_span, side="right"))
    return 3 * n_bin + z_bin


def rmse(pred, true):
    e = pred - true; e = e[np.isfinite(e)]
    return float(np.sqrt(np.mean(e ** 2))) if e.size else 1e9


def best_weights(pf, beamTVT, lk, true):
    """1カテゴリ内で (bw,hwt) を総当り最小化。返り値: (bw,hwt,rmse)。"""
    best = (0.0, 0.0, rmse(pf, true))
    for bw in BW_GRID:
        p1 = (1 - bw) * pf + bw * beamTVT
        for hwt in HW_GRID:
            r = rmse((1 - hwt) * p1 + hwt * lk, true)
            if r < best[2]: best = (float(bw), float(hwt), r)
    return best


def main():
    cats = well_cats()
    ne = np.array([cats[w][0] for w in cats]); zs = np.array([cats[w][1] for w in cats])
    n_thr, z_thr = N_THR, Z_THR
    if RECOMPUTE_THR:
        n_thr = float(np.median(ne)); z_thr = tuple(np.quantile(zs, [1/3, 2/3]))
        print(f"[thr再導出] N_THR={n_thr:.1f}  Z_THR={tuple(round(x,2) for x in z_thr)}")
    print(f"[cat6] N_THR={n_thr}  Z_THR={z_thr}  beam列={BEAM_COL}  banks={BANKS}")

    cols = ["well", "target", "last_known_tvt", BEAM_COL] + [f"pf_ancc_{i}" for i in BANKS]
    tb = pq.read_table(PQ, columns=cols).to_pandas()
    well = tb["well"].to_numpy()
    lk = tb["last_known_tvt"].to_numpy(np.float64)
    true = lk + tb["target"].to_numpy(np.float64)                 # true_TVT = last_known + 残差
    beamTVT = lk + tb[BEAM_COL].to_numpy(np.float64)              # beam は δ → 絶対TVT化
    # 行→cat6 code
    codemap = {w: code_of(cats[w][0], cats[w][1], n_thr, z_thr) for w in cats}
    code = np.array([codemap.get(w, 0) for w in well])
    fold = pd.Series(well).astype("category").cat.codes.to_numpy() % 5   # well単位GroupKFold(OOF転移チェック)

    result = {}
    for i in BANKS:
        pf = tb[f"pf_ancc_{i}"].to_numpy(np.float64)
        base = rmse(pf, true)
        BEAM_W, HOLD_W, cat_rmse = {}, {}, {}
        # --- in-sample: カテゴリ別 最適 (bw,hwt) ---
        for c in range(6):
            m = code == c
            if m.sum() == 0:
                BEAM_W[c] = 0.0; HOLD_W[c] = 0.0; cat_rmse[c] = None; continue
            bw, hwt, r = best_weights(pf[m], beamTVT[m], lk[m], true[m])
            BEAM_W[c] = bw; HOLD_W[c] = hwt; cat_rmse[c] = (rmse(pf[m], true[m]), r, int(m.sum()))
        # in-sample 全体適用
        pred = pf.copy()
        for c in range(6):
            m = code == c; bw = BEAM_W[c]; hwt = HOLD_W[c]
            p = (1 - bw) * pf[m] + bw * beamTVT[m]; pred[m] = (1 - hwt) * p + hwt * lk[m]
        insample = rmse(pred, true)
        # --- GroupKFold OOF(過学習/転移チェック): trainで重み決定→valに適用 ---
        oof = pf.copy()
        for f in range(5):
            tr = fold != f; va = fold == f
            for c in range(6):
                mtr = tr & (code == c); mva = va & (code == c)
                if mtr.sum() == 0 or mva.sum() == 0: continue
                bw, hwt, _ = best_weights(pf[mtr], beamTVT[mtr], lk[mtr], true[mtr])
                p = (1 - bw) * pf[mva] + bw * beamTVT[mva]; oof[mva] = (1 - hwt) * p + hwt * lk[mva]
        oof_rmse = rmse(oof, true)
        result[i] = dict(base=base, insample=insample, oof=oof_rmse,
                         BEAM_W=BEAM_W, HOLD_W=HOLD_W, cat=cat_rmse)
        print(f"\n=== pf_ancc_{i} ===  base={base:.4f} → in-sample={insample:.4f}  OOF={oof_rmse:.4f}")
        print(f"  {'code':>4} {'長さ':>4} {'z帯':>4} {'n_rows':>8} {'baseRMSE':>9} {'catRMSE':>9} {'BEAM_W':>7} {'HOLD_W':>7}")
        for c in range(6):
            cr = cat_rmse[c]
            if cr is None: continue
            print(f"  {c:>4} {'長' if c>=3 else '短':>4} {c%3:>4} {cr[2]:>8} {cr[0]:>9.3f} {cr[1]:>9.3f} {BEAM_W[c]:>7.2f} {HOLD_W[c]:>7.2f}")
        long_h = np.mean([HOLD_W[c] for c in (3, 4, 5)]); short_h = np.mean([HOLD_W[c] for c in (0, 1, 2)])
        print(f"  [仮説チェック] HOLD_W 平均: 長群(3-5)={long_h:.3f}  短群(0-2)={short_h:.3f}  "
              f"→ {'長い方がhold大 ○' if long_h > short_h else '長い方がhold小 ×(仮説と逆)'}")

    (ART / "beamhold_cat6_v69_2.json").write_text(json.dumps(
        {str(i): dict(base=r["base"], insample=r["insample"], oof=r["oof"],
                      BEAM_W={str(c): r["BEAM_W"][c] for c in range(6)},
                      HOLD_W={str(c): r["HOLD_W"][c] for c in range(6)}) for i, r in result.items()},
        indent=2, ensure_ascii=False))
    _plot(result)
    print(f"\n[saved] {ART/'beamhold_cat6_v69_2.json'} / .png")


def _plot(result):
    import matplotlib
    matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "Meiryo"
    banks = list(result)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(6); labels = [f"{c}\n{'長' if c>=3 else '短'}/z{c%3}" for c in range(6)]
    for i in banks:
        ax[0].plot(x, [result[i]["HOLD_W"][c] for c in range(6)], "o-", label=f"pf_{i}")
        ax[1].plot(x, [result[i]["BEAM_W"][c] for c in range(6)], "o-", label=f"pf_{i}")
    for a, t in zip(ax, ["HOLD_W(hold含有) — 仮説: 長群(code3-5)>短群(0-2)?", "BEAM_W(beam含有)"]):
        a.set_xticks(x); a.set_xticklabels(labels); a.set_xlabel("cat6 code (長さ×地形分散z)")
        a.set_ylabel("含有 weight"); a.set_title(t); a.grid(alpha=0.3); a.legend()
    ax[0].axvline(2.5, color="gray", ls="--", alpha=0.6)  # 短|長 の境
    fig.suptitle("v69_2 cat6 beam/hold 含有 再最適化(新PF4バンク)")
    fig.tight_layout(); fig.savefig(ART / "beamhold_cat6_v69_2.png", dpi=110)


if __name__ == "__main__":
    main()
