# -*- coding: utf-8 -*-
"""
typewell GR × horizontal GR 相関: 生 vs ハイパス
=================================================
狙い: 「真TVTで typewell から引いた期待GR」と「水平井の観測GR」の相関を、
      生(そのまま) と ハイパス(GR − 長窓移動平均) で比べ、窓長ごとに変化を見る。
      → 低周波(baseline/offset/ドリフト)を落とすと照合(相関)が良くなるか悪くなるかを直接判定。

判定:
  ハイパスで相関UP → 低周波は交絡(offset/ドリフトがズレていた) = ハイパス有効(照合が良化)
  ハイパスで相関DOWN → 低周波は共有信号だった = ハイパスは情報を捨てる(逆効果)
  相関がピークになる窓長 = 最適カットオフの目安

方法(各坑井):
  - eval行(TVT_input=NaN, TVT=真値あり)を対象。
  - 期待GR = typewell GR を真TVTで補間: exp_gr(row) = interp(TVT_row, tw_TVT, tw_GR)。
  - 観測GR = 水平井 GR(行)。
  - 生相関 = corr(obs_gr, exp_gr)。
  - ハイパス: obs_hp = obs − rollmean(obs, W), exp_hp = exp − rollmean(exp, W)(両方に同じ窓)。
             hp相関 = corr(obs_hp, exp_hp)。W を複数振る。
  - 坑井ごとの Pearson r を集計(中央値/平均, および 生から改善した坑井の割合)。

実行: rogii_claude/.venv/Scripts/python.exe "検証/gr_typewell_correlation.py"
※ Claude は実行しない(ユーザーが実行)。
"""

import glob
from pathlib import Path
import numpy as np
import pandas as pd

# ============================== CONFIG ==============================
PROJECT  = Path(r"C:\Users\kosaka256\Documents\rogii_claude")
DATA_DIR = PROJECT / "rogii-wellbore-geology-prediction" / "train"

N_WELLS  = None                        # None=全well, 整数で先頭N
WINDOWS  = [None, 300, 200, 100, 50, 30, 20, 10]   # None=生。数値=ハイパス窓(ft≈行数, dMD=1ft)
MIN_ROWS = 60                          # eval行がこれ未満の坑井はスキップ
# ===================================================================


def list_wells():
    return sorted({Path(p).stem.replace("__horizontal_well", "")
                   for p in glob.glob(str(DATA_DIR / "*__horizontal_well.csv"))})


def highpass(x, W):
    """x − 中央移動平均(窓W)。W=None ならそのまま。"""
    if W is None:
        return x
    s = pd.Series(x, dtype="float64")
    ma = s.rolling(int(W), center=True, min_periods=1).mean()
    return (s - ma).to_numpy()


def pearson(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if len(a) < 10 or a.std() < 1e-9 or b.std() < 1e-9:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def process_well(wid):
    """1坑井の (obs_gr, exp_gr) を eval行で返す。無効なら None。"""
    try:
        hw = pd.read_csv(DATA_DIR / f"{wid}__horizontal_well.csv", usecols=["GR", "TVT", "TVT_input"])
        tw = pd.read_csv(DATA_DIR / f"{wid}__typewell.csv", usecols=["TVT", "GR"]).sort_values("TVT")
    except Exception:
        return None
    if "TVT" not in hw.columns:
        return None
    ev = hw[hw["TVT_input"].isna()]
    if len(ev) < MIN_ROWS:
        return None
    tvt = ev["TVT"].to_numpy(float); obs = ev["GR"].to_numpy(float)
    tw_tvt = tw["TVT"].to_numpy(float); tw_gr = tw["GR"].to_numpy(float)
    ok = np.isfinite(tvt) & np.isfinite(obs)
    if ok.sum() < MIN_ROWS or len(tw_tvt) < 3:
        return None
    tvt, obs = tvt[ok], obs[ok]
    # obs の欠損は既に除外。exp = typewell GR @ 真TVT(範囲外は端clamp=np.interp既定)
    exp = np.interp(tvt, tw_tvt, tw_gr)
    return obs, exp


def main():
    print("=" * 70)
    print("typewell GR × horizontal GR 相関: 生 vs ハイパス")
    print("=" * 70)
    wells = list_wells()
    if N_WELLS:
        wells = wells[:N_WELLS]

    # 各坑井の (obs, exp) を先に取得(窓ごとに使い回す)
    data = []
    for w in wells:
        r = process_well(w)
        if r is not None:
            data.append(r)
    print(f"対象坑井: {len(data)} / {len(wells)}")
    if not data:
        print("有効な坑井なし"); return

    # 生相関(基準)を先に各坑井で計算
    raw_corr = np.array([pearson(obs, exp) for obs, exp in data])
    raw_med = np.nanmedian(raw_corr); raw_mean = np.nanmean(raw_corr)

    print("\n窓長ごとの相関(Pearson r, 坑井集計)")
    print(f"{'window(ft)':>10} | {'median r':>9} | {'mean r':>8} | {'Δmed vs raw':>11} | {'改善坑井%':>8}")
    print("-" * 60)
    for W in WINDOWS:
        rs = []
        for obs, exp in data:
            o = highpass(obs, W); e = highpass(exp, W)
            rs.append(pearson(o, e))
        rs = np.array(rs)
        med = np.nanmedian(rs); mean = np.nanmean(rs)
        if W is None:
            tag = "raw"; dmed = 0.0; imp = np.nan
        else:
            tag = f"{W}"; dmed = med - raw_med
            imp = float(np.mean((rs > raw_corr)[np.isfinite(rs) & np.isfinite(raw_corr)]) * 100)
        imp_s = "   -  " if not np.isfinite(imp) else f"{imp:6.1f}"
        print(f"{tag:>10} | {med:9.4f} | {mean:8.4f} | {dmed:+11.4f} | {imp_s:>8}")

    print("\n--- 解釈 ---")
    print(f"  生の相関: median={raw_med:.4f} mean={raw_mean:.4f}")
    print("  ・ハイパスで median r が上がる窓があれば → その帯(低周波)は交絡。ハイパスで照合が良化する見込み。")
    print("  ・どの窓でも下がる → 低周波は共有信号。ハイパスは情報を捨てるだけ(PFに入れない方が良い)。")
    print("  ・改善坑井% が高い窓ほど、広く効く安全なカットオフ。")
    print("  ※ これは真TVTを使った“相関の上限”診断。実PFは真TVTを使えないが、相関が上がる=照合の谷が深くなる示唆。")


if __name__ == "__main__":
    main()
