# -*- coding: utf-8 -*-
"""
v81-H — prefix-cutデータ拡張用の疑似データセット生成
=====================================================
各train井戸を「prefixの65%でカット」した疑似井戸に変換して物理CSVとして書き出す:
  - 行はprefix末端まで(真のeval区間は落とす)
  - TVT_input はカット行以降を NaN(=疑似eval区間。TVT列=真値は残る→create側でtargetになる)
  - typewell はコピー
出力: v81/data_cut65/train/*.csv  (createは V79_DATA_OVERRIDE でここを読む)
使い方: python v81/make_cut_dataset_v81.py
次: V79_DATA_OVERRIDE=.../v81/data_cut65 V79_OUT_OVERRIDE=.../v79/artifacts_v79/data/train_prefixcut.parquet \
    python v79/create_candidates_v79.py
    → train notebooks の BWD_PQ を train_prefixcut.parquet に向け USE_BACKWARD=True(=is_augmented指示列)
"""
import io, sys, glob, os, shutil
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
from pathlib import Path
import numpy as np, pandas as pd

P0 = Path(r"C:\Users\kosaka256\Documents\rogii_claude")
SRC = P0 / "rogii-wellbore-geology-prediction" / "train"
DST = P0 / "v81" / "data_cut65" / "train"
DST.mkdir(parents=True, exist_ok=True)
CUT = float(os.environ.get("CUT", "0.65"))
MIN_PSEUDO = 400                                   # 疑似eval最低行数

n_ok = n_skip = 0
for fp in sorted(glob.glob(str(SRC / "*__horizontal_well.csv"))):
    wid = os.path.basename(fp).split("__")[0]
    hw = pd.read_csv(fp)
    kn_idx = hw.index[hw["TVT_input"].notna()]
    ci = int(len(kn_idx) * CUT)
    if ci < 100 or len(kn_idx) - ci < MIN_PSEUDO:
        n_skip += 1; continue
    hw2 = hw.loc[:kn_idx[-1]].copy()               # prefix末端まで(真のevalは落とす)
    hw2.loc[kn_idx[ci + 1:], "TVT_input"] = np.nan
    hw2.to_csv(DST / f"{wid}__horizontal_well.csv", index=False)
    tw_src = SRC / f"{wid}__typewell.csv"
    if tw_src.exists():
        shutil.copyfile(tw_src, DST / f"{wid}__typewell.csv")
    n_ok += 1
print(f"[v81-H] 生成 {n_ok}井 / スキップ {n_skip}井(prefix短) → {DST}")
