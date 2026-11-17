# -*- coding: utf-8 -*-
"""v36 逆方向DA: 井戸を反転した train_backward.parquet を生成する(逆方向のみ実行)。
=====================================================================================
方式: **CSVレベルで井戸を反転**(行順反転 + MD' = MDmax − MD)し、toe側の実区間 K 行を
疑似既知帯(TVT_input := TVT)にした「逆向きの合成タスク」を作り、**既存の create パイプライン
(v28/create_train_parquet_v28.py の build_dataset_v28)にそのまま通す**。
→ md_since / PFアンカー(ls, ir) / beam / NCC / self grid 等の全特徴が鏡像世界で自動的に整合。

- **反転CSVは元のevalゾーンの行だけで構成**(元の既知帯は予測対象にしない=ユーザー指示。
  既知帯行はCSVごと除外する。末尾に残すとPFのprefix前提が壊れるため除外が唯一安全)。
- K = min(元既知帯の行数, 元evalの半分)、下限30。逆方向の行数 = **n_ev − K(事前区間の分だけ減る)**。
  n_ev − K < 50 の井戸はskip。
- 近傍(nbr)参照は他井戸の (GR, TVT) ペア集合なので行順反転の影響なし(binning は順序不変)。
- 出力に is_backward=1 列を付与。**順方向の parquet には触らない**。
- 学習(train_models_v36.ipynb): train = 順+逆(井戸単位GroupKFold、逆行は同じ井戸=同じfold側で
  リークなし)/ **valid・OOF評価 = 順方向のみ**(本番タスクと同じ分布で測る)。

出力: v36/artifacts_v36/data/train_backward.parquet
実行: .venv/Scripts/python.exe v36/create_train_backward_v36.py
(計算量は v28 create 1回分: 9本のsmooth PFパス+build_well。GPU数時間級)
"""
import os, sys, glob, shutil, time
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT = Path(r"C:\Users\kosaka256\Documents\rogii_claude")
sys.path.insert(0, str(PROJECT / "v28"))
import create_train_parquet_v28 as C

SRC_TRAIN = C.DATA_DIR_V26 / "train"
ART = PROJECT / "v36" / "artifacts_v36"
REV_ROOT = ART / "data" / "rev_csv"          # 反転CSVの置き場(REV_ROOT/train/*.csv)
REV_TRAIN = REV_ROOT / "train"
OUT = ART / "data" / "train_backward.parquet"
K_MIN, EV_MIN = 30, 50

# ---------- (1) 反転CSVの生成 ----------
REV_TRAIN.mkdir(parents=True, exist_ok=True)
wells = sorted({os.path.basename(q).split("__")[0] for q in glob.glob(str(SRC_TRAIN / "*__horizontal_well.csv"))})
n_ok = n_skip = 0
t0 = time.time()
for w in wells:
    hp = SRC_TRAIN / f"{w}__horizontal_well.csv"
    tp = SRC_TRAIN / f"{w}__typewell.csv"
    if not tp.exists():
        n_skip += 1; continue
    hw = pd.read_csv(hp)
    if "TVT" not in hw.columns or hw["TVT"].isna().all():
        n_skip += 1; continue
    kn = hw["TVT_input"].notna().to_numpy()
    n_kn = int(kn.sum()); n_ev = int((~kn).sum())
    K = min(n_kn, n_ev // 2)
    if K < K_MIN or (n_ev - K) < EV_MIN:
        n_skip += 1; continue
    ev = hw[~kn]                                            # ★元のevalゾーンのみ(既知帯は除外)
    hw2 = ev.iloc[::-1].reset_index(drop=True).copy()
    hw2["MD"] = float(ev["MD"].max()) - hw2["MD"]           # MD再び昇順
    head_tvt = hw2["TVT"].to_numpy()[:K]                    # toe側K行=疑似既知帯(真値あり)
    if not np.isfinite(head_tvt).all():
        n_skip += 1; continue
    tvt_in = np.full(len(hw2), np.nan)
    tvt_in[:K] = head_tvt
    hw2["TVT_input"] = tvt_in                               # 予測対象 = 元evalの n_ev−K 行のみ
    hw2.to_csv(REV_TRAIN / f"{w}__horizontal_well.csv", index=False)
    dst_tp = REV_TRAIN / f"{w}__typewell.csv"
    if not dst_tp.exists():
        shutil.copyfile(tp, dst_tp)                        # typewell はそのまま
    n_ok += 1
print(f"[rev] 反転CSV {n_ok}井戸(skip {n_skip}){time.time()-t0:.0f}s → {REV_TRAIN}")

# ---------- (2) 既存パイプラインを反転ディレクトリに向けて実行 ----------
C.DATA_DIR_V26 = REV_ROOT       # build_dataset_v28 / 近傍プール / _ref_grtvt が全てここを読む
C.SPLIT = "train"; C.IS_TRAIN = True; C.N_WELLS = None
C._TRAIN_CENT = None; C._REFC.clear()                     # 近傍キャッシュをリセット
print(f"[rev] build_dataset_v28 実行(device={C.V26_DEVICE}, n_seed={C.N_SEED}, lag={C.SMOOTH_LAG})")
df = C.build_dataset_v28()
df["is_backward"] = np.float32(1.0)
OUT.parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(OUT, index=False)
print(f"saved: {OUT}  rows={len(df)} cols={len(df.columns)}")
print("※ 学習は train_models_v36.ipynb(順=v36配置のtrain.parquet + 逆=本ファイルをconcat、検証は順のみ)")
print("※ rev_csv は再生成可能なので削除してよい")
