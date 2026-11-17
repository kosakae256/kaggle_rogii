"""
v9 の特徴量(~511列)を v11 1D-UNet で学習する(PF 再計算なし・現行パラメータ)。

  - 入力: v9/artifacts_v9/data/train.parquet
  - in_ch は特徴量数から自動設定(56 ではなく ~511)。
  - 成果物は artifacts_v11/v9feat/ に分離(56特徴モデルを壊さない)。

実行(ユーザー):
  .venv\\Scripts\\python.exe v11\\train_v9feat.py
"""

from pathlib import Path
import joblib
import torch

import train as T  # v11 train.py(run_cv_training 等を再利用)
from model import UNetRegConfig
from data_v9 import load_v9_well_data, PARQUET_DEFAULT


def main():
    print(f"─── v9 特徴量を読み込み: {PARQUET_DEFAULT} ───")
    well_data, feat_cols = load_v9_well_data()
    F = len(feat_cols)
    print(f"  wells={len(well_data)}  features(in_ch)={F}")

    # 成果物を v9feat サブディレクトリへ振り分け(run_cv_training はモジュール
    # グローバルの MODEL_DIR / NORM_FILE を参照するので、呼び出し前に差し替える)
    base = T.CFG.artifacts_path / "v9feat"
    base.mkdir(parents=True, exist_ok=True)
    T.MODEL_DIR = str(base / "models_sca_u2net")
    T.NORM_FILE = str(base / "norm_stats.pkl")
    joblib.dump(feat_cols, base / "feature_cols.pkl")  # test 時の列順再現用

    cfg_u = UNetRegConfig(in_ch=F)
    cfg_t = T.TrainConfig()

    models, oof_preds, mean, std, cv_rmse = T.run_cv_training(
        well_data, T.DEVICE, cfg_t, cfg_u
    )
    print(f"\n=== v9-feature UNet  CV(TVT) RMSE = {cv_rmse:.4f} ===")
    print(f"  models/oof saved under: {base}")


if __name__ == "__main__":
    main()
