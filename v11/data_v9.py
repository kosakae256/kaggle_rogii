"""
v9 の train.parquet を v11(1D-UNet)用の坑井別シーケンスに変換するローダ。

v9 build_well が吐いた ~511 特徴量を「そのまま」UNet の入力チャネルにする。
PF を再計算しないので最速で検証できる(現行 PF パラメータのまま)。

返り値の各 dict は WellSeqDataset / run_cv_training がそのまま使える形:
  {'wid', 'features'[L,F], 'target'[L], 'last_tvt'(float), 'md_since'[L], 'n_eval'}
"""

from pathlib import Path
import numpy as np
import pandas as pd

PARQUET_DEFAULT = (
    Path(__file__).resolve().parent.parent / "v9" / "artifacts_v9" / "data" / "train.parquet"
)

# 特徴量に含めない列。last_known_tvt は v9 と同様に特徴量として残す
# (絶対深度アンカー。test でも既知なのでリークではない)。
NON_FEATURES = ("well", "id", "target")


def _row_order(id_series: pd.Series) -> np.ndarray:
    """id = '{wid}_{rowidx}' の末尾整数 = 元の MD 順。坑井内の並べ替えに使う。"""
    return id_series.str.rsplit("_", n=1).str[-1].astype(np.int64).to_numpy()


def load_v9_well_data(parquet_path=PARQUET_DEFAULT, feature_cols=None, wells=None):
    """parquet を坑井別シーケンスのリストに変換する。

    parquet_path : v9 train.parquet
    feature_cols : 使う特徴量列(None なら NON_FEATURES 以外を全部)。
                   test 側と列順を揃えるために保存・再利用する想定。
    wells        : 指定すると該当坑井だけ読み込む(動作確認・省メモリ用)。
    戻り値       : (well_data_list, feature_cols)
    """
    read_kwargs = {}
    if wells is not None:
        read_kwargs["filters"] = [("well", "in", list(wells))]
    df = pd.read_parquet(parquet_path, **read_kwargs)

    if feature_cols is None:
        feature_cols = [c for c in df.columns if c not in NON_FEATURES]

    df = df.assign(_ord=_row_order(df["id"]))

    well_data = []
    for wid, sub in df.groupby("well", sort=False):
        sub = sub.sort_values("_ord")
        feats = sub[feature_cols].to_numpy(np.float32)
        feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
        md_since = (
            sub["md_since"].to_numpy(np.float32)
            if "md_since" in sub.columns
            else np.arange(len(sub), dtype=np.float32)
        )
        well_data.append({
            "wid": str(wid),
            "features": feats,  # nan_to_num が writable コピーを返す
            "target": sub["target"].to_numpy(np.float32).copy(),
            "last_tvt": float(sub["last_known_tvt"].iloc[0]),
            "md_since": md_since.copy(),
            "n_eval": len(sub),
        })
    return well_data, feature_cols


if __name__ == "__main__":
    wd, cols = load_v9_well_data()
    F = len(cols)
    L = [d["n_eval"] for d in wd]
    print(f"wells={len(wd)}  features={F}")
    print(f"seq len: min={min(L)} max={max(L)} mean={int(np.mean(L))}")
    print(f"sample features shape: {wd[0]['features'].shape}  last_tvt={wd[0]['last_tvt']:.2f}")
