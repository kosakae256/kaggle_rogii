# -*- coding: utf-8 -*-
"""v30 r1: 緩い錨の閉ループ軌跡(ρの階段)を特徴列にして train parquet を作り直す。
- 入力: v28 train.parquet + v30 preds_sig{key}.npz(CL_SIGS で選択)
- 追加列(σごと): cl{key}_res = cl − lk / cl{key}_vs_pf = cl − pf_ancc
  クロス: cl_spread = σ間 max−min / cl_disp = σ間 std(閉ループ同士の不一致=不確実性)
- prior そのもの・cl−prior は入れない(仕上げ済み予測の特徴化=v24罠を回避)
- 出力: v30/artifacts_v30/train_r1.parquet(元列+新列, pyarrowで省メモリ結合)
実行: rogii_claude/.venv/Scripts/python.exe v30/make_r1_parquet_v30.py
"""
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

PROJECT = Path(r"C:\Users\kosaka256\Documents\rogii_claude")
V28_PQ  = PROJECT / "v28" / "artifacts_v28" / "data" / "train.parquet"
OUT_DIR = PROJECT / "v30" / "artifacts_v30"
OUT_PQ  = OUT_DIR / "train_r1.parquet"

CL_SIGS = ["none", "192", "96"]  # ρの階段(none 0.71 / 192 0.85 / 96 0.90, 2026-07-06 診断より)

def main():
    import pandas as pd
    tbl = pq.read_table(V28_PQ)
    print("base parquet:", tbl.num_rows, "rows,", tbl.num_columns, "cols")
    g   = tbl.column("well").to_pandas().to_numpy()
    lk  = tbl.column("last_known_tvt").to_numpy(zero_copy_only=False).astype(np.float64)
    mds = tbl.column("md_since").to_numpy(zero_copy_only=False).astype(np.float64)
    pfa = tbl.column("pf_ancc").to_numpy(zero_copy_only=False).astype(np.float64)
    N = tbl.num_rows

    # 井戸ごとの行 index(md_since 昇順) — v30 npz の並びと一致
    order = {}
    for w in pd.unique(g):
        rows = np.where(g == w)[0]
        order[w] = rows[np.argsort(mds[rows], kind="stable")]

    cl_abs = {}
    for key in CL_SIGS:
        fp = OUT_DIR / f"preds_sig{key}.npz"
        if not fp.exists():
            raise FileNotFoundError(f"{fp} が無い(closed_loop_v30.py で σ={key} を先に回す)")
        z = np.load(fp, allow_pickle=True)
        arr = np.full(N, np.nan)
        miss = 0
        for w, o in order.items():
            k = f"s_{w}"
            if k in z and len(z[k]) == len(o):
                arr[o] = z[k]
            else:
                miss += 1
        cl_abs[key] = arr
        print(f"cl[{key}]: filled {np.isfinite(arr).sum()}/{N} rows (miss wells={miss})")

    stack = np.stack([cl_abs[k] for k in CL_SIGS], axis=0)
    newcols = {}
    for key in CL_SIGS:
        newcols[f"cl{key}_res"]   = (cl_abs[key] - lk).astype(np.float32)
        newcols[f"cl{key}_vs_pf"] = (cl_abs[key] - pfa).astype(np.float32)
    newcols["cl_spread"] = (np.nanmax(stack, 0) - np.nanmin(stack, 0)).astype(np.float32)
    newcols["cl_disp"]   = np.nanstd(stack, 0).astype(np.float32)

    for name, arr in newcols.items():
        if name in tbl.column_names:
            tbl = tbl.drop([name])
        tbl = tbl.append_column(name, pa.array(arr, type=pa.float32()))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pq.write_table(tbl, OUT_PQ)
    print(f"saved: {OUT_PQ} ({tbl.num_rows} rows, {tbl.num_columns} cols, +{len(newcols)}列: {list(newcols)})")

if __name__ == "__main__":
    main()
