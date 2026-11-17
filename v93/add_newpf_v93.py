# -*- coding: utf-8 -*-
"""newpf_train.npz(新PF delta/std)を train.parquet に列追加して train_plus1pf.parquet を作る。
   ★既存 train.parquet は上書きしない(新ファイル)。列名: newpf_delta / newpf_std。
実行: rogii_claude/.venv/Scripts/python.exe v93/add_newpf_v93.py"""
import io, sys
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
from pathlib import Path
import numpy as np, pyarrow as pa, pyarrow.parquet as pq

ART = Path(r"C:\Users\kosaka256\Documents\rogii_claude") / "v93" / "artifacts_v93"
SRC = ART / "data" / "train.parquet"
OUT = ART / "data" / "train_plus1pf.parquet"
d = np.load(ART / "newpf_train.npz", allow_pickle=True)
delta = d["delta"].astype(np.float32); std = d["std"].astype(np.float32)
pfin = pq.ParquetFile(SRC)
assert pfin.metadata.num_rows == len(delta), f"行数不一致 {pfin.metadata.num_rows} vs {len(delta)}"
print(f"[add newpf] {SRC.name}({pfin.metadata.num_rows}行) + newpf_delta/std -> {OUT.name}", flush=True)
writer = None; off = 0
for rg in range(pfin.num_row_groups):
    tbl = pfin.read_row_group(rg); n = tbl.num_rows
    tbl = tbl.append_column("newpf_delta", pa.array(delta[off:off + n]))
    tbl = tbl.append_column("newpf_std", pa.array(std[off:off + n]))
    off += n
    if writer is None:
        writer = pq.ParquetWriter(OUT, tbl.schema)
    writer.write_table(tbl)
writer.close()
print(f"saved {OUT}  rows={off}  (+newpf_delta, +newpf_std)")
