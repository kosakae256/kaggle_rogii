# -*- coding: utf-8 -*-
"""newpf_bank_train.npz(7番bank tw/self/nbr の delta+std 6列)を train.parquet に追加して
   train_plus.parquet を作る(★既存 train.parquet 上書きしない)。列: pf_{ancc,self,nbr}_{delta,std}_7"""
import io, sys
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
from pathlib import Path
import numpy as np, pyarrow as pa, pyarrow.parquet as pq

ART = Path(r"C:\Users\kosaka256\Documents\rogii_claude") / "v93" / "artifacts_v93"
SRC = ART / "data" / "train.parquet"; OUT = ART / "data" / "train_plus.parquet"
d = np.load(ART / "newpf_bank_train.npz", allow_pickle=True)
cols = [k for k in d.files if k != "id"]
pfin = pq.ParquetFile(SRC)
assert pfin.metadata.num_rows == len(d[cols[0]]), "行数不一致"
print(f"[add bank] {SRC.name} + {cols} -> {OUT.name}", flush=True)
arrs = {c: d[c].astype(np.float32) for c in cols}
writer = None; off = 0
for rg in range(pfin.num_row_groups):
    tbl = pfin.read_row_group(rg); n = tbl.num_rows
    for c in cols:
        tbl = tbl.append_column(c, pa.array(arrs[c][off:off + n]))
    off += n
    if writer is None:
        writer = pq.ParquetWriter(OUT, tbl.schema)
    writer.write_table(tbl)
writer.close()
print(f"saved {OUT}  rows={off} (+{len(cols)}列)")
