# -*- coding: utf-8 -*-
"""v96全特徴 + v97の値相違PF列(v97_接頭辞)を結合した train_combo.parquet を生成。
id順が完全一致なので行バッチで hstack。メモリ節約のため ParquetWriter で逐次書込。
出力: v97/artifacts_v97/data/train_combo.parquet"""
import sys, io, json, time
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
import pyarrow as pa, pyarrow.parquet as pq
from pathlib import Path

P96 = Path("v95/artifacts_v95b/data/train.parquet")
P97 = Path("v97/artifacts_v97/data/train.parquet")
OUT = Path("v97/artifacts_v97/data/train_combo.parquet")
diff = json.load(open("v97/artifacts_v97/v97_diff_cols.json"))   # v97の値相違427列
print(f"[combo] v96全列 + v97差分{len(diff)}列(v97_) を結合", flush=True)

pf96 = pq.ParquetFile(P96); pf97 = pq.ParquetFile(P97)
n96 = pf96.schema_arrow.names
BATCH = 400000; t0 = time.time(); writer = None; nrow = 0
it96 = pf96.iter_batches(batch_size=BATCH)
it97 = pf97.iter_batches(batch_size=BATCH, columns=diff)
for b96, b97 in zip(it96, it97):
    assert b96.num_rows == b97.num_rows, "行ズレ"
    cols = list(b96.columns); names = list(b96.schema.names)
    for j, c in enumerate(diff):                       # v97差分を v97_ 名で追加
        cols.append(b97.column(j)); names.append("v97_" + c)
    tbl = pa.Table.from_arrays(cols, names=names)
    if writer is None:
        writer = pq.ParquetWriter(OUT, tbl.schema, compression="zstd")
    writer.write_table(tbl); nrow += tbl.num_rows
    print(f"  {nrow} 行 ({time.time()-t0:.0f}s)", flush=True)
writer.close()
sch = pq.read_schema(OUT)
print(f"[combo] 完了: {OUT.name}  行={nrow} 列={len(sch.names)} ({time.time()-t0:.0f}s)", flush=True)
