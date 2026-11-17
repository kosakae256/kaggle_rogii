# -*- coding: utf-8 -*-
"""backward_feats_anchor.parquet(bpf列, 生成済) と train.parquet を行同順マージして
   train_fwdbwd_anchor.parquet を作る(build_backward の section6 と同じロジックの単独版)。
   build_backward の PF 再計算(96分)を避け、disk full で落ちたマージだけをやり直す。"""
import io, sys
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
from pathlib import Path
import pyarrow.parquet as pq

ART = Path(r"C:\Users\kosaka256\Documents\rogii_claude") / "v93" / "artifacts_v93"
TRAIN = ART / "data" / "train.parquet"
BF = ART / "backward_feats_anchor.parquet"
OUT = ART / "data" / "train_fwdbwd_anchor.parquet"

bdf = pq.read_table(BF)
bpf_names = [c for c in bdf.column_names if c.startswith("bpf_")]
pfin = pq.ParquetFile(TRAIN)
assert bdf.num_rows == pfin.metadata.num_rows, f"行数不一致 bf={bdf.num_rows} train={pfin.metadata.num_rows}"
print(f"[merge] bpf列={len(bpf_names)} 行={bdf.num_rows} -> {OUT.name}", flush=True)
bcols = {nm: bdf.column(nm) for nm in bpf_names}
writer = None; off = 0
for rg in range(pfin.num_row_groups):
    tbl = pfin.read_row_group(rg); n = tbl.num_rows
    for nm in bpf_names:
        tbl = tbl.append_column(nm, bcols[nm].slice(off, n))
    off += n
    if writer is None:
        writer = pq.ParquetWriter(OUT, tbl.schema)
    writer.write_table(tbl)
    if rg % 20 == 0:
        print(f"  rg {rg+1}/{pfin.num_row_groups} off={off}", flush=True)
writer.close()
print(f"saved: {OUT} rows={off} (+{len(bpf_names)} bpf cols)")
