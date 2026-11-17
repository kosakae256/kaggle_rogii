# -*- coding: utf-8 -*-
"""train_combo に bpf_v96 の候補60列(bpf_{rep}_delta/std_1..6, 5表現×6バンク)を positional 追記。
行の並びは検証済み100%一致=join不要。行グループ単位ストリーミングでRAM安全(train_combo=21GB)。
出力: v100/artifacts_v100/data/train_combo_bwd.parquet。実行: .venv/Scripts/python.exe v101/merge_bwd_v101.py"""
import time
import pyarrow as pa
import pyarrow.parquet as pq

TC = "v100/artifacts_v100/data/train_combo.parquet"
BP = "v100/artifacts_v100/bpf_v96.parquet"
OUT = "v100/artifacts_v100/data/train_combo_bwd.parquet"
REPS = ["ancc", "self", "nbr", "nbr_gr5", "self_graft"]
DELTA = [f"bpf_{r}_delta_{k}" for r in REPS for k in range(1, 7)]   # 30
STD = [f"bpf_{r}_std_{k}" for r in REPS for k in range(1, 7)]       # 30
BPCOLS = DELTA + STD

t0 = time.time()
bpt = pq.read_table(BP, columns=BPCOLS)                # 3.78M × 60 float32 ~0.9GB
print(f"[merge] bpf {len(BPCOLS)}列 load {bpt.num_rows}行 ({time.time()-t0:.0f}s)", flush=True)
pf = pq.ParquetFile(TC)
writer = None; off = 0
for i, b in enumerate(pf.iter_batches(batch_size=200000)):
    tb = pa.Table.from_batches([b]); n = tb.num_rows
    sl = bpt.slice(off, n)
    for c in BPCOLS:
        tb = tb.append_column(c, sl.column(c))
    if writer is None:
        writer = pq.ParquetWriter(OUT, tb.schema)
    writer.write_table(tb); off += n
    print(f"  batch{i} rows={off} ({time.time()-t0:.0f}s)", flush=True)
writer.close()
m = pq.ParquetFile(OUT)
print(f"[merge] saved {OUT} rows={m.metadata.num_rows} cols={len(m.schema.names)} "
      f"(+{len(BPCOLS)}bpf) ({time.time()-t0:.0f}s)", flush=True)
