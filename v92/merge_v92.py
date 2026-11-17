# -*- coding: utf-8 -*-
"""v92 特徴マージ(SPLIT対応)。
   train: anchored parquet(pf_*→pfg_*) + v69_4 parquet(no-anchor pf_*) → train_final。
   test : anchored parquet(pf_*→pfg_*) + noanchor parquet(v92 V92_ANCHOR=off, pf_*)   → test_final。
   env: MERGE_SPLIT(train/test) / MERGE_ANCHORED / MERGE_NOANCHOR / MERGE_OUT を上書き可。"""
import io,sys,os;sys.stdout=io.open(1,"w",encoding="utf-8",closefd=False)
import numpy as np,pandas as pd,pyarrow.parquet as pq,pyarrow as pa
from pathlib import Path
P=Path(os.environ.get("V92_ROOT", r"C:\Users\kosaka256\Documents\rogii_claude"))
SPLIT=os.environ.get("MERGE_SPLIT","train")
AV=P/"v92"/"artifacts_v92"/"data"
if SPLIT=="train":
    ANCHORED=os.environ.get("MERGE_ANCHORED", str(AV/"train.parquet"))            # v92 create(grfree)出力
    NOANCHOR=os.environ.get("MERGE_NOANCHOR", str(P/"v69"/"artifacts_v69_4"/"data"/"train.parquet"))  # v69_4流用
    OUT=os.environ.get("MERGE_OUT", str(AV/"train_final.parquet"))
else:
    ANCHORED=os.environ.get("MERGE_ANCHORED", str(AV/"test_anchored.parquet"))     # v92 create test(grfree)
    NOANCHOR=os.environ.get("MERGE_NOANCHOR", str(AV/"test_noanchor.parquet"))      # v92 create test(off)
    OUT=os.environ.get("MERGE_OUT", str(AV/"test_final.parquet"))
print(f"[merge {SPLIT}] anchored={ANCHORED}\n  noanchor={NOANCHOR}",flush=True)
a=pq.read_table(ANCHORED).to_pandas()
print(f"  anchored: {a.shape}",flush=True)
sc=pq.read_schema(NOANCHOR); pf_cols=[c for c in sc.names if c.startswith("pf_")]
b=pq.read_table(NOANCHOR,columns=(["well"] if "well" in sc.names else [])+pf_cols).to_pandas()
assert len(a)==len(b), f"行数不一致 anchored={len(a)} noanchor={len(b)}"
if "well" in a.columns and "well" in b.columns:
    assert (a["well"].to_numpy()==b["well"].to_numpy()).all(), "well順不一致"
    print("  ★行・well整合 OK",flush=True)
# anchored の pf_* → pfg_*
ren={c:("pfg_"+c[3:]) for c in a.columns if c.startswith("pf_")}
a=a.rename(columns=ren)
# no-anchor pf_* を合流
for c in pf_cols: a[c]=b[c].to_numpy()
print(f"  合流後: {a.shape}  (base + pf_*錨なし{len(pf_cols)} + pfg_*錨あり{len(ren)})",flush=True)
Path(OUT).parent.mkdir(parents=True,exist_ok=True)
pq.write_table(pa.Table.from_pandas(a,preserve_index=False),OUT,compression="zstd")
print(f"saved {OUT}  rows={len(a)} cols={a.shape[1]}",flush=True)
