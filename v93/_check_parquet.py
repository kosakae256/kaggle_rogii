# -*- coding: utf-8 -*-
import io,sys
sys.stdout=io.open(1,"w",encoding="utf-8",closefd=False)
import numpy as np, pyarrow.parquet as pq, pyarrow.compute as pc
from pathlib import Path
p=Path("v93/artifacts_v93/data/train.parquet")
pf=pq.ParquetFile(p); sc=pf.schema_arrow; N=pf.metadata.num_rows
numcols=[f.name for f in sc if str(f.type) in ("float","double","float32","float64","int32","int64","int16") ]
bad=[]; allnan=[]
for c in numcols:
    col=pq.read_table(p,columns=[c]).column(0)
    arr=col.to_numpy(zero_copy_only=False).astype(np.float64)
    fin=np.isfinite(arr); nf=int((~fin).sum())
    if nf>0:
        frac=fin.mean()
        bad.append((c,nf,frac))
        if frac==0: allnan.append(c)
print(f"数値列={len(numcols)}  行={N}")
print(f"非有限を含む列={len(bad)}  全NaN列={len(allnan)}")
if allnan: print("  ★全NaN列:", allnan[:20])
# 非有限率が高い(>1%)列
hi=[(c,nf,fr) for c,nf,fr in bad if fr<0.99]
print(f"非有限率>1%の列={len(hi)}")
for c,nf,fr in sorted(hi,key=lambda x:x[2])[:20]:
    print(f"    {c}: 非有限{nf}行 (有限率{fr*100:.2f}%)")
# 低率(<1%)の代表
lo=[(c,nf) for c,nf,fr in bad if fr>=0.99]
print(f"非有限率<1%(軽微)の列={len(lo)} 例:", [c for c,_ in lo[:8]])
