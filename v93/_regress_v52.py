# -*- coding: utf-8 -*-
"""回帰用: 旧v52 create を6井で回し df を temp parquet に保存(GUARD/mainを避け build_dataset_v28 直呼び)。"""
import os, sys, io
sys.stdout=io.open(1,"w",encoding="utf-8",closefd=False)
os.environ.setdefault("NWELLS","6")
SCR=r"C:\Users\KOSAKA~1\AppData\Local\Temp\claude\c--Users-kosaka256-Documents-rogii-claude\1c67796e-d953-4c10-8874-a500ac704725\scratchpad"
sys.path.insert(0,"v52")
import create_train_parquet_v52 as c52
df=c52.build_dataset_v28()
out=os.path.join(SCR,"df_v52_6.parquet")
df.to_parquet(out,index=False)
print(f"[regress v52] saved {out} rows={len(df)} cols={df.shape[1]} wells={df['well'].nunique()}",flush=True)
