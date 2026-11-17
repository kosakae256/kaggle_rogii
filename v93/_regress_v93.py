import os,sys,io,shutil
sys.stdout=io.open(1,"w",encoding="utf-8",closefd=False)
os.environ["NWELLS"]="6"
SCR=r"C:\Users\KOSAKA~1\AppData\Local\Temp\claude\c--Users-kosaka256-Documents-rogii-claude\1c67796e-d953-4c10-8874-a500ac704725\scratchpad"
sys.path.insert(0,"v93")
import create_v93
df=create_v93.build_dataset()
out=os.path.join(SCR,"df_v93_6.parquet")
df.to_parquet(out,index=False)
print(f"[regress v93] saved {out} rows={len(df)} cols={df.shape[1]} wells={df['well'].nunique()}",flush=True)
