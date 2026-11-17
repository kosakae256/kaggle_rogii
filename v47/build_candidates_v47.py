# -*- coding: utf-8 -*-
"""
v47 候補データ準備(lag16・7バンク単一parquet版)。
create_candidates_v47.py(7バンク=pf_ancc/self/nbr_1..7)の train.parquet から 21候補を統合し、
GR整合コスト・struct を付けて v47/artifacts_v47/candidates_v47.npz を作る。学習/推論は本artifactのみ参照。
実行: 先に create_candidates_v47.py(train) → 本スクリプト。
"""
import json, time, gc, pickle
from pathlib import Path
import numpy as np, pandas as pd
import pyarrow.parquet as pq
from scipy.ndimage import uniform_filter1d

PROJECT = Path(r"C:\Users\kosaka256\Documents\rogii_claude")
ART = PROJECT / "v47" / "artifacts_v47"; ART.mkdir(parents=True, exist_ok=True)
PQ  = ART / "data" / "train.parquet"                     # ← create_candidates_v47.py の出力
assert PQ.exists(), "先に python v47/create_candidates_v47.py(SPLIT=train)を実行してください"
FLAVORS = [("tw","pf_ancc"),("self","pf_self"),("nbr","pf_nbr")]
NBANK = 7
CTX_COLS = ["md_since","pf_nbr_has","pf_nbr_nndist"]
CAND_NAMES = [f"{fl}_b{b+1}" for b in range(NBANK) for (fl,_) in FLAVORS]   # 順序は不変(モードHMMはソート)
NC = len(CAND_NAMES)

t0=time.time()
base = pq.read_table(PQ, columns=["id","well","target","last_known_tvt"]+CTX_COLS)
ids = base.column("id").to_numpy(zero_copy_only=False).astype(str)
gname = base.column("well").to_pandas().to_numpy().astype(str)
y  = base.column("target").to_numpy(zero_copy_only=False).astype(np.float32)
lk = base.column("last_known_tvt").to_numpy(zero_copy_only=False).astype(np.float64)
mds = base.column("md_since").to_numpy(zero_copy_only=False).astype(np.float32)
CTX = np.stack([base.column(c).to_numpy(zero_copy_only=False).astype(np.float32) for c in CTX_COLS],1)
N = len(y)
DEL = np.empty((N,NC),np.float32); STD = np.empty((N,NC),np.float32); col=0
for b in range(NBANK):
    suf=f"_{b+1}"
    vt = pq.read_table(PQ, columns=[f"{pre}{suf}" for (_,pre) in FLAVORS]+[f"{pre}_std{suf}" for (_,pre) in FLAVORS])
    for (fl,pre) in FLAVORS:
        v = vt.column(f"{pre}{suf}").to_numpy(zero_copy_only=False).astype(np.float64)
        sdv = vt.column(f"{pre}_std{suf}").to_numpy(zero_copy_only=False).astype(np.float32)
        DEL[:,col]=(v-lk).astype(np.float32); STD[:,col]=np.nan_to_num(sdv,nan=50.0); col+=1
    del vt; gc.collect()
bad=~np.isfinite(DEL); DEL[bad]=0.0; STD[bad]=1e3
print(f"[v47build] N={N} 候補={NC}(7バンク×tw/self/nbr) ({time.time()-t0:.0f}s)")

# ---- GR整合コスト(公式CSV)+ struct(既存pkl) ----
TRAIN=PROJECT/"rogii-wellbore-geology-prediction"/"train"
STRUCT=pickle.loads((PROJECT/"v38"/"artifacts_v38"/"struct_prior_train.pkl").read_bytes())
W_ROLL,W_ROLL2=21,81
rownum=pd.Series(ids).str.rsplit("_",n=1,expand=True)[1].astype(int).to_numpy()
wells,wellcode=np.unique(gname,return_inverse=True)
GRC=np.zeros((N,NC),np.float32); GRC2=np.zeros((N,NC),np.float32); GRM=np.zeros((N,NC),np.float32)
STRD=np.zeros(N,np.float32); STRS=np.full(N,60.0,np.float32); t1=time.time()
for wi,wn in enumerate(wells):
    ri=np.where(wellcode==wi)[0]
    if len(ri)==0: continue
    o=np.argsort(rownum[ri]); ri=ri[o]; rn=rownum[ri]
    try:
        hw=pd.read_csv(TRAIN/f"{wn}__horizontal_well.csv",usecols=["GR"]); tw=pd.read_csv(TRAIN/f"{wn}__typewell.csv",usecols=["TVT","GR"]).sort_values("TVT")
    except Exception: continue
    ga=hw["GR"].astype(float).interpolate(limit_direction="both").to_numpy(); fb=float(np.nanmean(tw["GR"])) if np.isfinite(np.nanmean(tw["GR"])) else 0.0
    ga=np.where(np.isfinite(ga),ga,fb); g=ga[np.clip(rn,0,len(ga)-1)]
    ttv=tw["TVT"].to_numpy(float); tgr=tw["GR"].to_numpy(float)
    tvtk=lk[ri][:,None]+DEL[ri]; expg=np.interp(tvtk.ravel(),ttv,tgr).reshape(tvtk.shape); mis=g[:,None]-expg; amis=np.abs(mis)
    GRC[ri]=uniform_filter1d(amis,W_ROLL,axis=0,mode="nearest").astype(np.float32)
    GRC2[ri]=uniform_filter1d(amis,W_ROLL2,axis=0,mode="nearest").astype(np.float32)
    GRM[ri]=uniform_filter1d(mis,W_ROLL,axis=0,mode="nearest").astype(np.float32)
    sp=STRUCT.get(wn)
    if sp is not None and len(sp["tvt"])==len(ri):
        STRD[ri]=(np.asarray(sp["tvt"],float)-lk[ri]).astype(np.float32); STRS[ri]=np.asarray(sp["sig"],float).astype(np.float32)
    if (wi+1)%150==0: print(f"  GR/struct {wi+1}/{len(wells)} ({time.time()-t1:.0f}s)",flush=True)
print(f"[v47build] GRコスト/struct 完了 ({time.time()-t1:.0f}s)")

np.savez_compressed(ART/"candidates_v47.npz", DEL=DEL,STD=STD,y=y,lk=lk.astype(np.float32),mds=mds,ctx=CTX,
                    wellcode=wellcode.astype(np.int32),GRC=GRC,GRC2=GRC2,GRM=GRM,STRD=STRD,STRS=STRS)
np.save(ART/"ids_v47.npy",ids)
(ART/"candidates_v47_meta.json").write_text(json.dumps(dict(n_rows=int(N),cand_names=CAND_NAMES,ctx_cols=CTX_COLS,
    n_wells=int(len(wells)),wells=wells.tolist(),lag=16,note="lag16・7バンク。delta=絶対TVT-lk。学習/推論は本artifactのみ。"),ensure_ascii=False),encoding="utf-8")
orc=float(np.sqrt(np.mean(np.min(np.abs(DEL-y[:,None]),1)**2)))
print(f"[v47build] saved. mean候補={np.sqrt(np.mean((DEL.mean(1)-y)**2)):.4f}  ORACLE={orc:.4f}  ({time.time()-t0:.0f}s)")
