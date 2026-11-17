# -*- coding: utf-8 -*-
"""v90 + snapms(GRベース物理) + v74(GRフリー) 3脚stack。別系統で相互脱相関するか。v74行に整合しfield-out。"""
import sys,io,pickle,glob,time
try: sys.stdout=io.open(1,"w",encoding="utf-8",closefd=False)
except: pass
import numpy as np,pandas as pd,pyarrow.parquet as pq
from pathlib import Path
from sklearn.cluster import KMeans
P=Path(r"C:\Users\kosaka256\Documents\rogii_claude");TR=P/"rogii-wellbore-geology-prediction"/"train"
A74=P/"v74"/"artifacts_v74";A90=P/"v90"/"artifacts_v90"
N=5000;K=4;SLOPE=55/1000.;DEV=25/1000.;DIPSP=20/1000.;CONT=4.0;W_NN=0.62;NSEED=3
def gen(md,z,tl,zl,tmn,tmx,pmd,pch,rng):
    L=len(md);mr=md-md[0];dZ=z-zl
    pdip=float(np.clip(np.polyfit(pmd,pch,1)[0],-SLOPE,SLOPE)) if(len(pmd)>=5 and(pmd[-1]-pmd[0])>1) else 0.0
    xk=np.linspace(0.,mr[-1]+1e-6,K);seg=np.diff(xk)
    dn=pdip+rng.uniform(-DIPSP,DIPSP,size=(N,1))
    dyk=np.concatenate([np.zeros((N,1)),np.cumsum(rng.normal(0,DEV,(N,K-1))*seg[None,:],axis=1)],axis=1)
    dev=np.empty((N,L))
    for i in range(N):dev[i]=np.interp(mr,xk,dyk[i])
    cand=tl+dn*mr[None,:]+dev-dZ[None,:]
    ok=((cand>=tmn-5)&(cand<=tmx+5)).all(axis=1);ok&=(np.abs(cand[:,:min(30,L)]-tl)<CONT).all(axis=1)
    return cand[ok] if ok.any() else None
tb=pq.read_table(A90/"data"/"train.parquet",columns=["well","target"])
well=tb.column("well").to_numpy(zero_copy_only=False);y=tb.column("target").to_numpy(zero_copy_only=False).astype(float)
gbm=np.load(A90/"gbm_v41_oof.npy").astype(float)[:len(y)];nnz=np.load(A90/"nn_oof_combo3.npz")
nn=np.mean([nnz[k] for k in nnz.files],axis=0)[:len(y)];v90err=(1-W_NN)*gbm+W_NN*nn-y
ibw={}
for i,w in enumerate(well): ibw.setdefault(w,[]).append(i)
for w in ibw: ibw[w]=np.array(ibw[w])
PF=dict(np.load(P/"v89"/"artifacts_v89"/"pf_center_v89.npz"))
import os
V74F=os.environ.get("V74F","phaseI_oof_fieldout_pre.npz")
z74=np.load(A74/V74F,allow_pickle=True)
rows=[];t0=time.time();c=0
for f in sorted(glob.glob(str(TR/"*__horizontal_well.csv"))):
    wn=Path(f).name.split("__")[0]
    if wn not in PF or wn not in ibw or f"{wn}__err" not in z74.files: continue
    hw=pd.read_csv(f,usecols=["MD","Z","TVT","TVT_input","GR","X","Y"])
    try: tw=pd.read_csv(TR/f"{wn}__typewell.csv",usecols=["TVT","GR"])
    except: continue
    kn=hw["TVT_input"].notna().to_numpy()
    if kn.sum()<10 or (~kn).sum()<30: continue
    hps=int(np.where(kn)[0].max());ev=slice(hps+1,len(hw))
    z=hw["Z"].to_numpy(float);md=hw["MD"].to_numpy(float);tvt=hw["TVT"].to_numpy(float);ti=hw["TVT_input"].to_numpy(float)
    tt=tw["TVT"].to_numpy(float);m=np.isfinite(tt);tt=tt[m]
    if len(tt)<8: continue
    tl=float(ti[hps]);zl=float(z[hps]);ki=np.where(kn)[0];ki=ki[ki>=max(0,hps-80)]
    snaps=[]
    for s in range(NSEED):
        cd=gen(md[ev],z[ev],tl,zl,float(tt.min()),float(tt.max()),md[ki]-md[hps],(ti[ki]-tl)+(z[ki]-zl),np.random.default_rng(s))
        if cd is None: snaps=None;break
        r0=np.asarray(PF[wn],float)[:cd.shape[1]];snaps.append(cd[int(np.argmin(np.sqrt(np.mean((cd-r0[None,:])**2,axis=1))))])
    if snaps is None: continue
    L=len(snaps[0]);snapms=np.mean([s[:L] for s in snaps],0)
    true=tvt[ev][:L]
    v90e=v90err[ibw[wn]]
    e74=np.asarray(z74[f"{wn}__err"],float);rw=np.asarray(z74[f"{wn}__rows"]).astype(int);j=rw-(hps+1)
    ok=(j>=0)&(j<L)&(j<len(v90e))&np.isfinite(e74)&np.isfinite(true[np.clip(j,0,L-1)])
    j=j[ok];e74=e74[ok]
    if len(j)<5: continue
    rows.append(dict(v90=v90e[j],snap=(snapms[j]-true[j]),v74=e74,
                     cx=float(hw["X"].mean()),cy=float(hw["Y"].mean())))
    c+=1
    if c%200==0: print(f"  {c} ({time.time()-t0:.0f}s)",flush=True)
pickle.dump(rows,open(P/"v91"/f"stack_{V74F[:-4]}.pkl","wb"))
v90=np.concatenate([d["v90"] for d in rows]);snap=np.concatenate([d["snap"] for d in rows]);v74=np.concatenate([d["v74"] for d in rows])
r=lambda e:float(np.sqrt(np.mean(e**2)));base=r(v90)
cx=np.array([d["cx"] for d in rows]);cy=np.array([d["cy"] for d in rows])
print(f"\n[3脚stack {len(rows)}井] base={base:.4f}")
print(f"  ρ(v90,snap)={np.corrcoef(v90,snap)[0,1]:.3f} ρ(v90,v74)={np.corrcoef(v90,v74)[0,1]:.3f} ρ(snap,v74)={np.corrcoef(snap,v74)[0,1]:.3f}")
fb=KMeans(n_clusters=8,n_init=5,random_state=0).fit_predict(np.column_stack([cx,cy]))
gv=np.array([fb[i] for i,d in enumerate(rows) for _ in d["v90"]])  # 行ごとブロック
def fo_blend(legs):  # legs: list of arrays. field-out最適重み
    err=[]
    for b in np.unique(fb):
        trm=gv!=b;vam=gv==b
        from itertools import product
        best=(1e9,None)
        for ws in product(np.arange(0,0.26,0.05),repeat=len(legs)):
            if sum(ws)>0.5: continue
            e=(1-sum(ws))*v90[trm]+sum(w*legs[i][trm] for i,w in enumerate(ws))
            if r(e)<best[0]: best=(r(e),ws)
        ws=best[1];err.append((1-sum(ws))*v90[vam]+sum(w*legs[i][vam] for i,w in enumerate(ws)))
    return r(np.concatenate(err))
print(f"  v90+snap        field-out={fo_blend([snap]):.4f}")
print(f"  v90+v74         field-out={fo_blend([v74]):.4f}")
print(f"  ★v90+snap+v74   field-out={fo_blend([snap,v74]):.4f}")
