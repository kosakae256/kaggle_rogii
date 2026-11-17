# -*- coding: utf-8 -*-
"""全井 per-row 物理特徴を v90 parquet 行順に整合して生成→v91_physfeats.npz。
   B(v90統合)にもA(専用ゲート)にも使える。特徴は真値非依存・test再現可能。
   列: phys_snap_res(r0snap-lk), phys_gap(datum両義幅), phys_spread(候補広がり),
       phys_gr_resid(物理位置の局所GR残差), phys_topk_std(上位GR候補の位置分散), phys_move(r0からの移動)"""
import os,sys,io,pickle,time,glob
try: sys.stdout=io.open(1,"w",encoding="utf-8",closefd=False)
except: pass
import numpy as np,pandas as pd,pyarrow.parquet as pq
from pathlib import Path
P=Path(r"C:\Users\kosaka256\Documents\rogii_claude");TR=P/"rogii-wellbore-geology-prediction"/"train"
AV=P/"v89"/"artifacts_v89";A90=P/"v90"/"artifacts_v90"
N=5000;K=4;SLOPE=55/1000.;DEV=25/1000.;DIPSP=20/1000.;CONT=4.0
rng=np.random.default_rng(0)
def gen(md,z,tl,zl,tmn,tmx,pmd,pch):
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
# parquet 行順(well, last_known_tvt)
tb=pq.read_table(A90/"data"/"train.parquet",columns=["well","last_known_tvt"])
well=tb.column("well").to_numpy(zero_copy_only=False)
lk=tb.column("last_known_tvt").to_numpy(zero_copy_only=False).astype(float)
Ntot=len(well)
FEATS=["phys_snap_res","phys_gap","phys_spread","phys_gr_resid","phys_topk_std","phys_move"]
OUT={k:np.zeros(Ntot,np.float32) for k in FEATS}
ibw={}
for i,w in enumerate(well): ibw.setdefault(w,[]).append(i)
for w in ibw: ibw[w]=np.array(ibw[w])
PF=dict(np.load(AV/"pf_center_v89.npz"));t0=time.time();c=0;miss=0
for f in sorted(glob.glob(str(TR/"*__horizontal_well.csv"))):
    wn=Path(f).name.split("__")[0]
    if wn not in ibw: continue
    idx=ibw[wn]
    hw=pd.read_csv(f)
    try: tw=pd.read_csv(TR/f"{wn}__typewell.csv")
    except: continue
    kn=hw["TVT_input"].notna().to_numpy()
    if kn.sum()<10 or (~kn).sum()<30:
        miss+=1;continue
    hps=int(np.where(kn)[0].max());ev=slice(hps+1,len(hw))
    z=hw["Z"].to_numpy(float);md=hw["MD"].to_numpy(float)
    ti=hw["TVT_input"].to_numpy(float);gr=hw["GR"].to_numpy(float)
    tt=tw["TVT"].to_numpy(float);tg=tw["GR"].to_numpy(float);m=np.isfinite(tt)&np.isfinite(tg);tt,tg=tt[m],tg[m]
    if len(tt)<8: miss+=1;continue
    o=np.argsort(tt);tt,tg=tt[o],tg[o];tl=float(ti[hps]);zl=float(z[hps])
    ki=np.where(kn)[0];ki=ki[ki>=max(0,hps-80)]
    cand=gen(md[ev],z[ev],tl,zl,float(tt.min()),float(tt.max()),md[ki]-md[hps],(ti[ki]-tl)+(z[ki]-zl))
    if cand is None: miss+=1;continue
    L=cand.shape[1];obs=pd.Series(gr[ev]).interpolate(limit_direction="both").fillna(float(np.nanmean(tg))).rolling(5,center=True,min_periods=1).mean().to_numpy()
    gp=np.interp(cand,tt,tg);gmse=np.mean((gp-obs[None,:])**2,axis=1)
    r0=np.asarray(PF[wn],float)[:L];r0d=np.sqrt(np.mean((cand-r0[None,:])**2,axis=1))
    snap=cand[int(np.argmin(r0d))]
    topg=np.argsort(gmse)[:max(20,N//20)]
    # per-row特徴
    spread=cand.std(axis=0)                     # 候補群の行ごと広がり
    topk_std=cand[topg].std(axis=0)             # GR上位候補の行ごと位置分散=datum両義
    grr=np.abs(np.interp(snap,tt,tg)-obs)       # snap位置の局所GR残差
    n=min(L,len(idx))
    OUT["phys_snap_res"][idx[:n]]=(snap[:n]-tl).astype(np.float32)
    OUT["phys_gap"][idx[:n]]=topk_std[:n].astype(np.float32)
    OUT["phys_spread"][idx[:n]]=spread[:n].astype(np.float32)
    OUT["phys_gr_resid"][idx[:n]]=grr[:n].astype(np.float32)
    OUT["phys_topk_std"][idx[:n]]=topk_std[:n].astype(np.float32)
    OUT["phys_move"][idx[:n]]=np.abs(snap[:n]-r0[:n]).astype(np.float32)
    c+=1
    if c%150==0: print(f"  {c} ({time.time()-t0:.0f}s)",flush=True)
np.savez_compressed(P/"v91"/"physfeats_v91.npz",**OUT,well=well)
print(f"saved physfeats_v91.npz  {c}井 miss={miss}  rows={Ntot}")
