# -*- coding: utf-8 -*-
"""区間ゲート premise: 学習emission(U-Net)で v90 vs 物理 を区間判別。生GR(7.62)を超えるか。155井。"""
import os,sys,io,pickle,glob
try: sys.stdout=io.open(1,"w",encoding="utf-8",closefd=False)
except: pass
import numpy as np,pandas as pd,pyarrow.parquet as pq
from pathlib import Path
P=Path(r"C:\Users\kosaka256\Documents\rogii_claude");TR=P/"rogii-wellbore-geology-prediction"/"train"
A=P/"v64"/"artifacts_v64";AV=P/"v89"/"artifacts_v89";A90=P/"v90"/"artifacts_v90"
N=5000;K=4;SLOPE=55/1000.;DEV=25/1000.;DIPSP=20/1000.;CONT=4.0;W_NN=0.62;NB=90;ND=181;STEP=0.5;SEG=int(os.environ.get("SEG","150"))
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
tb=pq.read_table(A90/"data"/"train.parquet",columns=["well","target"])
well=tb.column("well").to_numpy(zero_copy_only=False);y=tb.column("target").to_numpy(zero_copy_only=False).astype(float)
gbm=np.load(A90/"gbm_v41_oof.npy").astype(float)[:len(y)];nnz=np.load(A90/"nn_oof_combo3.npz")
nn=np.mean([nnz[k] for k in nnz.files],axis=0)[:len(y)];v90err=(1-W_NN)*gbm+W_NN*nn-y
ibw={}
for i,w in enumerate(well): ibw.setdefault(w,[]).append(i)
for w in ibw: ibw[w]=np.array(ibw[w])
DIST=dict(np.load(A/"v89_dist_f0_dist.npz"));EVD=pickle.loads(open(AV/"learned_boot.pkl","rb").read());PF=dict(np.load(AV/"pf_center_v89.npz"))
def ulp(traj,st,bid,Pd):  # 各行のU-Net emission log尤度
    Ln=min(len(traj),len(st),len(bid));d=traj[:Ln]-st[:Ln];j=np.round(d/STEP+NB)
    jf=np.clip(j.astype(int),0,ND-1);lp=np.where((j>=0)&(j<ND),np.log(Pd[bid[:Ln],jf]+1e-6),-12.)
    return np.concatenate([lp,np.full(len(traj)-Ln,-12.)])
Eb=[];Eg=[];Eo=[];Egr=[]
for wn in [w for w in DIST if w in EVD and w in PF and w in ibw]:
    try:hw=pd.read_csv(TR/f"{wn}__horizontal_well.csv");tw=pd.read_csv(TR/f"{wn}__typewell.csv")
    except:continue
    kn=hw["TVT_input"].notna().to_numpy()
    if kn.sum()<10 or (~kn).sum()<30:continue
    hps=int(np.where(kn)[0].max());ev=slice(hps+1,len(hw))
    z=hw["Z"].to_numpy(float);md=hw["MD"].to_numpy(float);tvt=hw["TVT"].to_numpy(float)
    ti=hw["TVT_input"].to_numpy(float);gr=hw["GR"].to_numpy(float)
    tt=tw["TVT"].to_numpy(float);tg=tw["GR"].to_numpy(float);m=np.isfinite(tt)&np.isfinite(tg);tt,tg=tt[m],tg[m]
    if len(tt)<8:continue
    o=np.argsort(tt);tt,tg=tt[o],tg[o];tl=float(ti[hps]);zl=float(z[hps])
    ki=np.where(kn)[0];ki=ki[ki>=max(0,hps-80)]
    cand=gen(md[ev],z[ev],tl,zl,float(tt.min()),float(tt.max()),md[ki]-md[hps],(ti[ki]-tl)+(z[ki]-zl))
    if cand is None:continue
    true=tvt[ev];L=len(true);v90e=v90err[ibw[wn]];n=min(L,len(v90e))
    true=true[:n];cand=cand[:,:n];v90_tvt=true+v90e[:n]
    r0=np.asarray(PF[wn],float)[:n];phys=cand[int(np.argmin(np.sqrt(np.mean((cand-r0[None,:])**2,axis=1))))]
    st=np.asarray(EVD[wn]["st"],float);bid=np.asarray(EVD[wn]["bid"],int);Pd=np.asarray(DIST[wn],float)
    lpv=ulp(v90_tvt,st,bid,Pd)[:n];lpp=ulp(phys,st,bid,Pd)[:n]
    obs=pd.Series(gr[ev][:n]).interpolate(limit_direction="both").fillna(float(np.nanmean(tg))).rolling(5,center=True,min_periods=1).mean().to_numpy()
    grv=np.abs(np.interp(v90_tvt,tt,tg)-obs);grp=np.abs(np.interp(phys,tt,tg)-obs)
    fin=np.isfinite(true);out=v90_tvt.copy();outg=v90_tvt.copy();orc=v90_tvt.copy()
    for a in range(0,n,SEG):
        b=min(a+SEG,n);sl=slice(a,b)
        if np.nanmean(lpp[sl])>np.nanmean(lpv[sl]): out[sl]=phys[sl]          # U-Net区間ゲート
        if np.nanmean(grp[sl])<np.nanmean(grv[sl]): outg[sl]=phys[sl]         # 生GR区間ゲート
        if np.nanmean(np.abs(phys[sl]-true[sl]))<np.nanmean(np.abs(v90_tvt[sl]-true[sl])): orc[sl]=phys[sl]
    Eb.append((v90_tvt-true)[fin]);Eg.append((out-true)[fin]);Egr.append((outg-true)[fin]);Eo.append((orc-true)[fin])
r=lambda L:float(np.sqrt(np.mean(np.concatenate(L)**2)))
print(f"[U-Net区間ゲート SEG={SEG} {len(Eb)}井 fold0]")
print(f"  v90 base           = {r(Eb):.4f}")
print(f"  生GR区間ゲート       = {r(Egr):.4f}")
print(f"  ★U-Net区間ゲート    = {r(Eg):.4f}  ← 学習証拠なら当てられるか")
print(f"  区間oracle          = {r(Eo):.4f}")
