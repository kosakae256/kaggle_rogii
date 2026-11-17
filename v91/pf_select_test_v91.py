# -*- coding: utf-8 -*-
"""物理候補を PF流の尤度で選択できるか: global MSE vs 局所相関 vs 局所ベストシフト残差。155井fold0。"""
import os,sys,io,pickle,time
try: sys.stdout=io.open(1,"w",encoding="utf-8",closefd=False)
except: pass
import numpy as np,pandas as pd
from pathlib import Path
P=Path(r"C:\Users\kosaka256\Documents\rogii_claude");TR=P/"rogii-wellbore-geology-prediction"/"train"
A=P/"v64"/"artifacts_v64";AV=P/"v89"/"artifacts_v89"
N=5000;K=4;SLOPE=55/1000.;DEV=25/1000.;DIPSP=20/1000.;CONT=4.0;rng=np.random.default_rng(0)
def gen(md,z,tl,zl,tmn,tmx,pmd,pch):
    L=len(md);mr=md-md[0];dZ=z-zl
    pdip=float(np.clip(np.polyfit(pmd,pch,1)[0],-SLOPE,SLOPE)) if(len(pmd)>=5 and(pmd[-1]-pmd[0])>1) else 0.0
    xk=np.linspace(0.,mr[-1]+1e-6,K);seg=np.diff(xk)
    dn=pdip+rng.uniform(-DIPSP,DIPSP,size=(N,1))
    dyk=np.concatenate([np.zeros((N,1)),np.cumsum(rng.normal(0,DEV,(N,K-1))*seg[None,:],axis=1)],axis=1)
    dev=np.empty((N,L))
    for i in range(N):dev[i]=np.interp(mr,xk,dyk[i])
    cand=tl+dn*mr[None,:]+dev-dZ[None,:]
    ok=((cand>=tmn-5)&(cand<=tmx+5)).all(axis=1);nc=min(30,L);ok&=(np.abs(cand[:,:nc]-tl)<CONT).all(axis=1)
    return cand[ok] if ok.any() else None
DIST=dict(np.load(A/"v89_dist_f0_dist.npz"));EVD=pickle.loads(open(AV/"learned_boot.pkl","rb").read());PF=dict(np.load(AV/"pf_center_v89.npz"))
def wcorr(gp,obs,W=61):
    # 局所窓Pearson相関の平均(大=良). gp:(Nc,L) obs:(L,)
    Nc,L=gp.shape;s=max(1,L//40);idxs=range(0,L-W,s);acc=np.zeros(Nc);cnt=0
    o=obs
    for a in idxs:
        og=o[a:a+W];og=og-og.mean()
        gg=gp[:,a:a+W];gg=gg-gg.mean(axis=1,keepdims=True)
        num=(gg*og[None,:]).sum(1);den=np.sqrt((gg**2).sum(1)*(og**2).sum()+1e-9)
        acc+=num/den;cnt+=1
    return acc/max(cnt,1)
eo=[];em=[];ec=[];eu=[];er0=[]
for wn in [w for w in DIST if w in EVD and w in PF]:
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
    true=tvt[ev];fin=np.isfinite(true);L=len(true)
    obs=pd.Series(gr[ev]).interpolate(limit_direction="both").fillna(float(np.nanmean(tg))).rolling(5,center=True,min_periods=1).mean().to_numpy()
    gp=np.interp(cand,tt,tg)
    gmse=np.mean((gp-obs[None,:])**2,axis=1)                    # global MSE
    wc=wcorr(gp,obs)                                            # 局所相関(PF流)
    r0=np.asarray(PF[wn],float)[:L]
    eo.append((cand[int(np.argmin(np.sqrt(np.mean((cand-true[None,:])**2,axis=1))))]-true)[fin])
    em.append((cand[int(np.argmin(gmse))]-true)[fin])
    ec.append((cand[int(np.argmax(wc))]-true)[fin])
    er0.append((r0-true)[fin])
r=lambda L:float(np.sqrt(np.mean(np.concatenate(L)**2)))
print(f"[PF流選択テスト 155井]")
print(f"  oracle              = {r(eo):.3f}")
print(f"  素PF r0             = {r(er0):.3f}")
print(f"  global MSE 選択     = {r(em):.3f}  (既存20ft級)")
print(f"  局所相関 選択(PF流) = {r(ec):.3f}  ← PFの局所照合で20ftを破れるか")
