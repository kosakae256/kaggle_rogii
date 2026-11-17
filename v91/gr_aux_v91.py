# -*- coding: utf-8 -*-
"""物理ベース+GR補助: 物理ベース(r0)近傍の帯だけでGRを効かせ、遠方デコイを禁じる。
   GR照合を MSE / 順位相関(ノイズ頑健) で比較。U-Net不要。155井 fold0。"""
import os,sys,io,pickle
try: sys.stdout=io.open(1,"w",encoding="utf-8",closefd=False)
except: pass
import numpy as np,pandas as pd
from scipy.stats import rankdata
from pathlib import Path
P=Path(r"C:\Users\kosaka256\Documents\rogii_claude");TR=P/"rogii-wellbore-geology-prediction"/"train"
A=P/"v64"/"artifacts_v64";AV=P/"v89"/"artifacts_v89"
N=5000;K=4;SLOPE=55/1000.;DEV=25/1000.;DIPSP=20/1000.;CONT=4.0
DIST=dict(np.load(A/"v89_dist_f0_dist.npz"));PF=dict(np.load(AV/"pf_center_v89.npz"))
WELLS=[w for w in DIST if w in PF]
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
def collect():
    rng=np.random.default_rng(0);out=[]
    for wn in WELLS:
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
        cand=gen(md[ev],z[ev],tl,zl,float(tt.min()),float(tt.max()),md[ki]-md[hps],(ti[ki]-tl)+(z[ki]-zl),rng)
        if cand is None:continue
        true=tvt[ev];fin=np.isfinite(true);L=len(true)
        obs=pd.Series(gr[ev]).interpolate(limit_direction="both").fillna(float(np.nanmean(tg))).rolling(5,center=True,min_periods=1).mean().to_numpy()
        gp=np.interp(cand,tt,tg)
        gmse=np.mean((gp-obs[None,:])**2,axis=1)
        # 順位相関(ノイズ頑健): obsとgpをrank化しspearman的に(大=良). 局所窓平均。
        obr=rankdata(obs)
        gr_rank=np.array([np.corrcoef(rankdata(gp[i]),obr)[0,1] for i in range(len(cand))])
        r0=np.asarray(PF[wn],float)[:L]
        d0=np.sqrt(np.mean((cand-r0[None,:])**2,axis=1))
        out.append((cand,true,fin,gmse,gr_rank,d0,r0))
    return out
DATA=collect()
def rmse_of(sel):  # sel: list of chosen trajectories per well
    e=[np.concatenate([(sel[i]-DATA[i][1])[DATA[i][2]] for i in range(len(DATA))])]
    return float(np.sqrt(np.mean(np.concatenate(e)**2)))
# 基準
r0sel=[d[6][:len(d[1])] for d in DATA]
snap=[d[0][int(np.argmin(d[5]))] for d in DATA]
print(f"[物理ベース+GR補助 {len(DATA)}井]")
print(f"  素PF r0           = {rmse_of(r0sel):.3f}")
print(f"  r0スナップ(GR無)   = {rmse_of(snap):.3f}")
for BAND in [3,5,8,12,999]:
    # r0近傍BAND ft の帯内で GR-MSE最小
    mse_sel=[];rank_sel=[]
    for d in DATA:
        cand,_,_,gmse,grk,d0,_=d
        inb=d0<BAND
        if not inb.any(): inb=d0<=np.min(d0)+1e-9
        idx=np.where(inb)[0]
        mse_sel.append(cand[idx[int(np.argmin(gmse[idx]))]])
        rank_sel.append(cand[idx[int(np.argmax(grk[idx]))]])
    print(f"  BAND={BAND:>3}ft: 帯内GR-MSE={rmse_of(mse_sel):.3f}  帯内GR順位相関={rmse_of(rank_sel):.3f}")
