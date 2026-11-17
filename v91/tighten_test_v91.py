# -*- coding: utf-8 -*-
"""候補空間を物理拘束で狭める掃引: DIPSP/DEV/曲率拘束を締めて oracle と選択(GR/r0錨)がどう動くか。155井。"""
import os,sys,io,pickle
try: sys.stdout=io.open(1,"w",encoding="utf-8",closefd=False)
except: pass
import numpy as np,pandas as pd
from pathlib import Path
P=Path(r"C:\Users\kosaka256\Documents\rogii_claude");TR=P/"rogii-wellbore-geology-prediction"/"train"
A=P/"v64"/"artifacts_v64";AV=P/"v89"/"artifacts_v89"
K=4;SLOPE=55/1000.;CONT=4.0
DIST=dict(np.load(A/"v89_dist_f0_dist.npz"));EVD=pickle.loads(open(AV/"learned_boot.pkl","rb").read());PF=dict(np.load(AV/"pf_center_v89.npz"))
NB=90;ND=181;STEP=0.5
WELLS=[w for w in DIST if w in EVD and w in PF]

def run(N,DIPSP,DEV,CURV):
    rng=np.random.default_rng(0)
    eo=[];em=[];es=[];eu=[];nc_med=[]
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
        ki=np.where(kn)[0];ki=ki[ki>=max(0,hps-80)];pmd=md[ki]-md[hps];pch=(ti[ki]-tl)+(z[ki]-zl)
        mdv=md[ev];zv=z[ev];L=len(mdv);mr=mdv-mdv[0];dZ=zv-zl
        pdip=float(np.clip(np.polyfit(pmd,pch,1)[0],-SLOPE,SLOPE)) if(len(pmd)>=5 and(pmd[-1]-pmd[0])>1) else 0.0
        xk=np.linspace(0.,mr[-1]+1e-6,K);seg=np.diff(xk)
        dn=pdip+rng.uniform(-DIPSP,DIPSP,size=(N,1))
        dyk=np.concatenate([np.zeros((N,1)),np.cumsum(rng.normal(0,DEV,(N,K-1))*seg[None,:],axis=1)],axis=1)
        dev=np.empty((N,L))
        for i in range(N):dev[i]=np.interp(mr,xk,dyk[i])
        cand=tl+dn*mr[None,:]+dev-dZ[None,:]
        ok=((cand>=tt.min()-5)&(cand<=tt.max()+5)).all(axis=1)
        ok&=(np.abs(cand[:,:min(30,L)]-tl)<CONT).all(axis=1)
        # ★曲率拘束: 制御点偏差の2階差が小さい(geologyは急に曲がらない)
        if CURV<9:
            d2=np.abs(np.diff(dyk,2,axis=1)).max(axis=1) if K>=3 else np.zeros(N)
            ok&=(d2<CURV)
        if not ok.any():continue
        cand=cand[ok];true=tvt[ev];fin=np.isfinite(true)
        obs=pd.Series(gr[ev]).interpolate(limit_direction="both").fillna(float(np.nanmean(tg))).rolling(5,center=True,min_periods=1).mean().to_numpy()
        gmse=np.mean((np.interp(cand,tt,tg)-obs[None,:])**2,axis=1)
        r0=np.asarray(PF[wn],float)[:L]
        st=np.asarray(EVD[wn]["st"],float);bid=np.asarray(EVD[wn]["bid"],int);Pd=np.asarray(DIST[wn],float)
        Ln=min(L,len(st),len(bid));dl=cand[:,:Ln]-st[None,:Ln];j=np.round(dl/STEP+NB);jf=np.clip(j.astype(int),0,ND-1)
        us=np.where((j>=0)&(j<ND),np.log(Pd[bid[None,:Ln],jf]+1e-6),-12.).sum(1)
        zn=lambda v:(v-v.mean())/(v.std()+1e-9)
        r0d=np.sqrt(np.mean((cand-r0[None,:])**2,axis=1))
        eo.append((cand[int(np.argmin(np.sqrt(np.mean((cand-true[None,:])**2,axis=1))))]-true)[fin])
        em.append((cand[int(np.argmin(gmse))]-true)[fin])
        es.append((cand[int(np.argmax(-zn(r0d)))]-true)[fin])                       # r0スナップ
        eu.append((cand[int(np.argmax(-zn(r0d)+0.2*zn(us)))]-true)[fin])            # r0錨+U-Net
        nc_med.append(len(cand))
    r=lambda Lst:float(np.sqrt(np.mean(np.concatenate(Lst)**2)))
    print(f"  N={N} DIPSP={DIPSP*1000:.0f} DEV={DEV*1000:.0f} CURV={CURV}: oracle={r(eo):.2f} GRmse={r(em):.2f} r0snap={r(es):.2f} r0+UNet={r(eu):.2f} (候補中央{int(np.median(nc_med))})")

print("[候補空間を狭める掃引 155井]  基準=DIPSP20 DEV25 CURV=inf")
run(5000,20/1000.,25/1000.,99)
run(5000,10/1000.,25/1000.,99)
run(5000,10/1000.,12/1000.,99)
run(5000,5/1000.,8/1000.,99)
run(5000,10/1000.,25/1000.,1.5)   # 曲率拘束追加
run(5000,10/1000.,25/1000.,0.7)
