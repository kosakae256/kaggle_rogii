# -*- coding: utf-8 -*-
"""複数物理脚(snap/pmean/multiseed)を生成し per-well誤差+centroidを保存→field-out blend評価用。
   狙い: 学習なしで最も脱相関×高精度な脚を確定。multiseed=gen雑音除去で精度↑→blend↑期待。"""
import os,sys,io,pickle,time,glob
try: sys.stdout=io.open(1,"w",encoding="utf-8",closefd=False)
except: pass
import numpy as np,pandas as pd,pyarrow.parquet as pq
from pathlib import Path
P=Path(r"C:\Users\kosaka256\Documents\rogii_claude");TR=P/"rogii-wellbore-geology-prediction"/"train"
AV=P/"v89"/"artifacts_v89";A90=P/"v90"/"artifacts_v90"
N=5000;K=4;SLOPE=55/1000.;DEV=25/1000.;DIPSP=20/1000.;CONT=4.0;W_NN=0.62;NSEED=int(os.environ.get("NSEED","3"))
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
PF=dict(np.load(AV/"pf_center_v89.npz"));out={};t0=time.time();c=0
for f in sorted(glob.glob(str(TR/"*__horizontal_well.csv"))):
    wn=Path(f).name.split("__")[0]
    if wn not in PF or wn not in ibw: continue
    hw=pd.read_csv(f)
    try: tw=pd.read_csv(TR/f"{wn}__typewell.csv")
    except: continue
    kn=hw["TVT_input"].notna().to_numpy()
    if kn.sum()<10 or (~kn).sum()<30: continue
    hps=int(np.where(kn)[0].max());ev=slice(hps+1,len(hw))
    z=hw["Z"].to_numpy(float);md=hw["MD"].to_numpy(float);tvt=hw["TVT"].to_numpy(float)
    ti=hw["TVT_input"].to_numpy(float);gr=hw["GR"].to_numpy(float);x=hw["X"].to_numpy(float);yy=hw["Y"].to_numpy(float)
    tt=tw["TVT"].to_numpy(float);tg=tw["GR"].to_numpy(float);m=np.isfinite(tt)&np.isfinite(tg);tt,tg=tt[m],tg[m]
    if len(tt)<8: continue
    o=np.argsort(tt);tt,tg=tt[o],tg[o];tl=float(ti[hps]);zl=float(z[hps])
    ki=np.where(kn)[0];ki=ki[ki>=max(0,hps-80)]
    true=tvt[ev];L=len(true);v90e=v90err[ibw[wn]];n=min(L,len(v90e))
    r0=np.asarray(PF[wn],float)[:L]
    snaps=[];pmeans=[]
    obs=pd.Series(gr[ev]).interpolate(limit_direction="both").fillna(float(np.nanmean(tg))).rolling(5,center=True,min_periods=1).mean().to_numpy()
    okgen=True
    for s in range(NSEED):
        cand=gen(md[ev],z[ev],tl,zl,float(tt.min()),float(tt.max()),md[ki]-md[hps],(ti[ki]-tl)+(z[ki]-zl),np.random.default_rng(s))
        if cand is None: okgen=False;break
        r0d=np.sqrt(np.mean((cand-r0[None,:])**2,axis=1))
        snaps.append(cand[int(np.argmin(r0d))])
        wgt=np.exp(-(r0d-r0d.min())/max(np.std(r0d),1e-6));wgt/=wgt.sum()   # r0周りposterior mean
        pmeans.append((wgt[:,None]*cand).sum(0))
    if not okgen: continue
    snap0=snaps[0][:n];snapms=np.mean([s[:n] for s in snaps],0);pmean0=pmeans[0][:n];pmeanms=np.mean([s[:n] for s in pmeans],0)
    true=true[:n];fin=np.isfinite(true)
    out[wn]=dict(v90=(v90e[:n])[fin],snap=(snap0-true)[fin],snapms=(snapms-true)[fin],
                 pmean=(pmean0-true)[fin],pmeanms=(pmeanms-true)[fin],
                 cx=float(np.mean(x)),cy=float(np.mean(yy)))
    c+=1
    if c%150==0: print(f"  {c} ({time.time()-t0:.0f}s)",flush=True)
pickle.dump(out,open(P/"v91"/f"legs_ns{NSEED}_v91.pkl","wb"))
print(f"saved legs_v91.pkl {c}井")
