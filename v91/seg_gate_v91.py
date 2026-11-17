# -*- coding: utf-8 -*-
"""区間単位ゲートの premise 検証: 局所GR証拠でv90 vs 物理(r0snap)を区間ごとに判別できるか。
   スカラー井ゲート(-0.014)を超えるか。区間oracle(headroom)も。全773井。"""
import os,sys,io,pickle,time,glob
try: sys.stdout=io.open(1,"w",encoding="utf-8",closefd=False)
except: pass
import numpy as np,pandas as pd,pyarrow.parquet as pq
from pathlib import Path
P=Path(r"C:\Users\kosaka256\Documents\rogii_claude");TR=P/"rogii-wellbore-geology-prediction"/"train"
AV=P/"v89"/"artifacts_v89";A90=P/"v90"/"artifacts_v90"
N=5000;K=4;SLOPE=55/1000.;DEV=25/1000.;DIPSP=20/1000.;CONT=4.0;W_NN=0.62;SEG=int(os.environ.get("SEG","150"))
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
PF=dict(np.load(AV/"pf_center_v89.npz"))
E_base=[];E_gate=[];E_orc=[];E_wellgate=[];t0=time.time();c=0
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
    ti=hw["TVT_input"].to_numpy(float);gr=hw["GR"].to_numpy(float)
    tt=tw["TVT"].to_numpy(float);tg=tw["GR"].to_numpy(float);m=np.isfinite(tt)&np.isfinite(tg);tt,tg=tt[m],tg[m]
    if len(tt)<8: continue
    o=np.argsort(tt);tt,tg=tt[o],tg[o];tl=float(ti[hps]);zl=float(z[hps])
    ki=np.where(kn)[0];ki=ki[ki>=max(0,hps-80)]
    cand=gen(md[ev],z[ev],tl,zl,float(tt.min()),float(tt.max()),md[ki]-md[hps],(ti[ki]-tl)+(z[ki]-zl))
    if cand is None: continue
    true=tvt[ev];L=len(true)
    v90e=v90err[ibw[wn]];n=min(L,len(v90e))
    true=true[:n];cand=cand[:,:n];v90e=v90e[:n]
    v90_tvt=true+v90e
    r0=np.asarray(PF[wn],float)[:n];r0d=np.sqrt(np.mean((cand-r0[None,:])**2,axis=1));phys=cand[int(np.argmin(r0d))]
    obs=pd.Series(gr[ev][:n]).interpolate(limit_direction="both").fillna(float(np.nanmean(tg))).rolling(5,center=True,min_periods=1).mean().to_numpy()
    grv=np.abs(np.interp(v90_tvt,tt,tg)-obs)     # v90の局所GR残差(行)
    grp=np.abs(np.interp(phys,tt,tg)-obs)        # 物理の局所GR残差(行)
    out=v90_tvt.copy();out_orc=v90_tvt.copy();fin=np.isfinite(true)
    for a in range(0,n,SEG):
        b=min(a+SEG,n);sl=slice(a,b)
        # 局所GRゲート: 区間内でGR残差が小さい方を採用
        if np.nanmean(grp[sl])<np.nanmean(grv[sl]): out[sl]=phys[sl]
        # 区間oracle
        if np.nanmean(np.abs(phys[sl]-true[sl]))<np.nanmean(np.abs(v90_tvt[sl]-true[sl])): out_orc[sl]=phys[sl]
    E_base.append((v90_tvt-true)[fin]);E_gate.append((out-true)[fin]);E_orc.append((out_orc-true)[fin])
    # 井単位ゲート(局所GR全体で比較)
    wg=phys if np.nanmean(grp)<np.nanmean(grv) else v90_tvt
    E_wellgate.append((wg-true)[fin])
    c+=1
    if c%150==0: print(f"  {c} ({time.time()-t0:.0f}s)",flush=True)
r=lambda L:float(np.sqrt(np.mean(np.concatenate(L)**2)))
print(f"\n[区間ゲート SEG={SEG} {c}井]")
print(f"  v90 base            = {r(E_base):.4f}")
print(f"  局所GR 井単位ゲート   = {r(E_wellgate):.4f}")
print(f"  ★局所GR 区間ゲート   = {r(E_gate):.4f}")
print(f"  区間oracle(headroom) = {r(E_orc):.4f}")
