# -*- coding: utf-8 -*-
"""全773井 ヘッジゲート用データ収集: v90誤差/物理推定(r0snap,GRsel)/自己申告特徴/per-row誤差/centroid。
   U-Net不要=全井。保存→gate_v91.pyがfield-out CVで学習。"""
import os,sys,io,pickle,time
try: sys.stdout=io.open(1,"w",encoding="utf-8",closefd=False)
except: pass
import numpy as np,pandas as pd,pyarrow.parquet as pq
from pathlib import Path
P=Path(r"C:\Users\kosaka256\Documents\rogii_claude");TR=P/"rogii-wellbore-geology-prediction"/"train"
AV=P/"v89"/"artifacts_v89";A90=P/"v90"/"artifacts_v90"
N=5000;K=4;SLOPE=55/1000.;DEV=25/1000.;DIPSP=20/1000.;CONT=4.0;W_NN=0.62
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
    return (cand[ok],pdip) if ok.any() else (None,pdip)
# v90 OOF + PF残差特徴
cols=["well","target","pf_gr_resid0","pf_gr_best_resid","pf_gr_resid0_rstd21","last_known_tvt"]
tb=pq.read_table(A90/"data"/"train.parquet",columns=cols)
well=tb.column("well").to_numpy(zero_copy_only=False)
y=tb.column("target").to_numpy(zero_copy_only=False).astype(float)
pfr0=tb.column("pf_gr_resid0").to_numpy(zero_copy_only=False).astype(float)
pfrb=tb.column("pf_gr_best_resid").to_numpy(zero_copy_only=False).astype(float)
pfrs=tb.column("pf_gr_resid0_rstd21").to_numpy(zero_copy_only=False).astype(float)
gbm=np.load(A90/"gbm_v41_oof.npy").astype(float)[:len(y)];nnz=np.load(A90/"nn_oof_combo3.npz")
nn=np.mean([nnz[k] for k in nnz.files],axis=0)[:len(y)];v90err=(1-W_NN)*gbm+W_NN*nn-y
ibw={}
for i,w in enumerate(well): ibw.setdefault(w,[]).append(i)
for w in ibw: ibw[w]=np.array(ibw[w])
PF=dict(np.load(AV/"pf_center_v89.npz"))
out={};t0=time.time();c=0
import glob
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
    cand,pdip=gen(md[ev],z[ev],tl,zl,float(tt.min()),float(tt.max()),md[ki]-md[hps],(ti[ki]-tl)+(z[ki]-zl))
    if cand is None: continue
    true=tvt[ev];fin=np.isfinite(true);L=len(true)
    obs=pd.Series(gr[ev]).interpolate(limit_direction="both").fillna(float(np.nanmean(tg))).rolling(5,center=True,min_periods=1).mean().to_numpy()
    gp=np.interp(cand,tt,tg);gmse=np.mean((gp-obs[None,:])**2,axis=1)
    r0=np.asarray(PF[wn],float)[:L];r0d=np.sqrt(np.mean((cand-r0[None,:])**2,axis=1))
    snap=cand[int(np.argmin(r0d))]
    grsel=cand[int(np.argmin(gmse))]
    v90e=v90err[ibw[wn]];n=min(L,len(v90e));f=fin[:n]
    # 特徴(真値非依存)
    cmean=cand.mean(axis=1)                 # 候補ごと平均TVT
    topg=np.argsort(gmse)[:max(10,N//20)]   # GR上位5%
    feat=dict(
        L=float(n), gr_var=float(np.var(obs)), pdip=abs(pdip),
        cand_spread=float(np.std(cmean)),           # 候補群の広がり=曖昧さ幅
        topgr_spread=float(np.std(cmean[topg])),    # GR上位の広がり=datum両義
        snap_move=float(np.mean(np.abs(snap[:n]-r0[:n]))),  # 物理がr0を動かした量
        r0_rough=float(np.mean(np.abs(np.diff(r0[:n],2)))), # r0の非物理粗さ
        snap_gr_resid=float(np.sqrt(gmse[int(np.argmin(r0d))])),
        grsel_gr_resid=float(np.sqrt(gmse.min())),
        v90_pfr=float(np.nanmean(pfr0[ibw[wn]])), v90_pfrb=float(np.nanmean(pfrb[ibw[wn]])),
        v90_pfrs=float(np.nanmean(pfrs[ibw[wn]])),
    )
    out[wn]=dict(v90=(v90e[:n])[f], snap=((snap[:n]-true[:n]))[f], grsel=((grsel[:n]-true[:n]))[f],
                 cx=float(np.mean(x)), cy=float(np.mean(yy)), feat=feat)
    c+=1
    if c%150==0: print(f"  {c} ({time.time()-t0:.0f}s)",flush=True)
pickle.dump(out,open(P/"v91"/"hedge_data_v91.pkl","wb"))
# ヘッジoracle(全井)
def pool(k): return float(np.sqrt(np.mean(np.concatenate([d[k] for d in out.values()])**2)))
def hedge(a,b): return float(np.sqrt(np.mean(np.concatenate([out[w][a] if np.mean(out[w][a]**2)<=np.mean(out[w][b]**2) else out[w][b] for w in out])**2)))
nsnap=sum(1 for w in out if np.mean(out[w]['snap']**2)<np.mean(out[w]['v90']**2))
print(f"\n[全{len(out)}井] v90={pool('v90'):.3f} / r0snap={pool('snap'):.3f} / GRsel={pool('grsel'):.3f}")
print(f"  ★ヘッジoracle v90 vs r0snap = {hedge('v90','snap'):.3f}  (物理勝ち{nsnap}井)")
print(f"  ★ヘッジoracle v90 vs GRsel  = {hedge('v90','grsel'):.3f}")
print(f"  saved hedge_data_v91.pkl")
