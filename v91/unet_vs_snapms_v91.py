# -*- coding: utf-8 -*-
"""155井fold0: U-Net物理脚(r0錨+U-Net,7.49) vs snapms のv90 blend値比較→U-Net全fold投資判断。"""
import sys,io,pickle
try: sys.stdout=io.open(1,"w",encoding="utf-8",closefd=False)
except: pass
import numpy as np,pandas as pd,pyarrow.parquet as pq
from pathlib import Path
P=Path(r"C:\Users\kosaka256\Documents\rogii_claude");TR=P/"rogii-wellbore-geology-prediction"/"train"
A=P/"v64"/"artifacts_v64";AV=P/"v89"/"artifacts_v89";A90=P/"v90"/"artifacts_v90"
N=5000;K=4;SLOPE=55/1000.;DEV=25/1000.;DIPSP=20/1000.;CONT=4.0;W_NN=0.62;NB=90;ND=181;STEP=0.5;W_EV=0.2
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
LG=pickle.load(open(P/"v91"/"legs_v91.pkl","rb"))
zn=lambda v:(v-v.mean())/(v.std()+1e-9)
Ev=[];Eu=[];Es=[]
for wn in [w for w in DIST if w in EVD and w in PF and w in ibw and w in LG]:
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
    st=np.asarray(EVD[wn]["st"],float);bid=np.asarray(EVD[wn]["bid"],int);Pd=np.asarray(DIST[wn],float)
    Ln=min(L,len(st),len(bid));dl=cand[:,:Ln]-st[None,:Ln];j=np.round(dl/STEP+NB);jf=np.clip(j.astype(int),0,ND-1)
    us=np.where((j>=0)&(j<ND),np.log(Pd[bid[None,:Ln],jf]+1e-6),-12.).sum(1)
    obs=pd.Series(gr[ev]).interpolate(limit_direction="both").fillna(float(np.nanmean(tg))).rolling(5,center=True,min_periods=1).mean().to_numpy()
    gmse=np.mean((np.interp(cand,tt,tg)-obs[None,:])**2,axis=1)
    r0=np.asarray(PF[wn],float)[:L];r0d=np.sqrt(np.mean((cand-r0[None,:])**2,axis=1))
    uref=cand[int(np.argmax(-zn(r0d)+W_EV*(-zn(gmse)+zn(us))))]
    v90e=v90err[ibw[wn]];n=min(L,len(v90e),len(LG[wn]["v90"])+10**9)
    n=min(L,len(v90e))
    Ev.append((v90e[:n])[fin[:n]]);Eu.append((uref[:n]-true[:n])[fin[:n]])
    Es.append(LG[wn]["snapms"])   # 既に(pred-true)[fin]済み
# 長さ整合: Es は legs.pkl のfin後配列, Ev/Eu は今計算のfin後配列。同じ井なら同長のはず
v90=np.concatenate(Ev);uu=np.concatenate(Eu)
try: ss=np.concatenate(Es)
except: ss=None
r=lambda e:float(np.sqrt(np.mean(e**2)))
print(f"[155井 U-Net物理 vs snapms のblend比較] v90={r(v90):.4f}")
print(f"  U-Net物理: standalone={r(uu):.3f} ρ={np.corrcoef(v90,uu)[0,1]:.3f}")
for wq in [0.1,0.15,0.2,0.25,0.3]:
    print(f"     v90+U-Net w={wq}: {r((1-wq)*v90+wq*uu):.4f} (Δ{r((1-wq)*v90+wq*uu)-r(v90):+.4f})")
if ss is not None and len(ss)==len(v90):
    print(f"  snapms(参考同井): standalone={r(ss):.3f} ρ={np.corrcoef(v90,ss)[0,1]:.3f}")
    for wq in [0.1,0.15,0.2]:
        print(f"     v90+snapms w={wq}: {r((1-wq)*v90+wq*ss):.4f} (Δ{r((1-wq)*v90+wq*ss)-r(v90):+.4f})")
    # U-Net + snapms 両脚
    best=(r(v90),0,0)
    for w1 in np.arange(0,0.31,0.03):
        for w2 in np.arange(0,0.31,0.03):
            if w1+w2>0.5:continue
            e=(1-w1-w2)*v90+w1*uu+w2*ss
            if r(e)<best[0]:best=(r(e),w1,w2)
    print(f"  ★v90+U-Net+snapms 両脚: {best[0]:.4f} (w_unet={best[1]:.2f} w_snapms={best[2]:.2f}, Δ{best[0]-r(v90):+.4f})")
else:
    print("  snapms長さ不整合でスキップ(井集合差)")
