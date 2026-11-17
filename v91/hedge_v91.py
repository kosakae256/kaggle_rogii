# -*- coding: utf-8 -*-
"""多仮説ヘッジのheadroom測定: v90を物理候補でヘッジ(井戸/区間ごとに最良を取れたら)どこまで下がるか。
   置換でなくヘッジ=oracle retrievableの思想([[v34-seed-dump]])。155井fold0。"""
import os,sys,io,pickle
try: sys.stdout=io.open(1,"w",encoding="utf-8",closefd=False)
except: pass
import numpy as np,pandas as pd,pyarrow.parquet as pq
from pathlib import Path
P=Path(r"C:\Users\kosaka256\Documents\rogii_claude");TR=P/"rogii-wellbore-geology-prediction"/"train"
A=P/"v64"/"artifacts_v64";AV=P/"v89"/"artifacts_v89";A90=P/"v90"/"artifacts_v90"
N=5000;K=4;SLOPE=55/1000.;DEV=25/1000.;DIPSP=20/1000.;CONT=4.0;W_NN=0.62;NB=90;ND=181;STEP=0.5
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
DIST=dict(np.load(A/"v89_dist_f0_dist.npz"));EVD=pickle.loads(open(AV/"learned_boot.pkl","rb").read());PF=dict(np.load(AV/"pf_center_v89.npz"))
zn=lambda v:(v-v.mean())/(v.std()+1e-9)
rows=[]
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
    true=tvt[ev];fin=np.isfinite(true);L=len(true)
    st=np.asarray(EVD[wn]["st"],float);bid=np.asarray(EVD[wn]["bid"],int);Pd=np.asarray(DIST[wn],float)
    Ln=min(L,len(st),len(bid));dl=cand[:,:Ln]-st[None,:Ln];j=np.round(dl/STEP+NB);jf=np.clip(j.astype(int),0,ND-1)
    us=np.where((j>=0)&(j<ND),np.log(Pd[bid[None,:Ln],jf]+1e-6),-12.).sum(1)
    r0=np.asarray(PF[wn],float)[:L];r0d=np.sqrt(np.mean((cand-r0[None,:])**2,axis=1))
    ref=cand[int(np.argmax(-zn(r0d)+0.2*zn(us)))]
    orc=cand[int(np.argmin(np.sqrt(np.mean((cand-true[None,:])**2,axis=1))))]
    v90e=v90err[np.array(ibw[wn])];n=min(L,len(v90e));f=fin[:n]
    rms=lambda e:np.sqrt(np.mean(e[np.isfinite(e)]**2))
    rows.append(dict(v90=(v90e[:n])[f],phys=((ref[:n]-true[:n]))[f],orc=((orc[:n]-true[:n]))[f]))
def pool(key): return np.sqrt(np.mean(np.concatenate([r[key] for r in rows])**2))
def hedge(a,b):  # 井戸ごとに rms小さい方を採用
    return np.sqrt(np.mean(np.concatenate([r[a] if np.mean(r[a]**2)<=np.mean(r[b]**2) else r[b] for r in rows])**2))
print(f"[多仮説ヘッジ headroom {len(rows)}井]")
print(f"  v90単体                    = {pool('v90'):.3f}")
print(f"  物理選択(r0錨+U-Net)単体    = {pool('phys'):.3f}")
print(f"  物理oracle単体              = {pool('orc'):.3f}")
print(f"  ★ヘッジoracle v90 vs 物理選択 = {hedge('v90','phys'):.3f}  (井戸ごと完全ゲート)")
print(f"  ★ヘッジoracle v90 vs 物理oracle= {hedge('v90','orc'):.3f}  (生成の天井をゲートで拾えたら)")
# 何井が物理側で勝つか
nphys=sum(1 for r in rows if np.mean(r['phys']**2)<np.mean(r['v90']**2))
norc=sum(1 for r in rows if np.mean(r['orc']**2)<np.mean(r['v90']**2))
print(f"  物理選択がv90に勝つ井: {nphys}/{len(rows)} / 物理oracleが勝つ井: {norc}/{len(rows)}")

# ゲート実現性: 「物理選択がv90に勝つ」を自己申告特徴で分離できるか(AUC)
import numpy as np
feats={};lab=[]
for r in rows:
    lab.append(1 if np.mean(r['phys']**2)<np.mean(r['v90']**2) else 0)
lab=np.array(lab)
def auc(x,y):
    x=np.asarray(x,float);m=np.isfinite(x)
    x=x[m];yy=y[m]
    if len(np.unique(yy))<2: return .5
    r=np.argsort(np.argsort(x));n1=yy.sum();n0=len(yy)-n1
    return (r[yy==1].sum()-n1*(n1-1)/2)/(n1*n0)
# 特徴を rows作成時に保存してないので再計算せず, 保持済みのものだけ: v90/phys/orcのRMSE, umargin等は無い
# → 近似特徴: v90自身のrms(自信度代理), physのrms, 両者のrms差の符号なし要素は使えない(リーク)
# 実際に使えるゲート特徴(真値非依存)を rows に足すため再ループが要る→ここでは v90rms と physrms のみ相関確認(参考)
v90rms=np.array([np.sqrt(np.mean(r['v90']**2)) for r in rows])
physrms=np.array([np.sqrt(np.mean(r['phys']**2)) for r in rows])
print(f"\n[ゲート分離 参考] 物理勝ち{lab.sum()}井")
print(f"  AUC(v90rms→物理勝ち)  = {auc(v90rms,lab):.3f} (v90が自分で苦戦を申告できれば高い=リークだが上界)")
print(f"  AUC(physrms→物理勝ち) = {auc(-physrms,lab):.3f}")
