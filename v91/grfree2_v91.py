# -*- coding: utf-8 -*-
"""GRフリー脚 drift制御版: 減衰dip/平均回帰/damped。低ρ×RMSE~12狙い。field-out TUNED weightで評価
   (脚ごとにRMSE規模が違うので固定wでなくfield-out最適wが公平)。全773井。"""
import sys,io,pickle,glob,time
try: sys.stdout=io.open(1,"w",encoding="utf-8",closefd=False)
except: pass
import numpy as np,pandas as pd,pyarrow.parquet as pq
from pathlib import Path
from sklearn.cluster import KMeans
P=Path(r"C:\Users\kosaka256\Documents\rogii_claude");TR=P/"rogii-wellbore-geology-prediction"/"train"
A90=P/"v90"/"artifacts_v90";SLOPE=55/1000.;W_NN=0.62
tb=pq.read_table(A90/"data"/"train.parquet",columns=["well","target"])
well=tb.column("well").to_numpy(zero_copy_only=False);y=tb.column("target").to_numpy(zero_copy_only=False).astype(float)
gbm=np.load(A90/"gbm_v41_oof.npy").astype(float)[:len(y)];nnz=np.load(A90/"nn_oof_combo3.npz")
nn=np.mean([nnz[k] for k in nnz.files],axis=0)[:len(y)];v90err=(1-W_NN)*gbm+W_NN*nn-y
ibw={}
for i,w in enumerate(well): ibw.setdefault(w,[]).append(i)
for w in ibw: ibw[w]=np.array(ibw[w])
out={};t0=time.time();c=0
for f in sorted(glob.glob(str(TR/"*__horizontal_well.csv"))):
    wn=Path(f).name.split("__")[0]
    if wn not in ibw: continue
    hw=pd.read_csv(f,usecols=["MD","Z","TVT","TVT_input","X","Y"])
    kn=hw["TVT_input"].notna().to_numpy()
    if kn.sum()<10 or (~kn).sum()<30: continue
    hps=int(np.where(kn)[0].max());ev=slice(hps+1,len(hw))
    z=hw["Z"].to_numpy(float);md=hw["MD"].to_numpy(float);tvt=hw["TVT"].to_numpy(float);ti=hw["TVT_input"].to_numpy(float)
    try: tw=pd.read_csv(TR/f"{wn}__typewell.csv",usecols=["TVT"])
    except: continue
    tt=tw["TVT"].to_numpy(float);tt=tt[np.isfinite(tt)]
    if len(tt)<8: continue
    tcen=float(np.median(tt))
    tl=float(ti[hps]);zl=float(z[hps])
    ki=np.where(kn)[0];ki=ki[ki>=max(0,hps-80)]
    pmd=md[ki]-md[hps];pch=(ti[ki]-tl)+(z[ki]-zl)
    pdip=float(np.clip(np.polyfit(pmd,pch,1)[0],-SLOPE,SLOPE)) if(len(pmd)>=5 and(pmd[-1]-pmd[0])>1) else 0.0
    mr=md[ev]-md[hps];dZ=z[ev]-zl;true=tvt[ev];fin=np.isfinite(true)
    LAM=1500.  # 減衰長ft
    decay_dip=tl+pdip*LAM*(1-np.exp(-mr/LAM))-dZ            # dip減衰(遠方でflat=drift止め)
    damp=tl+pdip*mr*0.4-dZ                                   # dip 0.4倍
    change=pdip*LAM*(1-np.exp(-mr/LAM))
    revert=(tl+change-dZ)*np.exp(-mr/3000.)+ (tcen-dZ)*(1-np.exp(-mr/3000.))  # 平均回帰
    v90e=v90err[ibw[wn]];n=min(len(true),len(v90e))
    out[wn]=dict(v90=(v90e[:n])[fin[:n]],
                 decay=(decay_dip[:n]-true[:n])[fin[:n]],
                 damp=(damp[:n]-true[:n])[fin[:n]],
                 revert=(revert[:n]-true[:n])[fin[:n]],
                 cx=float(hw["X"].mean()),cy=float(hw["Y"].mean()))
    c+=1
    if c%200==0: print(f"  {c} ({time.time()-t0:.0f}s)",flush=True)
W=list(out);cx=np.array([out[w]["cx"] for w in W]);cy=np.array([out[w]["cy"] for w in W])
v90=np.concatenate([out[w]["v90"] for w in W]);r=lambda e:float(np.sqrt(np.mean(e**2)));base=r(v90)
fb=KMeans(n_clusters=8,n_init=5,random_state=0).fit_predict(np.column_stack([cx,cy]))
print(f"\n[GRフリー drift制御 {len(W)}井] v90 base={base:.4f}")
for lg in ["decay","damp","revert"]:
    a=np.concatenate([out[w][lg] for w in W]);rho=np.corrcoef(v90,a)[0,1]
    fo=[]
    for b in np.unique(fb):
        tr=[W[i] for i in range(len(W)) if fb[i]!=b];va=[W[i] for i in range(len(W)) if fb[i]==b]
        tv=np.concatenate([out[w]["v90"] for w in tr]);ta=np.concatenate([out[w][lg] for w in tr])
        wi=min([(r((1-w)*tv+w*ta),w) for w in np.arange(0,0.31,0.01)])[1]
        vv=np.concatenate([out[w]["v90"] for w in va]);aa=np.concatenate([out[w][lg] for w in va]);fo.append((1-wi)*vv+wi*aa)
    fo=r(np.concatenate(fo))
    # 全井最適w(参考)
    isw=min([(r((1-w)*v90+w*a),w) for w in np.arange(0,0.31,0.01)])
    print(f"  {lg:8s}: standalone={r(a):6.2f} ρ={rho:+.3f} | field-out={fo:.4f}(Δ{fo-base:+.4f}) 最適w={isw[1]:.2f}→{isw[0]:.4f}(Δ{isw[0]-base:+.4f})")
pickle.dump(out,open(P/"v91"/"grfree2_v91.pkl","wb"))
