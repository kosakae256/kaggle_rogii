# -*- coding: utf-8 -*-
"""ANCC面クリギングの素性診断: 井戸間距離分布 + ANCC予測誤差(=TVT誤差)を近傍数/距離別に。"""
import os,sys,io,glob
try: sys.stdout=io.open(1,"w",encoding="utf-8",closefd=False)
except: pass
import numpy as np,pandas as pd
from scipy.spatial import cKDTree
from pathlib import Path
TR=Path(r"C:\Users\kosaka256\Documents\rogii_claude")/"rogii-wellbore-geology-prediction"/"train"
files=sorted(glob.glob(str(TR/"*__horizontal_well.csv")))
wells={};cent=[]
for f in files:
    wn=Path(f).name.split("__")[0];hw=pd.read_csv(f)
    kn=hw["TVT_input"].notna().to_numpy()
    if kn.sum()<10 or (~kn).sum()<30: continue
    d=dict(x=hw["X"].to_numpy(float),y=hw["Y"].to_numpy(float),a=hw["ANCC"].to_numpy(float),
           z=hw["Z"].to_numpy(float),tvt=hw["TVT"].to_numpy(float),ti=hw["TVT_input"].to_numpy(float),kn=kn)
    wells[wn]=d;cent.append([d["x"].mean(),d["y"].mean()])
cent=np.array(cent);names=list(wells)
ct=cKDTree(cent);dd,_=ct.query(cent,k=2)
print(f"{len(wells)}井. 井戸centroid最近傍距離: 中央{np.median(dd[:,1]):.0f}ft 25%={np.percentile(dd[:,1],25):.0f} 75%={np.percentile(dd[:,1],75):.0f}")
# 全点tree(井ID付, DEC間引き)
DEC=6;XS=[];YS=[];AS=[];WID=[]
for i,wn in enumerate(names):
    d=wells[wn];s=slice(None,None,DEC)
    XS.append(d["x"][s]);YS.append(d["y"][s]);AS.append(d["a"][s]);WID.append(np.full(d["x"][s].shape,i))
XS=np.concatenate(XS);YS=np.concatenate(YS);AS=np.concatenate(AS);WID=np.concatenate(WID)
tree=cKDTree(np.column_stack([XS,YS]))
# 各井eval点で 他井最近傍ANCCとの誤差(=IDW k=8) と 最近傍距離
for KNN in [8]:
    E=[];NND=[]
    for i,wn in enumerate(names):
        d=wells[wn];kn=d["kn"];hps=int(np.where(kn)[0].max());ev=slice(hps+1,len(d["x"]))
        q=np.column_stack([d["x"][ev],d["y"][ev]])
        dist,idx=tree.query(q,k=200)
        anc=np.full(len(q),np.nan);nnd=np.full(len(q),np.nan)
        for j in range(len(q)):
            ii=idx[j];dj=dist[j];m=WID[ii]!=i
            ii=ii[m][:KNN];dj=dj[m][:KNN]
            if len(ii)==0: continue
            w=1/(dj**2+1e-6);anc[j]=np.sum(w*AS[ii])/np.sum(w);nnd[j]=dj[0]
        true=d["tvt"][ev];b=np.median(d["tvt"]-(d["a"]-d["z"]))  # 真b(ANCC誤差のみ分離)
        pred=anc-d["z"][ev]+b
        fin=np.isfinite(pred)&np.isfinite(true)
        if fin.any(): E.append((pred-true)[fin]);NND.append(nnd[fin])
    e=np.concatenate(E);nnd=np.concatenate(NND)
    print(f"  KNN={KNN}: ANCC面予測 TVT RMSE={np.sqrt(np.mean(e**2)):.3f} (真b使用, {len(E)}井)  eval点の他井最近傍距離 中央{np.median(nnd):.0f}ft")
    for thr in [200,500,1000]:
        mm=nnd<thr
        if mm.sum()>100: print(f"      他井最近傍<{thr}ft の点({mm.mean()*100:.0f}%): RMSE={np.sqrt(np.mean(e[mm]**2)):.3f}")
