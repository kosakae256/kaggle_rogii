# -*- coding: utf-8 -*-
"""ANCC面クリギング(局所平面, 座標中心化で数値安定)。真b/prefix-b。leave-well-out & field-out近似。"""
import os,sys,io,glob
try: sys.stdout=io.open(1,"w",encoding="utf-8",closefd=False)
except: pass
import numpy as np,pandas as pd
from scipy.spatial import cKDTree
from pathlib import Path
TR=Path(r"C:\Users\kosaka256\Documents\rogii_claude")/"rogii-wellbore-geology-prediction"/"train"
files=sorted(glob.glob(str(TR/"*__horizontal_well.csv")))
wells={};names=[]
for f in files:
    wn=Path(f).name.split("__")[0];hw=pd.read_csv(f)
    kn=hw["TVT_input"].notna().to_numpy()
    if kn.sum()<10 or (~kn).sum()<30: continue
    wells[wn]=dict(x=hw["X"].to_numpy(float),y=hw["Y"].to_numpy(float),a=hw["ANCC"].to_numpy(float),
           z=hw["Z"].to_numpy(float),tvt=hw["TVT"].to_numpy(float),ti=hw["TVT_input"].to_numpy(float),kn=kn);names.append(wn)
DEC=6;XS=[];YS=[];AS=[];WID=[]
for i,wn in enumerate(names):
    d=wells[wn];s=slice(None,None,DEC)
    XS.append(d["x"][s]);YS.append(d["y"][s]);AS.append(d["a"][s]);WID.append(np.full(d["x"][s].shape,i))
XS=np.concatenate(XS);YS=np.concatenate(YS);AS=np.concatenate(AS);WID=np.concatenate(WID)
tree=cKDTree(np.column_stack([XS,YS]))
KNN=30
def plane_pred(qx,qy,wi,min_dist=0.0):
    q=np.column_stack([qx,qy]);dist,idx=tree.query(q,k=400);out=np.full(len(q),np.nan)
    for j in range(len(q)):
        ii=idx[j];dj=dist[j];m=(WID[ii]!=wi)&(dj>=min_dist);ii=ii[m][:KNN];dj=dj[m][:KNN]
        if len(ii)<6: continue
        dx=XS[ii]-qx[j];dy=YS[ii]-qy[j]                     # ★局所座標中心化
        X=np.column_stack([np.ones(len(ii)),dx,dy]);w=np.sqrt(1/(dj**2+1e-6))
        try:
            beta=np.linalg.lstsq(X*w[:,None],AS[ii]*w,rcond=None)[0];out[j]=beta[0]  # 中心=切片
        except: pass
    return out
for MD_ in [0.0, 800.0]:
    E_true=[];E_pre=[]
    for i,wn in enumerate(names):
        d=wells[wn];kn=d["kn"];hps=int(np.where(kn)[0].max());ev=slice(hps+1,len(d["x"]));ki=np.where(kn)[0]
        anc=plane_pred(d["x"][ev],d["y"][ev],i,MD_)
        anck=plane_pred(d["x"][ki],d["y"][ki],i,MD_)
        true=d["tvt"][ev];z=d["z"]
        bt=np.median(d["tvt"]-(d["a"]-z));bp=np.nanmedian(d["ti"][ki]-(anck-z[ki]))
        fin=np.isfinite(anc)&np.isfinite(true)
        if fin.any():
            E_true.append((anc-z[ev]+bt-true)[fin])
            if np.isfinite(bp): E_pre.append((anc-z[ev]+bp-true)[fin])
    r=lambda L:float(np.sqrt(np.mean(np.concatenate(L)**2)))
    print(f"[局所平面クリギング 近傍除外min_dist={MD_:.0f}ft KNN={KNN}]")
    print(f"   +真b(面予測の実力)  = {r(E_true):.3f} ({len(E_true)}井)")
    print(f"   +prefix-b(実現値)   = {r(E_pre):.3f} ({len(E_pre)}井)")
