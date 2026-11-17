# -*- coding: utf-8 -*-
"""ANCC面クリギング最終: 局所平面(座標kft化+ridge+近傍レンジclip=安全)。leave-well-out, 近傍除外でfield-out近似。"""
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
    d=wells[wn];s=slice(None,None,DEC);xa=d["x"][s];ya=d["y"][s];aa=d["a"][s]
    fm=np.isfinite(aa)&np.isfinite(xa)&np.isfinite(ya)
    XS.append(xa[fm]);YS.append(ya[fm]);AS.append(aa[fm]);WID.append(np.full(fm.sum(),i))
XS=np.concatenate(XS);YS=np.concatenate(YS);AS=np.concatenate(AS);WID=np.concatenate(WID)
tree=cKDTree(np.column_stack([XS,YS]));KNN=20
def pred(qx,qy,wi,min_dist):
    q=np.column_stack([qx,qy]);dist,idx=tree.query(q,k=400);out=np.full(len(q),np.nan)
    for j in range(len(q)):
        ii=idx[j];dj=dist[j];m=(WID[ii]!=wi)&(dj>=min_dist);ii=ii[m][:KNN];dj=dj[m][:KNN]
        if len(ii)<4: continue
        av=AS[ii];lo,hi=av.min(),av.max()
        dx=(XS[ii]-qx[j])/1000.;dy=(YS[ii]-qy[j])/1000.
        X=np.column_stack([np.ones(len(ii)),dx,dy]);w=1/(dj**2+1e-6)
        A=(X*w[:,None]).T@X+1e-3*np.eye(3);bb=(X*w[:,None]).T@av
        try: v=np.linalg.solve(A,bb)[0]
        except: v=np.sum(w*av)/np.sum(w)
        out[j]=min(max(v,lo),hi)                            # 近傍レンジにclip(暴走防止)
    return out
for MD_ in [0.0, 800.0]:
    Et=[];Ep=[]
    for i,wn in enumerate(names):
        d=wells[wn];kn=d["kn"];hps=int(np.where(kn)[0].max());ev=slice(hps+1,len(d["x"]));ki=np.where(kn)[0]
        anc=pred(d["x"][ev],d["y"][ev],i,MD_);anck=pred(d["x"][ki],d["y"][ki],i,MD_)
        z=d["z"];true=d["tvt"][ev]
        bt=np.median((d["tvt"]-(d["a"]-z))[np.isfinite(d["a"])]);bp=np.nanmedian(d["ti"][ki]-(anck-z[ki]))
        fin=np.isfinite(anc)&np.isfinite(true)
        if fin.any():
            Et.append((anc-z[ev]+bt-true)[fin])
            if np.isfinite(bp):Ep.append((anc-z[ev]+bp-true)[fin])
    r=lambda L:float(np.sqrt(np.mean(np.concatenate(L)**2)))
    print(f"[ANCC面クリギング最終 min_dist={MD_:.0f}ft KNN={KNN} ({len(Et)}井)]")
    print(f"   +真b(面予測の純粋な実力)  = {r(Et):.3f}")
    print(f"   +prefix-b(実運用相当)     = {r(Ep):.3f}")
