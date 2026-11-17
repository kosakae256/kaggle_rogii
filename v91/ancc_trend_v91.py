# -*- coding: utf-8 -*-
"""ANCC面: 大域トレンド平面+残差IDW(安定)。leave-well-out。真b/prefix-b。近傍除外でfield-out近似。"""
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
    xa=d["x"][s];ya=d["y"][s];aa=d["a"][s];fm=np.isfinite(aa)&np.isfinite(xa)&np.isfinite(ya)
    XS.append(xa[fm]);YS.append(ya[fm]);AS.append(aa[fm]);WID.append(np.full(fm.sum(),i))
XS=np.concatenate(XS);YS=np.concatenate(YS);AS=np.concatenate(AS);WID=np.concatenate(WID)
mx,my=XS.mean(),YS.mean();dx=XS-mx;dy=YS-my
# 大域2次トレンド(安定): ANCC ~ [1,x,y,x2,y2,xy]
G=np.column_stack([np.ones_like(dx),dx,dy,dx*dx,dy*dy,dx*dy])
beta=np.linalg.lstsq(G,AS,rcond=None)[0]
def trend(x,y):
     x=x-mx;y=y-my;return beta[0]+beta[1]*x+beta[2]*y+beta[3]*x*x+beta[4]*y*y+beta[5]*x*y
RES=AS-trend(XS+0,YS+0)  # 残差(注: XS,YSは元座標)
RES=AS-(beta[0]+beta[1]*dx+beta[2]*dy+beta[3]*dx*dx+beta[4]*dy*dy+beta[5]*dx*dy)
print(f"大域トレンド残差 RMS={np.sqrt(np.mean(RES**2)):.2f}ft (トレンドで説明後の面の凹凸)")
tree=cKDTree(np.column_stack([XS,YS]));KNN=16
def pred(qx,qy,wi,min_dist):
    q=np.column_stack([qx,qy]);dist,idx=tree.query(q,k=400);out=np.full(len(q),np.nan)
    tr=trend(qx,qy)
    for j in range(len(q)):
        ii=idx[j];dj=dist[j];m=(WID[ii]!=wi)&(dj>=min_dist);ii=ii[m][:KNN];dj=dj[m][:KNN]
        if len(ii)<3: out[j]=tr[j];continue                 # 近傍無ければトレンドのみ
        w=1/(dj**2+1e-6);out[j]=tr[j]+np.sum(w*RES[ii])/np.sum(w)
    return out
for MD_ in [0.0, 800.0]:
    Et=[];Ep=[];Etr=[]
    for i,wn in enumerate(names):
        d=wells[wn];kn=d["kn"];hps=int(np.where(kn)[0].max());ev=slice(hps+1,len(d["x"]));ki=np.where(kn)[0]
        anc=pred(d["x"][ev],d["y"][ev],i,MD_);anck=pred(d["x"][ki],d["y"][ki],i,MD_)
        z=d["z"];true=d["tvt"][ev]
        bt=np.median(d["tvt"]-(d["a"]-z));bp=np.nanmedian(d["ti"][ki]-(anck-z[ki]))
        fin=np.isfinite(anc)&np.isfinite(true)
        if fin.any():
            Et.append((anc-z[ev]+bt-true)[fin])
            if np.isfinite(bp):Ep.append((anc-z[ev]+bp-true)[fin])
            Etr.append((trend(d["x"][ev],d["y"][ev])-z[ev]+bt-true)[fin])   # トレンドのみ+真b
    r=lambda L:float(np.sqrt(np.mean(np.concatenate(L)**2)))
    print(f"[トレンド+残差IDW min_dist={MD_:.0f}ft]")
    print(f"   大域トレンドのみ+真b     = {r(Etr):.3f}")
    print(f"   トレンド+近傍残差+真b    = {r(Et):.3f}  ← 面予測の実力")
    print(f"   トレンド+近傍残差+prefix-b= {r(Ep):.3f}  ← 実現値")
