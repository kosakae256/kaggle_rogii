# -*- coding: utf-8 -*-
"""ANCC面クリギング(局所平面フィット=universal kriging deg1で傾斜除去)。真b/prefix-b両方。
   比較: 自井prefix-dip外挿(近傍不使用) vs 近傍平面フィット。全773井 leave-well-out。"""
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
def plane_pred(qx,qy,wi):
    q=np.column_stack([qx,qy]);dist,idx=tree.query(q,k=300);out=np.full(len(q),np.nan)
    for j in range(len(q)):
        ii=idx[j];dj=dist[j];m=WID[ii]!=wi;ii=ii[m][:KNN];dj=dj[m][:KNN]
        if len(ii)<5: continue
        X=np.column_stack([np.ones(len(ii)),XS[ii],YS[ii]]);w=1/(dj**2+1e-6)
        try:
            beta=np.linalg.lstsq(X*w[:,None],AS[ii]*w,rcond=None)[0]
            out[j]=beta[0]+beta[1]*qx[j]+beta[2]*qy[j]
        except: pass
    return out
E_true=[];E_pre=[];E_self=[]
for i,wn in enumerate(names):
    d=wells[wn];kn=d["kn"];hps=int(np.where(kn)[0].max());ev=slice(hps+1,len(d["x"]));ki=np.where(kn)[0]
    anc=plane_pred(d["x"][ev],d["y"][ev],i)
    anck=plane_pred(d["x"][ki],d["y"][ki],i)
    true=d["tvt"][ev];z=d["z"]
    bt=np.median(d["tvt"]-(d["a"]-z))                      # 真b(ANCC誤差のみ)
    bp=np.nanmedian(d["ti"][ki]-(anck-z[ki]))             # prefix近傍からのb
    fin=np.isfinite(anc)&np.isfinite(true)
    if fin.any():
        E_true.append((anc-z[ev]+bt-true)[fin])
        if np.isfinite(bp): E_pre.append((anc-z[ev]+bp-true)[fin])
    # 自井prefix-dip外挿(近傍不使用baseline): ANCCをprefixの線形trendで外挿
    pm=d["x"][ki];# 使わず, MDベースdip
    md_ki=np.arange(len(ki));# 代わりに既知帯ANCC(真)からdip外挿は反則→ti使う
    # prefix dipは ti から: geo_change ~ (ti-last)+(z-zlast)
    tl=d["ti"][hps];zl=z[hps];mdrel=np.arange(len(d["x"]))
    kirel=ki-hps;pch=(d["ti"][ki]-tl)+(z[ki]-zl)
    if len(ki)>=5:
        dip=np.polyfit(kirel[kirel<=0] if (kirel<=0).any() else kirel,pch[kirel<=0] if (kirel<=0).any() else pch,1)[0] if len(np.unique(kirel))>1 else 0
    else: dip=0
    ev_rel=np.arange(hps+1,len(d["x"]))-hps
    self_pred=tl+dip*ev_rel-(z[ev]-zl)
    E_self.append((self_pred-true)[np.isfinite(true)])
r=lambda L:float(np.sqrt(np.mean(np.concatenate(L)**2)))
print(f"[ANCC面 局所平面クリギング {len(names)}井 leave-well-out KNN={KNN}]")
print(f"  自井prefix-dip外挿(近傍不使用)      = {r(E_self):.3f}  (baseline)")
print(f"  近傍平面フィット + 真b(ANCC誤差のみ) = {r(E_true):.3f}  ← 面予測の実力")
print(f"  近傍平面フィット + prefix-b(実運用)  = {r(E_pre):.3f}  ← 実現値")
