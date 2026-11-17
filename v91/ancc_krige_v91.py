# -*- coding: utf-8 -*-
"""★ANCC面クリギング検証: TVT=(ANCC-Z)+b。近傍他井からANCC(X,Y)を空間補間しTVT予測。
   leave-well-out / 距離しきいでfield-out近似。bは各井の既知帯(prefix)から。全773井。"""
import os,sys,io,glob
try: sys.stdout=io.open(1,"w",encoding="utf-8",closefd=False)
except: pass
import numpy as np,pandas as pd
from scipy.spatial import cKDTree
from pathlib import Path
TR=Path(r"C:\Users\kosaka256\Documents\rogii_claude")/"rogii-wellbore-geology-prediction"/"train"
DEC=8  # 各井 DEC点間引きでサンプル点に(面は滑らか)
KNN=12
files=sorted(glob.glob(str(TR/"*__horizontal_well.csv")))
# 全井の (X,Y,ANCC) サンプル点 + 井ID + 各井のフルデータ保持
XS=[];YS=[];AS=[];WID=[];wells={}
for wi,f in enumerate(files):
    wn=Path(f).name.split("__")[0]
    hw=pd.read_csv(f)
    x=hw["X"].to_numpy(float);y=hw["Y"].to_numpy(float);a=hw["ANCC"].to_numpy(float)
    z=hw["Z"].to_numpy(float);tvt=hw["TVT"].to_numpy(float);ti=hw["TVT_input"].to_numpy(float)
    kn=hw["TVT_input"].notna().to_numpy()
    if kn.sum()<10 or (~kn).sum()<30: continue
    wells[wn]=dict(x=x,y=y,a=a,z=z,tvt=tvt,ti=ti,kn=kn,wi=len(wells))
    s=slice(None,None,DEC)
    XS.append(x[s]);YS.append(y[s]);AS.append(a[s]);WID.append(np.full(x[s].shape,len(wells)-1))
XS=np.concatenate(XS);YS=np.concatenate(YS);AS=np.concatenate(AS);WID=np.concatenate(WID)
pts=np.column_stack([XS,YS]);tree=cKDTree(pts)
print(f"{len(wells)}井, サンプル点{len(XS)}")

def predict(wn, min_dist=0.0):
    """近傍他井(同井除外, min_dist以上離す)からANCCをIDW補間しTVT予測。"""
    d=wells[wn];x=d["x"];y=d["y"];z=d["z"];kn=d["kn"];wi=d["wi"]
    hps=int(np.where(kn)[0].max());ev=slice(hps+1,len(x))
    q=np.column_stack([x[ev],y[ev]])
    dist,idx=tree.query(q,k=KNN*4)   # 多めに取って同井/近すぎ除外後KNN使う
    anc=np.empty(len(q))
    for i in range(len(q)):
        ii=idx[i];dd=dist[i]
        m=(WID[ii]!=wi)&(dd>=min_dist)
        ii=ii[m][:KNN];dd=dd[m][:KNN]
        if len(ii)==0: anc[i]=np.nan;continue
        w=1.0/(dd**2+1e-6);anc[i]=np.sum(w*AS[ii])/np.sum(w)
    # b: 既知帯で TVT_input=(ANCC_pred-Z)+b だが既知帯はev外→既知帯のANCC_predも計算
    ki=np.where(kn)[0]
    qk=np.column_stack([x[ki],y[ki]])
    dk,ik=tree.query(qk,k=KNN*4);anck=np.empty(len(qk))
    for i in range(len(qk)):
        ii=ik[i];dd=dk[i];m=(WID[ii]!=wi)&(dd>=min_dist)
        ii=ii[m][:KNN];dd=dd[m][:KNN]
        anck[i]=np.sum((1/(dd**2+1e-6))*AS[ii])/np.sum(1/(dd**2+1e-6)) if len(ii) else np.nan
    b=np.nanmedian(d["ti"][ki]-(anck-z[ki]))
    pred=anc-z[ev]+b
    true=d["tvt"][ev];fin=np.isfinite(true)&np.isfinite(pred)
    return (pred-true)[fin]

for md_ in [0.0, 500.0, 1500.0]:
    E=[]
    for wn in wells: 
        e=predict(wn,md_)
        if len(e): E.append(e)
    e=np.concatenate(E);rmse=float(np.sqrt(np.mean(e**2)))
    print(f"  [近傍除外 min_dist={md_:>6.0f}ft] ANCC面クリギング TVT overall RMSE = {rmse:.3f}  ({len(E)}井)")
