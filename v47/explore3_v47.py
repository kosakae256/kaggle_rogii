# -*- coding: utf-8 -*-
"""反復3: 各枝(mode-rep 経路)に沿った「typewell GR」と「horizontal GR」の窓内相関(波形一致)を
判別特徴に加え、枝分類精度が 54% を超えるか。振幅ミスフィットより地質的に強いはずのtiebreak。v47 artifact のみ。"""
import os, time, numpy as np, pandas as pd
from pathlib import Path
from scipy.ndimage import uniform_filter1d
from sklearn.model_selection import GroupKFold
P=Path(r"C:\Users\kosaka256\Documents\rogii_claude"); ART=P/"v47"/"artifacts_v47"; TRAIN=P/"rogii-wellbore-geology-prediction"/"train"
z=np.load(ART/"candidates_v47.npz"); DEL=z["DEL"]; y=z["y"]; g=z["wellcode"].astype(np.int64); mds=z["mds"].astype(np.float64)
STD=z["STD"]; GRC=z["GRC"]; GRC2=z["GRC2"]; STRD=z["STRD"]; STRS=z["STRS"]; CTX=z["ctx"]; lk=z["lk"].astype(np.float64); N=len(y)
ids=np.load(ART/"ids_v47.npy",allow_pickle=True).astype(str); wells_name=__import__("json").loads((ART/"candidates_v47_meta.json").read_text(encoding="utf-8"))["wells"]
K=3; KT=21//K
a=np.argsort(DEL,1); rep=np.take_along_axis(DEL,a,1).reshape(N,K,KT).mean(2)          # [N,K]
gold=np.argmin(np.abs(rep-y[:,None]),1)
def rmse(p): e=p-y; return float(np.sqrt(np.mean(e*e)))

# ---- 各枝経路に沿った GR窓内相関(坑井ごとにスライディング相関を running-stat で高速計算) ----
rn=pd.Series(ids).str.rsplit("_",n=1,expand=True)[1].astype(int).to_numpy()
CORR=np.zeros((N,K),np.float32); t0=time.time()
for wi,wn in enumerate(wells_name):
    ri=np.where(g==wi)[0]
    if len(ri)==0: continue
    o=np.argsort(rn[ri]); ri=ri[o]; r=rn[ri]
    try:
        hw=pd.read_csv(TRAIN/f"{wn}__horizontal_well.csv",usecols=["GR"]); tw=pd.read_csv(TRAIN/f"{wn}__typewell.csv",usecols=["TVT","GR"]).sort_values("TVT")
    except Exception: continue
    hgr=hw["GR"].astype(float).interpolate(limit_direction="both").to_numpy(); fb=float(np.nanmean(tw["GR"]))
    hgr=np.where(np.isfinite(hgr),hgr,fb); h=hgr[np.clip(r,0,len(hgr)-1)]                # [n] 実測GR(eval行)
    ttv=tw["TVT"].to_numpy(float); tgr=tw["GR"].to_numpy(float)
    for W in [151]:                                                                       # 窓(行)
        for k in range(K):
            e=np.interp(lk[ri]+rep[ri,k],ttv,tgr)                                          # 枝kに沿った期待GR
            mh=uniform_filter1d(h,W,mode="nearest"); me=uniform_filter1d(e,W,mode="nearest")
            mhh=uniform_filter1d(h*h,W,mode="nearest"); mee=uniform_filter1d(e*e,W,mode="nearest"); mhe=uniform_filter1d(h*e,W,mode="nearest")
            cov=mhe-mh*me; sh=np.sqrt(np.maximum(mhh-mh*mh,1e-6)); se=np.sqrt(np.maximum(mee-me*me,1e-6))
            CORR[ri,k]=(cov/(sh*se)).astype(np.float32)
    if (wi+1)%200==0: print(f"  corr {wi+1}/{len(wells_name)} ({time.time()-t0:.0f}s)",flush=True)
print(f"GR相関計算 完了 ({time.time()-t0:.0f}s)  corr範囲 med={np.median(CORR):.3f}")

# argmax-corr 枝(ゼロ学習)= 波形一致が最良の枝を選ぶ
print(f"[zero] argmax-GR相関 枝: acc={np.mean(np.argmax(CORR,1)==gold):.3f} rmse={rmse(rep[np.arange(N),np.argmax(CORR,1)]):.4f}")

# ---- LightGBM 枝分類(旧特徴 + GR相関) ----
def tile(x): return np.take_along_axis(x,a,1).reshape(N,K,KT)
grc_m=tile(GRC).mean(2); grc2_m=tile(GRC2).mean(2); std_m=tile(STD).mean(2); spread_m=tile(DEL).std(2); vs=rep-STRD[:,None]
X0=np.concatenate([rep/30,grc_m/20,grc2_m/20,np.log1p(std_m),spread_m/30,vs/30],1)
Xg=np.stack([np.log1p(np.maximum(CTX[:,0],0)),CTX[:,1],np.log1p(np.maximum(CTX[:,2],0)),np.log1p(np.maximum(STRS,0))],1)
from lightgbm import LGBMClassifier
def run(X,tag):
    folds=list(GroupKFold(5).split(X,gold,groups=g)); om=np.zeros(N,np.int32); op=np.zeros((N,K),np.float32)
    for k,(tr,va) in enumerate(folds):
        sub=tr[::3]
        try: m=LGBMClassifier(n_estimators=400,num_leaves=63,learning_rate=0.05,subsample=0.8,colsample_bytree=0.8,reg_lambda=5.0,objective="multiclass",num_class=K,n_jobs=-1,verbose=-1,device_type="gpu").fit(X[sub],gold[sub])
        except Exception: m=LGBMClassifier(n_estimators=400,num_leaves=63,learning_rate=0.05,reg_lambda=5.0,objective="multiclass",num_class=K,n_jobs=-1,verbose=-1).fit(X[sub],gold[sub])
        p=m.predict_proba(X[va]); op[va]=p; om[va]=p.argmax(1)
    print(f"  [{tag}] acc={np.mean(om==gold):.3f} hard={rmse(rep[np.arange(N),om]):.4f} soft={rmse((op*rep).sum(1)):.4f}")
print("\n=== 反復3: GR波形相関を判別に追加 ===")
run(np.concatenate([X0,Xg],1).astype(np.float32),"旧特徴のみ")
run(np.concatenate([X0,Xg,CORR],1).astype(np.float32),"旧+GR相関")
print(f"  参考: 天井(oracle-mode)={np.sqrt(np.mean(np.min(np.abs(rep-y[:,None]),1)**2)):.4f}")
