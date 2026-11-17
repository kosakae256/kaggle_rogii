# -*- coding: utf-8 -*-
"""反復8: 選択/分類でなく「回帰」。LightGBMで y を直接回帰し、易しい行/硬い行で分解。
モードHMM(枝内平均で within-branch精度を捨てる)が取りこぼす易しい行の余地を測る。v47 artifact のみ。GPU軽。"""
import numpy as np, json
from pathlib import Path
from sklearn.model_selection import GroupKFold
from lightgbm import LGBMRegressor
ART=Path(r"C:\Users\kosaka256\Documents\rogii_claude")/"v47"/"artifacts_v47"
z=np.load(ART/"candidates_v47.npz"); DEL=z["DEL"]; y=z["y"]; g=z["wellcode"].astype(np.int64); STRD=z["STRD"]; STRS=z["STRS"]; GRC=z["GRC"]; GRC2=z["GRC2"]; STD=z["STD"]; CTX=z["ctx"]; N=len(y)
spread=np.percentile(DEL,90,1)-np.percentile(DEL,10,1); easy=spread<np.median(spread)
def rmse(p,mask): e=(p-y)[mask]; return float(np.sqrt(np.mean(e**2)))
print(f"易しい行={easy.mean()*100:.0f}%  易RMSE(mean候補)={rmse(DEL.mean(1),easy):.3f} 硬RMSE={rmse(DEL.mean(1),~easy):.3f}  全体={rmse(DEL.mean(1),np.ones(N,bool)):.3f}")
print(f"易 oracle={np.sqrt(np.mean(np.min(np.abs(DEL[easy]-y[easy,None]),1)**2)):.3f}  中央値候補 易={rmse(np.median(DEL,1),easy):.3f}")

# 特徴: ソート済み候補delta(21) + GRC(21) + std(21) + struct/spread/ctx
sd=np.sort(DEL,1)
X=np.concatenate([sd/30, np.sort(GRC,1)/20, np.log1p(np.sort(STD,1)),
                  (STRD/30)[:,None], (spread/30)[:,None], np.log1p(np.maximum(STRS,0))[:,None],
                  np.log1p(np.maximum(CTX[:,0:1],0)), CTX[:,1:2], np.log1p(np.maximum(CTX[:,2:3],0))],1).astype(np.float32)
folds=list(GroupKFold(5).split(X,y,groups=g)); oof=np.zeros(N,np.float32)
import time; t0=time.time()
for k,(tr,va) in enumerate(folds):
    sub=tr[::2]
    try: m=LGBMRegressor(n_estimators=600,num_leaves=127,learning_rate=0.04,subsample=0.8,colsample_bytree=0.8,reg_lambda=5.0,n_jobs=-1,verbose=-1,device_type="gpu").fit(X[sub],y[sub])
    except Exception: m=LGBMRegressor(n_estimators=600,num_leaves=127,learning_rate=0.04,reg_lambda=5.0,n_jobs=-1,verbose=-1).fit(X[sub],y[sub])
    oof[va]=m.predict(X[va]); print(f"  fold{k} ({time.time()-t0:.0f}s)",flush=True)
print("\n=== 反復8: LightGBM 直接回帰 ===")
print(f"  全体 RMSE  = {rmse(oof,np.ones(N,bool)):.4f}  (モードHMM soft=6.06, mean候補=7.14)")
print(f"  易しい行   = {rmse(oof,easy):.4f}  (mean候補 {rmse(DEL.mean(1),easy):.3f} / oracle {np.sqrt(np.mean(np.min(np.abs(DEL[easy]-y[easy,None]),1)**2)):.3f})")
print(f"  硬い行     = {rmse(oof,~easy):.4f}  (mean候補 {rmse(DEL.mean(1),~easy):.3f})")
np.save(ART/"oof_lgbm_reg.npy",oof)
