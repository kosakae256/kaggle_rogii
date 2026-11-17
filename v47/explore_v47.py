# -*- coding: utf-8 -*-
"""v47 方法探索ループ。各反復=1手法を軽く実装・検証。v47 artifact のみ参照。
反復1: LightGBM で datum枝(K=3)を分類できるか(証拠の判別力の切り分け)。
"""
import os, json, time, numpy as np
from pathlib import Path
from sklearn.model_selection import GroupKFold
ART=Path(r"C:\Users\kosaka256\Documents\rogii_claude")/"v47"/"artifacts_v47"
z=np.load(ART/"candidates_v47.npz")
DEL=z["DEL"]; y=z["y"]; g=z["wellcode"].astype(np.int64); N=len(y)
STD=z["STD"]; GRC=z["GRC"]; GRC2=z["GRC2"]; GRM=z["GRM"]; STRD=z["STRD"]; STRS=z["STRS"]; CTX=z["ctx"]
K=3; KT=21//K
print(f"N={N}  mean候補RMSE={np.sqrt(np.mean((DEL.mean(1)-y)**2)):.4f}")

# ---- 行レベルのモード(枝)代表・証拠 ----
a=np.argsort(DEL,1)
def tile(arr): return np.take_along_axis(arr,a,1).reshape(N,K,KT)
rep=tile(DEL).mean(2)                 # [N,K] 枝代表delta
grc_m=tile(GRC).mean(2); grc2_m=tile(GRC2).mean(2); grm_m=tile(GRM).mean(2)
std_m=tile(STD).mean(2); spread_m=tile(DEL).std(2)
vs_struct=rep-STRD[:,None]
gold=np.argmin(np.abs(rep-y[:,None]),1).astype(np.int32)   # oracle枝
orc_mode=float(np.sqrt(np.mean(np.min(np.abs(rep-y[:,None]),1)**2)))
print(f"K={K} oracle-mode(天井)={orc_mode:.4f}  gold枝分布={np.bincount(gold)/N}")

# ---- 特徴(枝ごと + 大域) ----
feats=[rep/30, grc_m/20, grc2_m/20, grm_m/20, np.log1p(std_m), spread_m/30, vs_struct/30]
X=np.concatenate([f for f in feats],1)   # [N,3*7=21]
Xg=np.stack([np.log1p(np.maximum(CTX[:,0],0)), CTX[:,1], np.log1p(np.maximum(CTX[:,2],0)),
             np.log1p(np.maximum(STRS,0)), (np.percentile(DEL,90,1)-np.percentile(DEL,10,1))/30],1)
X=np.concatenate([X,Xg],1).astype(np.float32)
print("特徴次元",X.shape[1])

# ---- ゼロ学習ベースライン(参考) ----
def rmse(pred): e=pred-y; return float(np.sqrt(np.mean(e*e)))
# GR尤度でcandidate加重(物理的soft)
for tau in [5,10,20]:
    w=np.exp(-GRC/tau); w/=w.sum(1,keepdims=True); print(f"  [zero] GR尤度加重 tau={tau}: {rmse((w*DEL).sum(1)):.4f}")
print(f"  [zero] argmin-GRC枝: {rmse(rep[np.arange(N),np.argmin(grc_m,1)]):.4f}   struct単体: {rmse(STRD):.4f}")

# ---- LightGBM 枝分類(5-fold OOF) ----
from lightgbm import LGBMClassifier
folds=list(GroupKFold(5).split(X,gold,groups=g))
oof_mode=np.zeros(N,np.int32); oof_proba=np.zeros((N,K),np.float32)
t0=time.time()
for k,(tr,va) in enumerate(folds):
    # 学習は間引き(高速)、予測は全va
    sub=tr[::3]
    m=LGBMClassifier(n_estimators=400,num_leaves=63,learning_rate=0.05,subsample=0.8,colsample_bytree=0.8,
                     reg_lambda=5.0,objective="multiclass",num_class=K,n_jobs=-1,verbose=-1,device_type="gpu")
    try: m.fit(X[sub],gold[sub])
    except Exception: m=LGBMClassifier(n_estimators=400,num_leaves=63,learning_rate=0.05,subsample=0.8,colsample_bytree=0.8,reg_lambda=5.0,objective="multiclass",num_class=K,n_jobs=-1,verbose=-1).fit(X[sub],gold[sub])
    p=m.predict_proba(X[va]); oof_proba[va]=p; oof_mode[va]=p.argmax(1)
    print(f"  fold{k} acc={np.mean(oof_mode[va]==gold[va]):.3f} ({time.time()-t0:.0f}s)",flush=True)
acc=np.mean(oof_mode==gold)
pred_hard=rep[np.arange(N),oof_mode]                       # 分類→枝代表(ハード)
pred_soft=(oof_proba*rep).sum(1)                            # 確率加重(ソフト)
print("\n=== 反復1: LightGBM 枝分類 ===")
print(f"  枝分類精度(OOF)         = {acc:.3f}  (ランダム=0.33, 最頻={np.max(np.bincount(gold))/N:.3f})")
print(f"  RMSE 分類→枝(ハード)    = {rmse(pred_hard):.4f}")
print(f"  RMSE 確率加重(ソフト)   = {rmse(pred_soft):.4f}")
print(f"  参考: mean候補={np.sqrt(np.mean((DEL.mean(1)-y)**2)):.4f}  soft床≈6.4  天井(oracle-mode)={orc_mode:.4f}")
np.save(ART/"oof_lgbm_mode.npy",oof_mode); np.save(ART/"oof_lgbm_proba.npy",oof_proba)
