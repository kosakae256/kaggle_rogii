# -*- coding: utf-8 -*-
"""反復7: 既知帯(TVT_input帯)のGR信頼度/系統ズレが eval帯の枝判別や候補信頼度に効くか。
選択路線 最後の未検証軸。v47 artifact + 公式CSV。GPU不要。"""
import numpy as np, pandas as pd, json, time
from pathlib import Path
from sklearn.model_selection import GroupKFold
P=Path(r"C:\Users\kosaka256\Documents\rogii_claude"); ART=P/"v47"/"artifacts_v47"; TRAIN=P/"rogii-wellbore-geology-prediction"/"train"
z=np.load(ART/"candidates_v47.npz"); DEL=z["DEL"]; y=z["y"]; g=z["wellcode"].astype(np.int64); lk=z["lk"].astype(np.float64); STRD=z["STRD"]; GRC=z["GRC"]; GRC2=z["GRC2"]; STD=z["STD"]; STRS=z["STRS"]; CTX=z["ctx"]; N=len(y)
wells=json.loads((ART/"candidates_v47_meta.json").read_text(encoding="utf-8"))["wells"]; W=len(wells)
K=3; KT=21//K; a=np.argsort(DEL,1); rep=np.take_along_axis(DEL,a,1).reshape(N,K,KT).mean(2); gold=np.argmin(np.abs(rep-y[:,None]),1)
def rmse(p,idx=None): e=(p-y) if idx is None else (p-y)[idx]; return float(np.sqrt(np.mean(e**2)))
CN=json.loads((ART/"candidates_v47_meta.json").read_text(encoding="utf-8"))["cand_names"]; jpfA=CN.index("tw_pfA")

# 既知帯特徴(坑井ごと)
kfeat=np.zeros((W,4)); t0=time.time()
for wi,wn in enumerate(wells):
    try:
        hw=pd.read_csv(TRAIN/f"{wn}__horizontal_well.csv",usecols=["GR","TVT_input"]); tw=pd.read_csv(TRAIN/f"{wn}__typewell.csv",usecols=["TVT","GR"]).sort_values("TVT")
    except Exception: continue
    ti=hw["TVT_input"].to_numpy(float); kn=np.isfinite(ti);
    if kn.sum()<20: continue
    hgr=hw["GR"].astype(float).interpolate(limit_direction="both").to_numpy(); fb=float(np.nanmean(tw["GR"])); hgr=np.where(np.isfinite(hgr),hgr,fb)
    ttv=tw["TVT"].to_numpy(float); tgr=tw["GR"].to_numpy(float)
    exp=np.interp(ti[kn],ttv,tgr); resid=hgr[kn]-exp
    # 系統datumズレ: known GRを±40ftシフトして最良一致するオフセット(0なら整合)
    offs=np.arange(-40,41,2.0); errs=[np.std(hgr[kn]-np.interp(ti[kn]+o,ttv,tgr)) for o in offs]
    best_off=offs[int(np.argmin(errs))]
    kfeat[wi]=[np.std(resid), np.mean(resid), best_off, np.std(resid)/(np.std(hgr[kn])+1e-6)]
    if (wi+1)%200==0: print(f"  known {wi+1}/{W} ({time.time()-t0:.0f}s)",flush=True)
print(f"既知帯特徴 完了 ({time.time()-t0:.0f}s)  resid_std中央値={np.median(kfeat[:,0]):.2f} best_off中央{np.median(kfeat[:,2]):.1f}")

# (1) 坑井別: GR信頼度は GR候補 vs struct の優劣を予測するか
def wrmse(p,wi): idx=(g==wi); e=p[idx]-y[idx]; return float(np.sqrt(np.mean(e**2))) if idx.any() else np.nan
grw=np.array([wrmse(DEL[:,jpfA],wi) for wi in range(W)])         # pfA(GR)候補の坑井RMSE
stw=np.array([wrmse(STRD,wi) for wi in range(W)])
ok=(kfeat[:,0]>0)&np.isfinite(grw)&np.isfinite(stw)
grwin=(grw<stw).astype(float)
print(f"\n(1) 既知帯GR信頼度 vs eval帯でのGR候補優位:")
print(f"    corr(known_resid_std, pfA坑井RMSE)={np.corrcoef(kfeat[ok,0],grw[ok])[0,1]:.3f}  corr(known_resid_std, GR勝ち)={np.corrcoef(kfeat[ok,0],grwin[ok])[0,1]:.3f}")
print(f"    corr(best_off絶対, pfA坑井RMSE)={np.corrcoef(np.abs(kfeat[ok,2]),grw[ok])[0,1]:.3f}")

# (2) 既知帯特徴を足すと枝分類精度が上がるか
from lightgbm import LGBMClassifier
def tile(x): return np.take_along_axis(x,a,1).reshape(N,K,KT)
grc_m=tile(GRC).mean(2); grc2_m=tile(GRC2).mean(2); std_m=tile(STD).mean(2); spread_m=tile(DEL).std(2); vs=rep-STRD[:,None]
X0=np.concatenate([rep/30,grc_m/20,grc2_m/20,np.log1p(std_m),spread_m/30,vs/30,
                   np.stack([np.log1p(np.maximum(CTX[:,0],0)),CTX[:,1],np.log1p(np.maximum(STRS,0))],1)],1)
Kf=kfeat[g]                                                     # 行に展開
def run(X,tag):
    folds=list(GroupKFold(5).split(X,gold,groups=g)); om=np.zeros(N,np.int32); op=np.zeros((N,K),np.float32)
    for k,(tr,va) in enumerate(folds):
        sub=tr[::3]
        try: m=LGBMClassifier(n_estimators=400,num_leaves=63,learning_rate=0.05,reg_lambda=5.0,objective="multiclass",num_class=K,n_jobs=-1,verbose=-1,device_type="gpu").fit(X[sub],gold[sub])
        except Exception: m=LGBMClassifier(n_estimators=400,num_leaves=63,learning_rate=0.05,reg_lambda=5.0,objective="multiclass",num_class=K,n_jobs=-1,verbose=-1).fit(X[sub],gold[sub])
        p=m.predict_proba(X[va]); op[va]=p; om[va]=p.argmax(1)
    print(f"    [{tag}] 枝acc={np.mean(om==gold):.3f}  soft_rmse={rmse((op*rep).sum(1)):.4f}")
print("\n(2) 既知帯特徴を枝分類に追加:")
run(X0.astype(np.float32),"旧特徴のみ")
run(np.concatenate([X0,Kf],1).astype(np.float32),"旧+既知帯")
print(f"    参考: 天井(oracle-mode)={rmse(rep[np.arange(N),gold]):.4f}")
