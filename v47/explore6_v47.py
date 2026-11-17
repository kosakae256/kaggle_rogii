# -*- coding: utf-8 -*-
"""反復6: 生GR残差パッチ[K=3ch,L]を小1D CNNで枝分類。線形相関(0.547)を非線形学習が超えるか。
硬い行のみ・サブサンプル・GPU軽量(他学習と並行)。v47 artifact + 公式CSV(GR)のみ。"""
import os, time, json, numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupKFold
import torch, torch.nn as nn
P=Path(r"C:\Users\kosaka256\Documents\rogii_claude"); ART=P/"v47"/"artifacts_v47"; TRAIN=P/"rogii-wellbore-geology-prediction"/"train"
z=np.load(ART/"candidates_v47.npz"); DEL=z["DEL"]; y=z["y"]; g=z["wellcode"].astype(np.int64); lk=z["lk"].astype(np.float64); N=len(y)
wells=json.loads((ART/"candidates_v47_meta.json").read_text(encoding="utf-8"))["wells"]
ids=np.load(ART/"ids_v47.npy",allow_pickle=True).astype(str); rn=pd.Series(ids).str.rsplit("_",n=1,expand=True)[1].astype(int).to_numpy()
K=3; KT=21//K; HALF=32; L=2*HALF
a=np.argsort(DEL,1); rep=np.take_along_axis(DEL,a,1).reshape(N,K,KT).mean(2); gold=np.argmin(np.abs(rep-y[:,None]),1)
spread=np.percentile(DEL,90,1)-np.percentile(DEL,10,1)
hard=spread>np.percentile(spread,55)                      # 硬い行(誤差集中)
rng=np.random.default_rng(0); sel=np.where(hard)[0]; sel=rng.choice(sel,min(400000,len(sel)),replace=False); selset=set(sel.tolist())
print(f"硬い行 {hard.sum()} → サブサンプル {len(sel)}  (L={L})")

# 残差パッチ抽出(坑井ごと)
Xp=np.zeros((len(sel),K,L),np.float32); lab=np.zeros(len(sel),np.int64); grp=np.zeros(len(sel),np.int64); gidx=np.zeros(len(sel),np.int64); fill=0; t0=time.time()
sel_by_w={}
for i in sel: sel_by_w.setdefault(int(g[i]),[]).append(i)
for wi,rows in sel_by_w.items():
    wn=wells[wi]; ri=np.where(g==wi)[0]; o=np.argsort(rn[ri]); ri=ri[o]; r=rn[ri]; n=len(ri)
    posmap={int(gr):j for j,gr in enumerate(ri)}
    try:
        hw=pd.read_csv(TRAIN/f"{wn}__horizontal_well.csv",usecols=["GR"]); tw=pd.read_csv(TRAIN/f"{wn}__typewell.csv",usecols=["TVT","GR"]).sort_values("TVT")
    except Exception: continue
    hgr=hw["GR"].astype(float).interpolate(limit_direction="both").to_numpy(); fb=float(np.nanmean(tw["GR"])); hgr=np.where(np.isfinite(hgr),hgr,fb)
    h=hgr[np.clip(r,0,len(hgr)-1)]; ttv=tw["TVT"].to_numpy(float); tgr=tw["GR"].to_numpy(float)
    resid=np.stack([h-np.interp(lk[ri]+rep[ri,k],ttv,tgr) for k in range(K)],0)   # [K,n]
    for gi in rows:
        j=posmap[int(gi)]; sl=slice(max(0,j-HALF),min(n,j+HALF)); pad=(HALF-(j-max(0,j-HALF)), HALF-(min(n,j+HALF)-j))
        patch=np.pad(resid[:,sl],((0,0),pad),mode="edge")[:, :L]
        patch=patch-patch.mean(1,keepdims=True)                                   # 各chゼロ平均(オフセット無視=波形勝負)
        Xp[fill]=patch; lab[fill]=gold[gi]; grp[fill]=wi; gidx[fill]=gi; fill+=1
Xp=Xp[:fill]; lab=lab[:fill]; grp=grp[:fill]; gidx=gidx[:fill]
print(f"パッチ抽出 {fill} ({time.time()-t0:.0f}s)")

# 小1D CNN(GPU軽量)
DEV="cuda" if torch.cuda.is_available() else "cpu"
try: torch.cuda.set_per_process_memory_fraction(0.35)
except Exception: pass
class CNN(nn.Module):
    def __init__(s):
        super().__init__()
        s.net=nn.Sequential(nn.Conv1d(K,32,7,padding=3),nn.GELU(),nn.Conv1d(32,32,5,padding=2),nn.GELU(),nn.MaxPool1d(2),
                            nn.Conv1d(32,48,5,padding=2),nn.GELU(),nn.AdaptiveAvgPool1d(1),nn.Flatten(),nn.Linear(48,K))
    def forward(s,x): return s.net(x)
def rmse_sel(predmode,idx): p=rep[idx,predmode]; e=p-y[idx]; return float(np.sqrt(np.mean(e**2)))
folds=list(GroupKFold(4).split(Xp,lab,groups=grp)); oofm=np.zeros(fill,np.int64); t0=time.time()
for k,(tr,va) in enumerate(folds):
    net=CNN().to(DEV); opt=torch.optim.AdamW(net.parameters(),lr=1e-3,weight_decay=1e-4); scaler=torch.cuda.amp.GradScaler(enabled=DEV=="cuda")
    Xt=torch.from_numpy(Xp[tr]); yt=torch.from_numpy(lab[tr]); B=1024
    for ep in range(8):
        net.train(); perm=torch.randperm(len(tr))
        for c0 in range(0,len(tr),B):
            bi=perm[c0:c0+B]; xb=Xt[bi].to(DEV); yb=yt[bi].to(DEV)
            with torch.autocast("cuda",enabled=DEV=="cuda"): out=net(xb); loss=nn.functional.cross_entropy(out,yb)
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
    net.eval()
    with torch.no_grad():
        pr=[]
        for c0 in range(0,len(va),4096):
            xb=torch.from_numpy(Xp[va[c0:c0+4096]]).to(DEV)
            with torch.autocast("cuda",enabled=DEV=="cuda"): pr.append(net(xb).argmax(1).cpu().numpy())
        oofm[va]=np.concatenate(pr)
    print(f"  fold{k} acc={np.mean(oofm[va]==lab[va]):.3f} ({time.time()-t0:.0f}s)",flush=True); del net,opt,scaler; torch.cuda.empty_cache() if DEV=="cuda" else None
acc=np.mean(oofm==lab)
print(f"\n=== 反復6: 生GR残差パッチ CNN 枝分類(硬い行) ===")
print(f"  CNN枝精度(硬い行OOF) = {acc:.3f}")
print(f"  同じ硬い行での参考: 最頻枝={np.max(np.bincount(lab))/fill:.3f}  ランダム=0.333")
print(f"  RMSE 硬い行: CNN選択={np.sqrt(np.mean((rep[gidx,oofm]-y[gidx])**2)):.4f}  mean候補={np.sqrt(np.mean((DEL[gidx].mean(1)-y[gidx])**2)):.4f}  oracle-mode={np.sqrt(np.mean(np.min(np.abs(rep[gidx]-y[gidx,None]),1)**2)):.4f}")
print(f"  比較: GR系LightGBM枝acc≈0.55(全行) / iter3線形相関追加=0.547")
