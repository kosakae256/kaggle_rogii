# -*- coding: utf-8 -*-
"""v48 反復7: PF不使用の 2D グリッド・ベイズフィルタ(状態=δ位置 × rate, momentum付き)。
遷移: δ←δ+rate(dip連続=物理), rate←rate+微小noise。emission=GR尤度, prior=struct。forward-backward。
momentumで一定dipを追跡・外挿→PF(8.1)に迫る & 周期をまたがず自己相似を抑える狙い。
"""
import os, time, pickle, numpy as np, pandas as pd, torch
import torch.nn.functional as F
from pathlib import Path
P=Path(r"C:\Users\kosaka256\Documents\rogii_claude"); TRAIN=P/"rogii-wellbore-geology-prediction"/"train"
STRUCT=pickle.loads((P/"v38"/"artifacts_v38"/"struct_prior_train.pkl").read_bytes())
DEV="cuda" if torch.cuda.is_available() else "cpu"
NW=int(os.environ.get("NW","150")); STRIDE=16; GR_SMOOTH=25
DD=float(os.environ.get("DD","0.5"))          # δ格子刻み(ft)
D=float(os.environ.get("D","30"))              # δ範囲 ±D
ND=int(round(2*D/DD))+1
VB=int(os.environ.get("VB","6"))               # rate範囲 ±VB (δ-bin/bin) = ±VB*DD ft/bin
NV=2*VB+1
gnp=np.linspace(-D,D,ND); grid=torch.tensor(gnp,dtype=torch.float32,device=DEV)
vvals=(np.arange(NV)-VB)                        # rate(δ-bin単位)
print(f"grid δ:{ND}({-D}..{D},{DD}ft)  rate:{NV}(±{VB*DD}ft/bin)  dev {DEV}")

wells=sorted(STRUCT.keys())[:NW]
def load(wn):
    hw=pd.read_csv(TRAIN/f"{wn}__horizontal_well.csv",usecols=["GR","TVT_input","TVT"]); tw=pd.read_csv(TRAIN/f"{wn}__typewell.csv",usecols=["TVT","GR"]).sort_values("TVT")
    ti=hw["TVT_input"].to_numpy(float); ev=~np.isfinite(ti); kn=np.isfinite(ti)
    if ev.sum()==0 or kn.sum()<10: return None
    g=hw["GR"].astype(float).interpolate(limit_direction="both").rolling(GR_SMOOTH,center=True,min_periods=1).mean().to_numpy()[ev]
    true=hw["TVT"].to_numpy(float)[ev]; sp=STRUCT.get(wn)
    if sp is None or len(sp["tvt"])!=ev.sum(): return None
    n=len(g); nb=(n+STRIDE-1)//STRIDE; bid=np.arange(n)//STRIDE
    def bm(a): o=np.zeros(nb); c=np.zeros(nb); np.add.at(o,bid,a); np.add.at(c,bid,1); return o/np.maximum(c,1)
    return dict(g=bm(g), st=bm(np.asarray(sp["tvt"],float)), sig=np.clip(bm(np.asarray(sp["sig"],float)),1,60),
               true=bm(true), tw_tvt=tw["TVT"].to_numpy(float), tw_gr=tw["GR"].to_numpy(float), lastk=float(ti[kn][-1]))
data=[d for d in (load(w) for w in wells) if d is not None]
print(f"wells {len(data)}")
print(f"[基準] struct単体 RMSE={np.sqrt(np.mean(np.concatenate([d['st']-d['true'] for d in data])**2)):.4f}")

# 動き: 各rate iv は δを (iv-VB) bin シフト。gather index を事前計算
jj=torch.arange(ND,device=DEV)
shift_idx=torch.stack([ (jj-(iv-VB)).clamp(0,ND-1) for iv in range(NV)],0)   # [NV,ND]
shift_idx_bwd=torch.stack([ (jj+(iv-VB)).clamp(0,ND-1) for iv in range(NV)],0)

def gauss1d(nbins,sig):
    x=torch.arange(nbins,device=DEV).float()-nbins//2; k=torch.exp(-0.5*(x/max(sig,1e-3))**2); return k/k.sum()

def run(SIGG,LPR,VN,SRATE,LRATE):
    # VN=rate過程noise(格子), SRATE... prior_v幅, LRATE=rate prior重み
    kv=gauss1d(NV, VN)                                          # rate遷移noiseカーネル(1D over v)
    pv=torch.exp(-0.5*((torch.tensor(vvals,device=DEV).float())/max(SRATE,1e-3))**2); pv=pv/pv.sum()  # rate prior(小dip favor)
    errs=[]; t0=time.time()
    for d in data:
        T=len(d["g"]); z=d["st"][:,None]+gnp[None,:]
        eg=np.interp(z.ravel(), d["tw_tvt"], d["tw_gr"]).reshape(T,ND)
        L=torch.tensor(np.exp(-0.5*((d["g"][:,None]-eg)/SIGG)**2),dtype=torch.float32,device=DEV)   # [T,ND]
        Pr=torch.tensor(np.exp(-0.5*(gnp[None,:]/d["sig"][:,None])**2),dtype=torch.float32,device=DEV)  # [T,ND]
        em=(L*Pr**LPR)                                          # [T,ND] emission(δ)
        pvL=(pv**LRATE).view(NV,1)
        def obs(t): b=em[t][None,:]*pvL; return b/b.sum().clamp_min(1e-30)    # [NV,ND]
        # forward
        A=[obs(0)]
        for t in range(1,T):
            B=torch.gather(A[-1],1,shift_idx)                  # δ←δ+rate 動き [NV,ND]
            B=F.conv1d(B.T.unsqueeze(1),kv.view(1,1,-1),padding=NV//2).squeeze(1).T[:NV]  # rate noise (over v)
            B=B.clamp_min(1e-30); B=B*obs(t); B=B/B.sum().clamp_min(1e-30); A.append(B)
        # backward
        Bk=[None]*T; Bk[-1]=torch.full((NV,ND),1.0/(NV*ND),device=DEV)
        for t in range(T-2,-1,-1):
            m=Bk[t+1]*obs(t+1)
            m=F.conv1d(m.T.unsqueeze(1),kv.view(1,1,-1),padding=NV//2).squeeze(1).T[:NV]
            m=torch.gather(m,1,shift_idx_bwd)
            m=m.clamp_min(1e-30); Bk[t]=m/m.sum().clamp_min(1e-30)
        dhat=[]
        for t in range(T):
            g2=(A[t]*Bk[t]); g2=g2/g2.sum().clamp_min(1e-30); md=g2.sum(0); dhat.append(float((md*grid).sum()))
        errs.append(d["st"]+np.array(dhat)-d["true"])
    e=np.concatenate(errs); return float(np.sqrt(np.mean(e**2))), time.time()-t0

print("\n=== v48 反復16: 2D momentum(rate prior緩め=rate持続を許す)===")
print("  1D best(同150坑井)≈8.5 / PF≈8.1")
for vn in [0.3,0.6]:                    # rate過程noise小=momentum保持
    for srate in [4.0,8.0,20.0]:        # rate prior緩め=非ゼロrate許容
        for lr in [1.0,0.3]:
            r,dt=run(SIGG=4,LPR=0.3,VN=vn,SRATE=srate,LRATE=lr); print(f"  VN={vn} SRATE={srate} LRATE={lr}: RMSE={r:.4f} ({dt:.0f}s)",flush=True)
