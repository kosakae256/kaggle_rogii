# -*- coding: utf-8 -*-
"""v48 反復1: PF不使用の軌跡最適化。
TVT軌跡 z(t) を勾配降下で最適化: E(z)= GRミスフィット(robust) + 平滑(2次差分) + struct事前分布 + last_known錨。
struct が datum を与え、GR が局所補正。粒子なし・微分可能・GPU・軽量。
struct単体≈9.7 / PF≈8.1 が比較基準。~8.0台なら PF-free 路線に見込み。
"""
import os, glob, time, pickle, numpy as np, pandas as pd, torch
from pathlib import Path
P=Path(r"C:\Users\kosaka256\Documents\rogii_claude"); TRAIN=P/"rogii-wellbore-geology-prediction"/"train"
STRUCT=pickle.loads((P/"v38"/"artifacts_v38"/"struct_prior_train.pkl").read_bytes())
DEV="cuda" if torch.cuda.is_available() else "cpu"
NW=int(os.environ.get("NW","150")); STEPS=int(os.environ.get("STEPS","300"))
LAM_SM=float(os.environ.get("LSM","3.0")); LAM_PR=float(os.environ.get("LPR","1.0")); HUBER=float(os.environ.get("HUB","8.0"))
GR_SMOOTH=25

wells=sorted(STRUCT.keys())[:NW]
def load(wn):
    hw=pd.read_csv(TRAIN/f"{wn}__horizontal_well.csv",usecols=["GR","TVT_input","TVT"])
    tw=pd.read_csv(TRAIN/f"{wn}__typewell.csv",usecols=["TVT","GR"]).sort_values("TVT")
    ti=hw["TVT_input"].to_numpy(float); ev=~np.isfinite(ti); kn=np.isfinite(ti)
    if ev.sum()==0 or kn.sum()<10: return None
    g=hw["GR"].astype(float).interpolate(limit_direction="both").to_numpy()
    g=pd.Series(g).rolling(GR_SMOOTH,center=True,min_periods=1).mean().to_numpy()[ev]
    true=hw["TVT"].to_numpy(float)[ev]; lastk=float(ti[kn][-1])
    sp=STRUCT.get(wn)
    if sp is None or len(sp["tvt"])!=ev.sum(): return None
    return dict(g=g, true=true, lastk=lastk, st=np.asarray(sp["tvt"],float), sig=np.asarray(sp["sig"],float),
               tw_tvt=tw["TVT"].to_numpy(float), tw_gr=tw["GR"].to_numpy(float))
data=[d for d in (load(w) for w in wells) if d is not None]
print(f"wells {len(data)}  device {DEV}")

def interp1d_b(z, xp, fp, mask_g):
    # z:[B,T]  xp:[B,G] (昇順)  fp:[B,G]  → f(z) 線形補間(微分可能)
    idx=torch.searchsorted(xp, z.clamp(xp[:, :1], xp[:, -1:]))
    idx=idx.clamp(1, xp.shape[1]-1)
    x0=torch.gather(xp,1,idx-1); x1=torch.gather(xp,1,idx); y0=torch.gather(fp,1,idx-1); y1=torch.gather(fp,1,idx)
    t=((z-x0)/(x1-x0+1e-6)).clamp(0,1); return y0+t*(y1-y0)

def run(LAM_SM,LAM_PR,HUBER):
    errs=[]; t0=time.time()
    B=32
    for c0 in range(0,len(data),B):
        sub=data[c0:c0+B]; b=len(sub)
        T=max(len(d["g"]) for d in sub); G=max(len(d["tw_tvt"]) for d in sub)
        g=torch.zeros(b,T); mask=torch.zeros(b,T); st=torch.zeros(b,T); sig=torch.ones(b,T); true=torch.zeros(b,T); lastk=torch.zeros(b,1)
        xp=torch.zeros(b,G); fp=torch.zeros(b,G)
        for i,d in enumerate(sub):
            n=len(d["g"]); g[i,:n]=torch.tensor(d["g"]); mask[i,:n]=1; st[i,:n]=torch.tensor(d["st"]); sig[i,:n]=torch.tensor(np.clip(d["sig"],1,60)); true[i,:n]=torch.tensor(d["true"]); lastk[i,0]=d["lastk"]
            gg=len(d["tw_tvt"]); xp[i,:gg]=torch.tensor(d["tw_tvt"]); fp[i,:gg]=torch.tensor(d["tw_gr"])
            if gg<G: xp[i,gg:]=d["tw_tvt"][-1]+torch.arange(1,G-gg+1).float()*1.0; fp[i,gg:]=d["tw_gr"][-1]
        g,mask,st,sig,true,lastk,xp,fp=[t.to(DEV) for t in (g,mask,st,sig,true,lastk,xp,fp)]
        delta=torch.zeros(b,T,device=DEV,requires_grad=True)
        opt=torch.optim.Adam([delta],lr=0.5)
        for step in range(STEPS):
            z=st+delta
            eg=interp1d_b(z,xp,fp,None)
            r=(g-eg)/HUBER
            data_t=(torch.sqrt(1+r*r)-1)*HUBER    # pseudo-Huber
            d2=z[:,2:]-2*z[:,1:-1]+z[:,:-2]
            sm=torch.zeros_like(z); sm[:,2:]=d2*d2
            pr=(delta/sig)**2
            anch=((z[:,:1]-lastk)**2)*0.5
            loss=((data_t+LAM_SM*sm+LAM_PR*pr)*mask).sum()/mask.sum()+anch.mean()
            opt.zero_grad(); loss.backward(); opt.step()
        z=(st+delta).detach()
        e=((z-true)*mask).cpu().numpy(); m=mask.cpu().numpy().astype(bool)
        errs.append(e[m])
    e=np.concatenate(errs); return float(np.sqrt(np.mean(e**2))), time.time()-t0

# struct単体 と hold の基準
st_all=np.concatenate([d["st"]-d["true"] for d in data]); print(f"[基準] struct単体 RMSE={np.sqrt(np.mean(st_all**2)):.4f}")
hold_all=np.concatenate([d["lastk"]-d["true"] for d in data]); print(f"[基準] last_known hold RMSE={np.sqrt(np.mean(hold_all**2)):.4f}")
print("\n=== v48 反復1: 軌跡最適化(struct prior) ===")
for lsm in [1.0,3.0,10.0]:
    for lpr in [0.3,1.0,3.0]:
        r,dt=run(lsm,lpr,HUBER); print(f"  LAM_SM={lsm:4.1f} LAM_PR={lpr:4.1f}: RMSE={r:.4f} ({dt:.0f}s)",flush=True)
