# -*- coding: utf-8 -*-
"""v48 反復2: PF不使用のグリッド・ベイズフィルタ(ヒストグラムfilter, 粒子なし)。
状態=struct からの偏差 δ の格子。各bin: GR尤度 L(δ)=exp(-((g-f(struct+δ))/σg)²/2) × struct prior N(δ;0,sig)。
遷移=平滑Gaussカーネル(dip連続性)。forward-backward で多峰posterior→ δ推定。
struct単体≈9.75 / PF≈8.1 が基準。多峰保持で struct を割れるか。
"""
import os, time, pickle, numpy as np, pandas as pd, torch
import torch.nn.functional as F
from pathlib import Path
P=Path(r"C:\Users\kosaka256\Documents\rogii_claude"); TRAIN=P/"rogii-wellbore-geology-prediction"/"train"
STRUCT=pickle.loads((P/"v38"/"artifacts_v38"/"struct_prior_train.pkl").read_bytes())
DEV="cuda" if torch.cuda.is_available() else "cpu"
NW=int(os.environ.get("NW","150")); STRIDE=int(os.environ.get("STRIDE","16"))
D=float(os.environ.get("D","45")); ND=int(os.environ.get("ND","181"))     # δ∈[-45,45] 181格子(0.5刻み)
SIGG=float(os.environ.get("SIGG","12"))     # GR尤度の分散
TAU=float(os.environ.get("TAU","1.2"))       # 遷移平滑(bin間でδが変わりうる幅)
GR_SMOOTH=25
grid=torch.linspace(-D,D,ND,device=DEV)      # δ格子
# 遷移カーネル(Gauss, 幅TAU格子)
_k=torch.arange(-ND//2,ND//2+1,device=DEV).float(); Tk=torch.exp(-0.5*(_k/max(TAU/(2*D/(ND-1)),1e-3))**2); Tk/=Tk.sum()

wells=sorted(STRUCT.keys())[:NW]
def load(wn):
    hw=pd.read_csv(TRAIN/f"{wn}__horizontal_well.csv",usecols=["GR","TVT_input","TVT"])
    tw=pd.read_csv(TRAIN/f"{wn}__typewell.csv",usecols=["TVT","GR"]).sort_values("TVT")
    ti=hw["TVT_input"].to_numpy(float); ev=~np.isfinite(ti); kn=np.isfinite(ti)
    if ev.sum()==0 or kn.sum()<10: return None
    g=hw["GR"].astype(float).interpolate(limit_direction="both")
    g=g.rolling(GR_SMOOTH,center=True,min_periods=1).mean().to_numpy()[ev]
    true=hw["TVT"].to_numpy(float)[ev]; sp=STRUCT.get(wn)
    if sp is None or len(sp["tvt"])!=ev.sum(): return None
    # self疑似typewell: 既知帯(TVT_input, GR)を TVTでbin平均
    gk=hw["GR"].astype(float).interpolate(limit_direction="both").rolling(GR_SMOOTH,center=True,min_periods=1).mean().to_numpy()[kn]
    tk=ti[kn]; order=np.argsort(tk); tk=tk[order]; gk=gk[order]
    # TVT重複を0.5ft binで平均
    if len(tk)>5:
        tb=np.round(tk/0.5)*0.5; ub,idx=np.unique(tb,return_inverse=True); sg_=np.zeros(len(ub)); cc=np.zeros(len(ub)); np.add.at(sg_,idx,gk); np.add.at(cc,idx,1); self_tvt=ub; self_gr=sg_/cc
    else: self_tvt=tk; self_gr=gk
    n=len(g); nb=(n+STRIDE-1)//STRIDE; bid=np.arange(n)//STRIDE
    def bm(a): o=np.zeros(nb); c=np.zeros(nb); np.add.at(o,bid,a); np.add.at(c,bid,1); return o/np.maximum(c,1)
    lastk=float(ti[kn][-1])
    # ★既知帯でGRキャリブレーション: g_known ≈ a·f(true_tvt_known)+b(horizontal と typewell の系統差)
    f_kn=np.interp(ti[kn], tw["TVT"].to_numpy(float), tw["GR"].to_numpy(float))
    gk2=hw["GR"].astype(float).interpolate(limit_direction="both").rolling(GR_SMOOTH,center=True,min_periods=1).mean().to_numpy()[kn]
    A=np.vstack([f_kn,np.ones_like(f_kn)]).T
    try: ca,cb=np.linalg.lstsq(A,gk2,rcond=None)[0]
    except Exception: ca,cb=1.0,0.0
    if not np.isfinite(ca) or ca<=0.2 or ca>5: ca,cb=1.0,0.0
    return dict(g=bm(g), st=bm(np.asarray(sp["tvt"],float)), sig=np.clip(bm(np.asarray(sp["sig"],float)),1,60),
               true=bm(true), tw_tvt=tw["TVT"].to_numpy(float), tw_gr=tw["GR"].to_numpy(float),
               self_tvt=self_tvt.astype(float), self_gr=self_gr.astype(float), lastk=lastk, ca=float(ca), cb=float(cb))
data=[d for d in (load(w) for w in wells) if d is not None]
print(f"wells {len(data)} device {DEV} grid {ND}({-D}..{D})")
st_all=np.concatenate([d["st"]-d["true"] for d in data]); print(f"[基準] struct単体(bin) RMSE={np.sqrt(np.mean(st_all**2)):.4f}")

def interp_np(z, xp, fp): return np.interp(z, xp, fp)

gnp=grid.cpu().numpy()
ANCH=float(os.environ.get("ANCH","0"))     # 境界(last_known)錨のσ。0で無効
_OFF=0.0                                    # prior中心オフセット(反復10で使用)
CAL=int(os.environ.get("CAL","0"))          # 既知帯GRキャリブレーション ON/OFF
ROBUST=float(os.environ.get("ROBUST","0"))  # 0=Gauss尤度, >0=Cauchy系(重い裾)
DETREND=int(os.environ.get("DETREND","0"))  # >0=長波長トレンド除去窓(高域=cyclicパターンで照合)
def run(SIGG,TAU,LPR,WSELF=0.0):
    dd=grid[:,None]-grid[None,:]; M=torch.exp(-0.5*(dd/TAU)**2); M=M/M.sum(0,keepdim=True).clamp_min(1e-30)  # 遷移[δ,δ']
    errs=[]; t0=time.time()
    for d in data:
        T=len(d["g"]); z=d["st"][:,None]+gnp[None,:]                          # [T,ND] 絶対TVT候補
        twg=d["tw_gr"]; gg=d["g"]
        if DETREND>0:
            twg=twg-pd.Series(twg).rolling(DETREND,center=True,min_periods=1).mean().to_numpy()
            gg=gg-pd.Series(gg).rolling(max(DETREND//STRIDE,3),center=True,min_periods=1).mean().to_numpy()
        eg=np.interp(z.ravel(), d["tw_tvt"], twg).reshape(T,ND)
        if CAL: eg=d["ca"]*eg+d["cb"]                                          # ★既知帯キャリブレーション適用
        rr=((gg[:,None]-eg)/SIGG)**2
        L=np.exp(-0.5*rr) if ROBUST==0 else (1.0/(1.0+rr))**ROBUST            # ROBUST>0=Cauchy系(重い裾)
        if WSELF>0 and len(d["self_tvt"])>5:                                  # ★self疑似typewell(既知帯)=混合
            smin,smax=d["self_tvt"][0],d["self_tvt"][-1]
            eg_s=np.interp(z.ravel(), d["self_tvt"], d["self_gr"]).reshape(T,ND)
            Ls=np.exp(-0.5*((d["g"][:,None]-eg_s)/(SIGG*1.5))**2)            # selfはsoftに
            inr=((z>=smin)&(z<=smax)).astype(float)                          # self範囲内
            L=np.where(inr>0, (1-WSELF)*L+WSELF*Ls, L)                        # 積でなく混合(潰さない)
        Pr=np.exp(-0.5*((gnp[None,:]-_OFF)/d["sig"][:,None])**2)               # struct prior(中心を_OFFずらせる)
        obs=torch.tensor((L*Pr**LPR),dtype=torch.float32,device=DEV); obs=obs/obs.sum(1,keepdim=True).clamp_min(1e-30)
        # ★境界固定: bin0 を last_known に強く錨(δ0≈lastk-struct0)。tight TAU で周期跨ぎを防ぐ
        if ANCH>0:
            d0=d["lastk"]-d["st"][0]; anc=torch.tensor(np.exp(-0.5*((gnp-d0)/ANCH)**2),dtype=torch.float32,device=DEV)
            obs[0]=obs[0]*anc; obs[0]=obs[0]/obs[0].sum().clamp_min(1e-30)
        a=torch.zeros(T,ND,device=DEV); a[0]=obs[0]; logZ=0.0
        for t in range(1,T):
            pred=M@a[t-1]; unn=pred.clamp_min(1e-30)*obs[t]; s=unn.sum(); logZ+=float(torch.log(s.clamp_min(1e-30))); a[t]=unn/s.clamp_min(1e-30)
        b=torch.zeros(T,ND,device=DEV); b[-1]=1.0/ND
        for t in range(T-2,-1,-1):
            msg=b[t+1]*obs[t+1]; pred=M@msg; b[t]=pred.clamp_min(1e-30); b[t]=b[t]/b[t].sum().clamp_min(1e-30)
        gam=a*b; gam=gam/gam.sum(1,keepdim=True).clamp_min(1e-30)
        dhat=(gam*grid[None,:]).sum(1).cpu().numpy()
        errs.append((d["st"]+dhat-d["true"], logZ/T))          # (err, 平均log-evidence)
    return errs, time.time()-t0

def run_rmse(SIGG,TAU,LPR,WSELF=0.0):
    errs,dt=run(SIGG,TAU,LPR,WSELF); e=np.concatenate([x[0] for x in errs]); return float(np.sqrt(np.mean(e**2))),dt

# ★反復10: prior中心を ±周期オフセット複数で走らせ、logZ最大の周期を坑井ごと選択
OFFS=[float(x) for x in os.environ.get("OFFS","-20,-13,-7,0,7,13,20").split(",")]
def run_multi(SIGG,TAU,LPR):
    global _OFF
    per=[]  # per offset: list of (err,logZ)
    for off in OFFS:
        _OFF=off; per.append(run(SIGG,TAU,LPR)[0])
    _OFF=0.0
    # 坑井ごと logZ最大のoffsetを選ぶ
    sel_err=[]; base_err=[]
    for wi in range(len(per[0])):
        lz=[per[oi][wi][1] for oi in range(len(OFFS))]; best=int(np.argmax(lz))
        sel_err.append(per[best][wi][0]); base_err.append(per[OFFS.index(0.0)][wi][0])
    se=np.concatenate(sel_err); be=np.concatenate(base_err)
    return float(np.sqrt(np.mean(se**2))), float(np.sqrt(np.mean(be**2)))

print(f"\n=== v48 全{len(data)}坑井: 本命config(struct/PF公平比較)===")
for sg in [4,5,6,7]:
    row=[]
    for tau in [0.3,0.35,0.45]:
        r,dt=run_rmse(sg,tau,0.3); row.append(f"TAU{tau}:{r:.3f}")
    print(f"  SIGG={sg}  "+"  ".join(row)+f"  ({dt:.0f}s/run)",flush=True)
