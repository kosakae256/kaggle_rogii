# -*- coding: utf-8 -*-
"""v48 グリッド・ベイズフィルタ(best: SIGG6/TAU0.35/LPR0.3)の予測を v47 の全行(parquet順)で生成し
特徴量として保存。gf_delta = 予測TVT - last_known(target同スケール), gf_std = posterior不確実性, gf_vs_struct。
train: ids_v47.npy(=parquet行順)に整列。test: submit側が gf_predict_rows() を呼ぶ。
出力: v48/artifacts_v48/gf_feature.npy [N,3] + gf_meta.json
"""
import os, time, json, pickle, numpy as np, pandas as pd, torch
from pathlib import Path
P=Path(r"C:\Users\kosaka256\Documents\rogii_claude"); TRAIN=P/"rogii-wellbore-geology-prediction"/"train"
OUT=P/"v48"/"artifacts_v48"; OUT.mkdir(parents=True,exist_ok=True)
DEV="cuda" if torch.cuda.is_available() else "cpu"
STRIDE=16; D=45.0; ND=181; GR_SMOOTH=25
SIGG=6.0; TAU=0.35; LPR=0.3            # ★v48 best config(773坑井 8.854)
gnp=np.linspace(-D,D,ND); grid=torch.tensor(gnp,dtype=torch.float32,device=DEV)

def gf_one(g_ev, struct, sig, tw_tvt, tw_gr, lastk):
    """eval行(rownum順)の GR/struct/typewell から grid filter 予測(z_row, std_row)を返す。"""
    n=len(g_ev); nb=(n+STRIDE-1)//STRIDE; bid=np.arange(n)//STRIDE
    def bm(a): o=np.zeros(nb); c=np.zeros(nb); np.add.at(o,bid,a); np.add.at(c,bid,1); return o/np.maximum(c,1)
    g=pd.Series(g_ev).rolling(GR_SMOOTH,center=True,min_periods=1).mean().to_numpy(); g=bm(g)
    st=bm(struct); sg=np.clip(bm(sig),1,60)
    z=st[:,None]+gnp[None,:]; eg=np.interp(z.ravel(),tw_tvt,tw_gr).reshape(nb,ND)
    L=np.exp(-0.5*((g[:,None]-eg)/SIGG)**2); Pr=np.exp(-0.5*(gnp[None,:]/sg[:,None])**2)
    obs=torch.tensor(L*Pr**LPR,dtype=torch.float32,device=DEV); obs=obs/obs.sum(1,keepdim=True).clamp_min(1e-30)
    dd=grid[:,None]-grid[None,:]; M=torch.exp(-0.5*(dd/TAU)**2); M=M/M.sum(0,keepdim=True).clamp_min(1e-30)
    a=torch.zeros(nb,ND,device=DEV); a[0]=obs[0]
    for t in range(1,nb):
        pred=M@a[t-1]; a[t]=(pred.clamp_min(1e-30)*obs[t]); a[t]=a[t]/a[t].sum().clamp_min(1e-30)
    b=torch.zeros(nb,ND,device=DEV); b[-1]=1.0/ND
    for t in range(nb-2,-1,-1):
        msg=b[t+1]*obs[t+1]; b[t]=(M@msg).clamp_min(1e-30); b[t]=b[t]/b[t].sum().clamp_min(1e-30)
    gam=a*b; gam=gam/gam.sum(1,keepdim=True).clamp_min(1e-30)
    dh=(gam*grid[None,:]).sum(1); var=(gam*(grid[None,:]-dh[:,None])**2).sum(1)
    dhat=dh.cpu().numpy(); dstd=np.sqrt(var.cpu().numpy())
    z_bin=st+dhat; d_from_st=dhat
    # bin→行 展開
    return z_bin[bid], dstd[bid], d_from_st[bid], st[bid]

def gf_predict_well(wn, rownums_sorted, struct_tvt, struct_sig, lastk, split_dir):
    """1坑井: CSV読込→eval行のgrid filter→ (gf_z_row, gf_std_row, gf_vs_struct_row)"""
    hw=pd.read_csv(Path(split_dir)/f"{wn}__horizontal_well.csv",usecols=["GR"])
    tw=pd.read_csv(Path(split_dir)/f"{wn}__typewell.csv",usecols=["TVT","GR"]).sort_values("TVT")
    ga=hw["GR"].astype(float).interpolate(limit_direction="both").to_numpy(); fb=float(np.nanmean(tw["GR"])) if np.isfinite(np.nanmean(tw["GR"])) else 0.0
    ga=np.where(np.isfinite(ga),ga,fb); g_ev=ga[np.clip(rownums_sorted,0,len(ga)-1)]
    z_row,std_row,dst_row,st_row=gf_one(g_ev,struct_tvt,struct_sig,tw["TVT"].to_numpy(float),tw["GR"].to_numpy(float),lastk)
    return z_row-lastk, std_row, dst_row     # gf_delta(target同スケール), gf_std, gf_vs_struct

if __name__=="__main__":
    ART=P/"v47"/"artifacts_v47"
    ids=np.load(ART/"ids_v47.npy",allow_pickle=True).astype(str)
    lk=np.load(ART/"candidates_v47.npz")["lk"].astype(np.float64)
    STRUCT=pickle.loads((P/"v38"/"artifacts_v38"/"struct_prior_train.pkl").read_bytes())
    pr=pd.Series(ids).str.rsplit("_",n=1,expand=True); wname=pr[0].to_numpy(); rn=pr[1].astype(int).to_numpy()
    N=len(ids); GF=np.full((N,3),np.nan,np.float32); t0=time.time()
    wells=pd.unique(wname); miss=0
    for wi,wn in enumerate(wells):
        idx=np.where(wname==wn)[0]; o=np.argsort(rn[idx]); idxs=idx[o]; rns=rn[idxs]
        sp=STRUCT.get(wn)
        if sp is None or len(sp["tvt"])!=len(idxs): miss+=1; continue
        lastk=float(lk[idxs][0])                       # last_known(candidates由来, 坑井内一定)
        try:
            gd,gs,gv=gf_predict_well(wn,rns,np.asarray(sp["tvt"],float),np.asarray(sp["sig"],float),lastk,TRAIN)
            GF[idxs,0]=gd; GF[idxs,1]=gs; GF[idxs,2]=gv
        except Exception as e:
            miss+=1
        if (wi+1)%150==0: print(f"  {wi+1}/{len(wells)} ({time.time()-t0:.0f}s)",flush=True)
    np.save(OUT/"gf_feature.npy",GF)
    (OUT/"gf_meta.json").write_text(json.dumps(dict(cols=["gf_delta","gf_std","gf_vs_struct"],
        config=dict(SIGG=SIGG,TAU=TAU,LPR=LPR,STRIDE=STRIDE),struct_miss=int(miss),n=int(N))))
    m=np.isfinite(GF[:,0]); y=np.load(ART/"candidates_v47.npz")["y"]
    print(f"gf_feature saved. 有効{m.mean()*100:.0f}%  gf_delta RMSE(vs target)={np.sqrt(np.mean((GF[m,0]-y[m])**2)):.4f}  ({time.time()-t0:.0f}s)")
