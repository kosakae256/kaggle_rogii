# -*- coding: utf-8 -*-
"""GR選択失敗の原因分解: (a)ノイズ obs_gr≠sim_gr か (b)datum縮退 真位置でも候補がGR等価か。
   hengck23の分岐点を実測。ノイズ主体ならノイズモデリングscorerが効く/縮退主体なら効かない。155井。"""
import os,sys,io,pickle
try: sys.stdout=io.open(1,"w",encoding="utf-8",closefd=False)
except: pass
import numpy as np,pandas as pd
from pathlib import Path
P=Path(r"C:\Users\kosaka256\Documents\rogii_claude");TR=P/"rogii-wellbore-geology-prediction"/"train"
A=P/"v64"/"artifacts_v64";AV=P/"v89"/"artifacts_v89"
N=5000;K=4;SLOPE=55/1000.;DEV=25/1000.;DIPSP=20/1000.;CONT=4.0;rng=np.random.default_rng(0)
DIST=dict(np.load(A/"v89_dist_f0_dist.npz"));PF=dict(np.load(AV/"pf_center_v89.npz"))
def gen(md,z,tl,zl,tmn,tmx,pmd,pch):
    L=len(md);mr=md-md[0];dZ=z-zl
    pdip=float(np.clip(np.polyfit(pmd,pch,1)[0],-SLOPE,SLOPE)) if(len(pmd)>=5 and(pmd[-1]-pmd[0])>1) else 0.0
    xk=np.linspace(0.,mr[-1]+1e-6,K);seg=np.diff(xk)
    dn=pdip+rng.uniform(-DIPSP,DIPSP,size=(N,1))
    dyk=np.concatenate([np.zeros((N,1)),np.cumsum(rng.normal(0,DEV,(N,K-1))*seg[None,:],axis=1)],axis=1)
    dev=np.empty((N,L))
    for i in range(N):dev[i]=np.interp(mr,xk,dyk[i])
    cand=tl+dn*mr[None,:]+dev-dZ[None,:]
    ok=((cand>=tmn-5)&(cand<=tmx+5)).all(axis=1);ok&=(np.abs(cand[:,:min(30,L)]-tl)<CONT).all(axis=1)
    return cand[ok] if ok.any() else None
noise_rms=[];true_rank=[];degen_cnt=[];gap_oracle_argmin=[];prefix_noise=[]
for wn in [w for w in DIST if w in PF]:
    try:hw=pd.read_csv(TR/f"{wn}__horizontal_well.csv");tw=pd.read_csv(TR/f"{wn}__typewell.csv")
    except:continue
    kn=hw["TVT_input"].notna().to_numpy()
    if kn.sum()<10 or (~kn).sum()<30:continue
    hps=int(np.where(kn)[0].max());ev=slice(hps+1,len(hw))
    z=hw["Z"].to_numpy(float);md=hw["MD"].to_numpy(float);tvt=hw["TVT"].to_numpy(float)
    ti=hw["TVT_input"].to_numpy(float);gr=hw["GR"].to_numpy(float)
    tt=tw["TVT"].to_numpy(float);tg=tw["GR"].to_numpy(float);m=np.isfinite(tt)&np.isfinite(tg);tt,tg=tt[m],tg[m]
    if len(tt)<8:continue
    o=np.argsort(tt);tt,tg=tt[o],tg[o];tl=float(ti[hps]);zl=float(z[hps])
    ki=np.where(kn)[0];ki=ki[ki>=max(0,hps-80)]
    cand=gen(md[ev],z[ev],tl,zl,float(tt.min()),float(tt.max()),md[ki]-md[hps],(ti[ki]-tl)+(z[ki]-zl))
    if cand is None:continue
    true=tvt[ev];fin=np.isfinite(true)
    obs=pd.Series(gr[ev]).interpolate(limit_direction="both").fillna(float(np.nanmean(tg))).rolling(5,center=True,min_periods=1).mean().to_numpy()
    # (a) ノイズ: 真TVTでのsim_gr vs obs_gr(=GR照合の下限)
    sim_true=np.interp(true,tt,tg)
    nz=(obs-sim_true)[fin]; noise_rms.append(float(np.sqrt(np.mean(nz**2))))
    # prefixノイズ(既知帯: obsとtypewell照合の実測ノイズ=学習可能なノイズ源)
    kt=ti[ki]; kg=gr[ki]; ksim=np.interp(kt,tt,tg); pn=kg-ksim; pn=pn[np.isfinite(pn)]
    if len(pn): prefix_noise.append(float(np.sqrt(np.mean(pn**2))))
    # 全候補のGR残差
    gres=np.sqrt(np.mean((np.interp(cand,tt,tg)-obs[None,:])**2,axis=1))
    tvt_err=np.sqrt(np.mean((cand-true[None,:])**2,axis=1))
    oi=int(np.argmin(tvt_err))                     # oracle候補
    res_oracle=gres[oi]; res_min=gres.min()
    gap_oracle_argmin.append(res_oracle-res_min)   # oracleのGR残差が最小からどれだけ上か
    # oracleのGR残差の順位(0=GR的にも最良)
    true_rank.append(float((gres<res_oracle).mean()))
    # 縮退: oracleと同等GR(<res_oracle*1.05)なのにTVT>10ft離れた"デコイ"の数
    decoy=(gres<res_oracle*1.05)&(tvt_err>10)
    degen_cnt.append(int(decoy.sum()))
r=lambda a:float(np.median(a))
print(f"[GR選択失敗の原因分解 {len(noise_rms)}井]")
print(f"  (a)ノイズ obs_gr−sim_gr@真TVT  RMS中央 = {r(noise_rms):.2f} (typewell照合の下限誤差)")
print(f"      prefix既知帯の実測ノイズ RMS中央 = {r(prefix_noise):.2f} (学習可能なノイズ源)")
print(f"  (b)縮退: oracleのGR残差の順位(0=GR最良,1=GR最悪) 中央 = {r(true_rank):.3f}")
print(f"      oracle残差−最小残差 の中央 = {r(gap_oracle_argmin):.3f} (0近=真位置がGR的にも最良)")
print(f"      GR等価(<oracle*1.05)かつTVT>10ft離れたデコイ数 中央 = {r(degen_cnt):.0f}/{N}")
print(f"\n読み: 順位が0近&gap小 なら真位置はGR最良だがデコイが等価=縮退(ノイズモデリング効かず)。")
print(f"      順位が高い(0.3+)なら真位置がGR最良でない=ノイズ/形不一致=ノイズモデリングscorerが効く余地。")
