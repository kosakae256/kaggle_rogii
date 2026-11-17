# -*- coding: utf-8 -*-
"""GRフリー物理脚: prefix dip外挿(錨)のみ, GR一切不使用。typewell範囲clipで drift抑制。
   複数錨バリアント比較 + v90とのfield-out固定weight blend。目標: RMSE~10 & 高脱相関。全773井。"""
import sys,io,pickle,glob,time
try: sys.stdout=io.open(1,"w",encoding="utf-8",closefd=False)
except: pass
import numpy as np,pandas as pd,pyarrow.parquet as pq
from pathlib import Path
from sklearn.cluster import KMeans
P=Path(r"C:\Users\kosaka256\Documents\rogii_claude");TR=P/"rogii-wellbore-geology-prediction"/"train"
A90=P/"v90"/"artifacts_v90";SLOPE=55/1000.;W_NN=0.62
tb=pq.read_table(A90/"data"/"train.parquet",columns=["well","target"])
well=tb.column("well").to_numpy(zero_copy_only=False);y=tb.column("target").to_numpy(zero_copy_only=False).astype(float)
gbm=np.load(A90/"gbm_v41_oof.npy").astype(float)[:len(y)];nnz=np.load(A90/"nn_oof_combo3.npz")
nn=np.mean([nnz[k] for k in nnz.files],axis=0)[:len(y)];v90err=(1-W_NN)*gbm+W_NN*nn-y
ibw={}
for i,w in enumerate(well): ibw.setdefault(w,[]).append(i)
for w in ibw: ibw[w]=np.array(ibw[w])
out={};t0=time.time();c=0
for f in sorted(glob.glob(str(TR/"*__horizontal_well.csv"))):
    wn=Path(f).name.split("__")[0]
    if wn not in ibw: continue
    hw=pd.read_csv(f)
    try: tw=pd.read_csv(TR/f"{wn}__typewell.csv")
    except: continue
    kn=hw["TVT_input"].notna().to_numpy()
    if kn.sum()<10 or (~kn).sum()<30: continue
    hps=int(np.where(kn)[0].max());ev=slice(hps+1,len(hw))
    z=hw["Z"].to_numpy(float);md=hw["MD"].to_numpy(float);tvt=hw["TVT"].to_numpy(float);ti=hw["TVT_input"].to_numpy(float)
    tt=tw["TVT"].to_numpy(float);tg=tw["GR"].to_numpy(float);m=np.isfinite(tt)&np.isfinite(tg);tt=tt[m]
    if len(tt)<8: continue
    tl=float(ti[hps]);zl=float(z[hps]);tmn,tmx=float(tt.min()),float(tt.max())
    ki=np.where(kn)[0];ki=ki[ki>=max(0,hps-80)]
    pmd=md[ki]-md[hps];pch=(ti[ki]-tl)+(z[ki]-zl)
    pdip=float(np.clip(np.polyfit(pmd,pch,1)[0],-SLOPE,SLOPE)) if(len(pmd)>=5 and(pmd[-1]-pmd[0])>1) else 0.0
    mdrel=md[ev]-md[hps];dZ=z[ev]-zl;true=tvt[ev];fin=np.isfinite(true)
    base=tl+pdip*mdrel-dZ                          # 純prefix dip外挿(GRフリー)
    base_clip=np.clip(base,tmn-3,tmx+3)            # typewell範囲clip
    flat=tl-dZ                                     # dip0(水平面) = 最保守錨
    v90e=v90err[ibw[wn]];n=min(len(true),len(v90e))
    x=hw["X"].to_numpy(float);yy=hw["Y"].to_numpy(float)
    out[wn]=dict(v90=(v90e[:n])[fin[:n]],base=(base[:n]-true[:n])[fin[:n]],
                 base_clip=(base_clip[:n]-true[:n])[fin[:n]],flat=(flat[:n]-true[:n])[fin[:n]],
                 cx=float(np.mean(x)),cy=float(np.mean(yy)))
    c+=1
    if c%200==0: print(f"  {c} ({time.time()-t0:.0f}s)",flush=True)
pickle.dump(out,open(P/"v91"/"grfree_v91.pkl","wb"))
W=list(out);cx=np.array([out[w]["cx"] for w in W]);cy=np.array([out[w]["cy"] for w in W])
v90=np.concatenate([out[w]["v90"] for w in W]);r=lambda e:float(np.sqrt(np.mean(e**2)));base_r=r(v90)
fb=KMeans(n_clusters=8,n_init=5,random_state=0).fit_predict(np.column_stack([cx,cy]))
print(f"\n[GRフリー物理脚 {len(W)}井] v90 base={base_r:.4f}")
for lg in ["flat","base","base_clip"]:
    a=np.concatenate([out[w][lg] for w in W]);rho=np.corrcoef(v90,a)[0,1]
    fo=[]
    for b in np.unique(fb):
        trW=[W[i] for i in range(len(W)) if fb[i]!=b];vaW=[W[i] for i in range(len(W)) if fb[i]==b]
        tv=np.concatenate([out[w]["v90"] for w in trW]);ta=np.concatenate([out[w][lg] for w in trW])
        wi=min([(r((1-w)*tv+w*ta),w) for w in np.arange(0,0.41,0.02)])[1]
        vv=np.concatenate([out[w]["v90"] for w in vaW]);va=np.concatenate([out[w][lg] for w in vaW])
        fo.append((1-wi)*vv+wi*va)
    fo=r(np.concatenate(fo));fx=r(0.9*v90+0.1*a)
    print(f"  {lg:10s}: standalone={r(a):.3f} ρ={rho:.3f} | field-out-w={fo:.4f}(Δ{fo-base_r:+.4f}) 固定w0.10={fx:.4f}(Δ{fx-base_r:+.4f})")
