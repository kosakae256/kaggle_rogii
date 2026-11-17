# -*- coding: utf-8 -*-
"""v74 GRフリー脚 × v90 blend (★正しい行アライメント: __rows全井index → v90 eval行=full_idx-(hps+1))。"""
import sys,io,pickle,glob
try: sys.stdout=io.open(1,"w",encoding="utf-8",closefd=False)
except: pass
import numpy as np,pandas as pd,pyarrow.parquet as pq
from pathlib import Path
from sklearn.cluster import KMeans
P=Path(r"C:\Users\kosaka256\Documents\rogii_claude");A74=P/"v74"/"artifacts_v74";A90=P/"v90"/"artifacts_v90"
TR=P/"rogii-wellbore-geology-prediction"/"train";W_NN=0.62
tb=pq.read_table(A90/"data"/"train.parquet",columns=["well","target"])
well=tb.column("well").to_numpy(zero_copy_only=False);y=tb.column("target").to_numpy(zero_copy_only=False).astype(float)
gbm=np.load(A90/"gbm_v41_oof.npy").astype(float)[:len(y)];nnz=np.load(A90/"nn_oof_combo3.npz")
nn=np.mean([nnz[k] for k in nnz.files],axis=0)[:len(y)];v90err=(1-W_NN)*gbm+W_NN*nn-y
ibw={}
for i,w in enumerate(well): ibw.setdefault(w,[]).append(i)
for w in ibw: ibw[w]=np.array(ibw[w])
# hps + centroid を各井のCSVから
hps_c={};cent={}
for f in sorted(glob.glob(str(TR/"*__horizontal_well.csv"))):
    wn=Path(f).name.split("__")[0]
    if wn not in ibw: continue
    hw=pd.read_csv(f,usecols=["TVT_input","X","Y"])
    kn=hw["TVT_input"].notna().to_numpy()
    if kn.sum()<10 or (~kn).sum()<30: continue
    hps_c[wn]=int(np.where(kn)[0].max());cent[wn]=(float(hw["X"].mean()),float(hw["Y"].mean()))
r=lambda e:float(np.sqrt(np.mean(e**2)))
def evalf(fn):
    z=np.load(A74/fn,allow_pickle=True);wells=sorted(set(k.split("__")[0] for k in z.files))
    Ev=[];Ea=[];CX=[];CY=[];Ws=[]
    for wn in wells:
        if f"{wn}__err" not in z.files or wn not in ibw or wn not in hps_c: continue
        e74=np.asarray(z[f"{wn}__err"],float);rw=np.asarray(z[f"{wn}__rows"]).astype(int)
        v90e=v90err[ibw[wn]];hp=hps_c[wn];ne=len(v90e)
        j=rw-(hp+1)                                   # v90 eval行index
        ok=(j>=0)&(j<ne)&np.isfinite(e74)
        if ok.sum()<5: continue
        Ev.append(v90e[j[ok]]);Ea.append(e74[ok]);CX.append(cent[wn][0]);CY.append(cent[wn][1]);Ws.append(wn)
    v90=np.concatenate(Ev);a=np.concatenate(Ea);rho=np.corrcoef(v90,a)[0,1];base=r(v90)
    fb=KMeans(n_clusters=8,n_init=5,random_state=0).fit_predict(np.column_stack([CX,CY]))
    fo=[]
    for b in np.unique(fb):
        tr=[i for i in range(len(Ws)) if fb[i]!=b];va=[i for i in range(len(Ws)) if fb[i]==b]
        tv=np.concatenate([Ev[i] for i in tr]);ta=np.concatenate([Ea[i] for i in tr])
        wi=min([(r((1-w)*tv+w*ta),w) for w in np.arange(0,0.41,0.02)])[1]
        vv=np.concatenate([Ev[i] for i in va]);aa=np.concatenate([Ea[i] for i in va]);fo.append((1-wi)*vv+wi*aa)
    return r(a),rho,base,r(np.concatenate(fo)),r(0.9*v90+0.1*a),len(Ws)
print("[v74 GRフリー脚 × v90 blend 正アライメント] base=対応間引き行のv90 RMSE(≈6.3なら正)")
for fn in ["phaseI_oof_best.npz","phaseD_oof_G3s5.npz","phaseI_oof_3k15.npz","phaseI_oof_fieldout_pre.npz","phaseD_oof_G.npz"]:
    try:
        s,rho,base,fo,fx,nw=evalf(fn)
        print(f"  {fn:30s}: GRフリー標準={s:6.2f} ρ={rho:+.3f} base={base:.3f} | field-out={fo:.4f}(Δ{fo-base:+.4f}) 固定w0.10={fx:.4f}(Δ{fx-base:+.4f}) [{nw}井]")
    except Exception as e: print(f"  {fn}: ERR {e}")
