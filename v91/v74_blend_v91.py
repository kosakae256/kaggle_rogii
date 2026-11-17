# -*- coding: utf-8 -*-
"""v74 GRフリー予測(各種phase OOF)をv90のblend脚として評価: standalone/ρ/field-out固定weight blend。"""
import sys,io,pickle,glob
try: sys.stdout=io.open(1,"w",encoding="utf-8",closefd=False)
except: pass
import numpy as np,pyarrow.parquet as pq
from pathlib import Path
from sklearn.cluster import KMeans
P=Path(r"C:\Users\kosaka256\Documents\rogii_claude");A74=P/"v74"/"artifacts_v74";A90=P/"v90"/"artifacts_v90"
W_NN=0.62
tb=pq.read_table(A90/"data"/"train.parquet",columns=["well","target"])
well=tb.column("well").to_numpy(zero_copy_only=False);y=tb.column("target").to_numpy(zero_copy_only=False).astype(float)
gbm=np.load(A90/"gbm_v41_oof.npy").astype(float)[:len(y)];nnz=np.load(A90/"nn_oof_combo3.npz")
nn=np.mean([nnz[k] for k in nnz.files],axis=0)[:len(y)];v90err=(1-W_NN)*gbm+W_NN*nn-y
ibw={}
for i,w in enumerate(well): ibw.setdefault(w,[]).append(i)
for w in ibw: ibw[w]=np.array(ibw[w])
cent=pickle.load(open(P/"v91"/"grfree_v91.pkl","rb"))  # cx,cy流用
r=lambda e:float(np.sqrt(np.mean(e**2)))
def evalfile(fn):
    z=np.load(A74/fn,allow_pickle=True);wells=sorted(set(k.split("__")[0] for k in z.files))
    Ev=[];Ea=[];CX=[];CY=[];Ws=[]
    for wn in wells:
        ek=f"{wn}__err"
        if ek not in z.files or wn not in ibw or wn not in cent: continue
        e74=np.asarray(z[ek],float);v90e=v90err[ibw[wn]]
        n=min(len(e74),len(v90e))
        if n<5: continue
        Ev.append(v90e[:n]);Ea.append(e74[:n]);CX.append(cent[wn]["cx"]);CY.append(cent[wn]["cy"]);Ws.append(wn)
    if not Ev: return None
    v90=np.concatenate(Ev);a=np.concatenate(Ea);rho=np.corrcoef(v90,a)[0,1];base=r(v90)
    fb=KMeans(n_clusters=8,n_init=5,random_state=0).fit_predict(np.column_stack([CX,CY]))
    fo=[]
    for b in np.unique(fb):
        tr=[i for i in range(len(Ws)) if fb[i]!=b];va=[i for i in range(len(Ws)) if fb[i]==b]
        tv=np.concatenate([Ev[i] for i in tr]);ta=np.concatenate([Ea[i] for i in tr])
        wi=min([(r((1-w)*tv+w*ta),w) for w in np.arange(0,0.41,0.02)])[1]
        vv=np.concatenate([Ev[i] for i in va]);aa=np.concatenate([Ea[i] for i in va]);fo.append((1-wi)*vv+wi*aa)
    fo=r(np.concatenate(fo));fx=r(0.9*v90+0.1*a)
    return (r(a),rho,base,fo,fx,len(Ws))
print("[v74 GRフリー脚 × v90 blend] (base=対応井のv90 RMSE)")
for fn in ["phaseD_oof_G3s5.npz","phaseD_oof_G.npz","phaseD_oof_G3.npz","phaseI_oof_best.npz","phaseI_oof_3k15.npz","phaseI_oof_fieldout_aug.npz","phaseI_oof_fieldout_pre.npz","phaseA_oof.npz","inpaint_oof.npz"]:
    try:
        res=evalfile(fn)
        if res: print(f"  {fn:32s}: standalone={res[0]:6.2f} ρ={res[1]:+.3f} base={res[2]:.3f} | field-out={res[3]:.4f}(Δ{res[3]-res[2]:+.4f}) 固定w0.10={res[4]:.4f}(Δ{res[4]-res[2]:+.4f}) [{res[5]}井]")
    except Exception as e: print(f"  {fn}: ERR {e}")
