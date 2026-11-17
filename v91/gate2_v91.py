# -*- coding: utf-8 -*-
"""ヘッジゲート改: v90の内部不確実性(PFバンク間不一致/beam std/sig残差)を特徴に追加。field-out CV。"""
import os,sys,io,pickle,warnings
warnings.filterwarnings("ignore")
try: sys.stdout=io.open(1,"w",encoding="utf-8",closefd=False)
except: pass
import numpy as np,pyarrow.parquet as pq
from pathlib import Path
import lightgbm as lgb
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score
P=Path(r"C:\Users\kosaka256\Documents\rogii_claude");A90=P/"v90"/"artifacts_v90"
D=pickle.load(open(P/"v91"/"hedge_data_v91.pkl","rb"));wells=list(D)
# v90内部不確実性をparquetから: pf_1..5残差の井内不一致 + beam_std + sig残差
unc_cols=["well"]+[f"pf_{i}_gr_resid0" for i in range(1,6)]+["beam_std_d","sig_gr_best_resid","pf_gr_resid0_rstd21","hyb_gr_resid0_rstd21"]
tb=pq.read_table(A90/"data"/"train.parquet",columns=unc_cols)
wl=tb.column("well").to_numpy(zero_copy_only=False)
mats={c:tb.column(c).to_numpy(zero_copy_only=False).astype(float) for c in unc_cols[1:]}
pfb=np.column_stack([mats[f"pf_{i}_gr_resid0"] for i in range(1,6)])
bankdis=np.nanstd(pfb,axis=1)   # PFバンク間不一致(行)
ibw={}
for i,w in enumerate(wl): ibw.setdefault(w,[]).append(i)
for w in ibw: ibw[w]=np.array(ibw[w])
def wagg(wn):
    idx=ibw.get(wn); 
    if idx is None: return [0,0,0,0]
    return [float(np.nanmean(bankdis[idx])),float(np.nanmean(mats["beam_std_d"][idx])),
            float(np.nanmean(mats["sig_gr_best_resid"][idx])),float(np.nanmean(mats["hyb_gr_resid0_rstd21"][idx]))]
FK=["L","gr_var","pdip","cand_spread","topgr_spread","snap_move","r0_rough","snap_gr_resid","grsel_gr_resid","v90_pfr","v90_pfrb","v90_pfrs"]
Xf=np.array([[D[w]["feat"][k] for k in FK]+wagg(w) for w in wells])
cx=np.array([D[w]["cx"] for w in wells]);cy=np.array([D[w]["cy"] for w in wells])
PHYS=os.environ.get("PHYS","snap")
v90rms=np.array([np.sqrt(np.mean(D[w]["v90"]**2)) for w in wells])
phrms=np.array([np.sqrt(np.mean(D[w][PHYS]**2)) for w in wells])
lab=(phrms<v90rms).astype(int);Ln=np.array([D[w]["feat"]["L"] for w in wells])
wgt=np.abs(v90rms-phrms)*np.sqrt(Ln)
pooled=lambda L:float(np.sqrt(np.mean(np.concatenate(L)**2)))
base=pooled([D[w]["v90"] for w in wells])
orc=pooled([D[w]["v90"] if v90rms[i]<=phrms[i] else D[w][PHYS] for i,w in enumerate(wells)])
def run(blocks,tag):
    p=np.zeros(len(wells))
    for b in np.unique(blocks):
        tr=blocks!=b;va=blocks==b
        if tr.sum()<20 or va.sum()<1: continue
        m=lgb.LGBMClassifier(n_estimators=250,num_leaves=15,learning_rate=0.03,min_child_samples=15,
                             subsample=0.8,colsample_bytree=0.8,verbose=-1)
        m.fit(Xf[tr],lab[tr],sample_weight=wgt[tr]);p[va]=m.predict_proba(Xf[va])[:,1]
    best=(base,None)
    for sc in [0.15,0.25,0.35,0.5,0.7,1.0]:
        w=np.clip(sc*p,0,1);rm=pooled([(1-w[i])*D[wl2]["v90"]+w[i]*D[wl2][PHYS] for i,wl2 in enumerate(wells)])
        if rm<best[0]: best=(rm,("soft",sc))
    for thr in [0.5,0.6,0.7,0.8]:
        rm=pooled([D[wl2][PHYS] if p[i]>thr else D[wl2]["v90"] for i,wl2 in enumerate(wells)])
        if rm<best[0]: best=(rm,("hard",thr))
    auc=roc_auc_score(lab,p) if len(np.unique(lab))>1 else .5
    print(f"  [{tag}] AUC={auc:.3f} best={best[0]:.4f} (base {base:.4f} oracle {orc:.4f} Δ{best[0]-base:+.4f}) {best[1]}")
print(f"物理={PHYS} {len(wells)}井 base={base:.4f} oracle={orc:.4f} 物理勝ち{lab.sum()} 特徴{Xf.shape[1]}次元")
for NB in [8,12]:
    fb=KMeans(n_clusters=NB,n_init=5,random_state=0).fit_predict(np.column_stack([cx,cy]));run(fb,f"field-out K={NB}")
run(np.random.default_rng(0).integers(0,8,len(wells)),"well-random")
