# -*- coding: utf-8 -*-
"""ヘッジゲート学習: v90 vs 物理(r0snap/GRsel) を井戸ごとに使い分け。★field-out CV(空間ブロック)必須。
   比較: field-out CV vs well-random CV(過学習露呈用) 両方報告。"""
import os,sys,io,pickle
try: sys.stdout=io.open(1,"w",encoding="utf-8",closefd=False)
except: pass
import numpy as np
from pathlib import Path
import lightgbm as lgb
from sklearn.cluster import KMeans
P=Path(r"C:\Users\kosaka256\Documents\rogii_claude")
D=pickle.load(open(P/"v91"/"hedge_data_v91.pkl","rb"))
wells=list(D)
PHYS=os.environ.get("PHYS","snap")   # snap or grsel
FKEYS=["L","gr_var","pdip","cand_spread","topgr_spread","snap_move","r0_rough","snap_gr_resid","grsel_gr_resid","v90_pfr","v90_pfrb","v90_pfrs"]
Xf=np.array([[D[w]["feat"][k] for k in FKEYS] for w in wells])
cx=np.array([D[w]["cx"] for w in wells]);cy=np.array([D[w]["cy"] for w in wells])
v90rms=np.array([np.sqrt(np.mean(D[w]["v90"]**2)) for w in wells])
phrms=np.array([np.sqrt(np.mean(D[w][PHYS]**2)) for w in wells])
# ラベル=物理が勝つ, 重み=|margin|×√L (overall寄与)
lab=(phrms<v90rms).astype(int)
Ln=np.array([D[w]["feat"]["L"] for w in wells])
wgt=np.abs(v90rms-phrms)*np.sqrt(Ln)
def pooled(werr_list): return float(np.sqrt(np.mean(np.concatenate(werr_list)**2)))
base=pooled([D[w]["v90"] for w in wells])
orc=pooled([D[w]["v90"] if v90rms[i]<=phrms[i] else D[w][PHYS] for i,w in enumerate(wells)])
def run_cv(blocks, tag):
    p=np.zeros(len(wells))
    for b in np.unique(blocks):
        tr=blocks!=b;va=blocks==b
        if tr.sum()<20 or va.sum()<1: p[va]=0;continue
        m=lgb.LGBMClassifier(n_estimators=200,num_leaves=15,learning_rate=0.03,
                             min_child_samples=15,subsample=0.8,colsample_bytree=0.8,verbose=-1)
        m.fit(Xf[tr],lab[tr],sample_weight=wgt[tr])
        p[va]=m.predict_proba(Xf[va])[:,1]
    # soft blend: w=scale*p を掃引
    best=(base,0,0)
    for sc in [0.15,0.25,0.35,0.5,0.7,1.0]:
        for thr in [0.0]:
            w=np.clip(sc*p,0,1)
            err=[ (1-w[i])*D[wl]["v90"]+w[i]*D[wl][PHYS] for i,wl in enumerate(wells)]
            rm=pooled(err)
            if rm<best[0]: best=(rm,sc,thr)
    # hard gate: p>thr で物理採用
    for thr in [0.5,0.6,0.7,0.8]:
        err=[ D[wl][PHYS] if p[i]>thr else D[wl]["v90"] for i,wl in enumerate(wells)]
        rm=pooled(err)
        if rm<best[0]: best=(rm,"hard",thr)
    from sklearn.metrics import roc_auc_score
    auc=roc_auc_score(lab,p) if len(np.unique(lab))>1 else .5
    print(f"  [{tag}] gate AUC={auc:.3f}  best blend RMSE={best[0]:.4f} (base {base:.4f}, oracle {orc:.4f})  param={best[1:]}")
    return best[0]
print(f"物理脚={PHYS}  {len(wells)}井  base(v90)={base:.4f}  ヘッジoracle={orc:.4f}  物理勝ち={lab.sum()}井")
# field-out: 空間KMeansブロック
for NB in [8,12]:
    fb=KMeans(n_clusters=NB,n_init=5,random_state=0).fit_predict(np.column_stack([cx,cy]))
    run_cv(fb, f"field-out K={NB}")
# well-random(過学習露呈)
rb=np.random.default_rng(0).integers(0,8,len(wells))
run_cv(rb, "well-random K=8")
