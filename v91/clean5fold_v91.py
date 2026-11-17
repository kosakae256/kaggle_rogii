# -*- coding: utf-8 -*-
"""★clean 5-fold: train/val完全分離。valのラベルもweight選択も一切使わない。
   各脚の構成: v90=well-out-of-fold OOF / snap=per-well推論(他井・ラベル不使用) / v74=well-out-of-fold OOF。
   weightはtrain 4foldのみで選びval foldに適用。val予測をpoolしてRMSE。5seed。"""
import sys,io,pickle
try: sys.stdout=io.open(1,"w",encoding="utf-8",closefd=False)
except: pass
import numpy as np
from pathlib import Path
from sklearn.model_selection import KFold
P=Path(r"C:\Users\kosaka256\Documents\rogii_claude")
rows=pickle.load(open(P/"v91"/"stack_v91.pkl","rb"));n=len(rows)  # v74=phaseI_best(well-CV=ランダム正解)
gl=np.array([len(d["v90"]) for d in rows]);wof=np.concatenate([[i]*gl[i] for i in range(n)])
V=np.concatenate([d["v90"] for d in rows]);S=np.concatenate([d["snap"] for d in rows]);U=np.concatenate([d["v74"] for d in rows])
r=lambda e:float(np.sqrt(np.mean(e**2)));base=r(V)
print(f"[clean 5-fold {n}井] base(v90 OOF)={base:.4f}")
print(f"  standalone(leak-free): snap={r(S):.3f} v74={r(U):.3f}")
def clean(legs,seeds=range(8)):
    outs=[]
    for sd in seeds:
        kf=KFold(5,shuffle=True,random_state=sd);err=np.zeros_like(V)
        for tr,va in kf.split(np.arange(n)):
            trw=set(tr.tolist());trm=np.array([wof[k] in trw for k in range(len(V))]);vam=~trm
            # ★train foldのみでweight選択(grid)。val fold(vam)は一切見ない
            if len(legs)==1:
                best=(1e9,0)
                for w in np.arange(0,0.31,0.01):
                    e=(1-w)*V[trm]+w*legs[0][trm]
                    if r(e)<best[0]:best=(r(e),w)
                w=best[1];err[vam]=(1-w)*V[vam]+w*legs[0][vam]
            else:
                best=(1e9,(0,0))
                for w1 in np.arange(0,0.26,0.02):
                    for w2 in np.arange(0,0.16,0.02):
                        if w1+w2>0.4:continue
                        e=(1-w1-w2)*V[trm]+w1*legs[0][trm]+w2*legs[1][trm]
                        if r(e)<best[0]:best=(r(e),(w1,w2))
                w1,w2=best[1];err[vam]=(1-w1-w2)*V[vam]+w1*legs[0][vam]+w2*legs[1][vam]
        outs.append(r(err))
    return np.mean(outs),np.std(outs)
for name,legs in [("v90+snap",[S]),("v90+snap+v74",[S,U])]:
    m,sd=clean(legs)
    print(f"  ★clean5fold(weight=train選択→val適用) {name}: {m:.4f}±{sd:.4f} (Δ{m-base:+.4f})")
# 参考: 固定weight(全データ=weight選択にval含む=リーク有)
print(f"  [参考:固定w全データ(weight peek)] snap0.10={r(0.9*V+0.1*S):.4f}({r(0.9*V+0.1*S)-base:+.4f}) snap0.10+v740.06={r(0.84*V+0.1*S+0.06*U):.4f}({r(0.84*V+0.1*S+0.06*U)-base:+.4f})")
