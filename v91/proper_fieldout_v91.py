# -*- coding: utf-8 -*-
"""★真のfield(5000ft連結成分)でGroupKFold=正しいfield-out。KMeans分断リークを修正して再評価。"""
import sys,io,pickle
try: sys.stdout=io.open(1,"w",encoding="utf-8",closefd=False)
except: pass
import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree
P=Path(r"C:\Users\kosaka256\Documents\rogii_claude")
rows=pickle.load(open(P/"v91"/"stack_phaseI_oof_fieldout_pre.pkl","rb"))
cx=np.array([d["cx"] for d in rows]);cy=np.array([d["cy"] for d in rows]);n=len(rows)
# 真field: 5000ft連結成分
tree=cKDTree(np.column_stack([cx,cy]));par=list(range(n))
def find(a):
    while par[a]!=a:par[a]=par[par[a]];a=par[a]
    return a
for i in range(n):
    for j in tree.query_ball_point([cx[i],cy[i]],5000.):
        if j>i:par[find(i)]=find(j)
field=np.array([find(i) for i in range(n)])
uf=np.unique(field);print(f"{n}井 {len(uf)}真field(5000ft連結). 最大{np.bincount(field).max()}井")
# field を5foldにGroupKFold(field単位で分割=同field全部同fold)
from sklearn.model_selection import GroupKFold
v90=[d["v90"] for d in rows];snap=[d["snap"] for d in rows];v74=[d["v74"] for d in rows]
gl=np.array([len(d["v90"]) for d in rows])
V=np.concatenate(v90);S=np.concatenate(snap);U=np.concatenate(v74)
r=lambda e:float(np.sqrt(np.mean(e**2)));base=r(V)
# 行→井index
wof=np.concatenate([[i]*gl[i] for i in range(n)])
print(f"base(v90)={base:.4f}")
# 固定weight(field非依存=不変)
print(f"[固定weight] snap0.10: {r(0.9*V+0.1*S):.4f}({r(0.9*V+0.1*S)-base:+.4f}) / snap0.10+v740.03: {r(0.87*V+0.1*S+0.03*U):.4f}({r(0.87*V+0.1*S+0.03*U)-base:+.4f})")
# ★正しいfield-out: field単位GroupKFold(5)でweight tune
def properfo(legs):
    gkf=GroupKFold(5);err=np.zeros_like(V)
    from itertools import product
    grid=[[round(w,2) for w in np.arange(0,0.31,0.03)] for _ in legs]
    for tr_f,va_f in gkf.split(np.arange(n),groups=field):
        trw=set(tr_f.tolist());trm=np.array([wof[k] in trw for k in range(len(V))]);vam=~trm
        best=(1e9,None)
        for ws in product(*grid):
            if sum(ws)>0.5:continue
            e=(1-sum(ws))*V[trm]+sum(w*legs[i][trm] for i,w in enumerate(ws))
            if r(e)<best[0]:best=(r(e),ws)
        ws=best[1];err[vam]=(1-sum(ws))*V[vam]+sum(w*legs[i][vam] for i,w in enumerate(ws))
    return r(err)
print(f"[★真field-out(tuned)] v90+snap: {properfo([S]):.4f} / v90+snap+v74: {properfo([S,U]):.4f}")
