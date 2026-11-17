# -*- coding: utf-8 -*-
"""field-outの意味を実データで図示: 坑井X-Y分布, fieldクラスタ, well-CV vs field-out holdout。"""
import sys,io,glob
try: sys.stdout=io.open(1,"w",encoding="utf-8",closefd=False)
except: pass
import numpy as np,pandas as pd
from pathlib import Path
from scipy.spatial import cKDTree
import matplotlib;matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"]="Meiryo"
P=Path(r"C:\Users\kosaka256\Documents\rogii_claude");TR=P/"rogii-wellbore-geology-prediction"/"train"
# 各坑井のcentroid
X=[];Y=[];WN=[]
for f in sorted(glob.glob(str(TR/"*__horizontal_well.csv"))):
    hw=pd.read_csv(f,usecols=["X","Y"])
    X.append(float(hw["X"].mean()));Y.append(float(hw["Y"].mean()));WN.append(Path(f).name.split("__")[0])
X=np.array(X);Y=np.array(Y);n=len(X)
# field=連結成分(5000ft以内で繋がる) union-find
EPS=5000.0
tree=cKDTree(np.column_stack([X,Y]));parent=list(range(n))
def find(a):
    while parent[a]!=a: parent[a]=parent[parent[a]];a=parent[a]
    return a
for i in range(n):
    for j in tree.query_ball_point([X[i],Y[i]],EPS):
        if j>i: parent[find(i)]=find(j)
lab=np.array([find(i) for i in range(n)])
uniq,inv,cnt=np.unique(lab,return_inverse=True,return_counts=True)
print(f"{n}坑井 → {len(uniq)}field(5000ft連結). field規模: 中央{int(np.median(cnt))} 最大{cnt.max()} 単独field={int((cnt==1).sum())}")
fid=inv  # 0..nfield-1
fig,ax=plt.subplots(1,3,figsize=(18,6))
rng=np.random.default_rng(0);cols=rng.random((len(uniq),3))
ax[0].scatter(X,Y,c=cols[fid],s=14);ax[0].set_title(f"全{n}坑井のX-Y分布 (色=field, {len(uniq)}fields, 5000ft連結)")
ax[0].set_xlabel("X");ax[0].set_ylabel("Y")
# 拡大: あるfield群
cx,cy=np.median(X),np.median(Y);msk=(np.abs(X-cx)<40000)&(np.abs(Y-cy)<40000)
ax[1].scatter(X[msk],Y[msk],c=cols[fid[msk]],s=40)
ax[1].set_title("拡大: fieldは空間クラスタ(同field=近接坑井群)")
ax[1].set_xlabel("X");ax[1].set_ylabel("Y")
# well-CV vs field-out の holdout概念図(同じ拡大領域で)
big=uniq[np.argmax(cnt)];fmsk=lab==big
ax[2].scatter(X,Y,c="lightgray",s=10,label="train")
# well-CV: fieldの中からランダムに20%をval(field-mateはtrainに残る=リーク)
vw=rng.random(n)<0.2
ax[2].scatter(X[vw],Y[vw],c="orange",s=14,label="well-CV val(散在, field-mate残る=リーク)")
# field-out: 丸ごとfieldをval
ax[2].scatter(X[fmsk],Y[fmsk],facecolors="none",edgecolors="red",s=60,linewidths=1.5,label="field-out val(field丸ごと=非リーク)")
ax[2].set_title("well-CV(散在,リーク) vs field-out(field丸ごと,非リーク)")
ax[2].legend(fontsize=8);ax[2].set_xlabel("X");ax[2].set_ylabel("Y")
plt.tight_layout();out=P/"v91"/"fieldout_explain.png";plt.savefig(out,dpi=100)
print("saved",out)
# 近接統計: 各坑井の最近傍他坑井距離
d,_=tree.query(np.column_stack([X,Y]),k=2)
print(f"最近傍坑井距離: 中央{np.median(d[:,1]):.0f}ft, <5000ft={100*(d[:,1]<5000).mean():.0f}%, >10000ft(孤立)={100*(d[:,1]>10000).mean():.0f}%")
