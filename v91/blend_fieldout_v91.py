# -*- coding: utf-8 -*-
"""物理snap脚のblend重みを field-out CVで検証(全井でw調整はin-sample→LB落ちる懸念)。
   空間ブロックでwを学習ブロックのみから決め、held-outに適用しpool。"""
import sys,io,pickle
try: sys.stdout=io.open(1,"w",encoding="utf-8",closefd=False)
except: pass
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
P=Path(r"C:\Users\kosaka256\Documents\rogii_claude")
D=pickle.load(open(P/"v91"/"hedge_data_v91.pkl","rb"));W=list(D)
cx=np.array([D[w]["cx"] for w in W]);cy=np.array([D[w]["cy"] for w in W])
r=lambda e:float(np.sqrt(np.mean(e**2)))
v90all=np.concatenate([D[w]["v90"] for w in W])
base=r(v90all)
def blend_at(w,keys):  # keys=井集合, 重みwでのpooled誤差配列
    return np.concatenate([(1-w)*D[k]["v90"]+w*D[k]["snap"] for k in keys])
# 全井最適w(in-sample=楽観)
ws=np.arange(0,0.41,0.01)
gi=int(np.argmin([r(blend_at(w,W)) for w in ws]));gw=ws[gi]
print(f"base={base:.4f}  in-sample最適 w={gw:.2f} → {r(blend_at(gw,W)):.4f}")
for NB in [6,10]:
    fb=KMeans(n_clusters=NB,n_init=5,random_state=0).fit_predict(np.column_stack([cx,cy]))
    err=[]
    for b in np.unique(fb):
        trW=[W[i] for i in range(len(W)) if fb[i]!=b];vaW=[W[i] for i in range(len(W)) if fb[i]==b]
        if not trW or not vaW: continue
        wi=ws[int(np.argmin([r(blend_at(w,trW)) for w in ws]))]  # 学習ブロックでw決定
        err.append(blend_at(wi,vaW))                              # held-outに適用
    fo=r(np.concatenate(err))
    print(f"  field-out K={NB}: {fo:.4f} (Δ{fo-base:+.4f})")
# 固定w=0.10(保守, 調整なし)も
for wf in [0.08,0.10,0.12]:
    print(f"  固定w={wf}(調整なし): {r(blend_at(wf,W)):.4f} (Δ{r(blend_at(wf,W))-base:+.4f})")
