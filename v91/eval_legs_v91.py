# -*- coding: utf-8 -*-
"""物理脚バリアントを field-out固定weight blend で評価し最強脱相関脚を確定。"""
import sys,io,pickle
try: sys.stdout=io.open(1,"w",encoding="utf-8",closefd=False)
except: pass
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
P=Path(r"C:\Users\kosaka256\Documents\rogii_claude")
D=pickle.load(open(P/"v91"/"legs_v91.pkl","rb"));W=list(D)
cx=np.array([D[w]["cx"] for w in W]);cy=np.array([D[w]["cy"] for w in W])
r=lambda e:float(np.sqrt(np.mean(e**2)))
v90=np.concatenate([D[w]["v90"] for w in W]);base=r(v90)
fb=KMeans(n_clusters=8,n_init=5,random_state=0).fit_predict(np.column_stack([cx,cy]))
print(f"[物理脚 field-out blend評価 {len(W)}井] base={base:.4f}")
legs=["snap","snapms","pmean","pmeanms"]
for lg in legs:
    a=np.concatenate([D[w][lg] for w in W]);rho=np.corrcoef(v90,a)[0,1]
    # 固定weight掃引(全井=in-sample上界)
    isbest=min([(r((1-w)*v90+w*a),w) for w in np.arange(0,0.41,0.02)])
    # field-out: 各ブロックのwを他ブロックで決め適用(honest)
    fo=[]
    for b in np.unique(fb):
        trW=[W[i] for i in range(len(W)) if fb[i]!=b];vaW=[W[i] for i in range(len(W)) if fb[i]==b]
        tv=np.concatenate([D[w]["v90"] for w in trW]);ta=np.concatenate([D[w][lg] for w in trW])
        wi=min([(r((1-w)*tv+w*ta),w) for w in np.arange(0,0.41,0.02)])[1]
        vv=np.concatenate([D[w]["v90"] for w in vaW]);va=np.concatenate([D[w][lg] for w in vaW])
        fo.append((1-wi)*vv+wi*va)
    fo=r(np.concatenate(fo))
    # 固定w=0.10(調整なし)
    fx=r(0.9*v90+0.1*a)
    print(f"  {lg:8s}: standalone={r(a):.3f} ρ={rho:.3f} | in-sample best={isbest[0]:.4f}(w{isbest[1]:.2f}) field-out-w={fo:.4f} 固定w0.10={fx:.4f}(Δ{fx-base:+.4f})")
# 最良脚 + snap の2脚blend(脱相関違いを重ねる)
print("\n[2脚重ね(固定weight)]")
for l1 in ["snapms","pmeanms"]:
    a1=np.concatenate([D[w][l1] for w in W])
    best=(base,0,0)
    for w1 in np.arange(0,0.3,0.02):
        for w2 in np.arange(0,0.2,0.02):
            e=(1-w1-w2)*v90+w1*a1+w2*np.concatenate([D[w]["snap"] for w in W])
            if r(e)<best[0]: best=(r(e),w1,w2)
    print(f"  v90+{l1}+snap: {best[0]:.4f} (w={best[1]:.2f},{best[2]:.2f})")
