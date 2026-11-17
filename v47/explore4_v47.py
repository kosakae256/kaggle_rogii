# -*- coding: utf-8 -*-
"""反復4: datum(struct基準オフセット)は空間相関するか。近傍train井戸のオフセットで自井戸の枝/TVTを決められるか。
GRと独立な"構造"の軸。v47 artifact + 公式CSV(XY)のみ。"""
import numpy as np, pandas as pd, json
from pathlib import Path
P=Path(r"C:\Users\kosaka256\Documents\rogii_claude"); ART=P/"v47"/"artifacts_v47"; TRAIN=P/"rogii-wellbore-geology-prediction"/"train"
z=np.load(ART/"candidates_v47.npz"); DEL=z["DEL"]; y=z["y"]; g=z["wellcode"].astype(np.int64); STRD=z["STRD"]; N=len(y)
wells=json.loads((ART/"candidates_v47_meta.json").read_text(encoding="utf-8"))["wells"]; W=len(wells)
K=3; KT=21//K; a=np.argsort(DEL,1); rep=np.take_along_axis(DEL,a,1).reshape(N,K,KT).mean(2); gold=np.argmin(np.abs(rep-y[:,None]),1)
def rmse(p,mask=None):
    e=(p-y); e=e[mask] if mask is not None else e; return float(np.sqrt(np.mean(e*e)))

# 井戸重心XY
XY=np.zeros((W,2))
for wi,wn in enumerate(wells):
    try: d=pd.read_csv(TRAIN/f"{wn}__horizontal_well.csv",usecols=["X","Y"]); XY[wi]=[d["X"].mean(),d["Y"].mean()]
    except Exception: XY[wi]=[np.nan,np.nan]
# 井戸ごと真オフセット(struct基準)= y - struct の平均、と枝
offw=np.array([np.mean((y-STRD)[g==wi]) for wi in range(W)])       # 真のdatumバイアス(struct比)
print(f"井戸オフセット: 平均{offw.mean():.2f} std{offw.std():.2f}  範囲[{np.percentile(offw,5):.1f},{np.percentile(offw,95):.1f}]")

# 近傍(leave-one-out)でオフセットを予測 -> 空間相関の強さ
from scipy.spatial import cKDTree
ok=np.isfinite(XY).all(1); idx=np.where(ok)[0]; tree=cKDTree(XY[idx])
for Knn in [3,5,10]:
    dist,nb=tree.query(XY[idx],k=Knn+1)                            # 自分含むのでKnn+1
    pred=np.array([np.mean(offw[idx[nb[i,1:]]]) for i in range(len(idx))])  # 自分除く近傍平均
    r=np.corrcoef(offw[idx],pred)[0,1]
    print(f"  k={Knn}: 近傍オフセット予測 vs 真オフセット  相関={r:.3f}  近傍距離中央値={np.median(dist[:,1]):.0f}")

# 近傍オフセットで struct を補正したら枝精度/RMSEは上がるか(行レベル)
Knn=8; dist,nb=tree.query(XY[idx],k=Knn+1); predoff=np.full(W,np.nan)
for i in range(len(idx)): predoff[idx[i]]=np.mean(offw[idx[nb[i,1:]]])
struct_corr=STRD+predoff[g]                                        # 近傍で補正した基準
# その補正基準に一番近い枝を選ぶ
sel=np.argmin(np.abs(rep-struct_corr[:,None]),1); acc=np.mean(sel==gold)
print("\n=== 反復4: 空間(近傍オフセット)で枝決定 ===")
print(f"  struct単体RMSE={rmse(STRD):.4f}  近傍補正struct RMSE={rmse(struct_corr):.4f}")
print(f"  近傍補正に最近接の枝: acc={acc:.3f}  rmse={rmse(rep[np.arange(N),sel]):.4f}")
print(f"  参考: GR系LightGBM枝acc≈0.54 / soft床6.4 / 天井(oracle-mode)={np.sqrt(np.mean(np.min(np.abs(rep-y[:,None]),1)**2)):.4f}")
