# -*- coding: utf-8 -*-
"""v51: 擬似nbr(確信井戸の予測パス注入)入り証拠pklを構築 → evidenceB3_pseudo.pkl"""
import os, sys, pickle, time
os.environ.setdefault("TAG","capaug"); os.environ.setdefault("CH","128")
os.environ.setdefault("DIM","96"); os.environ.setdefault("DILS","1,2,4,8,16,32")
os.environ.setdefault("BTAG","combo"); os.environ.setdefault("REFH","1")
os.environ.setdefault("DILT","1,2,4,8,16"); os.environ.setdefault("BCH","96")
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "v50"))
import stageB_v50 as sb
import stageA_v50 as sa

ART51 = Path(r"C:\Users\kosaka256\Documents\rogii_claude\v51\artifacts_v51")
PROF = pickle.loads((ART51/"prof_v51.pkl").read_bytes())
E50 = dict(np.load(sb.ART/"v50_stageBcombo_errs.npz"))
data = pickle.loads(sa.CACHE.read_bytes())
WELLS = sorted(PROF.keys()); XY = np.array([data[w]["xy"] for w in WELLS])
SH = np.arange(-20.0, 20.01, 0.5)
CONF_Q = 0.4; NBK = 8
margin = {}
for w in WELLS:
    pr = PROF[w]["prof"].astype(np.float64)
    margin[w] = float(pr[np.abs(SH)<=1.0].max() - pr[np.abs(SH)>=4.0].max())
th = np.quantile(list(margin.values()), CONF_Q)
conf = {w: margin[w] >= th for w in WELLS}
ZP = {w: sb.EVD[w]["y_ev"] + E50[w][:len(sb.EVD[w]["y_ev"])] for w in WELLS}
from scipy.spatial import cKDTree
_, idxs = cKDTree(XY).query(XY, k=NBK+1)
offs = np.arange(-sa.NB, sa.NB+1)
def nbr_points(j):
    c = data[WELLS[j]]
    knm = (~c["ev"]) & np.isfinite(c["y"])
    ys=[c["y"][knm].astype(np.float64)]; gs=[c["graw"][knm].astype(np.float64)]
    if conf[WELLS[j]]:
        zj = ZP[WELLS[j]]; fin = np.isfinite(sb.EVD[WELLS[j]]["y_ev"])
        gr_ev = c["graw"][c["ev"]][:len(zj)]
        ys.append(zj[fin]); gs.append(gr_ev[fin])
    return np.concatenate(ys), np.concatenate(gs)
OUT = {}
t0=time.time()
for i, w in enumerate(WELLS):
    c = data[w]; e = sb.EVD[w]
    K = len(c["gg"]); gmin = c["gmin"]
    ny,ng = [],[]
    for j in idxs[i,1:]:
        a,b = nbr_points(j); ny.append(a); ng.append(b)
    ny=np.concatenate(ny); ng=np.concatenate(ng)
    nbrg = np.full(K, np.nan)
    kyn = np.round((ny-gmin)/sa.STEP).astype(np.int64)
    okn = (kyn>=0)&(kyn<=K-1)
    if okn.sum()>=20:
        nsum=np.zeros(K); ncnt=np.zeros(K)
        np.add.at(nsum,kyn[okn],ng[okn]); np.add.at(ncnt,kyn[okn],1)
        with np.errstate(invalid="ignore"):
            nbrg = np.where(ncnt>0, nsum/np.maximum(ncnt,1), np.nan)
    st=c["st"].astype(np.float64); ks=np.round((st-gmin)/sa.STEP).astype(np.int64)
    cand=np.clip(ks[:,None]+offs[None,:],0,K-1)
    okc=(ks[:,None]+offs[None,:]>=0)&(ks[:,None]+offs[None,:]<=K-1)
    nc=np.abs(c["g25"][:,None]-nbrg[cand]); has=np.isfinite(nc)&okc
    nc=np.where(has,nc,60.0)
    T=int(c["ev"].sum()); nb_=(T+sa.STRIDE-1)//sa.STRIDE; bid=np.arange(T)//sa.STRIDE
    def bm(x):
        o=np.zeros((nb_,)+x.shape[1:]); cnt=np.zeros(nb_)
        np.add.at(o,bid,x); np.add.at(cnt,bid,1)
        return o/np.maximum(cnt,1)[(...,)+(None,)*(x.ndim-1)]
    ee = dict(e)
    EV2 = e["EV"].astype(np.float32).copy()
    EV2[8]=np.exp(-0.5*(bm(nc)/sa.SIGG)**2).astype(np.float32)
    EV2[9]=bm(has.astype(np.float64)).astype(np.float32)
    ee["EV"]=EV2.astype(np.float16)
    OUT[w]=ee
    if (i+1)%200==0: print(f"  {i+1}/{len(WELLS)} ({time.time()-t0:.0f}s)", flush=True)
(ART51/"evidenceB3_pseudo.pkl").write_bytes(pickle.dumps(OUT, protocol=4))
print(f"saved evidenceB3_pseudo.pkl ({len(OUT)} wells, conf={sum(conf.values())})")
