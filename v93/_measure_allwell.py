# -*- coding: utf-8 -*-
"""全井戸で struct20 の錨比較: 錨なし / GRフリー錨(sig=9ft実不確実性) / v66 struct錨(元struct20)。n_seed32。"""
import io,sys,os,json,pickle
sys.stdout=io.open(1,"w",encoding="utf-8",closefd=False)
import numpy as np
from pathlib import Path
sys.path.insert(0,"v93")
import pf_engine_v93 as pf
P=Path(r"C:\Users\kosaka256\Documents\rogii_claude")
NSEED=int(os.environ.get("NSEED","32"));CHUNK=int(os.environ.get("CHUNK","60"))
SIGFIX=float(os.environ.get("SIGFIX","9.0"))
st=json.loads(open("v93/artifacts_v93/struct20_init_params.json").read())
# 元の錨辞書(GRフリー)を退避
GRF=pickle.loads((P/"v93"/"artifacts_v93"/"grfree_anchor_train.pkl").read_bytes())
V66=pickle.loads((P/"v38"/"artifacts_v38"/"priors"/"prior_v66_leanv1cumul_train.pkl").read_bytes())
def set_anchor(src, sigfix=None):
    pf._ANCHOR.clear()
    for w,d in src.items():
        sig=np.asarray(d["sig"],float) if sigfix is None else np.full(len(d["tvt"]),sigfix,float)
        pf._ANCHOR[w]=dict(tvt=np.asarray(d["tvt"],float), sig=sig)
    pf.CACHE.clear()
def wells_and_rmse(P_):
    ws=[w for w in pf.list_wells() if pf.CACHE.setdefault(w,pf.build_cache(w)) is not None]
    ws=[w for w in ws if pf.CACHE[w] is not None]
    return ws, pf.overall_rmse_on(P_,ws,seed=99999,n_seeds=NSEED,chunk=CHUNK)
Poff=pf.full_params({**st,"use_anchor":False}); Pon=pf.full_params({**st,"use_anchor":True})
# 1) 錨なし(GRフリー辞書ロードでcache構築, 錨は使わない)
set_anchor(GRF, SIGFIX); ws,r_off=wells_and_rmse(Poff)
print(f"[全{len(ws)}井 struct20 n_seed{NSEED}]",flush=True)
print(f"  ① 錨なし                 = {r_off:.4f}",flush=True)
# 2) GRフリー錨(sig=SIGFIX)
set_anchor(GRF, SIGFIX); _,r_grf=wells_and_rmse(Pon)
print(f"  ② GRフリー錨(sig={SIGFIX}ft)  = {r_grf:.4f}  ({r_grf-r_off:+.4f})",flush=True)
# 2b) GRフリー錨(元のsig=3seed std)
set_anchor(GRF, None); _,r_grf0=wells_and_rmse(Pon)
print(f"  ②' GRフリー錨(元sig=3seed) = {r_grf0:.4f}  ({r_grf0-r_off:+.4f})",flush=True)
# 3) v66 struct錨(元struct20)
set_anchor(V66, None); _,r_v66=wells_and_rmse(Pon)
print(f"  ③ v66 struct錨(元struct20) = {r_v66:.4f}  ({r_v66-r_off:+.4f})",flush=True)
print(f"\n→ GRフリー錨(sig修正) vs 元struct20(v66錨): {r_grf-r_v66:+.4f}",flush=True)
