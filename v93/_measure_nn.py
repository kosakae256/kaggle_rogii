# -*- coding: utf-8 -*-
import io,sys,os,pickle,glob
sys.stdout=io.open(1,"w",encoding="utf-8",closefd=False)
import numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0,"v93")
import pf_banks_v93 as pf
TR=pf.DATA_DIR/"train"
SIM=pickle.loads((pf.ART/"sim_grfree_v93.pkl").read_bytes())
N=int(os.environ.get("NW","40"))
wells=sorted({os.path.basename(q).split("__")[0] for q in glob.glob(str(TR/"*__horizontal_well.csv"))})[:N]
P=pf.bank_param("pfA")
inps,trues=[],[]
for w in wells:
    hw=pd.read_csv(TR/f"{w}__horizontal_well.csv"); tw=pd.read_csv(TR/f"{w}__typewell.csv").sort_values("TVT")
    tt=tw["TVT"].to_numpy(float); tg=tw["GR"].to_numpy(float); m=np.isfinite(tt)&np.isfinite(tg)
    if m.sum()<8: continue
    x=pf.build_smoother_inputs(hw,tt[m],tg[m],P)
    if x is None: continue
    pf.attach_anchor(x,w,P["_physics"])
    sd=SIM.get(w)
    if sd is not None: x["_sim"]=sd["sim"]; x["_st"]=sd["st"]
    ev=hw[hw["TVT_input"].isna()]
    inps.append(x); trues.append(ev["TVT"].to_numpy(float))
def rmse(res):
    e=np.concatenate([res[k]["mean"]-trues[k] for k in range(len(inps))]); e=e[np.isfinite(e)]
    return float(np.sqrt(np.mean(e**2)))
print(f"[pfA {len(inps)}井 n_seed16] NN-emission(GRフリー帯)単体測定:",flush=True)
r_off=rmse(pf.run_smoother_ext(inps,P,seed=99999,n_seeds=16,chunk=8,w_nn=0.0))
print(f"  w_nn=0(なし)          = {r_off:.4f}",flush=True)
r_on =rmse(pf.run_smoother_ext(inps,P,seed=99999,n_seeds=16,chunk=8,w_nn=P["_w_nn"]))
print(f"  w_nn={P['_w_nn']}(GRフリーsim) = {r_on:.4f}  ({r_on-r_off:+.4f})",flush=True)
