# -*- coding: utf-8 -*-
"""STEP1: struct20 params × GRフリー錨(v91 OOF) の単体PF精度。非smooth, n_seed32, 少数well。
   比較: 錨なし vs GRフリー錨(anc=grfree tvt, ancs=grfree sig)。mean軌跡RMSE vs 真TVT。"""
import io,sys,os,pickle,json
sys.stdout=io.open(1,"w",encoding="utf-8",closefd=False)
import numpy as np,pandas as pd,glob,importlib.util
from pathlib import Path
P=Path(r"C:\Users\kosaka256\Documents\rogii_claude");TR=P/"rogii-wellbore-geology-prediction"/"train"
spec=importlib.util.spec_from_file_location("cv92", P/"v92"/"create_candidates_v92.py")
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
NSEED=int(os.environ.get("NSEED","32"));NW=int(os.environ.get("NW","20"));LAG=int(os.environ.get("LAG","0"))
# struct20 params
st20=json.loads((P/"v65"/"artifacts_v65"/"struct20_params.json").read_text())["best_params"]
raw=dict(st20);raw["n_seeds"]=NSEED
p=m._make_pf_params(raw,0);p["label"]="struct20";p["_raw"]=st20
print(f"struct20: anchor_mult={st20.get('anchor_mult')} use_anchor={st20.get('use_anchor')} gr_power={round(st20.get('gr_power',0),3)}")
# GRフリー錨
GRF=pickle.load(open(P/"v74"/"artifacts_v74"/"grfree_v2_anchor.pkl","rb"))
# well集合(錨あり & 有効)
wells=sorted({os.path.basename(q).split("__")[0] for q in glob.glob(str(TR/"*__horizontal_well.csv"))})
names=[];paths=[]
for w in wells:
    if len(names)>=NW: break
    hp=TR/f"{w}__horizontal_well.csv";tp=TR/f"{w}__typewell.csv"
    try: hw=pd.read_csv(hp);tw=pd.read_csv(tp).sort_values("TVT")
    except: continue
    if "TVT" not in hw.columns or w not in GRF: continue
    if m.build_smoother_inputs(hw,tw["TVT"].to_numpy(float),tw["GR"].to_numpy(float),st20) is None: continue
    names.append(w);paths.append((hp,tp))
print(f"測定well={len(names)} NSEED={NSEED} LAG={LAG}")
TRUE={}
for w,(hp,tp) in zip(names,paths):
    hw=pd.read_csv(hp);ti=hw["TVT_input"].to_numpy(float);tvt=hw["TVT"].to_numpy(float)
    hps=int(np.where(np.isfinite(ti))[0].max());TRUE[w]=tvt[hps+1:]
Pk={**st20,"smooth_lag":LAG,"n_seeds":NSEED,"use_phys":st20.get("use_phys",False),"phys_sig":float(st20.get("phys_sig",1.0))}
def run(anchor):
    inps=[]
    for w,(hp,tp) in zip(names,paths):
        hw=pd.read_csv(hp);tw=pd.read_csv(tp).sort_values("TVT")
        x=m.build_smoother_inputs(hw,tw["TVT"].to_numpy(float),tw["GR"].to_numpy(float),Pk)
        n=len(x["md"])
        if anchor=="grfree" and w in GRF and len(GRF[w]["tvt"])==n:
            x["anc"]=np.asarray(GRF[w]["tvt"],float); x["ancs"]=np.asarray(GRF[w]["sig"],float)
        else:
            x["anc"]=np.zeros(n); x["ancs"]=np.full(n,1e9)   # 錨OFF
        inps.append(x)
    outs=m.run_smoother_bank(inps,Pk,1,getattr(m,"SEED_V26",12345),NSEED,min(len(names),8))
    e=[]
    for i,w in enumerate(names):
        mn=np.asarray(outs[i]["mean"],float);tv=TRUE[w];k=min(len(mn),len(tv));e.append(mn[:k]-tv[:k])
    e=np.concatenate(e);return float(np.sqrt(np.mean(e[np.isfinite(e)]**2)))
print("struct20 錨なし 実行...",flush=True);r0=run("off")
print("struct20 GRフリー錨 実行...",flush=True);r1=run("grfree")
print(f"\n[struct20 単体PF, {len(names)}井, n_seed{NSEED}, 非smooth]")
print(f"  錨なし         = {r0:.3f}")
print(f"  ★GRフリー錨    = {r1:.3f}  ({r1-r0:+.3f})")
