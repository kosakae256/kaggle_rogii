4# -*- coding: utf-8 -*-
"""structgate(GRフリー錨)がPF軌跡をどれだけ改善するか。非smooth(smooth_lag=0), pf_1..5, 少数well/少seed。
   WITH=_attach_v75(GRフリー錨rtq/rtk) vs WITHOUT=錨なし。mean軌跡RMSE vs 真TVT。"""
import io,sys,os
sys.stdout=io.open(1,"w",encoding="utf-8",closefd=False)
os.environ["N_WELLS"]="1"   # import時のmain抑制には不要だが安全
import numpy as np,pandas as pd,glob,importlib.util
from pathlib import Path
P=Path(r"C:\Users\kosaka256\Documents\rogii_claude");TR=P/"rogii-wellbore-geology-prediction"/"train"
# v92 create をモジュールとして読込(main()は__main__ガードで走らない)
spec=importlib.util.spec_from_file_location("cv92", P/"v92"/"create_candidates_v92.py")
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
NSEED=int(os.environ.get("NSEED","16")); NW=int(os.environ.get("NW","8")); LAG=int(os.environ.get("LAG","0"))
# P_SETS 再構築
P_SETS=[]
for k,lab in enumerate(m.PF_BANK):
    raw=dict(m.PF_LIB[lab]);raw["n_seeds"]=NSEED
    p=m._make_pf_params(raw,k);p["label"]=lab;p["_raw"]=m.PF_LIB[lab];P_SETS.append(p)
# well集合(先頭NW, 有効なもの)
wells=sorted({os.path.basename(q).split("__")[0] for q in glob.glob(str(TR/"*__horizontal_well.csv"))})
paths=[];names=[]
for w in wells:
    if len(names)>=NW: break
    hp=TR/f"{w}__horizontal_well.csv";tp=TR/f"{w}__typewell.csv"
    try: hw=pd.read_csv(hp);tw=pd.read_csv(tp).sort_values("TVT")
    except: continue
    if "TVT" not in hw.columns: continue
    if m.build_smoother_inputs(hw,tw["TVT"].to_numpy(float),tw["GR"].to_numpy(float),P_SETS[0]["_raw"]) is None: continue
    if w not in m._V75_SP: continue   # 錨があるwellだけ(比較のため)
    names.append(w);paths.append((hp,tp))
print(f"測定well={len(names)} NSEED={NSEED} LAG={LAG}(0=非smooth)")
# 真TVT(eval行)
TRUE={}
for w,(hp,tp) in zip(names,paths):
    hw=pd.read_csv(hp);ti=hw["TVT_input"].to_numpy(float);tvt=hw["TVT"].to_numpy(float)
    hps=int(np.where(np.isfinite(ti))[0].max());TRUE[w]=tvt[hps+1:]
def run(withstruct):
    res={}
    for p in P_SETS:
        Pk={**p["_raw"],"smooth_lag":LAG,"n_seeds":NSEED}
        inps=[]
        for w,(hp,tp) in zip(names,paths):
            hw=pd.read_csv(hp);tw=pd.read_csv(tp).sort_values("TVT")
            x=m.build_smoother_inputs(hw,tw["TVT"].to_numpy(float),tw["GR"].to_numpy(float),Pk)
            x=m._attach_anchor(x,w,physics=False)          # 錨OFF(anc=0)
            if withstruct: x=m._attach_v75(x,w,use_sim=False)  # ★GRフリー structgate rtq/rtk
            inps.append(x)
        outs=m.run_smoother_bank(inps,Pk,1,m.SEED_V26 if hasattr(m,"SEED_V26") else 12345,NSEED,min(len(names),8))
        res[p["label"]]=outs
    return res
def rmse_of(res):
    out={}
    for lab,outs in res.items():
        e=[]
        for i,w in enumerate(names):
            mn=np.asarray(outs[i]["mean"],float);tv=TRUE[w];n=min(len(mn),len(tv))
            e.append(mn[:n]-tv[:n])
        e=np.concatenate(e);out[lab]=float(np.sqrt(np.mean(e[np.isfinite(e)]**2)))
    return out
print("錨なしPF実行...",flush=True); r0=rmse_of(run(False))
print("GRフリー錨ありPF実行...",flush=True); r1=rmse_of(run(True))
print("\n[structgate(GRフリー錨) 改善]  pf: 錨なし → 錨あり (Δ)")
for lab in m.PF_BANK:
    print(f"  {lab}: {r0[lab]:.3f} → {r1[lab]:.3f}  ({r1[lab]-r0[lab]:+.3f})")
allw0=np.mean([r0[l] for l in m.PF_BANK]);allw1=np.mean([r1[l] for l in m.PF_BANK])
print(f"  5本平均: {allw0:.3f} → {allw1:.3f} ({allw1-allw0:+.3f})")
