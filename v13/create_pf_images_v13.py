"""
v13 PF波形ビルダー(pf_1/2/3) ―― 2D-U-Net 用の生波形を作る(GPU・1回だけ)
================================================================================

各PFセット(pf_1/2/3)で 順方向PF(32本)+ 逆方向PF(各終端から4本=128本)を実行し、
**per-seed 生波形をすべて保存**(後で画像化・解析に使い回せる)。画像化は build_images_v13.py。

保存(各坑井 v13/artifacts_v13/img/{well}.npz):
  - 各セットの生波形: fwd_traj_pf_1/2/3 (32,T), bwd_traj_pf_1/2/3 (128,T)  ※絶対TVT, float16
  - alias: fwd_traj/bwd_traj = pf_1(既存の可視化/解析と互換)
  - center (T): pf_1 の順方向アンサンブル平均(=バンド中心, 全セット共通)
  - target_rel (T, train): 真TVT − center / ev_index / last_tvt

設計:
  - 自己完結(v9 exec/FI/DI 不要)。MD,X,Y,Z,GR,TVT_input + typewell だけ → test適用可。
  - メモリ対策: 坑井チャンク内で3セットを順に処理 → 坑井ごとに保存(全坑井を一度に持たない)。
  - 逆方向: MD反転、初期=順方向終端(pos, -rate)。

使い方:
  rogii_claude/.venv/Scripts/python.exe v13/create_pf_images_v13.py
"""

import os, time, glob
from pathlib import Path
import numpy as np
import pandas as pd
import torch

# ============================== CONFIG ==============================
PROJECT  = Path(r"C:\Users\kosaka256\Documents\rogii_claude")
DATA_DIR = PROJECT / "rogii-wellbore-geology-prediction" / "train"   # test時は .../test
OUT_DIR  = PROJECT / "v13" / "artifacts_v13" / "img"                 # 生波形の保存先(build_images の RAW_DIR と一致)
IS_TRAIN = True

N_WELLS     = None
PF_SETS     = ["pf_1", "pf_2", "pf_3"]   # 使うPFセット
FWD_SEEDS   = 32
BWD_PER_SEED= 4                          # 逆方向 = FWD_SEEDS * BWD_PER_SEED 本
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE       = torch.float32
SEED        = 0
WELL_CHUNK  = 260
# ===================================================================


PF_PARAM_SETS_RAW = {
    "pf_1": {"gr_power":2.1947763195255123,"gr_sig_def":114.11439615641156,"gr_sig_max":32.79346108797473,
             "gr_sig_min":7.495879205722816,"gr_sig_mult":2.7482077023565075,"hgr_smooth_r":3,
             "init_pos_std":0.43824435375436277,"init_rate_std":0.001840222365629066,"jump_prob":0.002,
             "jump_std":5.0,"likelihood_scale":20.0,"mom":0.998833416480357,"n_particles":600,"n_seeds":32,
             "pn":0.006788762926604484,"rate_clip":2.0,"resamp":0.060642973044944336,"rp":0.6864944651997384,
             "rr":0.00015873788792956448,"tvt_clip_margin":113.08092552242186,"tw_gr_smooth_r":0,
             "vn":0.0013871256161405515},
    "pf_2": {"gr_sig_def":71.37105679394953,"gr_sig_max":44.7373292550486,"gr_sig_min":6.520200142977573,
             "gr_sig_mult":1.9000654781073127,"init_pos_std":1.1039311766848976,"init_rate_std":0.009213697911348712,
             "likelihood_scale":18.584958539746772,"mom":0.9994265753132063,"n_particles":600,"n_seeds":32,
             "pn":0.011257078147676399,"resamp":0.30834451379977984,"rp":0.49711567429512854,
             "rr":0.00031680690540158854,"tvt_clip_margin":63.752292532272826,"vn":0.0010650490363481595,
             "rate_clip":0.0,"gr_power":2.0,"hgr_smooth_r":0,"tw_gr_smooth_r":0,"jump_prob":0.0,"jump_std":10.0},
    "pf_3": {"gr_power":2.074149546524975,"gr_sig_def":66.97913258656452,"gr_sig_max":50.843656140023995,
             "gr_sig_min":9.427442041004895,"gr_sig_mult":2.4485083695548218,"hgr_smooth_r":0,
             "init_pos_std":2.3279966476603975,"init_rate_std":0.03938758391370206,"jump_prob":0.002,
             "jump_std":5.0,"likelihood_scale":14.890876838217093,"mom":0.9999769474035675,"n_particles":600,
             "n_seeds":32,"pn":0.04394815893193551,"rate_clip":0.0,"resamp":0.054606961294962585,
             "rp":0.42322457711970995,"rr":0.002148810696831248,"tvt_clip_margin":91.4794941498565,
             "tw_gr_smooth_r":0,"vn":0.0006883646035942229},
}

def make_p(r):
    p = {}
    p["ANCC_N"]=int(r.get("n_particles",600)); p["PF_LIKELIHOOD_SCALE"]=float(r.get("likelihood_scale",20.0))
    p["PF_INIT_SPR"]=float(r.get("init_pos_std",0.5)); p["PF_INIT_V_STD"]=float(r.get("init_rate_std",0.01))
    p["PF_MOM"]=float(r.get("mom",0.999)); p["PF_VN"]=float(r.get("vn",0.001)); p["PF_PN"]=float(r.get("pn",0.005))
    p["PF_ROUGH_P"]=float(r.get("rp",0.5)); p["PF_ROUGH_V"]=float(r.get("rr",0.001)); p["PF_RESAMP"]=float(r.get("resamp",0.5))
    p["PF_GR_SIG_MIN"]=float(r.get("gr_sig_min",5.0)); p["PF_GR_SIG_MAX"]=float(r.get("gr_sig_max",50.0))
    p["PF_GR_SIG_DEF"]=float(r.get("gr_sig_def",100.0)); p["PF_GR_SIG_MULT"]=float(r.get("gr_sig_mult",2.0))
    p["PF_TVT_CLIP_MARGIN"]=float(r.get("tvt_clip_margin",50.0)); p["PF_RATE_CLIP"]=float(r.get("rate_clip",0.0))
    p["PF_GR_POWER"]=float(r.get("gr_power",2.0)); p["PF_HGR_SMOOTH_R"]=int(r.get("hgr_smooth_r",0))
    p["PF_TW_GR_SMOOTH_R"]=int(r.get("tw_gr_smooth_r",0)); p["PF_JUMP_PROB"]=float(r.get("jump_prob",0.0))
    p["PF_JUMP_STD"]=float(r.get("jump_std",0.0))
    p["ANCC_ALPHA"]=p["PF_MOM"]; p["ANCC_RN"]=p["PF_VN"]; p["ANCC_PN"]=p["PF_PN"]
    p["ANCC_IR"]=p["PF_INIT_V_STD"]; p["ANCC_IS"]=p["PF_INIT_SPR"]; p["ANCC_RP"]=p["PF_ROUGH_P"]; p["ANCC_RR"]=p["PF_ROUGH_V"]
    return p


# ----------------------- 入力作成(自己完結, セット別)-----------------------
def _smooth_radius_values(vals, fb, r):
    r = int(r); s = pd.Series(vals, dtype="float32").interpolate(limit_direction="both").fillna(float(fb))
    return s.to_numpy(np.float32) if r <= 0 else s.rolling(2*r+1, center=True, min_periods=1).mean().to_numpy(np.float32)

def _grid(tw_tvt, tw_gr, step=0.2):
    tmin=float(tw_tvt.min()); tmax=float(tw_tvt.max()); tvt_g=np.arange(tmin,tmax+step,step)
    return np.interp(tvt_g,tw_tvt,tw_gr).astype(np.float64), float(tmin), float(step)

def _gr_sig(hw, tw_tvt, tw_gr, p):
    gr_fb=float(np.nanmean(tw_gr)) if np.isfinite(np.nanmean(tw_gr)) else 0.
    hgr=_smooth_radius_values(hw["GR"].astype(float).to_numpy(),gr_fb,p["PF_HGR_SMOOTH_R"])
    kn=hw[hw["TVT_input"].notna()]
    if len(kn)<20: gs=float(p["PF_GR_SIG_DEF"])
    else:
        kpos=hw.index.get_indexer(kn.index); resid=hgr[kpos]-np.interp(kn["TVT_input"].values,tw_tvt,tw_gr)
        gs=float(np.nanstd(resid))
        if not np.isfinite(gs) or gs<=0: gs=float(p["PF_GR_SIG_DEF"])
    return float(np.clip(gs*p["PF_GR_SIG_MULT"],p["PF_GR_SIG_MIN"],p["PF_GR_SIG_MAX"]))

def build_inputs(hw, tw_tvt, tw_gr, p, is_train):
    tw_fb=float(np.nanmean(tw_gr)) if np.isfinite(np.nanmean(tw_gr)) else 0.
    tw_gr_pf=_smooth_radius_values(tw_gr,tw_fb,p["PF_TW_GR_SMOOTH_R"]).astype(np.float64)
    gs=_gr_sig(hw,tw_tvt,tw_gr_pf,p)
    kn=hw[hw["TVT_input"].notna()]; ev=hw[hw["TVT_input"].isna()]
    if len(ev)==0: return None
    ls=float(kn["TVT_input"].iloc[-1]+kn["Z"].iloc[-1]); last_tvt=float(kn["TVT_input"].iloc[-1])
    tail=kn.tail(30); dt=np.diff(tail["TVT_input"].values); dz=np.diff(tail["Z"].values); dm=np.diff(tail["MD"].values); m=dm>0
    ir=float(np.median((dt+dz)[m]/dm[m])) if m.sum()>=3 else 0.
    gg,gmin,gst=_grid(tw_tvt,tw_gr_pf)
    md_ev=ev["MD"].values.astype(np.float64); z_ev=ev["Z"].values.astype(np.float64)
    gr_full=hw["GR"].astype(float).interpolate(limit_direction="both").fillna(tw_fb).to_numpy(float)
    gr_proc=_smooth_radius_values(gr_full,tw_fb,p["PF_HGR_SMOOTH_R"])
    ev_pos=hw.index.get_indexer(ev.index); gr_ev=gr_proc[ev_pos].astype(np.float64)
    tvt_true = ev["TVT"].values.astype(np.float64) if (is_train and "TVT" in ev.columns) else None
    return dict(md=md_ev, z=z_ev, gr=gr_ev, gg=gg, gmin=gmin, gst=gst, gs=gs, ls=ls, ir=ir,
                N=int(p["ANCC_N"]), ev_index=ev.index.to_numpy(), last_tvt=last_tvt, tvt_true=tvt_true)


# ----------------------- GPU PF コア -----------------------
def _pf_core(md, z, gr, valid, grid, glen, vmin, step, gs, ls, ir, p, N, device, gen, dtype=DTYPE):
    B, T = md.shape
    ALPHA=p["ANCC_ALPHA"]; RN=p["ANCC_RN"]; PN=p["ANCC_PN"]; IR=p["ANCC_IR"]; IS=p["ANCC_IS"]
    RP=p["ANCC_RP"]; RR=p["ANCC_RR"]; RESAMP=p["PF_RESAMP"]; RATE_CLIP=p["PF_RATE_CLIP"]
    GR_POWER=p["PF_GR_POWER"]; JUMP_PROB=p["PF_JUMP_PROB"]; JUMP_STD=p["PF_JUMP_STD"]; CLIP=p["PF_TVT_CLIP_MARGIN"]
    def rn(): return torch.randn((B,N),generator=gen,device=device,dtype=dtype)
    pos=ls[:,None]+IS*rn(); rate=ir[:,None]+IR*rn()
    w=torch.full((B,N),1.0/N,device=device,dtype=dtype)
    log_lik=torch.zeros(B,device=device,dtype=torch.float64)
    pts=torch.zeros((B,T),device=device,dtype=dtype)
    tvt_lo=vmin-CLIP; tvt_hi=vmin+(glen.to(dtype)-1)*step+CLIP
    glast=torch.gather(grid,1,(glen-1).clamp_min(0).unsqueeze(1)); g0col=grid[:,0:1]
    pm=md[:,0]-1.0; arN=torch.arange(N,device=device,dtype=dtype); zero=torch.zeros((),device=device,dtype=dtype)
    for i in range(T):
        act=valid[:,i]; cur=md[:,i]; dm=(cur-pm).clamp_min(1.0)
        rate_n=ALPHA*rate+RN*rn()
        if RATE_CLIP>0: rate_n=rate_n.clamp(-RATE_CLIP,RATE_CLIP)
        pos_n=pos+rate_n*dm[:,None]+PN*rn()
        if JUMP_PROB>0 and JUMP_STD>0:
            jm=torch.rand((B,N),generator=gen,device=device,dtype=dtype)<JUMP_PROB
            pos_n=pos_n+torch.where(jm,JUMP_STD*rn(),zero)
        zi=z[:,i][:,None]; tvt=(pos_n-zi).clamp(tvt_lo[:,None],tvt_hi[:,None]); pos_n=tvt+zi
        am=act[:,None]; pos=torch.where(am,pos_n,pos); rate=torch.where(am,rate_n,rate)
        gri=gr[:,i]; obs=act & ~torch.isnan(gri)
        v=pos-zi; ii=(v-vmin[:,None])/step[:,None]; i0=torch.floor(ii).long()
        below=i0<0; above=i0>=(glen-1)[:,None]; i0c=i0.clamp_min(0); i0c=torch.minimum(i0c,(glen-2).clamp_min(0)[:,None])
        t=ii-i0c.to(dtype); gA=torch.gather(grid,1,i0c); gB=torch.gather(grid,1,i0c+1)
        eg=gA*(1-t)+gB*t; eg=torch.where(below,g0col,eg); eg=torch.where(above,glast,eg)
        d=((gri[:,None]-eg)/gs[:,None]).abs(); dp=d**GR_POWER
        lk=torch.where(dp<600.0,torch.exp(-0.5*dp),zero).clamp_min(1e-300)
        avg_lk=(w*lk).sum(1)
        log_lik=log_lik+torch.where(obs,torch.log(avg_lk.clamp_min(1e-300).double()),torch.zeros((),device=device,dtype=torch.float64))
        w_new=w*lk; ws=w_new.sum(1,keepdim=True)
        w_upd=torch.where(ws>0,w_new/ws.clamp_min(1e-300),torch.full_like(w,1.0/N)); w=torch.where(obs[:,None],w_upd,w)
        ne=(w*w).sum(1); need=act & ((1.0/ne)<(RESAMP*N))
        if bool(need.any()):
            cumw=torch.cumsum(w,1); u0=torch.rand((B,1),generator=gen,device=device,dtype=dtype)*(1.0/N)
            u=u0+arN[None,:]/N; idx=torch.searchsorted(cumw,u,right=False).clamp(0,N-1)
            pos_rs=torch.gather(pos,1,idx)+RP*rn(); rate_rs=torch.gather(rate,1,idx)+RR*rn()
            if RATE_CLIP>0: rate_rs=rate_rs.clamp(-RATE_CLIP,RATE_CLIP)
            nm=need[:,None]; pos=torch.where(nm,pos_rs,pos); rate=torch.where(nm,rate_rs,rate); w=torch.where(nm,torch.full_like(w,1.0/N),w)
        pts[:,i]=(w*(pos-zi)).sum(1); pm=torch.where(act,cur,pm)
    return pts, log_lik, (w*pos).sum(1), (w*rate).sum(1)

def _pad_wells(inps, device, reverse=False, dtype=DTYPE):
    W=len(inps); Tmax=max(len(x["md"]) for x in inps); Gmax=max(len(x["gg"]) for x in inps)
    md=torch.zeros((W,Tmax),dtype=dtype); z=torch.zeros((W,Tmax),dtype=dtype)
    gr=torch.full((W,Tmax),float("nan"),dtype=dtype); valid=torch.zeros((W,Tmax),dtype=torch.bool)
    grid=torch.zeros((W,Gmax),dtype=dtype); glen=torch.zeros(W,dtype=torch.long)
    vmin=torch.zeros(W,dtype=dtype); step=torch.zeros(W,dtype=dtype); gs=torch.zeros(W,dtype=dtype)
    ls=torch.zeros(W,dtype=dtype); ir=torch.zeros(W,dtype=dtype)
    for b,x in enumerate(inps):
        Tn=len(x["md"]); G=len(x["gg"]); mdv=x["md"].astype("float32"); zv=x["z"].astype("float32"); grv=x["gr"].astype("float32")
        if reverse: mdv=-mdv[::-1].copy(); zv=zv[::-1].copy(); grv=grv[::-1].copy()
        md[b,:Tn]=torch.from_numpy(mdv)
        if Tn<Tmax: md[b,Tn:]=md[b,Tn-1]
        z[b,:Tn]=torch.from_numpy(zv); gr[b,:Tn]=torch.from_numpy(grv); valid[b,:Tn]=True
        grid[b,:G]=torch.from_numpy(x["gg"].astype("float32"))
        if G<Gmax: grid[b,G:]=grid[b,G-1]
        glen[b]=G; vmin[b]=x["gmin"]; step[b]=x["gst"]; gs[b]=x["gs"]; ls[b]=x["ls"]; ir[b]=x["ir"]
    to=lambda tt: tt.to(device)
    return dict(md=to(md),z=to(z),gr=to(gr),valid=to(valid),grid=to(grid),glen=to(glen),
                vmin=to(vmin),step=to(step),gs=to(gs),ls=to(ls),ir=to(ir),Tmax=Tmax)

def _ens_weights(ll, scale):
    lk=ll-np.nanmax(ll); ww=np.exp(lk/max(scale,1e-6)); return ww/max(ww.sum(),1e-300)


def run_fwd(inps_sub, p, device, seed):
    W=len(inps_sub); S=FWD_SEEDS; pad=_pad_wells(inps_sub, device, reverse=False)
    rep=lambda tt: tt.repeat_interleave(S,0); gen=torch.Generator(device=device); gen.manual_seed(seed)
    pts,ll,posf,ratef=_pf_core(rep(pad["md"]),rep(pad["z"]),rep(pad["gr"]),rep(pad["valid"]),rep(pad["grid"]),
        pad["glen"].repeat_interleave(S),pad["vmin"].repeat_interleave(S),pad["step"].repeat_interleave(S),
        pad["gs"].repeat_interleave(S),pad["ls"].repeat_interleave(S),pad["ir"].repeat_interleave(S),
        p,int(p["ANCC_N"]),device,gen)
    return (pts.view(W,S,-1).float().cpu().numpy().astype(np.float16),
            ll.view(W,S).cpu().numpy(), posf.view(W,S).float().cpu().numpy(), ratef.view(W,S).float().cpu().numpy())

def run_bwd(inps_sub, tp, tr, p, device, seed):
    W=len(inps_sub); M=FWD_SEEDS*BWD_PER_SEED; pad=_pad_wells(inps_sub, device, reverse=True)
    rep=lambda tt: tt.repeat_interleave(M,0)
    tpv=torch.from_numpy(tp.reshape(-1)).to(device,DTYPE).repeat_interleave(BWD_PER_SEED)
    trv=torch.from_numpy(tr.reshape(-1)).to(device,DTYPE).repeat_interleave(BWD_PER_SEED)
    gen=torch.Generator(device=device); gen.manual_seed(seed)
    pts,_,_,_=_pf_core(rep(pad["md"]),rep(pad["z"]),rep(pad["gr"]),rep(pad["valid"]),rep(pad["grid"]),
        pad["glen"].repeat_interleave(M),pad["vmin"].repeat_interleave(M),pad["step"].repeat_interleave(M),
        pad["gs"].repeat_interleave(M), tpv, -trv, p, int(p["ANCC_N"]), device, gen)
    return pts.view(W,M,-1).float().cpu().numpy().astype(np.float16)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ps = {name: make_p(PF_PARAM_SETS_RAW[name]) for name in PF_SETS}
    scale1 = float(ps[PF_SETS[0]]["PF_LIKELIHOOD_SCALE"])
    print(f"device={DEVICE}  sets={PF_SETS}  fwd={FWD_SEEDS} bwd={FWD_SEEDS*BWD_PER_SEED}  chunk={WELL_CHUNK}")
    if DEVICE=="cuda": print("GPU:", torch.cuda.get_device_name(0))

    wells = sorted({os.path.basename(q).split("__")[0] for q in glob.glob(str(DATA_DIR / "*__horizontal_well.csv"))})
    if N_WELLS is not None: wells = wells[:N_WELLS]

    # 各坑井×各セットの入力を事前作成(+meta)
    print(f"[inputs] 入力作成中 ({len(wells)} 坑井 × {len(PF_SETS)} セット)...")
    inps_by_set = {name: [] for name in PF_SETS}; names = []; meta = []
    for w in wells:
        hp = DATA_DIR / f"{w}__horizontal_well.csv"; tp = DATA_DIR / f"{w}__typewell.csv"
        try:
            hw = pd.read_csv(hp); tw = pd.read_csv(tp).sort_values("TVT")
        except Exception:
            continue
        if IS_TRAIN and "TVT" not in hw.columns: continue
        tw_tvt = tw["TVT"].to_numpy(float); tw_gr = tw["GR"].to_numpy(float)
        dset = {name: build_inputs(hw, tw_tvt, tw_gr, ps[name], IS_TRAIN) for name in PF_SETS}
        if any(v is None for v in dset.values()): continue
        for name in PF_SETS: inps_by_set[name].append(dset[name])
        i0 = dset[PF_SETS[0]]
        meta.append(dict(ev_index=i0["ev_index"], last_tvt=i0["last_tvt"], tvt_true=i0["tvt_true"], T=len(i0["md"])))
        names.append(w)
    print(f"[inputs] 使用坑井: {len(names)}")

    t0 = time.perf_counter()
    for c in range(0, len(names), WELL_CHUNK):
        sub_names = names[c:c+WELL_CHUNK]; W = len(sub_names)
        set_fwd = {}; set_bwd = {}; ll_pf1 = None
        for name in PF_SETS:
            inps_sub = inps_by_set[name][c:c+WELL_CHUNK]
            pts_f, ll, tp_, tr_ = run_fwd(inps_sub, ps[name], DEVICE, SEED)
            pts_b = run_bwd(inps_sub, tp_, tr_, ps[name], DEVICE, SEED)
            set_fwd[name] = pts_f; set_bwd[name] = pts_b
            if name == PF_SETS[0]: ll_pf1 = ll
        if DEVICE == "cuda": torch.cuda.synchronize()
        for j in range(W):
            mj = meta[c+j]; T = mj["T"]
            fwd1 = set_fwd[PF_SETS[0]][j, :, :T].astype(np.float32)
            ww = _ens_weights(ll_pf1[j], scale1)
            center = (ww[:, None] * fwd1).sum(0).astype(np.float32)   # pf_1 アンサンブル=バンド中心
            out = dict(center=center, ev_index=mj["ev_index"].astype(np.int32), last_tvt=np.float32(mj["last_tvt"]))
            for name in PF_SETS:
                out[f"fwd_traj_{name}"] = set_fwd[name][j, :, :T]
                out[f"bwd_traj_{name}"] = set_bwd[name][j, :, :T][:, ::-1].copy()   # 順方向行順に戻す
            out["fwd_traj"] = out[f"fwd_traj_{PF_SETS[0]}"]; out["bwd_traj"] = out[f"bwd_traj_{PF_SETS[0]}"]  # alias
            if mj["tvt_true"] is not None:
                out["target_rel"] = (mj["tvt_true"] - center).astype(np.float32)
            np.savez_compressed(OUT_DIR / f"{sub_names[j]}.npz", **out)
        print(f"  chunk {c//WELL_CHUNK}: wells {c}..{c+W-1} done ({time.perf_counter()-t0:.0f}s)")

    print(f"\nsaved {len(names)} wells -> {OUT_DIR}  (sets={PF_SETS})")

if __name__ == "__main__":
    main()
