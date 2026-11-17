# -*- coding: utf-8 -*-
"""dec10(MD-10間引きPF)調査: GRフリー錨で、pf強化(Student-t/tempering)と nn_emission が効くか。
v102エンジン(pf_banks_v95, ps_combo=off)+ MD-10 decimation + np.interp補間。普通PF(fixedlag)・~60井。
単体PF overall RMSE で config比較 → 効くレバー/emission可否を確定。
実行: NWELLS=60 .venv/Scripts/python.exe v102/exp_dec10_levers.py"""
import os, sys, glob, pickle
os.environ["PS_COMBO_TAU"] = "0"                     # ★調査は ps_combo OFF で純粋に見る
os.environ.setdefault("ROGII_ART95", os.path.join(os.path.dirname(__file__), "artifacts_v102", "v95"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import pf_banks_v95 as pf

DEC = 10
NWELLS = int(os.environ.get("NWELLS", "60"))
SIM_PKL = os.environ.get("SIM_PKL", str(pf.PROJ / "v99" / "artifacts_v99" / "v97" / "sim_grfree_v97.pkl"))
TR = pf.DATA_DIR / "train"
wids = sorted({os.path.basename(q).split("__")[0] for q in glob.glob(str(TR / "*__horizontal_well.csv"))})[:NWELLS]
print(f"[dec10] 井={len(wids)} DEC={DEC} 錨=GRフリー(mult12) sim={os.path.basename(SIM_PKL)}", flush=True)
SIMD = pickle.loads(open(SIM_PKL, "rb").read())


def agg1(x, s, how="mean"):
    x = np.asarray(x, float); n = len(x); cuts = list(range(0, n, s)) + [n]
    return np.array([x[a:b].mean() if how == "mean" else x[a:b][(b - a) // 2] for a, b in zip(cuts[:-1], cuts[1:])])


def agg2(X, s):  # (n,B) を行方向にbin平均
    X = np.asarray(X, float); n = len(X); cuts = list(range(0, n, s)) + [n]
    return np.stack([X[a:b].mean(0) for a, b in zip(cuts[:-1], cuts[1:])], 0)


def decimate(x, emission):
    n = len(x["md"])
    if n < DEC * 2:
        xd = dict(x); xd["_full_md"] = np.asarray(x["md"], float); xd["_dec_md"] = xd["_full_md"]; return xd
    xd = dict(md=agg1(x["md"], DEC), z=agg1(x["z"], DEC), gr=agg1(x["gr"], DEC),
              gg=x["gg"], gmin=x["gmin"], gst=x["gst"], gs=x["gs"], ls=x["ls"], ir=x["ir"], N=x["N"],
              anc=agg1(x["anc"], DEC), ancs=agg1(x["ancs"], DEC), _wid=x.get("_wid"))
    if emission and "_st" in x:
        xd["_st"] = agg1(x["_st"], DEC); xd["_sim"] = agg2(x["_sim"], DEC)
    xd["_full_md"] = np.asarray(x["md"], float); xd["_dec_md"] = xd["md"]
    return xd


def run_cfg(name, robust_nu=0.0, temper_beta=1.0, emission=False):
    P = pf.bank_param("pf_1")                        # base params(レバーはここで上書き)
    P["anchor_mult"] = 12.0; P["robust_nu"] = robust_nu; P["temper_beta"] = temper_beta
    P["smooth_mode"] = "fixedlag"; P["smooth_lag"] = 16      # ★調査=普通PF
    inps = []; trues = []; fmds = []
    for w in wids:
        hw = pd.read_csv(TR / f"{w}__horizontal_well.csv"); tw = pd.read_csv(TR / f"{w}__typewell.csv").sort_values("TVT")
        tt = tw["TVT"].to_numpy(float); tg = tw["GR"].to_numpy(float); m = np.isfinite(tt) & np.isfinite(tg)
        if m.sum() < 8: continue
        x = pf.build_smoother_inputs(hw, tt[m], tg[m], P)
        if x is None: continue
        pf.attach_anchor(x, w, True)                # ★GRフリー錨を使う(physics=True)
        if emission:
            sd = SIMD.get(w)
            if sd is not None and len(sd.get("st", [])) == len(x["md"]):
                x["_sim"] = sd["sim"]; x["_st"] = sd["st"]
        xd = decimate(x, emission)
        inps.append(xd); trues.append(hw[hw["TVT_input"].isna()]["TVT"].to_numpy(float)); fmds.append((xd["_dec_md"], xd["_full_md"]))
    w_nn = 0.02 if emission else 0.0
    try:
        res = pf.run_smoother_ext(inps, P, seed=12345, n_seeds=32, chunk=48, w_nn=w_nn)
    except Exception as e:
        print(f"  [{name}] ★実行失敗: {type(e).__name__}: {e}"); return None
    errs = []
    for i in range(len(inps)):
        dm, fm = fmds[i]; mean_dec = np.asarray(res[i]["mean"], float)
        mean_full = mean_dec if len(dm) == len(fm) else np.interp(fm, dm, mean_dec)
        t = trues[i]; nn = min(len(mean_full), len(t)); d = mean_full[:nn] - t[:nn]; errs.append(d[np.isfinite(d)])
    e = np.concatenate(errs); rmse = float(np.sqrt(np.mean(e ** 2)))
    print(f"  [{name:28s}] overall RMSE = {rmse:.3f}", flush=True)
    return rmse


print("\n===== dec10 レバー調査(単体PF overall RMSE, 低いほど良い) =====")
run_cfg("dec10 base(錨mult12)")
run_cfg("dec10 +Student-t(nu5)", robust_nu=5.0)
run_cfg("dec10 +tempering(b0.85)", temper_beta=0.85)
run_cfg("dec10 +emission", emission=True)
run_cfg("dec10 +St+emission(no temper)", robust_nu=5.0, emission=True)   # ★本命候補
print("\n[done]")
