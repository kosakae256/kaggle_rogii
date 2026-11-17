# -*- coding: utf-8 -*-
"""v101 実験: 単体PFの seed=32 集約を「平均」でなく「2枝(mode)選択」にしたら精度が上がるか。
smooth/full は論点外。素の単体PF(pf_1, tw, emission無し)で per-seed 軌跡を capture し、
  baseline   = seed 尤度加重平均(現行)
  dominant   = 2枝クラスタで質量の大きいmodeを取る(ラベル不要=achievable)
  minspread  = 2枝クラスタで分散の小さいmodeを取る(achievable)
  oracle2    = 2枝のうち真値に近い方を取る(天井=枝が分かればここまで)
を overall(全eval行プール)RMSE で比較。forward-filter(pfw)と smoothed(ps)の両方で。
「2枝が実在するか(双峰井比率/gap分布)」も出す。実行:
  .venv/Scripts/python.exe v101/exp_seed_modes_v101.py   (env: BANK=pf_1 NWELLS=0(=全) TAU=5.0)
"""
import os, sys, glob
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "v99"))
import pf_banks_v95 as pf

BANK   = os.environ.get("BANK", "pf_1")
NWELLS = int(os.environ.get("NWELLS", "0"))          # 0=全井
NSEED  = int(os.environ.get("NSEED", "32"))
TAU    = float(os.environ.get("TAU", "5.0"))         # 双峰と見なす mode間gap 閾値(ft)
MASS_FLOOR = float(os.environ.get("MASS_FLOOR", "0.10"))
EMISSION = os.environ.get("EMISSION", "0") == "1"    # ★NN-emission ON/OFF
W_NN_ENV = os.environ.get("W_NN")                    # 未指定ならバンク固有 P['_w_nn'](pfA=0.01等)
SIM_PKL = os.environ.get("SIM_PKL", str(pf.PROJ / "v99" / "artifacts_v99" / "v97" / "sim_grfree_v97.pkl"))
os.environ.setdefault("PF_WELL_CHUNK", "48")

TR = pf.DATA_DIR / "train"
wids = sorted({os.path.basename(q).split("__")[0] for q in glob.glob(str(TR / "*__horizontal_well.csv"))})
if NWELLS > 0:
    wids = wids[:NWELLS]

P = pf.bank_param(BANK)
w_nn_use = (float(W_NN_ENV) if W_NN_ENV is not None else float(P["_w_nn"])) if EMISSION else 0.0
SIMD = {}
if EMISSION:
    import pickle
    with open(SIM_PKL, "rb") as _f:
        SIMD = pickle.load(_f)
    print(f"[exp] NN-emission ON: sim={len(SIMD)}井 <- {os.path.basename(SIM_PKL)}  w_nn={w_nn_use}", flush=True)
print(f"[exp] bank={BANK} physics={P['_physics']} emission={EMISSION} w_nn={w_nn_use} n_seed={NSEED} 井={len(wids)} tau={TAU}", flush=True)

inps, trues, keep = [], [], []
for w in wids:
    hw = pd.read_csv(TR / f"{w}__horizontal_well.csv")
    tw = pd.read_csv(TR / f"{w}__typewell.csv").sort_values("TVT")
    tt = tw["TVT"].to_numpy(float); tg = tw["GR"].to_numpy(float)
    m = np.isfinite(tt) & np.isfinite(tg)
    if m.sum() < 8:
        continue
    x = pf.build_smoother_inputs(hw, tt[m], tg[m], P)
    if x is None:
        continue
    pf.attach_anchor(x, w, P["_physics"])
    if EMISSION:
        sd = SIMD.get(w)
        if sd is not None and len(sd.get("st", [])) == len(x["md"]):
            x["_sim"] = sd["sim"]; x["_st"] = sd["st"]
    ev = hw[hw["TVT_input"].isna()]
    inps.append(x); trues.append(ev["TVT"].to_numpy(float)); keep.append(w)

nemit = sum(1 for x in inps if "_sim" in x)
# ---- capture キャッシュ(PFを毎回回さず解析を反復するため) ----
import pickle as _pk
_tagc = f"{BANK}_{'emit' if EMISSION else 'noemit'}_s{NSEED}_n{len(inps)}"
CACHE = pf.PROJ / "artifacts_v101" / f"cap_{_tagc}.pkl"
if CACHE.exists() and os.environ.get("RECAPTURE", "0") != "1":
    print(f"[exp] capture キャッシュ load <- {CACHE.name}", flush=True)
    with open(CACHE, "rb") as _f:
        cap = _pk.load(_f)
    res = cap["res"]; trues = cap["trues"]; keep = cap["keep"]
else:
    print(f"[exp] PF実行(capture) 井={len(inps)}  emission添付={nemit} ...", flush=True)
    res = pf.run_smoother_ext(inps, P, seed=12345, n_seeds=NSEED, chunk=48, w_nn=w_nn_use, capture=True)
    slim = [dict(ps=r["ps"], pfw=r["pfw"], ww=r["ww"]) for r in res]   # 解析に必要な最小限
    with open(CACHE, "wb") as _f:
        _pk.dump(dict(res=slim, trues=trues, keep=keep), _f, protocol=4)
    print(f"[exp] capture キャッシュ save -> {CACHE.name}", flush=True)


def split_two(u, ww):
    """1D尤度加重 最適2分割(閾値)。戻り: (idxA, idxB) seed index配列。"""
    order = np.argsort(u); us = u[order]; ws = ww[order]; S = len(u)
    best = None; best_wcss = np.inf
    for k in range(1, S):
        wa, wb = ws[:k], ws[k:]; sa, sb = wa.sum(), wb.sum()
        if sa <= 1e-9 or sb <= 1e-9:
            continue
        ma = (wa * us[:k]).sum() / sa; mb = (wb * us[k:]).sum() / sb
        wcss = (wa * (us[:k] - ma) ** 2).sum() + (wb * (us[k:] - mb) ** 2).sum()
        if wcss < best_wcss:
            best_wcss = wcss; best = k
    if best is None:
        return order, np.array([], int)
    return order[:best], order[best:]


ANCH = pf._ANCHOR   # GRフリー錨 {wid:{tvt,sig}}(独立第2推定器=枝判別の外部信号候補)


TAUS = float(os.environ.get("TAUS", "6.0"))    # per-seed 錨ソフト加重 温度(ft)


def analyze(traj_key):
    """traj_key='pfw'(forward filter) or 'ps'(smoothed)。各井の per-seed 軌跡でmode/per-seed解析。"""
    e_base, e_dom, e_msp, e_orc, e_anc = [], [], [], [], []   # 全eval行の (pred-true)
    e_pso, e_psll, e_psanc, e_pssoft, e_pscmb = [], [], [], [], []   # per-seed 選択系
    rowbest = []                                              # per-row oracle(絶対誤差)
    per = []                                              # per-well 記録
    for k in range(len(inps)):
        r = res[k]; true = trues[k]
        traj = r[traj_key].astype(np.float64)             # (S,T)
        ww = r["ww"].astype(np.float64); ww = ww / max(ww.sum(), 1e-12)
        S, T = traj.shape
        base = (ww[:, None] * traj).sum(0)                # seed加重平均(現行)
        u = traj.mean(1)                                  # per-seed level(eval平均)
        sp = ANCH.get(keep[k]); atvt = None
        if sp is not None and len(np.asarray(sp["tvt"], float)) == T:
            atvt = np.asarray(sp["tvt"], float)           # eval行に整列した錨TVT

        # ===== per-seed 選択(ユーザー狙い: per-seed oracle を取りに行く) =====
        rs = np.sqrt(((traj - true[None, :]) ** 2).mean(1))   # per-seed 井RMSE
        pso = traj[int(np.argmin(rs))]                        # per-seed oracle(井, 真値使用=天井)
        rowbest.append(np.min(np.abs(traj - true[None, :]), axis=0))   # per-row oracle
        psll = traj[int(np.argmax(ww))]                       # 最尤 seed(achievable)
        if atvt is not None:
            da = ((traj - atvt[None, :]) ** 2).mean(1)        # 各seedの錨からの距離
            psanc = traj[int(np.argmin(da))]                  # 錨最近 seed(achievable)
            wsoft = np.exp(-(da - da.min()) / (2 * TAUS ** 2)); wsoft = wsoft / wsoft.sum()
            pssoft = (wsoft[:, None] * traj).sum(0)           # 錨ソフト加重(achievable)
            wcmb = ww * np.exp(-(da - da.min()) / (2 * TAUS ** 2)); wcmb = wcmb / max(wcmb.sum(), 1e-12)
            pscmb = (wcmb[:, None] * traj).sum(0)             # 尤度×錨 合成ソフト加重(achievable)
        else:
            psanc = base; pssoft = base; pscmb = base
        e_pso.append(pso - true); e_psll.append(psll - true)
        e_psanc.append(psanc - true); e_pssoft.append(pssoft - true); e_pscmb.append(pscmb - true)
        A, B = split_two(u, ww)
        if len(B) == 0:
            dom = base; msp = base; orc = base; anc = base; gap = 0.0; mmin = 1.0
        else:
            wA = ww[A] / max(ww[A].sum(), 1e-12); wB = ww[B] / max(ww[B].sum(), 1e-12)
            cA = (wA[:, None] * traj[A]).sum(0); cB = (wB[:, None] * traj[B]).sum(0)
            mA = ww[A].sum(); mB = ww[B].sum()
            gap = abs((wB * u[B]).sum() - (wA * u[A]).sum())
            spA = np.sqrt(max((wA * (u[A] - (wA * u[A]).sum()) ** 2).sum(), 0.0))
            spB = np.sqrt(max((wB * (u[B] - (wB * u[B]).sum()) ** 2).sum(), 0.0))
            mmin = min(mA, mB)
            bimodal = (gap >= TAU) and (mmin >= MASS_FLOOR)
            if bimodal:
                dom = cA if mA >= mB else cB
                msp = cA if spA <= spB else cB
                rA = np.sqrt(np.mean((cA - true) ** 2)); rB = np.sqrt(np.mean((cB - true) ** 2))
                orc = cA if rA <= rB else cB
                if atvt is not None:                       # ★錨に近いmodeを選ぶ(achievable な独立信号)
                    dA = np.mean((cA - atvt) ** 2); dB = np.mean((cB - atvt) ** 2)
                    anc = cA if dA <= dB else cB
                else:
                    anc = base
            else:
                dom = base; msp = base; orc = base; anc = base
        e_base.append(base - true); e_dom.append(dom - true)
        e_msp.append(msp - true); e_orc.append(orc - true); e_anc.append(anc - true)
        rb = np.sqrt(np.mean((base - true) ** 2)); ro = np.sqrt(np.mean((orc - true) ** 2))
        per.append(dict(well=keep[k], n=T, gap=gap, mmin=mmin,
                        bimodal=(len(B) > 0 and gap >= TAU and mmin >= MASS_FLOOR),
                        rmse_base=rb, rmse_orc=ro))

    def ov(elist):
        e = np.concatenate(elist); e = e[np.isfinite(e)]
        return float(np.sqrt(np.mean(e ** 2)))
    dfp = pd.DataFrame(per)
    print(f"\n===== [{traj_key}] overall RMSE (全{len(inps)}井プール) =====", flush=True)
    print(f"  baseline (seed平均, 現行) = {ov(e_base):.4f}")
    print(f"  dominant (質量大mode)     = {ov(e_dom):.4f}   [achievable]")
    print(f"  minspread(分散小mode)     = {ov(e_msp):.4f}   [achievable]")
    print(f"  anchor   (錨に近いmode)   = {ov(e_anc):.4f}   [★achievable/独立信号]")
    print(f"  oracle2  (真値に近いmode) = {ov(e_orc):.4f}   [mode天井]")
    print(f"  --- per-seed(32本から選ぶ) ---")
    print(f"  ps_ll    (最尤seed)       = {ov(e_psll):.4f}   [achievable]")
    print(f"  ps_anchor(錨最近seed)     = {ov(e_psanc):.4f}   [★achievable/独立信号]")
    print(f"  ps_soft  (錨ソフト加重)   = {ov(e_pssoft):.4f}   [★achievable/独立信号]")
    print(f"  ps_combo (尤度×錨 加重)   = {ov(e_pscmb):.4f}   [★achievable/合成]")
    print(f"  ps_oracle(井:最良seed)    = {ov(e_pso):.4f}   [★★per-seed天井(井)]")
    e_row = np.concatenate(rowbest); print(f"  ps_oracle(行:最良seed)    = {float(np.sqrt(np.mean(e_row[np.isfinite(e_row)]**2))):.4f}   [★★per-seed天井(行)]")
    nb = int(dfp["bimodal"].sum())
    print(f"  双峰井(gap>={TAU}ft & 両mass>={MASS_FLOOR}) = {nb}/{len(dfp)} ({100*nb/len(dfp):.1f}%)")
    if nb:
        sub = dfp[dfp["bimodal"]]
        gain = (sub["rmse_base"] - sub["rmse_orc"])
        print(f"  双峰井のみ: oracle2 の井別RMSE改善 mean={gain.mean():.3f} / median={gain.median():.3f} / max={gain.max():.3f}")
        print(f"  gap分布(双峰井): p50={sub['gap'].median():.1f} p90={sub['gap'].quantile(.9):.1f} max={sub['gap'].max():.1f} ft")
    return dfp, dict(base=ov(e_base), dom=ov(e_dom), msp=ov(e_msp), orc=ov(e_orc), nb=nb)


dfw, sw = analyze("pfw")     # forward filter = PF自体
dfs, ss = analyze("ps")      # smoothed(参考)

# ---- グラフ(Meiryo) ----
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "Meiryo"
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    # (1) per-well RMSE ECDF: baseline vs oracle2 (pfw)
    for col, lab, c in [("rmse_base", "baseline(seed平均)", "C0"), ("rmse_orc", "oracle2(枝選択)", "C3")]:
        v = np.sort(dfw[col].to_numpy()); ax[0].plot(v, np.linspace(0, 1, len(v)), label=lab, color=c)
    ax[0].set_title("pfw 井別RMSE ECDF"); ax[0].set_xlabel("well RMSE(ft)"); ax[0].set_ylabel("CDF")
    ax[0].set_xlim(0, np.percentile(dfw["rmse_base"], 98)); ax[0].legend(); ax[0].grid(alpha=.3)
    # (2) gap ヒスト
    ax[1].hist(dfw["gap"], bins=40, color="C1"); ax[1].axvline(TAU, color="k", ls="--", label=f"tau={TAU}")
    ax[1].set_title("mode間 gap 分布(pfw)"); ax[1].set_xlabel("gap(ft)"); ax[1].legend(); ax[1].grid(alpha=.3)
    # (3) 散布: baseline vs oracle2 井別RMSE(双峰井を赤)
    bm = dfw["bimodal"].to_numpy()
    ax[2].scatter(dfw["rmse_base"][~bm], dfw["rmse_orc"][~bm], s=8, alpha=.4, label="単峰", color="C0")
    ax[2].scatter(dfw["rmse_base"][bm], dfw["rmse_orc"][bm], s=14, alpha=.7, label="双峰", color="C3")
    lim = np.percentile(dfw["rmse_base"], 99); ax[2].plot([0, lim], [0, lim], "k--", lw=.8)
    ax[2].set_xlim(0, lim); ax[2].set_ylim(0, lim); ax[2].set_title("baseline vs oracle2(井別)")
    ax[2].set_xlabel("baseline RMSE"); ax[2].set_ylabel("oracle2 RMSE"); ax[2].legend(); ax[2].grid(alpha=.3)
    fig.suptitle(f"v101 単体PF seed集約 [{BANK}] 平均 vs 2枝  (pfw: base {sw['base']:.3f} / dominant {sw['dom']:.3f} / oracle2 {sw['orc']:.3f})")
    fig.tight_layout()
    _tag = f"{BANK}_emit" if EMISSION else BANK
    out = os.path.join(os.path.dirname(__file__), "..", "artifacts_v101", f"seed_modes_{_tag}.png")
    fig.savefig(out, dpi=110); print(f"\n[graph] saved {os.path.abspath(out)}")
except Exception as e:
    print("[graph] skip:", e)

print("\n[done]")
