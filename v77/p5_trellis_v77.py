# -*- coding: utf-8 -*-
"""
v77 P5 — (offset × dip) exact forward-backward trellis(決定的・乱数なし)
=========================================================================
P6との違い: 状態に「dip偏差」(K16錨からの傾き乖離, ft/行)を持ち、
  offset_{t+1} = offset_t + dip_t,  dip はゆっくり変化(persistence)。
= doc1のtrellis形(50ft節点)の行レベル連続版。オフセットが慣性を持つ=ジグザグ禁止。
emission: GR尤度(gauss^POW) × 錨prior(σ(dd))。出力=posterior mean。
使い方: NW=60 python v77/p5_trellis_v77.py
"""
import os, io, sys, glob, time, pickle
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
from pathlib import Path
import numpy as np, pandas as pd
import torch

P0 = Path(r"C:\Users\kosaka256\Documents\rogii_claude")
TRAIN = P0 / "rogii-wellbore-geology-prediction" / "train"
ART = P0 / "v77" / "artifacts_v77"; ART.mkdir(parents=True, exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
NW = int(os.environ.get("NW", "60"))
BAND = float(os.environ.get("BAND", "40"))
STEP = float(os.environ.get("STEP", "1.0"))       # offset格子(ft)
NDIP = int(os.environ.get("NDIP", "11"))          # dip偏差格子本数(奇数)
DIPMAX = float(os.environ.get("DIPMAX", "0.05"))  # dip偏差最大(ft/行)
SDIP = float(os.environ.get("SDIP", "0.2"))       # dip遷移σ(格子単位)
ANCW = float(os.environ.get("ANCW", "0.01"))
EMT = float(os.environ.get("EMT", "1.0"))         # emission温度(<1=軟化)
WSTRIDE = int(os.environ.get("WSTRIDE", "0"))
POW = float(os.environ.get("POW", "1.7826896851395453"))
SIGDIV = float(os.environ.get("SIGDIV", "1.0"))
GRSIG = dict(mn=11.865793233968713, mx=44.98927964612316, df=119.25397464636981, mult=3.058318309943017)
ANC = pickle.loads(Path(os.environ.get("ANCPKL", str(ART / "k16_anchor_train.pkl"))).read_bytes())

offs = torch.arange(-BAND, BAND + STEP / 2, STEP, device=DEV)
NO = len(offs)
dips = torch.linspace(-DIPMAX, DIPMAX, NDIP, device=DEV)          # ft/行
# dip→offset移動: 1行あたり dips ft。offset格子上の移動は非整数→2点分配(線形)
shift = dips / STEP
sh_lo = torch.floor(shift).long(); sh_frac = shift - sh_lo.float()
# dip遷移行列 (NDIP,NDIP): ガウス(σ=SDIP格子)
dgrid = torch.arange(NDIP, device=DEV, dtype=torch.float32)
Tdip = torch.exp(-0.5 * ((dgrid[:, None] - dgrid[None, :]) / SDIP) ** 2)
Tdip = Tdip / Tdip.sum(1, keepdim=True)


_ar = torch.arange(NO, device=DEV)[:, None]                       # (NO,1)
_srcA = _ar - sh_lo[None, :]                                       # (NO,NDIP) 移動元index
_srcB = _srcA - 1
_mA = (_srcA >= 0) & (_srcA < NO); _mB = (_srcB >= 0) & (_srcB < NO)
_gA = _srcA.clamp(0, NO - 1); _gB = _srcB.clamp(0, NO - 1)
_srcA2 = _ar + sh_lo[None, :]; _srcB2 = _srcA2 + 1
_mA2 = (_srcA2 >= 0) & (_srcA2 < NO); _mB2 = (_srcB2 >= 0) & (_srcB2 < NO)
_gA2 = _srcA2.clamp(0, NO - 1); _gB2 = _srcB2.clamp(0, NO - 1)
_fr = sh_frac[None, :]


def shift_offset(p):
    """p:(NO,NDIP)。dip列dごとにoffset方向へ shift[d] 移動(線形分配)。ベクトル化。"""
    A = p.gather(0, _gA) * _mA; B = p.gather(0, _gB) * _mB
    return (1 - _fr) * A + _fr * B


def run_well(wid):
    if wid not in ANC: return None
    hw = pd.read_csv(TRAIN / f"{wid}__horizontal_well.csv")
    tw = pd.read_csv(TRAIN / f"{wid}__typewell.csv").sort_values("TVT")
    kn = hw[hw["TVT_input"].notna()]; ev = hw[hw["TVT_input"].isna()]
    if len(ev) < 60 or len(kn) < 20: return None
    tw_tvt = tw["TVT"].to_numpy(float); tw_gr = tw["GR"].to_numpy(float)
    m = np.isfinite(tw_tvt) & np.isfinite(tw_gr); tw_tvt, tw_gr = tw_tvt[m], tw_gr[m]
    if len(tw_tvt) < 8: return None
    gr = hw["GR"].astype(float).interpolate(limit_direction="both").fillna(float(np.nanmean(tw_gr))).to_numpy(float)
    kpos = hw.index.get_indexer(kn.index); epos = hw.index.get_indexer(ev.index)
    resid = gr[kpos] - np.interp(kn["TVT_input"].to_numpy(float), tw_tvt, tw_gr)
    gs = float(np.nanstd(resid))
    if not np.isfinite(gs) or gs <= 0: gs = GRSIG["df"]
    gs = float(np.clip(gs * GRSIG["mult"], GRSIG["mn"], GRSIG["mx"])) / SIGDIV
    a = ANC[wid]
    T = min(len(ev), len(a["tvt"]))
    A = a["tvt"][:T]; sig_dd = a["sig"][:T]
    ge = gr[epos][:T]; tv = ev["TVT"].to_numpy(float)[:T]
    # emission (T,NO)
    At = torch.tensor(A, device=DEV, dtype=torch.float32)
    tvtr = At[:, None] + offs[None, :]
    ttv = torch.tensor(tw_tvt, device=DEV, dtype=torch.float32)
    tgr = torch.tensor(tw_gr, device=DEV, dtype=torch.float32)
    ii = torch.searchsorted(ttv, tvtr.clamp(ttv[0], ttv[-1]).contiguous()).clamp(1, len(ttv) - 1)
    wv = ((tvtr - ttv[ii - 1]) / (ttv[ii] - ttv[ii - 1]).clamp_min(1e-6)).clamp(0, 1)
    egr = tgr[ii - 1] * (1 - wv) + tgr[ii] * wv
    grt = torch.tensor(ge, device=DEV, dtype=torch.float32)
    d = ((grt[:, None] - egr) / gs).abs()
    loge_gr = EMT * (-0.5 * d ** POW)
    sdd = torch.tensor(sig_dd, device=DEV, dtype=torch.float32).clamp_min(1.0)
    loge_anc = -0.5 * ANCW * (offs[None, :] / sdd[:, None]) ** 2
    hook = globals().get("_NN_HOOK")
    if hook is not None and hook[0] == wid:                       # P8: NN emission注入
        _, lognn, wnn, repl = hook
        ln = lognn[:loge_gr.shape[0]].to(loge_gr.device)
        loge = (ln * wnn + loge_anc) if repl == "1" else (loge_gr + ln * wnn + loge_anc)
    else:
        loge = loge_gr + loge_anc
    E = torch.exp(loge - loge.max(1, keepdim=True).values)        # (T,NO) prob域
    # forward
    alphas = []
    p = torch.zeros(NO, NDIP, device=DEV)
    j0 = int(torch.argmin(offs.abs()))
    p[j0] = 1.0 / NDIP                                            # 開始: offset0, dip一様
    for t in range(T):
        p = p * E[t][:, None]
        p = p / p.sum().clamp_min(1e-30)
        alphas.append(p.clone())
        p = shift_offset(p @ Tdip)
    # backward
    betas = [None] * T
    b = torch.ones(NO, NDIP, device=DEV)
    for t in range(T - 1, -1, -1):
        betas[t] = b
        bb = b * E[t][:, None]
        # 逆遷移: offsetを−shift移動→dip逆混合(Tdip対称)
        bb = shift_offset_neg(bb)
        b = bb @ Tdip.T
        b = b / b.sum().clamp_min(1e-30)
    om = np.empty(T, dtype=np.float64)
    for t in range(T):
        g = alphas[t] * betas[t]
        g = g / g.sum().clamp_min(1e-30)
        om[t] = float((g.sum(1) * offs).sum())
    pred = A + om
    e = pred - tv
    return dict(pred=pred.astype(np.float32), err=e.astype(np.float32), true=tv,
                anc_err=(A - tv).astype(np.float32))


def shift_offset_neg(p):
    """shift_offsetの随伴(逆方向移動)。"""
    A = p.gather(0, _gA2) * _mA2; B = p.gather(0, _gB2) * _mB2
    return (1 - _fr) * A + _fr * B


def main():
    wids = sorted({os.path.basename(p).split("__")[0] for p in glob.glob(str(TRAIN / "*__horizontal_well.csv"))})
    wids = wids[::WSTRIDE] if WSTRIDE > 0 else wids[:NW]
    es, ea, per, outs = [], [], {}, {}
    t0 = time.time()
    for i, w in enumerate(wids):
        try:
            r = run_well(w)
        except Exception as ex:
            print(f"  {w}: ERROR {type(ex).__name__} {ex}"); continue
        if r is None: continue
        m = np.isfinite(r["err"])
        es.append(r["err"][m]); ea.append(r["anc_err"][m])
        per[w] = float(np.sqrt(np.mean(r["err"][m] ** 2))); outs[w] = r
    e = np.concatenate(es); a = np.concatenate(ea)
    print(f"\n=== P5 (offset×dip) trellis ({len(per)}井) ANCW={ANCW} NDIP={NDIP} DIPMAX={DIPMAX} SDIP={SDIP} SIGDIV={SIGDIV} ===")
    print(f"  P5 pooled = {np.sqrt(np.mean(e**2)):.4f}   (K16錨単体 = {np.sqrt(np.mean(a**2)):.4f})")
    print(f"  per-well 中央 = {np.median(list(per.values())):.3f}")
    fp = ART / f"p5_out{os.environ.get('TAG','')}_{len(per)}w.pkl"
    fp.write_bytes(pickle.dumps(outs, protocol=4))
    print(f"[saved] {fp.name}  total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
