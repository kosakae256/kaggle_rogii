# -*- coding: utf-8 -*-
"""
v78 P20 — 井戸レベル大域潜在の周辺化(族6: FBが扱えない大域変数のベイズ混合)
==========================================================================
潜在: GRレベルシフト b(±2×既知区間残差std/√n級) × typewellストレッチ s(振幅倍率)。
各(b,s)で P6型FB(lp13/EMT)を回し、**周辺尤度 logZ(=正規化前logsumexpの累積)** でソフト混合。
点推定補正(TWREF)は過信で死んだ—周辺化は「わからなさ」を保つ。決定的。
使い方: WSTRIDE=5 python v77/p20_marglat_v77.py
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
WSTRIDE = int(os.environ.get("WSTRIDE", "0"))
BAND = 40.0; STEP = 0.5
SRW = float(os.environ.get("SRW", "0.02"))
ANCW = float(os.environ.get("ANCW", "0.01"))
EMT = float(os.environ.get("EMT", "0.15"))
POW = 1.7826896851395453
GRMA = int(os.environ.get("GRMA", "13")); TWMA = float(os.environ.get("TWMA", "6"))
NB = int(os.environ.get("NB", "5"))              # biasグリッド本数
NS = int(os.environ.get("NS", "3"))              # stretchグリッド本数
SMAX = float(os.environ.get("SMAX", "0.15"))     # stretch±(1±SMAX)
MTEMP = float(os.environ.get("MTEMP", "1.0"))    # 潜在混合の温度(logZ/行 スケール)
GRSIG = dict(mn=11.865793233968713, mx=44.98927964612316, df=119.25397464636981, mult=3.058318309943017)
ANC = pickle.loads((ART / "k16_anchor_train.pkl").read_bytes())
offs = torch.arange(-BAND, BAND + STEP / 2, STEP, device=DEV)
S = len(offs)


def fb_run(loge, ker, kw):
    """FB: posterior mean(オフセット) と logZ/T を返す"""
    T_ = loge.shape[0]
    def blur(v):
        p = torch.exp(v - v.max())
        p = torch.nn.functional.conv1d(p.view(1, 1, -1), ker, padding=kw).view(-1)
        return torch.log(p.clamp_min(1e-30))
    alpha = torch.zeros(T_, S, device=DEV); beta = torch.zeros(T_, S, device=DEV)
    logz = 0.0
    a = -0.5 * (offs / 3.0) ** 2 + loge[0]
    c = torch.logsumexp(a, 0); logz += float(c); alpha[0] = a - c
    for t in range(1, T_):
        a = blur(alpha[t - 1]) + loge[t]
        c = torch.logsumexp(a, 0); logz += float(c); alpha[t] = a - c
    b = torch.zeros(S, device=DEV)
    beta[T_ - 1] = b
    for t in range(T_ - 2, -1, -1):
        b = blur(beta[t + 1] + loge[t + 1])
        b = b - b.max(); beta[t] = b
    post = torch.softmax(alpha + beta, dim=1)
    om = (post * offs[None, :]).sum(1)
    return om.cpu().numpy(), logz / T_


def run_well(wid):
    if wid not in ANC: return None
    hw = pd.read_csv(TRAIN / f"{wid}__horizontal_well.csv")
    tw = pd.read_csv(TRAIN / f"{wid}__typewell.csv").sort_values("TVT")
    kn = hw[hw["TVT_input"].notna()]; ev = hw[hw["TVT_input"].isna()]
    if len(ev) < 60 or len(kn) < 20: return None
    tw_tvt = tw["TVT"].to_numpy(float); tw_gr0 = tw["GR"].to_numpy(float)
    m = np.isfinite(tw_tvt) & np.isfinite(tw_gr0); tw_tvt, tw_gr0 = tw_tvt[m], tw_gr0[m]
    if len(tw_tvt) < 8: return None
    gr = hw["GR"].astype(float).interpolate(limit_direction="both").fillna(float(np.nanmean(tw_gr0))).to_numpy(float)
    if GRMA > 1: gr = np.convolve(gr, np.ones(GRMA) / GRMA, mode="same")
    step = float(np.median(np.diff(tw_tvt)))
    wpt = max(3, int(round(TWMA / max(step, 1e-3))) | 1)
    tw_gr = np.convolve(tw_gr0, np.ones(wpt) / wpt, mode="same") if TWMA > 0 else tw_gr0
    kpos = hw.index.get_indexer(kn.index); epos = hw.index.get_indexer(ev.index)
    kt = kn["TVT_input"].to_numpy(float)
    resid = gr[kpos] - np.interp(kt, tw_tvt, tw_gr)
    rs = float(np.nanstd(resid)); rm = float(np.nanmedian(resid))
    gs = rs if np.isfinite(rs) and rs > 0 else GRSIG["df"]
    gs = float(np.clip(gs * GRSIG["mult"], GRSIG["mn"], GRSIG["mx"]))
    a = ANC[wid]
    T = min(len(ev), len(a["tvt"]))
    A = a["tvt"][:T]; sdd = np.clip(a["sig"][:T], 1.0, None)
    ge = gr[epos][:T]; tv = ev["TVT"].to_numpy(float)[:T]
    twmu = float(np.nanmean(tw_gr))
    # 潜在グリッド: b=残差中央値まわり±2SE級 / s=振幅倍率
    bse = max(rs / np.sqrt(max(len(kt), 1)), 2.0)
    bgrid = rm + np.linspace(-2, 2, NB) * bse if NB > 1 else np.array([0.0])
    sgrid = 1 + np.linspace(-SMAX, SMAX, NS) if NS > 1 else np.array([1.0])
    At = torch.tensor(A, device=DEV, dtype=torch.float32)
    tvtr = At[:, None] + offs[None, :]
    ttv = torch.tensor(tw_tvt, device=DEV, dtype=torch.float32)
    tgrt = torch.tensor(tw_gr, device=DEV, dtype=torch.float32)
    ii = torch.searchsorted(ttv, tvtr.clamp(ttv[0], ttv[-1]).contiguous()).clamp(1, len(ttv) - 1)
    wv = ((tvtr - ttv[ii - 1]) / (ttv[ii] - ttv[ii - 1]).clamp_min(1e-6)).clamp(0, 1)
    egr0 = tgrt[ii - 1] * (1 - wv) + tgrt[ii] * wv           # (T,S) 基準typewell GR
    grt = torch.tensor(ge, device=DEV, dtype=torch.float32)
    sddt = torch.tensor(sdd, device=DEV, dtype=torch.float32)
    lanc = -0.5 * ANCW * (offs[None, :] / sddt[:, None]) ** 2
    ksig = max(SRW / STEP, 0.05)
    kw = int(max(2, np.ceil(4 * ksig)))
    kx = torch.arange(-kw, kw + 1, device=DEV, dtype=torch.float32)
    ker = torch.exp(-0.5 * (kx / ksig) ** 2); ker = (ker / ker.sum()).view(1, 1, -1)
    oms, lzs = [], []
    for b_ in bgrid:
        for s_ in sgrid:
            egr = twmu + (egr0 - twmu) * float(s_)
            d = ((grt[:, None] - float(b_)) - egr).abs() / gs
            loge = EMT * (-0.5 * d ** POW) + lanc
            om, lz = fb_run(loge, ker, kw)
            oms.append(om); lzs.append(lz)
    lz = np.array(lzs)
    wgt = np.exp((lz - lz.max()) / MTEMP); wgt /= wgt.sum()
    om = np.sum(np.stack(oms, 0) * wgt[:, None], 0)
    pred = A + om
    return dict(pred=pred.astype(np.float32), err=(pred - tv).astype(np.float32),
                anc_err=(A - tv).astype(np.float32), wmax=float(wgt.max()))


def main():
    wids = sorted({os.path.basename(p).split("__")[0] for p in glob.glob(str(TRAIN / "*__horizontal_well.csv"))})
    wids = wids[::WSTRIDE] if WSTRIDE > 0 else wids[:NW]
    es, ea, per, outs, wmx = [], [], {}, {}, []
    t0 = time.time()
    for i, w in enumerate(wids):
        try:
            r = run_well(w)
        except Exception as ex:
            print(f"  {w}: ERROR {type(ex).__name__} {ex}"); continue
        if r is None: continue
        m = np.isfinite(r["err"])
        es.append(r["err"][m]); ea.append(r["anc_err"][m])
        per[w] = float(np.sqrt(np.mean(r["err"][m] ** 2))); outs[w] = r; wmx.append(r["wmax"])
        if (i + 1) % 40 == 0:
            print(f"  {i+1}/{len(wids)} 暫定 {np.sqrt(np.mean(np.concatenate(es)**2)):.3f} ({time.time()-t0:.0f}s)", flush=True)
    e = np.concatenate(es); a = np.concatenate(ea)
    print(f"\n=== P20 潜在周辺化 ({len(per)}井) NB={NB} NS={NS} SMAX={SMAX} MTEMP={MTEMP} EMT={EMT} ===")
    print(f"  pooled = {np.sqrt(np.mean(e**2)):.4f}   (K16錨単体 = {np.sqrt(np.mean(a**2)):.4f})  中央 {np.median(list(per.values())):.3f}")
    print(f"  潜在重みmax中央 = {np.median(wmx):.2f}(1.0=点推定化, 1/{NB*NS}={1/(NB*NS):.2f}=一様)")
    fp = ART / f"p20_out{os.environ.get('TAG','')}_{len(per)}w.pkl"
    fp.write_bytes(pickle.dumps(outs, protocol=4))
    print(f"[saved] {fp.name}  total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
