# -*- coding: utf-8 -*-
"""
v78 P21' — (offset × GRレベルドリフト) 2次元同時FB(決定的)
===========================================================
P20(井戸一定bias周辺化)の拡張: biasを「MDに沿ってゆっくり動く状態」として同時推定。
状態: (offset 81 × bias NBS) / 遷移: offsetぼかし ⊗ biasぼかし(独立) / emission: gr−bias−tw(A+off)。
ツールドリフト・洗掘等のMD変化するGRレベル誤差をFBが吸収できるかの検証。
使い方: WSTRIDE=25 python v77/p21_biasdrift_v77.py
"""
import os, io, sys, glob, time, pickle
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
from pathlib import Path
import numpy as np, pandas as pd
import torch
import torch.nn.functional as Fnn

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
NBS = int(os.environ.get("NBS", "7"))            # bias格子本数
BMAXSE = float(os.environ.get("BMAXSE", "2.0"))  # bias範囲(±SE倍)
BRW = float(os.environ.get("BRW", "0.02"))       # biasドリフトσ(格子単位/行)
GRSIG = dict(mn=11.865793233968713, mx=44.98927964612316, df=119.25397464636981, mult=3.058318309943017)
ANC = pickle.loads((ART / "k16_anchor_train.pkl").read_bytes())
offs = torch.arange(-BAND, BAND + STEP / 2, STEP, device=DEV)
S = len(offs)


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
    if GRMA > 1: gr = np.convolve(gr, np.ones(GRMA) / GRMA, mode="same")
    step = float(np.median(np.diff(tw_tvt)))
    wpt = max(3, int(round(TWMA / max(step, 1e-3))) | 1)
    tw_gr = np.convolve(tw_gr, np.ones(wpt) / wpt, mode="same") if TWMA > 0 else tw_gr
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
    bse = max(rs / np.sqrt(max(len(kt), 1)), 2.0)
    bgrid = torch.tensor(rm + np.linspace(-BMAXSE, BMAXSE, NBS) * bse, device=DEV, dtype=torch.float32)
    At = torch.tensor(A, device=DEV, dtype=torch.float32)
    tvtr = At[:, None] + offs[None, :]
    ttv = torch.tensor(tw_tvt, device=DEV, dtype=torch.float32)
    tgrt = torch.tensor(tw_gr, device=DEV, dtype=torch.float32)
    ii = torch.searchsorted(ttv, tvtr.clamp(ttv[0], ttv[-1]).contiguous()).clamp(1, len(ttv) - 1)
    wv = ((tvtr - ttv[ii - 1]) / (ttv[ii] - ttv[ii - 1]).clamp_min(1e-6)).clamp(0, 1)
    egr = tgrt[ii - 1] * (1 - wv) + tgrt[ii] * wv                 # (T,S)
    grt = torch.tensor(ge, device=DEV, dtype=torch.float32)
    d = (grt[:, None, None] - bgrid[None, None, :] - egr[:, :, None]).abs() / gs   # (T,S,B)
    loge = EMT * (-0.5 * d ** POW)
    sddt = torch.tensor(sdd, device=DEV, dtype=torch.float32)
    loge = loge - 0.5 * ANCW * (offs[None, :, None] / sddt[:, None, None]) ** 2
    # 遷移カーネル(offset方向, bias方向)
    ksig = max(SRW / STEP, 0.05)
    kw = int(max(2, np.ceil(4 * ksig)))
    kx = torch.arange(-kw, kw + 1, device=DEV, dtype=torch.float32)
    ko = torch.exp(-0.5 * (kx / ksig) ** 2); ko = (ko / ko.sum()).view(1, 1, -1)
    bsig = max(BRW, 0.02)
    bw = int(max(1, np.ceil(4 * bsig)))
    bx = torch.arange(-bw, bw + 1, device=DEV, dtype=torch.float32)
    kb = torch.exp(-0.5 * (bx / bsig) ** 2); kb = (kb / kb.sum()).view(1, 1, -1)
    def blur(v):                                                   # (S,B) 対数→確率域で2方向独立ぼかし
        p = torch.exp(v - v.max())
        p = Fnn.conv1d(p.t().reshape(NBS, 1, S), ko, padding=kw).reshape(NBS, S).t()
        p = Fnn.conv1d(p.reshape(S, 1, NBS), kb, padding=bw).reshape(S, NBS)
        return torch.log(p.clamp_min(1e-30))
    alpha = torch.zeros(T, S, NBS, device=DEV); beta = torch.zeros(T, S, NBS, device=DEV)
    prior0 = (-0.5 * (offs / 3.0) ** 2)[:, None] + torch.zeros(NBS, device=DEV)[None, :]
    alpha[0] = prior0 + loge[0]; alpha[0] -= alpha[0].max()
    for t in range(1, T):
        alpha[t] = blur(alpha[t - 1]) + loge[t]; alpha[t] -= alpha[t].max()
    beta[T - 1] = 0
    for t in range(T - 2, -1, -1):
        beta[t] = blur(beta[t + 1] + loge[t + 1]); beta[t] -= beta[t].max()
    post = torch.softmax((alpha + beta).reshape(T, -1), dim=1).reshape(T, S, NBS)
    om = (post.sum(2) * offs[None, :]).sum(1).cpu().numpy()
    pred = A + om
    return dict(pred=pred.astype(np.float32), err=(pred - tv).astype(np.float32),
                anc_err=(A - tv).astype(np.float32))


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
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(wids)} 暫定 {np.sqrt(np.mean(np.concatenate(es)**2)):.3f} ({time.time()-t0:.0f}s)", flush=True)
    e = np.concatenate(es); a = np.concatenate(ea)
    print(f"\n=== P21' (offset×biasドリフト) 2D-FB ({len(per)}井) NBS={NBS} BRW={BRW} EMT={EMT} ===")
    print(f"  pooled = {np.sqrt(np.mean(e**2)):.4f}   (K16錨単体 = {np.sqrt(np.mean(a**2)):.4f})  中央 {np.median(list(per.values())):.3f}")
    fp = ART / f"p21_out{os.environ.get('TAG','')}_{len(per)}w.pkl"
    fp.write_bytes(pickle.dumps(outs, protocol=4))
    print(f"[saved] {fp.name}  total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
