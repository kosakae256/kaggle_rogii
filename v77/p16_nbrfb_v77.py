# -*- coding: utf-8 -*-
"""
v77 P16 — 近傍疑似typewellの grid-FB(K16錨帯, 決定的)
======================================================
emissionの参照曲線を「XY最近傍のtrain井(自分以外=LOO)のGR-TVT曲線」に置換したP6。
v28の教訓: neighbor=外部情報で有効(self=冗長)。tw/低域/PFと違う証拠源=脱相関脚候補。
NBK: 使う近傍数(emission log尤度を等重み平均)。
使い方: NW=60 python v77/p16_nbrfb_v77.py
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
BAND = float(os.environ.get("BAND", "40")); STEP = float(os.environ.get("STEP", "0.5"))
SRW = float(os.environ.get("SRW", "0.02"))
ANCW = float(os.environ.get("ANCW", "0.01"))
POW = float(os.environ.get("POW", "1.7826896851395453"))
NBK = int(os.environ.get("NBK", "1"))
SELFC = os.environ.get("SELFC", "0")             # v81: 1=参照曲線を自分のprefix(TVT_input,GR)に差替
DEVOVR = os.environ.get("DEVICE", "")
if DEVOVR: DEV = DEVOVR
TVBIN = float(os.environ.get("TVBIN", "0.5"))    # 近傍曲線のTVTビン幅(ft)
GRSIG = dict(mn=11.865793233968713, mx=44.98927964612316, df=119.25397464636981, mult=3.058318309943017)
ANC = pickle.loads((ART / "k16_anchor_train.pkl").read_bytes())
offs = torch.arange(-BAND, BAND + STEP / 2, STEP, device=DEV)
S = len(offs)

# --- 近傍インデックス(全train井の中心XYとGR-TVT曲線) ---
IDX_FP = ART / "p16_nbr_index.pkl"
if IDX_FP.exists():
    NIDX = pickle.loads(IDX_FP.read_bytes())
else:
    NIDX = {}
    t0 = time.time()
    for fp in sorted(glob.glob(str(TRAIN / "*__horizontal_well.csv"))):
        wid = os.path.basename(fp).split("__")[0]
        w = pd.read_csv(fp, usecols=["X", "Y", "GR", "TVT"])
        m = np.isfinite(w["TVT"].to_numpy()) & np.isfinite(w["GR"].to_numpy())
        if m.sum() < 200: continue
        tvt = w["TVT"].to_numpy()[m]; gr = w["GR"].to_numpy()[m]
        # TVTビン平均曲線
        b = np.round(tvt / TVBIN).astype(int)
        ub, inv = np.unique(b, return_inverse=True)
        cg = np.bincount(inv, gr) / np.bincount(inv)
        NIDX[wid] = dict(x=float(np.nanmean(w["X"])), y=float(np.nanmean(w["Y"])),
                         tv=(ub * TVBIN).astype(np.float32), gr=cg.astype(np.float32))
    IDX_FP.write_bytes(pickle.dumps(NIDX, protocol=4))
    print(f"[index built] {len(NIDX)}井 ({time.time()-t0:.0f}s)", flush=True)


def run_well(wid):
    if wid not in ANC or wid not in NIDX: return None
    hw = pd.read_csv(TRAIN / f"{wid}__horizontal_well.csv")
    kn = hw[hw["TVT_input"].notna()]; ev = hw[hw["TVT_input"].isna()]
    if len(ev) < 60 or len(kn) < 20: return None
    gr = hw["GR"].astype(float).interpolate(limit_direction="both")
    gr = gr.fillna(float(np.nanmean(gr))).to_numpy(float)
    kpos = hw.index.get_indexer(kn.index); epos = hw.index.get_indexer(ev.index)
    # 近傍選択(自分以外, 距離昇順) / SELFC=1なら自分のprefix曲線のみ
    if SELFC == "1":
        m_ = kn["TVT_input"].notna().to_numpy() & np.isfinite(gr[kpos])
        stv = kn["TVT_input"].to_numpy(float)[m_]; sgr = gr[kpos][m_]
        b = np.round(stv / TVBIN).astype(int)
        ub, inv = np.unique(b, return_inverse=True)
        cg = np.bincount(inv, sgr) / np.bincount(inv)
        NIDX[wid + "_self"] = dict(x=0.0, y=0.0, tv=(ub * TVBIN).astype(np.float32), gr=cg.astype(np.float32))
        cands = [(0.0, wid + "_self")]
    else:
        x0, y0 = NIDX[wid]["x"], NIDX[wid]["y"]
        cands = sorted(((np.hypot(NIDX[n]["x"] - x0, NIDX[n]["y"] - y0), n)
                        for n in NIDX if n != wid))
    a = ANC[wid]
    T = min(len(ev), len(a["tvt"]))
    A = a["tvt"][:T]; sig_dd = a["sig"][:T]
    ge = gr[epos][:T]; tv = ev["TVT"].to_numpy(float)[:T]
    At = torch.tensor(A, device=DEV, dtype=torch.float32)
    tvtr = At[:, None] + offs[None, :]
    kt = kn["TVT_input"].to_numpy(float)
    loge = None; used = 0; dists = []
    for dist, nb in cands[:NBK * 3]:
        if used >= NBK: break
        ntv = NIDX[nb]["tv"]; ngr = NIDX[nb]["gr"]
        # 帯がカバーされているか(80%以上)
        lo, hi = float(np.nanmin(A) - BAND), float(np.nanmax(A) + BAND)
        cov = ((ntv >= lo) & (ntv <= hi)).sum() * TVBIN / max(hi - lo, 1e-6)
        if cov < 0.6: continue
        resid = gr[kpos] - np.interp(kt, ntv, ngr)
        bias = float(np.nanmedian(resid))
        if not np.isfinite(bias): bias = 0.0
        gs = float(np.nanstd(resid - bias))
        if not np.isfinite(gs) or gs <= 0: gs = GRSIG["df"]
        gs = float(np.clip(gs * GRSIG["mult"], GRSIG["mn"], GRSIG["mx"]))
        ttv = torch.tensor(ntv, device=DEV, dtype=torch.float32)
        tgr = torch.tensor(ngr, device=DEV, dtype=torch.float32)
        ii = torch.searchsorted(ttv, tvtr.clamp(ttv[0], ttv[-1]).contiguous()).clamp(1, len(ttv) - 1)
        wv = ((tvtr - ttv[ii - 1]) / (ttv[ii] - ttv[ii - 1]).clamp_min(1e-6)).clamp(0, 1)
        egr = tgr[ii - 1] * (1 - wv) + tgr[ii] * wv
        grt = torch.tensor(ge, device=DEV, dtype=torch.float32)
        d = (((grt[:, None] - bias) - egr) / gs).abs()
        l_ = -0.5 * d ** POW
        loge = l_ if loge is None else loge + l_
        used += 1; dists.append(dist)
    if loge is None: return None
    loge = loge / used
    sdd = torch.tensor(sig_dd, device=DEV, dtype=torch.float32).clamp_min(1.0)
    loge = loge - 0.5 * ANCW * (offs[None, :] / sdd[:, None]) ** 2
    ksig = max(SRW / STEP, 0.05)
    kw = int(max(2, np.ceil(4 * ksig)))
    kx = torch.arange(-kw, kw + 1, device=DEV, dtype=torch.float32)
    ker = (torch.exp(-0.5 * (kx / ksig) ** 2)); ker = (ker / ker.sum()).view(1, 1, -1)
    def blur(v):
        p = torch.exp(v - v.max())
        p = torch.nn.functional.conv1d(p.view(1, 1, -1), ker, padding=kw).view(-1)
        return torch.log(p.clamp_min(1e-30))
    T_ = loge.shape[0]
    alpha = torch.zeros(T_, S, device=DEV); beta = torch.zeros(T_, S, device=DEV)
    alpha[0] = -0.5 * (offs / 3.0) ** 2 + loge[0]
    for t in range(1, T_):
        alpha[t] = blur(alpha[t - 1]) + loge[t]; alpha[t] -= alpha[t].max()
    beta[T_ - 1] = 0
    for t in range(T_ - 2, -1, -1):
        beta[t] = blur(beta[t + 1] + loge[t + 1]); beta[t] -= beta[t].max()
    post = torch.softmax(alpha + beta, dim=1)
    om = (post * offs[None, :]).sum(1).cpu().numpy()
    pred = A + om
    return dict(pred=pred.astype(np.float32), err=(pred - tv).astype(np.float32),
                anc_err=(A - tv).astype(np.float32), nbd=float(np.mean(dists)))


def main():
    wids = sorted({os.path.basename(p).split("__")[0] for p in glob.glob(str(TRAIN / "*__horizontal_well.csv"))})
    ws = int(os.environ.get("WSTRIDE", "0"))
    wids = wids[::ws] if ws > 0 else wids[:NW]
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
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(wids)} 暫定 {np.sqrt(np.mean(np.concatenate(es)**2)):.3f} ({time.time()-t0:.0f}s)", flush=True)
    e = np.concatenate(es); a = np.concatenate(ea)
    print(f"\n=== P16 nbr-FB ({len(per)}井) NBK={NBK} ANCW={ANCW} SRW={SRW} ===")
    print(f"  pooled = {np.sqrt(np.mean(e**2)):.4f}   (K16錨単体 = {np.sqrt(np.mean(a**2)):.4f})  中央 {np.median(list(per.values())):.3f}")
    fp = ART / f"p16_out{os.environ.get('TAG','')}_{len(per)}w.pkl"
    fp.write_bytes(pickle.dumps(outs, protocol=4))
    print(f"[saved] {fp.name}  total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
