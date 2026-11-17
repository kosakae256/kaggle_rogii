# -*- coding: utf-8 -*-
"""
v77 P10 — 大域スプライン・オフセット最適化(非逐次・一括推定)
=============================================================
offset(row) = 区分線形スプライン(KNOTSノット値 v_j の線形補間)。
損失 = Σ行 robust GRコスト(A+off) + ANCW·Σ(off/σ_dd)² + CURV·Σ(Δ²v)²。
Adamで一括最適化(逐次伝播なし・ドリフト無縁)。init=0(錨盆地)。
使い方: NW=60 python v77/p10_globalfit_v77.py  (KNOTS/ANCW/CURV/GRMA/STEPS)
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
KNOTS = int(os.environ.get("KNOTS", "12"))
ANCW = float(os.environ.get("ANCW", "1.0"))
CURV = float(os.environ.get("CURV", "10.0"))
STEPS = int(os.environ.get("STEPS", "400"))
LR = float(os.environ.get("LR", "0.5"))
GRMA = int(os.environ.get("GRMA", "1"))
POW = float(os.environ.get("POW", "1.7826896851395453"))
GRSIG = dict(mn=11.865793233968713, mx=44.98927964612316, df=119.25397464636981, mult=3.058318309943017)
ANC = pickle.loads((ART / "k16_anchor_train.pkl").read_bytes())
torch.manual_seed(777)


def load(wid):
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
    kpos = hw.index.get_indexer(kn.index); epos = hw.index.get_indexer(ev.index)
    resid = gr[kpos] - np.interp(kn["TVT_input"].to_numpy(float), tw_tvt, tw_gr)
    gs = float(np.nanstd(resid))
    if not np.isfinite(gs) or gs <= 0: gs = GRSIG["df"]
    gs = float(np.clip(gs * GRSIG["mult"], GRSIG["mn"], GRSIG["mx"]))
    a = ANC[wid]
    T = min(len(ev), len(a["tvt"]))
    return dict(A=a["tvt"][:T], sdd=a["sig"][:T], ge=gr[epos][:T],
                tw_tvt=tw_tvt, tw_gr=tw_gr, gs=gs, true=ev["TVT"].to_numpy(float)[:T])


def run(c):
    T = len(c["ge"])
    At = torch.tensor(c["A"], device=DEV, dtype=torch.float32)
    ttv = torch.tensor(c["tw_tvt"], device=DEV, dtype=torch.float32)
    tgr = torch.tensor(c["tw_gr"], device=DEV, dtype=torch.float32)
    grt = torch.tensor(c["ge"], device=DEV, dtype=torch.float32)
    sdd = torch.tensor(c["sdd"], device=DEV, dtype=torch.float32).clamp_min(1.0)
    # ノット→行への線形補間行列(固定): rows t の位置を [0,KNOTS-1] に写像
    tt = torch.linspace(0, KNOTS - 1, T, device=DEV)
    j0 = tt.floor().long().clamp(0, KNOTS - 2); fr = (tt - j0.float()).clamp(0, 1)
    v = torch.zeros(KNOTS, device=DEV, requires_grad=True)
    opt = torch.optim.Adam([v], lr=LR)
    for it in range(STEPS):
        off = v[j0] * (1 - fr) + v[j0 + 1] * fr
        q = (At + off).clamp(ttv[0], ttv[-1])
        ii = torch.searchsorted(ttv, q.detach().contiguous()).clamp(1, len(ttv) - 1)
        wv = ((q - ttv[ii - 1]) / (ttv[ii] - ttv[ii - 1]).clamp_min(1e-6)).clamp(0, 1)
        egr = tgr[ii - 1] * (1 - wv) + tgr[ii] * wv
        cost_gr = (((grt - egr) / c["gs"]).abs() + 1e-4) ** POW
        cost_anc = ANCW * (off / sdd) ** 2
        cost_cv = CURV * ((v[2:] - 2 * v[1:-1] + v[:-2]) ** 2).sum()
        loss = cost_gr.mean() + cost_anc.mean() + cost_cv / T
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        off = (v[j0] * (1 - fr) + v[j0 + 1] * fr).cpu().numpy()
    return c["A"] + off


def main():
    wids = sorted({os.path.basename(p).split("__")[0] for p in glob.glob(str(TRAIN / "*__horizontal_well.csv"))})[:NW]
    es, ea, per, outs = [], [], {}, {}
    t0 = time.time()
    for i, w in enumerate(wids):
        c = load(w)
        if c is None: continue
        try:
            pred = run(c)
        except Exception as ex:
            print(f"  {w}: ERROR {type(ex).__name__} {ex}"); continue
        e = pred - c["true"]; m = np.isfinite(e)
        es.append(e[m]); ea.append((c["A"] - c["true"])[m])
        per[w] = float(np.sqrt(np.mean(e[m] ** 2)))
        outs[w] = dict(pred=pred.astype(np.float32))
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(wids)} 暫定 {np.sqrt(np.mean(np.concatenate(es)**2)):.3f} ({time.time()-t0:.0f}s)", flush=True)
    e = np.concatenate(es); a = np.concatenate(ea)
    print(f"\n=== P10 大域スプライン ({len(per)}井) KNOTS={KNOTS} ANCW={ANCW} CURV={CURV} GRMA={GRMA} ===")
    print(f"  pooled = {np.sqrt(np.mean(e**2)):.4f}   (K16錨単体 = {np.sqrt(np.mean(a**2)):.4f})  中央 {np.median(list(per.values())):.3f}")
    fp = ART / f"p10_out{os.environ.get('TAG','')}_{len(per)}w.pkl"
    fp.write_bytes(pickle.dumps(outs, protocol=4))
    print(f"[saved] {fp.name}  total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
