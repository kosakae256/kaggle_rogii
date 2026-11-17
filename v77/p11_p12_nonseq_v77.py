# -*- coding: utf-8 -*-
"""
v77 P11/P12 — 非逐次推定(ユーザー方針: 物理錨があれば逐次でない推論が可能になる)
================================================================================
P11: 行独立事後 — 遷移なし。各行の posterior(o) ∝ exp(GR尤度 + 錨prior)、posterior-mean。
     オプションで事後に中央値フィルタ(MED行)のみ(推定自体は完全に行独立)。
P12: 区間定数オフセット — 井戸をSEGLEN行の区間に分割し、各区間で「区間全体のGR照合コスト
     +錨prior」を候補オフセット一括評価 → softmin。区間中心間は線形補間。各区間は独立=非逐次。
使い方: METHOD=p11 NW=60 python v77/p11_p12_nonseq_v77.py / METHOD=p12 ...
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
METHOD = os.environ.get("METHOD", "p11")
NW = int(os.environ.get("NW", "60"))
BAND = float(os.environ.get("BAND", "40")); STEP = float(os.environ.get("STEP", "0.5"))
ANCW = float(os.environ.get("ANCW", "1.0"))       # 錨prior重み(P11/P12は行・区間単位=1.0が自然)
POW = float(os.environ.get("POW", "1.7826896851395453"))
GRMA = int(os.environ.get("GRMA", "1"))
TSOFT = float(os.environ.get("TSOFT", "1.0"))     # softmin温度
SEGLEN = int(os.environ.get("SEGLEN", "128"))     # P12区間長(行)
MED = int(os.environ.get("MED", "0"))             # P11: 事後中央値フィルタ(0=なし)
GRSIG = dict(mn=11.865793233968713, mx=44.98927964612316, df=119.25397464636981, mult=3.058318309943017)
ANC = pickle.loads((ART / "k16_anchor_train.pkl").read_bytes())
offs = torch.arange(-BAND, BAND + STEP / 2, STEP, device=DEV)
S = len(offs)


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


def emission(c):
    """(T,S) GR log尤度(行ごと) と 錨prior log(行ごと)"""
    At = torch.tensor(c["A"], device=DEV, dtype=torch.float32)
    tvtr = At[:, None] + offs[None, :]
    ttv = torch.tensor(c["tw_tvt"], device=DEV, dtype=torch.float32)
    tgr = torch.tensor(c["tw_gr"], device=DEV, dtype=torch.float32)
    ii = torch.searchsorted(ttv, tvtr.clamp(ttv[0], ttv[-1]).contiguous()).clamp(1, len(ttv) - 1)
    wv = ((tvtr - ttv[ii - 1]) / (ttv[ii] - ttv[ii - 1]).clamp_min(1e-6)).clamp(0, 1)
    egr = tgr[ii - 1] * (1 - wv) + tgr[ii] * wv
    grt = torch.tensor(c["ge"], device=DEV, dtype=torch.float32)
    lg = -0.5 * (((grt[:, None] - egr) / c["gs"]).abs()) ** POW
    sdd = torch.tensor(c["sdd"], device=DEV, dtype=torch.float32).clamp_min(1.0)
    la = -0.5 * (offs[None, :] / sdd[:, None]) ** 2
    return lg, la


def run_p11(c):
    lg, la = emission(c)
    post = torch.softmax((lg + ANCW * la) / TSOFT, dim=1)
    om = (post * offs[None, :]).sum(1).cpu().numpy()
    if MED > 1:
        om = pd.Series(om).rolling(MED, center=True, min_periods=1).median().to_numpy()
    return c["A"] + om


def run_p12(c):
    lg, la = emission(c)
    T = lg.shape[0]
    nseg = max(1, int(np.ceil(T / SEGLEN)))
    cen = []; ov = []
    for s0 in range(0, T, SEGLEN):
        s1 = min(s0 + SEGLEN, T)
        seg = (lg[s0:s1].mean(0) + ANCW * la[s0:s1].mean(0)) / TSOFT   # 区間平均コスト(候補一括)
        p = torch.softmax(seg * (s1 - s0) ** 0.5, 0)                    # 長さ√で信頼度スケール
        ov.append(float((p * offs).sum())); cen.append((s0 + s1 - 1) / 2)
    om = np.interp(np.arange(T), cen, ov)
    return c["A"] + om


def main():
    wids = sorted({os.path.basename(p).split("__")[0] for p in glob.glob(str(TRAIN / "*__horizontal_well.csv"))})[:NW]
    es, ea, per, outs = [], [], {}, {}
    t0 = time.time()
    fn = run_p11 if METHOD == "p11" else run_p12
    for w in wids:
        c = load(w)
        if c is None: continue
        try:
            pred = fn(c)
        except Exception as ex:
            print(f"  {w}: ERROR {type(ex).__name__} {ex}"); continue
        e = pred - c["true"]; m = np.isfinite(e)
        es.append(e[m]); ea.append((c["A"] - c["true"])[m])
        per[w] = float(np.sqrt(np.mean(e[m] ** 2)))
        outs[w] = dict(pred=pred.astype(np.float32))
    e = np.concatenate(es); a = np.concatenate(ea)
    print(f"\n=== {METHOD} 非逐次 ({len(per)}井) ANCW={ANCW} GRMA={GRMA} SEGLEN={SEGLEN} MED={MED} TSOFT={TSOFT} ===")
    print(f"  pooled = {np.sqrt(np.mean(e**2)):.4f}   (K16錨単体 = {np.sqrt(np.mean(a**2)):.4f})  中央 {np.median(list(per.values())):.3f}")
    fp = ART / f"{METHOD}_out{os.environ.get('TAG','')}_{len(per)}w.pkl"
    fp.write_bytes(pickle.dumps(outs, protocol=4))
    print(f"[saved] {fp.name}  total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
