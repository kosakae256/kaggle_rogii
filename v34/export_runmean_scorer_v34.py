# -*- coding: utf-8 -*-
"""f_runmean スコアラの提出用エクスポート: train全井戸で TCNスコアラを full-train 学習し、
runmean_scorer.pt(state_dict)+ runmean_scorer_cfg.json を保存する。
前提: seed_dump_pf{1,2,3}.npz / v28_4 parquet / rgrid_cache.npz(ranker_runmean_v34.ipynb 実行済)。
実行: .venv/Scripts/python.exe v34/export_runmean_scorer_v34.py(GPU ~10分)
"""
import json, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch, torch.nn as nn, torch.nn.functional as F
warnings.filterwarnings("ignore")

PROJECT = Path(r"C:\Users\kosaka256\Documents\rogii_claude")
V28_ART = PROJECT / "v28" / "artifacts_v28"
V284_PQ = PROJECT / "v28" / "artifacts_v28_4" / "data" / "train.parquet"
ART = PROJECT / "v34" / "artifacts_v34"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
STRIDE, TAU, EPOCHS, LR, CH_N = 4, 0.5, 20, 3e-4, 32
BETA_RULE = 1.5          # 提出ルール: softmax β1.5 + 等重みshrink λ0.9(OOF 6.7346, どちらもプラトー=頑健)
SHRINK = 0.9             # w = λ·softmax + (1-λ)·等重み(combination puzzle処方箋, 2026-07-07実測)
torch.manual_seed(0)

PF_BANKS = ["pf1", "pf2", "pf3"]; R_BANKS = ["rp1", "rp2", "rp3"]
BANKS = PF_BANKS + R_BANKS
SRCS = ("tw", "self", "nbr"); FLTS = (("sm", "ps"), ("fw", "pfw"))
B_ID = {b: i for i, b in enumerate(BANKS)}; S_ID = {"tw": 0, "self": 1, "nbr": 2}
META_DIM = len(BANKS) + 3 + 1 + 3
RCOL = {("rp1", "tw"): "pf_ancc_1", ("rp2", "tw"): "pf_ancc_2", ("rp3", "tw"): "pf_ancc_3",
        ("rp1", "self"): "pf_self_1", ("rp2", "self"): "pf_self_2", ("rp3", "self"): "pf_self_3",
        ("rp1", "nbr"): "pf_nbr_1", ("rp2", "nbr"): "pf_nbr_2", ("rp3", "nbr"): "pf_nbr_3"}
RSTD = {"rp1": "pf_ancc_std_1", "rp2": "pf_ancc_std_2", "rp3": "pf_ancc_std_3"}

tbl = pq.read_table(V28_ART / "data" / "train.parquet",
                    columns=["well", "target", "last_known_tvt", "pf_nbr_has"])
g = tbl.column("well").to_pandas().to_numpy()
lk = tbl.column("last_known_tvt").to_numpy(zero_copy_only=False).astype(np.float64)
tvt_true = lk + tbl.column("target").to_numpy(zero_copy_only=False).astype(np.float64)
has = tbl.column("pf_nbr_has").to_numpy(zero_copy_only=False).astype(np.float64)
wid = list(pd.unique(g)); rows_of = {w: np.where(g == w)[0] for w in wid}
DUMPS = {b: np.load(ART / f"seed_dump_pf{b[-1]}.npz") for b in PF_BANKS}
KEYS = {b: set(DUMPS[b].files) for b in PF_BANKS}
t4 = pq.read_table(V284_PQ, columns=["well"] + sorted(set(RCOL.values()) | set(RSTD.values())))
assert (t4.column("well").to_pandas().to_numpy() == g).all()
C4 = {c: t4.column(c).to_numpy(zero_copy_only=False).astype(np.float64) for c in t4.schema.names if c != "well"}
del t4
_rg = np.load(ART / "rgrid_cache.npz")
RGRID = {}
for key in _rg.files:
    w, b, kind = key.split("|")
    RGRID.setdefault((w, b), {})[kind] = _rg[key]

# ---- RM(候補+meta)構築: ranker_runmean_v34.ipynb C3 と同一 ----
t0 = time.time(); RM = {}
for w in wid:
    r = rows_of[w]
    nbr_ok = np.nanmean(has[r]) > 0.5
    trs, meta = [], []
    Tmin = len(r)
    for b in PF_BANKS:
        z = DUMPS[b]
        for src in SRCS:
            if src == "nbr" and not nbr_ok: continue
            if f"{w}|ps_{src}" not in KEYS[b]: continue
            ww_ = z[f"{w}|ww_{src}"]
            for flt, pref in FLTS:
                ps = z[f"{w}|{pref}_{src}"]
                Tmin = min(Tmin, ps.shape[1])
                tr_ = (ww_[:, None] * ps.astype(np.float64)).sum(0)
                dse = float(np.nanstd(np.nanmedian(ps.astype(np.float64) - tr_[None, :ps.shape[1]], axis=1)))
                went = float(-(ww_ * np.log(np.maximum(ww_, 1e-12))).sum())
                trs.append(tr_); meta.append((b, src, flt, dse, went, float(ww_.max())))
    for b in R_BANKS:
        if (w, b) not in RGRID: continue
        dse = float(np.nanmedian(C4[RSTD[b]][r]))
        for src in SRCS:
            if src == "nbr" and not nbr_ok: continue
            v = C4[RCOL[(b, src)]][r]
            if np.isfinite(v).sum() < max(10, 0.3 * len(r)): continue
            trs.append(v.copy()); meta.append((b, src, "sm", dse, 0.0, 0.0))
    if len(trs) < 4 or Tmin < 5: continue
    TR = np.stack([t_[:Tmin] for t_ in trs], 0)
    err = np.nanmedian(np.abs(TR - tvt_true[r[:Tmin]][None, :]), axis=1)
    ok = np.isfinite(err)
    if ok.sum() < 4: continue
    RM[w] = dict(TR=TR[ok].astype(np.float32), T=Tmin, err=err[ok].astype(np.float32),
                 meta=[m for m, o in zip(meta, ok) if o],
                 cons=np.nanmedian(TR[ok], axis=0).astype(np.float32),
                 p10=np.nanpercentile(TR[ok], 10, axis=0).astype(np.float32),
                 p90=np.nanpercentile(TR[ok], 90, axis=0).astype(np.float32))
print(f"RM {len(RM)} wells({time.time()-t0:.0f}s)")

def _grids_of(w, b):
    if b in PF_BANKS:
        z = DUMPS[b]
        out = dict(meta=z[f"{w}|meta"], gr=z[f"{w}|gr"], md=z[f"{w}|md"], zz=z[f"{w}|z"],
                   ir=float(z[f"{w}|ir"]), gg_tw=z[f"{w}|gg_tw"], gg_self=z[f"{w}|gg_self"])
        if f"{w}|gg_nbr" in KEYS[b]: out["gg_nbr"] = z[f"{w}|gg_nbr"]
        return out
    d = RGRID.get((w, b))
    if d is None: return None
    out = dict(meta=d["meta"], gr=d["gr"], md=d["md"], zz=d["z"], ir=float(d["ir"][0]),
               gg_tw=d["gg_tw"], gg_self=d["gg_self"])
    if "gg_nbr" in d: out["gg_nbr"] = d["gg_nbr"]
    return out

def build_ch(w):
    d = RM[w]; T = d["T"]
    TR = d["TR"][:, ::STRIDE].astype(np.float32)
    cons = d["cons"][::STRIDE]; p10 = d["p10"][::STRIDE]; p90 = d["p90"][::STRIDE]
    n = len(TR)
    g0 = _grids_of(w, "pf1")
    md_ = g0["md"].astype(np.float32)[:T][::STRIDE]; zz = g0["zz"].astype(np.float32)[:T][::STRIDE]
    ir0 = g0["ir"]
    off = np.clip((TR - cons[None, :]) / 10.0, -6, 6)
    env = np.clip((TR - p10[None, :]) / (p90 - p10 + 1e-3)[None, :] - 0.5, -3, 3)
    dmd = np.maximum(np.diff(md_), 1.0)
    dip = np.zeros_like(TR)
    dip[:, 1:] = np.clip(np.nan_to_num((np.diff(TR + zz[None, :], axis=1) / dmd[None, :] - ir0) * 100.0), -4, 4)
    RES = {k: np.zeros_like(TR) for k in ("tw", "self", "nbr")}
    for i, (b, src, flt, dse, went, wmax) in enumerate(d["meta"]):
        gb = _grids_of(w, b)
        if gb is None: continue
        gmin, gst, gs = gb["meta"]; gr = gb["gr"].astype(np.float32)[:T][::STRIDE]
        for nmk in ("tw", "self", "nbr"):
            gg = gb.get(f"gg_{nmk}")
            if gg is None: continue
            gg = gg.astype(np.float32); G = len(gg)
            ii = np.clip(np.nan_to_num((TR[i] - gmin) / gst), 0.0, G - 1.000001)
            i0 = ii.astype(int); t = ii - i0
            eg = gg[i0] * (1 - t) + gg[np.minimum(i0 + 1, G - 1)] * t
            RES[nmk][i] = np.clip(np.nan_to_num((gr - eg) / max(float(gs), 1e-3)), -4, 4)
    grz = g0["gr"].astype(np.float32)[:T][::STRIDE]
    grz = np.clip((grz - np.nanmedian(grz)) / (np.nanstd(grz) + 1e-3), -4, 4)
    grz = np.nan_to_num(np.broadcast_to(grz, (n, len(grz))))
    X = np.stack([np.nan_to_num(off), np.nan_to_num(env), RES["tw"], RES["self"], RES["nbr"], dip, grz], 1)
    M = np.zeros((n, META_DIM), np.float32)
    for i, (b, src, flt, dse, went, wmax) in enumerate(d["meta"]):
        M[i, B_ID[b]] = 1; M[i, len(BANKS) + S_ID[src]] = 1
        M[i, len(BANKS) + 3] = float(flt == "sm")
        M[i, len(BANKS) + 4] = min(dse, 25.0) / 5.0
        M[i, len(BANKS) + 5] = went / 3.5; M[i, len(BANKS) + 6] = wmax
    return np.ascontiguousarray(X).astype(np.float16), M

class ChanLN(nn.Module):
    def __init__(self, c):
        super().__init__(); self.ln = nn.LayerNorm(c)
    def forward(self, x): return self.ln(x.transpose(1, 2)).transpose(1, 2)
class Scorer(nn.Module):
    def __init__(self, cin=7, ch=CH_N, dmeta=META_DIM):
        super().__init__()
        self.inp = nn.Conv1d(cin, ch, 1)
        self.blocks = nn.ModuleList()
        for dil in (1, 2, 4, 8):
            self.blocks.append(nn.ModuleList([
                nn.Conv1d(ch, ch, 5, padding=2 * dil, dilation=dil), ChanLN(ch), nn.Conv1d(ch, ch, 1)]))
        self.emb = nn.Linear(2 * ch, 64)
        self.mp = nn.Sequential(nn.Linear(dmeta, 64), nn.GELU(), nn.Linear(64, 64))
        self.head = nn.Sequential(nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 1))
    def forward(self, x, meta):
        h = self.inp(x)
        for cv, ln, pw in self.blocks:
            h = h + pw(F.gelu(ln(cv(h))))
        e = self.emb(torch.cat([h.mean(-1), h.amax(-1)], -1)) + self.mp(meta)
        return self.head(e).squeeze(-1)

print("チャネル前計算 ...", flush=True); t0 = time.time()
CH = {}; MB = {}; PSTAR = {}
for w in RM:
    xw, mw = build_ch(w)
    CH[w] = torch.from_numpy(xw).to(DEV); MB[w] = torch.from_numpy(mw).to(DEV)
    e = RM[w]["err"].astype(np.float64)
    ez = ((e - e.mean()) / (e.std() + 1e-9)).astype(np.float32)
    PSTAR[w] = F.softmax(torch.from_numpy(-ez / TAU).to(DEV), 0)
print(f"  done {time.time()-t0:.0f}s")

mdl = Scorer().to(DEV)
opt = torch.optim.AdamW(mdl.parameters(), lr=LR, weight_decay=1e-4)
rng = np.random.default_rng(0)
wells_all = list(RM)
for ep in range(EPOCHS):
    mdl.train(); tot = 0.0
    for j in rng.permutation(len(wells_all)):
        w = wells_all[j]
        s = mdl(CH[w].float(), MB[w])
        loss = -(PSTAR[w] * F.log_softmax(s, 0)).sum()
        opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss)
    if ep % 5 == 4:
        print(f"ep{ep+1}: loss {tot/len(wells_all):.4f}", flush=True)

torch.save(mdl.state_dict(), ART / "runmean_scorer.pt")
cfg = dict(banks=BANKS, srcs=list(SRCS), meta_dim=META_DIM, stride=STRIDE, ch=CH_N,
           rule=dict(agg="softmax", beta=BETA_RULE, shrink=SHRINK),
           provenance=dict(oof_softmax_b15=6.7485, oof_shrink09=6.7346, cons=7.1055, oracle1=4.8783,
                           n_wells=len(RM), created="2026-07-07"))
(ART / "runmean_scorer_cfg.json").write_text(json.dumps(cfg, indent=1))
print("saved:", ART / "runmean_scorer.pt", "/", ART / "runmean_scorer_cfg.json")
