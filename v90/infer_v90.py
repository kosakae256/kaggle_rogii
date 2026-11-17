# -*- coding: utf-8 -*-
"""
v90 推論 — test.parquet(v90 create出力) から GBM(9var×5fold) + TCN(combo3 3seed×5fold) blend + pp → submission.csv。
================================================================================
v69 submit_nn の推論部を v90用に(struct-free/ v90モデル/ gfは0埋め初版)。ローカル検証用。
使い方: python v90/infer_v90.py   (v90/artifacts_v90/data/test.parquet と各モデルを読む)
"""
import os, sys, io, glob, json, math, time
try: sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
except Exception: pass
import numpy as np, pandas as pd, joblib
import pyarrow.parquet as pq
import torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path

PROJ = Path(os.environ.get("V90_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
ART = PROJ / "v90" / "artifacts_v90"
WORK = Path(os.environ.get("V90_WORK", str(ART)))
def _detect_data():
    if os.environ.get("V90_DATA"): return Path(os.environ["V90_DATA"])
    for _p in ["/kaggle/input/rogii-wellbore-geology-prediction",
               "/kaggle/input/competitions/rogii-wellbore-geology-prediction",
               str(PROJ / "rogii-wellbore-geology-prediction")]:
        if os.path.exists(_p): return Path(_p)
    _h = glob.glob("/kaggle/input/**/sample_submission.csv", recursive=True)
    return Path(os.path.dirname(_h[0])) if _h else (PROJ / "rogii-wellbore-geology-prediction")
DATA = _detect_data()
TESTD = DATA / "test"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
t0 = time.time()

# ---- test_df ----
test_df = pq.read_table(WORK / "data" / "test.parquet").to_pandas()
print(f"test rows {len(test_df)} cols {test_df.shape[1]} ({time.time()-t0:.0f}s)")
features = json.loads((ART / "features.json").read_text())
for c in features:
    if c not in test_df.columns: test_df[c] = 0.0    # gf等 未生成列は0埋め(初版)
Xte = test_df[features].astype("float32").to_numpy()
lk = test_df["last_known_tvt"].to_numpy(np.float64)
mds = test_df["md_since"].to_numpy(np.float64)
pf_res = (test_df["pf_ancc"].to_numpy(np.float64) - lk) if "pf_ancc" in test_df.columns else np.zeros(len(test_df))

# ---- GBM(models_fwd) ----
_mroot = ART / "models_fwd"
preds = []
for d in sorted(glob.glob(str(_mroot / "*"))):
    fps = sorted(glob.glob(str(Path(d) / "fold*.pkl")))
    if not fps: continue
    preds.append(np.mean([joblib.load(fp).predict(Xte) for fp in fps], axis=0))
    print(f"  GBM {Path(d).name}: {len(fps)} folds")
res = np.mean(preds, axis=0).astype(np.float64)
print(f"GBM variants used: {len(preds)} ({time.time()-t0:.0f}s)")

# ---- TCN(combo3) ----
PACK = json.loads((ART / "submit_pack_nn.json").read_text())
CFG = PACK["cfg"]; A = PACK["arch"]; MTP_OFFS = A.get("mtp_offs", [4, 16, 64])

class ChanLN(nn.Module):
    def __init__(self, c): super().__init__(); self.ln = nn.LayerNorm(c)
    def forward(self, x): return self.ln(x.transpose(1, 2)).transpose(1, 2)
class Block(nn.Module):
    def __init__(self, c, dil, k, drop):
        super().__init__(); self.conv = nn.Conv1d(c, c, k, padding=dil * (k - 1) // 2, dilation=dil)
        self.ln = ChanLN(c); self.pw = nn.Conv1d(c, c, 1); self.do = nn.Dropout(drop)
    def forward(self, x, m):
        h = self.conv(x * m); h = F.gelu(h); h = self.ln(h); h = self.do(self.pw(h)); return (x + h) * m
class SelfAttn(nn.Module):
    def __init__(self, ch, heads, drop):
        super().__init__(); self.h = heads; self.dk = ch // heads; self.p = drop
        self.qkv = nn.Linear(ch, 3 * ch); self.proj = nn.Linear(ch, ch)
    def forward(self, x, kmask):
        B, L, C = x.shape; q, k, v = self.qkv(x).chunk(3, -1)
        q = q.view(B, L, self.h, self.dk).transpose(1, 2); k = k.view(B, L, self.h, self.dk).transpose(1, 2)
        v = v.view(B, L, self.h, self.dk).transpose(1, 2)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=kmask[:, None, None, :], dropout_p=0.0)
        return self.proj(o.transpose(1, 2).reshape(B, L, C))
class PrefixEnc(nn.Module):
    def __init__(self, c_in, ctx):
        super().__init__(); self.c1 = nn.Conv1d(c_in, 64, 5, padding=2)
        self.c2 = nn.Conv1d(64, 64, 5, padding=8, dilation=4); self.fc = nn.Linear(128, ctx)
    def forward(self, x, m):
        h = F.gelu(self.c1(x * m)); h = F.gelu(self.c2(h)) * m; s = m.sum(-1)
        mean = h.sum(-1) / s.clamp(min=1.0); mx = (h + (m - 1.0) * 1e4).amax(-1)
        mx = torch.where(s > 0, mx, torch.zeros_like(mx)); return self.fc(torch.cat([mean, mx], -1))
class SeqNet(nn.Module):
    def __init__(self, k_in, head, ch, ksize, dilations, drop, use_attn, attn_heads, ch_drop,
                 use_ctx=False, pre_c=4, ctx_dim=64, use_mtp=False):
        super().__init__(); self.head = head; self.use_ctx = use_ctx
        self.inp = nn.Conv1d(k_in, ch, 1); self.indrop = nn.Dropout1d(ch_drop)
        if use_ctx: self.pre = PrefixEnc(pre_c, ctx_dim); self.ctxp = nn.Linear(ctx_dim, ch)
        self.blocks = nn.ModuleList([Block(ch, d, ksize, drop) for d in dilations])
        self.use_attn = use_attn
        if use_attn: self.attn = SelfAttn(ch, attn_heads, drop); self.aln = nn.LayerNorm(ch)
        self.out = nn.Conv1d(ch, 5 if head == "mdn2" else 1, 1)
        self.mtp = nn.Conv1d(ch, len(MTP_OFFS), 1) if use_mtp else None
    def forward(self, x, m, px=None, pm=None):
        h = self.inp(self.indrop(x)) * m
        if self.use_ctx and px is not None: h = (h + self.ctxp(self.pre(px, pm))[:, :, None]) * m
        for b in self.blocks: h = b(h, m)
        if self.use_attn:
            q = self.aln(h.transpose(1, 2)); a = self.attn(q, (m[:, 0, :] > 0.5))
            h = (h + a.transpose(1, 2)) * m
        return self.out(h), (self.mtp(h) if self.mtp is not None else None)

NN = []
for _s in PACK["seeds"]:
    for _k in range(int(PACK["n_splits"])):
        m = SeqNet(len(PACK["features"]), "mdn2", A["ch"], A["ksize"], A["dilations"], A["drop"],
                   A["use_attn"], A["attn_heads"], A["ch_drop"], use_ctx=bool(CFG.get("prefix")),
                   pre_c=A["pre_c"], ctx_dim=A["ctx_dim"], use_mtp=bool(CFG.get("mtp", False))).to(DEV)
        m.load_state_dict(torch.load(ART / "nn" / f"{PACK['run']}_mdn2_s{_s}_fold{_k}.pt", map_location=DEV))
        m.eval(); NN.append(m)
print(f"TCN: {len(NN)} models (run={PACK['run']} w_nn={PACK['w_nn']})")

feats29 = PACK["features"]; a2r = PACK["abs2res"]
med = np.array(PACK["med"], np.float64); scl = np.array(PACK["scale"], np.float64)
Xnn = np.empty((len(test_df), len(feats29)), np.float32)
for j, c in enumerate(feats29):
    v = test_df[c].to_numpy(np.float64) if c in test_df.columns else np.full(len(test_df), np.nan)
    if a2r[j]: v = v - lk
    Xnn[:, j] = (v - med[j]) / scl[j]
Xnn = np.clip(np.nan_to_num(Xnn, nan=0.0, posinf=8.0, neginf=-8.0), -8.0, 8.0)
wc = test_df["well"].to_numpy()
PREF = {}
if CFG.get("prefix"):
    gm = PACK["prefix_stats"]["gmed"]; gq = PACK["prefix_stats"]["giqr"]; ts = PACK["t_scale"]; plen = int(A["pre_len"])
    for w in pd.unique(wc):
        fp = TESTD / f"{w}__horizontal_well.csv"
        if not fp.exists(): continue
        dfp = pd.read_csv(fp, usecols=["MD", "Z", "GR", "TVT_input"])
        kn = dfp.dropna(subset=["MD", "Z", "GR", "TVT_input"]).tail(plen)
        if len(kn) < 8: continue
        a = kn.to_numpy(np.float64)
        arr = np.stack([np.clip((a[:, 2] - gm) / gq, -8, 8), (a[:, 3] - a[-1, 3]) / ts,
                        (a[:, 1] - a[-1, 1]) / 50.0, (a[:, 0] - a[-1, 0]) / 1000.0], axis=1)
        PREF[w] = np.nan_to_num(arr, nan=0.0, posinf=8.0, neginf=-8.0).astype(np.float32)

def nn_forward(xseq, w):
    x = torch.from_numpy(np.ascontiguousarray(xseq.T)[None]).to(DEV)
    m = torch.ones(1, 1, xseq.shape[0], device=DEV); px = pm = None
    if CFG.get("prefix"):
        pa = PREF.get(w)
        if pa is None: px = torch.zeros(1, int(A["pre_c"]), 1, device=DEV); pm = torch.zeros(1, 1, 1, device=DEV)
        else: px = torch.from_numpy(np.ascontiguousarray(pa.T)[None]).to(DEV); pm = torch.ones(1, 1, len(pa), device=DEV)
    ps = []
    for mdl in NN:
        o = mdl(x, m, px, pm)[0][0].float().cpu().numpy()
        pl = 1.0 / (1.0 + np.exp(-o[2])); ps.append(pl * o[0] + (1.0 - pl) * o[1])
    return np.mean(ps, axis=0)

acc = np.zeros(len(test_df), np.float64)
with torch.no_grad():
    for w in pd.unique(wc):
        ix = np.where(wc == w)[0]; o = ix[np.argsort(mds[ix], kind="stable")]
        p = nn_forward(Xnn[o], w)
        if CFG.get("flip"): p = 0.5 * (p + nn_forward(np.ascontiguousarray(Xnn[o][::-1]), w)[::-1])
        acc[o] = p
res_nn = acc * PACK["t_scale"] + PACK["t_mean"]
wnn = float(PACK["w_nn"]); res = (1.0 - wnn) * res + wnn * res_nn
print(f"blend w_nn={wnn:.2f} ({time.time()-t0:.0f}s)")

PP = json.loads((ART / "pp_params.json").read_text())
d = res * (1 - PP["w_pf"]) + pf_res * PP["w_pf"]
if PP.get("tau"): d = d * (1.0 - np.exp(-np.maximum(mds, 0.0) / float(PP["tau"])))
d = d * float(PP.get("alpha", 1.0)); tvt = lk + d
pred_df = pd.DataFrame({"id": test_df["id"].values, "tvt": tvt})

ss = pd.read_csv(DATA / "sample_submission.csv")
sub = ss[["id"]].merge(pred_df, on="id", how="left")
miss = sub["tvt"].isna()
if miss.any():
    lkmap = {}
    for wid in set(i.rsplit("_", 1)[0] for i in sub.loc[miss, "id"]):
        v = pd.read_csv(TESTD / f"{wid}__horizontal_well.csv", usecols=["TVT_input"])["TVT_input"].dropna()
        lkmap[wid] = float(v.iloc[-1]) if len(v) else 0.0
    sub.loc[miss, "tvt"] = [lkmap[i.rsplit("_", 1)[0]] for i in sub.loc[miss, "id"]]
    print(f"lk fallback: {int(miss.sum())} rows")
sub.to_csv("submission.csv", index=False)
print(f"submission.csv: {len(sub)} rows, NaN={int(sub['tvt'].isna().sum())} ({time.time()-t0:.0f}s)")
