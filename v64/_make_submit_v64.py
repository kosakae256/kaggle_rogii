# submit_unet_v64.ipynb を生成する(1回実行用)
import json, sys
sys.stdout = open(1, "w", encoding="utf-8", closefd=False)

def code(src): return dict(cell_type="code", metadata={}, execution_count=None, outputs=[], source=src.splitlines(keepends=True))
def md(src): return dict(cell_type="markdown", metadata={}, source=src.splitlines(keepends=True))

c0 = """# v64 — PF-free 2D UNet 提出(9ch GR照合像, 実行時構築)
- dataset: **rogii-v64-artifacts**(= artifacts_v64 の ckpt類 + struct_field_v38.npz。画像pklは不要=test側で実行時構築)
- チェーン: test struct prior 構築 → 9ch画像(PF不使用) → UNet(5fold平均, submit_pack_v64.json の ckpt 群) → z = st + dh
- 出力は生値(pp/SGなし)= ローカルCVと同条件"""

c1 = r'''# ===== 設定 + test struct prior 実行時構築(struct_field_v38.npz) =====
import os, json, time, glob as _glob, pickle
import numpy as np, pandas as pd
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F

_C = [Path("/kaggle/input/rogii-v64-artifacts"), Path(r"C:\Users\kosaka256\Documents\rogii_claude\v64\artifacts_v64")]
ART = next(p for p in _C if p.exists())
def _detect_data():
    for _p in ["/kaggle/input/rogii-wellbore-geology-prediction",
               "/kaggle/input/competitions/rogii-wellbore-geology-prediction",
               r"C:\Users\kosaka256\Documents\rogii_claude\rogii-wellbore-geology-prediction"]:
        if os.path.exists(_p): return Path(_p)
    _h = _glob.glob("/kaggle/input/**/sample_submission.csv", recursive=True)
    return Path(os.path.dirname(_h[0]))
DATA = _detect_data(); TESTD = DATA / "test"; TRAIND = DATA / "train"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
print("ART:", ART, "| DATA:", DATA, "| device:", DEV)

FZ = np.load(ART / "struct_field_v38.npz")
def build_test_struct(test_dir):
    """test坑井の構造prior(k16 dip場)を実行時構築(v46系。返り値 {wid:{tvt,sig}} eval行=最終既知行の次以降)。"""
    FLD = FZ["field"]; KAP = FZ["kappa"]; CEN = FZ["centers"]; SGC = FZ["sigs"]
    TH = float(FZ["theta0"]); K = int(FZ["K"]); KK = int(FZ["knn_k"]); KH = float(FZ["knn_h"])
    GT = float(FZ["gate"]); KB = FZ["kbins"]; SMIN = float(FZ["sig_min"]); SMAX = float(FZ["sig_max"])
    d = {}
    for fp in _glob.glob(str(Path(test_dir) / "*__horizontal_well.csv")):
        wid = os.path.basename(fp).split("__")[0]
        try:
            df = pd.read_csv(fp, usecols=["X", "Y", "Z", "TVT_input"])
            X = df["X"].to_numpy(); Y = df["Y"].to_numpy(); Z = df["Z"].to_numpy(); ti = df["TVT_input"].to_numpy()
            fin = np.where(np.isfinite(ti))[0]
            if len(fin) == 0: continue
            s = int(fin.max()); ndz = -np.diff(Z)[s:]; n = len(ndz)
            if n < 2: continue
            edges = np.linspace(0, n, K + 1)
            segid = np.clip(np.searchsorted(edges[1:], np.arange(1, n + 1.0), side="left"), 0, K - 1)
            mid = np.empty((K, 2)); proj = np.empty(K)
            for j in range(K):
                f0 = s + 1 + int(edges[j]); f1 = min(s + 1 + max(int(edges[j + 1]) - 1, int(edges[j])), len(X) - 1)
                az = np.degrees(np.arctan2(Y[f1] - Y[f0], X[f1] - X[f0])) % 360
                mid[j] = ((X[f0] + X[f1]) / 2, (Y[f0] + Y[f1]) / 2); proj[j] = np.cos(np.radians(az - TH))
            Dh = np.empty(K); dd = np.empty(K)
            for j in range(K):
                d2 = (FLD[:, 0] - mid[j, 0]) ** 2 + (FLD[:, 1] - mid[j, 1]) ** 2
                sel = np.argpartition(d2, KK)[:KK]
                wv = np.exp(np.maximum(-d2[sel] / (2 * KH ** 2), -700))
                Dh[j] = np.sum(wv * FLD[sel, 2]) / np.sum(wv); dd[j] = np.sqrt(np.median(d2[sel]))
            gg = ndz + (Dh * proj)[segid]; gg = np.where(np.abs(proj[segid]) < GT, 0.0, gg)
            kstep = KAP[np.digitize(dd, KB[1:-1])][segid]
            tvt = float(ti[s]) + np.cumsum(kstep * gg)
            sig = np.clip(np.interp(dd[segid], CEN, SGC), SMIN, SMAX)
            d[wid] = dict(tvt=tvt.astype(np.float64), sig=sig.astype(np.float64))
        except Exception:
            continue
    print(f"[struct] test実行時構築: {len(d)} wells")
    return d
t0 = time.time()
STRUCT_T = build_test_struct(TESTD)'''

c2 = r'''# ===== 9ch画像ビルダ部品(build_images_v64.py と同一手順)+ train近傍プール =====
NB = 90; ND = 181; STEP = 0.5; STR = 16
OFF = (np.arange(ND) - NB) * STEP
W16, W64 = 16, 64; SIG_LK = 3.0

def smooth(v, fb, r=2):
    s = pd.Series(v, dtype="float64").interpolate(limit_direction="both").fillna(fb)
    return s.rolling(2 * r + 1, center=True, min_periods=1).mean().to_numpy()

def grid_from_pairs(tvt, gr, gmin, gmax):
    g = np.arange(gmin, gmax + STEP, STEP)
    out = np.full(len(g), np.nan)
    idx = np.round((tvt - gmin) / STEP).astype(int)
    ok = (idx >= 0) & (idx < len(g)) & np.isfinite(gr) & np.isfinite(tvt)
    if ok.sum():
        ssum = np.zeros(len(g)); scnt = np.zeros(len(g))
        np.add.at(ssum, idx[ok], gr[ok]); np.add.at(scnt, idx[ok], 1)
        m = scnt > 0
        out[m] = ssum[m] / scnt[m]
    return g, out

def winsum(A, W):
    c = np.cumsum(np.concatenate([np.zeros((1,) + A.shape[1:]), A], 0), 0)
    lo = np.clip(np.arange(len(A)) - W // 2, 0, len(A))
    hi = np.clip(np.arange(len(A)) + W - W // 2, 0, len(A))
    return c[hi] - c[lo], (hi - lo).astype(np.float64)

def ncc_img(g, T, W):
    V = np.isfinite(T); Tz = np.where(V, T, 0.0); gs = g[:, None] * V
    n, _ = winsum(V.astype(np.float64), W)
    sg, _ = winsum(gs, W); st_, _ = winsum(Tz, W)
    sgg, _ = winsum((g[:, None] ** 2) * V, W); stt, _ = winsum(Tz ** 2, W)
    sgt, _ = winsum(g[:, None] * Tz, W)
    n = np.maximum(n, 1)
    mg = sg / n; mt = st_ / n
    vg = np.maximum(sgg / n - mg ** 2, 1e-6); vt = np.maximum(stt / n - mt ** 2, 1e-6)
    return np.clip((sgt / n - mg * mt) / np.sqrt(vg * vt), -1, 1), (n / W)

# --- train 近傍プール(既知prefix行の (TVT_input, GR)。test坑井は除外不要) ---
print("train近傍プール構築中 ...")
TRW = sorted(set(os.path.basename(f).split("__")[0] for f in _glob.glob(str(TRAIND / "*__horizontal_well.csv"))))
XYs = []; KN = {}
for w in TRW:
    h = pd.read_csv(TRAIND / f"{w}__horizontal_well.csv", usecols=["X", "Y", "GR", "TVT_input"])
    kn = h["TVT_input"].notna().to_numpy()
    XYs.append((float(h["X"].mean()), float(h["Y"].mean())))
    KN[w] = (h["TVT_input"].to_numpy(float)[kn], h["GR"].to_numpy(float)[kn])
XY = np.array(XYs)
from scipy.spatial import cKDTree
tree = cKDTree(XY)
print(f"  pool: {len(TRW)} wells ({time.time()-t0:.0f}s)")

def build_image(wid):
    hw = pd.read_csv(TESTD / f"{wid}__horizontal_well.csv")
    tw = pd.read_csv(TESTD / f"{wid}__typewell.csv").sort_values("TVT")
    t_tvt = tw["TVT"].to_numpy(float); t_gr = tw["GR"].to_numpy(float)
    fin = np.isfinite(t_tvt) & np.isfinite(t_gr); t_tvt, t_gr = t_tvt[fin], t_gr[fin]
    sp = STRUCT_T.get(wid)
    if sp is None or len(t_tvt) < 50: return None
    ti = hw["TVT_input"].to_numpy(float)
    finq = np.where(np.isfinite(ti))[0]
    if len(finq) < 10: return None
    s = int(finq.max())
    evi = np.arange(s + 1, len(hw))                    # eval行 = 最終既知行の次以降(struct と同整列)
    st = np.asarray(sp["tvt"], np.float64); sigr = np.asarray(sp["sig"], np.float64)
    T = min(len(evi), len(st))
    if T < 20: return None
    evi = evi[:T]
    fb = float(np.nanmean(t_gr))
    gr_all = smooth(hw["GR"].to_numpy(float), fb, 2)
    g = gr_all[evi]; stv = st[:T]
    kn_m = np.isfinite(ti); kn_tvt = ti[kn_m]
    resid = gr_all[kn_m] - np.interp(kn_tvt, t_tvt, smooth(t_gr, fb, 2))
    sigw = float(np.nanstd(resid)); sigw = 15.0 if not np.isfinite(sigw) or sigw < 3 else max(sigw, 5.0)
    pos = stv[:, None] + OFF[None, :]
    tws = smooth(t_gr, fb, 2)
    Ttw = np.interp(pos, t_tvt, tws)
    gmin, gmax = pos.min() - 1, pos.max() + 1
    gs_, vs_ = grid_from_pairs(kn_tvt, gr_all[kn_m], gmin, gmax)
    Tself = np.interp(pos, gs_, np.where(np.isfinite(vs_), vs_, np.interp(gs_, t_tvt, tws)))
    Aself = np.interp(pos, gs_, np.isfinite(vs_).astype(float))
    cx, cy = float(hw["X"].mean()), float(hw["Y"].mean())
    _, nidx = tree.query([cx, cy], k=8)
    js = [TRW[j] for j in np.atleast_1d(nidx)]
    ny = np.concatenate([KN[j][0] for j in js]); ng = np.concatenate([KN[j][1] for j in js])
    gn_, vn_ = grid_from_pairs(ny, ng, gmin, gmax)
    Tnbr = np.interp(pos, gn_, np.where(np.isfinite(vn_), vn_, np.interp(gn_, t_tvt, tws)))
    Anbr = np.interp(pos, gn_, np.isfinite(vn_).astype(float))
    d16, _ = winsum((g[:, None] - Ttw), W16)
    c0_ = (d16 / W16) / sigw
    c1_, _ = ncc_img(g, Ttw, W16)
    c2_, _ = ncc_img(g, Ttw, W64)
    c3_, _ = ncc_img(g, Tself, W16)
    c4_, _ = ncc_img(g, Tnbr, W16)
    a5s, _ = winsum(Aself, W16); a5n, _ = winsum(Anbr, W16)
    c5_ = 0.5 * (a5s / W16) + 0.5 * (a5n / W16)
    nb_ = (T + STR - 1) // STR
    sel = np.clip(np.arange(nb_) * STR + STR // 2, 0, T - 1)
    sig_b = np.clip(sigr[:T], 1.0, 45.0)[sel]
    c6_b = np.exp(-0.5 * (OFF[None, :] / sig_b[:, None]) ** 2)
    lk = float(ti[s]); zl_b = lk - stv[sel]
    c7_b = np.exp(-0.5 * ((OFF[None, :] - zl_b[:, None]) / SIG_LK) ** 2)
    md_all = hw["MD"].to_numpy(float)
    mds_row = np.maximum(md_all[evi] - float(md_all[s]), 0.0)
    c8_b = np.repeat((np.clip(mds_row[sel] / 1000.0, 0, 3.0) / 3.0)[:, None], ND, 1)
    EV = np.concatenate([np.stack([c[sel] for c in (c0_, c1_, c2_, c3_, c4_, c5_)], 0),
                         np.stack([c6_b, c7_b, c8_b], 0)], 0).astype(np.float32)
    bid = np.minimum(np.arange(T) // STR, nb_ - 1)
    ids = [f"{wid}_{int(i)}" for i in evi]
    return EV, stv, bid, ids'''

c3 = r'''# ===== UNet2D(train_unet_v64.py と同一)+ ckpt ロード =====
PACK = (json.loads((ART / "submit_pack_v64.json").read_text())
        if (ART / "submit_pack_v64.json").exists()
        else dict(ckpts=sorted(os.path.basename(p) for p in _glob.glob(str(ART / "v64_unet*_f*.pt"))), ch=128))
CH = int(PACK.get("ch", 128)); CIN = 9
print("ckpts:", PACK["ckpts"])
assert PACK["ckpts"], "UNet ckpt が見つからない(submit_pack_v64.json か v64_unet*_f*.pt を dataset に)"

class UNet2D(nn.Module):
    def __init__(self, cin=CIN, ch=CH):
        super().__init__()
        def blk(ci, co):
            return nn.Sequential(nn.Conv2d(ci, co, 3, padding=1), nn.GELU(),
                                 nn.Conv2d(co, co, 3, padding=1), nn.GELU(), nn.GroupNorm(8, co))
        c1, c2, c3, c4 = ch // 2, ch, ch, ch
        self.e1 = blk(cin, c1); self.e2 = blk(c1, c2); self.e3 = blk(c2, c3); self.bott = blk(c3, c4)
        self.pool = nn.AvgPool2d(2)
        self.u3 = nn.ConvTranspose2d(c4, c3, 2, stride=2); self.d3 = blk(c3 + c3, c3)
        self.u2 = nn.ConvTranspose2d(c3, c2, 2, stride=2); self.d2 = blk(c2 + c2, c2)
        self.u1 = nn.ConvTranspose2d(c2, c1, 2, stride=2); self.d1 = blk(c1 + c1, c1)
        self.gru = nn.GRU(c1, ch, batch_first=True, bidirectional=True)
        self.mix = nn.Linear(2 * ch, c1)
        self.out = nn.Conv2d(c1, 1, 1)
        self.ref = nn.Linear(2 * ch, 1)
    def forward(self, x):
        B, C, T, D = x.shape
        pt = (-T) % 8; pdd = (-D) % 8
        if pt or pdd: x = F.pad(x, (0, pdd, 0, pt))
        h1 = self.e1(x)
        h2 = self.e2(self.pool(h1))
        h3 = self.e3(self.pool(h2))
        hb = self.bott(self.pool(h3))
        d3 = self.d3(torch.cat([self.u3(hb), h3], 1))
        d2 = self.d2(torch.cat([self.u2(d3), h2], 1))
        d1 = self.d1(torch.cat([self.u1(d2), h1], 1))
        d1 = d1[:, :, :T, :D]
        g = d1.mean(3).transpose(1, 2)
        g2, _ = self.gru(g)
        h = d1 + self.mix(g2).transpose(1, 2)[:, :, :, None]
        return self.out(h)[:, 0], 2.0 * torch.tanh(self.ref(g2)[:, :, 0])

NETS = []
for nmk in PACK["ckpts"]:
    net = UNet2D().to(DEV)
    net.load_state_dict(torch.load(ART / nmk, map_location=DEV), strict=False)
    net.eval(); NETS.append(net)
offs_t = torch.arange(-NB, NB + 1, device=DEV).float() * STEP
print(f"loaded {len(NETS)} nets")'''

c4 = r'''# ===== 推論: test全坑井 → z = st + dh =====
t1 = time.time()
wids = sorted(set(os.path.basename(f).split("__")[0] for f in _glob.glob(str(TESTD / "*__horizontal_well.csv"))))
pids = []; pvals = []; nskip = 0
with torch.no_grad():
    for k, wid in enumerate(wids):
        r = build_image(wid)
        if r is None:
            nskip += 1; continue
        EV, stv, bid, ids = r
        x = torch.from_numpy(EV)[None].to(DEV)
        nb_ = EV.shape[1]
        dhs = []
        for net in NETS:
            lg, rf = net(x)
            P = torch.softmax(lg[0, :nb_], -1)
            dhs.append(((P @ offs_t) + rf[0, :nb_]).cpu().numpy())
        dh = np.mean(dhs, 0)
        z = stv + np.asarray(dh, np.float64)[bid]
        pids += ids; pvals += list(z)
        if (k + 1) % 50 == 0: print(f"  {k+1}/{len(wids)} ({time.time()-t1:.0f}s)", flush=True)
pred_df = pd.DataFrame({"id": pids, "tvt": pvals})
print(f"predicted: {len(pred_df)} rows / skip={nskip} wells ({time.time()-t1:.0f}s)")'''

c5 = r'''# ===== submission(欠け行は lk でフォールバック) =====
ss = pd.read_csv(DATA / "sample_submission.csv")
sub = ss[["id"]].merge(pred_df, on="id", how="left")
miss = sub["tvt"].isna()
if miss.any():
    lkmap = {}
    for wid in set(i.rsplit("_", 1)[0] for i in sub.loc[miss, "id"]):
        h = pd.read_csv(TESTD / f"{wid}__horizontal_well.csv", usecols=["TVT_input"])
        v = h["TVT_input"].dropna()
        lkmap[wid] = float(v.iloc[-1]) if len(v) else 0.0
    sub.loc[miss, "tvt"] = [lkmap[i.rsplit("_", 1)[0]] for i in sub.loc[miss, "id"]]
    print(f"lkフォールバック: {int(miss.sum())} rows")
sub.to_csv("submission.csv", index=False)
print("submission.csv:", len(sub), "rows")'''

nb = dict(nbformat=4, nbformat_minor=5,
          metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                    "language_info": {"name": "python"}},
          cells=[md(c0), code(c1), code(c2), code(c3), code(c4), code(c5)])
json.dump(nb, open("v64/submit_unet_v64.ipynb", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("written v64/submit_unet_v64.ipynb")
