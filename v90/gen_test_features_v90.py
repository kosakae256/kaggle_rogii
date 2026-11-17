# -*- coding: utf-8 -*-
"""
v90 提出 test側特徴生成(★自己完結: v50/v89/v64 を import せず全inline)。
================================================================================
boot chain(ddoof→boot=softmax 5fold平均, v50 capaug emission像) で各testウェルの boot中心を求め:
  ① boot中心の学習emission c_learn(build_ev の EV[0], 5fold平均) を band 化 → sim_boot_test_v90.pkl {wid:(band,st,bid)}
  ② boot中心に SDF段(変位投票) → sdf_pred_test_v90.npz {wid: TVT}
create_candidates_v90.py SPLIT=test がこれを読む。モデルは bundle(dataset)から探索、出力は WORK(書込可)へ。
env: V90_ROOT(コード/モデル root) / V90_WORK(出力先, 既定=V90_ROOT/v90/artifacts_v90)
"""
import os, sys, io, glob, time, json, pickle
try: sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
except Exception: pass
import numpy as np, pandas as pd
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F

DEV = "cuda" if torch.cuda.is_available() else "cpu"
FAM = "capaug"; LOGIT = 8.0; PROD = True; TAU = 4.0
NB = 90; ND = 181; STEP = 0.5; STR = 16; W16, W64 = 16, 64; SIG_LK = 3.0
OFF = (np.arange(ND) - NB) * STEP
CAP_CH, CAP_DIM, CAP_DILS = 128, 96, (1, 2, 4, 8, 16, 32)
DILS_DDIL = [(1, 1), (2, 1), (4, 2), (8, 4), (16, 8)]
_ROOT = Path(os.environ.get("V90_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
WORK = Path(os.environ.get("V90_WORK", str(_ROOT / "v90" / "artifacts_v90"))); WORK.mkdir(parents=True, exist_ok=True)


def _find(fname, extra=()):
    """モデルファイルを bundle/local の候補dirから探す。"""
    cands = list(extra) + [_ROOT / "v90" / "artifacts_v90", _ROOT / "v64" / "artifacts_v64",
                           _ROOT / "v50" / "artifacts_v50", _ROOT]
    for c in glob.glob("/kaggle/input/*"): cands.append(Path(c))
    for c in glob.glob("/kaggle/input/*/*"): cands.append(Path(c))
    for d in cands:
        if (Path(d) / fname).exists(): return Path(d) / fname
    raise FileNotFoundError(fname)


def _detect_data():
    if os.environ.get("V90_DATA"): return Path(os.environ["V90_DATA"])
    for _p in ["/kaggle/input/rogii-wellbore-geology-prediction",
               "/kaggle/input/competitions/rogii-wellbore-geology-prediction",
               str(_ROOT / "rogii-wellbore-geology-prediction")]:
        if os.path.exists(_p): return Path(_p)
    _h = glob.glob("/kaggle/input/**/sample_submission.csv", recursive=True)
    return Path(os.path.dirname(_h[0]))
DATA = _detect_data(); TESTD = DATA / "test"
print(f"ROOT={_ROOT} WORK={WORK} DATA={DATA} DEV={DEV}")


def _rz(v):
    med = np.nanmedian(v); iqr = np.nanpercentile(v, 75) - np.nanpercentile(v, 25)
    return np.clip((v - med) / max(iqr / 1.349, 1e-6), -8, 8)
def _rm(v, w): return pd.Series(v).rolling(w, center=True, min_periods=1).mean().to_numpy()
def _rs(v, w): return pd.Series(v).rolling(w, center=True, min_periods=2).std().fillna(0).to_numpy()
def _detr(v, w=301):
    m = _rm(v, w); s = _rs(v, w); return np.clip((v - m) / np.maximum(s, 1e-3), -6, 6)
def feats_h(gr, md, z):
    g = pd.Series(gr).interpolate(limit_direction="both").to_numpy()
    g = np.where(np.isfinite(g), g, np.nanmedian(g)); g = np.where(np.isfinite(g), g, 0.0)
    z0 = _rz(g); s5, s21, s81 = _rm(z0, 5), _rm(z0, 21), _rm(z0, 81)
    d5 = np.gradient(s5) * 10.0; v21 = _rs(z0, 21); dt = _detr(z0, 301)
    dmd = np.maximum(np.gradient(md), 1e-3); dzdm = np.clip(np.gradient(z) / dmd, -0.2, 0.2) * 5.0
    return np.nan_to_num(np.stack([z0, s5, s21, s81, d5, v21, dt, dzdm], 1).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
def feats_t(gg):
    z0 = _rz(gg); s3, s9, s33 = _rm(z0, 3), _rm(z0, 9), _rm(z0, 33)
    d3 = np.gradient(s3) * 10.0; v9 = _rs(z0, 9); dt = _detr(z0, 301)
    return np.nan_to_num(np.stack([z0, s3, s9, s33, d3, v9, dt], 1).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
def smooth(v, fb, r=2):
    s = pd.Series(v, dtype="float64").interpolate(limit_direction="both").fillna(fb)
    return s.rolling(2 * r + 1, center=True, min_periods=1).mean().to_numpy()
def grid_from_pairs(tvt, gr, gmin, gmax):
    g = np.arange(gmin, gmax + STEP, STEP); out = np.full(len(g), np.nan)
    idx = np.round((tvt - gmin) / STEP).astype(int)
    ok = (idx >= 0) & (idx < len(g)) & np.isfinite(gr) & np.isfinite(tvt)
    if ok.sum():
        ssum = np.zeros(len(g)); scnt = np.zeros(len(g))
        np.add.at(ssum, idx[ok], gr[ok]); np.add.at(scnt, idx[ok], 1)
        m = scnt > 0; out[m] = ssum[m] / scnt[m]
    return g, out
def winsum(A, W):
    c = np.cumsum(np.concatenate([np.zeros((1,) + A.shape[1:]), A], 0), 0)
    lo = np.clip(np.arange(len(A)) - W // 2, 0, len(A)); hi = np.clip(np.arange(len(A)) + W - W // 2, 0, len(A))
    return c[hi] - c[lo], (hi - lo).astype(np.float64)
def ncc_img(g, T, W):
    V = np.isfinite(T); Tz = np.where(V, T, 0.0)
    n, _ = winsum(V.astype(np.float64), W)
    sg, _ = winsum(g[:, None] * V, W); st_, _ = winsum(Tz, W)
    sgg, _ = winsum((g[:, None] ** 2) * V, W); stt, _ = winsum(Tz ** 2, W)
    sgt, _ = winsum(g[:, None] * Tz, W); n = np.maximum(n, 1)
    mg = sg / n; mt = st_ / n
    vg = np.maximum(sgg / n - mg ** 2, 1e-6); vt = np.maximum(stt / n - mt ** 2, 1e-6)
    return np.clip((sgt / n - mg * mt) / np.sqrt(vg * vt), -1, 1), (n / W)


class Enc1D(nn.Module):
    def __init__(self, cin, ch, dout, dils=CAP_DILS):
        super().__init__(); self.inp = nn.Conv1d(cin, ch, 5, padding=2); self.blocks = nn.ModuleList()
        for d in dils:
            self.blocks.append(nn.Sequential(nn.Conv1d(ch, ch, 5, padding=2 * d, dilation=d), nn.GELU(), nn.Conv1d(ch, ch, 1)))
        self.norm = nn.ModuleList([nn.GroupNorm(8, ch) for _ in dils]); self.out = nn.Conv1d(ch, dout, 1)
    def forward(self, x):
        h = self.inp(x.t()[None])
        for blk, nm in zip(self.blocks, self.norm): h = nm(h + blk(h))
        return F.normalize(self.out(h)[0].t(), dim=1)
def _backbone(self, cin, ch):
    self.inp = nn.Conv2d(cin, ch, (5, 5), padding=(2, 2)); self.blocks = nn.ModuleList(); self.norms = nn.ModuleList()
    for dt, dd in DILS_DDIL:
        self.blocks.append(nn.Sequential(nn.Conv2d(ch, ch, (5, 5), padding=(2 * dt, 2 * dd), dilation=(dt, dd)), nn.GELU(), nn.Conv2d(ch, ch, 1)))
        self.norms.append(nn.GroupNorm(8, ch))
    self.gru = nn.GRU(ch, ch, batch_first=True, bidirectional=True); self.mix = nn.Linear(2 * ch, ch); self.ref = nn.Linear(2 * ch, 1)
def _trunk(self, x):
    h = self.inp(x)
    for blk, nm in zip(self.blocks, self.norms): h = nm(h + blk(h))
    g = h.mean(3).transpose(1, 2); g2, _ = self.gru(g); h = h + self.mix(g2).transpose(1, 2)[:, :, :, None]
    return h, g2
class DecV(nn.Module):
    def __init__(self, cin, ch=128): super().__init__(); _backbone(self, cin, ch); self.out = nn.Conv2d(ch, 1, 1)
    def forward(self, x): h, g2 = _trunk(self, x); return self.out(h)[:, 0], 2.0 * torch.tanh(self.ref(g2)[:, :, 0])
class DecVSDF(nn.Module):
    def __init__(self, cin, ch=128): super().__init__(); _backbone(self, cin, ch); self.disp = nn.Conv2d(ch, 1, 1)
    def forward(self, x): h, g2 = _trunk(self, x); return self.disp(h)[:, 0], 2.0 * torch.tanh(self.ref(g2)[:, :, 0])
def load_enc(f):
    ck = torch.load(_find(f"stageA_enc{FAM}_f{f}.pt"), map_location=DEV)
    a = Enc1D(8, CAP_CH, CAP_DIM).to(DEV); a.load_state_dict(ck["encA"]); a.eval()
    b = Enc1D(7, CAP_CH, CAP_DIM).to(DEV); b.load_state_dict(ck["encB"]); b.eval(); return a, b
def load_soft(tag, f):
    sd = torch.load(_find(f"v89_{tag}_f{f}.pt"), map_location=DEV)
    net = DecV(sd["inp.weight"].shape[1]).to(DEV); net.load_state_dict(sd, strict=True); net.eval(); return net
def load_sdf(f):
    sd = torch.load(_find(f"v89_sdf_ddil_f{f}.pt"), map_location=DEV)
    net = DecVSDF(sd["inp.weight"].shape[1]).to(DEV); net.load_state_dict(sd, strict=True); net.eval(); return net

t0 = time.time()
ENC = {f: load_enc(f) for f in range(5)}
MODELS = {"ddoof": {f: load_soft("ddoof", f) for f in range(5)}, "boot": {f: load_soft("boot", f) for f in range(5)},
          "sdf": {f: load_sdf(f) for f in range(5)}}
offs_t = torch.arange(-NB, NB + 1, device=DEV).float() * STEP
print(f"loaded encoders + ddoof/boot/sdf ({time.time()-t0:.0f}s)")


def build_ev(hw, tw, center, encA, encB):
    kn = hw["TVT_input"].notna().to_numpy(); ev = ~kn
    if kn.sum() < 10 or ev.sum() < 100: return None
    tw_tvt = tw["TVT"].to_numpy(float); tw_gr = tw["GR"].to_numpy(float)
    m = np.isfinite(tw_tvt) & np.isfinite(tw_gr); tw_tvt, tw_gr = tw_tvt[m], tw_gr[m]
    if len(tw_tvt) < 8: return None
    gmin, gmax = float(tw_tvt.min()), float(tw_tvt.max())
    gg = np.interp(np.arange(gmin, gmax + STEP, STEP), tw_tvt, tw_gr).astype(np.float32)
    Fh = feats_h(hw["GR"].to_numpy(float), hw["MD"].to_numpy(float), hw["Z"].to_numpy(float)); Ft = feats_t(gg)
    with torch.no_grad():
        a = encA(torch.from_numpy(Fh).to(DEV)).cpu().numpy(); b = encB(torch.from_numpy(Ft).to(DEV)).cpu().numpy()
    a_ev = a[ev]
    fb = float(np.nanmean(tw_gr)); gr_all = smooth(hw["GR"].to_numpy(float), fb, 2)
    g = gr_all[ev]; kn_tvt = hw["TVT_input"].to_numpy(float)[kn]; lk = float(kn_tvt[-1])
    z_all = hw["Z"].to_numpy(float); md_all = hw["MD"].to_numpy(float); evidx = np.where(ev)[0]
    if center is not None:
        stv = np.asarray(center, np.float64)
        if len(stv) < len(g): stv = np.pad(stv, (0, len(g) - len(stv)), mode="edge")
    else:
        stv = np.full(len(g), lk, np.float64)
    T = min(len(g), len(stv), len(a_ev)); g = g[:T]; stv = stv[:T]; a_ev = a_ev[:T]; evidx = evidx[:T]
    pos = stv[:, None] + OFF[None, :]
    idx = np.clip(np.round((pos - gmin) / STEP).astype(int), 0, len(gg) - 1)
    cl = np.einsum("td,tnd->tn", a_ev, b[idx]) * LOGIT; c_learn = 1.0 / (1.0 + np.exp(-cl))
    tws = smooth(tw_gr, fb, 2); Ttw = np.interp(pos, tw_tvt, tws)
    resid = gr_all[kn] - np.interp(kn_tvt, tw_tvt, tws); sig = float(np.nanstd(resid))
    sig = 15.0 if not np.isfinite(sig) or sig < 3 else max(sig, 5.0)
    c_raw = np.exp(-0.5 * ((g[:, None] - Ttw) / sig) ** 2); c_ncc, _ = ncc_img(g, Ttw, W64)
    gs_, vs_ = grid_from_pairs(kn_tvt, gr_all[kn], pos.min() - 1, pos.max() + 1)
    Tself = np.interp(pos, gs_, np.where(np.isfinite(vs_), vs_, np.interp(gs_, tw_tvt, tws)))
    c_self, _ = ncc_img(g, Tself, W16)
    nb_ = (T + STR - 1) // STR; sel = np.clip(np.arange(nb_) * STR + STR // 2, 0, T - 1)
    c_lk = np.repeat(np.exp(-0.5 * (OFF / SIG_LK) ** 2)[None, :], T, 0)
    md_lk = float(md_all[kn][-1]); c_mds = np.repeat((np.clip((md_all[ev][:T] - md_lk) / 1000, 0, 3) / 3)[:, None], ND, 1)
    z_lk = float(z_all[kn][-1]); c_z = np.repeat((np.clip((z_all[ev][:T] - z_lk) / 50, -3, 3) / 3)[:, None], ND, 1)
    chs = [c_learn, c_raw, c_ncc, c_self, c_lk, c_mds, c_z]
    if PROD: chs += [c_learn * c_raw, c_learn * np.clip(c_ncc, 0, 1)]
    EV = np.stack([c[sel] for c in chs], 0).astype(np.float32)
    bid = np.minimum(np.arange(T) // STR, nb_ - 1)
    return EV, stv, bid, evidx, T
def readout_sdf(disp, ref):
    vote = offs_t[None, None, :] + disp; w = torch.softmax(-disp.abs() / TAU, dim=-1)
    return (w * vote).sum(-1) + ref
def stage(hw, tw, center, models5, is_sdf, want_band=False):
    Asum = rfsum = base = bandsum = None
    for f in range(5):
        r = build_ev(hw, tw, center, ENC[f][0], ENC[f][1])
        if r is None: return None
        EV, stv, bid, evidx, T = r
        with torch.no_grad():
            o, rf = models5[f](torch.from_numpy(EV)[None].to(DEV))
            A = o[0, :EV.shape[1]] if is_sdf else torch.softmax(o[0, :EV.shape[1]], -1)
        Asum = A if Asum is None else Asum + A
        rfv = rf[0, :EV.shape[1]]; rfsum = rfv if rfsum is None else rfsum + rfv
        if want_band:
            bb = np.asarray(EV[0], np.float32); bandsum = bb if bandsum is None else bandsum + bb
        base = (stv, bid, evidx)
    stv, bid, evidx = base; Aavg = Asum / 5.0; rfavg = rfsum / 5.0
    if is_sdf: dh = readout_sdf(Aavg[None], rfavg[None])[0].cpu().numpy()
    else: dh = ((Aavg @ offs_t) + rfavg).cpu().numpy()
    z = stv + np.asarray(dh, np.float64)[bid]
    if want_band: return z, evidx, (bandsum / 5.0).astype(np.float16), np.asarray(stv, np.float32), np.asarray(bid, np.int32)
    return z, evidx


def main():
    wids = sorted(set(os.path.basename(f).split("__")[0] for f in glob.glob(str(TESTD / "*__horizontal_well.csv"))))
    sim = {}; sdf = {}; nskip = 0; t1 = time.time()
    for k, wid in enumerate(wids):
        try:
            hw = pd.read_csv(TESTD / f"{wid}__horizontal_well.csv")
            tw = pd.read_csv(TESTD / f"{wid}__typewell.csv").sort_values("TVT")
        except Exception:
            nskip += 1; continue
        o1 = stage(hw, tw, None, MODELS["ddoof"], is_sdf=False)          # ddoof
        if o1 is None: nskip += 1; continue
        o2 = stage(hw, tw, o1[0], MODELS["boot"], is_sdf=False, want_band=True)   # boot(+band)
        if o2 is None: nskip += 1; continue
        ctr, evidx, band, st0, bid0 = o2
        sim[wid] = dict(band=band, st=st0, bid=bid0)
        o3 = stage(hw, tw, ctr, MODELS["sdf"], is_sdf=True)              # SDF pred
        if o3 is not None: sdf[wid] = np.asarray(o3[0], np.float32)
        if (k + 1) % 25 == 0: print(f"  {k+1}/{len(wids)} ({time.time()-t1:.0f}s)", flush=True)
    pickle.dump(sim, open(WORK / "sim_boot_test_v90.pkl", "wb"), protocol=4)
    np.savez_compressed(WORK / "sdf_pred_test_v90.npz", **sdf)
    print(f"saved {WORK}/sim_boot_test_v90.pkl ({len(sim)}井) + sdf_pred_test_v90.npz ({len(sdf)}井) skip={nskip}")


if __name__ == "__main__":
    main()
