# -*- coding: utf-8 -*-
"""v89 SDF転用 — 出力ヘッドを softmax-δ分類 → 符号付き距離場(変位回帰)に差替。
================================================================================
入力(EV画像/bootstrap中心)は softmax版と同一(learned_boot2.pkl等を流用)。archはddil/unet。
各セル(t,δ)が「真δまでの符号付きオフセット s=dtrue-offs[δ]」を回帰(target clamp±R)。
読み出し: 各セルが true δ を offs[δ]+s と投票、w=softmax(-|s|/τ)で近傍セル重み付け平均 → δ_hat。
  env: EVPKL / FOLD / ARCH(ddil既定) / SAVE / R(clamp ft=20) / TAU(投票温度ft=4) / CELLW(近傍重み ft=12)
       AUXCE(softmax補助ロス重み=0) / VFRAC / TOKMAX / SEED / MAXMIN / SAVE_OOF / SAVE_DIST
softmax版(train_unet_v89.py)と同じ 5fold/early stop/中心=EV pkl。standalone RMSE と boot2 OOFとのρを測る。
"""
import os, sys, json, time, pickle
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

P0 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART = os.path.join(P0, "v64", "artifacts_v64"); os.makedirs(ART, exist_ok=True)
EVPKL = os.environ.get("EVPKL", os.path.join(P0, "v89", "artifacts_v89", "learned_boot2.pkl"))
ARCH = os.environ.get("ARCH", "ddil")
FOLD = int(os.environ.get("FOLD", "0"))
VFRAC = float(os.environ.get("VFRAC", "0.25")); TOKMAX = int(os.environ.get("TOKMAX", "600")); BW = 4
CH = int(os.environ.get("CHN", "128")); PAT = 12; LRPAT = 4; MAXMIN = float(os.environ.get("MAXMIN", "70"))
R = float(os.environ.get("R", "20.0"))          # 変位target clamp(ft)
TAU = float(os.environ.get("TAU", "4.0"))       # 投票温度(ft)
CELLW = float(os.environ.get("CELLW", "12.0"))  # 近傍セル重み(ft)。大=全セル均等
AUXCE = float(os.environ.get("AUXCE", "0.0"))   # softmax補助ロス
LAM = float(os.environ.get("LAM", "0.05"))      # δ_hat の MSE 重み
BDROP = float(os.environ.get("BDROP", "0.15")); NOISE = 0.02
NB = int(os.environ.get("NBIN", "90")); ND = 2 * NB + 1; STEP = 0.5
DEV = "cuda" if torch.cuda.is_available() else "cpu"
if DEV == "cuda": torch.cuda.set_per_process_memory_fraction(VFRAC)
SEED = int(os.environ.get("SEED", "0"))
import random
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
SAVE = os.environ.get("SAVE", f"v89_sdf_{ARCH}_f{FOLD}")

t0 = time.time()
EVD = pickle.loads(open(EVPKL, "rb").read())
WELLS = sorted(EVD.keys())
CIN = int(EVD[WELLS[0]]["EV"].shape[0]); NPF = max(CIN - 11, 0)
print(f"pkl={os.path.basename(EVPKL)} wells={len(WELLS)} CIN={CIN} ARCH={ARCH} R={R} TAU={TAU} CELLW={CELLW} ({time.time()-t0:.0f}s)", flush=True)

DILS = {"base": [(1, 1), (2, 1), (4, 1), (8, 1), (16, 1)],
        "ddil": [(1, 1), (2, 1), (4, 2), (8, 4), (16, 8)],
        "unet": [(1, 1)]}[ARCH]


class DecV(nn.Module):
    """train_unet_v89.py の DecV と同型。out=変位場(ft), ref=行バイアス。"""
    def __init__(self, cin=CIN, ch=CH):
        super().__init__()
        self.inp = nn.Conv2d(cin, ch, (5, 5), padding=(2, 2))
        self.blocks = nn.ModuleList(); self.norms = nn.ModuleList()
        for dt, dd in DILS:
            self.blocks.append(nn.Sequential(
                nn.Conv2d(ch, ch, (5, 5), padding=(2 * dt, 2 * dd), dilation=(dt, dd)), nn.GELU(),
                nn.Conv2d(ch, ch, 1)))
            self.norms.append(nn.GroupNorm(8, ch))
        self.gru = nn.GRU(ch, ch, batch_first=True, bidirectional=True)
        self.mix = nn.Linear(2 * ch, ch)
        self.disp = nn.Conv2d(ch, 1, 1)                 # 符号付き変位場(ft)
        nn.init.zeros_(self.disp.weight); nn.init.zeros_(self.disp.bias)
        self.ref = nn.Linear(2 * ch, 1)
        nn.init.zeros_(self.ref.weight); nn.init.zeros_(self.ref.bias)
        self.ce = nn.Conv2d(ch, 1, 1) if AUXCE > 0 else None   # 補助softmax(任意)
    def forward(self, x):
        h = self.inp(x)
        for blk, nm in zip(self.blocks, self.norms):
            h = nm(h + blk(h))
        g = h.mean(3).transpose(1, 2)
        g2, _ = self.gru(g)
        h = h + self.mix(g2).transpose(1, 2)[:, :, :, None]
        disp = self.disp(h)[:, 0]                        # (B,T,D) ft
        ce = self.ce(h)[:, 0] if self.ce is not None else None
        return disp, 2.0 * torch.tanh(self.ref(g2)[:, :, 0]), ce


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
        self.disp = nn.Conv2d(c1, 1, 1); nn.init.zeros_(self.disp.weight); nn.init.zeros_(self.disp.bias)
        self.ref = nn.Linear(2 * ch, 1); nn.init.zeros_(self.ref.weight); nn.init.zeros_(self.ref.bias)
        self.ce = nn.Conv2d(c1, 1, 1) if AUXCE > 0 else None
    def forward(self, x):
        B, C, T, D = x.shape
        pt = (-T) % 8; pdd = (-D) % 8
        if pt or pdd: x = F.pad(x, (0, pdd, 0, pt))
        h1 = self.e1(x); h2 = self.e2(self.pool(h1)); h3 = self.e3(self.pool(h2)); hb = self.bott(self.pool(h3))
        d3 = self.d3(torch.cat([self.u3(hb), h3], 1))
        d2 = self.d2(torch.cat([self.u2(d3), h2], 1))
        d1 = self.d1(torch.cat([self.u1(d2), h1], 1))[:, :, :T, :D]
        g = d1.mean(3).transpose(1, 2); g2, _ = self.gru(g)
        h = d1 + self.mix(g2).transpose(1, 2)[:, :, :, None]
        disp = self.disp(h)[:, 0]; ce = self.ce(h)[:, 0] if self.ce is not None else None
        return disp, 2.0 * torch.tanh(self.ref(g2)[:, :, 0]), ce


def make_batch(wns):
    Tm = max(EVD[w]["EV"].shape[1] for w in wns); B = len(wns)
    x = torch.zeros(B, CIN, Tm, ND); m = torch.zeros(B, Tm)
    dt = torch.full((B, Tm), float("nan")); wg = torch.zeros(B, Tm)
    for i, w in enumerate(wns):
        e = EVD[w]; T = e["EV"].shape[1]
        x[i, :, :T] = torch.from_numpy(e["EV"].astype(np.float32)); m[i, :T] = 1.0
        dt[i, :T] = torch.from_numpy(np.asarray(e["dtrue"], np.float32))
        wg[i, :T] = torch.from_numpy(np.asarray(e["wgt"], np.float32))
    return x.to(DEV), m.to(DEV), dt.to(DEV), wg.to(DEV)


offs_t = torch.arange(-NB, NB + 1, device=DEV).float() * STEP


def readout(disp, ref):
    """変位投票: 各セル vote=offs+disp, 近傍(|disp|小)を softmax(-|disp|/τ)で重み付け平均。"""
    vote = offs_t[None, None, :] + disp
    w = torch.softmax(-disp.abs() / TAU, dim=-1)
    return (w * vote).sum(-1) + ref


def bucket(ws, rng):
    lens = {w: EVD[w]["EV"].shape[1] for w in ws}
    order = rng.permutation(len(ws)); ws_ = [ws[j] for j in order]
    wl = [w for i in range(0, len(ws_), 64) for w in sorted(ws_[i:i + 64], key=lambda x: lens[x])]
    out = []; cur = []
    for w in wl:
        t = cur + [w]
        if cur and (len(t) * max(lens[x] for x in t) > TOKMAX or len(t) > BW): out.append(cur); cur = [w]
        else: cur = t
    if cur: out.append(cur)
    return out


def batch_loss(wns, train):
    x, m, dtr, wg = make_batch(wns)
    if train:
        if BDROP > 0 and NPF > 0:
            msk = (torch.rand(x.shape[0], NPF, device=DEV) > BDROP).float()[:, :, None, None]
            x[:, 10:10 + NPF] = x[:, 10:10 + NPF] * msk
        x = x + NOISE * torch.randn_like(x)
    disp, rf, ce = net(x)
    ok = m.bool() & torch.isfinite(dtr) & (wg > 0)
    if not ok.any(): return None
    dtn = torch.nan_to_num(dtr)
    # 符号付き変位target(clamp±R) と 近傍重み(cell weight)
    s_tgt = (dtn[:, :, None] - offs_t[None, None, :]).clamp(-R, R)
    cw = torch.exp(-(dtn[:, :, None] - offs_t[None, None, :]).abs() / CELLW)   # 近傍セル重視
    okc = ok[:, :, None].expand_as(s_tgt)
    sl = (F.smooth_l1_loss(disp, s_tgt, reduction="none", beta=2.0) * cw)[okc].sum() / (cw[okc].sum() + 1e-9)
    dmean = readout(disp, rf)
    se = (dmean - dtn) ** 2
    wgn = wg / wg[ok].mean()
    loss = sl + LAM * (se[ok] * wgn[ok]).mean()
    if AUXCE > 0 and ce is not None:
        kt = (dtr / STEP + NB); okce = ok & (kt >= 0) & (kt <= ND - 1)
        if okce.any():
            ktc = torch.nan_to_num(kt, nan=float(NB)).clamp(0, ND - 1)
            d = (torch.arange(ND, device=DEV)[None, None, :] - ktc[:, :, None])
            tgt = torch.exp(-0.5 * (d / 2.0) ** 2); tgt = tgt / tgt.sum(-1, keepdim=True).clamp_min(1e-12)
            lsm = F.log_softmax(ce, dim=-1); cel = -(tgt * lsm).sum(-1)
            loss = loss + AUXCE * (cel[okce] * wgn[okce]).mean()
    return loss


val = WELLS[FOLD::5]; trn = [w for w in WELLS if w not in set(val)]
net = (UNet2D() if ARCH == "unet" else DecV()).to(DEV)
LR = float(os.environ.get("LR", "3e-4"))
opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-4)
sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=LRPAT)
rng = np.random.default_rng(FOLD)


def val_loss():
    net.eval(); tot = 0.0; n = 0
    with torch.no_grad():
        for wns in bucket(val, np.random.default_rng(1)):
            l = batch_loss(wns, False)
            if l is None: continue
            tot += float(l) * len(wns); n += len(wns)
    return tot / max(n, 1)


best = float("inf"); best_state = None; bad = 0; ep = 0; t1 = time.time()
while True:
    ep += 1; net.train(); tl = []
    for wns in bucket(trn, rng):
        try:
            l = batch_loss(wns, True)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); continue
        if l is None or not torch.isfinite(l): continue
        opt.zero_grad(set_to_none=True); l.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step(); tl.append(float(l.detach()))
    vl = val_loss(); sch.step(vl); star = ""
    if vl < best - 1e-4:
        best = vl; best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}; bad = 0; star = " *best"
    else: bad += 1
    print(f"ep{ep} train={np.mean(tl):.4f} val={vl:.4f}{star} (bad={bad}, {time.time()-t1:.0f}s)", flush=True)
    if bad >= PAT or (time.time() - t1) / 60 > MAXMIN:
        print(f"stop @ep{ep} best={best:.4f}"); break
net.load_state_dict(best_state); net.eval()
dh = {}
with torch.no_grad():
    for w in val:
        x, m, dtr, wg = make_batch([w]); T = EVD[w]["EV"].shape[1]
        disp, rf, _ = net(x)
        dh[w] = readout(disp[:, :T], rf[:, :T])[0].cpu().numpy()
es = []
for w in val:
    e = EVD[w]; z = e["st"] + np.asarray(dh[w], np.float64)[e["bid"]]
    er = z - e["y_ev"]; es.append(er[np.isfinite(er)])
r = float(np.sqrt(np.mean(np.concatenate(es) ** 2)))
print(f"\n[{SAVE}] SDF fold{FOLD} RMSE = {r:.4f}  (softmax boot2 fold0=10.88/overall11.94 と同fold比較)")
if os.environ.get("SAVE_OOF", "0") == "1":
    _oof = {w: (EVD[w]["st"] + np.asarray(dh[w], np.float64)[EVD[w]["bid"]]).astype("float32") for w in val}
    np.savez_compressed(os.path.join(ART, f"{SAVE}_oof.npz"), **_oof)
    print(f"[oof] saved {SAVE}_oof.npz ({len(_oof)} wells)")
torch.save(net.state_dict(), os.path.join(ART, f"{SAVE}.pt"))
json.dump(dict(rmse=r, arch=ARCH, R=R, TAU=TAU, CELLW=CELLW, AUXCE=AUXCE, best_val=best, epochs=ep,
               pkl=os.path.basename(EVPKL)), open(os.path.join(ART, f"{SAVE}.json"), "w"), indent=1)
