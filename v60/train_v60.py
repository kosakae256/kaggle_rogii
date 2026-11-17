# v60 trainer — DecoderC系の変種学習(fold0)。v54 create と同居する小スライス運転。
#   env: ARCH=base|ddil / SDFH=0|1 / EVPKL(既定=v53 evidenceC) / FOLD=0 / VFRAC=0.25 / TOKMAX=600
#        PRE=合成evidenceUで事前学習するptパス(あればロード) / SAVE=保存名
#   学習作法: epoch上限なし・early stop(pat12)・best val loss・ReduceLROnPlateau
import os, sys, json, time, gc, pickle, math
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

P0 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART = os.path.join(P0, "v60", "artifacts_v60"); os.makedirs(ART, exist_ok=True)
EVPKL = os.environ.get("EVPKL", os.path.join(P0, "v53", "artifacts_v53", "evidenceC_v53.pkl"))
ARCH = os.environ.get("ARCH", "ddil"); SDFH = os.environ.get("SDFH", "0") == "1"
FOLD = int(os.environ.get("FOLD", "0"))
VFRAC = float(os.environ.get("VFRAC", "0.25")); TOKMAX = int(os.environ.get("TOKMAX", "600")); BW = 4
CH = int(os.environ.get("CHN", "128")); LAM = 0.02; SDW = float(os.environ.get("SDW", "0.3"))
PAT = 12; LRPAT = 4; MAXMIN = float(os.environ.get("MAXMIN", "70"))
BDROP = float(os.environ.get("BDROP", "0.15")); NOISE = 0.02
ND = 181; NB = 90; STEP = 0.5
DEV = "cuda" if torch.cuda.is_available() else "cpu"
if DEV == "cuda":
    torch.cuda.set_per_process_memory_fraction(VFRAC)
SEED = int(os.environ.get("SEED", "0"))
import random
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
SAVE = os.environ.get("SAVE", f"v60_{ARCH}{'_sdf' if SDFH else ''}_f{FOLD}")

t0 = time.time()
EVD = pickle.loads(open(EVPKL, "rb").read())
WELLS = sorted(EVD.keys())
CIN = int(EVD[WELLS[0]]["EV"].shape[0])
NPF = max(CIN - 11, 0)
print(f"pkl={os.path.basename(EVPKL)} wells={len(WELLS)} CIN={CIN} ARCH={ARCH} SDFH={SDFH} ({time.time()-t0:.0f}s)", flush=True)

DILS = {"base": [(1, 1), (2, 1), (4, 1), (8, 1), (16, 1)],
        "ddil": [(1, 1), (2, 1), (4, 2), (8, 4), (16, 8)],
        "unet": [(1, 1)]}[ARCH]

class UNet2D(nn.Module):
    """(時間×δ)面の砂時計: down×3(δも時間も1/2ずつ)+skip。列softmax読み出しはDecVと共通仕様。"""
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
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)
        self.ref = nn.Linear(2 * ch, 1)
        nn.init.zeros_(self.ref.weight); nn.init.zeros_(self.ref.bias)
        self.sdf = nn.Conv2d(c1, 1, 1) if SDFH else None
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
        sdf = self.sdf(h)[:, 0] if self.sdf is not None else None
        return self.out(h)[:, 0], 2.0 * torch.tanh(self.ref(g2)[:, :, 0]), sdf

class DecV(nn.Module):
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
        self.out = nn.Conv2d(ch, 1, 1)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)
        self.ref = nn.Linear(2 * ch, 1)
        nn.init.zeros_(self.ref.weight); nn.init.zeros_(self.ref.bias)
        self.sdf = nn.Conv2d(ch, 1, 1) if SDFH else None
    def forward(self, x):
        h = self.inp(x)
        for blk, nm in zip(self.blocks, self.norms):
            h = nm(h + blk(h))
        g = h.mean(3).transpose(1, 2)
        g2, _ = self.gru(g)
        h = h + self.mix(g2).transpose(1, 2)[:, :, :, None]
        sdf = self.sdf(h)[:, 0] if self.sdf is not None else None
        return self.out(h)[:, 0], 2.0 * torch.tanh(self.ref(g2)[:, :, 0]), sdf

def make_batch(wns):
    Tm = max(EVD[w]["EV"].shape[1] for w in wns); B = len(wns)
    x = torch.zeros(B, CIN, Tm, ND); m = torch.zeros(B, Tm)
    dt = torch.full((B, Tm), float("nan")); wg = torch.zeros(B, Tm)
    for i, w in enumerate(wns):
        e = EVD[w]; T = e["EV"].shape[1]
        x[i, :, :T] = torch.from_numpy(e["EV"].astype(np.float32))
        m[i, :T] = 1.0
        dt[i, :T] = torch.from_numpy(np.asarray(e["dtrue"], np.float32))
        wg[i, :T] = torch.from_numpy(np.asarray(e["wgt"], np.float32))
    return x.to(DEV), m.to(DEV), dt.to(DEV), wg.to(DEV)

offs_t = torch.arange(-NB, NB + 1, device=DEV).float() * STEP

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
    lg, rf, sdf = net(x)
    kt = (dtr / STEP + NB)
    ok = m.bool() & torch.isfinite(dtr) & (kt >= 0) & (kt <= ND - 1) & (wg > 0)
    if not ok.any(): return None
    ktc = torch.nan_to_num(kt, nan=float(NB)).clamp(0, ND - 1)
    d = (torch.arange(ND, device=DEV)[None, None, :] - ktc[:, :, None])
    tgt = torch.exp(-0.5 * (d / 2.0) ** 2); tgt = tgt / tgt.sum(-1, keepdim=True).clamp_min(1e-12)
    lsm = F.log_softmax(lg, dim=-1)
    ce = -(tgt * lsm).sum(-1)
    P = lsm.exp(); dmean = (P * offs_t[None, None, :]).sum(-1) + rf
    se = (dmean - torch.nan_to_num(dtr)) ** 2
    wgn = wg / wg[ok].mean()
    loss = (ce[ok] * wgn[ok]).mean() + LAM * (se[ok] * wgn[ok]).mean()
    if SDFH and sdf is not None:
        dist_t = (d.abs() * STEP).clamp(0, 10.0) / 10.0
        sl = F.smooth_l1_loss(torch.sigmoid(sdf)[ok], dist_t[ok])
        loss = loss + SDW * sl
    return loss

val = WELLS[FOLD::5]; trn = [w for w in WELLS if w not in set(val)]
net = (UNet2D() if ARCH == "unet" else DecV()).to(DEV)
PRE = os.environ.get("PRE")
if PRE and os.path.exists(PRE):
    sd_ = torch.load(PRE, map_location=DEV)
    net.load_state_dict(sd_, strict=False)
    print(f"[pretrain] loaded {os.path.basename(PRE)}", flush=True)
opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4)
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
        x, m, dtr, wg = make_batch([w])
        T = EVD[w]["EV"].shape[1]
        lg, rf, _ = net(x)
        P = torch.softmax(lg[0, :T], -1)
        dh[w] = ((P @ offs_t) + rf[0, :T]).cpu().numpy()
es = []
for w in val:
    e = EVD[w]; z = e["st"] + np.asarray(dh[w], np.float64)[e["bid"]]
    er = z - e["y_ev"]; es.append(er[np.isfinite(er)])
r = float(np.sqrt(np.mean(np.concatenate(es) ** 2)))
print(f"\n[{SAVE}] fold{FOLD} RMSE = {r:.4f}  (基準 DecoderC f0=5.047)")
torch.save(net.state_dict(), os.path.join(ART, f"{SAVE}.pt"))
json.dump(dict(rmse=r, arch=ARCH, sdf=SDFH, best_val=best, epochs=ep),
          open(os.path.join(ART, f"{SAVE}.json"), "w"), indent=1)
