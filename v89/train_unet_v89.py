# v64 trainer — PF-free 2D UNet(9ch GR照合像+物理prior)。Tucker 3位再現路線。
#   env: PHASE=fit|pre / ARCH=unet|base|ddil / EVPKL / FOLD / VFRAC / TOKMAX / SEED / SAVE
#        PRE=事前学習ckptパス(fit時にロード=T1) / REAL_PKL+STMAP(pre時のdonor-foldリーク除外)
#   PHASE=pre: EVPKL=pseudo画像pklで事前学習(val=10%固定, FOLD>=0ならそのfoldのval donor由来pseudoを除外)
#   PHASE=fit: 実5-fold(v60と同一作法: early stop pat12 / best val / ReduceLROnPlateau)
import os, sys, json, time, gc, pickle, math
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

P0 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART = os.path.join(P0, "v64", "artifacts_v64"); os.makedirs(ART, exist_ok=True)
EVPKL = os.environ.get("EVPKL", os.path.join(ART, "imagesP_v64.pkl"))
ARCH = os.environ.get("ARCH", "unet"); SDFH = os.environ.get("SDFH", "0") == "1"
PHASE = os.environ.get("PHASE", "fit")   # fit=実5fold / pre=pseudo事前学習
FOLD = int(os.environ.get("FOLD", "0"))
VFRAC = float(os.environ.get("VFRAC", "0.25")); TOKMAX = int(os.environ.get("TOKMAX", "600")); BW = 4
CH = int(os.environ.get("CHN", "128")); LAM = 0.02; SDW = float(os.environ.get("SDW", "0.3"))
PAT = 12; LRPAT = 4; MAXMIN = float(os.environ.get("MAXMIN", "70"))
SHIFT_AUG = int(os.environ.get("SHIFT_AUG", "0"))   # ★train時: δ方向±Kbinランダムシフト(中心不信の学習=転移/事前学習対策)
BDROP = float(os.environ.get("BDROP", "0.15")); NOISE = 0.02
CONW = float(os.environ.get("CONW", "0.0"))   # ★v89 B: δ経路の連続性(血管修復)ペナルティ重み
NB = int(os.environ.get("NBIN", "90")); ND = 2 * NB + 1; STEP = 0.5   # v64b: NBIN=120(±60ft)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
if DEV == "cuda":
    torch.cuda.set_per_process_memory_fraction(VFRAC)
SEED = int(os.environ.get("SEED", "0"))
import random
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
SAVE = os.environ.get("SAVE", (f"v64_pre_f{FOLD}" if PHASE == "pre" else f"v89_{ARCH}{'_sdf' if SDFH else ''}_f{FOLD}"))

t0 = time.time()
EVD = pickle.loads(open(EVPKL, "rb").read())
WELLS = sorted(EVD.keys())
# ---- v64 DA混合: fit時に第2pkl(pseudo)の井戸をエポック毎に混ぜる ----
DA_PKL = os.environ.get("DA_PKL", "")
DA_RATIO = float(os.environ.get("DA_RATIO", "0.5"))
DA_WELLS = []; DA_FOLD = {}
if DA_PKL:
    _dvd = pickle.loads(open(DA_PKL, "rb").read())
    _sm = os.environ.get("STMAP", "")
    _don = {r["id"]: r["donor"] for r in json.loads(open(_sm, encoding="utf-8").read())["wells"]} if _sm else {}
    _valr = set(WELLS[FOLD::5])
    for _w, _e in _dvd.items():
        if _don.get(_w, "") in _valr: continue      # donor-foldリーク除外
        EVD["DA::" + _w] = _e; DA_WELLS.append("DA::" + _w)
    print(f"[DA] {os.path.basename(DA_PKL)}: {len(DA_WELLS)} wells (ratio={DA_RATIO})", flush=True)
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
    if train and SHIFT_AUG > 0:
        for _i in range(x.shape[0]):
            _k = int(torch.randint(-SHIFT_AUG, SHIFT_AUG + 1, (1,)).item())
            if _k == 0: continue
            _sh = torch.zeros_like(x[_i])
            if _k > 0: _sh[:, :, _k:] = x[_i][:, :, :-_k]
            else:      _sh[:, :, :_k] = x[_i][:, :, -_k:]
            x[_i] = _sh                                   # 窓外から入る領域は0(=証拠なし)
            dtr[_i] = dtr[_i] + _k * STEP                  # 真値の見かけ位置も同シフト(窓外はokマスクで自然に除外)
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
    if CONW > 0:                                  # ★血管修復: 隣接MD binのδジャンプをrobust(L1)に罰する
        dd = (dmean[:, 1:] - dmean[:, :-1]).abs()
        okc = ok[:, 1:] & ok[:, :-1]
        if okc.any():
            loss = loss + CONW * dd[okc].mean()
    if SDFH and sdf is not None:
        dist_t = (d.abs() * STEP).clamp(0, 10.0) / 10.0
        sl = F.smooth_l1_loss(torch.sigmoid(sdf)[ok], dist_t[ok])
        loss = loss + SDW * sl
    return loss

if PHASE == "pre":
    # 事前学習: pseudo画像pkl。FOLD>=0なら「実foldのval坑井をdonorとするpseudo」を除外(リーク防止)
    pool = list(WELLS)
    if FOLD >= 0:
        rpkl = os.environ.get("REAL_PKL", os.path.join(ART, "imagesP_v64.pkl"))
        smap = os.environ.get("STMAP", "")
        if smap and os.path.exists(rpkl):
            _rw = sorted(pickle.loads(open(rpkl, "rb").read()).keys())
            _valr = set(_rw[FOLD::5])
            _don = {r["id"]: r["donor"] for r in json.loads(open(smap, encoding="utf-8").read())["wells"]}
            pool = [w for w in pool if _don.get(w, "") not in _valr]
            print(f"[pre] donor-fold{FOLD}リーク除外後: {len(pool)}/{len(WELLS)} wells", flush=True)
    val = pool[::10]; trn = [w for w in pool if w not in set(val)]
else:
    val = WELLS[FOLD::5]; trn = [w for w in WELLS if w not in set(val)]
net = (UNet2D() if ARCH == "unet" else DecV()).to(DEV)
PRE = os.environ.get("PRE")
if PRE and os.path.exists(PRE):
    sd_ = torch.load(PRE, map_location=DEV)
    net.load_state_dict(sd_, strict=False)
    print(f"[pretrain] loaded {os.path.basename(PRE)}", flush=True)
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
    ep_trn = trn
    if DA_WELLS:
        _nda = min(len(DA_WELLS), int(len(trn) * DA_RATIO))
        ep_trn = trn + [DA_WELLS[j] for j in rng.choice(len(DA_WELLS), size=_nda, replace=False)]
    for wns in bucket(ep_trn, rng):
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
dh = {}; _dist = {}
with torch.no_grad():
    for w in val:
        x, m, dtr, wg = make_batch([w])
        T = EVD[w]["EV"].shape[1]
        lg, rf, _ = net(x)
        P = torch.softmax(lg[0, :T], -1)
        dh[w] = ((P @ offs_t) + rf[0, :T]).cpu().numpy()
        if os.environ.get("SAVE_DIST","0")=="1":
            _dist[w] = P.detach().cpu().numpy().astype("float16")
es = []
for w in val:
    e = EVD[w]; z = e["st"] + np.asarray(dh[w], np.float64)[e["bid"]]
    er = z - e["y_ev"]; es.append(er[np.isfinite(er)])
r = float(np.sqrt(np.mean(np.concatenate(es) ** 2)))
print(f"\n[{SAVE}] {PHASE} fold{FOLD} RMSE = {r:.4f}"
      + ("  (pseudo val=T0ゲート)" if PHASE == "pre" else "  (scratch f0基準は初回実測を docs/v64.md に記録)"))
if os.environ.get("SAVE_OOF","0")=="1":
    _oof={}
    for w in val:
        e=EVD[w]; _oof[w]=(e["st"]+np.asarray(dh[w],np.float64)[e["bid"]]).astype("float32")
    np.savez_compressed(os.path.join(ART, f"{SAVE}_oof.npz"), **_oof)
    print(f"[oof] saved {SAVE}_oof.npz ({len(_oof)} wells)")
if os.environ.get("SAVE_DIST","0")=="1":
    np.savez_compressed(os.path.join(ART, f"{SAVE}_dist.npz"), **_dist); print(f"[dist] saved {SAVE}_dist.npz")
torch.save(net.state_dict(), os.path.join(ART, f"{SAVE}.pt"))
json.dump(dict(rmse=r, arch=ARCH, sdf=SDFH, best_val=best, epochs=ep, phase=PHASE, pkl=os.path.basename(EVPKL), pre=os.environ.get("PRE", "")),
          open(os.path.join(ART, f"{SAVE}.json"), "w"), indent=1)
