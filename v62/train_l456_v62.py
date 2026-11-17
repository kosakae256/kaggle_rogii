# v62 L4/L5/L6 トレーナ — v52特徴の「別表現」3種(docs/v61,v62系スタッキング計画)
#   L4 selfsim: 自己相似バイアス付きTransformer(再訪整合: |med_i−med_j| が小さいbin同士を結ぶ)
#   L5 curveattn: エンジン16本を曲線トークン化し、bin別cross-attentionで凸結合+微小残差
#   L6 spectral: 多重スケール(4/16/64bin)分解特徴の dilated Conv1d
#   共通: 入力= seqfeat_v62.pkl(D=δ16本,S=幅,C=文脈)、メタ= imagesF_v62.pkl
#   env: ARCH=l4|l5|l6 / FOLD / SEED / VFRAC / SAVE。学習作法=early stop, best val loss, plateau LR。
import os, sys, json, time, gc, pickle, random
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

P0 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART = os.path.join(P0, "v62", "artifacts_v62")
ARCH = os.environ.get("ARCH", "l4")
FOLD = int(os.environ.get("FOLD", "0"))
SEED = int(os.environ.get("SEED", "0"))
VFRAC = float(os.environ.get("VFRAC", "0.5"))
DIM = int(os.environ.get("DIM", "64")); PAT = 12; LRPAT = 4
MAXMIN = float(os.environ.get("MAXMIN", "45")); BW = 6; TOKB = 3000
TAU_SIM = float(os.environ.get("TAU_SIM", "3.0"))   # L4: 類似バイアスの距離スケール(ft)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
if DEV == "cuda": torch.cuda.set_per_process_memory_fraction(VFRAC)
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
SAVE = os.environ.get("SAVE", f"v62_{ARCH}_f{FOLD}")

t0 = time.time()
SF = pickle.loads(open(os.path.join(ART, "seqfeat_v62.pkl"), "rb").read())
IM = pickle.loads(open(os.path.join(ART, "imagesF_v62.pkl"), "rb").read())
WELLS = sorted(set(SF) & set(IM))
K = SF[WELLS[0]]["D"].shape[1]
print(f"wells={len(WELLS)} K={K} ARCH={ARCH} ({time.time()-t0:.0f}s)", flush=True)

class L4(nn.Module):
    def __init__(self, k=K, d=DIM, nl=3, nh=4):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(2 * k + 3 + 1, d), nn.GELU(), nn.Linear(d, d))
        self.layers = nn.ModuleList([nn.ModuleDict(dict(
            q=nn.Linear(d, d), kk=nn.Linear(d, d), v=nn.Linear(d, d), o=nn.Linear(d, d),
            n1=nn.LayerNorm(d), n2=nn.LayerNorm(d),
            ff=nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d)))) for _ in range(nl)])
        self.nh = nh; self.d = d
        self.head = nn.Linear(d, 1)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)
    def forward(self, D, S, C, m):
        med = D.median(2, keepdim=True).values                     # (B,T,1)
        x = torch.cat([D - med, torch.log(S), C, med / 10.0], -1)
        h = self.inp(x)
        B, T, d = h.shape
        # 自己相似バイアス: −|med_i − med_j|/τ − |i−j|/64(近すぎ自明対の抑制なし・素直に)
        dm = (med[:, :, 0][:, :, None] - med[:, :, 0][:, None, :]).abs()
        ti = torch.arange(T, device=D.device).float()
        dt_ = (ti[None, :, None] - ti[None, None, :]).abs()
        bias = -dm / TAU_SIM - dt_ / 96.0
        bias = bias.masked_fill(~(m[:, None, :].bool()), -1e9)
        for L in self.layers:
            hn = L["n1"](h)
            q = L["q"](hn).view(B, T, self.nh, -1).transpose(1, 2)
            kk = L["kk"](hn).view(B, T, self.nh, -1).transpose(1, 2)
            v = L["v"](hn).view(B, T, self.nh, -1).transpose(1, 2)
            att = (q @ kk.transpose(-1, -2)) / (q.shape[-1] ** 0.5) + bias[:, None]
            h = h + L["o"]((att.softmax(-1) @ v).transpose(1, 2).reshape(B, T, d))
            h = h + L["ff"](L["n2"](h))
        return self.head(h)[..., 0] + med[..., 0]                  # 残差=medからのずれ

class L5(nn.Module):
    def __init__(self, k=K, d=DIM):
        super().__init__()
        self.k = k
        self.enc = nn.Sequential(nn.Conv1d(2, d, 9, padding=4), nn.GELU(),
                                 nn.Conv1d(d, d, 9, padding=4 * 4, dilation=4), nn.GELU(),
                                 nn.Conv1d(d, d, 9, padding=4 * 16, dilation=16))
        self.emb = nn.Embedding(k, d)
        self.qry = nn.Sequential(nn.Linear(3 + 4, d), nn.GELU(), nn.Linear(d, d))
        self.res = nn.Linear(d, 1)
        nn.init.zeros_(self.res.weight); nn.init.zeros_(self.res.bias)
    def forward(self, D, S, C, m):
        B, T, k = D.shape
        med = D.median(2, keepdim=True).values
        x = torch.stack([D - med, torch.log(S)], 2)                # (B,T,2,K)
        x = x.permute(0, 3, 2, 1).reshape(B * k, 2, T)
        hc = self.enc(x).reshape(B, k, -1, T).permute(0, 3, 1, 2)  # (B,T,K,d)
        hc = hc + self.emb.weight[None, None]
        stats = torch.cat([C, med / 10.0, D.std(2, keepdim=True), (D - med).abs().lt(2).float().mean(2, keepdim=True),
                           torch.zeros_like(med)], -1)
        q = self.qry(stats)                                        # (B,T,d)
        att = (hc @ q[..., None])[..., 0] / (hc.shape[-1] ** 0.5)  # (B,T,K)
        w = att.softmax(-1)
        out = (w * D).sum(-1)
        return out + 2.0 * torch.tanh(self.res(q)[..., 0])

class L6(nn.Module):
    def __init__(self, k=K, d=DIM):
        super().__init__()
        cin = (2 * k + 3) * 3
        self.net = nn.Sequential(nn.Conv1d(cin, d, 5, padding=2), nn.GELU(),
                                 nn.Conv1d(d, d, 5, padding=2 * 4, dilation=4), nn.GELU(),
                                 nn.Conv1d(d, d, 5, padding=2 * 16, dilation=16), nn.GELU(),
                                 nn.Conv1d(d, d, 5, padding=2 * 64, dilation=64))
        self.head = nn.Linear(d, 1)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)
    def forward(self, D, S, C, m):
        med = D.median(2, keepdim=True).values
        base = torch.cat([D - med, torch.log(S), C], -1).transpose(1, 2)   # (B,c,T)
        outs = [base]
        for w_ in (4, 16):                                        # 多重スケール(detrendピラミッド)
            sm = F.avg_pool1d(F.pad(base, (w_ // 2, w_ - w_ // 2 - 1), mode="replicate"), w_, stride=1)
            outs.append(base - sm)
        h = self.net(torch.cat(outs, 1)).transpose(1, 2)
        return self.head(h)[..., 0] + med[..., 0]

class L7(nn.Module):
    """純選択(ranking)版: L5と同骨格だが凸結合のみ(残差なし)。forwardは(out, w)を返す。"""
    def __init__(self, k=K, d=DIM):
        super().__init__()
        self.k = k
        self.enc = nn.Sequential(nn.Conv1d(2, d, 9, padding=4), nn.GELU(),
                                 nn.Conv1d(d, d, 9, padding=4 * 4, dilation=4), nn.GELU(),
                                 nn.Conv1d(d, d, 9, padding=4 * 16, dilation=16))
        self.emb = nn.Embedding(k, d)
        self.qry = nn.Sequential(nn.Linear(3 + 4, d), nn.GELU(), nn.Linear(d, d))
    def forward(self, D, S, C, m):
        B, T, k = D.shape
        med = D.median(2, keepdim=True).values
        x = torch.stack([D - med, torch.log(S)], 2)
        x = x.permute(0, 3, 2, 1).reshape(B * k, 2, T)
        hc = self.enc(x).reshape(B, k, -1, T).permute(0, 3, 1, 2)
        hc = hc + self.emb.weight[None, None]
        stats = torch.cat([C, med / 10.0, D.std(2, keepdim=True), (D - med).abs().lt(2).float().mean(2, keepdim=True),
                           torch.zeros_like(med)], -1)
        q = self.qry(stats)
        att = (hc @ q[..., None])[..., 0] / (hc.shape[-1] ** 0.5)
        w = att.softmax(-1)
        return (w * D).sum(-1), w

TAU_R = float(os.environ.get("TAU_R", "1.5"))   # listwiseターゲット温度(ft)

def make_batch(wns):
    Tm = max(SF[w]["D"].shape[0] for w in wns); B = len(wns)
    D = torch.zeros(B, Tm, K); S = torch.ones(B, Tm, K); C = torch.zeros(B, Tm, 3)
    yv = torch.full((B, Tm), float("nan")); m = torch.zeros(B, Tm); wg = torch.zeros(B, Tm)
    for i, w in enumerate(wns):
        f = SF[w]; e = IM[w]; T = f["D"].shape[0]
        D[i, :T] = torch.from_numpy(f["D"]); S[i, :T] = torch.from_numpy(f["S"]); C[i, :T] = torch.from_numpy(f["C"])
        yv[i, :T] = torch.from_numpy(np.asarray(e["dtrue"], np.float32))
        wg[i, :T] = torch.from_numpy(np.asarray(e["wgt"], np.float32)); m[i, :T] = 1
    return D.to(DEV), S.to(DEV), C.to(DEV), yv.to(DEV), m.to(DEV), wg.to(DEV)

def bucket(ws, rng):
    lens = {w: SF[w]["D"].shape[0] for w in ws}
    order = rng.permutation(len(ws)); ws_ = [ws[j] for j in order]
    wl = [w for i in range(0, len(ws_), 64) for w in sorted(ws_[i:i + 64], key=lambda x: lens[x])]
    out = []; cur = []
    for w in wl:
        t = cur + [w]
        if cur and (len(t) * max(lens[x] for x in t) > TOKB or len(t) > BW): out.append(cur); cur = [w]
        else: cur = t
    if cur: out.append(cur)
    return out

val = WELLS[FOLD::5]; trn = [w for w in WELLS if w not in set(val)]
net = {"l4": L4, "l5": L5, "l6": L6, "l7": L7}[ARCH]().to(DEV)
opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4)
sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=LRPAT)
rng = np.random.default_rng(SEED)

def run(wns, train):
    D, S, C, yv, m, wg = make_batch(wns)
    if train:
        D = D + 0.1 * torch.randn_like(D)                          # 軽い座標ノイズ
    if ARCH == "l7":
        out, w = net(D, S, C, m)
    else:
        out = net(D, S, C, m); w = None
    ok = m.bool() & torch.isfinite(yv) & (wg > 0)
    if not ok.any(): return None
    wgn = wg / wg[ok].mean()
    mse = ((out - torch.nan_to_num(yv)) ** 2 * wgn)[ok].mean()
    if ARCH == "l7":
        # listwise ranking: 正解に近い候補ほど高確率、のターゲット分布とのCE
        dist = (D - torch.nan_to_num(yv)[..., None]).abs()
        qt = torch.softmax(-dist / TAU_R, -1)
        ce = -(qt * torch.log(w.clamp_min(1e-9))).sum(-1)
        return (ce * wgn)[ok].mean() + 0.1 * mse
    return mse

def val_loss():
    net.eval(); tot = 0.0; n = 0
    with torch.no_grad():
        for wns in bucket(val, np.random.default_rng(1)):
            l = run(wns, False)
            if l is None: continue
            tot += float(l) * len(wns); n += len(wns)
    return tot / max(n, 1)

best = float("inf"); best_state = None; bad = 0; ep = 0; t1 = time.time()
while True:
    ep += 1; net.train(); tl = []
    for wns in bucket(trn, rng):
        try:
            l = run(wns, True)
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
    print(f"ep{ep} train={np.mean(tl):.3f} val={vl:.3f}{star} (bad={bad}, {time.time()-t1:.0f}s)", flush=True)
    if bad >= PAT or (time.time() - t1) / 60 > MAXMIN:
        print(f"stop @ep{ep} best={best:.3f}"); break
net.load_state_dict(best_state); net.eval()
dh = {}
with torch.no_grad():
    for w in val:
        D, S, C, yv, m, wg = make_batch([w])
        T = SF[w]["D"].shape[0]
        o_ = net(D, S, C, m)
        if ARCH == "l7": o_ = o_[0]
        dh[w] = o_[0, :T].cpu().numpy()
es = []
for w in val:
    e = IM[w]; z = e["st"] + np.asarray(dh[w], np.float64)[e["bid"]]
    er = z - e["y_ev"]; es.append(er[np.isfinite(er)])
r = float(np.sqrt(np.mean(np.concatenate(es) ** 2)))
print(f"\n[{SAVE}] fold{FOLD} RMSE = {r:.4f}  (v52基準 f{FOLD}: blend {[5.0558,5.7178,6.1925,6.8998,6.4186][FOLD]})")
os.makedirs(os.path.join(ART, "nn62"), exist_ok=True)
torch.save(net.state_dict(), os.path.join(ART, "nn62", f"{SAVE}.pt"))
json.dump(dict(rmse=r, arch=ARCH, seed=SEED, epochs=ep, best_val=best),
          open(os.path.join(ART, "nn62", f"{SAVE}.json"), "w"), indent=1)
