# -*- coding: utf-8 -*-
"""
v83 — L5(曲線トークンcross-attention)のK16基盤再建トレーナ(train_l456_v62のフォーク, IM依存除去)
入力: v83/artifacts_v83/seqfeat_v83.pkl(帯中心=K16錨, K=20トークン)
env: FOLD=0..4 / SEED / ARCH=l5|l7 / VFRAC / MAXMIN。学習作法はv62と同一(early stop, plateau LR)。
出力: nn83/{SAVE}.pt/.json + oof_{SAVE}.pkl(井戸別 行レベルδ予測=stackingで使用)
"""
import os, sys, json, time, gc, pickle, random, io
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

P0 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART = os.path.join(P0, "v83", "artifacts_v83")
ARCH = os.environ.get("ARCH", "l5")
FOLD = int(os.environ.get("FOLD", "0"))
SEED = int(os.environ.get("SEED", "0"))
VFRAC = float(os.environ.get("VFRAC", "0.85"))
DIM = int(os.environ.get("DIM", "64")); PAT = 12; LRPAT = 4
MAXMIN = float(os.environ.get("MAXMIN", "45")); BW = 6; TOKB = 3000
DEV = "cuda" if torch.cuda.is_available() else "cpu"
if DEV == "cuda": torch.cuda.set_per_process_memory_fraction(VFRAC)
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
SAVE = os.environ.get("SAVE", f"v83_{ARCH}_f{FOLD}_s{SEED}")

t0 = time.time()
SF = pickle.loads(open(os.path.join(ART, "seqfeat_v83.pkl"), "rb").read())
WELLS = sorted(SF)
K = SF[WELLS[0]]["D"].shape[1]
print(f"wells={len(WELLS)} K={K} ARCH={ARCH} fold={FOLD} seed={SEED} ({time.time()-t0:.0f}s)", flush=True)


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
        x = torch.stack([D - med, torch.log(S)], 2)
        x = x.permute(0, 3, 2, 1).reshape(B * k, 2, T)
        hc = self.enc(x).reshape(B, k, -1, T).permute(0, 3, 1, 2)
        hc = hc + self.emb.weight[None, None]
        stats = torch.cat([C, med / 10.0, D.std(2, keepdim=True),
                           (D - med).abs().lt(2).float().mean(2, keepdim=True),
                           torch.zeros_like(med)], -1)
        q = self.qry(stats)
        att = (hc @ q[..., None])[..., 0] / (hc.shape[-1] ** 0.5)
        w = att.softmax(-1)
        out = (w * D).sum(-1)
        return out + 2.0 * torch.tanh(self.res(q)[..., 0])


class L7(nn.Module):
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
        stats = torch.cat([C, med / 10.0, D.std(2, keepdim=True),
                           (D - med).abs().lt(2).float().mean(2, keepdim=True),
                           torch.zeros_like(med)], -1)
        q = self.qry(stats)
        att = (hc @ q[..., None])[..., 0] / (hc.shape[-1] ** 0.5)
        w = att.softmax(-1)
        return (w * D).sum(-1), w


TAU_R = float(os.environ.get("TAU_R", "1.5"))


def make_batch(wns):
    Tm = max(SF[w]["D"].shape[0] for w in wns); B = len(wns)
    D = torch.zeros(B, Tm, K); S = torch.ones(B, Tm, K); C = torch.zeros(B, Tm, 3)
    yv = torch.full((B, Tm), float("nan")); m = torch.zeros(B, Tm); wg = torch.zeros(B, Tm)
    for i, w in enumerate(wns):
        f = SF[w]; T = f["D"].shape[0]
        D[i, :T] = torch.from_numpy(f["D"]); S[i, :T] = torch.from_numpy(f["S"]); C[i, :T] = torch.from_numpy(f["C"])
        yv[i, :T] = torch.from_numpy(np.asarray(f["dtrue"], np.float32))
        wg[i, :T] = torch.from_numpy(np.asarray(f["wgt"], np.float32)); m[i, :T] = 1
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
net = {"l5": L5, "l7": L7}[ARCH]().to(DEV)
opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4)
sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=LRPAT)
rng = np.random.default_rng(SEED)


def run(wns, train):
    D, S, C, yv, m, wg = make_batch(wns)
    if train:
        D = D + 0.1 * torch.randn_like(D)
    if ARCH == "l7":
        out, w = net(D, S, C, m)
    else:
        out = net(D, S, C, m); w = None
    ok = m.bool() & torch.isfinite(yv) & (wg > 0)
    if not ok.any(): return None
    wgn = wg / wg[ok].mean()
    mse = ((out - torch.nan_to_num(yv)) ** 2 * wgn)[ok].mean()
    if ARCH == "l7":
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
    else:
        bad += 1
    print(f"ep{ep} train={np.mean(tl):.3f} val={vl:.3f}{star} (bad={bad}, {time.time()-t1:.0f}s)", flush=True)
    if bad >= PAT or (time.time() - t1) / 60 > MAXMIN:
        print(f"stop @ep{ep} best={best:.3f}"); break

net.load_state_dict(best_state); net.eval()
os.makedirs(os.path.join(ART, "nn83"), exist_ok=True)
dh = {}; es = []
with torch.no_grad():
    for w in val:
        D, S, C, yv, m, wg = make_batch([w])
        T = SF[w]["D"].shape[0]
        o_ = net(D, S, C, m)
        if ARCH == "l7": o_ = o_[0]
        db = o_[0, :T].cpu().numpy()
        f = SF[w]
        dr = db[f["bid"]]                                    # 行レベルδ予測
        dh[w] = dr.astype(np.float32)
        er = dr - f["dtrue_rows"].astype(np.float64)
        es.append(er[np.isfinite(er)])
r = float(np.sqrt(np.mean(np.concatenate(es) ** 2)))
print(f"\n[{SAVE}] fold{FOLD} 行RMSE = {r:.4f}  (v62 L5参考: 単seed fold横断5.93)")
torch.save(net.state_dict(), os.path.join(ART, "nn83", f"{SAVE}.pt"))
pickle.dump(dh, open(os.path.join(ART, "nn83", f"oof_{SAVE}.pkl"), "wb"), protocol=4)
json.dump(dict(rmse=r, arch=ARCH, seed=SEED, fold=FOLD, epochs=ep, best_val=best),
          open(os.path.join(ART, "nn83", f"{SAVE}.json"), "w"), indent=1)
