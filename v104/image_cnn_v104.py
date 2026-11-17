# -*- coding: utf-8 -*-
"""v104 画像CNN脚: seqfeat の候補δ予測(D,S,DX)を「位置T × δビンB」のコンセンサス密度画像にして 2D CNN で δ を回帰。
L5(候補attention)/TCN(1D系列)と別モダリティ=脱相関脚を狙う。det GroupKFold, mdn無しの素回帰(soft-argmax)。
env: SMOKE(1=10井1fold5ep) / NSEED(既定1) / EPOCHS(既定40) / CH(既定48) / BINS(既定48) / R(既定36)
実行: .venv/Scripts/python.exe v104/image_cnn_v104.py"""
import os, sys, io, time, pickle, math
try:
    sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
except Exception:
    pass
from pathlib import Path
import numpy as np, torch, torch.nn as nn
from sklearn.model_selection import GroupKFold

V104 = Path(__file__).resolve().parent
ART = V104 / "artifacts_v104"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SMOKE = os.environ.get("SMOKE", "0") == "1"
NSEED = int(os.environ.get("NSEED", "1"))
EPOCHS = int(os.environ.get("EPOCHS", "6" if SMOKE else "40"))
CH = int(os.environ.get("CH", "48")); B = int(os.environ.get("BINS", "48")); R = float(os.environ.get("R", "36"))
SIGMA = 1.6
bc = np.linspace(-R + R / B, R - R / B, B).astype(np.float32)   # δビン中心
bc_t = torch.tensor(bc, device=DEV)

print(f"[imgcnn] load seqfeat ... B={B} R={R} CH={CH} EPOCHS={EPOCHS} SMOKE={SMOKE}", flush=True)
SF = pickle.loads((ART / "seqfeat_v104_cf.pkl").read_bytes())
wells = sorted(SF.keys())
if SMOKE:
    wells = wells[:10]
y_all = np.load(ART / "l5_v104baseg-1_oof.npz")["y"]
Nrows = len(y_all)


def build_img(f):
    D = f["D"].astype(np.float64); S = f["S"].astype(np.float64); DX = f["DX"]
    T = D.shape[0]
    med = np.median(D, 1, keepdims=True)                       # (T,1) 位置ごとの中央候補
    o = D - med                                                 # (T,K) 中央からのオフセット
    w1 = 1.0 / (S + 0.5)
    ll = DX[:, :, 0].astype(np.float64)                         # loglik(z正規化済)
    w2 = np.exp(ll - ll.max(1, keepdims=True))
    diff = o[:, :, None] - bc[None, None, :]                    # (T,K,B)
    g = np.exp(-0.5 * (diff / SIGMA) ** 2)
    d1 = (w1[:, :, None] * g).sum(1); d2 = (w2[:, :, None] * g).sum(1)   # (T,B)
    d1 /= d1.sum(1, keepdims=True) + 1e-9; d2 /= d2.sum(1, keepdims=True) + 1e-9
    img = np.stack([d1, d2], 0).astype(np.float32)             # (2,T,B)
    return img, med[:, 0].astype(np.float32)


# 画像 + メタを事前構築
IMG = {}; MED = {}; DT = {}; WG = {}; ROWS = {}; BID = {}
g_group = []
t0 = time.time()
for w in wells:
    f = SF[w]
    img, med = build_img(f)
    IMG[w] = img; MED[w] = med
    DT[w] = f["dtrue"].astype(np.float32); WG[w] = f["wgt"].astype(np.float32)
    ROWS[w] = f["rows"]; BID[w] = f["bid"]
    g_group.append(w)
print(f"[imgcnn] 画像構築 {len(wells)}井 ({time.time()-t0:.0f}s)", flush=True)


class ImgCNN(nn.Module):
    def __init__(s, ch=CH, nb=4):
        super().__init__()
        s.inp = nn.Conv2d(2, ch, 3, padding=1)
        s.c = nn.ModuleList([nn.Conv2d(ch, ch, 3, padding=1) for _ in range(nb)])
        s.n = nn.ModuleList([nn.GroupNorm(8, ch) for _ in range(nb)])
        s.out = nn.Conv2d(ch, 1, 3, padding=1)
    def forward(s, x):                                          # x:(Bt,2,T,B)
        h = torch.relu(s.inp(x))
        for c, n in zip(s.c, s.n):
            h = torch.relu(n(c(h)) + h)
        logit = s.out(h)[:, 0]                                  # (Bt,T,B)
        p = torch.softmax(logit, -1)
        off = (p * bc_t).sum(-1)                                # (Bt,T) δオフセット(soft-argmax)
        return off


def batches(wl, bs=16, shuffle=True):
    wl = list(wl)
    if shuffle:
        np.random.shuffle(wl)
    wl = sorted(wl, key=lambda w: IMG[w].shape[1])              # 長さ順=pad最小
    for i in range(0, len(wl), bs):
        yield wl[i:i + bs]


def collate(ws):
    Tm = max(IMG[w].shape[1] for w in ws); n = len(ws)
    X = np.zeros((n, 2, Tm, B), np.float32); med = np.zeros((n, Tm), np.float32)
    dt = np.zeros((n, Tm), np.float32); wg = np.zeros((n, Tm), np.float32)
    for i, w in enumerate(ws):
        T = IMG[w].shape[1]
        X[i, :, :T] = IMG[w]; med[i, :T] = MED[w]; dt[i, :T] = DT[w]; wg[i, :T] = WG[w]
    return (torch.from_numpy(X).to(DEV), torch.from_numpy(med).to(DEV),
            torch.from_numpy(dt).to(DEV), torch.from_numpy(wg).to(DEV), ws)


def run():
    gkf = GroupKFold(n_splits=2 if SMOKE else 5)
    groups = np.array(wells); folds = list(gkf.split(wells, groups=groups))
    oof_seed = []
    for seed in range(NSEED):
        torch.manual_seed(seed); np.random.seed(seed)
        pred = np.full(Nrows, np.nan)
        for k, (tri, vai) in enumerate(folds):
            trw = [wells[i] for i in tri]; vaw = [wells[i] for i in vai]
            net = ImgCNN().to(DEV); opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
            best = (1e9, None)
            for ep in range(EPOCHS):
                net.train()
                for ws in batches(trw):
                    X, med, dt, wg, _ = collate(ws)
                    off = net(X); dp = med + off
                    loss = ((dp - dt) ** 2 * wg).sum() / wg.sum().clamp(min=1)
                    opt.zero_grad(); loss.backward(); opt.step()
                # val
                net.eval(); se = 0.0; sw = 0.0
                with torch.no_grad():
                    for ws in batches(vaw, shuffle=False):
                        X, med, dt, wg, _ = collate(ws)
                        dp = med + net(X); se += ((dp - dt) ** 2 * wg).sum().item(); sw += wg.sum().item()
                vr = (se / sw) ** 0.5
                if vr < best[0]:
                    best = (vr, {kk: vv.detach().clone() for kk, vv in net.state_dict().items()})
                if ep < 2 or ep == EPOCHS - 1:
                    print(f"  seed{seed} fold{k} ep{ep} val={vr:.4f}", flush=True)
            net.load_state_dict(best[1]); net.eval()
            with torch.no_grad():
                for ws in batches(vaw, shuffle=False):
                    X, med, dt, wg, wl = collate(ws)
                    dp = (med + net(X)).cpu().numpy()
                    for i, w in enumerate(wl):
                        T = IMG[w].shape[1]
                        pv = dp[i, :T]                            # bin予測 → eval行へ
                        pred[ROWS[w]] = pv[BID[w]]
            print(f"  seed{seed} fold{k} best_val={best[0]:.4f}", flush=True)
        m = np.isfinite(pred) & np.isfinite(y_all)
        rmse = float(np.sqrt(np.mean((pred[m] - y_all[m]) ** 2)))
        print(f"[imgcnn] seed{seed} OOF = {rmse:.4f}", flush=True)
        oof_seed.append(pred)
    ens = np.nanmean(np.stack(oof_seed, 0), 0)
    m = np.isfinite(ens) & np.isfinite(y_all)
    print(f"[imgcnn] {NSEED}seed OOF = {np.sqrt(np.mean((ens[m]-y_all[m])**2)):.4f}  (L5 5.5549 / TCN 6.01)", flush=True)
    np.savez(ART / "imgcnn_oof.npz", y=y_all, **{f"s{s}": oof_seed[s] for s in range(NSEED)})
    print("[imgcnn] imgcnn_oof.npz 保存", flush=True)


if __name__ == "__main__":
    run()
