# -*- coding: utf-8 -*-
"""
v50 Stage B — 学習デコーダ(固定FBの全置換): 証拠ボリューム → 2D CNN + 時間bi-GRU → δ分布
================================================================================================
Stage A の結論「表現は伸びるのに固定FB+固定較正で decode が飽和(8.15)」を受け、デコーダを学習化。
  入力(時間bin × δ181bin): [学習類似度(OOFエンコーダ), 生尤度, struct prior^0.3, δ座標, σ]
  出力: 各時間binの δ softmax 分布 → posterior mean → z = struct + δ̂
  損失: soft CE(真δビン, σ=2bin)+ λ·bin重み付きMSE(posterior mean)
リーク対策: 類似度は各坑井の OOF エンコーダ(その坑井が val の fold の capaug)で生成。
デコーダ自体も GroupKFold 5fold(Stage A と同一分割)で学習=全773坑井の正直評価。

実行: python v50/stageB_v50.py            (evidence cache 無ければ自動生成)
env : BEPOCHS=40 BCH=48 FOLDS=0,1,2,3,4 ...
"""
import os, sys, json, time, math, pickle
os.environ.setdefault("TAG", "capaug"); os.environ.setdefault("CH", "128")
os.environ.setdefault("DIM", "96"); os.environ.setdefault("DILS", "1,2,4,8,16,32")
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import stageA_v50 as sa

P0 = sa.P0; ART = sa.ART; DEV = sa.DEV
if DEV == "cuda":
    torch.cuda.set_per_process_memory_fraction(float(os.environ.get("VFRAC", "0.55")))   # ★12GB機で~6.7GB上限
ND = 2 * sa.NB + 1                                   # 181
BEPOCHS = int(os.environ.get("BEPOCHS", "40"))
BCH     = int(os.environ.get("BCH", "48"))           # デコーダCNN幅
BLR     = float(os.environ.get("BLR", "3e-4"))
LAM     = float(os.environ.get("LAM", "0.02"))       # posterior-mean MSE 重み(ft²スケール)
FOLDS   = [int(x) for x in os.environ.get("FOLDS", "0,1,2,3,4").split(",")]
BSEED   = int(os.environ.get("BSEED", "0"))
BTAG    = os.environ.get("BTAG", "")            # デコーダ実験タグ
TOKMAX  = int(os.environ.get("TOKMAX", "2400"))  # バッチ上限 = 井戸数×最大時間長(VRAM制御)
BFLIP   = os.environ.get("BFLIP", "0") == "1"    # ①時間反転aug+TTA
REFH    = os.environ.get("REFH", "0") == "1"     # ②回帰精緻化ヘッド(±2ft補正)
NBK_E   = int(os.environ.get("NBRK", "8"))       # ③nbr近傍数(8=従来)
NBRW    = os.environ.get("NBRW", "0") == "1"     # ③nbr距離重み付きプール
DILT    = tuple(int(x) for x in os.environ.get("DILT", "1,2,4,8").split(","))  # ④デコーダ時間dilation
EV_FP   = ART / (f"evidenceB3_v50" + (f"_s{sa.STRIDE}" if sa.STRIDE != 16 else "") + (f"_nb{NBK_E}{int(NBRW)}" if (NBK_E != 8 or NBRW) else "") + ".pkl")   # v3: self+nbr証拠10ch
if os.environ.get("EVFP"): EV_FP = Path(os.environ["EVFP"])          # ★証拠pklの明示指定(v51 擬似nbr等)

# ============ 1) 証拠ボリューム cache(OOFエンコーダで生成) ============
def build_evidence():
    data = pickle.loads(sa.CACHE.read_bytes())
    wells = sorted(data.keys())
    offs = np.arange(-sa.NB, sa.NB + 1)
    # ★nbr: 全坑井の既知区間(TVT_input行の y,GR)と平均XYを前collect
    KN = {}
    XY = np.array([data[w]["xy"] for w in wells])
    for w in wells:
        c = data[w]; knm = (~c["ev"]) & np.isfinite(c["y"])
        KN[w] = (c["y"][knm].astype(np.float64), c["graw"][knm].astype(np.float64))
    NBK = NBK_E
    out = {}; t0 = time.time()
    offs_t = torch.tensor(offs * sa.STEP, dtype=torch.float32, device=DEV)
    NDl = len(offs)
    ddm = offs_t[:, None] - offs_t[None, :]
    Mfb = torch.exp(-0.5 * (ddm / 0.25) ** 2); Mfb = Mfb / Mfb.sum(0, keepdim=True).clamp_min(1e-30)
    def fb_gamma(obs_np, nb_):
        obs = torch.tensor(obs_np, dtype=torch.float32, device=DEV)
        obs = obs / obs.sum(1, keepdim=True).clamp_min(1e-30)
        aF = torch.zeros(nb_, NDl, device=DEV); aF[0] = obs[0]
        for i in range(1, nb_):
            pr = Mfb @ aF[i - 1]; aF[i] = (pr.clamp_min(1e-30) * obs[i]); aF[i] = aF[i] / aF[i].sum().clamp_min(1e-30)
        bB = torch.zeros(nb_, NDl, device=DEV); bB[-1] = 1.0 / NDl
        for i in range(nb_ - 2, -1, -1):
            ms = bB[i + 1] * obs[i + 1]; bB[i] = (Mfb @ ms).clamp_min(1e-30); bB[i] = bB[i] / bB[i].sum().clamp_min(1e-30)
        gam = aF * bB; gam = gam / gam.sum(1, keepdim=True).clamp_min(1e-30)
        return gam.cpu().numpy()
    with torch.no_grad():
        for f in range(5):
            ck = torch.load(ART / f"stageA_enccapaug_f{f}.pt", map_location=DEV)
            encA = sa.Enc1D(8, sa.CH_ENC, sa.DIM, sa.DILS).to(DEV); encA.load_state_dict(ck["encA"]); encA.eval()
            encB = sa.Enc1D(7, sa.CH_ENC, sa.DIM, sa.DILS).to(DEV); encB.load_state_dict(ck["encB"]); encB.eval()
            for wn in wells[f::5]:                    # ★各坑井は自分が val の fold の enc = OOF
                c = data[wn]
                ev_idx = np.where(c["ev"])[0]; y_ev = c["y"][ev_idx].astype(np.float64)
                a = encA(torch.from_numpy(c["Fh"]).to(DEV))[ev_idx]
                b = encB(torch.from_numpy(c["Ft"]).to(DEV))
                K = len(c["gg"]); gmin = c["gmin"]
                st = c["st"].astype(np.float64); sg = np.clip(c["sg"].astype(np.float64), 1, 60)
                ks = np.round((st - gmin) / sa.STEP).astype(np.int64)
                cand = np.clip(ks[:, None] + offs[None, :], 0, K - 1)
                okc = (ks[:, None] + offs[None, :] >= 0) & (ks[:, None] + offs[None, :] <= K - 1)
                sims = (a[:, None, :] * b[torch.tensor(cand, device=DEV)]).sum(-1).cpu().numpy()
                sims = np.where(okc, sims, -1.0)
                cost = np.abs(c["g25"][:, None] - c["gg"][cand]); cost[~okc] = 60.0
                T = len(ev_idx); nb_ = (T + sa.STRIDE - 1) // sa.STRIDE; bid = np.arange(T) // sa.STRIDE
                def bm(x):
                    o = np.zeros((nb_,) + x.shape[1:]); cnt = np.zeros(nb_)
                    np.add.at(o, bid, x); np.add.at(cnt, bid, 1)
                    return o / np.maximum(cnt, 1)[(...,) + (None,) * (x.ndim - 1)]
                sgb = np.clip(bm(sg), 1, 60); stb = bm(st)
                Pr = np.exp(-0.5 * ((offs * sa.STEP)[None, :] / sgb[:, None]) ** 2)
                L_raw = np.exp(-0.5 * (bm(cost) / sa.SIGG) ** 2)
                # ★nbr証拠: XY近傍 k=8 坑井の既知区間プール → 擬似typewell格子 → GR整合
                wi_ = wells.index(wn)
                dxy = np.sqrt(((XY - XY[wi_]) ** 2).sum(1)); dxy[wi_] = 1e18
                nbrs = np.argsort(dxy)[:NBK]
                ny = np.concatenate([KN[wells[j]][0] for j in nbrs]) if len(nbrs) else np.array([])
                ng = np.concatenate([KN[wells[j]][1] for j in nbrs]) if len(nbrs) else np.array([])
                nw_ = (np.concatenate([np.full(len(KN[wells[j]][0]), 1.0 / (1.0 + dxy[j] / 500.0)) for j in nbrs])
                       if (NBRW and len(nbrs)) else np.ones(len(ny)))
                nbrg = np.full(K, np.nan)
                if len(ny) >= 20:
                    kyn = np.round((ny - gmin) / sa.STEP).astype(np.int64)
                    okn = (kyn >= 0) & (kyn <= K - 1)
                    if okn.sum() >= 20:
                        nsum = np.zeros(K); ncnt = np.zeros(K)
                        np.add.at(nsum, kyn[okn], ng[okn] * nw_[okn]); np.add.at(ncnt, kyn[okn], nw_[okn])
                        with np.errstate(invalid="ignore"):
                            nbrg = np.where(ncnt > 0, nsum / np.maximum(ncnt, 1), np.nan)
                nc = np.abs(c["g25"][:, None] - nbrg[cand])
                has_nbr = np.isfinite(nc)
                nc = np.where(has_nbr, nc, 60.0)
                L_nbr = np.exp(-0.5 * (bm(nc) / sa.SIGG) ** 2)
                H_nbr = bm(has_nbr.astype(np.float64))
                # ★self証拠: 既知区間(TVT_input)の (y,GR) を 0.5ft グリッド化 → 各候補位置のGR整合
                knm = (~c["ev"]) & np.isfinite(c["y"])
                selfg = np.full(K, np.nan)
                if knm.sum() >= 10:
                    ky = np.round((c["y"][knm].astype(np.float64) - gmin) / sa.STEP).astype(np.int64)
                    kg = c["graw"][knm].astype(np.float64)
                    okk = (ky >= 0) & (ky <= K - 1)
                    if okk.sum() >= 10:
                        ssum = np.zeros(K); scnt = np.zeros(K)
                        np.add.at(ssum, ky[okk], kg[okk]); np.add.at(scnt, ky[okk], 1)
                        with np.errstate(invalid="ignore"):
                            selfg = np.where(scnt > 0, ssum / np.maximum(scnt, 1), np.nan)
                sc = np.abs(c["g25"][:, None] - selfg[cand])              # (T,181) NaN=self情報なし
                has_self = np.isfinite(sc)
                sc = np.where(has_self, sc, 60.0)
                L_self = np.exp(-0.5 * (bm(sc) / sa.SIGG) ** 2)
                H_self = bm(has_self.astype(np.float64))
                simb = bm(sims)
                ls_ck = math.exp(ck["logit_s"])
                Prow = torch.softmax(torch.tensor(np.where(okc, sims, -1e4), dtype=torch.float32) * ls_ck * 0.25, dim=1).numpy()
                Pb = bm(Prow)
                gam = fb_gamma(L_raw * np.maximum(Pb, 1e-12) ** 0.5 * Pr ** sa.LPR, nb_)   # ★固定FB(hybrid)事後
                EV = np.stack([simb,                                       # 学習類似度(OOF)
                               L_raw,                                      # 生尤度
                               Pr ** sa.LPR,                               # struct prior
                               np.tile(np.tanh(offs * sa.STEP / 20.0), (nb_, 1)),   # δ座標
                               np.tile((sgb / 20.0)[:, None], (1, ND)),    # prior信頼度
                               (gam * 30.0).astype(np.float64),            # ★FB(hybrid)事後γ(×30でfp16域へ)
                               L_self,                                     # ★self証拠(既知区間GR整合)
                               H_self,                                     # self情報の有無
                               L_nbr,                                      # ★nbr証拠(近傍既知区間GR整合)
                               H_nbr,                                      # nbr情報の有無
                               ], 0).astype(np.float16)                    # (10, nb, 181)
                dtrue = bm(np.where(np.isfinite(y_ev), y_ev, np.nan)) - stb
                wgt = np.bincount(bid, weights=np.isfinite(y_ev).astype(float), minlength=nb_)
                out[wn] = dict(EV=EV, dtrue=dtrue.astype(np.float32), wgt=wgt.astype(np.float32),
                               st=st.astype(np.float32), bid=bid.astype(np.int32),
                               y_ev=y_ev.astype(np.float32))
            print(f"  evidence fold{f} ({time.time()-t0:.0f}s)", flush=True)
    EV_FP.write_bytes(pickle.dumps(out, protocol=4))
    print(f"evidence saved: {len(out)} wells -> {EV_FP.name}")
    return out

EVD = pickle.loads(EV_FP.read_bytes()) if EV_FP.exists() else build_evidence()
WELLS = sorted(EVD.keys())
print(f"evidence: {len(WELLS)} wells")

# ============ 2) デコーダ ============
class DecoderB(nn.Module):
    """(B,5,T,181) → (B,T,181) logits。2D CNN(時間dilated)+ 時間GRU 大域文脈。"""
    def __init__(self, cin=10, ch=BCH):
        super().__init__()
        self.inp = nn.Conv2d(cin, ch, (5, 5), padding=(2, 2))
        self.blocks = nn.ModuleList(); self.norms = nn.ModuleList()
        for dt in DILT:                               # 時間方向 dilated(δ方向は等方k5)
            self.blocks.append(nn.Sequential(
                nn.Conv2d(ch, ch, (5, 5), padding=(2 * dt, 2), dilation=(dt, 1)), nn.GELU(),
                nn.Conv2d(ch, ch, 1)))
            self.norms.append(nn.GroupNorm(8, ch))
        self.gru = nn.GRU(ch, ch, batch_first=True, bidirectional=True)
        self.mix = nn.Linear(2 * ch, ch)
        self.out = nn.Conv2d(ch, 1, 1)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)   # 安定化: 一様分布から開始
        self.ref = nn.Linear(2 * ch, 1) if REFH else None
        if self.ref is not None:
            nn.init.zeros_(self.ref.weight); nn.init.zeros_(self.ref.bias)
    def forward(self, x, mask):                       # x:(B,5,T,181) mask:(B,T)
        h = self.inp(x)
        for blk, nm in zip(self.blocks, self.norms):
            h = nm(h + blk(h))
        g = h.mean(3).transpose(1, 2)                 # (B,T,ch) δ平均=時間系列
        g2, _ = self.gru(g)
        gm = self.mix(g2).transpose(1, 2)[:, :, :, None]  # (B,ch,T,1)
        h = h + gm                                    # 大域文脈をδ全binへ放送
        ref = (2.0 * torch.tanh(self.ref(g2)[:, :, 0])) if self.ref is not None else None
        return self.out(h)[:, 0], ref                 # (B,T,181), (B,T)|None

def make_batch(wns):
    Tm = max(EVD[w]["EV"].shape[1] for w in wns); B = len(wns)
    x = torch.zeros(B, 10, Tm, ND); m = torch.zeros(B, Tm)
    dt = torch.full((B, Tm), float("nan")); wg = torch.zeros(B, Tm)
    for i, w in enumerate(wns):
        e = EVD[w]; T = e["EV"].shape[1]
        x[i, :, :T] = torch.from_numpy(e["EV"].astype(np.float32))
        m[i, :T] = 1.0
        dt[i, :T] = torch.from_numpy(e["dtrue"]); wg[i, :T] = torch.from_numpy(e["wgt"])
    return x.to(DEV), m.to(DEV), dt.to(DEV), wg.to(DEV)

def train_fold(f, trn):
    torch.manual_seed(BSEED * 100 + f); rng = np.random.default_rng(BSEED * 100 + f)
    net = DecoderB().to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=BLR, weight_decay=1e-4)
    offs_t = torch.arange(-sa.NB, sa.NB + 1, device=DEV).float() * sa.STEP
    BW = 6
    lens = {w: EVD[w]["EV"].shape[1] for w in trn}
    def make_batches(order):
        """シャッフル→局所長さソート→トークン上限(井戸数×最大長 ≤ TOKMAX)でグループ化。"""
        ws_ = [trn[j] for j in order]
        wl = [w for i in range(0, len(ws_), 64) for w in sorted(ws_[i:i + 64], key=lambda x: lens[x])]
        batches = []; cur = []
        for w in wl:
            t = cur + [w]; Tmx = max(lens[x] for x in t)
            if cur and (len(t) * Tmx > TOKMAX or len(t) > BW):
                batches.append(cur); cur = [w]
            else:
                cur = t
        if cur: batches.append(cur)
        return batches
    for ep in range(BEPOCHS):
        for pg in opt.param_groups:
            pg["lr"] = BLR * 0.5 * (1 + math.cos(math.pi * ep / max(BEPOCHS - 1, 1)))
        net.train(); order = rng.permutation(len(trn)); losses = []; nskip = 0
        for wns in make_batches(order):
            try:
                x, m, dtr, wg = make_batch(wns)
                if BFLIP and rng.random() < 0.5:
                    x = torch.flip(x, dims=[2]); dtr = torch.flip(dtr, dims=[1])
                    wg = torch.flip(wg, dims=[1]); m = torch.flip(m, dims=[1])
                logits, refv = net(x, m)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache(); nskip += 1; continue
            kt = (dtr / sa.STEP + sa.NB)
            ok = m.bool() & torch.isfinite(dtr) & (kt >= 0) & (kt <= ND - 1) & (wg > 0)
            if not ok.any(): continue
            ktc = torch.nan_to_num(kt, nan=float(sa.NB)).clamp(0, ND - 1)   # ★NaN無害化(0×NaN勾配汚染の防止)
            d = (torch.arange(ND, device=DEV)[None, None, :] - ktc[:, :, None])
            tgt = torch.exp(-0.5 * (d / 2.0) ** 2); tgt = tgt / tgt.sum(-1, keepdim=True).clamp_min(1e-12)
            lsm = F.log_softmax(logits, dim=-1)
            ce = -(tgt * lsm).sum(-1)
            P = lsm.exp(); dmean = (P * offs_t[None, None, :]).sum(-1)
            if refv is not None: dmean = dmean + refv
            se = (dmean - torch.nan_to_num(dtr)) ** 2
            wgn = wg / wg[ok].mean()
            loss = (ce[ok] * wgn[ok]).mean() + LAM * (se[ok] * wgn[ok]).mean()
            if not torch.isfinite(loss):
                opt.zero_grad(set_to_none=True); nskip += 1; continue
            opt.zero_grad(set_to_none=True); loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            if not torch.isfinite(gn):
                opt.zero_grad(set_to_none=True); nskip += 1; continue
            opt.step(); losses.append(float(loss))
        if (ep + 1) % 5 == 0 or ep < 3:
            print(f"  [f{f}] ep{ep+1}/{BEPOCHS} loss={np.mean(losses) if losses else float('nan'):.4f} skip={nskip}", flush=True)
    torch.save(net.state_dict(), ART / f"stageB_dec{BTAG}_f{f}.pt")
    return net

@torch.no_grad()
def eval_fold(net, val):
    net.eval(); errs = []
    offs_t = torch.arange(-sa.NB, sa.NB + 1, device=DEV).float() * sa.STEP
    for w in val:
        e = EVD[w]
        x, m, _, _ = make_batch([w])
        def _dec(xx):
            lg, rf = net(xx, m)
            Pp = torch.softmax(lg, dim=-1)
            dm = (Pp * offs_t[None, None, :]).sum(-1)
            if rf is not None: dm = dm + rf
            return dm
        dmf = _dec(x)
        if BFLIP:
            dmf = 0.5 * (dmf + torch.flip(_dec(torch.flip(x, dims=[2])), dims=[1]))   # TTA: 正順+逆順
        dh = dmf[0, :e["EV"].shape[1]].cpu().numpy()
        z = e["st"] + dh[e["bid"]]
        err = z - e["y_ev"]
        errs.append(err[np.isfinite(err)])
    return errs

if __name__ == "__main__":
    agg = []; per_fold = {}
    for f in FOLDS:
        val = WELLS[f::5]; trn = [w for w in WELLS if w not in set(val)]
        fp = ART / f"stageB_dec{BTAG}_f{f}.pt"
        if fp.exists():
            net = DecoderB().to(DEV); net.load_state_dict(torch.load(fp, map_location=DEV))
            print(f"[f{f}] decoder load")
        else:
            t0 = time.time(); net = train_fold(f, trn)
            print(f"[f{f}] trained ({time.time()-t0:.0f}s)")
        errs = eval_fold(net, val)
        e = np.concatenate(errs)
        r = float(np.sqrt(np.mean(e ** 2)))
        per_fold[f] = r; agg.extend(errs)
        print(f"[foldB{f}] StageB RMSE = {r:.4f}", flush=True)
    e = np.concatenate(agg)
    total = float(np.sqrt(np.mean(e ** 2)))
    print(f"\n=== v50 Stage B(学習デコーダ)folds={FOLDS} プールRMSE = {total:.4f} ===")
    print("  参考: capaug hybrid固定FB 8.156 / v38錨PF 8.132 / v50×v38 blend 7.319")
    json.dump(dict(per_fold=per_fold, total=total),
              open(ART / f"stageB_results{BTAG}.json", "w"), indent=1)
