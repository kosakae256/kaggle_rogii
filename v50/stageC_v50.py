# -*- coding: utf-8 -*-
"""
v50 Stage C — K本パス仮説の生成 + 検証器による soft 混合
==========================================================
C0 実測: posterior mean 7.778 に対し「正しい枝コミット」oracle 5.17(ヘッジ代2.6ft)。
その回収を狙う:
  C1 生成: 井戸ごとにシフトプロファイル(δ̂+s 窓±6ftの事後質量対数和, s∈±20ft)の
           極大 top-K(≤6, 3ft分離)→ 各シフトのモード条件付き事後平均パス。仮説0=base。
  C2 検証: 仮説ごとの証拠集計特徴(質量/類似度/生尤度/nbr/self/prior逸脱/シフト量/シェア)
           → MLP → 井戸内 listwise softmax → soft混合(hard死・soft生の教訓)。
学習: combo デコーダ(凍結)の posterior 上で、同一 GroupKFold。まず fold0。
実行: python v50/stageC_v50.py    (env FOLDS=0, CEPOCHS=300 ...)
"""
import os, sys, json, math, pickle, time
os.environ.setdefault("TAG", "capaug"); os.environ.setdefault("CH", "128")
os.environ.setdefault("DIM", "96"); os.environ.setdefault("DILS", "1,2,4,8,16,32")
os.environ.setdefault("BTAG", "combo"); os.environ.setdefault("REFH", "1")
os.environ.setdefault("DILT", "1,2,4,8,16"); os.environ.setdefault("BCH", "96")
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import stageB_v50 as sb
import stageA_v50 as sa

DEV = sb.DEV
FOLDS   = [int(x) for x in os.environ.get("FOLDS", "0").split(",")]
CEPOCHS = int(os.environ.get("CEPOCHS", "60"))
KHYP    = int(os.environ.get("KHYP", "6"))
MW      = 12                                     # モード窓 ±12bin=6ft
SH      = np.arange(-20.0, 20.01, 0.5)
SEP     = 6                                      # 極大の分離(bin=3ft)
HYP_FP  = sb.ART / "stageC_hyps.pkl"

offs_np = (np.arange(-sa.NB, sa.NB + 1) * sa.STEP).astype(np.float64)

# ============ C1: 仮説生成(全坑井, comboデコーダ凍結) ============
def gen_hypotheses():
    offs_t = torch.arange(-sa.NB, sa.NB + 1, device=DEV).float() * sa.STEP
    out = {}
    with torch.no_grad():
        for f in range(5):
            net = sb.DecoderB().to(DEV)
            net.load_state_dict(torch.load(sb.ART / f"stageB_dec{sb.BTAG}_f{f}.pt", map_location=DEV))
            net.eval()
            for w in sb.WELLS[f::5]:
                e = sb.EVD[w]
                x, m, _, _ = sb.make_batch([w])
                lg, rf = net(x, m)
                T = e["EV"].shape[1]
                P = torch.softmax(lg[0, :T], dim=-1)
                dh = (P * offs_t[None, :]).sum(-1)
                if rf is not None: dh = dh + rf[0, :T]
                dh = dh.cpu().numpy(); Pn = P.cpu().numpy().astype(np.float64)
                nb_ = len(dh)
                # 窓質量: cumsum で全シフト一括
                csum = np.concatenate([np.zeros((nb_, 1)), np.cumsum(Pn, 1)], 1)   # (nb,182)
                kk = np.round((dh[:, None] + SH[None, :]) / sa.STEP).astype(np.int64) + sa.NB  # (nb,S)
                lo = np.clip(kk - MW, 0, 181); hi = np.clip(kk + MW + 1, 0, 181)
                mass = np.take_along_axis(csum, hi, 1) - np.take_along_axis(csum, lo, 1)       # (nb,S)
                prof = np.log(np.maximum(mass, 1e-9)).mean(0)                                   # (S,)
                # 極大 top-K(分離 SEP*0.5ft)
                order = np.argsort(-prof); chosen = []
                for si in order:
                    if abs(SH[si]) <= 0.25: continue                 # base(0)と重複するシフトは除外
                    if all(abs(si - c) >= SEP for c in chosen): chosen.append(int(si))
                    if len(chosen) >= KHYP - 1: break
                shifts = [0.0] + [float(SH[c]) for c in chosen]      # 常に K 本(base+K-1)
                # 各仮説パス = モード条件付き事後平均
                paths = []; feats = []; tracks = []
                ch = e["EV"].astype(np.float32)                     # (10,nb,181)
                for s in shifts:
                    ki = np.clip(np.round((dh + s) / sa.STEP).astype(np.int64) + sa.NB, 0, 180)
                    l2 = np.maximum(ki - MW, 0); h2 = np.minimum(ki + MW + 1, 181)
                    dcond = np.empty(nb_); msum = np.empty(nb_)
                    for t in range(nb_):
                        pw = Pn[t, l2[t]:h2[t]]; sw = pw.sum()
                        msum[t] = sw
                        dcond[t] = (pw * offs_np[l2[t]:h2[t]]).sum() / max(sw, 1e-12)
                    kpi = np.clip(np.round(dcond / sa.STEP).astype(np.int64) + sa.NB, 0, 180)
                    gt = lambda c: ch[c][np.arange(nb_), kpi]                      # binトラック
                    g = lambda c: float(gt(c).mean())
                    feats.append([float(np.log(np.maximum(msum, 1e-9)).mean()),   # 事後質量(パス適合)
                                  g(0), g(1), g(5),                               # 類似度/生尤度/γ
                                  g(8), g(9), g(6), g(7),                          # nbr/nbr有/self/self有
                                  g(2),                                            # prior
                                  s, abs(s),
                                  float(np.mean(np.abs(dcond - dh)))])             # baseからの乖離
                    tracks.append(np.stack([np.log(np.maximum(msum, 1e-9)),
                                            gt(0), gt(1), gt(5), gt(8), gt(2)], 0))  # (6,nb)
                    paths.append(dcond.astype(np.float32))
                feats = np.array(feats, np.float32)
                ms = feats[:, 0]; feats = np.concatenate([feats, (ms - ms.max())[:, None]], 1)  # 質量シェア
                out[w] = dict(paths=np.stack(paths), feats=feats, shifts=np.array(shifts, np.float32),
                              tracks=np.stack(tracks).astype(np.float16))          # (K,6,nb)
            print(f"  C1 fold{f} done", flush=True)
    HYP_FP.write_bytes(pickle.dumps(out, protocol=4))
    print(f"hyps saved: {len(out)} wells")
    return out

HYPS = pickle.loads(HYP_FP.read_bytes()) if HYP_FP.exists() else gen_hypotheses()
NF = HYPS[sb.WELLS[0]]["feats"].shape[1]
print(f"hypotheses: {len(HYPS)} wells, feats {NF}")

# ============ C2: 検証器(listwise softmax → soft混合) ============
class Verifier(nn.Module):
    # スカラー特徴 + binトラック(6ch)を conv1d で読む listwise 検証器
    def __init__(self, nf, h=64, ct=32):
        super().__init__()
        self.c1 = nn.Conv1d(6, ct, 7, padding=3)
        self.c2 = nn.Conv1d(ct, ct, 7, padding=12, dilation=4)
        self.net = nn.Sequential(nn.Linear(nf + 2 * ct, h), nn.GELU(),
                                 nn.Linear(h, h), nn.GELU(), nn.Linear(h, 1))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)
    def forward(self, x, tr, tm):              # x:(B,K,F) tr:(B,K,6,T) tm:(B,T)
        B, K, C, T = tr.shape
        mrep = tm.repeat_interleave(K, 0)[:, None, :]
        h = F.gelu(self.c1(tr.reshape(B * K, C, T)))
        h = F.gelu(self.c2(h)) * mrep
        s_ = tm.sum(1).repeat_interleave(K)[:, None].clamp_min(1.0)
        mean = h.sum(-1) / s_
        mx = (h + (mrep - 1) * 1e4).amax(-1)
        z = torch.cat([x.reshape(B * K, -1), mean, mx], 1)
        return self.net(z)[..., 0].view(B, K)

def well_tensors(w):
    """仮説パスの bin誤差(vs dtrue)と特徴を返す。"""
    h = HYPS[w]; e = sb.EVD[w]
    dtr = e["dtrue"]; wgt = e["wgt"]
    ok = np.isfinite(dtr) & (wgt > 0)
    if ok.sum() < 4: return None
    E = h["paths"][:, ok] - dtr[ok][None, :]           # (K,nbo)
    return h["feats"], E.astype(np.float32), wgt[ok].astype(np.float32)

def run_fold(f):
    val = sb.WELLS[f::5]; trn = [w for w in sb.WELLS if w not in set(val)]
    data_t = [(w, well_tensors(w)) for w in trn]; data_t = [(w, d) for w, d in data_t if d]
    torch.manual_seed(f); rng = np.random.default_rng(f)
    ver = Verifier(NF).to(DEV)
    opt = torch.optim.AdamW(ver.parameters(), lr=1e-3, weight_decay=1e-4)
    # 特徴/トラックの正規化統計(train側)
    Xs = np.stack([d[0] for _, d in data_t])                    # (N,K,F)
    mu = Xs.mean((0, 1)); sd = np.maximum(Xs.std((0, 1)), 1e-6)
    tcat = np.concatenate([HYPS[w]["tracks"].astype(np.float32).reshape(KHYP, 6, -1)[:, :, ::4].reshape(6, -1)
                           if False else HYPS[w]["tracks"].astype(np.float32).transpose(1, 0, 2).reshape(6, -1)[:, ::7]
                           for w, _ in data_t[:120]], 1)
    tmu = tcat.mean(1); tsd = np.maximum(tcat.std(1), 1e-6)
    def prep(w):
        h = HYPS[w]
        x = (h["feats"] - mu) / sd
        tr = (h["tracks"].astype(np.float32) - tmu[None, :, None]) / tsd[None, :, None]
        return x.astype(np.float32), tr.astype(np.float32)
    # 教師(listwise soft target)
    P_t = {}; E_t = {}; W_t = {}
    for w, d in data_t:
        E = torch.from_numpy(d[1]).to(DEV); Wt = torch.from_numpy(d[2]).to(DEV)
        rms = ((E ** 2 * Wt[None, :]).sum(1) / Wt.sum()).sqrt()
        P_t[w] = torch.softmax(-(rms - rms.min()) / 1.5, dim=0)
        E_t[w] = E; W_t[w] = Wt
    wl = [w for w, _ in data_t]
    BWC = 24
    for ep in range(CEPOCHS):
        ver.train(); order = rng.permutation(len(wl)); tot = []
        for i0_ in range(0, len(wl), BWC):
            ws_ = [wl[j] for j in order[i0_:i0_ + BWC]]
            Tm = max(HYPS[w]["tracks"].shape[2] for w in ws_)
            xb = torch.zeros(len(ws_), KHYP, NF); tb = torch.zeros(len(ws_), KHYP, 6, Tm)
            tm = torch.zeros(len(ws_), Tm)
            for i, w in enumerate(ws_):
                x, tr = prep(w); T = tr.shape[2]
                xb[i] = torch.from_numpy(x); tb[i, :, :, :T] = torch.from_numpy(tr); tm[i, :T] = 1.0
            xb, tb, tm = xb.to(DEV), tb.to(DEV), tm.to(DEV)
            lg = ver(xb, tb, tm)
            wsoft = torch.softmax(lg, 1)
            ce = torch.stack([-(P_t[w] * F.log_softmax(lg[i], 0)).sum() for i, w in enumerate(ws_)]).mean()
            mse = torch.stack([(((wsoft[i][:, None] * E_t[w]).sum(0)) ** 2 * W_t[w]).sum() / W_t[w].sum()
                               for i, w in enumerate(ws_)]).mean()
            loss = float(os.environ.get('CE_W', '0.2')) * ce + float(os.environ.get('MSE_W', '1.0')) * mse
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); tot.append(float(loss))
        if (ep + 1) % 20 == 0: print(f"  [C2 f{f}] ep{ep+1} loss={np.mean(tot):.4f}", flush=True)
    # ---- val ----
    ver.eval(); errs = []
    with torch.no_grad():
        for w in val:
            h = HYPS[w]; e = sb.EVD[w]
            x, tr = prep(w); T = tr.shape[2]
            xb = torch.from_numpy(x)[None].to(DEV)
            tb = torch.from_numpy(tr)[None].to(DEV)
            tm = torch.ones(1, T, device=DEV)
            wsoft = torch.softmax(ver(xb, tb, tm)[0], 0).cpu().numpy()
            mix = (wsoft[:, None] * h["paths"]).sum(0)
            z = e["st"] + mix[e["bid"]]
            err = z - e["y_ev"]; errs.append(err[np.isfinite(err)])
    ee = np.concatenate(errs)
    return float(np.sqrt(np.mean(ee ** 2))), errs

if __name__ == "__main__":
    agg = []; pf = {}
    for f in FOLDS:
        r, errs = run_fold(f)
        pf[f] = r; agg.extend(errs)
        print(f"[C2 fold{f}] StageC RMSE = {r:.4f}", flush=True)
    ee = np.concatenate(agg)
    print(f"\n=== v50 Stage C folds={FOLDS} プールRMSE = {np.sqrt(np.mean(ee**2)):.4f} ===")
    print("  参考: combo(posterior mean) fold0=5.771 全体7.778 / oracleB 5.175")
    json.dump(dict(per_fold=pf, total=float(np.sqrt(np.mean(ee ** 2)))),
              open(sb.ART / "stageC_results.json", "w"), indent=1)
