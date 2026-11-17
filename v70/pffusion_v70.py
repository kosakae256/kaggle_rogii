# -*- coding: utf-8 -*-
"""
v70 PF-fusion — 複数PFの評価・統合(iTransformer_PF_fusion.md 準拠)
=====================================================================================
仕様(v70/iTransformer_PF_fusion.md): 各PFを「変数(token)」化し、per-PF Temporal Encoder →
cross-PF Attention → Gating/Fusion Head で時刻ごとに信頼度αを推定 → TVT = Σ α_k·pf_ancc_k。
- §4.1 PF単位トークン化(全特徴フラット化を避け、PF間比較の帰納バイアスを持たせる)。
- §5 損失 = L_state(融合RMSE) + λ_o·KL(soft-oracle q ‖ α) + λ_s·切替罰(既定0)。
- §8 residual Δx は既定OFF(まず純convex fusionで選択器が機能してるか確認)。
- §6 convex-fusion oracle(3.555)を上限基準に。
- struct判断: 融合出力は struct-free PF で束縛→coreはstruct不要。既定 STRUCT=0。STRUCT=1でgating補助文脈に s_tvt/s_sig を追加(転移レバーになるので任意)。
入力PF: v70/artifacts_v70/multipf_s{stride}.pkl(=build_multipf_v70。既定v28の9本 struct-free)。
使い方(cache/companion は既存を流用):
  EVAL_STRIDE=4 .venv/Scripts/python.exe v70/build_multipf_v70.py            # 無ければ
  CV_MODE=field .venv/Scripts/python.exe v70/pffusion_v70.py                 # struct無し融合(本命)
  CV_MODE=field RESID=1 .venv/Scripts/python.exe v70/pffusion_v70.py         # §8 残差補正も
env: CV_MODE(field) STRUCT(0) EVAL_STRIDE(4) EPOCHS(15) BATCH(8) LR(1.5e-3) DMODEL(64)
     ATTN_HEADS(4) TCN_LEVELS(4) DROP(0.1) TAU(1.0) LAMBDA_O(0.3) TAU_O(4.0) RESID(0) CORR(30) N_WELLS(0)
"""
import os, sys, json, math, time, pickle
from pathlib import Path
import numpy as np
sys.stdout = open(1, "w", encoding="utf-8", closefd=False)
PROJ = Path(r"C:\Users\kosaka256\Documents\rogii_claude")
sys.path.insert(0, str(PROJ / "v68"))
import importlib.util as _u
_sp = _u.spec_from_file_location("g68", str(PROJ / "v68" / "gridfilter_v68.py")); g68 = _u.module_from_spec(_sp); _sp.loader.exec_module(g68)
import torch, torch.nn as nn, torch.nn.functional as F

OUT = PROJ / "v70" / "artifacts_v70"; OUT.mkdir(parents=True, exist_ok=True)
CV_MODE = g68.CV_MODE; USE_STRUCT = g68.USE_STRUCT; CENTER_KEY = g68.CENTER_KEY
GR_SIG = g68.GR_SIG; GSTEP = g68.GSTEP; SEED = g68.SEED; N_FIELDS = g68.N_FIELDS; EVAL_STRIDE = g68.EVAL_STRIDE
DR = g68.DR_EFF
EPOCHS = int(os.environ.get("EPOCHS", "15")); BATCH = int(os.environ.get("BATCH", "8")); LR = float(os.environ.get("LR", "1.5e-3"))
DMODEL = int(os.environ.get("DMODEL", "64")); HEADS = int(os.environ.get("ATTN_HEADS", "4")); TCN_LV = int(os.environ.get("TCN_LEVELS", "4")); DROP = float(os.environ.get("DROP", "0.1"))
TAU = float(os.environ.get("TAU", "1.0")); LAMBDA_O = float(os.environ.get("LAMBDA_O", "0.3")); TAU_O = float(os.environ.get("TAU_O", "4.0"))
RESID = os.environ.get("RESID", "0") == "1"; CORR = float(os.environ.get("CORR", "30"))
MA_WINS = [3, 7, 15]; DEC = 10
DEVICE = "cuda" if (os.environ.get("DEVICE", "auto") != "cpu" and torch.cuda.is_available()) else "cpu"


def _ma(x, w):
    import pandas as pd
    return pd.Series(x, dtype="float64").rolling(w, center=True, min_periods=1).mean().to_numpy()

def _dec(x, s):
    n = len(x); cuts = sorted(set([0, n] + list(range(s, n, s)))); xb, mb = [], []
    for a, b in zip(cuts[:-1], cuts[1:]): xb.append(float(np.mean(x[a:b]))); mb.append((a + b) / 2.0)
    return np.interp(np.arange(n), mb, xb)

def add_reps(w):
    h = np.asarray(w["hgr"], float)
    w["ma7"] = _ma(h, 7); w["ma15"] = _ma(h, 15); w["dec"] = _dec(h, DEC)
    w["center"] = np.asarray(w[CENTER_KEY], float)
    return w


def collate(batch, dev):
    B = len(batch); T = max(len(w["hgr"]) for w in batch); G = max(w["gn"] for w in batch); K = batch[0]["pf_banks"].shape[0]
    def pT(k, f=0.0):
        a = np.full((B, T), f, np.float32)
        for i, w in enumerate(batch): a[i, :len(w[k])] = w[k]
        return a
    def pG(k):
        a = np.zeros((B, G), np.float32)
        for i, w in enumerate(batch): a[i, :len(w[k])] = w[k]
        return a
    def pKT(k):
        a = np.zeros((B, K, T), np.float32)
        for i, w in enumerate(batch): a[i, :, :w[k].shape[1]] = w[k]
        return a
    m = np.zeros((B, T), np.float32)
    for i, w in enumerate(batch): m[i, :len(w["hgr"])] = 1.0
    t = lambda x: torch.tensor(x, device=dev)
    return dict(hgr=t(pT("hgr")), ma7=t(pT("ma7")), ma15=t(pT("ma15")), dec=t(pT("dec")),
                center=t(pT("center")), s_sig=t(pT("s_sig", 30.0)), s_tvt=t(pT("s_tvt")), tgt=t(pT("target")), mask=t(m),
                pf_banks=t(pKT("pf_banks")), pf_std=t(pKT("pf_std")), pf_med=t(pT("pf_med")),
                pf_disagree=t(pT("pf_disagree")),
                tw=t(pG("tw_grid")), sf=t(pG("self_grid")), sfv=t(pG("self_val")), nb=t(pG("nbr_grid")), nbv=t(pG("nbr_val")),
                g0=t(np.array([w["g0"] for w in batch], np.float32)), gn=t(np.array([w["gn"] for w in batch], np.int64)))


def _gather(grid, idx):
    return torch.gather(grid, 1, idx.reshape(idx.shape[0], -1).clamp(0, grid.shape[1] - 1)).reshape(idx.shape)

def build_tokens(bt):
    """per-PF token特徴 [B,T,K,F] と 共有文脈 [B,T,Fc] を作る。全てGR/PF由来。"""
    B, T = bt["hgr"].shape; g0 = bt["g0"][:, None, None]; gn = bt["gn"]
    def gr_at(grid, cand):
        idx = ((cand - g0) / GSTEP).round().long(); idx = torch.minimum(idx.clamp_min(0), (gn - 1)[:, None, None])
        return _gather(grid, idx)
    banks = bt["pf_banks"]                                  # [B,K,T]
    bnk = banks.permute(0, 2, 1)                            # [B,T,K]
    sc = max(float(DR), 1.0)
    rel_med = (bnk - bt["pf_med"][:, :, None]) / sc         # 合意からのずれ
    log_std = torch.log1p(bt["pf_std"].permute(0, 2, 1).clamp_min(0))
    ma7 = bt["ma7"][:, :, None]
    gr_tw = (ma7 - gr_at(bt["tw"], bnk)) / GR_SIG           # 各PF位置のGR不一致
    svv = _gather(bt["sfv"], ((bnk - g0) / GSTEP).round().long().clamp_min(0).minimum((gn - 1)[:, None, None]))
    nvv = _gather(bt["nbv"], ((bnk - g0) / GSTEP).round().long().clamp_min(0).minimum((gn - 1)[:, None, None]))
    gr_sf = (ma7 - gr_at(bt["sf"], bnk)) / GR_SIG * svv
    gr_nb = (ma7 - gr_at(bt["nb"], bnk)) / GR_SIG * nvv
    pf_feats = torch.stack([rel_med, log_std, gr_tw, gr_sf, gr_nb], -1)   # [B,T,K,F=5]
    # 共有文脈: GR多表現(z-score) + disagreement + rowfrac (+ struct 任意)
    reps = torch.stack([bt["hgr"], bt["ma7"], bt["ma15"], bt["dec"]], -1); mm = bt["mask"][:, :, None]
    mu = (reps * mm).sum(1, keepdim=True) / mm.sum(1, keepdim=True).clamp_min(1)
    sd = torch.sqrt(((reps - mu) ** 2 * mm).sum(1, keepdim=True) / mm.sum(1, keepdim=True).clamp_min(1) + 1e-6)
    repz = (reps - mu) / sd
    rowf = torch.linspace(0, 1, T, device=bt["hgr"].device)[None, :, None].expand(B, T, 1)
    disag = (bt["pf_disagree"] / sc).clamp(0, 3)[:, :, None]
    ctx = [repz, disag, rowf]
    if USE_STRUCT:
        ctx += [((bt["s_tvt"] - bt["pf_med"]) / sc)[:, :, None], (bt["s_sig"] / 30.0).clamp(0, 3)[:, :, None]]
    ctx = torch.cat(ctx, -1)                                # [B,T,Fc]
    return pf_feats, ctx


class TCN(nn.Module):
    def __init__(self, d, levels, drop, k=5):
        super().__init__(); self.bl = nn.ModuleList()
        for i in range(levels):
            dil = 2 ** i
            self.bl.append(nn.Sequential(nn.Conv1d(d, d, k, padding=(k - 1) // 2 * dil, dilation=dil), nn.GELU(), nn.Dropout(drop),
                                         nn.Conv1d(d, d, k, padding=(k - 1) // 2 * dil, dilation=dil)))
    def forward(self, h):                 # [Bx,T,d]
        x = h.transpose(1, 2)
        for b in self.bl: x = x + b(x)
        return x.transpose(1, 2)


class PFFusion(nn.Module):
    def __init__(self, Fpf, Fc, K, d=DMODEL):
        super().__init__()
        self.K = K
        self.pf_in = nn.Linear(Fpf, d); self.pf_tcn = TCN(d, TCN_LV, DROP)          # per-PF Temporal Encoder(共有)
        self.ctx_in = nn.Linear(Fc, d); self.ctx_tcn = TCN(d, TCN_LV, DROP)
        self.tok_emb = nn.Parameter(torch.randn(K + 1, d) * 0.02)                    # token種別(PF/文脈)
        enc = nn.TransformerEncoderLayer(d, HEADS, d * 2, DROP, batch_first=True, activation="gelu")
        self.cross = nn.TransformerEncoder(enc, 2)                                   # cross-PF Attention
        self.gate = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        nn.init.zeros_(self.gate[-1].weight); nn.init.zeros_(self.gate[-1].bias)     # init=一様α=平均から開始
        self.resid = nn.Linear(d, 1) if RESID else None
        if self.resid is not None: nn.init.zeros_(self.resid.weight); nn.init.zeros_(self.resid.bias)

    def forward(self, pf_feats, ctx, banks):     # pf_feats[B,T,K,F] ctx[B,T,Fc] banks[B,T,K]
        B, T, K, Fp = pf_feats.shape
        e = self.pf_in(pf_feats)                                       # [B,T,K,d]
        e = self.pf_tcn(e.permute(0, 2, 1, 3).reshape(B * K, T, -1)).reshape(B, K, T, -1).permute(0, 2, 1, 3)  # per-PF時間符号化
        ec = self.ctx_tcn(self.ctx_in(ctx))[:, :, None, :]            # [B,T,1,d]
        toks = torch.cat([e, ec], 2) + self.tok_emb[None, None]       # [B,T,K+1,d]
        h = self.cross(toks.reshape(B * T, K + 1, -1)).reshape(B, T, K + 1, -1)
        s = self.gate(h[:, :, :K, :]).squeeze(-1) / TAU               # [B,T,K] logits
        alpha = torch.softmax(s, -1)
        tvt = (alpha * banks).sum(-1)
        if self.resid is not None:
            tvt = tvt + torch.tanh(self.resid(h.mean(2)).squeeze(-1)) * CORR         # §8 系統誤差補正(任意)
        return tvt, alpha, s                                          # s(logits)= KL を log_softmax で安定計算


def run(model, bt):
    pf_feats, ctx = build_tokens(bt); banks = bt["pf_banks"].permute(0, 2, 1)
    return model(pf_feats, ctx, banks), banks


def loss_fn(tvt, s, banks, bt):
    m = bt["mask"]
    Ls = (((tvt - bt["tgt"]) ** 2) * m).sum() / m.sum().clamp_min(1)      # §5 L_state
    if LAMBDA_O > 0:
        err = (banks - bt["tgt"][:, :, None]) ** 2                        # [B,T,K]
        logq = torch.log_softmax(-err / TAU_O, -1)                        # soft-oracle(log)
        logalpha = torch.log_softmax(s, -1)                              # ★log_softmax=安定(log(α→0)の勾配爆発を回避)
        kl = (logq.exp() * (logq - logalpha)).sum(-1)
        Lo = (kl * m).sum() / m.sum().clamp_min(1)
        return Ls + LAMBDA_O * Lo, float(Ls), float(Lo)
    return Ls, float(Ls), 0.0


def train_fold(wtr, wva, Fpf, Fc, K):
    model = PFFusion(Fpf, Fc, K).to(DEVICE); opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    batches = g68._len_buckets(wtr, BATCH); rng = np.random.RandomState(SEED); order = np.arange(len(batches))
    for ep in range(EPOCHS):
        model.train(); rng.shuffle(order); tl = ts = to = 0.0; nb = 0; t0 = time.time()
        for bi in order:
            bt = collate([wtr[j] for j in batches[bi]], DEVICE)
            (tvt, alpha, s), banks = run(model, bt)
            L, ls, lo = loss_fn(tvt, s, banks, bt)
            if not torch.isfinite(L):
                opt.zero_grad(); continue                       # NaN/inf は捨てる(保険)
            opt.zero_grad(); L.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
            tl += L.item(); ts += ls; to += lo; nb += 1
        print(f"    ep{ep+1}/{EPOCHS} L={tl/nb:.3f} state={ts/nb:.3f}(rmse~{math.sqrt(ts/nb):.3f}) oracle={to/nb:.3f} {time.time()-t0:.0f}s", flush=True)
    model.eval(); oof = {}
    with torch.no_grad():
        for bidx in g68._len_buckets(wva, BATCH):
            batch = [wva[j] for j in bidx]; bt = collate(batch, DEVICE); (tvt, _, _), _ = run(model, bt)
            for b, w in enumerate(batch): oof[w["wid"]] = tvt[b, :len(w["hgr"])].cpu().numpy()
    return oof


def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    wells = g68.load_cache()
    _nw = int(os.environ.get("N_WELLS", "0")) or None
    if _nw: wells = wells[:_nw]
    for w in wells: add_reps(w)
    pfby = pickle.loads((OUT / f"multipf_s{EVAL_STRIDE}.pkl").read_bytes()) if (OUT / f"multipf_s{EVAL_STRIDE}.pkl").exists() else {}
    if not pfby:
        print(f"★multipf_s{EVAL_STRIDE}.pkl 無し → 先に `EVAL_STRIDE={EVAL_STRIDE} python v70/build_multipf_v70.py`"); return
    K = pfby[next(iter(pfby))]["ancc"].shape[0]; npf = 0
    for w in wells:
        n = len(w["hgr"]); d = pfby.get(w["wid"]); ctr = np.asarray(w["center"], float)
        if d is not None and d["ancc"].shape[1] >= n:
            A = np.asarray(d["ancc"][:, :n], float); w["pf_banks"] = A; w["pf_std"] = np.asarray(d["std"][:, :n], float)
            w["pf_med"] = np.nanmedian(A, 0); w["pf_disagree"] = np.nanstd(A, 0); npf += 1
        else:
            w["pf_banks"] = np.tile(ctr, (K, 1)); w["pf_std"] = np.full((K, n), 30.0); w["pf_med"] = ctr.copy(); w["pf_disagree"] = np.zeros(n)
    Fpf, Fc = [x.shape[-1] for x in build_tokens(collate(wells[:2], DEVICE))]
    print(f"[pffusion] wells {len(wells)} PF付き{npf} K={K}バンク Fpf={Fpf} Fc={Fc} STRUCT={int(USE_STRUCT)} RESID={int(RESID)} λo={LAMBDA_O} CV={CV_MODE} dev={DEVICE}")
    from sklearn.model_selection import GroupKFold
    if CV_MODE == "field":
        fld = g68.make_fields(wells); nf = min(5, len(set(fld))); asg = np.zeros(len(wells), int)
        for k, (_, va) in enumerate(GroupKFold(nf).split(wells, groups=fld)): asg[va] = k
        folds = list(range(nf)); print(f"[cv] leave-field-out {len(set(fld))}fields→{nf}fold")
    else:
        g = np.array([w["wid"] for w in wells]); asg = np.zeros(len(wells), int)
        for k, (_, va) in enumerate(GroupKFold(5).split(wells, groups=g)): asg[va] = k
        folds = list(range(5)); print("[cv] well GroupKFold(5)")
    oof = {}; t0 = time.time()
    for k in folds:
        tr = [wells[i] for i in np.where(asg != k)[0]]; va = [wells[i] for i in np.where(asg == k)[0]]
        print(f"  fold {k}: train {len(tr)} / val {len(va)}", flush=True); oof.update(train_fold(tr, va, Fpf, Fc, K))
    perw, pool = {}, []
    for w in wells:
        if w["wid"] not in oof: continue
        p = oof[w["wid"]]; tg = w["target"]; n = min(len(p), len(tg)); e = p[:n] - tg[:n]; fin = np.isfinite(e)
        if fin.any(): perw[w["wid"]] = float(np.sqrt(np.mean(e[fin] ** 2))); pool.append(e[fin])
    ov = float(np.sqrt(np.mean(np.concatenate(pool) ** 2)))
    # 参考: median / convex-oracle(共通行)
    mp = []; co = []
    for w in wells:
        if w["wid"] not in oof: continue
        A = w["pf_banks"]; tg = np.asarray(w["target"], float); n = min(A.shape[1], len(tg)); A = A[:, :n]; tg = tg[:n]
        mp.append(np.nanmedian(A, 0) - tg); co.append(np.clip(tg, A.min(0), A.max(0)) - tg)
    rmed = float(np.sqrt(np.mean(np.concatenate(mp) ** 2))); rconv = float(np.sqrt(np.mean(np.concatenate(co) ** 2)))
    tag = f"{CV_MODE}_{'st' if USE_STRUCT else 'free'}{'_resid' if RESID else ''}"
    print(f"\n[pffusion] {tag}  overall RMSE = {ov:.4f}   (median {rmed:.3f} / convex-oracle {rconv:.3f})")
    print(f"          坑井別中央 {np.median(list(perw.values())):.4f}  ({len(perw)}坑井, {time.time()-t0:.0f}s)")
    np.save(OUT / f"fusion_oof_{tag}.npy", np.array([v for w in wells if w["wid"] in oof for v in oof[w["wid"]]], np.float32))
    (OUT / f"fusion_results_{tag}.json").write_text(json.dumps(dict(overall=ov, median=rmed, convex_oracle=rconv,
        struct=USE_STRUCT, resid=RESID, lambda_o=LAMBDA_O, K=K, n=len(perw), cv=CV_MODE), indent=2))
    print(f"[saved] fusion_results_{tag}.json")


if __name__ == "__main__":
    main()
