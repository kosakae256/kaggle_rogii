# -*- coding: utf-8 -*-
"""
v71 — PF-free / filter-free iTransformer on δ-band emission image(v50B の証拠形 × v70 iTransformer核)
=====================================================================================================
発端(ユーザー 2026-07-15): 「v70 の iTransformer を v50 StageB の形(δ帯 emission画像)で、**PFなし**で。
  まっすぐになってた線が、うまくGR特性を理解してくれるんじゃないか。まっすぐにする場合ペナルティを与えてもいい」。
確定した設計方針(AskUserQuestion):
  - GR証拠 = **生マルチ表現マッチ**(v70/v68 cache の tw/self/nbr グリッドを δ帯へラスタライズ)。PF・FBフィルタは一切使わない。
  - コミット機構 = **純iTransformer + 直進ペナルティだけ**で commit できるかの純粋検証(FBグリッド事後も入れない)。

構造:
  - 入力 = **struct中心 δ帯(±DR, 91bin)の emission画像** E[B,C,T,Dd]:
      ch = { tw / self / nbr の GR一致尤度 exp(-0.5·((maGR-refGR)/GR_SIG)²), struct prior(δ), self/nbr被覆 } = 6ch。
  - 核 = **iTransformer変数attention(ch間: どの表現を信じるか)** → **時間TCN(dip連続)** → per-row δ logits。
  - 出力 = δ softmax → 期待δ → **TVT = center + δ**。decoder は zero-init、logits に **LOGB·log(emission集約)** を加算
      → 学習開始は「GR emissionに追従」(=錨δ0に居座らない)。これが構造的な"直進しない"デフォルト。
  - 損失 = CE(真δ soft) + W_MSE·δ²(posterior-mean) + **W_EMIT·emission追従**(pを高emission δへ) + STRAIGHT_PEN(明示直進罰,既定0)。
  - 転移ガードレール = leave-field-out(NN脚のLB非転移を早期検出。v34/v62/v69)。

使い方(cache は v68/v70 と共通):
  CV_MODE=field STRUCT=1 .venv/Scripts/python.exe v71/itransformer_img_v71.py
env: CV_MODE(field) STRUCT(1) EPOCHS(15) BATCH(8) LR(1.5e-3) DMODEL(64) ATTN_HEADS(4) NVAR_LAYERS(2)
     TCN_LEVELS(4) DROP(0.1) LOGB(1.0) W_EMIT(0.3) W_MSE(1.0) TGT_SIG(1.5) STRAIGHT_PEN(0.0) N_WELLS(0=全)
"""
import os, sys, json, math, time
from pathlib import Path
import numpy as np
sys.stdout = open(1, "w", encoding="utf-8", closefd=False)
PROJ = Path(r"C:\Users\kosaka256\Documents\rogii_claude")
sys.path.insert(0, str(PROJ / "v68"))
import importlib.util as _u
_sp = _u.spec_from_file_location("g68", str(PROJ / "v68" / "gridfilter_v68.py"))
g68 = _u.module_from_spec(_sp); _sp.loader.exec_module(g68)
import torch
import torch.nn as nn

OUT = PROJ / "v71" / "artifacts_v71"; OUT.mkdir(parents=True, exist_ok=True)
CV_MODE = g68.CV_MODE; USE_STRUCT = g68.USE_STRUCT; CENTER_KEY = g68.CENTER_KEY
GR_SIG = g68.GR_SIG; GSTEP = g68.GSTEP; DR = float(g68.DR_EFF); DSTEP = float(g68.DSTEP_EFF)
SEED = g68.SEED; EVAL_STRIDE = g68.EVAL_STRIDE
EPOCHS = int(os.environ.get("EPOCHS", "15")); BATCH = int(os.environ.get("BATCH", "8"))
LR = float(os.environ.get("LR", "1.5e-3")); DMODEL = int(os.environ.get("DMODEL", "64"))
HEADS = int(os.environ.get("ATTN_HEADS", "4")); NVAR = int(os.environ.get("NVAR_LAYERS", "2"))
TCN_LV = int(os.environ.get("TCN_LEVELS", "4")); DROP = float(os.environ.get("DROP", "0.1"))
LOGB = float(os.environ.get("LOGB", "1.0"))          # logits への emission log-bias(構造的な"GR追従"デフォルト)
W_EMIT = float(os.environ.get("W_EMIT", "0.3"))      # 直進ペナルティ本体: p を高emission δ へ
W_MSE = float(os.environ.get("W_MSE", "1.0"))
TGT_SIG = float(os.environ.get("TGT_SIG", "1.5"))    # soft CE 目標のσ(δbin単位)
STRAIGHT_PEN = float(os.environ.get("STRAIGHT_PEN", "0.0"))   # 明示直進罰(δ≈0にmass&emission非中心なら罰)
STRAIGHT_MARGIN = float(os.environ.get("STRAIGHT_MARGIN", "4.0"))  # ±この範囲を"直進(錨近傍)"とみなす
DEVICE = "cuda" if (os.environ.get("DEVICE", "auto") != "cpu" and torch.cuda.is_available()) else "cpu"


# ============================ 前処理 / collate(PFなし) ============================
def _ma(x, w):
    import pandas as pd
    return pd.Series(x, dtype="float64").rolling(w, center=True, min_periods=1).mean().to_numpy()

def add_reps(w):
    w["ma7"] = _ma(np.asarray(w["hgr"], float), 7)
    w["center"] = np.asarray(w[CENTER_KEY], float)
    return w

def collate(batch, device):
    B = len(batch); T = max(len(w["hgr"]) for w in batch); G = max(w["gn"] for w in batch)
    def padT(key, fill=0.0):
        a = np.full((B, T), fill, np.float32)
        for i, w in enumerate(batch): a[i, :len(w[key])] = w[key]
        return a
    def padG(key):
        a = np.zeros((B, G), np.float32)
        for i, w in enumerate(batch): a[i, :len(w[key])] = w[key]
        return a
    mask = np.zeros((B, T), np.float32)
    for i, w in enumerate(batch): mask[i, :len(w["hgr"])] = 1.0
    t = lambda x: torch.tensor(x, device=device)
    return dict(hgr=t(padT("hgr")), ma7=t(padT("ma7")), center=t(padT("center")),
                s_sig=t(padT("s_sig", 30.0)), tgt=t(padT("target")), mask=t(mask),
                tw_grid=t(padG("tw_grid")), self_grid=t(padG("self_grid")), self_val=t(padG("self_val")),
                nbr_grid=t(padG("nbr_grid")), nbr_val=t(padG("nbr_val")),
                g0=t(np.array([w["g0"] for w in batch], np.float32)),
                gn=t(np.array([w["gn"] for w in batch], np.int64)))


# ============================ δ帯 emission画像(PF/FB不使用) ============================
def build_emission_image(bt, delta):
    """struct中心δ帯の GR一致尤度画像 E[B,C,T,Dd] と 集約emission分布 Eagg[B,T,Dd] を返す。"""
    B, T = bt["hgr"].shape; dev = bt["hgr"].device; Dd = delta.shape[0]
    g0 = bt["g0"][:, None, None]; gn = bt["gn"]
    cand = bt["center"][:, :, None] + delta[None, None, :]                 # [B,T,Dd] TVT候補
    idx = ((cand - g0) / GSTEP).round().long()
    idx = torch.minimum(idx.clamp_min(0), (gn - 1)[:, None, None])
    idxf = idx.reshape(B, -1)
    def gath(grid): return torch.gather(grid, 1, idxf).reshape(B, T, Dd)
    ma7 = bt["ma7"][:, :, None]
    def emis(grid, valgrid):
        c = (ma7 - gath(grid)) / GR_SIG; e = torch.exp(-0.5 * c * c)
        if valgrid is not None:
            v = gath(valgrid); return e * v, v
        return e, torch.ones_like(e)
    e_tw, _ = emis(bt["tw_grid"], None)
    e_sf, v_sf = emis(bt["self_grid"], bt["self_val"])
    e_nb, v_nb = emis(bt["nbr_grid"], bt["nbr_val"])
    if USE_STRUCT:
        ssig = bt["s_sig"][:, :, None].clamp_min(3.0)
        e_st = torch.exp(-0.5 * (delta[None, None, :] / ssig) ** 2)        # struct prior(δ, 中心0)
    else:
        e_st = torch.zeros(B, T, Dd, device=dev)
    E = torch.stack([e_tw, e_sf, e_nb, e_st, v_sf, v_nb], 1)               # [B,6,T,Dd]
    Eagg = (e_tw + e_sf + e_nb) / (1.0 + v_sf + v_nb)                      # 被覆重みGR emission集約
    Eagg = Eagg / Eagg.sum(-1, keepdim=True).clamp_min(1e-6)              # per-row 分布化
    return E, Eagg


# ============================ モデル: iTransformer変数attn(ch間) + 時間TCN ============================
class ChanAttn(nn.Module):
    """各MD行で ch(表現)間 self-attention = iTransformer の inverted attention。"""
    def __init__(self, C, Dd, d):
        super().__init__()
        self.emb = nn.Linear(Dd, d); self.cpos = nn.Parameter(torch.randn(C, d) * 0.02)
        enc = nn.TransformerEncoderLayer(d, HEADS, d * 2, DROP, batch_first=True, activation="gelu")
        self.tr = nn.TransformerEncoder(enc, NVAR)
    def forward(self, E):                                    # E[B,C,T,Dd]
        B, C, T, Dd = E.shape
        x = self.emb(E.permute(0, 2, 1, 3).reshape(B * T, C, Dd)) + self.cpos[None]
        return self.tr(x).mean(1).reshape(B, T, -1)          # ch間attention→プール [B,T,d]

class TCN(nn.Module):
    def __init__(self, d, levels, drop, k=5):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(levels):
            dil = 2 ** i
            self.blocks.append(nn.Sequential(
                nn.Conv1d(d, d, k, padding=(k - 1) // 2 * dil, dilation=dil), nn.GELU(), nn.Dropout(drop),
                nn.Conv1d(d, d, k, padding=(k - 1) // 2 * dil, dilation=dil)))
    def forward(self, h):
        x = h.transpose(1, 2)
        for b in self.blocks: x = x + b(x)
        return x.transpose(1, 2)

class ITrImg(nn.Module):
    def __init__(self, C, Dd, d=DMODEL):
        super().__init__()
        self.chan = ChanAttn(C, Dd, d); self.tcn = TCN(d, TCN_LV, DROP); self.norm = nn.LayerNorm(d)
        self.rowvec = nn.Parameter(torch.randn(d) * 0.02)
        self.dec = nn.Linear(d, Dd)
        nn.init.zeros_(self.dec.weight); nn.init.zeros_(self.dec.bias)   # 開始=emission追従(logits=LOGB·logEagg)
    def forward(self, E, logEagg, rowfrac):
        z = self.chan(E) + rowfrac[:, :, None] * self.rowvec[None, None, :]
        h = self.norm(z + self.tcn(z))
        return self.dec(h) + LOGB * logEagg                             # logits[B,T,Dd]


# ============================ run / 損失 / CV ============================
def run(model, bt, delta):
    E, Eagg = build_emission_image(bt, delta)
    logEagg = torch.log(Eagg.clamp_min(1e-6))
    B, T = bt["hgr"].shape
    rowfrac = torch.linspace(0, 1, T, device=E.device)[None, :].expand(B, T)
    logits = model(E, logEagg, rowfrac)
    p = torch.softmax(logits, -1)
    dpred = (p * delta[None, None, :]).sum(-1)                          # 期待δ
    return bt["center"] + dpred, logits, Eagg, dpred

def loss_fn(logits, Eagg, dpred, bt, delta):
    dtrue = (bt["tgt"] - bt["center"]).clamp(-DR, DR)                   # [B,T]
    q = torch.exp(-0.5 * ((delta[None, None, :] - dtrue[:, :, None]) / (TGT_SIG * DSTEP)) ** 2)
    q = q / q.sum(-1, keepdim=True).clamp_min(1e-6)
    logp = torch.log_softmax(logits, -1); p = logp.exp()
    L_ce = -(q * logp).sum(-1)                                         # 真δへコミット
    L_mse = ((dpred - dtrue) / DR) ** 2
    L_emit = -(p * torch.log(Eagg.clamp_min(1e-6))).sum(-1)            # ★直進ペナルティ: pを高emission δへ
    L = L_ce + W_MSE * L_mse + W_EMIT * L_emit
    if STRAIGHT_PEN > 0:                                               # 明示直進罰: 錨近傍(±margin)にmass&emissionは非中心
        near = (delta.abs() < STRAIGHT_MARGIN).float()[None, None, :]
        off = 1.0 - (Eagg * near).sum(-1)                             # emissionが中心に無い度合い
        L = L + STRAIGHT_PEN * (p * near).sum(-1) * off
    m = bt["mask"]
    return (L * m).sum() / m.sum().clamp_min(1)

def train_fold(wtr, wva, delta, C):
    Dd = delta.shape[0]
    model = ITrImg(C, Dd).to(DEVICE); opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    batches = g68._len_buckets(wtr, BATCH); rng = np.random.RandomState(SEED); order = np.arange(len(batches))
    for ep in range(EPOCHS):
        model.train(); rng.shuffle(order); tot = 0.0; nb = 0; t0 = time.time()
        for bi in order:
            bt = collate([wtr[j] for j in batches[bi]], DEVICE)
            tvt, logits, Eagg, dpred = run(model, bt, delta)
            loss = loss_fn(logits, Eagg, dpred, bt, delta)
            if not torch.isfinite(loss): opt.zero_grad(); continue
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
            tot += loss.item(); nb += 1
        print(f"    ep{ep+1}/{EPOCHS} loss={tot/max(nb,1):.4f} {time.time()-t0:.0f}s", flush=True)
    model.eval(); oof = {}
    with torch.no_grad():
        for bidx in g68._len_buckets(wva, BATCH):
            batch = [wva[j] for j in bidx]; bt = collate(batch, DEVICE); tvt = run(model, bt, delta)[0]
            for b, w in enumerate(batch): oof[w["wid"]] = tvt[b, :len(w["hgr"])].cpu().numpy()
    return oof

def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    wells = g68.load_cache()
    _nw = int(os.environ.get("N_WELLS", "0")) or None
    if _nw: wells = wells[:_nw]
    if CENTER_KEY not in wells[0]:
        print(f"★cacheに'{CENTER_KEY}'無し。EVAL_STRIDE={EVAL_STRIDE}で v68 cache 作り直し"); return
    for w in wells: add_reps(w)
    delta = torch.arange(-DR, DR + 1e-6, DSTEP, device=DEVICE, dtype=torch.float32)
    C = build_emission_image(collate(wells[:2], DEVICE), delta)[0].shape[1]
    print(f"[v71 iTr-img] wells {len(wells)} CV={CV_MODE} STRUCT={int(USE_STRUCT)} C={C} δ={len(delta)}bin "
          f"d={DMODEL} LOGB={LOGB} W_EMIT={W_EMIT} STRAIGHT_PEN={STRAIGHT_PEN} device={DEVICE}")
    from sklearn.model_selection import GroupKFold
    if CV_MODE == "field":
        fld = g68.make_fields(wells); nf = min(5, len(set(fld))); assign = np.zeros(len(wells), int)
        for k, (_, va) in enumerate(GroupKFold(nf).split(wells, groups=fld)): assign[va] = k
        folds = list(range(nf)); print(f"[cv] leave-field-out {len(set(fld))}fields→{nf}fold")
    else:
        g = np.array([w["wid"] for w in wells]); assign = np.zeros(len(wells), int)
        for k, (_, va) in enumerate(GroupKFold(min(5, len(wells))).split(wells, groups=g)): assign[va] = k
        folds = list(range(min(5, len(wells)))); print("[cv] well GroupKFold")
    oof = {}; t0 = time.time()
    for k in folds:
        tr = [wells[i] for i in np.where(assign != k)[0]]; va = [wells[i] for i in np.where(assign == k)[0]]
        if not tr or not va: continue
        print(f"  fold {k}: train {len(tr)} / val {len(va)}", flush=True)
        oof.update(train_fold(tr, va, delta, C))
    perw, pool, spool = {}, [], []
    for w in wells:
        if w["wid"] not in oof: continue
        p = oof[w["wid"]]; tg = w["target"]; ct = w["center"]; n = min(len(p), len(tg))
        e = p[:n] - tg[:n]; se = ct[:n] - tg[:n]; fin = np.isfinite(e)
        if fin.any(): perw[w["wid"]] = float(np.sqrt(np.mean(e[fin] ** 2))); pool.append(e[fin]); spool.append(se[fin])
    overall = float(np.sqrt(np.mean(np.concatenate(pool) ** 2)))
    base = float(np.sqrt(np.mean(np.concatenate(spool) ** 2)))
    tag = f"{CV_MODE}_{'st' if USE_STRUCT else 'free'}"
    print(f"\n[v71 iTr-img] {tag}  overall RMSE = {overall:.4f}  ({'struct' if USE_STRUCT else 'flat'}錨基準 {base:.4f})")
    print(f"          坑井別中央 {np.median(list(perw.values())):.4f}  ({len(perw)}坑井, {time.time()-t0:.0f}s)")
    (OUT / f"results_{tag}.json").write_text(json.dumps(dict(overall=overall, base=base, struct=USE_STRUCT,
        per_well_median=float(np.median(list(perw.values()))), n=len(perw), cv=CV_MODE,
        cfg=dict(C=C, dmodel=DMODEL, nvar=NVAR, tcn=TCN_LV, epochs=EPOCHS, logb=LOGB, w_emit=W_EMIT,
                 straight_pen=STRAIGHT_PEN, eval_stride=EVAL_STRIDE)), indent=2))
    print(f"[saved] results_{tag}.json")


if __name__ == "__main__":
    main()
