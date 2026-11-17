# -*- coding: utf-8 -*-
"""
v70 — GR多表現 × iTransformer(inverted attention) — PF不使用・struct on/offトグル
=====================================================================================
発端(ユーザー 2026-07-15, v69のstruct/PF依存で転移崩壊→クリーンにやり直し):
  「iTransformer を v70 で実装。PFは使わない。structはon/off切替。GRの表現(感覚を狭める=間引き/
   MA/self/neighbor)で相関が変わるので、それらを適用した入力を作る」。

方針:
  - **PF完全不使用**。入力は GR の多表現マッチ特徴のみ(v68 の PFフリー cache=tw/self/nbr グリッド+struct を流用)。
  - **GR多表現(=相関が変わる表現群を全部チャネル化)**:
      well GR: raw / MA{3,7,15} / 間引き(MD-DEC bin平均→補間)
      参照(疑似typewell): tw / self / nbr の各グリッド
      各行で δ 窓のローカル照合(best/2nd-bestコスト=双峰シグナル, PF不使用)も算出。
  - **struct on/off トグル(STRUCT=1/0)**: on=structを δ の錨中心+信頼度チャネルに / off=flat hold(base_free)錨・struct chゼロ。
  - **モデル= iTransformer の核(表現チャネル間の inverted attention)+ MD方向 temporal(連続性)= axial**。
      純iTransformerはMD次元をtoken埋込に畳む→行ごと予測に不向きなので、変数(表現)軸attention + 時間軸TCN の2軸に。
  - **出力= δ分布→posterior-mean**(双峰を平均で潰さずヘッジ)。TVT = 錨 + δ。
  - **転移ガードレール= leave-field-out**(NN脚のLB非転移を早期検出。v34/v62/v69の教訓)。

使い方(cache は v68 と共通。先に無ければ v68 で作る):
  .venv/Scripts/python.exe v68/gridfilter_v68.py cache            # 無ければ(EVAL_STRIDE=4既定)
  CV_MODE=field STRUCT=1 .venv/Scripts/python.exe v70/itransformer_v70.py     # struct有り
  CV_MODE=field STRUCT=0 .venv/Scripts/python.exe v70/itransformer_v70.py     # struct無し
env: CV_MODE(field) STRUCT(1) EVAL_STRIDE(4) EPOCHS(15) BATCH(8) LR(1.5e-3) DMODEL(64) NVAR_LAYERS(2)
     ATTN_HEADS(4) TCN_LEVELS(4) DROP(0.1) LMR(30) LMS(3) N_WELLS(0=全) N_FIELDS(16)
"""
import os, sys, json, math, time, pickle
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
import torch.nn.functional as F

OUT = PROJ / "v70" / "artifacts_v70"; OUT.mkdir(parents=True, exist_ok=True)
CV_MODE  = g68.CV_MODE
USE_STRUCT = g68.USE_STRUCT            # STRUCT env(v68と共通)
CENTER_KEY = g68.CENTER_KEY            # s_tvt / base_free
GR_SIG = g68.GR_SIG; GSTEP = g68.GSTEP
DR = g68.DR_EFF; DSTEP = g68.DSTEP_EFF                 # 出力δグリッド(struct無しは広く粗く=v68流用)
SEED = g68.SEED; N_FIELDS = g68.N_FIELDS; EVAL_STRIDE = g68.EVAL_STRIDE
EPOCHS = int(os.environ.get("EPOCHS", "15")); BATCH = int(os.environ.get("BATCH", "8"))
LR = float(os.environ.get("LR", "1.5e-3")); DMODEL = int(os.environ.get("DMODEL", "64"))
NVAR = int(os.environ.get("NVAR_LAYERS", "2")); HEADS = int(os.environ.get("ATTN_HEADS", "4"))
TCN_LV = int(os.environ.get("TCN_LEVELS", "4")); DROP = float(os.environ.get("DROP", "0.1"))
LMR = float(os.environ.get("LMR", "30")); LMS = float(os.environ.get("LMS", "3"))   # ローカル照合窓(±LMR, step LMS)
USE_PF = os.environ.get("PF", "1") == "1"     # ★v28 の既存PF出力(pf_ancc/std)を入力chに(既定on)
CORR = float(os.environ.get("CORR", "40"))    # ★PF残差学習の補正幅(±CORR ft)。PF土台+小補正=必ずPFを使う
MA_WINS = [3, 7, 15]; DEC = 10
DEVICE = "cuda" if (os.environ.get("DEVICE", "auto") != "cpu" and torch.cuda.is_available()) else "cpu"


# ============================ 多表現 GR 前処理(numpy, per well) ============================
def _ma(x, w):
    import pandas as pd
    return pd.Series(x, dtype="float64").rolling(w, center=True, min_periods=1).mean().to_numpy()

def _dec_interp(x, s):
    n = len(x); cuts = sorted(set([0, n] + list(range(s, n, s))))
    xb, mb = [], []
    for a, b in zip(cuts[:-1], cuts[1:]):
        xb.append(float(np.mean(x[a:b]))); mb.append((a + b) / 2.0)
    return np.interp(np.arange(n), mb, xb)

def add_reps(w):
    """well GR の多表現(raw/MA/間引き)を追加。struct/flat 錨中心も選ぶ。"""
    hgr = np.asarray(w["hgr"], float)
    w["ma3"] = _ma(hgr, MA_WINS[0]); w["ma7"] = _ma(hgr, MA_WINS[1]); w["ma15"] = _ma(hgr, MA_WINS[2])
    w["dec"] = _dec_interp(hgr, DEC)
    w["center"] = np.asarray(w[CENTER_KEY], float)     # s_tvt(struct) or base_free
    return w


# ============================ collate ============================
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
    def padKT(key):                          # [B, K, T](各PFバンク)
        K = batch[0][key].shape[0]; a = np.zeros((B, K, T), np.float32)
        for i, w in enumerate(batch): a[i, :, :w[key].shape[1]] = w[key]
        return a
    mask = np.zeros((B, T), np.float32)
    for i, w in enumerate(batch): mask[i, :len(w["hgr"])] = 1.0
    t = lambda x: torch.tensor(x, device=device)
    out = dict(hgr=t(padT("hgr")), ma3=t(padT("ma3")), ma7=t(padT("ma7")), ma15=t(padT("ma15")), dec=t(padT("dec")),
               center=t(padT("center")), s_sig=t(padT("s_sig", 30.0)), tgt=t(padT("target")), mask=t(mask),
               pf_med=t(padT("pf_med")), pf_disagree=t(padT("pf_disagree")), pf_range=t(padT("pf_range")), pf_banks=t(padKT("pf_banks")),
               tw_grid=t(padG("tw_grid")), self_grid=t(padG("self_grid")), self_val=t(padG("self_val")),
               nbr_grid=t(padG("nbr_grid")), nbr_val=t(padG("nbr_val")),
               g0=t(np.array([w["g0"] for w in batch], np.float32)),
               gn=t(np.array([w["gn"] for w in batch], np.int64)))
    return out


# ============================ 多表現マッチ特徴(PF不使用) ============================
def _gather(grid, idx):   # grid[B,G] idx[B,T,*] -> [B,T,*]
    sh = idx.shape; return torch.gather(grid, 1, idx.reshape(sh[0], -1).clamp(0, grid.shape[1] - 1)).reshape(sh)

def build_feats(bt, dloc):
    """per-MD 変数(表現)特徴 [B,T,V] を作る。全て GR 由来(PF不使用)。"""
    B, T = bt["hgr"].shape; dev = bt["hgr"].device
    g0 = bt["g0"][:, None, None]; gn = bt["gn"]
    center = bt["center"]
    def gr_at(grid, cand):   # cand[B,T,*] TVT -> grid GR
        idx = ((cand - g0) / GSTEP).round().long()
        idx = torch.minimum(idx.clamp_min(0), (gn - 1)[:, None, None])
        return _gather(grid, idx)
    # --- well GR 多表現(z-score per well) ---
    reps = torch.stack([bt["hgr"], bt["ma3"], bt["ma7"], bt["ma15"], bt["dec"]], -1)   # [B,T,5]
    m = bt["mask"][:, :, None]
    mu = (reps * m).sum(1, keepdim=True) / m.sum(1, keepdim=True).clamp_min(1)
    sd = torch.sqrt(((reps - mu) ** 2 * m).sum(1, keepdim=True) / m.sum(1, keepdim=True).clamp_min(1) + 1e-6)
    reps_z = (reps - mu) / sd                                                          # [B,T,5]
    # --- 錨(center)での照合残差(ma7 vs 各参照) ---
    c0 = center[:, :, None]
    tw_c = gr_at(bt["tw_grid"], c0)[:, :, 0]; sf_c = gr_at(bt["self_grid"], c0)[:, :, 0]; nb_c = gr_at(bt["nbr_grid"], c0)[:, :, 0]
    sv = _gather(bt["self_val"], ((c0 - g0) / GSTEP).round().long().clamp_min(0).minimum((gn - 1)[:, None, None]))[:, :, 0]
    nv = _gather(bt["nbr_val"], ((c0 - g0) / GSTEP).round().long().clamp_min(0).minimum((gn - 1)[:, None, None]))[:, :, 0]
    res_tw = (bt["ma7"] - tw_c) / GR_SIG
    res_sf = (bt["ma7"] - sf_c) / GR_SIG * sv
    res_nb = (bt["ma7"] - nb_c) / GR_SIG * nv
    # --- ローカル δ 照合(best/2nd-best/best位置=双峰シグナル。PF不使用) ---
    dloc_t = torch.arange(-LMR, LMR + 1e-6, LMS, device=dev)               # [Dloc]
    cand = center[:, :, None] + dloc_t[None, None, :]                      # [B,T,Dloc]
    def cost_of(grid, valid_grid):
        eg = gr_at(grid, cand)                                             # [B,T,Dloc]
        cst = ((bt["ma7"][:, :, None] - eg) / GR_SIG).abs()               # 小=一致
        vg = gr_at(valid_grid, cand) if valid_grid is not None else torch.ones_like(cst)
        cst = torch.where(vg > 0.5, cst, torch.full_like(cst, 9.9))        # 無効ビンはペナルティ
        best, bi = cst.min(-1)                                             # best cost, argmin
        c2 = cst.clone(); c2.scatter_(-1, bi[:, :, None], 9.9); second = c2.min(-1)[0]
        bestd = dloc_t[bi] / LMR                                           # best δ(正規化)
        gap = (second - best)                                             # 大=非双峰(明瞭) / 小=双峰(曖昧)
        return best, bestd, gap
    twb, twd, twg = cost_of(bt["tw_grid"], None)
    sfb, sfd, sfg = cost_of(bt["self_grid"], bt["self_val"])
    nbb, nbd, nbg = cost_of(bt["nbr_grid"], bt["nbr_val"])
    # --- struct(toggle) ---
    ssig = (bt["s_sig"] / 30.0).clamp(0, 3) if USE_STRUCT else torch.zeros_like(bt["hgr"])
    st_on = torch.full_like(bt["hgr"], 1.0 if USE_STRUCT else 0.0)
    # --- position/meta ---
    rowfrac = torch.linspace(0, 1, T, device=dev)[None, :].expand(B, T)
    feats = torch.stack([
        reps_z[..., 0], reps_z[..., 1], reps_z[..., 2], reps_z[..., 3], reps_z[..., 4],    # GR多表現
        res_tw, res_sf, res_nb, sv, nv,                                                    # 錨での照合残差+valid
        twb, twd, twg, sfb, sfd, sfg, nbb, nbd, nbg,                                        # ローカルδ照合(best/pos/gap)
        ssig, st_on, rowfrac,                                                               # struct+meta
    ], -1)                                                                                  # [B,T,V]
    if USE_PF:   # ★複数PF: median基準 + 各バンクδ + 食い違い(disagreement=硬い坑井のシグナル) + PF位置GR一致
        med = bt["pf_med"]; sc = max(float(DR), 1.0)
        med_dlt = (med - center) / sc
        disag = (bt["pf_disagree"] / sc).clamp(0, 3)                  # バンク間std(大=曖昧=硬い)
        rng = (bt["pf_range"] / sc).clamp(0, 5)
        bank_dlt = (bt["pf_banks"] - center[:, None, :]) / sc         # [B,K,T] 各PFの錨からのδ
        p0 = med[:, :, None]
        pf_tw = (bt["ma7"] - gr_at(bt["tw_grid"], p0)[:, :, 0]) / GR_SIG
        pfsv = _gather(bt["self_val"], ((p0 - g0) / GSTEP).round().long().clamp_min(0).minimum((gn - 1)[:, None, None]))[:, :, 0]
        pf_sf = (bt["ma7"] - gr_at(bt["self_grid"], p0)[:, :, 0]) / GR_SIG * pfsv
        extra = torch.stack([med_dlt, disag, rng, pf_tw, pf_sf], -1)  # [B,T,5]
        bnk = bt["pf_banks"].permute(0, 2, 1)                          # [B,T,K]
        grk = ((bt["ma7"][:, :, None] - gr_at(bt["tw_grid"], bnk)) / GR_SIG)   # [B,T,K] ★各バンク位置のGR不一致=選択の根拠
        feats = torch.cat([feats, extra, bank_dlt.permute(0, 2, 1), grk], -1)  # +5 +K(δ) +K(GRコスト)
    return feats


# ============================ モデル: iTransformer核(変数attn) + 時間TCN ============================
class VarAttn(nn.Module):
    """各MD行で 表現チャネル間 self-attention(= iTransformer の inverted attention)。"""
    def __init__(self, V, d, heads, layers, drop):
        super().__init__()
        self.emb = nn.Linear(1, d)
        self.vpos = nn.Parameter(torch.randn(V, d) * 0.02)
        enc = nn.TransformerEncoderLayer(d, heads, d * 2, drop, batch_first=True, activation="gelu")
        self.tr = nn.TransformerEncoder(enc, layers)
    def forward(self, x):                     # x[B,T,V]
        B, T, V = x.shape
        h = self.emb(x.reshape(B * T, V, 1)) + self.vpos[None]     # [B*T,V,d]
        h = self.tr(h)                                             # 変数間attention
        return h.mean(1).reshape(B, T, -1)                        # 変数プール -> [B,T,d]

class TCN(nn.Module):
    def __init__(self, d, levels, drop, k=5):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(levels):
            dil = 2 ** i
            self.blocks.append(nn.Sequential(
                nn.Conv1d(d, d, k, padding=(k - 1) // 2 * dil, dilation=dil), nn.GELU(), nn.Dropout(drop),
                nn.Conv1d(d, d, k, padding=(k - 1) // 2 * dil, dilation=dil)))
    def forward(self, h):                     # h[B,T,d]
        x = h.transpose(1, 2)
        for b in self.blocks: x = x + b(x)
        return x.transpose(1, 2)

class ITr(nn.Module):
    def __init__(self, V, Dout, K=1, d=DMODEL):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(V, d), nn.GELU())      # ★直接チャネル読み(mean-pool希釈を回避)
        self.var = VarAttn(V, d, HEADS, NVAR, DROP)               # iTransformer変数attention(補助路)
        self.tcn = TCN(d, TCN_LV, DROP)
        self.norm = nn.LayerNorm(d)
        # mode: sel=バンク選択(既定, PF要) / dist=δ分布 / reg=δ回帰
        self.mode = "sel" if (os.environ.get("HEAD_SEL", "1") == "1" and USE_PF) else ("dist" if os.environ.get("HEAD_DIST", "0") == "1" else "reg")
        odim = (K + 1) if self.mode == "sel" else (Dout if self.mode == "dist" else 1)   # sel: K重み + 1補正(ハイブリッド)
        self.head = nn.Linear(d, odim)
        if self.mode in ("sel", "reg"):                           # ★ゼロ初期化: sel=一様重み(=バンク平均)/reg=PF素通し から開始(安定)
            nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)
    def forward(self, x, delta):              # x[B,T,V]
        h = self.inp(x) + self.var(x)
        h = self.norm(h + self.tcn(h))
        if self.mode == "dist":
            return (torch.softmax(self.head(h), -1) * delta[None, None, :]).sum(-1)
        if self.mode == "sel":
            return self.head(h)                                   # [B,T,K] バンクlogits(softmaxはrun)
        return torch.tanh(self.head(h)).squeeze(-1)              # δ回帰 raw[-1,1]


# ============================ 学習 / CV ============================
def run(model, bt, delta):
    x = build_feats(bt, delta)
    out = model(x, delta)
    if model.mode == "sel":                                  # ★選択+補正ハイブリッド: TVT = Σ w_k·pf_k + 補正
        banks = bt["pf_banks"].permute(0, 2, 1)              # [B,T,K]
        w = torch.softmax(out[..., :-1], -1)                 # per-row バンク加重
        corr = torch.tanh(out[..., -1]) * CORR               # 範囲外への補正(±CORR)
        return (w * banks).sum(-1) + corr
    if model.mode == "dist":
        return bt["center"] + out
    if USE_PF:
        return bt["pf_med"] + CORR * out                     # δ回帰(複数PF残差, center相殺)
    return bt["center"] + float(DR) * out

def train_fold(wtr, wva, delta, V, K):
    model = ITr(V, len(delta), K).to(DEVICE); opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    batches = g68._len_buckets(wtr, BATCH); rng = np.random.RandomState(SEED); order = np.arange(len(batches))
    for ep in range(EPOCHS):
        model.train(); rng.shuffle(order); tot = 0.0; nb = 0; t0 = time.time()
        for bi in order:
            bt = collate([wtr[j] for j in batches[bi]], DEVICE)
            tvt = run(model, bt, delta)
            loss = (((tvt - bt["tgt"]) ** 2) * bt["mask"]).sum() / bt["mask"].sum().clamp_min(1)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
            tot += loss.item(); nb += 1
        print(f"    ep{ep+1}/{EPOCHS} mse={tot/max(nb,1):.3f}(rmse~{math.sqrt(tot/max(nb,1)):.3f}) {time.time()-t0:.0f}s", flush=True)
    model.eval(); oof = {}
    with torch.no_grad():
        for bidx in g68._len_buckets(wva, BATCH):
            batch = [wva[j] for j in bidx]; bt = collate(batch, DEVICE); tvt = run(model, bt, delta)
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
    # ★複数PF(multipf_s{stride}.pkl = v28 の9バンク等)を付与。基準=median / 食い違い=硬いシグナル。欠けは錨。
    _pfp = OUT / f"multipf_s{EVAL_STRIDE}.pkl"
    _pfby = pickle.loads(_pfp.read_bytes()) if _pfp.exists() else {}
    K_PF = _pfby[next(iter(_pfby))]["ancc"].shape[0] if _pfby else 1
    _npf = 0
    for w in wells:
        n = len(w["hgr"]); d = _pfby.get(w["wid"]); ctr = np.asarray(w["center"], float)
        if d is not None and d["ancc"].shape[1] >= n:
            A = np.asarray(d["ancc"][:, :n], float)                 # [K, n] 各PFバンク
            w["pf_med"] = np.nanmedian(A, 0); w["pf_banks"] = A     # median=基準 / 各バンク=ch
            w["pf_disagree"] = np.nanstd(A, 0); w["pf_range"] = np.nanmax(A, 0) - np.nanmin(A, 0)
            _npf += 1
        else:
            w["pf_med"] = ctr.copy(); w["pf_banks"] = np.tile(ctr, (K_PF, 1))
            w["pf_disagree"] = np.zeros(n); w["pf_range"] = np.zeros(n)
    if USE_PF and not _pfby:
        print(f"★PF=on だが multipf_s{EVAL_STRIDE}.pkl 無し → 先に `EVAL_STRIDE={EVAL_STRIDE} python v70/build_multipf_v70.py`")
    print(f"[v70] 複数PF付き {_npf}/{len(wells)}  ({K_PF}バンク median基準, PF入力={'on' if USE_PF else 'off'})")
    delta = torch.arange(-DR, DR + 1e-6, DSTEP, device=DEVICE, dtype=torch.float32)
    V = build_feats(collate(wells[:2], DEVICE), delta).shape[-1]
    _mode = "sel(バンク選択)" if (os.environ.get("HEAD_SEL", "1") == "1" and USE_PF) else ("dist" if os.environ.get("HEAD_DIST", "0") == "1" else "reg(PF残差)")
    print(f"[v70 iTr] wells {len(wells)} CV={CV_MODE} STRUCT={int(USE_STRUCT)} V={V} K={K_PF} mode={_mode} δ={len(delta)}bin d={DMODEL} device={DEVICE}")
    from sklearn.model_selection import GroupKFold
    if CV_MODE == "field":
        fld = g68.make_fields(wells); nf = min(5, len(set(fld))); assign = np.zeros(len(wells), int)
        for k, (_, va) in enumerate(GroupKFold(nf).split(wells, groups=fld)): assign[va] = k
        folds = list(range(nf)); print(f"[cv] leave-field-out {len(set(fld))}fields→{nf}fold")
    else:
        g = np.array([w["wid"] for w in wells]); assign = np.zeros(len(wells), int)
        for k, (_, va) in enumerate(GroupKFold(5).split(wells, groups=g)): assign[va] = k
        folds = list(range(5)); print("[cv] well GroupKFold(5)")
    oof = {}; t0 = time.time()
    for k in folds:
        tr = [wells[i] for i in np.where(assign != k)[0]]; va = [wells[i] for i in np.where(assign == k)[0]]
        print(f"  fold {k}: train {len(tr)} / val {len(va)}", flush=True)
        oof.update(train_fold(tr, va, delta, V, K_PF))
    perw, pool, spool = {}, [], []
    for w in wells:
        if w["wid"] not in oof: continue
        p = oof[w["wid"]]; tg = w["target"]; ct = w["center"]; n = min(len(p), len(tg))
        e = p[:n] - tg[:n]; se = ct[:n] - tg[:n]; fin = np.isfinite(e)
        if fin.any(): perw[w["wid"]] = float(np.sqrt(np.mean(e[fin] ** 2))); pool.append(e[fin]); spool.append(se[fin])
    overall = float(np.sqrt(np.mean(np.concatenate(pool) ** 2)))
    base = float(np.sqrt(np.mean(np.concatenate(spool) ** 2)))
    tag = f"{CV_MODE}_{'st' if USE_STRUCT else 'free'}"
    print(f"\n[v70 iTr] {tag}  overall RMSE = {overall:.4f}  ({'struct' if USE_STRUCT else 'flat'}錨基準 {base:.4f})")
    print(f"          坑井別中央 {np.median(list(perw.values())):.4f}  ({len(perw)}坑井, {time.time()-t0:.0f}s)")
    np.save(OUT / f"itr_oof_{tag}.npy", np.array([v for w in wells if w["wid"] in oof for v in oof[w["wid"]]], np.float32))
    (OUT / f"results_{tag}.json").write_text(json.dumps(dict(overall=overall, base=base, struct=USE_STRUCT,
        per_well_median=float(np.median(list(perw.values()))), n=len(perw), cv=CV_MODE,
        cfg=dict(V=V, dmodel=DMODEL, nvar=NVAR, tcn=TCN_LV, epochs=EPOCHS, eval_stride=EVAL_STRIDE)), indent=2))
    print(f"[saved] itr_oof_{tag}.npy / results_{tag}.json")


if __name__ == "__main__":
    main()
