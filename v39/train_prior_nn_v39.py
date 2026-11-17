# -*- coding: utf-8 -*-
"""
v39 構造prior付き系列NN(v29b SeqNet+mdn2 を土台に、GRフリー幾何priorを注入)
==============================================================================
v38の構造事前分布 p(TVT|幾何)=N(struct_tvt, sig_struct^2) を per-well TCN(v29b)に食わせる。
prefix/mtp は落とし(priorが実験対象)、v29bの SeqNet(TCN Block+SelfAttn)+ mdn2 を踏襲。

3パターン(PATTERN で切替。P2 は P1 の重みを再利用しreadoutだけ変える=学習は2回):
  p0_control : target=TVT-last_known, 構造ch無し, prior損失無し, mdn平均readout   … priorのlift基準
  p1_prior   : target=TVT-struct(残差化), +struct_res/sig の2ch, prior損失, mdn平均readout
  p2_tiebrk  : p1と同一重み。readoutを prior重み付き w_k∝pi_k·exp(-0.5(mu_k/sig)^2) に(datum両義のタイブレーク)

依存: v38/artifacts_v38/struct_prior_train.pkl(先に struct_prior_v38.py 実行)/ v29 gbm_oof・feat_rank / v28 parquet。
評価: overall(pp後)と 硬い核(v32 hardcore_diag)を両方。priorの価値はtailに集中するので硬い核を必ず見る。
実行: rogii_claude/.venv/Scripts/python.exe v39/train_prior_nn_v39.py
      PATTERN を p0→p1→p2 の順に書き換えて3回(p2はp1のチェックポイントを読むのでp1後に実行)。
"""
import os, json, math, time, gc, copy, glob, warnings, pickle
from pathlib import Path
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.model_selection import GroupKFold
warnings.filterwarnings("ignore")

# ============================== CONFIG ==============================
PROJECT = Path(r"C:\Users\kosaka256\Documents\rogii_claude")
V28_ART = PROJECT / "v28" / "artifacts_v28"
V29_ART = PROJECT / "v29" / "artifacts_v29"
ART     = PROJECT / "v39" / "artifacts_v39"; (ART / "nn").mkdir(parents=True, exist_ok=True)
PARQUET = V28_ART / "data" / "train.parquet"
STRUCT_PKL = PROJECT / "v38" / "artifacts_v38" / "struct_prior_train.pkl"
HARDCORE_CSV = PROJECT / "v32" / "artifacts_v32" / "hardcore_diag.csv"

PATTERN   = "c_r014"            # ★ p0_control / p1_prior / p2_tiebrk / c_r014(=p1 + r0/r1/r4特徴)
LAMBDA_PRIOR = 0.05               # prior損失 λ*mean((r_hat/sig)^2)(p1/p2のみ)
FLIP      = True                  # 反転aug + TTA(v29b b_flip 相当)
N_SPLITS  = 5
TOP_K     = 192
SEEDS     = [0, 1]                # v29は[0,1,2]。labは2で。
EPOCHS    = 50
LR, WD, BATCH_WELLS, CROP = 3e-4, 1e-4, 16, 1024
NOISE, EMA_DECAY, CLIP, WARM_FRAC = 0.02, 0.999, 1.0, 0.05
EVAL_TOK  = 60_000
# --- model(v29b) ---
CH, KSIZE, DROP = 192, 5, 0.10
DILATIONS = [1, 2, 4, 8, 16, 32, 64, 128]
USE_ATTN, ATTN_HEADS, CH_DROP = True, 4, 0.10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

USE_STRUCT = PATTERN in ("p1_prior", "p2_tiebrk", "c_r014")
TIEBREAK   = PATTERN == "p2_tiebrk"
USE_RBANK  = PATTERN == "c_r014"   # ★c: r0/r1/r4_seed32 の軌跡特徴を v28_4 parquet から追加
CKPT_TAG   = ("ctrl" if PATTERN == "p0_control" else "prior_r" if USE_RBANK else "prior")   # p1/p2 は重み共有
# v28_4 parquet は同一列名の中身が r0/r1/r4(RCOL対応: pf_ancc_1/2/3 = r0/r1/r4(tw), self/nbr同様)
PARQUET_R = PROJECT / "v28" / "artifacts_v28_4" / "data" / "train.parquet"
R_SRC = ["pf_ancc_1", "pf_ancc_2", "pf_ancc_3", "pf_self_1", "pf_self_2", "pf_self_3",
         "pf_nbr_1", "pf_nbr_2", "pf_nbr_3",
         "pf_self_vs_tw_1", "pf_self_vs_tw_2", "pf_self_vs_tw_3",
         "pf_nbr_vs_tw_1", "pf_nbr_vs_tw_2", "pf_nbr_vs_tw_3",
         "pf_agree_std_1", "pf_agree_std_2", "pf_agree_std_3"]
R_NAMES = ["r_" + c[3:] for c in R_SRC]   # r_ancc_1(=r0 tw) など。名前衝突回避
FORCE_FEATS = ["md_since", "pf_ancc", "pf_ancc_2", "pf_ancc_3",
               "pf_self", "pf_self_2", "pf_self_3", "pf_nbr", "pf_nbr_2", "pf_nbr_3",
               "pf_self_vs_tw", "pf_self_vs_tw_2", "pf_self_vs_tw_3",
               "pf_nbr_vs_tw", "pf_nbr_vs_tw_2", "pf_nbr_vs_tw_3",
               "pf_self_mean", "pf_nbr_mean", "pf_self_pstd", "pf_nbr_pstd",
               "pf_self_mean_vs_tw", "pf_nbr_mean_vs_tw",
               "pf_agree_std", "pf_agree_std_2", "pf_agree_std_3",
               "pf_nbr_has", "pf_nbr_nndist", "pf_nbr_nrefs", "z_grad"]
pp_params = (json.loads((V28_ART / "pp_params.json").read_text())
             if (V28_ART / "pp_params.json").exists() else {"alpha": 1.0, "tau": 85, "w_pf": 0.09})
print(f"PATTERN={PATTERN}  USE_STRUCT={USE_STRUCT} TIEBREAK={TIEBREAK} FLIP={FLIP} | device={DEVICE}")

# ============================== FEATURES ==============================
features_all = json.loads((V28_ART / "features.json").read_text())
rank = json.loads((V29_ART / "feat_rank.json").read_text())
rank = [c for c in rank if c != "last_known_tvt"]
feats_nn = [c for c in FORCE_FEATS if c in features_all]
for c in rank:
    if len(feats_nn) >= TOP_K: break
    if c not in feats_nn: feats_nn.append(c)
print("NN features:", len(feats_nn))

# ============================== DATA ==============================
import pyarrow.parquet as pq
need = list(dict.fromkeys(["well", "target", "last_known_tvt", "md_since", "pf_ancc"] + feats_nn))
tbl = pq.read_table(PARQUET, columns=need)
N = tbl.num_rows
g   = tbl.column("well").to_pandas().to_numpy()
y   = tbl.column("target").to_numpy(zero_copy_only=False).astype(np.float64)
lk  = tbl.column("last_known_tvt").to_numpy(zero_copy_only=False).astype(np.float64)
mds = tbl.column("md_since").to_numpy(zero_copy_only=False).astype(np.float64)
pf_abs = tbl.column("pf_ancc").to_numpy(zero_copy_only=False).astype(np.float64)
NB = len(feats_nn); NR_EXT = len(R_SRC) if USE_RBANK else 0
Xf = np.empty((N, NB + NR_EXT), np.float32)      # ★c: r列ぶんも最初から確保(concatコピー回避)
for j, c in enumerate(feats_nn):
    Xf[:, j] = tbl.column(c).to_numpy(zero_copy_only=False).astype(np.float32)
del tbl; gc.collect()
if USE_RBANK:
    # v28_4(行順一致は検証済)から r0/r1/r4 軌跡・派生を追加。列名は r_ 接頭で衝突回避
    tbl4 = pq.read_table(PARQUET_R, columns=R_SRC)
    assert tbl4.num_rows == N, (tbl4.num_rows, N)
    for j, c in enumerate(R_SRC):
        Xf[:, NB + j] = tbl4.column(c).to_numpy(zero_copy_only=False).astype(np.float32)
    del tbl4; gc.collect()
    feats_nn = feats_nn + R_NAMES
    print(f"c_r014: r0/r1/r4 特徴 {NR_EXT} 列追加 → 計 {len(feats_nn)}")

folds = list(GroupKFold(n_splits=N_SPLITS).split(np.zeros(N), y, groups=g))
samp = np.arange(0, N, max(1, N // 120000))
abs2res = []
for j in range(Xf.shape[1]):
    v = Xf[samp, j].astype(np.float64); m = np.isfinite(v)
    ok = m.sum() > 100 and np.nanstd(v[m]) > 1e-9
    c_ = float(np.corrcoef(v[m], lk[samp][m])[0, 1]) if ok else 0.0
    a = bool(c_ > 0.98); abs2res.append(a)
    if a: Xf[:, j] = Xf[:, j] - lk.astype(np.float32)
med = np.nanmedian(Xf, axis=0); q75 = np.nanpercentile(Xf, 75, axis=0); q25 = np.nanpercentile(Xf, 25, axis=0)
scl = np.maximum((q75 - q25) / 1.349, 1e-6)
Xf = np.clip(np.nan_to_num((Xf - med) / scl, nan=0.0, posinf=8.0, neginf=-8.0), -8.0, 8.0).astype(np.float32)

gbm_oof = np.load(V29_ART / "gbm_oof.npy")
wid_order = pd.unique(g)
well_rows = {}
for w in wid_order:
    idx = np.where(g == w)[0]
    well_rows[w] = idx[np.argsort(mds[idx], kind="stable")]   # MD順(=構造prior ev順)
fold_wells = [(pd.unique(g[tr]), pd.unique(g[va])) for tr, va in folds]

# ---- 構造prior を全行に整列(well_rows=MD順に代入) ----
STRUCT = pickle.loads(STRUCT_PKL.read_bytes()) if STRUCT_PKL.exists() else {}
struct_res = np.zeros(N, np.float64)          # struct_tvt - last_known(残差空間)
sig = np.full(N, 1e9, np.float64)             # 欠損=1e9(prior無効)
n_anch = 0
for w in wid_order:
    r = well_rows[w]; sp = STRUCT.get(w)
    if sp is not None and len(sp["tvt"]) == len(r):
        struct_res[r] = np.asarray(sp["tvt"], np.float64) - lk[r]
        sig[r] = np.asarray(sp["sig"], np.float64); n_anch += 1
print(f"struct aligned: {n_anch}/{len(wid_order)} wells (欠損は prior無効)")
real = sig < 1e8

# ---- アクティブ target(p0=from-lk, p1/p2=from-struct) ----
if USE_STRUCT:
    yt = y - struct_res            # residual-from-struct(GR補正ぶんだけ)
else:
    yt = y.copy()
t_mean = float(yt.mean()); t_scale = float(yt.std())
yn = ((yt - t_mean) / t_scale).astype(np.float32)

# ---- 入力チャネル: base feats (+ struct_res, sig を p1/p2 で2ch追加) ----
if USE_STRUCT:
    s_med = float(np.median(struct_res[real])) if real.any() else 0.0
    s_scl = float(max(np.subtract(*np.percentile(struct_res[real], [75, 25])) / 1.349, 1e-6)) if real.any() else 1.0
    sr_ch = np.clip((struct_res - s_med) / s_scl, -8, 8).astype(np.float32)
    sg_ch = np.where(real, sig, np.nan)
    sg_med = float(np.nanmedian(sg_ch)); sg_iqr = float(max(np.nanpercentile(sg_ch, 75) - np.nanpercentile(sg_ch, 25), 1e-6))
    sg_ch = np.clip(np.nan_to_num((sg_ch - sg_med) / sg_iqr, nan=8.0), -8, 8).astype(np.float32)
    X = np.concatenate([Xf, sr_ch[:, None], sg_ch[:, None]], axis=1)
else:
    X = Xf
K_IN = X.shape[1]
sig_f = np.clip(sig, 1e-3, 1e9).astype(np.float32)   # prior損失/tiebreak用の生sig(ft)
print(f"rows {N} | wells {len(wid_order)} | K_IN {K_IN} | t_mean/scale {t_mean:.3f}/{t_scale:.3f}")

# ============================== MODEL(v29b SeqNet) ==============================
class ChanLN(nn.Module):
    def __init__(self, c): super().__init__(); self.ln = nn.LayerNorm(c)
    def forward(self, x): return self.ln(x.transpose(1, 2)).transpose(1, 2)
class Block(nn.Module):
    def __init__(self, c, dil, k, drop):
        super().__init__()
        self.conv = nn.Conv1d(c, c, k, padding=dil * (k - 1) // 2, dilation=dil)
        self.ln = ChanLN(c); self.pw = nn.Conv1d(c, c, 1); self.do = nn.Dropout(drop)
    def forward(self, x, m):
        h = self.conv(x * m); h = F.gelu(h); h = self.ln(h); h = self.do(self.pw(h))
        return (x + h) * m
class SelfAttn(nn.Module):
    def __init__(self, ch, heads, drop):
        super().__init__(); self.h = heads; self.dk = ch // heads; self.p = drop
        self.qkv = nn.Linear(ch, 3 * ch); self.proj = nn.Linear(ch, ch)
    def forward(self, x, kmask):
        B, L, C = x.shape
        q, k, v = self.qkv(x).chunk(3, -1)
        q = q.view(B, L, self.h, self.dk).transpose(1, 2); k = k.view(B, L, self.h, self.dk).transpose(1, 2)
        v = v.view(B, L, self.h, self.dk).transpose(1, 2)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=kmask[:, None, None, :],
                                           dropout_p=self.p if self.training else 0.0)
        return self.proj(o.transpose(1, 2).reshape(B, L, C))
class SeqNet(nn.Module):
    def __init__(self, k_in, ch, ksize, dilations, drop, use_attn, attn_heads, ch_drop):
        super().__init__()
        self.inp = nn.Conv1d(k_in, ch, 1); self.indrop = nn.Dropout1d(ch_drop)
        self.blocks = nn.ModuleList([Block(ch, d, ksize, drop) for d in dilations])
        self.use_attn = use_attn
        if use_attn: self.attn = SelfAttn(ch, attn_heads, drop); self.aln = nn.LayerNorm(ch)
        self.out = nn.Conv1d(ch, 5, 1)   # mdn2: mu1,mu2,logit_pi,ls1,ls2
    def forward(self, x, m):
        h = self.inp(self.indrop(x)) * m
        for b in self.blocks: h = b(h, m)
        if self.use_attn:
            q = self.aln(h.transpose(1, 2)); a = self.attn(q, (m[:, 0, :] > 0.5))
            h = (h + a.transpose(1, 2)) * m
        return self.out(h)

def mdn_params(out):
    mu1, mu2 = out[:, 0], out[:, 1]; pl = torch.sigmoid(out[:, 2])
    ls1 = out[:, 3].clamp(-3.0, 2.0); ls2 = out[:, 4].clamp(-3.0, 2.0)
    return mu1, mu2, pl, ls1, ls2

def mdn_loss(out, yb, mb, sg):
    mu1, mu2, pl, ls1, ls2 = mdn_params(out)
    def logN(mu, ls): return -0.5 * ((yb - mu) / ls.exp()) ** 2 - ls - 0.5 * math.log(2 * math.pi)
    nll = -torch.logsumexp(torch.stack([torch.log(pl + 1e-6) + logN(mu1, ls1),
                                        torch.log(1 - pl + 1e-6) + logN(mu2, ls2)]), dim=0)
    mean = pl * mu1 + (1 - pl) * mu2
    loss = (nll * mb).sum() / mb.sum() + 0.1 * ((mean - yb) ** 2 * mb).sum() / mb.sum()
    if USE_STRUCT and LAMBDA_PRIOR > 0:                # prior: 幾何(=0)からの逸脱を sig で正規化して罰
        r_phys = mean * t_scale + t_mean               # residual-from-struct(ft)
        pr = ((r_phys / sg) ** 2 * mb).sum() / mb.sum()
        loss = loss + LAMBDA_PRIOR * pr
    return loss

def readout(out, sg):
    """予測(正規化 target 空間)。p2 は prior重み付きモード選択。"""
    mu1, mu2, pl, ls1, ls2 = mdn_params(out)
    if not TIEBREAK:
        return pl * mu1 + (1 - pl) * mu2
    m1 = mu1 * t_scale + t_mean; m2 = mu2 * t_scale + t_mean     # ft(from-struct)
    lw1 = torch.log(pl + 1e-6) - 0.5 * (m1 / sg) ** 2
    lw2 = torch.log(1 - pl + 1e-6) - 0.5 * (m2 / sg) ** 2
    w1 = torch.sigmoid(lw1 - lw2)                                # softmax over 2
    return w1 * mu1 + (1 - w1) * mu2

class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items() if v.dtype.is_floating_point}
    def update(self, model):
        for k, v in model.state_dict().items():
            if k in self.shadow: self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
    def copy_to(self, model):
        sd = model.state_dict()
        for k, v in self.shadow.items(): sd[k].copy_(v)

def make_batch(wids, train, rng, flip_all=False):
    sel = []
    for w in wids:
        r = well_rows[w]
        if flip_all or (train and FLIP and rng.random() < 0.5): r = r[::-1]
        if train and len(r) > CROP:
            s = int(rng.integers(0, len(r) - CROP + 1)); r = r[s:s + CROP]
        sel.append((w, r))
    Lm = max(len(r) for _, r in sel); B = len(sel)
    xb = np.zeros((B, Lm, K_IN), np.float32); yb = np.zeros((B, Lm), np.float32)
    mb = np.zeros((B, Lm), np.float32); sgb = np.ones((B, Lm), np.float32) * 1e9
    for i, (w, r) in enumerate(sel):
        xb[i, :len(r)] = X[r]; yb[i, :len(r)] = yn[r]; mb[i, :len(r)] = 1.0; sgb[i, :len(r)] = sig_f[r]
    x = torch.from_numpy(xb).to(DEVICE).transpose(1, 2)
    return (x, torch.from_numpy(yb).to(DEVICE), torch.from_numpy(mb).to(DEVICE),
            torch.from_numpy(sgb).to(DEVICE), sel)

@torch.no_grad()
def _predict_once(model, wids, flip):
    model.eval(); out = np.zeros(N, np.float64)
    order = sorted(wids, key=lambda w: len(well_rows[w])); i = 0
    while i < len(order):
        Lm = len(well_rows[order[i]]); j = i + 1
        while j < len(order):
            Lc = max(Lm, len(well_rows[order[j]]))
            if (j - i + 1) * Lc > EVAL_TOK: break
            Lm = Lc; j += 1
        x, _, mb, sgb, sel = make_batch(order[i:j], False, None, flip_all=flip)
        with torch.autocast(DEVICE, dtype=torch.bfloat16, enabled=(DEVICE == "cuda")):
            o = model(x, mb.unsqueeze(1))
        p = readout(o.float(), sgb).cpu().numpy()
        for b, (w, r) in enumerate(sel): out[r] = p[b, :len(r)]
        i = j
    return out

def predict_wells(model, wids):
    p = _predict_once(model, wids, False)
    if FLIP: p = 0.5 * (p + _predict_once(model, wids, True))
    return p

def cosine_lr(step, total, warm):
    if step < warm: return step / max(1, warm)
    t = (step - warm) / max(1, total - warm); return 0.5 * (1.0 + math.cos(math.pi * t))

def train_fold(seed, k):
    trw, vaw = fold_wells[k]
    fp = ART / "nn" / f"{CKPT_TAG}_s{seed}_fold{k}.pt"
    model = SeqNet(K_IN, CH, KSIZE, DILATIONS, DROP, USE_ATTN, ATTN_HEADS, CH_DROP).to(DEVICE)
    if fp.exists():                       # p2 は p1 の重みをそのまま読む(readoutだけ変わる)
        model.load_state_dict(torch.load(fp, map_location=DEVICE))
        return predict_wells(model, list(vaw)), "load"
    torch.manual_seed(1000 * seed + k); rng = np.random.default_rng(1000 * seed + k)
    trw = list(trw); vaw = list(vaw)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    total = math.ceil(len(trw) / BATCH_WELLS) * EPOCHS; warm = int(total * WARM_FRAC)
    ema = EMA(model, EMA_DECAY); best = (1e18, None); step = 0
    va_idx = np.concatenate([well_rows[w] for w in vaw])
    for ep in range(EPOCHS):
        model.train(); order = rng.permutation(len(trw))
        for i in range(0, len(order), BATCH_WELLS):
            x, yb, mb, sgb, _ = make_batch([trw[j] for j in order[i:i + BATCH_WELLS]], True, rng)
            if NOISE: x = x + NOISE * torch.randn_like(x)
            for pg in opt.param_groups: pg["lr"] = LR * cosine_lr(step, total, warm)
            with torch.autocast(DEVICE, dtype=torch.bfloat16, enabled=(DEVICE == "cuda")):
                out = model(x, mb.unsqueeze(1))
            loss = mdn_loss(out.float(), yb, mb, sgb)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP); opt.step(); ema.update(model); step += 1
        bak = copy.deepcopy(model.state_dict()); ema.copy_to(model)
        p = predict_wells(model, vaw)      # readout(p0/p1=mean, p2=tiebreak); 予測は from-active-target 正規化
        pr = p[va_idx] * t_scale + t_mean                       # active-target(ft)
        pr = pr + (struct_res[va_idx] if USE_STRUCT else 0.0)   # → from-lk 残差に戻す
        rm = float(np.sqrt(np.mean((pr - y[va_idx]) ** 2)))
        if not np.isfinite(rm): raise RuntimeError(f"val RMSE non-finite ep={ep}")
        if rm < best[0]: best = (rm, copy.deepcopy(model.state_dict()))
        model.load_state_dict(bak)
    model.load_state_dict(best[1]); torch.save(best[1], fp)
    return predict_wells(model, vaw), f"fit best={best[0]:.4f}"

# ============================== RUN ==============================
def to_fromlk(p_norm, idx=None):           # readout(正規化) → from-lk 残差(ft)。idx指定でsubset対応
    r = p_norm * t_scale + t_mean
    sr = (struct_res if idx is None else struct_res[idx]) if USE_STRUCT else 0.0
    return r + sr

nn_oof = {}
for seed in SEEDS:
    oof = np.zeros(N, np.float64); t0 = time.time()
    for k in range(N_SPLITS):
        p, tag = train_fold(seed, k)
        va = folds[k][1]; oof[va] = p[va]
        gc.collect(); torch.cuda.empty_cache() if DEVICE == "cuda" else None
        pr = to_fromlk(p[va], va)
        print(f"[{PATTERN}] s{seed} fold{k} [{tag}] rmse={np.sqrt(np.mean((pr - y[va]) ** 2)):.4f}", flush=True)
    nn_oof[f"s{seed}"] = to_fromlk(oof)
    print(f"== [{PATTERN}] s{seed} OOF raw rmse={np.sqrt(np.mean((nn_oof[f's{seed}'] - y) ** 2)):.4f} ({time.time()-t0:.0f}s)")
np.savez(ART / f"nn_oof_{PATTERN}.npz", **nn_oof)

# ---- 評価: pp後 overall / 硬い核 / GBMブレンド ----
pf_res = pf_abs - lk; w_pf = float(pp_params.get("w_pf", 0.09))
tau = pp_params.get("tau", 85); alpha = float(pp_params.get("alpha", 1.0))
def apply_pp(res):
    d = res * (1 - w_pf) + pf_res * w_pf
    if tau: d = d * (1.0 - np.exp(-np.maximum(mds, 0.0) / float(tau)))
    return d * alpha
def rmse_on(res, mask=None):
    e = apply_pp(res) - y; e = e if mask is None else e[mask]; e = e[np.isfinite(e)]
    return float(np.sqrt(np.mean(e ** 2)))
try:
    hd = pd.read_csv(HARDCORE_CSV); hard = set(hd[hd["hard"] == True].iloc[:, 0].astype(str))
except Exception:
    hard = set()
hmask = np.array([w in hard for w in g])
ens = np.mean(list(nn_oof.values()), axis=0)
ws = np.arange(0.0, 1.001, 0.02)
scs = [rmse_on((1 - a) * gbm_oof + a * ens) for a in ws]
bi = int(np.argmin(scs)); wb = float(ws[bi]); res_b = (1 - wb) * gbm_oof + wb * ens
rec = dict(pattern=PATTERN, lambda_prior=LAMBDA_PRIOR, flip=FLIP, seeds=len(SEEDS),
           nn_overall=rmse_on(ens), nn_hard=rmse_on(ens, hmask),
           gbm_overall=rmse_on(gbm_oof), gbm_hard=rmse_on(gbm_oof, hmask),
           blend_w=wb, blend_overall=float(scs[bi]), blend_hard=rmse_on(res_b, hmask))
res_fp = ART / "results.json"
allres = json.loads(res_fp.read_text()) if res_fp.exists() else {}
allres[PATTERN] = rec; res_fp.write_text(json.dumps(allres, indent=1))
print(f"\n[{PATTERN}]  NN overall={rec['nn_overall']:.4f} hard={rec['nn_hard']:.4f}")
print(f"[{PATTERN}]  GBM      overall={rec['gbm_overall']:.4f} hard={rec['gbm_hard']:.4f}")
print(f"[{PATTERN}]  blend w={wb:.2f} overall={rec['blend_overall']:.4f} hard={rec['blend_hard']:.4f}")
