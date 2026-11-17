# -*- coding: utf-8 -*-
"""v93 相補脚: FT-Transformer(連続特徴tokenizer+self-attn) タブラーNN. 719特徴(struct-free), GroupKFold5(L1/L2/L5整合), 3seed OOF -> tabnn_oof.npz. 木(GBM)と別族で脱相関狙い.

設計メモ:
- これは PER-ROW 回帰(各評価行が1サンプル)。L2 TCN(系列)とは異なり行を独立に扱う。
- 特徴 = features_l1.json(719列, struct-free = L1 GBM と同一集合)から last_known_tvt を除外 -> 718入力。
  env TOPK(既定256)で feat_rank_nn.json(L1 GBM gain 降順)上位を選抜し attention を O(TOPK^2) で現実的に。
  TOPK=0 で全718列使用。
- FT-Transformer: 連続特徴tokenizer(x_j*W_j+b_j -> (B,F,d))+ [CLS] -> pre-LN Transformer(N blocks,
  MHSA + GELU-FFN)-> CLS出力 -> LayerNorm -> MLP(d->d->1)。
- attention は FLOAT32(bf16 autocast なし)= T4 bf16 math-fallback OOM 回避 + 安定性(MEMORY note)。
- CV = GroupKFold(5).split(zeros(N), y, groups=well) = L1 GBM / L2 TCN / L5 と完全同一 -> OOF が stack で整列。
- 出力 = artifacts_v93/tabnn_oof.npz(tab_s0/s1/s2/tab_mean/y, per-row, TARGET単位)。
- resumable: per-(seed,fold) の val 予測を tabnn/ にキャッシュ。既存はロードして再学習スキップ。
- SMOKE=1: 1seed x 1fold, wells~40, TOPK=64, epochs~2 で forward/backward と oof shape 確認のみ。

cp932 安全: em-dash は使わず '->'。stdout を utf-8 に。struct-free(parquet + feat_rank_nn.json のみ消費)。
"""
import os, io, sys, json, math, time, gc, copy, re, warnings
# --- cp932 環境でも UTF-8 で print できるように ---
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
sys.stderr = io.open(2, "w", encoding="utf-8", closefd=False)

import numpy as np, pandas as pd
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
import pyarrow.parquet as pq
from sklearn.model_selection import GroupKFold
warnings.filterwarnings("ignore")

SMOKE = os.environ.get("SMOKE", "0") == "1"

# =====================================================================
# 設定
# =====================================================================
PROJECT = Path(r"C:\Users\kosaka256\Documents\rogii_claude")
ART     = PROJECT / "v93" / "artifacts_v93"
PARQUET = ART / "data" / "train.parquet"
FEATS_JSON = ART / "features_l1.json"      # struct-free 719列(gf/v50b/struct なし)
RANK_JSON  = ART / "feat_rank_nn.json"     # L1 GBM gain importance 降順
CKPT = ART / "tabnn"
CKPT.mkdir(parents=True, exist_ok=True)

N_SPLITS = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- モデル(FT-Transformer)ハイパラ ---
DIM     = int(os.environ.get("DIM", "96"))       # トークン次元 d
N_BLOCK = int(os.environ.get("NBLOCK", "3"))     # Transformer ブロック数
N_HEAD  = int(os.environ.get("NHEAD", "8"))      # attention ヘッド数
FF_MULT = int(os.environ.get("FFMULT", "2"))     # FFN 隠れ = FF_MULT * d
DROP    = float(os.environ.get("DROP", "0.1"))
TOPK    = int(os.environ.get("TOPK", "128"))     # 0 = 全特徴

# --- 学習 ---
SEEDS  = [int(s) for s in os.environ.get("SEEDS", "0,1,2").split(",") if s.strip() != ""]
EPOCHS = int(os.environ.get("EPOCHS", "40"))
LR     = float(os.environ.get("LR", "1e-3"))
WD     = float(os.environ.get("WD", "1e-5"))
BATCH  = int(os.environ.get("BATCH", "2048"))  # ★bf16(AMP)時はflash attentionでO(B*T*d)=省メモリ→8192可。
#   AMP=off(float32)時は attention 実体化(B*heads*T*T)でOOM注意→TOPK256なら BATCH<=2048。
# ★bf16 autocast: ローカル5070は flash/efficient SDPA で大幅高速&省メモリ。
#   (docstring の float32 固定は Kaggle 推論=T4 の話。ローカル学習は bf16 でOK。重みは float32 推論でも動く)
import contextlib
USE_AMP = (DEVICE == "cuda") and (os.environ.get("AMP", "off").lower() in ("bf16", "1", "on", "true"))
def amp_ctx():
    return torch.autocast("cuda", dtype=torch.bfloat16) if USE_AMP else contextlib.nullcontext()
CLIP   = float(os.environ.get("CLIP", "1.0"))
WARM_FRAC = 0.05
PATIENCE  = int(os.environ.get("PATIENCE", "6"))  # early stop
EMA_DECAY = float(os.environ.get("EMA", "0.999"))
HUBER_DELTA = float(os.environ.get("HUBER", "1.0"))  # 標準化ターゲット上の Huber delta

# --- スモーク縮小 ---
if SMOKE:
    SEEDS = [0]; EPOCHS = 3; TOPK = 64; SMOKE_WELLS = 40
    print("*** SMOKE MODE: 1seed x 1fold, wells<=%d, TOPK=%d, epochs=%d ***"
          % (SMOKE_WELLS, TOPK, EPOCHS))

print(f"device={DEVICE} torch={torch.__version__} | SMOKE={SMOKE} | "
      f"DIM={DIM} NBLOCK={N_BLOCK} NHEAD={N_HEAD} FFMULT={FF_MULT} TOPK={TOPK} SEEDS={SEEDS}")

# =====================================================================
# 特徴選択(struct-free, last_known_tvt 除外, 任意で TOPK 選抜)
# =====================================================================
features_all = json.loads(FEATS_JSON.read_text())
assert isinstance(features_all, list), "features_l1.json は列名リストのはず"
# struct/gf/v50b が紛れていないことを実行時にも確認(struct 使用禁止方針)
_forbidden = [c for c in features_all if re.search(r"(^gf_|v50b|struct|v38|v41)", c)]
assert not _forbidden, f"struct/gf/v50b 系が features_l1.json に混入: {_forbidden[:10]}"

feats_base = [c for c in features_all if c != "last_known_tvt"]   # 718 入力(lk 除外)
if TOPK and TOPK > 0:
    rank = json.loads(RANK_JSON.read_text())
    assert isinstance(rank, list), "feat_rank_nn.json は列名リスト(降順)のはず"
    rank = [c for c in rank if c != "last_known_tvt" and c in set(feats_base)]
    feats_nn = rank[:TOPK]
    # 万一 rank が短い場合の補填(通常 719 -> 発生しない)
    if len(feats_nn) < TOPK:
        for c in feats_base:
            if c not in feats_nn:
                feats_nn.append(c)
            if len(feats_nn) >= TOPK:
                break
    print(f"TOPK={TOPK} 選抜: feat_rank_nn.json 上位 -> 入力 {len(feats_nn)}列")
else:
    feats_nn = feats_base
    print(f"features_l1.json: {len(features_all)}列 (struct-free) -> 入力 {len(feats_nn)}列 (last_known_tvt除外)")

# 再確認: 選抜後にも禁止列が無いこと
assert not [c for c in feats_nn if re.search(r"(^gf_|v50b|struct|v38|v41)", c)], "選抜後に禁止列混入"

# =====================================================================
# データ読込(メモリ安全: X 事前確保 + 列チャンク読み)
# =====================================================================
_pq_names = set(pq.read_schema(PARQUET).names)
assert all(c in _pq_names for c in feats_nn), "全特徴が parquet に存在する前提"
META = ["well", "target"]


def _load_full():
    """全 773 井を列チャンクで X に直接書込(コピー最小化, Arrow buffer は都度解放)。"""
    N = pq.read_metadata(PARQUET).num_rows
    meta_tbl = pq.read_table(PARQUET, columns=META)
    g = meta_tbl.column("well").to_pandas().to_numpy()
    y = meta_tbl.column("target").to_numpy(zero_copy_only=False).astype(np.float64)
    del meta_tbl; gc.collect()
    X = np.empty((N, len(feats_nn)), np.float32)   # TOPK=256 -> ~3.87GB / 718 -> ~10.9GB
    CHUNK = 64
    for s in range(0, len(feats_nn), CHUNK):
        cols = feats_nn[s:s + CHUNK]
        t = pq.read_table(PARQUET, columns=cols)
        for jj in range(len(cols)):
            X[:, s + jj] = t.column(jj).to_numpy(zero_copy_only=False).astype(np.float32)
        del t; gc.collect()
    return N, g, y, X


def _load_smoke(nw):
    """先頭 nw 井のみ batch 読みで軽量ロード(井は連続配置前提)。"""
    pf = pq.ParquetFile(PARQUET)
    need = list(dict.fromkeys(META + feats_nn))
    parts = []; seen = []; done = False
    for batch in pf.iter_batches(batch_size=200_000, columns=need):
        d = batch.to_pydict()
        parts.append(d)
        for w in d["well"]:
            if w not in seen:
                seen.append(w)
                if len(seen) > nw:
                    done = True
        if done:
            break
    keep = set(seen[:nw])
    allw = np.concatenate([np.asarray(p["well"], dtype=object) for p in parts])
    mask = np.array([w in keep for w in allw])
    N = int(mask.sum())
    g = allw[mask]
    y = np.concatenate([np.asarray(p["target"]) for p in parts])[mask].astype(np.float64)
    X = np.empty((N, len(feats_nn)), np.float32)
    for j, c in enumerate(feats_nn):
        X[:, j] = np.concatenate([np.asarray(p[c]) for p in parts])[mask].astype(np.float32)
    return N, g, y, X


t0 = time.time()
N, g, y, X = (_load_smoke(SMOKE_WELLS) if SMOKE else _load_full())
F_IN = X.shape[1]
print(f"loaded: rows={N} wells={len(pd.unique(g))} F={F_IN} ({time.time()-t0:.0f}s)")

# ---- fold(L1 GBM / L2 TCN / L5 と完全同一 = per-row groups=well) ----
folds = list(GroupKFold(n_splits=N_SPLITS).split(np.zeros(N), y, groups=g))

# ---- robust 標準化(統計は subsample で算出, 変換は in-place で 11GB コピー回避) ----
samp = np.arange(0, N, max(1, N // 200000))
Xs = X[samp]
med = np.nanmedian(Xs, axis=0).astype(np.float32)
q75 = np.nanpercentile(Xs, 75, axis=0).astype(np.float32)
q25 = np.nanpercentile(Xs, 25, axis=0).astype(np.float32)
scl = np.maximum((q75 - q25) / 1.349, 1e-6).astype(np.float32)
del Xs; gc.collect()
X -= med                                    # in-place broadcast
X /= scl                                    # in-place
np.nan_to_num(X, copy=False, nan=0.0, posinf=8.0, neginf=-8.0)
np.clip(X, -8.0, 8.0, out=X)                # in-place

# ---- ターゲット標準化(学習安定化。予測は target 単位へ復号) ----
t_mean = float(y.mean()); t_scale = float(y.std())
yn = ((y - t_mean) / t_scale).astype(np.float32)
print(f"rows {N} | F {F_IN} | t_mean/scale {round(t_mean,3)}/{round(t_scale,3)}")


# =====================================================================
# モデル定義: FT-Transformer(連続特徴 tokenizer + [CLS] + pre-LN Transformer)
# =====================================================================
class FeatureTokenizer(nn.Module):
    """各連続特徴 x_j -> token = x_j * W_j + b_j。W,b は (F,d) の学習パラメータ。"""
    def __init__(self, n_feat, d):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_feat, d))
        self.bias = nn.Parameter(torch.empty(n_feat, d))
        nn.init.normal_(self.weight, std=0.02)
        nn.init.normal_(self.bias, std=0.02)

    def forward(self, x):                       # x: (B, F)
        return x.unsqueeze(-1) * self.weight + self.bias   # (B, F, d)


class TransformerBlock(nn.Module):
    """pre-LN: MHSA(SDPA float32) + GELU-FFN, residual + dropout。"""
    def __init__(self, d, n_head, ff_mult, drop):
        super().__init__()
        assert d % n_head == 0, "DIM は NHEAD で割り切れること"
        self.h = n_head; self.dk = d // n_head; self.p = drop
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, ff_mult * d), nn.GELU(),
                                nn.Dropout(drop), nn.Linear(ff_mult * d, d))
        self.do = nn.Dropout(drop)

    def forward(self, x):                        # x: (B, T, d)
        B, T, D = x.shape
        h = self.ln1(x)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        q = q.view(B, T, self.h, self.dk).transpose(1, 2)
        k = k.view(B, T, self.h, self.dk).transpose(1, 2)
        v = v.view(B, T, self.h, self.dk).transpose(1, 2)
        # SDPA を FLOAT32 で(bf16 autocast は使わない = T4 math-fallback OOM 回避 + 安定)
        a = F.scaled_dot_product_attention(q, k, v, dropout_p=self.p if self.training else 0.0)
        a = self.proj(a.transpose(1, 2).reshape(B, T, D))
        x = x + self.do(a)
        x = x + self.do(self.ff(self.ln2(x)))
        return x


class FTTransformer(nn.Module):
    def __init__(self, n_feat, d, n_block, n_head, ff_mult, drop):
        super().__init__()
        self.tok = FeatureTokenizer(n_feat, d)
        self.cls = nn.Parameter(torch.empty(1, 1, d)); nn.init.normal_(self.cls, std=0.02)
        self.blocks = nn.ModuleList([TransformerBlock(d, n_head, ff_mult, drop)
                                     for _ in range(n_block)])
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(),
                                  nn.Dropout(drop), nn.Linear(d, 1))

    def forward(self, x):                        # x: (B, F)
        t = self.tok(x)                          # (B, F, d)
        cls = self.cls.expand(t.shape[0], -1, -1)
        t = torch.cat([cls, t], dim=1)           # (B, F+1, d)
        for blk in self.blocks:
            t = blk(t)
        return self.head(t[:, 0]).squeeze(-1)    # CLS -> (B,)


class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()
                       if v.dtype.is_floating_point}
    def update(self, model):
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
    def state_dict(self):
        return {k: v.clone() for k, v in self.shadow.items()}


def new_model():
    return FTTransformer(F_IN, DIM, N_BLOCK, N_HEAD, FF_MULT, DROP).to(DEVICE)


_tmp = new_model()
N_PARAM = sum(p.numel() for p in _tmp.parameters())
print(f"model: FT-Transformer d={DIM} blocks={N_BLOCK} heads={N_HEAD} ff={FF_MULT}*d "
      f"tokens={F_IN + 1}(CLS+{F_IN}) params={N_PARAM:,}")
del _tmp; gc.collect()
if DEVICE == "cuda":
    torch.cuda.empty_cache()


def cosine_lr(step, total, warm):
    if step < warm:
        return step / max(1, warm)
    t = (step - warm) / max(1, total - warm)
    return 0.5 * (1.0 + math.cos(math.pi * t))


@torch.no_grad()
def predict_rows(model, idx):
    """idx(行indexのndarray)-> 予測(標準化ターゲット単位, ndarray)。"""
    model.eval()
    out = np.empty(len(idx), np.float32)
    for i in range(0, len(idx), BATCH):
        bidx = idx[i:i + BATCH]
        xb = torch.from_numpy(X[bidx]).to(DEVICE)
        with amp_ctx():
            pb = model(xb)
        out[i:i + len(bidx)] = pb.float().cpu().numpy()
    return out


def train_fold(seed, k):
    """(seed,fold)を学習し val 行の予測(TARGET単位)を返す。early stop + EMA。"""
    tr, va = folds[k]
    tr = np.asarray(tr); va = np.asarray(va)
    torch.manual_seed(1000 * seed + k)
    rng = np.random.default_rng(1000 * seed + k)

    model = new_model()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    ema = EMA(model, EMA_DECAY)
    steps_ep = math.ceil(len(tr) / BATCH)
    total = steps_ep * EPOCHS; warm = int(total * WARM_FRAC); step = 0
    best = (1e18, None); bad = 0

    for ep in range(EPOCHS):
        ep_t0 = time.time()
        model.train()
        perm = rng.permutation(len(tr))
        for i in range(0, len(perm), BATCH):
            bidx = tr[perm[i:i + BATCH]]
            xb = torch.from_numpy(X[bidx]).to(DEVICE)
            yb = torch.from_numpy(yn[bidx]).to(DEVICE)
            for pg in opt.param_groups:
                pg["lr"] = LR * cosine_lr(step, total, warm)
            with amp_ctx():
                pred = model(xb)
            loss = F.huber_loss(pred.float(), yb, delta=HUBER_DELTA)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            opt.step(); ema.update(model); step += 1
            if SMOKE and step >= 4:
                break   # スモークは数ステップで打ち切り(forward/backward 確認のみ)
        if not torch.isfinite(loss).item():
            raise RuntimeError(f"loss が非有限 (seed{seed} fold{k} ep{ep})")
        # --- EMA 重みで val 評価 ---
        bak = copy.deepcopy(model.state_dict())
        sd = model.state_dict()
        for kk, vv in ema.state_dict().items():
            sd[kk].copy_(vv)
        vp = predict_rows(model, va) * t_scale + t_mean
        rm = float(np.sqrt(np.mean((vp - y[va]) ** 2)))
        if not np.isfinite(rm):
            raise RuntimeError(f"val RMSE が非有限 (seed{seed} fold{k} ep{ep})")
        if rm < best[0] - 1e-5:
            best = (rm, vp.copy()); bad = 0
        else:
            bad += 1
        model.load_state_dict(bak)
        print(f"  seed{seed} fold{k} ep{ep+1}/{EPOCHS} val={rm:.4f} best={best[0]:.4f} "
              f"bad={bad}/{PATIENCE} ({time.time()-ep_t0:.0f}s/ep)", flush=True)
        if bad >= PATIENCE:
            break

    del model, opt, ema; gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return best[1], best[0], va


# =====================================================================
# 学習ループ: seed x fold(resumable per-(seed,fold) val-pred キャッシュ)
# =====================================================================
folds_to_run = [0] if SMOKE else list(range(N_SPLITS))
seed_oofs = {}
_run_t0 = time.time(); _folds_total = len(SEEDS) * len(folds_to_run); _folds_done = 0
print(f"[tabnn] AMP={'bf16' if USE_AMP else 'off(float32)'} BATCH={BATCH} EPOCHS={EPOCHS} "
      f"total {_folds_total} folds(={len(SEEDS)}seed x {len(folds_to_run)}fold)", flush=True)
for seed in SEEDS:
    oof = np.zeros(N, np.float64); filled = np.zeros(N, bool); tt = time.time()
    for k in folds_to_run:
        cache = CKPT / f"tab_s{seed}_f{k}_valpred.npz"
        if cache.exists() and not SMOKE:
            z = np.load(cache)
            oof[z["rows"]] = z["pred"]; filled[z["rows"]] = True
            print(f"seed{seed} fold{k} [load cache] val={len(z['rows'])}", flush=True)
            continue
        vp, rm, va = train_fold(seed, k)
        oof[va] = vp; filled[va] = True
        if not SMOKE:
            np.savez_compressed(cache, rows=va.astype(np.int64), pred=vp.astype(np.float64))
        _folds_done += 1; _el = time.time() - _run_t0
        _eta = _el / max(_folds_done, 1) * (_folds_total - _folds_done)
        print(f"seed{seed} fold{k} [fit] val={len(va)} rmse={rm:.4f}  "
              f"[{_folds_done}/{_folds_total} fold  経過{_el/60:.0f}分  残りETA~{_eta/60:.0f}分]", flush=True)
    seed_oofs[seed] = oof
    if SMOKE:
        va0 = folds[0][1]
        rm0 = float(np.sqrt(np.mean((oof[va0] - y[va0]) ** 2)))
        print(f"== SMOKE seed{seed} fold0-val rmse={rm0:.4f} oof.shape={oof.shape} "
              f"filled={int(filled.sum())} ({time.time()-tt:.0f}s)")
    else:
        rm = float(np.sqrt(np.mean((oof[filled] - y[filled]) ** 2)))
        print(f"== seed{seed} OOF pooled RMSE(target) = {rm:.4f} "
              f"(filled {int(filled.sum())}/{N}) ({time.time()-tt:.0f}s)", flush=True)

# =====================================================================
# 保存(tabnn_oof.npz)+ 3seed アンサンブル pooled RMSE
# =====================================================================
save = {f"tab_s{s}": seed_oofs[s].astype(np.float32) for s in SEEDS}
mean_oof = np.mean([seed_oofs[s] for s in SEEDS], axis=0)
save["tab_mean"] = mean_oof.astype(np.float32)
save["y"] = y.astype(np.float32)

if SMOKE:
    out_fp = ART / "tabnn_oof_SMOKE.npz"
    np.savez_compressed(out_fp, **save)
    print(f"saved {out_fp.name} keys={list(save.keys())} tab_s0.shape={save['tab_s'+str(SEEDS[0])].shape}")
    print("done (SMOKE).")
else:
    out_fp = ART / "tabnn_oof.npz"
    np.savez_compressed(out_fp, **save)
    print(f"saved {out_fp.name} keys={list(save.keys())}")
    for s in SEEDS:
        print(f"  seed{s} OOF pooled RMSE(target) = {np.sqrt(np.mean((seed_oofs[s]-y)**2)):.4f}")
    rm_ens = float(np.sqrt(np.mean((mean_oof - y) ** 2)))
    print(f"== {len(SEEDS)}-seed ensemble OOF pooled RMSE(target) = {rm_ens:.4f}")
    print("done.")
