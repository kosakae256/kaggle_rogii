# -*- coding: utf-8 -*-
"""v93 相補脚: ResNet-MLP タブラーNN. 719特徴(struct-free), GroupKFold5(L1/L2/L5整合), 3seed OOF -> resnet_oof.npz.
木(GBM)と別族の NN で脱相関狙い。FT-Transformer が 5070 で激遅だったため、attention 無しの ResNet-MLP に。

設計:
- PER-ROW 回帰(各評価行1サンプル)。入力=features_l1.json(719, struct-free)から last_known_tvt 除外 -> 718。
  MLP は特徴数に安いので既定 TOPK=0(全718)。TOPK>0 で feat_rank_nn.json 上位選抜も可。
- ResNet-MLP(rtdl系): 入力Linear -> N x [LN -> Linear(d->h) -> GELU -> Drop -> Linear(h->d) -> Drop -> +残差] -> head。
  ★attention なし = 桁違いに速い。float32 で十分(AMP不要)。
- CV = GroupKFold(5).split(zeros(N), y, groups=well) = L1/L2/L5 と完全同一 -> OOF が stack で整列。
- 出力 = artifacts_v93/resnet_oof.npz(res_s0/s1/s2/res_mean/y, per-row, TARGET単位)。resumable(resnet_ckpt/)。
- 実行: cd v93 && ../.venv/Scripts/python.exe train_resnet_v93.py
  SMOKE=1 で 1seed x 1fold x ~40井 x 2ep(動作確認)。env: TOPK/DIM/NBLOCK/HID/DROP/BATCH/SEEDS/EPOCHS 上書き可。

cp932安全: em-dash 不使用(-> )。stdout utf-8。struct-free(parquet + feat_rank_nn.json のみ)。
"""
import os, io, sys, json, math, time, gc, copy, re, warnings, contextlib
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
sys.stderr = io.open(2, "w", encoding="utf-8", closefd=False)
import numpy as np, pandas as pd
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
import pyarrow.parquet as pq
from sklearn.model_selection import GroupKFold
warnings.filterwarnings("ignore")

SMOKE = os.environ.get("SMOKE", "0") == "1"

PROJECT = Path(r"C:\Users\kosaka256\Documents\rogii_claude")
ART = PROJECT / "v93" / "artifacts_v93"
PARQUET = ART / "data" / "train.parquet"
FEATS_JSON = ART / "features_l1.json"
RANK_JSON = ART / "feat_rank_nn.json"
CKPT = ART / "resnet_ckpt"; CKPT.mkdir(parents=True, exist_ok=True)

N_SPLITS = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- モデル(ResNet-MLP)ハイパラ ---
DIM     = int(os.environ.get("DIM", "256"))        # 内部次元 d
N_BLOCK = int(os.environ.get("NBLOCK", "4"))       # 残差ブロック数
HID     = int(os.environ.get("HID", str(2 * DIM))) # ブロック内隠れ幅
DROP    = float(os.environ.get("DROP", "0.15"))
TOPK    = int(os.environ.get("TOPK", "0"))         # 0 = 全特徴(MLPは安いので既定全部)

# --- 学習 ---
SEEDS  = [int(s) for s in os.environ.get("SEEDS", "0,1,2").split(",") if s.strip() != ""]
EPOCHS = int(os.environ.get("EPOCHS", "40"))
LR     = float(os.environ.get("LR", "1e-3"))
WD     = float(os.environ.get("WD", "1e-5"))
BATCH  = int(os.environ.get("BATCH", "8192"))      # MLPは attention 無し -> 大バッチOK(速い)
CLIP   = float(os.environ.get("CLIP", "1.0"))
WARM_FRAC = 0.05
PATIENCE  = int(os.environ.get("PATIENCE", "6"))
EMA_DECAY = float(os.environ.get("EMA", "0.999"))
HUBER_DELTA = float(os.environ.get("HUBER", "1.0"))
# AMP は既定 off(float32)。MLP は float32 で十分速い。
USE_AMP = (DEVICE == "cuda") and (os.environ.get("AMP", "off").lower() in ("bf16", "fp16", "1", "on", "true"))
_AMP_DT = torch.float16 if os.environ.get("AMP", "off").lower() == "fp16" else torch.bfloat16
def amp_ctx():
    return torch.autocast("cuda", dtype=_AMP_DT) if USE_AMP else contextlib.nullcontext()

if SMOKE:
    SEEDS = [0]; EPOCHS = 2; TOPK = 128; SMOKE_WELLS = 40
    print("*** SMOKE MODE: 1seed x 1fold, wells<=%d, TOPK=%d, epochs=%d ***" % (SMOKE_WELLS, TOPK, EPOCHS))

print(f"device={DEVICE} torch={torch.__version__} | SMOKE={SMOKE} | "
      f"ResNet-MLP DIM={DIM} NBLOCK={N_BLOCK} HID={HID} DROP={DROP} TOPK={TOPK} BATCH={BATCH} SEEDS={SEEDS}")

# =====================================================================
# 特徴選択(struct-free, last_known_tvt 除外, 任意で TOPK)
# =====================================================================
features_all = json.loads(FEATS_JSON.read_text())
assert isinstance(features_all, list)
_forbidden = [c for c in features_all if re.search(r"(^gf_|v50b|struct|v38|v41)", c)]
assert not _forbidden, f"struct/gf/v50b 混入: {_forbidden[:10]}"
feats_base = [c for c in features_all if c != "last_known_tvt"]
if TOPK and TOPK > 0:
    rank = json.loads(RANK_JSON.read_text())
    rank = [c for c in rank if c != "last_known_tvt" and c in set(feats_base)]
    feats_nn = rank[:TOPK]
    for c in feats_base:
        if len(feats_nn) >= TOPK:
            break
        if c not in feats_nn:
            feats_nn.append(c)
    print(f"TOPK={TOPK} 選抜 -> 入力 {len(feats_nn)}列")
else:
    feats_nn = feats_base
    print(f"全特徴 -> 入力 {len(feats_nn)}列 (last_known_tvt 除外)")
assert not [c for c in feats_nn if re.search(r"(^gf_|v50b|struct|v38|v41)", c)]

# =====================================================================
# データ読込(メモリ安全: X 事前確保 + 列チャンク)
# =====================================================================
_pq_names = set(pq.read_schema(PARQUET).names)
assert all(c in _pq_names for c in feats_nn)
META = ["well", "target"]


def _load_full():
    N = pq.read_metadata(PARQUET).num_rows
    mt = pq.read_table(PARQUET, columns=META)
    g = mt.column("well").to_pandas().to_numpy()
    y = mt.column("target").to_numpy(zero_copy_only=False).astype(np.float64)
    del mt; gc.collect()
    X = np.empty((N, len(feats_nn)), np.float32)
    CH = 64
    for s in range(0, len(feats_nn), CH):
        cols = feats_nn[s:s + CH]
        t = pq.read_table(PARQUET, columns=cols)
        for jj in range(len(cols)):
            X[:, s + jj] = t.column(jj).to_numpy(zero_copy_only=False).astype(np.float32)
        del t; gc.collect()
    return N, g, y, X


def _load_smoke(nw):
    pf = pq.ParquetFile(PARQUET); need = list(dict.fromkeys(META + feats_nn))
    parts = []; seen = []; done = False
    for b in pf.iter_batches(batch_size=200_000, columns=need):
        d = b.to_pydict(); parts.append(d)
        for w in d["well"]:
            if w not in seen:
                seen.append(w)
                if len(seen) > nw:
                    done = True
        if done:
            break
    keep = set(seen[:nw])
    allw = np.concatenate([np.asarray(p["well"], dtype=object) for p in parts])
    mask = np.array([w in keep for w in allw]); N = int(mask.sum())
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

folds = list(GroupKFold(n_splits=N_SPLITS).split(np.zeros(N), y, groups=g))

# robust 標準化(subsample 統計 + in-place 変換)
samp = np.arange(0, N, max(1, N // 200000)); Xs = X[samp]
med = np.nanmedian(Xs, axis=0).astype(np.float32)
scl = np.maximum((np.nanpercentile(Xs, 75, axis=0) - np.nanpercentile(Xs, 25, axis=0)) / 1.349, 1e-6).astype(np.float32)
del Xs; gc.collect()
X -= med; X /= scl
np.nan_to_num(X, copy=False, nan=0.0, posinf=8.0, neginf=-8.0); np.clip(X, -8.0, 8.0, out=X)

t_mean = float(y.mean()); t_scale = float(y.std())
yn = ((y - t_mean) / t_scale).astype(np.float32)
print(f"rows {N} | F {F_IN} | t_mean/scale {round(t_mean,3)}/{round(t_scale,3)}")


# =====================================================================
# モデル: ResNet-MLP (rtdl系, LayerNorm)
# =====================================================================
class ResBlock(nn.Module):
    """LN -> Linear(d->h) -> GELU -> Drop -> Linear(h->d) -> Drop -> +残差。"""
    def __init__(self, d, h, drop):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.lin1 = nn.Linear(d, h)
        self.lin2 = nn.Linear(h, d)
        self.do = nn.Dropout(drop)

    def forward(self, x):
        h = self.ln(x)
        h = self.do(F.gelu(self.lin1(h)))
        h = self.do(self.lin2(h))
        return x + h


class ResNetMLP(nn.Module):
    def __init__(self, n_feat, d, n_block, h, drop):
        super().__init__()
        self.inp = nn.Linear(n_feat, d)
        self.blocks = nn.ModuleList([ResBlock(d, h, drop) for _ in range(n_block)])
        self.head = nn.Sequential(nn.LayerNorm(d), nn.GELU(), nn.Linear(d, 1))

    def forward(self, x):                    # x: (B, F)
        x = self.inp(x)
        for b in self.blocks:
            x = b(x)
        return self.head(x).squeeze(-1)


class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items() if v.dtype.is_floating_point}
    def update(self, model):
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
    def state_dict(self):
        return {k: v.clone() for k, v in self.shadow.items()}


def new_model():
    return ResNetMLP(F_IN, DIM, N_BLOCK, HID, DROP).to(DEVICE)


_tmp = new_model(); N_PARAM = sum(p.numel() for p in _tmp.parameters())
print(f"model: ResNet-MLP d={DIM} blocks={N_BLOCK} hid={HID} params={N_PARAM:,}")
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
    model.eval(); out = np.empty(len(idx), np.float32)
    for i in range(0, len(idx), BATCH):
        bidx = idx[i:i + BATCH]
        xb = torch.from_numpy(X[bidx]).to(DEVICE)
        with amp_ctx():
            pb = model(xb)
        out[i:i + len(bidx)] = pb.float().cpu().numpy()
    return out


def train_fold(seed, k):
    tr, va = folds[k]; tr = np.asarray(tr); va = np.asarray(va)
    torch.manual_seed(1000 * seed + k); rng = np.random.default_rng(1000 * seed + k)
    model = new_model()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    ema = EMA(model, EMA_DECAY)
    steps_ep = math.ceil(len(tr) / BATCH); total = steps_ep * EPOCHS; warm = int(total * WARM_FRAC); step = 0
    best = (1e18, None); bad = 0
    for ep in range(EPOCHS):
        ep_t0 = time.time(); model.train(); perm = rng.permutation(len(tr))
        for i in range(0, len(perm), BATCH):
            bidx = tr[perm[i:i + BATCH]]
            xb = torch.from_numpy(X[bidx]).to(DEVICE); yb = torch.from_numpy(yn[bidx]).to(DEVICE)
            for pgp in opt.param_groups:
                pgp["lr"] = LR * cosine_lr(step, total, warm)
            with amp_ctx():
                pred = model(xb)
            loss = F.huber_loss(pred.float(), yb, delta=HUBER_DELTA)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            opt.step(); ema.update(model); step += 1
            if SMOKE and step >= 4:
                break
        if not torch.isfinite(loss).item():
            raise RuntimeError(f"loss 非有限 (seed{seed} fold{k} ep{ep})")
        bak = copy.deepcopy(model.state_dict()); sd = model.state_dict()
        for kk, vv in ema.state_dict().items():
            sd[kk].copy_(vv)
        vp = predict_rows(model, va) * t_scale + t_mean
        rm = float(np.sqrt(np.mean((vp - y[va]) ** 2)))
        if not np.isfinite(rm):
            raise RuntimeError(f"val RMSE 非有限 (seed{seed} fold{k} ep{ep})")
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
# 学習ループ(seed x fold, resumable)
# =====================================================================
folds_to_run = [0] if SMOKE else list(range(N_SPLITS))
seed_oofs = {}
_run_t0 = time.time(); _folds_total = len(SEEDS) * len(folds_to_run); _folds_done = 0
print(f"[resnet] AMP={'on' if USE_AMP else 'off(float32)'} total {_folds_total} folds", flush=True)
for seed in SEEDS:
    oof = np.zeros(N, np.float64); filled = np.zeros(N, bool); tt = time.time()
    for k in folds_to_run:
        cache = CKPT / f"res_s{seed}_f{k}_valpred.npz"
        if cache.exists() and not SMOKE:
            z = np.load(cache); oof[z["rows"]] = z["pred"]; filled[z["rows"]] = True
            print(f"seed{seed} fold{k} [load cache] val={len(z['rows'])}", flush=True); continue
        vp, rm, va = train_fold(seed, k)
        oof[va] = vp; filled[va] = True
        if not SMOKE:
            np.savez_compressed(cache, rows=va.astype(np.int64), pred=vp.astype(np.float64))
        _folds_done += 1; _el = time.time() - _run_t0
        _eta = _el / max(_folds_done, 1) * (_folds_total - _folds_done)
        print(f"seed{seed} fold{k} [fit] val={len(va)} rmse={rm:.4f}  "
              f"[{_folds_done}/{_folds_total} fold  経過{_el/60:.0f}分  残りETA~{_eta/60:.0f}分]", flush=True)
    seed_oofs[seed] = oof
    if not SMOKE:
        rm = float(np.sqrt(np.mean((oof[filled] - y[filled]) ** 2)))
        print(f"== seed{seed} OOF pooled RMSE(target) = {rm:.4f} (filled {int(filled.sum())}/{N}) "
              f"({time.time()-tt:.0f}s)", flush=True)

# =====================================================================
# OOF 保存 + 相補性(GBM/TCN/L5 と blend)判定
# =====================================================================
if not SMOKE:
    keys = {f"res_s{s}": seed_oofs[s] for s in SEEDS}
    mean_oof = np.mean([seed_oofs[s] for s in SEEDS], axis=0)
    np.savez_compressed(ART / "resnet_oof.npz", **keys, res_mean=mean_oof, y=y)
    filled_all = np.ones(N, bool) if len(SEEDS) else np.zeros(N, bool)
    rmse = lambda p: float(np.sqrt(np.mean((p - y) ** 2)))
    print(f"\n=== ResNet-MLP OOF ===")
    for s in SEEDS:
        print(f"  seed{s} = {rmse(seed_oofs[s]):.4f}")
    print(f"  {len(SEEDS)}seed ensemble = {rmse(mean_oof):.4f}", flush=True)
    # 相補性: 全fold埋まっている時のみ(部分実行では skip)
    try:
        gbm = np.load(ART / "gbm_l1_oof.npy").astype(np.float64)
        nnz = np.load(ART / "nn_oof_combo.npz"); tcn = np.mean([nnz[k] for k in nnz.files if k.startswith("mdn2")], 0).astype(np.float64)
        l5 = np.load(ART / "l5_oof.npz")["l5_mean"].astype(np.float64)
        from scipy.optimize import nnls
        legs = {"gbm": gbm, "tcn": tcn, "l5": l5, "resnet": mean_oof}
        print("\n=== 相補性(全脚 OOF, 全行) ===")
        for nm in ("gbm", "tcn", "l5"):
            print(f"  corr(resnet,{nm})={np.corrcoef(mean_oof, legs[nm])[0,1]:.3f}  {nm}単体={rmse(legs[nm]):.4f}")
        M3 = np.stack([gbm, tcn, l5], 1); w3, _ = nnls(M3, y); r3 = rmse(M3 @ w3)
        M4 = np.stack([gbm, tcn, l5, mean_oof], 1); w4, _ = nnls(M4, y); r4 = rmse(M4 @ w4)
        print(f"  3-way{{gbm,tcn,l5}} nnls = {r3:.4f}")
        print(f"  4-way +resnet nnls = {r4:.4f}  重み={np.round(w4,3)}  利得={r4-r3:+.4f}")
        print(f"  VERDICT: {'相補的(効く)' if (r4 < r3 - 1e-4 and w4[3] > 1e-3) else '相補性乏しい'}", flush=True)
    except Exception as e:
        print(f"[相補性] 全脚OOF未揃い/部分実行のため skip ({type(e).__name__})", flush=True)
print("done.", flush=True)
