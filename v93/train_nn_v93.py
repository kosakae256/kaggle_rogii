# -*- coding: utf-8 -*-
"""v93 L2: 系列NN(TCN+SelfAttn+PrefixEnc, mdn2 combo). struct-free features_l1.json消費. L1と同一GroupKFold5. GBMとblend用OOF(nn_oof_combo.npz)

v52 train_nn_v52.ipynb の忠実移植版(struct-free化)。
- v52 は features.json(781列, gf/v50b含む)から LightGBM importance で TOP_K=192 を選抜していた。
  v93 は STRUCT-FREE のため features_l1.json(719列, gf/v50b/struct 一切なし = v93 L1 GBM と同一特徴集合)を
  そのまま NN 入力に使う。gf_feature.npy / v50b_feature.npy / struct 系 artifact は一切読まない。
- モデル(SeqNet: 拡張TCN + SelfAttn + PrefixEnc, mdn2 head)/学習(head_loss=MDN NLL+0.1*MSE, MTP補助,
  AdamW, EMA, cosine LR, flip TTA, prefix encoder, CROP, t_scale/t_mean 正規化)は v52 verbatim。
- CV = GroupKFold(5) by well, v52 と同一 = `GroupKFold(5).split(np.zeros(N), y, groups=g)` → L1 GBM と揃う。
- 出力 = artifacts_v93/nn_oof_combo.npz(mdn2_s0/s1/s2, 各 oof*t_scale+t_mean = target単位)。
- 実行前スモーク: 環境変数 SMOKE=1 で 1seed x 1fold, wells~40, epochs~2 だけ走らせ, forward/backward と
  oof shape を確認(重い本学習はしない)。

cp932 安全: em-dash は使わず '->'。stdout を utf-8 に。
"""
import os, io, sys, json, math, time, gc, copy, glob, re, warnings
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
# 設定 + プリセット選択(v52 と同一)
# =====================================================================
PROJECT = Path(r"C:\Users\kosaka256\Documents\rogii_claude")
ART     = PROJECT / "v93" / "artifacts_v93"
# ---- 入力切替(env V93_INPUT) ----
#   "fwdbwd"(既定): train_fwdbwd.parquet + features_l1_fwdbwd.json + feat_rank_nn_fwdbwd.json(GBM fwdbwd 実行が生成)。
#                   逆方向 bpf_ を FORCE_PATTERNS で強制投入して TCN でも確実にテストする。
#   "fwd"        : train.parquet + features_l1.json + feat_rank_nn.json(既存 forward baseline)。
V93_INPUT = os.environ.get("V93_INPUT", "fwdbwd").lower()
_IMAP = {                                          # V93_INPUT -> (parquet 名, 特徴/rank の接尾 INSUF)
    "fwd":           ("train.parquet",               ""),
    "fwdbwd":        ("train_fwdbwd.parquet",        "_fwdbwd"),
    "fwdbwd_anchor": ("train_fwdbwd_anchor.parquet", "_fwdbwd_anchor"),
}
if V93_INPUT not in _IMAP:
    raise SystemExit(f"[v93 L2 TCN] 未知の V93_INPUT={V93_INPUT} (fwd / fwdbwd / fwdbwd_anchor)")
_fn, INSUF = _IMAP[V93_INPUT]
PARQUET = ART / "data" / _fn
FEATS_JSON = ART / (f"features_l1.json" if INSUF == "" else f"features_l1{INSUF}.json")
RANK_JSON = ART / (f"feat_rank_nn.json" if INSUF == "" else f"feat_rank_nn{INSUF}.json")
for _p in (PARQUET, FEATS_JSON, RANK_JSON):
    if not _p.exists():
        raise SystemExit(f"[v93 L2 TCN] 入力欠損: {_p}\n  V93_INPUT={V93_INPUT}。fwdbwd は "
                         f"build_backward_v93.py -> train_l1_gbm_v93.py(V93_INPUT=fwdbwd) の順で生成される。")
DATA    = PROJECT / "rogii-wellbore-geology-prediction"
(ART / "nn").mkdir(parents=True, exist_ok=True)
print(f"[v93 L2 TCN] V93_INPUT={V93_INPUT}  入力={PARQUET.name}  feats={FEATS_JSON.name}  rank={RANK_JSON.name}")

# ---- プリセット(v52 と同一。★combo が v29b 実測ベスト) ----
PRESETS = {
    "base":     dict(flip=False, prefix=False, gbmch=False),
    "b_flip":   dict(flip=True,  prefix=False, gbmch=False),
    "d_prefix": dict(flip=False, prefix=True,  gbmch=False),
    "combo":    dict(flip=True,  prefix=True,  gbmch=False),   # ★採用構成
    "full":     dict(flip=True,  prefix=True,  gbmch=False, mtp=True),
}
RUN = "combo"
CFG = PRESETS[RUN]

N_SPLITS = 5

# --- モデル(v52/v29b と同一) ---
CH = 192; KSIZE = 5; DILATIONS = [1, 2, 4, 8, 16, 32, 64, 128]; DROP = 0.10
USE_ATTN = True; ATTN_HEADS = 4; CH_DROP = 0.10
PRE_LEN = 512; PRE_C = 4; CTX_DIM = 64      # prefix エンコーダ
MTP_OFFS = [4, 16, 64]; MTP_W = 0.1
# --- 学習(v52/v29b と同一) ---
HEADS = ["mdn2"]; SEEDS = [0, 1, 2]
EPOCHS = 60; LR = 3e-4; WD = 1e-4; BATCH_WELLS = 16; CROP = 1024
NOISE = 0.02; EMA_DECAY = 0.999; CLIP = 1.0; WARM_FRAC = 0.05
EVAL_TOK = 60_000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- スモーク時の縮小 ---
if SMOKE:
    SEEDS = [0]; EPOCHS = 2; SMOKE_WELLS = 40
    print("*** SMOKE MODE: 1seed x 1fold, wells<=%d, epochs=%d ***" % (SMOKE_WELLS, EPOCHS))

# pp_params(v93 では未生成。無ければ v52 既定値)
pp_fp = ART / "pp_params.json"
pp_params = (json.loads(pp_fp.read_text()) if pp_fp.exists()
             else {"alpha": 1.0, "tau": 85, "w_pf": 0.09})
print(f"RUN={RUN} {CFG} | device={DEVICE} torch={torch.__version__} | SMOKE={SMOKE}")

# =====================================================================
# 特徴選択(v93: features_l1.json を直接使用 = L1 GBM と同一集合)
#   v52 は importance で TOP_K 選抜していたが, v93 は struct-free の 719 列をそのまま入力にする。
#   last_known_tvt は v52 と同様に NN 入力から除外(残差化で恒常0列になるため)。
# =====================================================================
features_all = json.loads(FEATS_JSON.read_text())
assert isinstance(features_all, list), "features_l1.json は列名リストのはず"
# struct/gf/v50b が紛れていないことを実行時にも確認(struct 使用禁止方針)
_forbidden = [c for c in features_all if re.search(r"(^gf_|v50b|struct|v38|v41)", c)]
assert not _forbidden, f"struct/gf/v50b 系が features_l1.json に混入: {_forbidden[:10]}"
feats_nn = [c for c in features_all if c != "last_known_tvt"]   # v52 と同様 lk を除外

# ★TOP_K env: v52忠実の192選抜(FORCE_PATTERNS強制 + L1 GBM importance上位)。gf/v50bは除外(struct-free)。
#   未指定=719全部(既定)。指定時は出力を SUFFIX で別名にし 719版と共存させる。
#   ★既定=None(全特徴。top-k で絞らない方が精度が高い=ユーザー方針 2026-07-29)。TOP_K=192 env で従来の絞り込みも可。
TOP_K = int(os.environ["TOP_K"]) if os.environ.get("TOP_K") else None
SUFFIX = (f"_top{TOP_K}{INSUF}" if TOP_K else INSUF)   # ★INSUF で fwd/fwdbwd の出力・ckpt を分離(RESUME事故防止)
if TOP_K:
    FORCE_PATTERNS = [                                  # v52 と同一, ただし gf_/v50b_ を除外
        r"pf_(ancc|self|nbr)(_[1-9])?", r"pf_(self|nbr)_vs_tw(_[1-9])?", r"pf_self_vs_nbr(_[1-9])?",
        r"pf_(self|nbr)_(mean|pstd)", r"pf_(self|nbr)_mean_vs_tw", r"pf_agree_std(_[1-9])?",
        r"pf_nbr_(has|nndist|nrefs)", r"md_since", r"z_grad",
    ]
    if V93_INPUT != "fwd":
        # ★逆方向 bpf_ は full-mirror で ~287列ある。全部強制すると TOP_K=192 を超えて破綻するので
        #   「コア(推定/自己近傍/新規=到達誤差・位置・fwd比較)」だけ強制し、残りは importance に任せる。
        FORCE_PATTERNS += [
            r"bpf_ancc(_[1-6])?", r"bpf_ancc_(mean|med|std_between)",
            r"bpf_(self|nbr)(_[1-6])?", r"bpf_(self|nbr)_mean",
            r"bpf_arrival(_err|_abs)?(_[1-6])?", r"bpf_md_from_toe", r"bpf_pos_frac",
            r"bpf_mean_vs_fwd", r"bpf_vs_fwd_[1-6]",
        ]
    rank = json.loads(RANK_JSON.read_text())            # L1 GBM gain importance 降順(fwdbwd は bpf_ 含む)
    rank = [c for c in rank if c != "last_known_tvt"]
    _force_re = [re.compile(p + "$") for p in FORCE_PATTERNS]
    sel = [c for c in features_all if c != "last_known_tvt" and any(r.fullmatch(c) for r in _force_re)]
    n_force = len(sel)
    n_bpf = sum(1 for c in sel if c.startswith("bpf_"))
    for c in rank:
        if len(sel) >= TOP_K:
            break
        if c not in sel:
            sel.append(c)
    feats_nn = sel
    print(f"★TOP_K={TOP_K} 選抜(v52忠実): 強制{n_force}(うち bpf_={n_bpf}) + importance上位 = NN入力{len(feats_nn)}列")
else:
    print(f"features_l1.json: {len(features_all)}列 (struct-free) -> NN入力 {len(feats_nn)}列 (last_known_tvt除外)")

# =====================================================================
# データ読込(メモリ安全: X 事前確保 + 列チャンク読み) + 前処理(v52 と同一) + prefix
# =====================================================================
_pq_names = set(pq.read_schema(PARQUET).names)
assert all(c in _pq_names for c in feats_nn), "全特徴が parquet に存在する前提(v93 は npy 外部特徴なし)"
META = ["well", "target", "last_known_tvt", "md_since", "pf_ancc"]


def _load_full():
    """全 773 井を列チャンクで X に直接書込(コピー最小化, Arrow buffer は都度解放)。"""
    N = pq.read_metadata(PARQUET).num_rows
    meta_tbl = pq.read_table(PARQUET, columns=META)
    g      = meta_tbl.column("well").to_pandas().to_numpy()
    y      = meta_tbl.column("target").to_numpy(zero_copy_only=False).astype(np.float64)
    lk     = meta_tbl.column("last_known_tvt").to_numpy(zero_copy_only=False).astype(np.float64)
    mds    = meta_tbl.column("md_since").to_numpy(zero_copy_only=False).astype(np.float64)
    pf_abs = meta_tbl.column("pf_ancc").to_numpy(zero_copy_only=False).astype(np.float64)
    del meta_tbl; gc.collect()
    X = np.empty((N, len(feats_nn)), np.float32)   # ~10.9GB (718 x 3.78M)
    CHUNK = 64
    for s in range(0, len(feats_nn), CHUNK):
        cols = feats_nn[s:s + CHUNK]
        t = pq.read_table(PARQUET, columns=cols)   # Arrow buffer は cols 分のみ
        for jj, c in enumerate(cols):
            X[:, s + jj] = t.column(jj).to_numpy(zero_copy_only=False).astype(np.float32)
        del t; gc.collect()
    return N, g, y, lk, mds, pf_abs, X


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
    # 連結
    allw = np.concatenate([np.asarray(p["well"], dtype=object) for p in parts])
    mask = np.array([w in keep for w in allw])
    N = int(mask.sum())
    g   = allw[mask]
    def col(name, dt):
        return np.concatenate([np.asarray(p[name]) for p in parts])[mask].astype(dt)
    y      = col("target", np.float64)
    lk     = col("last_known_tvt", np.float64)
    mds    = col("md_since", np.float64)
    pf_abs = col("pf_ancc", np.float64)
    X = np.empty((N, len(feats_nn)), np.float32)
    for j, c in enumerate(feats_nn):
        X[:, j] = np.concatenate([np.asarray(p[c]) for p in parts])[mask].astype(np.float32)
    return N, g, y, lk, mds, pf_abs, X


t0 = time.time()
N, g, y, lk, mds, pf_abs, X = (_load_smoke(SMOKE_WELLS) if SMOKE else _load_full())
print(f"loaded: rows={N} wells={len(pd.unique(g))} K={X.shape[1]} ({time.time()-t0:.0f}s)")

# ---- fold(v52 と完全同一 = L1 GBM と揃う) ----
folds = list(GroupKFold(n_splits=N_SPLITS).split(np.zeros(N), y, groups=g))

# ---- 絶対TVT -> 残差化(lk と corr>0.98 の列は lk を引く。v52 と同一) ----
samp = np.arange(0, N, max(1, N // 120000))
abs2res = []
for j in range(X.shape[1]):
    v = X[samp, j].astype(np.float64); m = np.isfinite(v)
    ok = m.sum() > 100 and np.nanstd(v[m]) > 1e-9
    c_ = float(np.corrcoef(v[m], lk[samp][m])[0, 1]) if ok else 0.0
    a = bool(c_ > 0.98); abs2res.append(a)
    if a:
        X[:, j] = X[:, j] - lk.astype(np.float32)
print("絶対TVT->残差化:", int(sum(abs2res)), "列")

# ---- robust 標準化(メモリ安全: 統計は subsample で算出, 変換は in-place) ----
# v52 は全行 nanmedian/nanpercentile だが 718列×3.78M では全配列コピーで OOM 危険。
# robust スケールの分位点は subsample で十分精度が出るため samp 部分行で算出(結果はほぼ同一)。
Xs = X[samp]
med = np.nanmedian(Xs, axis=0).astype(np.float32)
q75 = np.nanpercentile(Xs, 75, axis=0).astype(np.float32)
q25 = np.nanpercentile(Xs, 25, axis=0).astype(np.float32)
scl = np.maximum((q75 - q25) / 1.349, 1e-6).astype(np.float32)
del Xs; gc.collect()
X -= med                                   # in-place broadcast
X /= scl                                    # in-place
np.nan_to_num(X, copy=False, nan=0.0, posinf=8.0, neginf=-8.0)
np.clip(X, -8.0, 8.0, out=X)               # in-place

t_mean = float(y.mean()); t_scale = float(y.std())
yn = ((y - t_mean) / t_scale).astype(np.float32)

# ---- 井戸ごとの行順(md_since 昇順, v52 と同一) ----
wid_order = pd.unique(g)
well_rows = {}
for w in wid_order:
    idx = np.where(g == w)[0]
    well_rows[w] = idx[np.argsort(mds[idx], kind="stable")]
fold_wells = [(pd.unique(g[tr]), pd.unique(g[va])) for tr, va in folds]

# ---- 既知ゾーン(prefix)系列の抽出(キャッシュ, v52 と同一手順) ----
PREF = {}
if CFG["prefix"]:
    pre_fp = ART / "nn" / "prefix_cache_nn.npz"
    if pre_fp.exists() and not SMOKE:
        z = np.load(pre_fp, allow_pickle=False)
        PREF = {w: np.nan_to_num(z[w], nan=0.0, posinf=8.0, neginf=-8.0)
                for w in z.files if w in set(wid_order)}
        print("prefix cache load:", len(PREF), "wells")
    else:
        grs = []; raw = {}
        for w in wid_order:
            f = DATA / "train" / f"{w}__horizontal_well.csv"
            if not f.exists():
                continue
            df = pd.read_csv(f, usecols=["MD", "Z", "GR", "TVT_input"])
            kn = df.dropna(subset=["MD", "Z", "GR", "TVT_input"]).tail(PRE_LEN)
            if len(kn) < 8:
                continue
            raw[w] = kn.to_numpy(np.float64)   # [MD, Z, GR, TVT_input]
            grs.append(raw[w][:, 2])
        allgr = np.concatenate(grs)
        gmed = float(np.nanmedian(allgr))
        giqr = float(max(np.nanpercentile(allgr, 75) - np.nanpercentile(allgr, 25), 1e-6))
        for w, a in raw.items():
            gr_n = np.clip((a[:, 2] - gmed) / giqr, -8, 8)
            tvt_rel = (a[:, 3] - a[-1, 3]) / t_scale
            z_rel = (a[:, 1] - a[-1, 1]) / 50.0
            md_rel = (a[:, 0] - a[-1, 0]) / 1000.0
            PREF[w] = np.nan_to_num(np.stack([gr_n, tvt_rel, z_rel, md_rel], axis=1),
                                    nan=0.0, posinf=8.0, neginf=-8.0).astype(np.float32)
        if not SMOKE:
            np.savez_compressed(pre_fp, **PREF)
        print("prefix extracted:", len(PREF), "wells | gr med/iqr", round(gmed, 1), round(giqr, 1))

K_IN = X.shape[1]
print(f"rows {N} | wells {len(wid_order)} | K {K_IN} | t_mean/scale {round(t_mean,3)}/{round(t_scale,3)}")


# =====================================================================
# モデル定義(SeqNet + PrefixEnc)+ ユーティリティ  ---  v52 verbatim
# =====================================================================
class ChanLN(nn.Module):
    def __init__(self, c):
        super().__init__(); self.ln = nn.LayerNorm(c)
    def forward(self, x):
        return self.ln(x.transpose(1, 2)).transpose(1, 2)


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
        super().__init__()
        self.h = heads; self.dk = ch // heads; self.p = drop
        self.qkv = nn.Linear(ch, 3 * ch); self.proj = nn.Linear(ch, ch)
    def forward(self, x, kmask):
        B, L, C = x.shape
        q, k, v = self.qkv(x).chunk(3, -1)
        q = q.view(B, L, self.h, self.dk).transpose(1, 2)
        k = k.view(B, L, self.h, self.dk).transpose(1, 2)
        v = v.view(B, L, self.h, self.dk).transpose(1, 2)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=kmask[:, None, None, :],
                                           dropout_p=self.p if self.training else 0.0)
        return self.proj(o.transpose(1, 2).reshape(B, L, C))


class PrefixEnc(nn.Module):
    """既知ゾーン系列 -> 井戸コンテキストベクトル(masked mean+max pooling)。"""
    def __init__(self, c_in, ctx):
        super().__init__()
        self.c1 = nn.Conv1d(c_in, 64, 5, padding=2)
        self.c2 = nn.Conv1d(64, 64, 5, padding=8, dilation=4)
        self.fc = nn.Linear(128, ctx)
    def forward(self, x, m):  # x:(B,C,Lp), m:(B,1,Lp)
        h = F.gelu(self.c1(x * m)); h = F.gelu(self.c2(h)) * m
        s = m.sum(-1)                                   # (B,1)
        mean = h.sum(-1) / s.clamp(min=1.0)
        mx = (h + (m - 1.0) * 1e4).amax(-1)
        mx = torch.where(s > 0, mx, torch.zeros_like(mx))
        return self.fc(torch.cat([mean, mx], dim=-1)) * (s > 0).float()


class SeqNet(nn.Module):
    def __init__(self, k_in, head, ch, ksize, dilations, drop, use_attn, attn_heads, ch_drop,
                 use_ctx=False, pre_c=4, ctx_dim=64, use_mtp=False):
        super().__init__()
        self.head = head; self.use_ctx = use_ctx
        self.inp = nn.Conv1d(k_in, ch, 1); self.indrop = nn.Dropout1d(ch_drop)
        if use_ctx:
            self.pre = PrefixEnc(pre_c, ctx_dim); self.ctxp = nn.Linear(ctx_dim, ch)
        self.blocks = nn.ModuleList([Block(ch, d, ksize, drop) for d in dilations])
        self.use_attn = use_attn
        if use_attn:
            self.attn = SelfAttn(ch, attn_heads, drop)
            self.aln = nn.LayerNorm(ch)
        self.out = nn.Conv1d(ch, 5 if head == "mdn2" else 1, 1)
        self.mtp = nn.Conv1d(ch, len(MTP_OFFS), 1) if use_mtp else None
    def forward(self, x, m, px=None, pm=None):
        h = self.inp(self.indrop(x)) * m
        if self.use_ctx and px is not None:
            ctx = self.pre(px, pm)
            h = (h + self.ctxp(ctx)[:, :, None]) * m
        for b in self.blocks:
            h = b(h, m)
        if self.use_attn:
            q = self.aln(h.transpose(1, 2))
            a = self.attn(q, (m[:, 0, :] > 0.5))
            h = (h + a.transpose(1, 2)) * m
        mo = self.mtp(h) if self.mtp is not None else None
        return self.out(h), mo


def head_loss(out, yb, mb, head):
    if head == "mse":
        p = out[:, 0]
        return ((p - yb) ** 2 * mb).sum() / mb.sum(), p
    mu1, mu2 = out[:, 0], out[:, 1]
    pl = torch.sigmoid(out[:, 2])
    ls1 = out[:, 3].clamp(-3.0, 2.0); ls2 = out[:, 4].clamp(-3.0, 2.0)
    def logN(mu, ls):
        return -0.5 * ((yb - mu) / ls.exp()) ** 2 - ls - 0.5 * math.log(2 * math.pi)
    nll = -torch.logsumexp(torch.stack([torch.log(pl + 1e-6) + logN(mu1, ls1),
                                        torch.log(1 - pl + 1e-6) + logN(mu2, ls2)]), dim=0)
    mean = pl * mu1 + (1 - pl) * mu2
    loss = (nll * mb).sum() / mb.sum() + 0.1 * ((mean - yb) ** 2 * mb).sum() / mb.sum()
    return loss, mean


def predict_head(out, head):
    if head == "mse":
        return out[:, 0]
    pl = torch.sigmoid(out[:, 2]); return pl * out[:, 0] + (1 - pl) * out[:, 1]


class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()
                       if v.dtype.is_floating_point}
    def update(self, model):
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
    def copy_to(self, model):
        sd = model.state_dict()
        for k, v in self.shadow.items():
            sd[k].copy_(v)


def make_batch(wids, train, rng, flip_all=False):
    """wids(井戸idリスト)-> (x, y, mask, sel, px, pm)。flip_all=True は TTA 用に全井戸反転。"""
    sel = []
    for w in wids:
        r = well_rows[w]
        if flip_all or (train and CFG["flip"] and rng.random() < 0.5):
            r = r[::-1]
        if train and len(r) > CROP:
            s = int(rng.integers(0, len(r) - CROP + 1)); r = r[s:s + CROP]
        sel.append((w, r))
    Lm = max(len(r) for _, r in sel); B = len(sel)
    xb = np.zeros((B, Lm, K_IN), np.float32)
    yb = np.zeros((B, Lm), np.float32); mb = np.zeros((B, Lm), np.float32)
    for i, (w, r) in enumerate(sel):
        xb[i, :len(r)] = X[r]; yb[i, :len(r)] = yn[r]; mb[i, :len(r)] = 1.0
    x = torch.from_numpy(xb).to(DEVICE).transpose(1, 2)
    yb = torch.from_numpy(yb).to(DEVICE); mb = torch.from_numpy(mb).to(DEVICE)
    px = pm = None
    if CFG["prefix"]:
        Lp = max(len(PREF.get(w, np.zeros((1, PRE_C), np.float32))) for w, _ in sel)
        pxb = np.zeros((B, Lp, PRE_C), np.float32); pmb = np.zeros((B, Lp), np.float32)
        for i, (w, _) in enumerate(sel):
            a = PREF.get(w)
            if a is not None:
                pxb[i, :len(a)] = a; pmb[i, :len(a)] = 1.0
        px = torch.from_numpy(pxb).to(DEVICE).transpose(1, 2)
        pm = torch.from_numpy(pmb).to(DEVICE).unsqueeze(1)
    return x, yb, mb, sel, px, pm


@torch.no_grad()
def _predict_once(model, head, wids, flip):
    model.eval(); out = np.zeros(N, np.float64)
    order = sorted(wids, key=lambda w: len(well_rows[w]))
    i = 0
    while i < len(order):
        Lm = len(well_rows[order[i]]); j = i + 1
        while j < len(order):
            Lc = max(Lm, len(well_rows[order[j]]))
            if (j - i + 1) * Lc > EVAL_TOK:
                break
            Lm = Lc; j += 1
        x, _, mb, sel, px, pm = make_batch(order[i:j], False, None, flip_all=flip)
        with torch.autocast(DEVICE, dtype=torch.bfloat16, enabled=(DEVICE == "cuda")):
            o = model(x, mb.unsqueeze(1), px, pm)[0]
        p = predict_head(o.float(), head).cpu().numpy()
        for b, (w, r) in enumerate(sel):
            out[r] = p[b, :len(r)]   # r 反転でも scatter で正位置
        i = j
    return out


def predict_wells(model, head, wids):
    p = _predict_once(model, head, wids, False)
    if CFG["flip"]:
        p = 0.5 * (p + _predict_once(model, head, wids, True))   # TTA: 正順+逆順平均
    return p


# =====================================================================
# 学習(RUN プリセット): mdn2 x SEEDS x fold  ---  v52 verbatim
# =====================================================================
def cosine_lr(step, total, warm):
    if step < warm:
        return step / max(1, warm)
    t = (step - warm) / max(1, total - warm)
    return 0.5 * (1.0 + math.cos(math.pi * t))


def new_model(head):
    return SeqNet(K_IN, head, CH, KSIZE, DILATIONS, DROP, USE_ATTN, ATTN_HEADS, CH_DROP,
                  use_ctx=CFG["prefix"], pre_c=PRE_C, ctx_dim=CTX_DIM,
                  use_mtp=CFG.get("mtp", False)).to(DEVICE)


def train_fold(head, seed, k):
    trw, vaw = fold_wells[k]
    fp = ART / "nn" / f"{RUN}{SUFFIX}_{head}_s{seed}_fold{k}.pt"
    model = new_model(head)
    if fp.exists() and not SMOKE:
        model.load_state_dict(torch.load(fp, map_location=DEVICE))
        return predict_wells(model, head, list(vaw)), "load"
    torch.manual_seed(1000 * seed + k); rng = np.random.default_rng(1000 * seed + k)
    trw = list(trw); vaw = list(vaw)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    steps_ep = math.ceil(len(trw) / BATCH_WELLS)
    total = steps_ep * EPOCHS; warm = int(total * WARM_FRAC)
    ema = EMA(model, EMA_DECAY); best = (1e18, None); step = 0
    va_idx = np.concatenate([well_rows[w] for w in vaw])
    for ep in range(EPOCHS):
        model.train(); order = rng.permutation(len(trw))
        for i in range(0, len(order), BATCH_WELLS):
            x, yb, mb, _, px, pm = make_batch([trw[j] for j in order[i:i + BATCH_WELLS]], True, rng)
            if NOISE:
                x = x + NOISE * torch.randn_like(x)
            for pg in opt.param_groups:
                pg["lr"] = LR * cosine_lr(step, total, warm)
            with torch.autocast(DEVICE, dtype=torch.bfloat16, enabled=(DEVICE == "cuda")):
                out, mo = model(x, mb.unsqueeze(1), px, pm)
            loss, _ = head_loss(out.float(), yb, mb, head)
            if CFG.get("mtp", False) and mo is not None:
                for _oi, _kk in enumerate(MTP_OFFS):
                    if yb.shape[1] > _kk:
                        _dt = yb[:, _kk:] - yb[:, :-_kk]
                        _vm = mb[:, _kk:] * mb[:, :-_kk]
                        if _vm.sum() > 0:
                            _pk = mo[:, _oi, :-_kk].float()
                            loss = loss + MTP_W * ((_pk - _dt) ** 2 * _vm).sum() / _vm.sum()
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            opt.step(); ema.update(model); step += 1
            if SMOKE and step >= 4:
                break   # スモークは数ステップで打ち切り(forward/backward 確認のみ)
        bak = copy.deepcopy(model.state_dict()); ema.copy_to(model)
        p = predict_wells(model, head, vaw)
        rm = float(np.sqrt(np.mean((p[va_idx] * t_scale + t_mean - y[va_idx]) ** 2)))
        if not np.isfinite(rm):
            raise RuntimeError(f"val RMSE が非有限 (ep={ep}): 入力/損失にNaN。prefix等の入力を確認")
        if rm < best[0]:
            best = (rm, copy.deepcopy(model.state_dict()))
        model.load_state_dict(bak)
    model.load_state_dict(best[1])
    if not SMOKE:
        torch.save(best[1], fp)
    return predict_wells(model, head, vaw), f"fit best={best[0]:.4f}"


nn_oof = {}
folds_to_run = [0] if SMOKE else list(range(N_SPLITS))
for head in HEADS:
    for seed in SEEDS:
        key = f"{head}_s{seed}"; oof = np.zeros(N, np.float64); tt = time.time()
        for k in folds_to_run:
            p, tag = train_fold(head, seed, k)
            va = folds[k][1]; oof[va] = p[va]
            gc.collect()
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
            _rm = np.sqrt(np.mean((p[va] * t_scale + t_mean - y[va]) ** 2))
            print(f"[{RUN}] {key} fold{k} [{tag}] rmse={_rm:.4f}", flush=True)
        nn_oof[key] = oof * t_scale + t_mean
        if SMOKE:
            va0 = folds[0][1]
            _rm = np.sqrt(np.mean((nn_oof[key][va0] - y[va0]) ** 2))
            print(f"== [{RUN}] {key} SMOKE fold0-val rmse={_rm:.4f} oof.shape={nn_oof[key].shape}"
                  f" ({time.time()-tt:.0f}s)")
        else:
            print(f"== [{RUN}] {key} OOF raw rmse={np.sqrt(np.mean((nn_oof[key]-y)**2)):.4f}"
                  f" ({time.time()-tt:.0f}s)")

oof_out = ART / (f"nn_oof_{RUN}_SMOKE.npz" if SMOKE else f"nn_oof_{RUN}{SUFFIX}.npz")
np.savez(oof_out, **nn_oof)
print(f"saved {oof_out.name} keys={list(nn_oof.keys())} shape={nn_oof[HEADS[0]+'_s'+str(SEEDS[0])].shape}")

# 3seed アンサンブル(本学習時のみ意味あり)
if not SMOKE and len(nn_oof) > 1:
    ens = np.mean(list(nn_oof.values()), axis=0)
    print(f"== [{RUN}] {len(nn_oof)}-seed ensemble OOF raw rmse={np.sqrt(np.mean((ens-y)**2)):.4f}")

# =====================================================================
# 評価 + (存在すれば)GBM blend + results/submit_pack 書き出し  ---  v52 準拠
# =====================================================================
if not SMOKE:
    pf_res = pf_abs - lk
    w_pf = float(pp_params.get("w_pf", 0.09)); tau = pp_params.get("tau", 85)
    alpha = float(pp_params.get("alpha", 1.0))

    def apply_pp_res(res):
        d = res * (1 - w_pf) + pf_res * w_pf
        if tau:
            d = d * (1.0 - np.exp(-np.maximum(mds, 0.0) / float(tau)))
        return d * alpha

    def raw_rmse(res):
        return float(np.sqrt(np.mean((res - y) ** 2)))

    def overall(res):
        return float(np.sqrt(np.mean((apply_pp_res(res) - y) ** 2)))

    def wells_over(res, thr=10.0):
        e = apply_pp_res(res) - y
        return int(sum(1 for w in wid_order if np.sqrt(np.mean(e[well_rows[w]] ** 2)) > thr))

    ens = np.mean(list(nn_oof.values()), axis=0)
    rec = dict(cfg=CFG, crop=CROP, seeds=len(SEEDS),
               nn_raw=raw_rmse(ens), nn_pp=overall(ens), nn_over10=wells_over(ens))
    print(f"[{RUN}] NN単体: raw={rec['nn_raw']:.4f} pp={rec['nn_pp']:.4f} >10={rec['nn_over10']}")

    # v93 L1 GBM OOF が(将来)存在すれば blend を探索。無ければ NN 単体のみ記録。
    gbm_candidates = [ART / "gbm_l1_oof.npy", ART / "gbm_v93_oof.npy", ART / "gbm_oof.npy"]
    gbm_fp = next((p for p in gbm_candidates if p.exists()), None)
    if gbm_fp is not None:
        gbm_oof = np.load(gbm_fp).astype(np.float64)
        assert len(gbm_oof) == N, (len(gbm_oof), N)
        ws = np.arange(0.0, 1.001, 0.02)
        scs = [overall((1 - w) * gbm_oof + w * ens) for w in ws]
        bi = int(np.argmin(scs)); wb = float(ws[bi])
        res_b = (1 - wb) * gbm_oof + wb * ens
        rec.update(blend_w=wb, blend_pp=float(scs[bi]), blend_over10=wells_over(res_b))
        print(f"[{RUN}] blend : w={wb:.2f} pp={rec['blend_pp']:.4f} >10={rec['blend_over10']}"
              f"  (GBM単体 {overall(gbm_oof):.4f}, src={gbm_fp.name})")
    else:
        rec.update(blend_w=1.0, blend_pp=rec["nn_pp"], blend_over10=rec["nn_over10"])
        print(f"[{RUN}] GBM OOF 未検出 -> blend スキップ(候補: {[p.name for p in gbm_candidates]})")

    res_fp = ART / f"results_nn{SUFFIX}.json"
    allres = json.loads(res_fp.read_text()) if res_fp.exists() else {}
    allres[RUN] = rec
    res_fp.write_text(json.dumps(allres, indent=1))

    # ---- submit 用パック(推論側が読む。prefix GR 統計を CSV から再計算) ----
    gmed = giqr = None
    if CFG.get("prefix"):
        _grs = []
        for _w in wid_order:
            _f = DATA / "train" / f"{_w}__horizontal_well.csv"
            if not _f.exists():
                continue
            _dfp = pd.read_csv(_f, usecols=["MD", "Z", "GR", "TVT_input"])
            _kn = _dfp.dropna(subset=["MD", "Z", "GR", "TVT_input"]).tail(PRE_LEN)
            if len(_kn) < 8:
                continue
            _grs.append(_kn["GR"].to_numpy(np.float64))
        _all = np.concatenate(_grs)
        gmed = float(np.nanmedian(_all))
        giqr = float(max(np.nanpercentile(_all, 75) - np.nanpercentile(_all, 25), 1e-6))
    pack = dict(run=RUN, cfg=CFG, w_nn=rec.get("blend_w", 1.0), oof_blend_pp=rec.get("blend_pp"),
                seeds=SEEDS, n_splits=N_SPLITS,
                features=feats_nn, abs2res=abs2res, med=med.tolist(), scale=scl.tolist(),
                t_mean=t_mean, t_scale=t_scale,
                arch=dict(ch=CH, ksize=KSIZE, dilations=DILATIONS, drop=DROP, use_attn=USE_ATTN,
                          attn_heads=ATTN_HEADS, ch_drop=CH_DROP,
                          pre_len=PRE_LEN, pre_c=PRE_C, ctx_dim=CTX_DIM, mtp_offs=MTP_OFFS),
                prefix_stats=dict(gmed=gmed, giqr=giqr))
    (ART / f"submit_pack_nn{SUFFIX}.json").write_text(json.dumps(pack))
    print(f"submit_pack_nn{SUFFIX}.json saved: run={RUN} w_nn={pack['w_nn']} prefix_stats=({gmed}, {giqr})")

print("done.")
