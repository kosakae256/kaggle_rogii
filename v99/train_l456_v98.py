# -*- coding: utf-8 -*-
"""v93 L4/L5/L6/L7 トレーナ -- v62 の segment/bin レベル「別表現」脚の STRUCT-FREE 版。

v62 train_l456_v62.py を再現するが:
  (a) ARCH = os.environ.get("ARCH","l5") で L4/L5/L6/L7 を切替。全クラスを v62 から VERBATIM
      再現(K=14 = struct-free 14 軌跡 = v62 の 16 から gf/v50b を除いた値)。
        L4 selfsim : 自己相似バイアス付き Transformer(|med_i-med_j|/TAU_SIM + |i-j| バイアス)
        L5 curveattn: エンジン14本を曲線トークン化し bin 別 cross-attention で凸結合+微小残差
        L6 spectral : (2K+3)*3 ch の多重 dilation Conv1d(検波ピラミッド)
        L7 ranking  : L5 骨格だが凸結合のみ(残差なし)。forward は (out, w) を返す(listwise CE)
  (b) 入力 = seqfeat_v93.pkl(D/S/C/dtrue/wgt/bid/rows/T)。imagesF の画像 ch は使わない。
  (c) CV = GroupKFold(5) by well。★v93 L1 GBM / L2 TCN と同一の per-row GroupKFold を
      再現して well->fold を作り、OOF をスタック用に完全整列させる
      (L1/L2: `GroupKFold(5).split(np.zeros(N), y, groups=g)`, g=parquet行順の well)。
  (d) 損失/最適化/bucket batching/early-stop は v62 と同一(L5 は重み付き MSE。TAU_R は l7 用で L5 未使用)。
      v62 L5 に EMA は無いので EMA も無し。
  (e) 3 seeds(SEEDS 環境変数, 既定 0,1,2)。per-row OOF を TARGET 単位(= TVT - last_known_tvt)で保存。
      出力: v93/artifacts_v93/{ARCH}_oof.npz(keys={ARCH}_s0/s1/s2 / {ARCH}_mean / y)。
  resumable: per (seed,fold) の val 予測を {ARCH}_ckpt/ にキャッシュ(ARCH ごとに別ディレクトリ)。

env:
  ARCH=l4|l5|l6|l7  (既定 l5)
  SMOKE=1  -> 1 seed(0) / 1 fold(0) / ~40 wells / 数エポックで forward-backward と OOF shape だけ確認。
  SEEDS="0,1,2"  DIM=64  VFRAC=0.5  MAXMIN=45  MAXEP  TAU_R=1.5(l7)  TAU_SIM=3.0(l4)

★struct禁止: seqfeat_v93.pkl(v93 parquet 単体・struct-free)しか読まない。gf/v50b/struct 一切なし。
cp932 安全出力・em-dash 不使用('->')。
"""
import io, sys
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
import os, json, time, gc, pickle, random
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.model_selection import GroupKFold

P0 = Path(r"C:\Users\kosaka256\Documents\rogii_claude")
ART = Path(os.environ.get("ROGII_ART98", str(P0 / "v98" / "artifacts_v99")))   # ★v98
# --- 入力切替(env V93_INPUT): fwd(既定, seqfeat_v93.pkl K=14) / fwdbwd / fwdbwd_anchor(bpf ch 込み K=26) ---
V93_INPUT = os.environ.get("V93_INPUT", "combo").lower()   # ★v98: combo既定
_L5MAP = {"fwd": ("train.parquet", ""),
          "combo": ("train_combo.parquet", "_fcombo"),      # ★v98: combo(v96+v97結合)
          "fwdbwd": ("train_fwdbwd.parquet", "_fwdbwd"),
          "fwdbwd_anchor": ("train_fwdbwd_anchor.parquet", "_fwdbwd_anchor")}
if V93_INPUT not in _L5MAP:
    raise SystemExit(f"[L456] 未知の V93_INPUT={V93_INPUT}")
_fn, _osuf_base = _L5MAP[V93_INPUT]
# ★実験: L5_SF で seqfeat 入力を上書き / L5_OSUF で出力・ckpt suffix を上書き(既存を上書きしない)
_osuf = os.environ.get("L5_OSUF", _osuf_base)
# fold/well/target は base train.parquet で決まる(bpf 列は fold 無関係。variant parquet は消えていても可)。
PQ = ART / "data" / _fn                             # ★v98: combo なら train_combo.parquet で fold を作る
SF_FP = Path(os.environ["L5_SF"]) if os.environ.get("L5_SF") else ART / (f"seqfeat_v95b{_osuf_base}.pkl")

ARCH = os.environ.get("ARCH", "l5")                # v93: L4/L5/L6/L7 切替
assert ARCH in ("l4", "l5", "l6", "l7", "l5x", "l5b"), f"ARCH must be l4|l5|l6|l7|l5x|l5b, got {ARCH}"
L5B_GW = float(os.environ.get("L5B_GW", "0.5"))    # 枝分離L5のゲートBCE重み
L5B_EPS = float(os.environ.get("L5B_EPS", "0.05"))  # MCL 敗者の最小重み(ヘッド崩壊防止)
CKPT = ART / f"{ARCH}{_osuf}_ckpt"; CKPT.mkdir(parents=True, exist_ok=True)   # variant 毎に別ckpt(resume事故防止)

DIM = int(os.environ.get("DIM", "64")); PAT = 12; LRPAT = 4
VFRAC = float(os.environ.get("VFRAC", "0.5"))
MAXMIN = float(os.environ.get("MAXMIN", "45")); BW = 6; TOKB = int(os.environ.get("TOKB", "3000"))
L5_TOPK = int(os.environ.get("L5_TOPK", "0"))   # >0 で L5 の候補softmaxを top-k に絞る(0=全候補=従来)
N_SPLITS = 5
SMOKE = os.environ.get("SMOKE", "0") == "1"
MAXEP = int(os.environ.get("MAXEP", "2" if SMOKE else "100000"))
SEEDS = [int(s) for s in os.environ.get("SEEDS", "0" if SMOKE else "0,1,2,3,4").split(",") if s != ""]   # ★v98: 5seed
TAU_R = float(os.environ.get("TAU_R", "1.5"))     # l7: listwise ターゲット温度(ft)
TAU_SIM = float(os.environ.get("TAU_SIM", "3.0"))  # l4: 自己相似バイアスの距離スケール(ft)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
if DEV == "cuda":
    torch.cuda.set_per_process_memory_fraction(VFRAC)

t0 = time.time()
SF = pickle.loads(SF_FP.read_bytes())
# ---- ★v98 DA: aug seqfeat(逆/順) を接尾辞付き well名で SF へ統合(train側専用・val/OOFには入れない) ----
AUG_KEYS = set()
if os.environ.get("V98_DA", "1") == "1" and not SMOKE:
    for tag, ev in [("__b", "L5_SF_BWD"), ("__f", "L5_SF_FWD")]:
        sp = os.environ.get(ev)
        if not sp or not Path(sp).exists():
            print(f"[v98 DA-L5] {ev} 無し -> スキップ"); continue
        sfa = pickle.loads(Path(sp).read_bytes())
        for w, d in sfa.items():
            SF[str(w) + tag] = d; AUG_KEYS.add(str(w) + tag)
        print(f"[v98 DA-L5] +{Path(sp).name} 井={len(sfa)} (tag{tag})", flush=True)
WELLS_ALL = sorted(SF.keys())
K = SF[WELLS_ALL[0]]["D"].shape[1]
CDIM = SF[WELLS_ALL[0]]["C"].shape[1]           # ★C(context)次元を自動導出(CEXTRA対応)
NF = SF[WELLS_ALL[0]]["DX"].shape[2] if "DX" in SF[WELLS_ALL[0]] else 0   # ★候補別特徴数(CANDFEAT)
assert 14 <= K <= 200, f"K は 14(fwd)/26(fwdbwd)/46(combo10バンク)/71(v100 15バンク)/101(+bwd v96) 等。got {K}"
print(f"wells={len(WELLS_ALL)} K={K} V93_INPUT={V93_INPUT} SF={SF_FP.name} ARCH={ARCH} "
      f"SMOKE={SMOKE} SEEDS={SEEDS} ({time.time()-t0:.0f}s)", flush=True)


# ======================= L5(v62 VERBATIM。K=14) =======================
class L5(nn.Module):
    def __init__(self, k=K, d=DIM):
        super().__init__()
        self.k = k
        self.enc = nn.Sequential(nn.Conv1d(2 + NF, d, 9, padding=4), nn.GELU(),
                                 nn.Conv1d(d, d, 9, padding=4 * 4, dilation=4), nn.GELU(),
                                 nn.Conv1d(d, d, 9, padding=4 * 16, dilation=16))
        self.emb = nn.Embedding(k, d)
        self.qry = nn.Sequential(nn.Linear(CDIM + 4, d), nn.GELU(), nn.Linear(d, d))
        self.res = nn.Linear(d, 1)
        nn.init.zeros_(self.res.weight); nn.init.zeros_(self.res.bias)

    def forward(self, D, S, C, m, DX=None):
        B, T, k = D.shape
        med = D.median(2, keepdim=True).values
        x = torch.stack([D - med, torch.log(S)] + ([DX[..., _f] for _f in range(DX.shape[-1])] if DX is not None else []), 2)                # (B,T,2,K)
        x = x.permute(0, 3, 2, 1).reshape(B * k, 2 + (DX.shape[-1] if DX is not None else 0), T)
        hc = self.enc(x).reshape(B, k, -1, T).permute(0, 3, 1, 2)  # (B,T,K,d)
        hc = hc + self.emb.weight[None, None]
        stats = torch.cat([C, med / 10.0, D.std(2, keepdim=True),
                           (D - med).abs().lt(2).float().mean(2, keepdim=True),
                           torch.zeros_like(med)], -1)
        q = self.qry(stats)                                        # (B,T,d)
        att = (hc @ q[..., None])[..., 0] / (hc.shape[-1] ** 0.5)  # (B,T,K)
        if L5_TOPK and 0 < L5_TOPK < att.shape[-1]:                 # ★top-k softmax: 上位k候補内で再正規化, 残りは0
            thr = att.topk(L5_TOPK, dim=-1).values[..., -1:]
            att = att.masked_fill(att < thr, float("-inf"))
        w = att.softmax(-1)
        out = (w * D).sum(-1)
        return out + 2.0 * torch.tanh(self.res(q)[..., 0])


# ======================= L4(v62 VERBATIM。K=14) =======================
class L4(nn.Module):
    def __init__(self, k=K, d=DIM, nl=3, nh=4):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(2 * k + 3 + 1, d), nn.GELU(), nn.Linear(d, d))
        self.layers = nn.ModuleList([nn.ModuleDict(dict(
            q=nn.Linear(d, d), kk=nn.Linear(d, d), v=nn.Linear(d, d), o=nn.Linear(d, d),
            n1=nn.LayerNorm(d), n2=nn.LayerNorm(d),
            ff=nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d)))) for _ in range(nl)])
        self.nh = nh; self.d = d
        self.head = nn.Linear(d, 1)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)

    def forward(self, D, S, C, m, DX=None):
        med = D.median(2, keepdim=True).values                     # (B,T,1)
        x = torch.cat([D - med, torch.log(S), C, med / 10.0], -1)
        h = self.inp(x)
        B, T, d = h.shape
        # self-similarity bias: -|med_i - med_j|/TAU_SIM - |i-j|/96
        dm = (med[:, :, 0][:, :, None] - med[:, :, 0][:, None, :]).abs()
        ti = torch.arange(T, device=D.device).float()
        dt_ = (ti[None, :, None] - ti[None, None, :]).abs()
        bias = -dm / TAU_SIM - dt_ / 96.0
        bias = bias.masked_fill(~(m[:, None, :].bool()), -1e9)
        for L in self.layers:
            hn = L["n1"](h)
            q = L["q"](hn).view(B, T, self.nh, -1).transpose(1, 2)
            kk = L["kk"](hn).view(B, T, self.nh, -1).transpose(1, 2)
            v = L["v"](hn).view(B, T, self.nh, -1).transpose(1, 2)
            att = (q @ kk.transpose(-1, -2)) / (q.shape[-1] ** 0.5) + bias[:, None]
            h = h + L["o"]((att.softmax(-1) @ v).transpose(1, 2).reshape(B, T, d))
            h = h + L["ff"](L["n2"](h))
        return self.head(h)[..., 0] + med[..., 0]                  # 残差 = med からのずれ


# ======================= L6(v62 VERBATIM。K=14) =======================
class L6(nn.Module):
    def __init__(self, k=K, d=DIM):
        super().__init__()
        cin = (2 * k + 3) * 3
        self.net = nn.Sequential(nn.Conv1d(cin, d, 5, padding=2), nn.GELU(),
                                 nn.Conv1d(d, d, 5, padding=2 * 4, dilation=4), nn.GELU(),
                                 nn.Conv1d(d, d, 5, padding=2 * 16, dilation=16), nn.GELU(),
                                 nn.Conv1d(d, d, 5, padding=2 * 64, dilation=64))
        self.head = nn.Linear(d, 1)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)

    def forward(self, D, S, C, m, DX=None):
        med = D.median(2, keepdim=True).values
        base = torch.cat([D - med, torch.log(S), C], -1).transpose(1, 2)   # (B,c,T)
        outs = [base]
        for w_ in (4, 16):                                        # 多重スケール(detrend ピラミッド)
            sm = F.avg_pool1d(F.pad(base, (w_ // 2, w_ - w_ // 2 - 1), mode="replicate"), w_, stride=1)
            outs.append(base - sm)
        h = self.net(torch.cat(outs, 1)).transpose(1, 2)
        return self.head(h)[..., 0] + med[..., 0]


# ======================= L7(v62 VERBATIM。K=14。forward は (out, w) を返す) =======================
class L7(nn.Module):
    """純選択(ranking)版: L5 と同骨格だが凸結合のみ(残差なし)。forward は (out, w) を返す。"""
    def __init__(self, k=K, d=DIM):
        super().__init__()
        self.k = k
        self.enc = nn.Sequential(nn.Conv1d(2 + NF, d, 9, padding=4), nn.GELU(),
                                 nn.Conv1d(d, d, 9, padding=4 * 4, dilation=4), nn.GELU(),
                                 nn.Conv1d(d, d, 9, padding=4 * 16, dilation=16))
        self.emb = nn.Embedding(k, d)
        self.qry = nn.Sequential(nn.Linear(CDIM + 4, d), nn.GELU(), nn.Linear(d, d))

    def forward(self, D, S, C, m, DX=None):
        B, T, k = D.shape
        med = D.median(2, keepdim=True).values
        x = torch.stack([D - med, torch.log(S)] + ([DX[..., _f] for _f in range(DX.shape[-1])] if DX is not None else []), 2)
        x = x.permute(0, 3, 2, 1).reshape(B * k, 2 + (DX.shape[-1] if DX is not None else 0), T)
        hc = self.enc(x).reshape(B, k, -1, T).permute(0, 3, 1, 2)
        hc = hc + self.emb.weight[None, None]
        stats = torch.cat([C, med / 10.0, D.std(2, keepdim=True),
                           (D - med).abs().lt(2).float().mean(2, keepdim=True),
                           torch.zeros_like(med)], -1)
        q = self.qry(stats)
        att = (hc @ q[..., None])[..., 0] / (hc.shape[-1] ** 0.5)
        w = att.softmax(-1)
        return (w * D).sum(-1), w


# ======================= L5X(L5 + 候補間セルフアテンション。候補が互いの一致/枝構造を見てから選択) =======================
class L5X(nn.Module):
    def __init__(self, k=K, d=DIM, nh=4, nca=int(os.environ.get("L5X_NCA", "2"))):
        super().__init__()
        self.k = k; self.nh = nh
        self.enc = nn.Sequential(nn.Conv1d(2 + NF, d, 9, padding=4), nn.GELU(),
                                 nn.Conv1d(d, d, 9, padding=4 * 4, dilation=4), nn.GELU(),
                                 nn.Conv1d(d, d, 9, padding=4 * 16, dilation=16))
        self.emb = nn.Embedding(k, d)
        # 候補間セルフアテンション(K次元で候補同士が attend)
        self.ca = nn.ModuleList([nn.ModuleDict(dict(
            q=nn.Linear(d, d), kk=nn.Linear(d, d), v=nn.Linear(d, d), o=nn.Linear(d, d),
            n1=nn.LayerNorm(d), n2=nn.LayerNorm(d),
            ff=nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d)))) for _ in range(nca)])
        self.qry = nn.Sequential(nn.Linear(CDIM + 4, d), nn.GELU(), nn.Linear(d, d))
        self.res = nn.Linear(d, 1)
        nn.init.zeros_(self.res.weight); nn.init.zeros_(self.res.bias)

    def forward(self, D, S, C, m, DX=None):
        B, T, k = D.shape
        med = D.median(2, keepdim=True).values
        x = torch.stack([D - med, torch.log(S)] + ([DX[..., _f] for _f in range(DX.shape[-1])] if DX is not None else []), 2)
        x = x.permute(0, 3, 2, 1).reshape(B * k, 2 + (DX.shape[-1] if DX is not None else 0), T)
        hc = self.enc(x).reshape(B, k, -1, T).permute(0, 3, 1, 2)   # (B,T,K,d)
        hc = hc + self.emb.weight[None, None]
        h = hc.reshape(B * T, k, -1)                                 # (B*T, K, d) 候補=系列
        for L in self.ca:
            hn = L["n1"](h)
            q = L["q"](hn).view(B * T, k, self.nh, -1).transpose(1, 2)
            kk = L["kk"](hn).view(B * T, k, self.nh, -1).transpose(1, 2)
            v = L["v"](hn).view(B * T, k, self.nh, -1).transpose(1, 2)
            att = (q @ kk.transpose(-1, -2)) / (q.shape[-1] ** 0.5)
            h = h + L["o"]((att.softmax(-1) @ v).transpose(1, 2).reshape(B * T, k, -1))
            h = h + L["ff"](L["n2"](h))
        hc = h.reshape(B, T, k, -1)
        stats = torch.cat([C, med / 10.0, D.std(2, keepdim=True),
                           (D - med).abs().lt(2).float().mean(2, keepdim=True), torch.zeros_like(med)], -1)
        q = self.qry(stats)
        att = (hc @ q[..., None])[..., 0] / (hc.shape[-1] ** 0.5)
        w = att.softmax(-1)
        return (w * D).sum(-1) + 2.0 * torch.tanh(self.res(q)[..., 0])


class L5B(nn.Module):
    """★枝分離L5(branch-separated): 候補エンコードはL5共通、クエリを2本(A/B)にして
       同じ候補群への2つのsoftmax重み→2推定 outA/outB。MCL(勝者のみ罰)損失で2ヘッドが
       datum二枝に分化。ゲートgが枝を選び推論= g·outA+(1-g)·outB(確信でコミット/不確実で平均)。
       forward戻り: (out=ゲート後, (outA,outB,glogit))  ← o[0]が推論/OOF, o[1]が学習損失用。"""
    def __init__(self, k=K, d=DIM):
        super().__init__()
        self.k = k
        self.enc = nn.Sequential(nn.Conv1d(2 + NF, d, 9, padding=4), nn.GELU(),
                                 nn.Conv1d(d, d, 9, padding=4 * 4, dilation=4), nn.GELU(),
                                 nn.Conv1d(d, d, 9, padding=4 * 16, dilation=16))
        self.emb = nn.Embedding(k, d)
        self.qryA = nn.Sequential(nn.Linear(CDIM + 4, d), nn.GELU(), nn.Linear(d, d))
        self.qryB = nn.Sequential(nn.Linear(CDIM + 4, d), nn.GELU(), nn.Linear(d, d))
        self.resA = nn.Linear(d, 1); self.resB = nn.Linear(d, 1)
        for r in (self.resA, self.resB):
            nn.init.zeros_(r.weight); nn.init.zeros_(r.bias)
        self.gate = nn.Sequential(nn.Linear(CDIM + 4, d), nn.GELU(), nn.Linear(d, 1))

    def forward(self, D, S, C, m, DX=None):
        B, T, k = D.shape
        med = D.median(2, keepdim=True).values
        nf = DX.shape[-1] if DX is not None else 0
        x = torch.stack([D - med, torch.log(S)] + ([DX[..., _f] for _f in range(nf)] if DX is not None else []), 2)
        x = x.permute(0, 3, 2, 1).reshape(B * k, 2 + nf, T)
        hc = self.enc(x).reshape(B, k, -1, T).permute(0, 3, 1, 2)   # (B,T,K,d)
        hc = hc + self.emb.weight[None, None]
        stats = torch.cat([C, med / 10.0, D.std(2, keepdim=True),
                           (D - med).abs().lt(2).float().mean(2, keepdim=True), torch.zeros_like(med)], -1)
        scale = hc.shape[-1] ** 0.5

        def head(qry, res):
            q = qry(stats)
            att = (hc @ q[..., None])[..., 0] / scale
            w = att.softmax(-1)
            return (w * D).sum(-1) + 2.0 * torch.tanh(res(q)[..., 0])
        outA = head(self.qryA, self.resA)
        outB = head(self.qryB, self.resB)
        glogit = self.gate(stats)[..., 0]
        g = torch.sigmoid(glogit)
        out = g * outA + (1.0 - g) * outB
        return out, (outA, outB, glogit)


ARCHES = {"l4": L4, "l5": L5, "l6": L6, "l7": L7, "l5x": L5X, "l5b": L5B}


# ======================= バッチ(v62 make_batch を seqfeat 内 dtrue/wgt に対応) =======================
def make_batch(wns):
    Tm = max(SF[w]["D"].shape[0] for w in wns); B = len(wns)   # T = nb(bin 数)
    D = torch.zeros(B, Tm, K); S = torch.ones(B, Tm, K); C = torch.zeros(B, Tm, CDIM); DX = (torch.zeros(B, Tm, K, NF) if NF > 0 else None)
    yv = torch.full((B, Tm), float("nan")); m = torch.zeros(B, Tm); wg = torch.zeros(B, Tm)
    for i, w in enumerate(wns):
        f = SF[w]; T = f["D"].shape[0]
        D[i, :T] = torch.from_numpy(f["D"]); S[i, :T] = torch.from_numpy(f["S"]); C[i, :T] = torch.from_numpy(f["C"])
        yv[i, :T] = torch.from_numpy(np.asarray(f["dtrue"], np.float32))
        wg[i, :T] = torch.from_numpy(np.asarray(f["wgt"], np.float32)); m[i, :T] = 1
        if NF > 0: DX[i, :T] = torch.from_numpy(SF[w]["DX"])
    return D.to(DEV), S.to(DEV), C.to(DEV), yv.to(DEV), m.to(DEV), wg.to(DEV), (DX.to(DEV) if DX is not None else None)


def bucket(ws, rng):
    lens = {w: SF[w]["D"].shape[0] for w in ws}
    order = rng.permutation(len(ws)); ws_ = [ws[j] for j in order]
    wl = [w for i in range(0, len(ws_), 64) for w in sorted(ws_[i:i + 64], key=lambda x: lens[x])]
    out = []; cur = []
    for w in wl:
        t = cur + [w]
        if cur and (len(t) * max(lens[x] for x in t) > TOKB or len(t) > BW):
            out.append(cur); cur = [w]
        else:
            cur = t
    if cur: out.append(cur)
    return out


def run(net, wns, train):
    D, S, C, yv, m, wg, DX = make_batch(wns)
    if train:
        D = D + 0.1 * torch.randn_like(D)                          # 軽い座標ノイズ(v62)
    o = net(D, S, C, m, DX)
    out, w = (o[0], o[1]) if isinstance(o, tuple) else (o, None)   # l7 は (out, w)
    ok = m.bool() & torch.isfinite(yv) & (wg > 0)
    if not ok.any(): return None
    wgn = wg / wg[ok].mean()
    if ARCH == "l5b" and train:
        # ★MCL(hindsight): 勝ちヘッドのみ回帰罰(敗者はeps重み=崩壊防止)+ ゲートは勝者を予測
        outA, outB, glogit = w
        yv2 = torch.nan_to_num(yv)
        eA = (outA - yv2) ** 2; eB = (outB - yv2) ** 2
        aWin = (eA <= eB).float()
        wA = aWin * (1.0 - L5B_EPS) + (1.0 - aWin) * L5B_EPS
        reg = wA * eA + (1.0 - wA) * eB
        gbce = torch.nn.functional.binary_cross_entropy_with_logits(glogit, aWin, reduction="none")
        return ((reg + L5B_GW * gbce) * wgn)[ok].mean()
    mse = ((out - torch.nan_to_num(yv)) ** 2 * wgn)[ok].mean()     # L4/L5/L6/l5b(val) = 重み付き MSE(l5bはゲート後out)
    if ARCH == "l7":
        # v62 と同一の listwise ranking: 正解に近い候補ほど高確率のターゲット分布との CE
        dist = (D - torch.nan_to_num(yv)[..., None]).abs()
        qt = torch.softmax(-dist / TAU_R, -1)
        ce = -(qt * torch.log(w.clamp_min(1e-9))).sum(-1)
        return (ce * wgn)[ok].mean() + 0.1 * mse
    return mse


def val_loss(net, val):
    net.eval(); tot = 0.0; n = 0
    with torch.no_grad():
        for wns in bucket(val, np.random.default_rng(1)):
            l = run(net, wns, False)
            if l is None: continue
            tot += float(l) * len(wns); n += len(wns)
    return tot / max(n, 1)


def train_fold(trn, val, seed):
    """v62 の学習ループを 1 fold 分再現。best val state を返す。"""
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    net = ARCHES[ARCH]().to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=LRPAT)
    rng = np.random.default_rng(seed)
    best = float("inf"); best_state = None; bad = 0; ep = 0; t1 = time.time()
    while True:
        ep += 1; net.train(); tl = []
        for wns in bucket(trn, rng):
            try:
                l = run(net, wns, True)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache(); continue
            if l is None or not torch.isfinite(l): continue
            opt.zero_grad(set_to_none=True); l.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step(); tl.append(float(l.detach()))
        vl = val_loss(net, val); sch.step(vl); star = ""
        if vl < best - 1e-4:
            best = vl; best_state = {k_: v.detach().clone() for k_, v in net.state_dict().items()}
            bad = 0; star = " *best"
        else:
            bad += 1
        print(f"  s{seed} ep{ep} train={np.mean(tl) if tl else float('nan'):.3f} val={vl:.3f}{star} "
              f"(bad={bad}, {time.time()-t1:.0f}s)", flush=True)
        if bad >= PAT or ep >= MAXEP or (time.time() - t1) / 60 > MAXMIN:
            print(f"  s{seed} stop @ep{ep} best={best:.3f}", flush=True)
            break
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    return net


def predict_wells(net, wells):
    """val well -> per-row 予測(TARGET 単位)を {well: (rows, pred_row)} で返す。"""
    out = {}
    with torch.no_grad():
        for w in wells:
            f = SF[w]; T = f["D"].shape[0]
            D, S, C, yv, m, wg, DX = make_batch([w])
            o = net(D, S, C, m, DX)
            o = o[0] if isinstance(o, tuple) else o                       # l7 は (out, w)
            dh = o[0, :T].cpu().numpy().astype(np.float64)                # per-bin 予測
            pred_row = dh[f["bid"]]                                        # per-row に展開
            out[w] = (f["rows"], pred_row)
    return out


# ======================= per-row GroupKFold(L1/L2 と同一)で well->fold =======================
def build_folds():
    meta = pq.read_table(PQ, columns=["well", "target"])
    g = meta.column("well").to_pandas().to_numpy()
    y = meta.column("target").to_numpy(zero_copy_only=False).astype(np.float64)
    del meta; gc.collect()
    N = len(g)
    folds = list(GroupKFold(n_splits=N_SPLITS).split(np.zeros(N), y, groups=g))
    fold_of = {}
    import pandas as pd
    for k, (tr, va) in enumerate(folds):
        for w in pd.unique(g[va]):
            fold_of[w] = k
    for w in AUG_KEYS:                       # ★aug は real well の fold を継承(train側専用)
        real = w[:-3]
        if real in fold_of:
            fold_of[w] = fold_of[real]
    return N, y, fold_of


# ======================= 実行 =======================
if SMOKE:
    # スモーク: 全 well GroupKFold は重い(N=3.78M string)ので回避。~40 well を簡易分割。
    N = pq.ParquetFile(PQ).metadata.num_rows
    y = pq.read_table(PQ, columns=["target"]).column("target").to_numpy(
        zero_copy_only=False).astype(np.float64)
    WELLS = WELLS_ALL[:40]
    val = WELLS[0::5]; trn = [w for w in WELLS if w not in set(val)]
    print(f"[SMOKE] N={N} trn={len(trn)} val={len(val)} maxep={MAXEP}", flush=True)
    oof = np.zeros(N, np.float64)
    net = train_fold(trn, val, SEEDS[0])
    preds = predict_wells(net, val)
    for w, (rows, pr) in preds.items():
        oof[rows] = pr
    vmask = np.concatenate([preds[w][0] for w in val])
    rm = float(np.sqrt(np.mean((oof[vmask] - y[vmask]) ** 2)))
    print(f"[SMOKE] one seed/fold done. oof.shape={oof.shape} (==N:{oof.shape[0]==N}) "
          f"val-rows={len(vmask)} val rmse(per-row target)={rm:.4f}")
    print("[SMOKE] OK: model builds + forward/backward ran + OOF shape correct.")
    sys.exit(0)

# --- 本番: per-row GroupKFold(L1/L2 整列)で全 seed x fold ---
N, y, fold_of = build_folds()
WELLS = [w for w in WELLS_ALL if w in fold_of]
missing = [w for w in WELLS_ALL if w not in fold_of]
if missing:
    print(f"[warn] {len(missing)} wells not in fold map (skipped): {missing[:5]}", flush=True)
print(f"folds built: N={N} wells_used={len(WELLS)} "
      f"(fold sizes: {[sum(1 for w in WELLS if fold_of[w]==k) for k in range(N_SPLITS)]})", flush=True)

seed_oofs = {}
for seed in SEEDS:
    oof = np.zeros(N, np.float64); filled = np.zeros(N, bool)
    _fold_only = os.environ.get("FOLD_ONLY")
    for k in range(N_SPLITS):
        if _fold_only is not None and k != int(_fold_only):
            continue                                                        # ★A/B用: 指定foldのみ学習
        cache = CKPT / f"{ARCH}_s{seed}_f{k}_valpred.npz"
        val = [w for w in WELLS if fold_of[w] == k and w not in AUG_KEYS]   # ★val=main(順)のみ
        trn = [w for w in WELLS if fold_of[w] != k]                          # ★train=main+aug(fold≠kでaug自動含)
        if cache.exists():
            z = np.load(cache)
            oof[z["rows"]] = z["pred"]; filled[z["rows"]] = True
            print(f"seed{seed} fold{k} [load cache] val={len(val)}", flush=True)
            continue
        print(f"seed{seed} fold{k} [train] trn={len(trn)} val={len(val)}", flush=True)
        net = train_fold(trn, val, seed * 100 + k)
        preds = predict_wells(net, val)
        all_rows = np.concatenate([preds[w][0] for w in val])
        all_pred = np.concatenate([preds[w][1] for w in val])
        oof[all_rows] = all_pred; filled[all_rows] = True
        np.savez_compressed(cache, rows=all_rows, pred=all_pred)
        # ★test 推論用にモデル state_dict を保存(K/ARCH/DIM は推論側で一致させる)
        torch.save({"state": net.state_dict(), "K": K, "arch": ARCH, "dim": DIM},
                   CKPT / f"{ARCH}_s{seed}_f{k}_model.pt")
        del net; gc.collect()
        if DEV == "cuda": torch.cuda.empty_cache()
    rm = float(np.sqrt(np.mean((oof[filled] - y[filled]) ** 2)))
    seed_oofs[seed] = oof
    print(f"== seed{seed} OOF per-row(target) RMSE = {rm:.4f}  (filled {filled.sum()}/{N})", flush=True)

# --- 保存 + 3-seed アンサンブル ---
mean_oof = np.mean([seed_oofs[s] for s in SEEDS], axis=0)
save = {f"{ARCH}_s{s}": seed_oofs[s].astype(np.float32) for s in SEEDS}
save[f"{ARCH}_mean"] = mean_oof.astype(np.float32)
save["y"] = y.astype(np.float32)
np.savez_compressed(ART / f"{ARCH}{_osuf}_oof.npz", **save)   # variant 毎に別出力
finite = np.isfinite(mean_oof) & (mean_oof != 0)
rm_ens = float(np.sqrt(np.mean((mean_oof - y) ** 2)))
print(f"\n=== v93 {ARCH.upper()} (struct-free segment-level leg) ===")
for s in SEEDS:
    print(f"  seed{s:d} OOF RMSE = {np.sqrt(np.mean((seed_oofs[s]-y)**2)):.4f}")
print(f"  {len(SEEDS)}-seed ensemble OOF RMSE = {rm_ens:.4f}")
print(f"  saved {ART / (ARCH + '_oof.npz')}  keys={list(save.keys())}")
