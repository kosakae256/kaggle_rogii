# -*- coding: utf-8 -*-
"""
v40 微分可能グリッド・ベイズフィルタ(学習型emission × 物理prior)
====================================================================
固定PFのGR尤度を「共有小TCNの学習型emission」に置換し、構造幾何prior(v38)で暴走を止めつつ
end-to-endで真TVTに最適化。信念=偏差グリッド上の分布(多峰=bimodal datumを保持)。

  belief0 = softmax(-0.5 (d/sig0)^2)
  for t:  bp = belief @ Tmat(diffusion);  bp = (1-leak)*bp + leak*prior_t;  bp/=sum
          L = gate_t * (-0.5 ((gr_t - eg[t,:])/sig_t)^2)         # eg=typewell(struct_t+d)
          belief = normalize(bp * exp(L));  pred_dev_t = sum(belief*d)
  pred_tvt = struct_tvt + pred_dev

★転移安全: emission NN は局所・相対のGR文脈のみ(XY/field/well-id を見ない)→ フィールド指紋を記憶不能。
依存: v38/struct_prior_train.pkl / v28 parquet(整列・fold・gbm blend用) / 公式CSV。
実行: rogii_claude/.venv/Scripts/python.exe v40/train_grid_filter_v40.py
"""
import os, json, math, time, gc, copy, glob, pickle, warnings
from pathlib import Path
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.model_selection import GroupKFold
warnings.filterwarnings("ignore"); np.seterr(all="ignore")

# ============================== CONFIG ==============================
PROJECT = Path(r"C:\Users\kosaka256\Documents\rogii_claude")
DATA    = PROJECT / "rogii-wellbore-geology-prediction" / "train"
V28_ART = PROJECT / "v28" / "artifacts_v28"
V29_ART = PROJECT / "v29" / "artifacts_v29"
V36_ART = PROJECT / "v36" / "artifacts_v36"
ART     = PROJECT / "v40" / "artifacts_v40"; (ART / "ck").mkdir(parents=True, exist_ok=True)
STRUCT_PKL = PROJECT / "v38" / "artifacts_v38" / "struct_prior_train.pkl"
HARDCORE_CSV = PROJECT / "v32" / "artifacts_v32" / "hardcore_diag.csv"
PARQUET = V28_ART / "data" / "train.parquet"

DMAX, DSTEP = 32.0, 0.5              # 偏差グリッド ±32ft / 0.5ft = 129ビン
HGR = 2                              # 坑井GR 平滑半径
N_SPLITS = 5
SEEDS = [0]
EPOCHS = 40
LR, WD, BATCH_WELLS, CROP = 3e-4, 1e-4, 12, 768
CH_EMIT, EMIT_DIL, DROP = 48, [1, 2, 4, 8, 16, 32], 0.05
EMA_DECAY, CLIP, WARM_FRAC, NOISE = 0.999, 1.0, 0.05, 0.0
EVAL_TOK = 40_000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DGRID = np.arange(-DMAX, DMAX + 1e-6, DSTEP).astype(np.float64); G = len(DGRID)
pp_params = json.loads((V36_ART / "pp_params.json").read_text()) if (V36_ART / "pp_params.json").exists() else {"alpha":1.0,"tau":85,"w_pf":0.09}
print(f"v40 | device={DEVICE} | grid G={G} (±{DMAX}/{DSTEP}ft) | CROP={CROP}")

# ============================== 整列・fold(parquet) ==============================
import pyarrow.parquet as pq
_t = pq.read_table(PARQUET, columns=["well", "md_since", "last_known_tvt", "pf_ancc"]).to_pydict()
g_par = np.array(_t["well"]); mds_par = np.asarray(_t["md_since"], float)
lk_par = np.asarray(_t["last_known_tvt"], float); pf_par = np.asarray(_t["pf_ancc"], float)
Npar = len(g_par)
wid_order = pd.unique(g_par)
well_rows = {w: (np.where(g_par == w)[0])[np.argsort(mds_par[np.where(g_par == w)[0]], kind="stable")] for w in wid_order}
folds_row = list(GroupKFold(n_splits=N_SPLITS).split(np.zeros(Npar), np.zeros(Npar), groups=g_par))
well2fold = {}
for k, (_, va) in enumerate(folds_row):
    for w in pd.unique(g_par[va]): well2fold[w] = k
gbm_oof = np.load(V36_ART / "gbm_v36_oof.npy").astype(np.float64) if (V36_ART / "gbm_v36_oof.npy").exists() else None
try:
    hd = pd.read_csv(HARDCORE_CSV); HARD = set(hd[hd["hard"] == True].iloc[:, 0].astype(str))
except Exception:
    HARD = set()

# ============================== 坑井キャッシュ(CSV + struct) ==============================
STRUCT = pickle.loads(STRUCT_PKL.read_bytes())
def _smooth(v, r):
    if r <= 0: return v
    return pd.Series(v).rolling(2 * r + 1, center=True, min_periods=1).mean().to_numpy()

CACHE = {}; skipped = 0
t0 = time.time()
for w in wid_order:
    sp = STRUCT.get(w)
    if sp is None: skipped += 1; continue
    fp = DATA / f"{w}__horizontal_well.csv"; tp = DATA / f"{w}__typewell.csv"
    if not fp.exists() or not tp.exists(): skipped += 1; continue
    hw = pd.read_csv(fp, usecols=["Z", "GR", "TVT_input", "TVT"])
    ti = hw["TVT_input"].to_numpy(); evm = ~np.isfinite(ti)
    n_ev = int(evm.sum())
    if n_ev != len(sp["tvt"]) or n_ev != len(well_rows[w]): skipped += 1; continue
    tw = pd.read_csv(tp, usecols=["TVT", "GR"]).sort_values("TVT")
    tw_tvt = tw["TVT"].to_numpy(float); tw_gr = tw["GR"].to_numpy(float)
    tw_fb = float(np.nanmedian(tw_gr)); tw_iqr = float(max(np.nanpercentile(tw_gr, 75) - np.nanpercentile(tw_gr, 25), 1e-6))
    gr_all = pd.Series(hw["GR"].to_numpy(float)).interpolate(limit_direction="both").fillna(tw_fb).to_numpy()
    gr_ev = _smooth(gr_all, HGR)[evm]
    true_ev = hw["TVT"].to_numpy(float)[evm]
    stv = np.asarray(sp["tvt"], float); sig = np.asarray(sp["sig"], float)
    lk = float(stv[0] - 0.0)  # struct[0]=last_known+resid[0]; anchor=last_known_tvt(=parquet lkと一致)
    lk = float(lk_par[well_rows[w][0]])
    eg = np.interp((stv[:, None] + DGRID[None, :]).ravel(), tw_tvt, tw_gr).reshape(n_ev, G).astype(np.float16)
    gr_n = np.clip((gr_ev - tw_fb) / tw_iqr, -8, 8)
    dgr = np.gradient(gr_n)
    twstd = eg.astype(np.float32).std(axis=1)          # typewell局所識別力(偏差方向のGR変動)
    CACHE[w] = dict(gr=gr_ev.astype(np.float32), eg=eg, stv=stv.astype(np.float32),
                    sig=np.clip(sig, 1e-3, 60).astype(np.float32), true=true_ev.astype(np.float32),
                    gr_n=gr_n.astype(np.float32), dgr=dgr.astype(np.float32), twstd=twstd, lk=lk)
ALL = list(CACHE.keys())
print(f"cache: {len(ALL)} wells (skip {skipped}) {time.time()-t0:.0f}s")
# 文脈の大域正規化(twstd, sig)
_tw = np.concatenate([CACHE[w]["twstd"] for w in ALL]); tw_m, tw_s = float(np.median(_tw)), float(_tw.std() + 1e-6)
_sg = np.concatenate([CACHE[w]["sig"] for w in ALL]); sg_m, sg_s = float(np.median(_sg)), float(_sg.std() + 1e-6)
for w in ALL:
    c = CACHE[w]
    c["ctx"] = np.stack([c["gr_n"], c["dgr"], (c["twstd"] - tw_m) / tw_s, (c["sig"] - sg_m) / sg_s], axis=1).astype(np.float32)  # (n,4)
del g_par; gc.collect()

# ============================== モデル ==============================
class ChanLN(nn.Module):
    def __init__(self, c): super().__init__(); self.ln = nn.LayerNorm(c)
    def forward(self, x): return self.ln(x.transpose(1, 2)).transpose(1, 2)
class Blk(nn.Module):
    def __init__(self, c, d, k=5, drop=0.05):
        super().__init__(); self.conv = nn.Conv1d(c, c, k, padding=d * (k - 1) // 2, dilation=d)
        self.ln = ChanLN(c); self.pw = nn.Conv1d(c, c, 1); self.do = nn.Dropout(drop)
    def forward(self, x, m):
        h = self.conv(x * m); h = F.gelu(h); h = self.ln(h); h = self.do(self.pw(h)); return (x + h) * m
class GridFilter(nn.Module):
    def __init__(self, G, dgrid):
        super().__init__()
        self.register_buffer("d", torch.tensor(dgrid, dtype=torch.float32))
        self.inp = nn.Conv1d(4, CH_EMIT, 1)
        self.blocks = nn.ModuleList([Blk(CH_EMIT, d, drop=DROP) for d in EMIT_DIL])
        self.head = nn.Conv1d(CH_EMIT, 2, 1)
        self.log_diff = nn.Parameter(torch.tensor(math.log(2.0 / DSTEP)))   # 拡散幅(グリッド単位)
        self.leak_logit = nn.Parameter(torch.tensor(-2.0))
        self.log_sig_base = nn.Parameter(torch.tensor(math.log(10.0)))
        with torch.no_grad(): self.head.bias[0] = 2.0                        # gate 初期 ~1(GRを一旦信じる)
    def emit(self, ctx, m):
        h = self.inp(ctx * m)
        for b in self.blocks: h = b(h, m)
        o = self.head(h)
        gate = torch.sigmoid(o[:, 0])                                        # (B,T)
        sigma = torch.exp(self.log_sig_base + o[:, 1]).clamp(2.0, 60.0)      # (B,T)
        return gate, sigma
    def tmat(self):
        dd = (self.d[None, :] - self.d[:, None]) / (F.softplus(self.log_diff) + 1e-3)
        K = torch.exp(-0.5 * dd * dd); return K / K.sum(-1, keepdim=True).clamp_min(1e-20)
    def forward(self, ctx, gr, eg, sigst, stv, m):
        # ctx(B,4,T) gr(B,T) eg(B,T,G) sigst(B,T) stv(B,T) m(B,1,T)
        gate, sigma = self.emit(ctx, m)                                      # (B,T)
        Tm = self.tmat(); dgrid = self.d[None, :]                            # (G,G),(1,G)
        B, T = gr.shape; mm = m[:, 0, :]                                     # (B,T)
        s0 = sigst[:, 0:1].clamp_min(1e-3)
        belief = torch.softmax(-0.5 * (dgrid / s0) ** 2, dim=-1)             # (B,G)
        leak = torch.sigmoid(self.leak_logit)
        preds = torch.zeros(B, T, device=gr.device, dtype=torch.float32)
        for t in range(T):
            bp = belief @ Tm
            st = sigst[:, t:t + 1].clamp_min(1e-3)
            prior_t = torch.softmax(-0.5 * (dgrid / st) ** 2, dim=-1)
            bp = (1 - leak) * bp + leak * prior_t
            bp = bp / bp.sum(-1, keepdim=True).clamp_min(1e-20)
            r = (gr[:, t:t + 1] - eg[:, t, :]) / sigma[:, t:t + 1]
            L = -0.5 * r * r * gate[:, t:t + 1]
            logb = torch.log(bp.clamp_min(1e-20)) + L
            logb = logb - logb.max(-1, keepdim=True).values
            bn = torch.exp(logb); bn = bn / bn.sum(-1, keepdim=True).clamp_min(1e-20)
            mt = mm[:, t:t + 1]
            belief = mt * bn + (1 - mt) * belief
            preds[:, t] = (belief * dgrid).sum(-1)
        return stv + preds                                                    # pred_tvt (B,T)

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

# ============================== バッチ ==============================
def make_batch(wids, train, rng):
    sel = []
    for w in wids:
        n = len(CACHE[w]["gr"]); a, b = 0, n
        if train and n > CROP: a = int(rng.integers(0, n - CROP + 1)); b = a + CROP
        sel.append((w, a, b))
    L = max(b - a for _, a, b in sel); B = len(sel)
    ctx = np.zeros((B, 4, L), np.float32); gr = np.zeros((B, L), np.float32); eg = np.zeros((B, L, G), np.float32)
    sig = np.ones((B, L), np.float32); stv = np.zeros((B, L), np.float32); tru = np.zeros((B, L), np.float32); m = np.zeros((B, L), np.float32)
    for i, (w, a, b) in enumerate(sel):
        c = CACHE[w]; L0 = b - a
        ctx[i, :, :L0] = c["ctx"][a:b].T; gr[i, :L0] = c["gr"][a:b]; eg[i, :L0] = c["eg"][a:b].astype(np.float32)
        sig[i, :L0] = c["sig"][a:b]; stv[i, :L0] = c["stv"][a:b]; tru[i, :L0] = c["true"][a:b]; m[i, :L0] = 1.0
    to = lambda x: torch.from_numpy(x).to(DEVICE)
    return (to(ctx), to(gr), to(eg), to(sig), to(stv), to(tru), to(m[:, None, :]), sel)

@torch.no_grad()
def predict(model, wids):
    model.eval(); out = {}
    order = sorted(wids, key=lambda w: len(CACHE[w]["gr"])); i = 0
    while i < len(order):
        Lm = len(CACHE[order[i]]["gr"]); j = i + 1
        while j < len(order):
            Lc = max(Lm, len(CACHE[order[j]]["gr"]))
            if (j - i + 1) * Lc > EVAL_TOK: break
            Lm = Lc; j += 1
        ctx, gr, eg, sig, stv, tru, m, sel = make_batch(order[i:j], False, None)
        with torch.autocast(DEVICE, dtype=torch.bfloat16, enabled=(DEVICE == "cuda")):
            p = model(ctx, gr, eg, sig, stv, m).float().cpu().numpy()
        for b, (w, a, e) in enumerate(sel): out[w] = p[b, :e - a]
        i = j
    return out

def cosine_lr(step, total, warm):
    if step < warm: return step / max(1, warm)
    t = (step - warm) / max(1, total - warm); return 0.5 * (1 + math.cos(math.pi * t))

def masked_mse(pred, tru, m):
    e = (pred - tru) * m[:, 0, :]; return (e * e).sum() / m.sum().clamp_min(1.0)

# ============================== 学習 ==============================
def train_fold(seed, k):
    vaw = [w for w in ALL if well2fold[w] == k]; trw = [w for w in ALL if well2fold[w] != k]
    fp = ART / "ck" / f"v40_s{seed}_fold{k}.pt"
    model = GridFilter(G, DGRID).to(DEVICE)
    if fp.exists():
        model.load_state_dict(torch.load(fp, map_location=DEVICE)); return predict(model, vaw), "load"
    torch.manual_seed(1000 * seed + k); rng = np.random.default_rng(1000 * seed + k)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    total = math.ceil(len(trw) / BATCH_WELLS) * EPOCHS; warm = int(total * WARM_FRAC)
    ema = EMA(model, EMA_DECAY); best = (1e18, None); step = 0
    for ep in range(EPOCHS):
        model.train(); order = rng.permutation(len(trw))
        for i in range(0, len(order), BATCH_WELLS):
            ctx, gr, eg, sig, stv, tru, m, _ = make_batch([trw[j] for j in order[i:i + BATCH_WELLS]], True, rng)
            for pg in opt.param_groups: pg["lr"] = LR * cosine_lr(step, total, warm)
            with torch.autocast(DEVICE, dtype=torch.bfloat16, enabled=(DEVICE == "cuda")):
                pred = model(ctx, gr, eg, sig, stv, m)
            loss = masked_mse(pred.float(), tru, m)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP); opt.step(); ema.update(model); step += 1
        bak = copy.deepcopy(model.state_dict()); ema.copy_to(model)
        pr = predict(model, vaw)
        e = np.concatenate([pr[w] - CACHE[w]["true"] for w in vaw]); rm = float(np.sqrt(np.mean(e ** 2)))
        if rm < best[0]: best = (rm, copy.deepcopy(model.state_dict()))
        model.load_state_dict(bak)
        print(f"[v40] s{seed} f{k} ep{ep} val_overall={rm:.4f} (best {best[0]:.4f})", flush=True)
    model.load_state_dict(best[1]); torch.save(best[1], fp)
    return predict(model, vaw), f"best={best[0]:.4f}"

def rmse(pred, true, mask=None):
    e = pred - true; e = e if mask is None else e[mask]; e = e[np.isfinite(e)]
    return float(np.sqrt(np.mean(e ** 2)))

for seed in SEEDS:
    oof_tvt = {}; t0 = time.time()
    for k in range(N_SPLITS):
        pr, tag = train_fold(seed, k); oof_tvt.update(pr)
        gc.collect(); torch.cuda.empty_cache() if DEVICE == "cuda" else None
        print(f"[v40] s{seed} fold{k} [{tag}]", flush=True)
    # overall / hard(全行プール, 坑井の真値から)
    allp = np.concatenate([oof_tvt[w] for w in ALL]); allt = np.concatenate([CACHE[w]["true"] for w in ALL])
    hardm = np.concatenate([np.full(len(CACHE[w]["true"]), w in HARD) for w in ALL])
    print(f"\n== [v40 s{seed}] overall={rmse(allp, allt):.4f}  hard={rmse(allp, allt, hardm):.4f}  ({time.time()-t0:.0f}s)")
    # parquet整列の残差 oof を保存(gbm/v39 blend用)
    oof_res = np.full(Npar, np.nan)
    for w in ALL: oof_res[well_rows[w]] = oof_tvt[w] - CACHE[w]["lk"]
    np.save(ART / f"v40_oof_resid_s{seed}.npy", oof_res)
    print(f"saved {ART/f'v40_oof_resid_s{seed}.npy'}")

# ============================== gbm(v36)ブレンド(pp+SG, 即go/no-go) ==============================
if gbm_oof is not None:
    from scipy.signal import savgol_filter
    PP = pp_params; SG_W, SG_P = 17, 3
    y_abs = lk_par + gbm_oof * 0  # placeholder; 真値は CACHE から parquet整列で作る
    true_res = np.full(Npar, np.nan)
    for w in ALL: true_res[well_rows[w]] = CACHE[w]["true"] - CACHE[w]["lk"]
    ok = np.isfinite(true_res) & np.isfinite(oof_res)
    def chain(res):
        d = res * (1 - PP["w_pf"]) + (pf_par - lk_par) * PP["w_pf"]
        if PP.get("tau"): d = d * (1.0 - np.exp(-np.maximum(mds_par, 0.0) / float(PP["tau"])))
        tvt = lk_par + d * float(PP.get("alpha", 1.0)); out = tvt.copy()
        for w in ALL:
            idx = well_rows[w]; v = tvt[idx]; nn_ = len(v); wl = min(SG_W, nn_)
            if wl % 2 == 0: wl -= 1
            if wl >= SG_P + 2: out[idx] = savgol_filter(v, wl, SG_P)
        return out
    hard_all = np.array([w in HARD for w in g_par]) if False else None
    true_abs = lk_par + true_res
    hmask = np.zeros(Npar, bool)
    for w in ALL:
        if w in HARD: hmask[well_rows[w]] = True
    def ov(res, m=None):
        e = (chain(res) - true_abs); e = e[ok] if m is None else e[ok & m]; e = e[np.isfinite(e)]
        return float(np.sqrt(np.mean(e ** 2)))
    g_ov, g_hd = ov(gbm_oof), ov(gbm_oof, hmask)
    print(f"\n[blend] v36-GBM単体: overall={g_ov:.4f} hard={g_hd:.4f}")
    best = (g_ov, 0.0)
    for wq in np.round(np.arange(0, 0.71, 0.05), 2):
        r = ov((1 - wq) * gbm_oof + wq * oof_res); h = ov((1 - wq) * gbm_oof + wq * oof_res, hmask)
        print(f"   w={wq:.2f} overall={r:.4f} hard={h:.4f}" + (" *" if r < best[0] else ""))
        if r < best[0]: best = (r, wq)
    print(f"[blend] best overall={best[0]:.4f} at w={best[1]:.2f} (Δ {best[0]-g_ov:+.4f})")
