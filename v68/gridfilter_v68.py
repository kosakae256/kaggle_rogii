# -*- coding: utf-8 -*-
"""
v68 GF-MR — 多表現マッチテンソル × 学習emission × 微分可能グリッドベイズフィルタ
=====================================================================
「全表現を1つのモデルに入れて oracle を狙う」単一メソッド。
  - 各 eval 行 i × 候補オフセット δ(struct 中心)で、全表現のマッチコストを channel として積む:
      tw-raw / tw-MA / self / self-valid / nbr / nbr-valid / struct-prior
  - **学習emission**(小さい共有MLP: C→H→1)が (i,δ) 毎に全 channel を融合 → emission logit。
    ★真δは tw+self+nbr の全chで合意/偽δ(cycle alias)は tw だけ → emission が真δで鋭くなる(disambiguation)。
  - **微分可能グリッドベイズフィルタ**(forward-backward, Gaussian遷移=dip連続 prior)で δ の posterior。
    posterior-mean δ を出力(双峰ヘッジ, error_anatomy §4)→ TVT = struct_tvt + pmean。
  - **転移ガードレール**: emission は小容量MLP(過学習面小)+ **leave-field-out 検証**(NN脚のLB非転移を早期検出, v34/v67)。
物理(グリッドフィルタ)は転移する構造、学習は emission だけ = v64 全CNNの field過学習を避ける設計。

やること: このスクリプトは train(OOF+leave-field-out)まで。実行はユーザー(CLAUDE.md)。
実行(★EVAL_STRIDE はキャッシュ時に効く=cache名に埋め込み。stride を変えたら cache/train で同じ値を渡す。
  train だけに付けても、その stride の cache が無ければ自動で作る):
  EVAL_STRIDE=4 .venv/Scripts/python.exe v68/gridfilter_v68.py cache          # 間引きグリッド構築(T→1/4, 初回のみ)
  CV_MODE=field EVAL_STRIDE=4 .venv/Scripts/python.exe v68/gridfilter_v68.py train
★メモリ注意: build_channels が (B,T,D=91,7ch) を実体化 → VRAM ∝ B×T。B=8/T=4847で~0.1GB(可)、
  B=48だと~5.9GB(+勾配で溢れる)。BATCHは8前後に据え置き、速度はEVAL_STRIDE(T削減=メモリも削減)で稼ぐ。
env: CV_MODE(field|well) N_FIELDS(16) EPOCHS(15) BATCH(8) LR(2e-3) HID(24)
     DR(45) DSTEP(1) GSTEP(0.5) GR_SIG(15) TRANS_SIG0(2.0) MA(10) MAXNBR(3) NBR_DIST(1500)
     N_WELLS(0=全; スモーク) SEED(20250714) DEVICE(auto)
     EVAL_STRIDE(1) ★eval行をstride間引き=T(逐次ループ長)を1/strideに=速度∝1/stride。cache時に適用(cache名にs{stride})。
                     4-8推奨(MD間引きと同思想=denoise+高速。CV採点は間引き行のRMSE≈全行RMSE)
出力: v68/artifacts_v68/gf_oof_{cv}.npy + results_{cv}.json + png(oracle/struct比較)
"""
import os, sys, json, glob, pickle, math, time
from pathlib import Path
import numpy as np, pandas as pd
sys.stdout = open(1, "w", encoding="utf-8", closefd=False)
PROJECT = Path(r"C:\Users\kosaka256\Documents\rogii_claude")
OUT = PROJECT / "v68" / "artifacts_v68"; OUT.mkdir(parents=True, exist_ok=True)
TRAIN_DIR = PROJECT / "rogii-wellbore-geology-prediction" / "train"
STRUCT_PKL = PROJECT / "v38" / "artifacts_v38" / "priors" / "prior_v66_leanv1cumul_train.pkl"

CV_MODE  = os.environ.get("CV_MODE", "field")
N_FIELDS = int(os.environ.get("N_FIELDS", "16"))
EPOCHS   = int(os.environ.get("EPOCHS", "15"))
BATCH    = int(os.environ.get("BATCH", "8"))
LR       = float(os.environ.get("LR", "2e-3"))
HID      = int(os.environ.get("HID", "24"))
DR       = float(os.environ.get("DR", "45"))
DSTEP    = float(os.environ.get("DSTEP", "1"))
GSTEP    = float(os.environ.get("GSTEP", "0.5"))    # 参照グリッド刻み(共通TVT軸)
GR_SIG   = float(os.environ.get("GR_SIG", "15"))
TRANS_SIG0 = float(os.environ.get("TRANS_SIG0", "2.0"))   # 遷移(dip連続)初期σ(grid単位)
MA       = int(os.environ.get("MA", "10"))
MAXNBR   = int(os.environ.get("MAXNBR", "3"))
NBR_DIST = float(os.environ.get("NBR_DIST", "1500"))
N_WELLS  = int(os.environ.get("N_WELLS", "0")) or None
SEED     = int(os.environ.get("SEED", "20250714"))
EVAL_STRIDE = int(os.environ.get("EVAL_STRIDE", "4"))
USE_STRUCT = os.environ.get("STRUCT", "0") == "1"       # 1=struct窓中心+struct ch / 0=struct-free(base_free中心, 6ch)
CENTER_KEY = "s_tvt" if USE_STRUCT else "base_free"
NCH = 7 if USE_STRUCT else 6   # ch: tw-raw, tw-ma, self, self-valid, nbr, nbr-valid [, struct]
# struct-free は錨(base_free=flat hold≈naive 15)が弱い→窓を広く+粗く(構造残差を包む=well-posed化)
DR_FREE    = float(os.environ.get("DR_FREE", "150"))    # struct無し時のδ半幅(広い)
DSTEP_FREE = float(os.environ.get("DSTEP_FREE", "2"))   # struct無し時のδ刻み(粗く=bin数を抑える)
DR_EFF    = DR    if USE_STRUCT else DR_FREE            # 実効δ半幅(モード依存)
DSTEP_EFF = DSTEP if USE_STRUCT else DSTEP_FREE
GRID_MARGIN = max(DR, DR_FREE) + 5                       # cacheグリッドは両モードを包む(=1cacheで両対応)
CACHE = OUT / f"cache_v68_s{EVAL_STRIDE}.pkl"   # ★strideをファイル名に=train時にstride指定で正しい間引きcacheを自動使用/dense黙殺防止


def _ma(x, w):
    if w <= 1: return np.asarray(x, float)
    return pd.Series(x, dtype="float64").rolling(w, center=True, min_periods=1).mean().to_numpy()

def _bin_to_grid(tvt, gr, g0, gn):
    """(TVT,GR) を共通TVTグリッド[g0 + j*GSTEP]に median 集約。返: (grid値, valid)。"""
    out = np.full(gn, np.nan, np.float32)
    m = np.isfinite(tvt) & np.isfinite(gr); tvt, gr = tvt[m], gr[m]
    if len(gr):
        idx = np.round((tvt - g0) / GSTEP).astype(int); ok = (idx >= 0) & (idx < gn)
        s = pd.Series(gr[ok]).groupby(idx[ok]).median()
        out[s.index.to_numpy()] = s.to_numpy().astype(np.float32)
    val = np.isfinite(out).astype(np.float32); out = np.nan_to_num(out, nan=0.0)
    return out, val


def _centroids():
    fp = OUT / "centroids.json"
    if fp.exists():
        d = json.loads(fp.read_text()); return d["wids"], np.array(d["xy"])
    wids, xy = [], []
    for p in sorted(glob.glob(str(TRAIN_DIR / "*__horizontal_well.csv"))):
        h = pd.read_csv(p, usecols=["X", "Y"]); wids.append(os.path.basename(p).split("__")[0])
        xy.append((float(h["X"].mean()), float(h["Y"].mean())))
    fp.write_text(json.dumps(dict(wids=wids, xy=[list(p) for p in xy]))); return wids, np.array(xy)


def build_cache():
    struct = pickle.loads(STRUCT_PKL.read_bytes())
    cwids, cxy = _centroids()
    fps = sorted(glob.glob(str(TRAIN_DIR / "*__horizontal_well.csv")))
    if N_WELLS: fps = fps[:N_WELLS]
    wells = []
    for fp in fps:
        wid = os.path.basename(fp).split("__")[0]
        sp = struct.get(wid)
        if sp is None: continue
        hw = pd.read_csv(fp); tw = pd.read_csv(fp.replace("__horizontal_well", "__typewell")).sort_values("TVT")
        if "TVT" not in hw.columns or hw["TVT"].isna().all(): continue
        kn = hw[hw["TVT_input"].notna()]; ev = hw[hw["TVT_input"].isna()]
        if len(kn) < 10 or len(ev) < 5: continue
        nev_full = len(ev)
        if len(sp["tvt"]) != nev_full: continue
        ev = ev.iloc[::EVAL_STRIDE]; keep = np.arange(nev_full)[::EVAL_STRIDE]
        s_tvt = np.asarray(sp["tvt"], np.float32)[keep]; s_sig = np.asarray(sp["sig"], np.float32)[keep]
        # struct-free 窓中心: 末端既知TVTを水平保持(flat hold=naive baseline≈15, 有界)。構造残差は広いδ窓で拾う
        tvk = kn["TVT_input"].to_numpy(float); tvk = tvk[np.isfinite(tvk)]
        base_free = np.full(len(s_tvt), float(tvk[-1]) if len(tvk) else float(s_tvt[0]), np.float32)
        tw_tvt = tw["TVT"].to_numpy(float); tw_gr = tw["GR"].to_numpy(float)
        m = np.isfinite(tw_tvt) & np.isfinite(tw_gr); tw_tvt, tw_gr = tw_tvt[m], tw_gr[m]
        if len(tw_tvt) < 5: continue
        # 共通グリッド: struct±DR と struct-free anchor±DR_FREE と typewell を包む(GRID_MARGIN=両モード対応)
        g0 = float(min(tw_tvt.min(), s_tvt.min() - GRID_MARGIN, base_free.min() - GRID_MARGIN))
        g1 = float(max(tw_tvt.max(), s_tvt.max() + GRID_MARGIN, base_free.max() + GRID_MARGIN))
        gn = int(np.ceil((g1 - g0) / GSTEP)) + 1
        tw_grid = np.interp(g0 + np.arange(gn) * GSTEP, tw_tvt, tw_gr).astype(np.float32)  # tw 全域 valid
        # self = 自prefix(TVT_input, GR)
        self_grid, self_val = _bin_to_grid(kn["TVT_input"].to_numpy(float), kn["GR"].to_numpy(float), g0, gn)
        # nbr = 近傍train(≤NBR_DIST, ≤MAXNBR, 自己除外)の (TVT, GR)
        cx, cy = float(hw["X"].mean()), float(hw["Y"].mean())
        dd = np.hypot(cxy[:, 0] - cx, cxy[:, 1] - cy); order = np.argsort(dd)
        refs = [cwids[j] for j in order if cwids[j] != wid and dd[j] <= NBR_DIST][:MAXNBR]
        if len(refs) >= 2:
            nt, ng = [], []
            for r in refs:
                try:
                    rh = pd.read_csv(TRAIN_DIR / f"{r}__horizontal_well.csv", usecols=["TVT", "GR"])
                    nt.append(rh["TVT"].to_numpy(float)); ng.append(rh["GR"].to_numpy(float))
                except Exception: pass
            if nt:
                nbr_grid, nbr_val = _bin_to_grid(np.concatenate(nt), np.concatenate(ng), g0, gn)
            else:
                nbr_grid, nbr_val = np.zeros(gn, np.float32), np.zeros(gn, np.float32)
        else:
            nbr_grid, nbr_val = np.zeros(gn, np.float32), np.zeros(gn, np.float32)
        ev_pos = hw.index.get_indexer(ev.index)
        hgr = hw["GR"].astype(float).interpolate(limit_direction="both").fillna(float(np.nanmean(tw_gr))).to_numpy(float)[ev_pos].astype(np.float32)
        wells.append(dict(wid=wid, hgr=hgr, hgr_ma=_ma(hgr, MA).astype(np.float32),
                          s_tvt=s_tvt, s_sig=s_sig, base_free=base_free, g0=np.float32(g0), gn=gn,
                          tw_grid=tw_grid, self_grid=self_grid, self_val=self_val,
                          nbr_grid=nbr_grid, nbr_val=nbr_val,
                          target=ev["TVT"].to_numpy(np.float32), cx=cx, cy=cy))
    print(f"[cache] wells {len(wells)}  (nbr有 {sum(1 for w in wells if w['nbr_val'].any())})")
    CACHE.write_bytes(pickle.dumps(wells)); print(f"[saved] {CACHE}")
    return wells


def load_cache():
    return pickle.loads(CACHE.read_bytes()) if CACHE.exists() else build_cache()


# ============================ モデル ============================
import torch
import torch.nn as nn
import torch.nn.functional as F


class GFMR(nn.Module):
    def __init__(self, nch=NCH, hid=HID):
        super().__init__()
        self.emis = nn.Sequential(nn.Linear(nch, hid), nn.SiLU(), nn.Linear(hid, hid), nn.SiLU(), nn.Linear(hid, 1))
        self.log_trans = nn.Parameter(torch.tensor(math.log(TRANS_SIG0)))   # 遷移σ(dip連続, 学習)

    def kernel(self, device):
        s = torch.exp(self.log_trans).clamp(0.3, 12.0)
        K = int((s * 4).clamp(3, 40).item()) | 1
        x = torch.arange(K, device=device, dtype=torch.float32) - K // 2
        k = torch.exp(-0.5 * (x / s) ** 2); return (k / k.sum())[None, None, :]


def _gather(grid, idx):
    """grid:(B,G) idx:(B,T,D)→(B,T,D)。per-well grid から候補TVTのGR。"""
    B, T, D = idx.shape
    return torch.gather(grid, 1, idx.reshape(B, T * D)).reshape(B, T, D)


def build_channels(bt, delta, use_struct=True, center_key="s_tvt"):
    """マッチテンソル (B,T,D,C) を作る。cand = center + δ を共通グリッドへ index。
    use_struct=True → struct-prior chを含む7ch(既定=GF-MR)。False → 6ch(struct無し)。
    center_key: 窓の中心TVT。struct='s_tvt' / struct-free='base_free'。"""
    center = bt[center_key]; g0 = bt["g0"][:, None, None]; gn = bt["gn"]
    cand = center[:, :, None] + delta[None, None, :]               # (B,T,D)
    idx = ((cand - g0) / GSTEP).round().long().clamp_min(0)
    idx = torch.minimum(idx, (gn - 1)[:, None, None])
    tw = _gather(bt["tw_grid"], idx); sf = _gather(bt["self_grid"], idx); sfv = _gather(bt["self_val"], idx)
    nb = _gather(bt["nbr_grid"], idx); nbv = _gather(bt["nbr_val"], idx)
    hgr = bt["hgr"][:, :, None]; hgr_ma = bt["hgr_ma"][:, :, None]
    d_tw = (hgr - tw) / GR_SIG
    d_twma = (hgr_ma - tw) / GR_SIG
    d_self = ((hgr - sf) / GR_SIG) * sfv
    d_nbr = ((hgr - nb) / GR_SIG) * nbv
    chans = [d_tw, d_twma, d_self, sfv, d_nbr, nbv]
    if use_struct:
        ssig = bt["s_sig"][:, :, None].clamp(2.0, 60.0)
        struct = torch.exp(-0.5 * (delta[None, None, :] / ssig) ** 2)   # struct prior bump
        chans.append(struct)
    return torch.stack(chans, -1)                                 # (B,T,D,C) C=7(struct)/6(free)


def bayes_filter(E, ker, delta, s_sig0):
    """微分可能グリッドベイズフィルタ(forward-backward)。E:(B,T,D)=emission分布 → posterior-mean δ, γ。
    s_sig0:(B,1) 初期prior σ(struct=s_sig[:,0] / struct-free=広い定数)。"""
    B, T, D = E.shape; device = E.device
    a = torch.exp(-0.5 * (delta[None, :] / s_sig0) ** 2); a = a / a.sum(1, keepdim=True).clamp_min(1e-30)
    alphas = []
    for i in range(T):
        if i > 0:
            a = F.conv1d(a[:, None, :], ker, padding=ker.shape[-1] // 2)[:, 0, :]
        a = a * E[:, i]; a = a / a.sum(1, keepdim=True).clamp_min(1e-30)
        alphas.append(a)
    alpha = torch.stack(alphas, 1)
    b = torch.ones(B, D, device=device) / D; betas = [None] * T; betas[T - 1] = b
    for i in range(T - 2, -1, -1):
        be = b * E[:, i + 1]
        b = F.conv1d(be[:, None, :], ker, padding=ker.shape[-1] // 2)[:, 0, :]
        b = b / b.sum(1, keepdim=True).clamp_min(1e-30); betas[i] = b
    beta = torch.stack(betas, 1)
    g = alpha * beta; g = g / g.sum(2, keepdim=True).clamp_min(1e-30)  # posterior γ
    pmean = (g * delta[None, None, :]).sum(2)                      # posterior-mean δ (B,T)
    return pmean, g


def run_batch(model, bt, delta, device, use_struct=True, center_key="s_tvt"):
    ch = build_channels(bt, delta, use_struct=use_struct, center_key=center_key)   # (B,T,D,C)
    logit = model.emis(ch).squeeze(-1)                             # (B,T,D)
    E = torch.exp(torch.log_softmax(logit, dim=2))                 # emission(δ分布)
    ker = model.kernel(device)
    if use_struct:
        s_sig0 = bt["s_sig"][:, 0:1].clamp(2, 60)
    else:
        s_sig0 = torch.full((ch.shape[0], 1), float(delta[-1]) / 2, device=device)   # struct無し=窓幅に応じた広い初期prior
    pmean, g = bayes_filter(E, ker, delta, s_sig0)
    tvt = bt[center_key] + pmean
    return tvt, g


def collate(batch, device):
    B = len(batch); T = max(len(w["hgr"]) for w in batch); G = max(w["gn"] for w in batch)
    def padT(key):
        a = np.zeros((B, T), np.float32)
        for i, w in enumerate(batch): a[i, :len(w[key])] = w[key]
        return a
    def padG(key):
        a = np.zeros((B, G), np.float32)
        for i, w in enumerate(batch): a[i, :len(w[key])] = w[key]
        return a
    mask = np.zeros((B, T), np.float32)
    for i, w in enumerate(batch): mask[i, :len(w["hgr"])] = 1.0
    t = lambda x: torch.tensor(x, device=device)
    out = dict(hgr=t(padT("hgr")), hgr_ma=t(padT("hgr_ma")), s_tvt=t(padT("s_tvt")), s_sig=t(padT("s_sig")),
               tgt=t(padT("target")), mask=t(mask),
               tw_grid=t(padG("tw_grid")), self_grid=t(padG("self_grid")), self_val=t(padG("self_val")),
               nbr_grid=t(padG("nbr_grid")), nbr_val=t(padG("nbr_val")),
               g0=t(np.array([w["g0"] for w in batch], np.float32)),
               gn=t(np.array([w["gn"] for w in batch], np.int64)))
    if "base_free" in batch[0]:                                   # struct-free 窓中心(あれば)
        out["base_free"] = t(padT("base_free"))
    return out


def make_fields(wells):
    from sklearn.cluster import KMeans
    xy = np.array([[w["cx"], w["cy"]] for w in wells]); k = min(N_FIELDS, len(wells))
    return KMeans(n_clusters=k, random_state=SEED, n_init=10).fit(xy).labels_


def _len_buckets(wells, bs):
    """長さ順に並べて連続バッチ化(padding無駄を最小化)。返す=各バッチのindexリスト。"""
    idx = sorted(range(len(wells)), key=lambda i: len(wells[i]["hgr"]))
    return [idx[i:i + bs] for i in range(0, len(idx), bs)]


def train_fold(wtr, wva, delta, device):
    model = GFMR().to(device); opt = torch.optim.Adam(model.parameters(), lr=LR)
    batches = _len_buckets(wtr, BATCH)                     # 長さバケット(padding削減)。バッチ順のみ毎epochシャッフル
    rng = np.random.RandomState(SEED); border = np.arange(len(batches))
    Ttot = sum(len(w["hgr"]) for w in wtr)                 # 進捗ETA用(処理行総数)
    for ep in range(EPOCHS):
        model.train(); rng.shuffle(border); tot = 0.0; nb = 0; done_rows = 0; t0 = time.time(); last = t0
        for bi in border:
            batch = [wtr[j] for j in batches[bi]]
            bt = collate(batch, device)
            tvt, _ = run_batch(model, bt, delta, device, use_struct=USE_STRUCT, center_key=CENTER_KEY)
            loss = (((tvt - bt["tgt"]) ** 2) * bt["mask"]).sum() / bt["mask"].sum().clamp_min(1)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
            tot += float(loss); nb += 1; done_rows += sum(len(w["hgr"]) for w in batch)
            if time.time() - last > 20:                   # ★エポック内進捗(20秒ごと): 行進捗+ETA
                el = time.time() - t0; eta = el / max(done_rows, 1) * (Ttot - done_rows)
                print(f"      ep{ep+1} {nb}/{len(batches)}b rows{done_rows//1000}k/{Ttot//1000}k "
                      f"rmse~{math.sqrt(tot/max(nb,1)):.2f} {el:.0f}s eta{eta:.0f}s", flush=True); last = time.time()
        print(f"    ep{ep+1}/{EPOCHS} mse={tot/max(nb,1):.3f} (rmse~{math.sqrt(tot/max(nb,1)):.3f}) {time.time()-t0:.0f}s", flush=True)
    model.eval(); oof = {}
    with torch.no_grad():
        for bidx in _len_buckets(wva, BATCH):
            batch = [wva[j] for j in bidx]; bt = collate(batch, device)
            tvt, _ = run_batch(model, bt, delta, device, use_struct=USE_STRUCT, center_key=CENTER_KEY)
            for b, w in enumerate(batch): oof[w["wid"]] = tvt[b, :len(w["hgr"])].cpu().numpy()
    return oof


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "train"
    if mode == "cache":
        build_cache(); return
    torch.manual_seed(SEED); np.random.seed(SEED)
    device = "cuda" if (os.environ.get("DEVICE", "auto") != "cpu" and torch.cuda.is_available()) else "cpu"
    wells = load_cache()
    delta = torch.arange(-DR_EFF, DR_EFF + 1e-6, DSTEP_EFF, device=device, dtype=torch.float32)
    print(f"[v68] wells {len(wells)} CV={CV_MODE} STRUCT={int(USE_STRUCT)}(center={CENTER_KEY}) device={device} δ=±{DR_EFF:.0f}/{DSTEP_EFF:.0f}={len(delta)}bin NCH={NCH} HID={HID}")
    if CENTER_KEY not in wells[0]:
        print(f"★cacheに'{CENTER_KEY}'が無い(古いcache)。EVAL_STRIDE={EVAL_STRIDE}でcacheを作り直してください: python gridfilter_v68.py cache"); return
    if CV_MODE == "field":
        from sklearn.model_selection import GroupKFold
        fld = make_fields(wells); nf = min(5, len(set(fld)))     # 16フィールド→5-fold(各foldで~3フィールドをholdout)
        assign = np.zeros(len(wells), int)
        for k, (_, va) in enumerate(GroupKFold(nf).split(wells, groups=fld)): assign[va] = k
        folds = list(range(nf))
        print(f"[cv] leave-field-out: {len(set(fld))}fields → {nf}-fold GroupKFold on fields(holdout fieldはtrainに不在)")
    else:
        from sklearn.model_selection import GroupKFold
        g = np.array([w["wid"] for w in wells]); assign = np.zeros(len(wells), int)
        for k, (_, va) in enumerate(GroupKFold(5).split(wells, groups=g)): assign[va] = k
        folds = list(range(5)); print("[cv] well GroupKFold(5)")
    oof = {}; t0 = time.time()
    for k in folds:
        tr = [wells[i] for i in np.where(assign != k)[0]]; va = [wells[i] for i in np.where(assign == k)[0]]
        print(f"  fold {k}: train {len(tr)} / val {len(va)}", flush=True)
        oof.update(train_fold(tr, va, delta, device))
    # 評価: overall RMSE + 中心baseline(struct or struct-free) + oracle(δ格子で真に最も近い=到達不能上限)
    perw, pool, spool, opool = {}, [], [], []
    for w in wells:
        if w["wid"] not in oof: continue
        p = oof[w["wid"]]; tgt = w["target"]; ctr = w[CENTER_KEY]; n = min(len(p), len(tgt))
        e = p[:n] - tgt[:n]; se = ctr[:n] - tgt[:n]
        oe = (np.clip(np.round((tgt[:n] - ctr[:n]) / DSTEP_EFF), -DR_EFF / DSTEP_EFF, DR_EFF / DSTEP_EFF) * DSTEP_EFF + ctr[:n]) - tgt[:n]
        fin = np.isfinite(e)
        if fin.any():
            perw[w["wid"]] = float(np.sqrt(np.mean(e[fin] ** 2))); pool.append(e[fin]); spool.append(se[fin]); opool.append(oe[fin])
    pool = np.concatenate(pool); overall = float(np.sqrt(np.mean(pool ** 2)))
    struct_rmse = float(np.sqrt(np.mean(np.concatenate(spool) ** 2)))
    oracle = float(np.sqrt(np.mean(np.concatenate(opool) ** 2)))
    tag = f"{CV_MODE}_{'st' if USE_STRUCT else 'free'}"
    base_lbl = "struct基準" if USE_STRUCT else "anchor基準(struct-free)"
    print(f"\n[v68 GF-MR] {tag}  overall RMSE = {overall:.4f}   ({base_lbl} {struct_rmse:.4f} / δ格子oracle {oracle:.4f})")
    print(f"           坑井別中央 {np.median(list(perw.values())):.4f}  ({len(perw)}坑井, {time.time()-t0:.0f}s)")
    ids, arr = [], []
    for w in wells:
        if w["wid"] in oof:
            for j, v in enumerate(oof[w["wid"]]): ids.append(f"{w['wid']}_{j}"); arr.append(v)
    np.save(OUT / f"gf_oof_{tag}.npy", np.array(arr, np.float32))
    (OUT / f"results_{tag}.json").write_text(json.dumps(dict(
        overall=overall, struct=struct_rmse, oracle_dgrid=oracle, per_well_median=float(np.median(list(perw.values()))),
        n_wells=len(perw), cv=CV_MODE, cfg=dict(dr=DR, dstep=DSTEP, gstep=GSTEP, gr_sig=GR_SIG, hid=HID,
        epochs=EPOCHS, n_fields=N_FIELDS, ma=MA, eval_stride=EVAL_STRIDE)), indent=2))
    print(f"[saved] gf_oof_{tag}.npy / results_{tag}.json")


if __name__ == "__main__":
    main()
