# -*- coding: utf-8 -*-
"""
v50 Stage A — GR対応点の対比学習(表現がゴーストを分離できるかの分水嶺測定)
==============================================================================
PF不使用。水平坑の行特徴 a_t と typewell 格子特徴 b_k を対比学習(InfoNCE+ゴーストhinge)し、
  (1) 行レベル: 真値近傍 vs 最強遠方競合(ゴースト)の判別精度・top1 — 生GRコストと同一土俵で比較
  (2) A2: 学習類似度を v48 grid filter の emission に差し替えて FBデコード → val坑井プールRMSE
      比較線 = naive(last_known複製 ≈ rmse15) / struct単体 / 生emission grid filter(v48)
を測る。学習は train坑井のみ・評価は val坑井のみ(井戸レベル分割)。

実行: python v50/stageA_v50.py            (cache 無ければ自動build → train → eval → plot)
env : EPOCHS=25 DIM=64 NW=10000 STEP=0.5 BAND=45 ...
"""
import os, io, json, time, glob, math, pickle, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F

warnings.filterwarnings("ignore"); np.seterr(all="ignore")
P0    = Path(r"C:\Users\kosaka256\Documents\rogii_claude")
TRAIN = P0 / "rogii-wellbore-geology-prediction" / "train"
ART   = P0 / "v50" / "artifacts_v50"; ART.mkdir(parents=True, exist_ok=True)
DEV   = "cuda" if torch.cuda.is_available() else "cpu"

STEP   = float(os.environ.get("STEP", "0.5"))     # typewell格子刻み(ft)
BAND   = float(os.environ.get("BAND", "45"))      # 候補バンド半幅(ft)
NB     = int(round(BAND / STEP))                  # バンド半幅(ビン)
DIM    = int(os.environ.get("DIM", "64"))
CH_ENC = int(os.environ.get("CH", "96"))          # エンコーダ幅
DILS   = tuple(int(x) for x in os.environ.get("DILS", "1,2,4,8,16").split(","))
EPOCHS = int(os.environ.get("EPOCHS", "25"))
LR     = float(os.environ.get("LR", "3e-4"))
COS_LR = os.environ.get("COS", "1") == "1"        # cosine LR減衰
AUG    = os.environ.get("AUG", "0") == "1"        # 低周波ゲイン/ドリフト包絡+ノイズ増強
NOISE_F = float(os.environ.get("NOISE_F", "0.05"))
TAG    = os.environ.get("TAG", "")                # 実験タグ(encファイル名/ablation log)
SYNTH_EP = int(os.environ.get("SYNTH_EP", "0"))   # 合成井戸事前学習エポック数(0=off)
RPW    = int(os.environ.get("RPW", "384"))        # 1坑井から1epochに使う行数
NW     = int(os.environ.get("NW", "10000"))
SEED   = int(os.environ.get("SEED", "0"))
NEAR_B = int(os.environ.get("NEAR", "4"))         # 「真値近傍」= ±4bin(2ft)
FAR_B  = int(os.environ.get("FAR", "12"))         # 「遠方競合」= ±12bin(6ft)以遠
# --- A2 デコード(v48 best 構成。TAU/LPR は hybrid 用に再調整可) ---
STRIDE = int(os.environ.get("STRIDE", "16")); GRS = 25; SIGG = 6.0
TAU = float(os.environ.get("TAU", "0.35")); LPR = float(os.environ.get("LPR", "0.3"))
TEMP_A2 = float(os.environ.get("TEMP_A2", "1.0"))  # 学習emissionのsoftmax温度(logit×scaleに追加)

CACHE = ART / "cacheA_v50.pkl"
STRUCT = pickle.loads((P0 / "v38" / "artifacts_v38" / "struct_prior_train.pkl").read_bytes())

# ============================ 特徴量(表現の素材) ============================
def _rz(v):
    """ロバストz(median/IQR)。キャリブレーション差に強い正規化。"""
    med = np.nanmedian(v); iqr = np.nanpercentile(v, 75) - np.nanpercentile(v, 25)
    return np.clip((v - med) / max(iqr / 1.349, 1e-6), -8, 8)

def _rm(v, w):
    return pd.Series(v).rolling(w, center=True, min_periods=1).mean().to_numpy()

def _rs(v, w):
    return pd.Series(v).rolling(w, center=True, min_periods=2).std().fillna(0).to_numpy()

def _detr(v, w=301):
    m = _rm(v, w); s = _rs(v, w)
    return np.clip((v - m) / np.maximum(s, 1e-3), -6, 6)

def feats_h(gr, md, z):
    """水平坑 行特徴 (L,8)。全て test 実在列(GR/MD/Z)のみ。"""
    g = pd.Series(gr).interpolate(limit_direction="both").to_numpy()
    g = np.where(np.isfinite(g), g, np.nanmedian(g)); g = np.where(np.isfinite(g), g, 0.0)
    z0 = _rz(g)
    s5, s21, s81 = _rm(z0, 5), _rm(z0, 21), _rm(z0, 81)
    d5 = np.gradient(s5) * 10.0
    v21 = _rs(z0, 21)
    dt = _detr(z0, 301)
    dmd = np.maximum(np.gradient(md), 1e-3)
    dzdm = np.clip(np.gradient(z) / dmd, -0.2, 0.2) * 5.0
    X = np.stack([z0, s5, s21, s81, d5, v21, dt, dzdm], 1).astype(np.float32)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

def feats_t(gg):
    """typewell 格子特徴 (K,7)。gg = GR on 0.5ft TVT grid。"""
    z0 = _rz(gg)
    s3, s9, s33 = _rm(z0, 3), _rm(z0, 9), _rm(z0, 33)
    d3 = np.gradient(s3) * 10.0
    v9 = _rs(z0, 9)
    dt = _detr(z0, 301)
    X = np.stack([z0, s3, s9, s33, d3, v9, dt], 1).astype(np.float32)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

# ============================ キャッシュ構築 ============================
def build_cache():
    wells = sorted(STRUCT.keys())[:NW]
    out = {}; t0 = time.time()
    for wi, wn in enumerate(wells):
        try:
            hw = pd.read_csv(TRAIN / f"{wn}__horizontal_well.csv")
            tw = pd.read_csv(TRAIN / f"{wn}__typewell.csv").sort_values("TVT")
        except Exception:
            continue
        if "TVT" not in hw.columns or len(tw) < 8: continue
        ev = hw["TVT_input"].isna().to_numpy()
        if ev.sum() == 0 or (~ev).sum() < 10: continue
        sp = STRUCT.get(wn)
        if sp is None or len(sp["tvt"]) != int(ev.sum()): continue
        y_all = hw["TVT_input"].to_numpy(float)
        y_all = np.where(np.isfinite(y_all), y_all, hw["TVT"].to_numpy(float))
        tw_tvt = tw["TVT"].to_numpy(float); tw_gr = tw["GR"].to_numpy(float)
        m = np.isfinite(tw_tvt) & np.isfinite(tw_gr)
        tw_tvt, tw_gr = tw_tvt[m], tw_gr[m]
        if len(tw_tvt) < 8: continue
        gmin, gmax = float(tw_tvt.min()), float(tw_tvt.max())
        gg = np.interp(np.arange(gmin, gmax + STEP, STEP), tw_tvt, tw_gr)
        Fh = feats_h(hw["GR"].to_numpy(float), hw["MD"].to_numpy(float), hw["Z"].to_numpy(float))
        xy = np.array([np.nanmean(hw["X"].to_numpy(float)), np.nanmean(hw["Y"].to_numpy(float))])
        Ft = feats_t(gg)
        # 生emission(AUC/A2基準)用: eval行順の g_s25(v48と同一の作り)
        ga = pd.Series(hw["GR"].to_numpy(float)).interpolate(limit_direction="both").to_numpy()
        fb = float(np.nanmean(tw_gr)) if np.isfinite(np.nanmean(tw_gr)) else 0.0
        ga = np.where(np.isfinite(ga), ga, fb)
        g_ev = ga[ev]
        g_s25 = pd.Series(g_ev).rolling(GRS, center=True, min_periods=1).mean().to_numpy()
        out[wn] = dict(Fh=Fh, Ft=Ft, gmin=gmin, gg=gg.astype(np.float32),
                       y=y_all.astype(np.float32), ev=ev,
                       st=np.asarray(sp["tvt"], np.float32), sg=np.asarray(sp["sig"], np.float32),
                       g25=g_s25.astype(np.float32), graw=ga.astype(np.float32),
                       xy=xy.astype(np.float64))
        if (wi + 1) % 150 == 0: print(f"  cache {wi+1}/{len(wells)} ({time.time()-t0:.0f}s)", flush=True)
    CACHE.write_bytes(pickle.dumps(out, protocol=4))
    print(f"cache saved: {len(out)} wells ({time.time()-t0:.0f}s)")
    return out

# ============================ モデル ============================
class Enc1D(nn.Module):
    def __init__(self, cin, ch, dout, dils=(1, 2, 4, 8, 16)):
        super().__init__()
        self.inp = nn.Conv1d(cin, ch, 5, padding=2)
        self.blocks = nn.ModuleList()
        for d in dils:
            self.blocks.append(nn.Sequential(
                nn.Conv1d(ch, ch, 5, padding=2 * d, dilation=d), nn.GELU(),
                nn.Conv1d(ch, ch, 1)))
        self.norm = nn.ModuleList([nn.GroupNorm(8, ch) for _ in dils])
        self.out = nn.Conv1d(ch, dout, 1)
    def forward(self, x):                      # x:(L,C) → (L,D) L2正規化
        h = self.inp(x.t()[None])
        for blk, nm in zip(self.blocks, self.norm):
            h = nm(h + blk(h))
        return F.normalize(self.out(h)[0].t(), dim=1)

# ============================ 学習 ============================
def _slow_env(L, amp):
    """低周波包絡(4〜8ノット線形補間)。キャリブレーションの緩いドリフトを模す。"""
    k = int(4 + torch.randint(0, 5, (1,)).item())
    v = 1.0 + amp * (torch.rand(k) * 2 - 1)
    return F.interpolate(v[None, None].to(DEV), size=L, mode="linear", align_corners=True)[0, 0]

def augment_h(xh):
    """水平坑特徴の増強: GRチャネル(0..6)に低周波ゲイン× + 振幅チャネル(0..3)にドリフト+ + 白ノイズ。"""
    L = xh.shape[0]
    xh = xh.clone()
    xh[:, :7] = xh[:, :7] * _slow_env(L, 0.15)[:, None]
    xh[:, :4] = xh[:, :4] + (_slow_env(L, 1.0)[:, None] - 1.0) * 0.4
    return xh + NOISE_F * torch.randn_like(xh)

def synth_well(c, rng, T=2400):
    """typewellから対応既知の合成水平坑を生成: パス=OUドリフト+緩い正弦, GR=f(y)+ノイズ。
    返り値: (Fh合成, kt_float配列)  ※真の対応が構成的に既知=事前学習の無限教師。"""
    gg = c["gg"].astype(np.float64); K = len(gg); gmin = c["gmin"]
    lo, hi = gmin + (NB + 4) * STEP, gmin + (K - 1 - NB - 4) * STEP
    if hi <= lo: return None
    y0 = rng.uniform(lo, hi)
    dip = np.zeros(T); d = rng.normal(0, 0.01)
    th = rng.uniform(0.005, 0.03); sd = rng.uniform(0.001, 0.008)
    for t in range(1, T):
        d = (1 - th) * d + rng.normal(0, sd); dip[t] = np.clip(d, -0.08, 0.08)
    y = y0 + np.cumsum(dip)
    y += rng.uniform(1, 6) * np.sin(np.linspace(0, rng.uniform(1, 4) * np.pi, T) + rng.uniform(0, 6))
    y = np.clip(y, lo, hi)
    g = np.interp(y, gmin + np.arange(K) * STEP, gg)
    g = g * (1 + 0.1 * np.sin(np.linspace(0, rng.uniform(1, 3) * np.pi, T)))          # 低周波ゲイン
    g = g + rng.normal(0, 0.06 * max(np.std(gg), 1e-3), T)                             # 白ノイズ
    z = -np.cumsum(dip) * rng.uniform(0.6, 1.0)                                        # TVT≈-Z+構造 の緩い模倣
    Fh = feats_h(g, np.arange(T, dtype=float), z)
    return Fh, (y - gmin) / STEP

def train(data, val=None, tag=""):
    wells = sorted(data.keys())
    if val is None: val = wells[::5]
    val = set(val); trn = [w for w in wells if w not in val]
    print(f"[{TAG}{tag or '/fold0'}] train {len(trn)} wells / val {len(val)} wells"
          f" ch{CH_ENC} dim{DIM} dils{DILS} ep{EPOCHS} aug={AUG}", flush=True)
    torch.manual_seed(SEED); rng = np.random.default_rng(SEED)
    encA = Enc1D(8, CH_ENC, DIM, DILS).to(DEV); encB = Enc1D(7, CH_ENC, DIM, DILS).to(DEV)
    logit_s = nn.Parameter(torch.tensor(2.3, device=DEV))          # exp(2.3)≈10
    opt = torch.optim.AdamW(list(encA.parameters()) + list(encB.parameters()) + [logit_s],
                            lr=LR, weight_decay=1e-4)
    offs = torch.arange(-NB, NB + 1, device=DEV)
    # soft target: 真値ビン中心 σ=2bin(1ft)
    tgt0 = torch.exp(-0.5 * (offs.float() / 2.0) ** 2); tgt0 = (tgt0 / tgt0.sum())
    far_m = (offs.abs() >= FAR_B)
    t0 = time.time(); step = 0
    def _step(xh, Ft_np, kf, K):
        """1坑井ぶんの対比損失+更新(実/合成 共通)。"""
        lab = np.where(np.isfinite(kf) & (kf >= NB) & (kf <= K - 1 - NB))[0]
        if len(lab) < 32: return None
        rows = rng.choice(lab, size=min(RPW, len(lab)), replace=False)
        kt = torch.tensor(np.round(kf[rows]).astype(np.int64), device=DEV)
        a = encA(xh)[rows]
        b = encB(torch.from_numpy(Ft_np).to(DEV))
        cand = kt[:, None] + offs[None, :]
        sims = (a[:, None, :] * b[cand]).sum(-1)
        logits = sims * logit_s.exp().clamp(1.0, 100.0)
        loss = -(tgt0[None, :] * F.log_softmax(logits, dim=1)).sum(1).mean()
        near = sims[:, NB - NEAR_B: NB + NEAR_B + 1].max(1).values
        far = sims.masked_fill(~far_m[None, :], -1e4).max(1).values
        loss = loss + 0.5 * F.relu(far - near + 0.10).mean()
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(list(encA.parameters()) + list(encB.parameters()), 1.0)
        opt.step(); return float(loss)
    if SYNTH_EP > 0:                       # ★R3: 合成井戸で事前学習(対応が構成的に既知)
        for ep in range(SYNTH_EP):
            encA.train(); encB.train(); losses = []
            for wi in rng.permutation(len(trn)):
                c = data[trn[wi]]
                sw = synth_well(c, rng)
                if sw is None: continue
                Fh_s, kf = sw
                ls_v = _step(torch.from_numpy(Fh_s).to(DEV), c["Ft"], kf, len(c["gg"]))
                if ls_v is not None: losses.append(ls_v)
            print(f"[synth] ep{ep+1}/{SYNTH_EP} loss={np.mean(losses):.4f} ({time.time()-t0:.0f}s)", flush=True)
    for ep in range(EPOCHS):
        if COS_LR:
            for pg in opt.param_groups:
                pg["lr"] = LR * 0.5 * (1.0 + math.cos(math.pi * ep / max(EPOCHS - 1, 1)))
        encA.train(); encB.train(); order = rng.permutation(len(trn)); losses = []
        for wi in order:
            c = data[trn[wi]]
            y = c["y"]; gmin = c["gmin"]; K = len(c["gg"])
            kf = (y - gmin) / STEP
            lab = np.where(np.isfinite(kf) & (kf >= NB) & (kf <= K - 1 - NB))[0]
            if len(lab) < 32: continue
            rows = rng.choice(lab, size=min(RPW, len(lab)), replace=False)
            kt = torch.tensor(np.round(kf[rows]).astype(np.int64), device=DEV)
            xh = torch.from_numpy(c["Fh"]).to(DEV)
            if AUG: xh = augment_h(xh)
            a = encA(xh)[rows]                                             # (R,D)
            b = encB(torch.from_numpy(c["Ft"]).to(DEV))                    # (K,D)
            cand = kt[:, None] + offs[None, :]                             # (R,181)
            sims = (a[:, None, :] * b[cand]).sum(-1)                       # cosine
            logits = sims * logit_s.exp().clamp(1.0, 100.0)
            loss = -(tgt0[None, :] * F.log_softmax(logits, dim=1)).sum(1).mean()
            near = sims[:, NB - NEAR_B: NB + NEAR_B + 1].max(1).values
            far = sims.masked_fill(~far_m[None, :], -1e4).max(1).values
            loss = loss + 0.5 * F.relu(far - near + 0.10).mean()           # ゴーストhinge(cosine margin)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(list(encA.parameters()) + list(encB.parameters()), 1.0)
            opt.step(); losses.append(float(loss)); step += 1
        print(f"ep{ep+1}/{EPOCHS} loss={np.mean(losses):.4f} scale={float(logit_s.exp()):.1f}"
              f" ({time.time()-t0:.0f}s)", flush=True)
    torch.save(dict(encA=encA.state_dict(), encB=encB.state_dict(), logit_s=float(logit_s)),
               ART / f"stageA_enc{TAG}{tag}.pt")
    return encA, encB, sorted(val), float(logit_s.exp())

# ============================ 評価 ============================
TEMPS = [float(x) for x in os.environ.get("TEMPS", "0.3,0.5,1.0,2.0").split(",")]

@torch.no_grad()
def evaluate(data, encA, encB, val, logit_scale=8.0):
    encA.eval(); encB.eval()
    offs = np.arange(-NB, NB + 1)
    accs = dict(raw=[], lrn=[], top1_raw=[], top1_lrn=[])
    rmse = dict(naive=[], struct=[], gf_raw=[])
    HWS = [float(x) for x in os.environ.get("HWS", "0.5").split(",")]   # ハイブリッド: L_raw × P_lrn^HW
    for tp in TEMPS:
        rmse[f"gf_lrn_t{tp:g}"] = []
        for hw in HWS: rmse[f"hyb_t{tp:g}_w{hw:g}"] = []
    grid_t = torch.tensor(offs * STEP, dtype=torch.float32, device=DEV)
    ND = len(offs)
    dd = grid_t[:, None] - grid_t[None, :]
    M = torch.exp(-0.5 * (dd / TAU) ** 2); M = M / M.sum(0, keepdim=True).clamp_min(1e-30)
    for wn in val:
        c = data[wn]
        ev_idx = np.where(c["ev"])[0]
        y_ev = c["y"][ev_idx]
        a = encA(torch.from_numpy(c["Fh"]).to(DEV))[ev_idx]     # (T,D)
        b = encB(torch.from_numpy(c["Ft"]).to(DEV))             # (K,D)
        K = len(c["gg"]); gmin = c["gmin"]
        st, sg = c["st"].astype(np.float64), np.clip(c["sg"].astype(np.float64), 1, 60)
        ks = np.round((st - gmin) / STEP).astype(np.int64)
        cand = np.clip(ks[:, None] + offs[None, :], 0, K - 1)
        okc = (ks[:, None] + offs[None, :] >= 0) & (ks[:, None] + offs[None, :] <= K - 1)
        candt = torch.tensor(cand, device=DEV)
        sims = (a[:, None, :] * b[candt]).sum(-1).cpu().numpy()          # (T,181) cosine
        sims[~okc] = -1e4
        cost = np.abs(c["g25"][:, None] - c["gg"][cand])                 # 生emission(v48)
        cost[~okc] = 1e4
        # --- 行レベル判別(真値がバンド内の行のみ) ---
        kt = (y_ev - gmin) / STEP
        toff = np.round(kt - ks).astype(np.int64) + NB
        inb = (toff >= NEAR_B) & (toff <= 2 * NB - NEAR_B) & np.isfinite(y_ev)
        for t in np.where(inb)[0]:
            o = toff[t]
            nearm = np.zeros(ND, bool); nearm[max(0, o - NEAR_B): o + NEAR_B + 1] = True
            farm = np.abs(np.arange(ND) - o) >= FAR_B
            if farm.sum() == 0: continue
            accs["lrn"].append(float(sims[t, nearm].max() > sims[t, farm].max()))
            accs["raw"].append(float(cost[t, nearm].min() < cost[t, farm].min()))
            accs["top1_lrn"].append(float(abs(int(np.argmax(sims[t])) - o) <= 6))
            accs["top1_raw"].append(float(abs(int(np.argmin(cost[t])) - o) <= 6))
        # --- A2: grid filter デコード(bin=16行, FB) ---
        T = len(ev_idx); nb_ = (T + STRIDE - 1) // STRIDE; bid = np.arange(T) // STRIDE
        def bm(x):
            o = np.zeros((nb_,) + x.shape[1:]); cnt = np.zeros(nb_)
            np.add.at(o, bid, x); np.add.at(cnt, bid, 1)
            return o / np.maximum(cnt, 1)[(...,) + (None,) * (x.ndim - 1)]
        sgb = np.clip(bm(sg), 1, 60); stb = bm(st)
        Pr = np.exp(-0.5 * ((offs * STEP)[None, :] / sgb[:, None]) ** 2)
        def fb(obs_np):
            obs = torch.tensor(obs_np, dtype=torch.float32, device=DEV)
            obs = obs / obs.sum(1, keepdim=True).clamp_min(1e-30)
            aF = torch.zeros(nb_, ND, device=DEV); aF[0] = obs[0]
            for i in range(1, nb_):
                pred = M @ aF[i - 1]; aF[i] = (pred.clamp_min(1e-30) * obs[i])
                aF[i] = aF[i] / aF[i].sum().clamp_min(1e-30)
            bB = torch.zeros(nb_, ND, device=DEV); bB[-1] = 1.0 / ND
            for i in range(nb_ - 2, -1, -1):
                msg = bB[i + 1] * obs[i + 1]; bB[i] = (M @ msg).clamp_min(1e-30)
                bB[i] = bB[i] / bB[i].sum().clamp_min(1e-30)
            gam = aF * bB; gam = gam / gam.sum(1, keepdim=True).clamp_min(1e-30)
            dh = (gam * grid_t[None, :]).sum(1).cpu().numpy()
            return stb + dh
        L_raw = np.exp(-0.5 * (bm(cost) / SIGG) ** 2) * Pr ** LPR
        z_raw = fb(L_raw)[bid]
        fin = np.isfinite(y_ev)
        lk = c["y"][~c["ev"]][-1] if (~c["ev"]).any() else y_ev[0]
        rmse["naive"].append(np.full(fin.sum(), lk) - y_ev[fin])
        rmse["struct"].append(st[fin] - y_ev[fin])
        rmse["gf_raw"].append(z_raw[fin] - y_ev[fin])
        # 学習emission: 行softmax(学習済スケール×温度)→ビン内で確率平均(混合)→prior^LPR
        sims_v = np.where(okc, sims, -1e4)
        L_raw0 = np.exp(-0.5 * (bm(cost) / SIGG) ** 2)                 # prior抜きの生尤度
        for tp in TEMPS:
            Prow = torch.softmax(torch.tensor(sims_v, dtype=torch.float32) * logit_scale * tp, dim=1).numpy()
            Pb = bm(Prow)
            z_lrn = fb(Pb * Pr ** LPR)[bid]
            rmse[f"gf_lrn_t{tp:g}"].append(z_lrn[fin] - y_ev[fin])
            for hw in HWS:
                z_hyb = fb(L_raw0 * np.maximum(Pb, 1e-12) ** hw * Pr ** LPR)[bid]   # 積の専門家
                rmse[f"hyb_t{tp:g}_w{hw:g}"].append(z_hyb[fin] - y_ev[fin])
    res = {}
    for k in accs: res[k] = float(np.mean(accs[k]))
    res["n_rows"] = len(accs["raw"])
    for k in rmse:
        e = np.concatenate(rmse[k]); e = e[np.isfinite(e)]
        res[f"rmse_{k}"] = float(np.sqrt(np.mean(e ** 2)))
    return res, rmse, accs

# ============================ main ============================
if __name__ == "__main__":
    if CACHE.exists():
        data = pickle.loads(CACHE.read_bytes())
        print(f"cache load: {len(data)} wells")
    else:
        data = build_cache()
    if os.environ.get("FOLD5", "0") == "1":
        # ===== 5-fold: 全773坑井の正直な full-set 評価 =====
        wells = sorted(data.keys()); agg_e = {}; agg_a = {}
        for f in range(5):
            valf = wells[f::5]; fp = ART / f"stageA_enc{TAG}_f{f}.pt"
            if fp.exists():
                ck = torch.load(fp, map_location=DEV)
                encA = Enc1D(8, CH_ENC, DIM, DILS).to(DEV); encA.load_state_dict(ck["encA"])
                encB = Enc1D(7, CH_ENC, DIM, DILS).to(DEV); encB.load_state_dict(ck["encB"])
                ls_ = math.exp(ck["logit_s"]); print(f"[f{f}] enc load")
            else:
                encA, encB, _, ls_ = train(data, val=valf, tag=f"_f{f}")
            res_f, rm_f, ac_f = evaluate(data, encA, encB, valf, logit_scale=ls_)
            for k in rm_f: agg_e.setdefault(k, []).extend(rm_f[k])
            for k in ac_f: agg_a.setdefault(k, []).extend(ac_f[k])
            print(f"[fold{f}] raw={res_f['raw']:.3f} lrn={res_f['lrn']:.3f} "
                  + "  ".join(f"{k[5:]}={v:.3f}" for k, v in sorted(res_f.items()) if k.startswith("rmse_")),
                  flush=True)
        res = {k: float(np.mean(v)) for k, v in agg_a.items()}
        res["n_rows"] = len(agg_a["raw"])
        for k in agg_e:
            e = np.concatenate(agg_e[k]); e = e[np.isfinite(e)]
            res[f"rmse_{k}"] = float(np.sqrt(np.mean(e ** 2)))
        print("\n=== v50 Stage A 5-fold 全773坑井(正直) ===")
        print(f"  行レベル判別: raw={res['raw']:.3f}  learned={res['lrn']:.3f} (n={res['n_rows']})")
        print(f"  top1(±3ft) : raw={res['top1_raw']:.3f}  learned={res['top1_lrn']:.3f}")
        for k in sorted(res):
            if k.startswith("rmse_"): print(f"  {k[5:]:16s} = {res[k]:.4f}")
        (ART / f"stageA_fold5_results{TAG}.json").write_text(json.dumps(res, indent=1))
        raise SystemExit(0)
    if os.environ.get("EVAL_ONLY", "0") == "1" and (ART / f"stageA_enc{TAG}.pt").exists():
        ck = torch.load(ART / f"stageA_enc{TAG}.pt", map_location=DEV)
        encA = Enc1D(8, CH_ENC, DIM, DILS).to(DEV); encA.load_state_dict(ck["encA"])
        encB = Enc1D(7, CH_ENC, DIM, DILS).to(DEV); encB.load_state_dict(ck["encB"])
        ls_ = math.exp(ck["logit_s"])
        wells = sorted(data.keys()); val = wells[::5]
        print(f"EVAL_ONLY: enc load (scale={ls_:.1f}) val {len(val)} wells")
    else:
        encA, encB, val, ls_ = train(data)
    res, _, _ = evaluate(data, encA, encB, val, logit_scale=ls_)
    print("\n=== v50 Stage A 結果(val坑井のみ) ===")
    print(f"  行レベル判別(真値近傍 vs 遠方競合): raw={res['raw']:.3f}  learned={res['lrn']:.3f}"
          f"   (n={res['n_rows']})")
    print(f"  top1(±3ft):                         raw={res['top1_raw']:.3f}  learned={res['top1_lrn']:.3f}")
    line = "  ".join(f"gf_lrn(t{tp:g})={res[f'rmse_gf_lrn_t{tp:g}']:.3f}" for tp in TEMPS)
    print(f"  A2 プールRMSE: naive={res['rmse_naive']:.3f}  struct={res['rmse_struct']:.3f}"
          f"  gf_raw={res['rmse_gf_raw']:.3f}\n                 {line}")
    for k in sorted(res):
        if k.startswith("rmse_hyb"): print(f"                 {k[5:]}={res[k]:.3f}")
    (ART / "stageA_results.json").write_text(json.dumps(res, indent=1))
    ab_fp = ART / "stageA_ablation.json"
    ab = json.loads(ab_fp.read_text()) if ab_fp.exists() else {}
    ab[TAG or "base"] = dict(cfg=dict(ch=CH_ENC, dim=DIM, dils=list(DILS), ep=EPOCHS, aug=AUG),
                             **{k: v for k, v in res.items() if isinstance(v, (int, float))})
    ab_fp.write_text(json.dumps(ab, indent=1))
