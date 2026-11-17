# -*- coding: utf-8 -*-
"""
v68 GF-IMG — マッチテンソル(MD×δ×表現)を「画像」として 2D CNN(UNet-lite)で解く版
=====================================================================================
GF-MR(gridfilter_v68)は各(MD,δ)セルを点ごとにMLPで見るだけ=δ帯がMD方向に連なる/孤立を見られない。
本版は match tensor を画像 (B, C, H=MD, W=δ) とみなし **2D UNet** に通す:
  → CNN が「同じδ帯が多数のMD行で一貫して光る=本物 / チラつく=cycle alias」という 2D 文脈で
    emission を鋭くする(= double-datum disambiguation)。v64(生GR窓→LB9.46 field過学習)と違い、
    入力は **struct/anchor で δ 整列済み**なので条件が良く汎化が期待できる。
HEAD:
  filter(既定) = CNN emission → 微分可能グリッドベイズフィルタ(gf.bayes_filter, dip連続prior) → posterior-mean δ
  direct        = CNN emission → softmax(δ) → posterior-mean δ(純画像。時間平滑はCNNのMD方向RFが担う=多様性脚)
出力 = center + pmean。center は STRUCT=1→s_tvt / STRUCT=0→base_free(struct-free)。

★struct 有り無し両対応: STRUCT=1/0(gf と共通env)。cache/CV/幾何は gridfilter_v68 を再利用。
実行(cache は gf と共通。STRUCT/EVAL_STRIDE を合わせること):
  EVAL_STRIDE=4 .venv/Scripts/python.exe v68/gridfilter_v68.py cache            # 無ければ先にcache(gf側)
  CV_MODE=field STRUCT=1 .venv/Scripts/python.exe v68/gridimg_v68.py            # struct有り 画像
  CV_MODE=field STRUCT=0 .venv/Scripts/python.exe v68/gridimg_v68.py            # struct無し 画像
  HEAD=direct CV_MODE=field .venv/Scripts/python.exe v68/gridimg_v68.py         # 純画像(フィルタ無し)
img専用env: IMG_BATCH(4) HEAD(filter|direct) CH(16,32,48) IMG_EPOCHS(=gf.EPOCHS) IMG_LR(=gf.LR)
  ★メモリ: CNN特徴マップは (B,c,T/2,W) 級 → VRAM ∝ B×T×c。IMG_BATCH=4 既定(重ければ 2 or EVAL_STRIDE↑)。
gf共通env: CV_MODE STRUCT EVAL_STRIDE N_FIELDS DR DSTEP GSTEP GR_SIG MA N_WELLS SEED
"""
import os, sys, json, math, time
from pathlib import Path
import numpy as np
sys.stdout = open(1, "w", encoding="utf-8", closefd=False)
sys.path.insert(0, str(Path(__file__).parent))
import gridfilter_v68 as gf
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- img専用 env ----
IMG_BATCH  = int(os.environ.get("IMG_BATCH", "4"))
HEAD       = os.environ.get("HEAD", "filter")            # filter | direct
CH         = tuple(int(x) for x in os.environ.get("CH", "16,32,48").split(","))
IMG_EPOCHS = int(os.environ.get("IMG_EPOCHS", str(gf.EPOCHS)))
IMG_LR     = float(os.environ.get("IMG_LR", str(gf.LR)))
DEVICE     = "cuda" if (os.environ.get("DEVICE", "auto") != "cpu" and torch.cuda.is_available()) else "cpu"


def cblock(cin, cout, stride=(1, 1)):
    # padding最小・norm無し(padded行汚染を避ける)。小容量UNet=過学習ガード
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride=stride, padding=1), nn.SiLU(),
                         nn.Conv2d(cout, cout, 3, padding=1), nn.SiLU())


class UNetLite(nn.Module):
    """MD(H)方向のみ downsample(δ=W は全解像度維持=oracle 0.3 の δ 分解能を保つ)。"""
    def __init__(self, cin, ch=CH):
        super().__init__()
        c1, c2, c3 = ch
        self.stem = cblock(cin, c1, stride=(2, 1))    # H/2
        self.e1 = cblock(c1, c2, stride=(2, 1))       # H/4
        self.e2 = cblock(c2, c3, stride=(2, 1))       # H/8
        self.u1 = cblock(c3 + c2, c2)
        self.u0 = cblock(c2 + c1, c1)
        self.head = nn.Conv2d(c1, 1, 1)

    def forward(self, x):                              # x:(B,C,H,W)
        H, W = x.shape[-2:]
        s = self.stem(x); a = self.e1(s); b = self.e2(a)
        u = self.u1(torch.cat([F.interpolate(b, size=a.shape[-2:], mode="nearest"), a], 1))
        u = self.u0(torch.cat([F.interpolate(u, size=s.shape[-2:], mode="nearest"), s], 1))
        o = self.head(u)                              # (B,1,H/2,W)
        return F.interpolate(o, size=(H, W), mode="bilinear", align_corners=False)  # (B,1,H,W)


class GFIMG(nn.Module):
    def __init__(self, nch=gf.NCH):
        super().__init__()
        self.cnn = UNetLite(nch)
        self.log_trans = nn.Parameter(torch.tensor(math.log(gf.TRANS_SIG0)))

    def kernel(self, device):
        s = torch.exp(self.log_trans).clamp(0.3, 12.0)
        K = int((s * 4).clamp(3, 40).item()) | 1
        x = torch.arange(K, device=device, dtype=torch.float32) - K // 2
        k = torch.exp(-0.5 * (x / s) ** 2); return (k / k.sum())[None, None, :]


def run_img(model, bt, delta, device):
    ch = gf.build_channels(bt, delta, use_struct=gf.USE_STRUCT, center_key=gf.CENTER_KEY)  # (B,T,D,C)
    x = ch.permute(0, 3, 1, 2).contiguous()                       # (B,C,T,D) 画像
    logit = model.cnn(x).squeeze(1)                               # (B,T,D)
    if HEAD == "filter":
        E = torch.exp(torch.log_softmax(logit, dim=2))
        ker = model.kernel(device)
        if gf.USE_STRUCT:
            s_sig0 = bt["s_sig"][:, 0:1].clamp(2, 60)
        else:
            s_sig0 = torch.full((x.shape[0], 1), float(delta[-1]) / 2, device=device)
        pmean, _ = gf.bayes_filter(E, ker, delta, s_sig0)
    else:                                                         # direct: 純画像(CNNのMD方向RFで時間平滑)
        pmean = (torch.softmax(logit, dim=2) * delta[None, None, :]).sum(2)
    return bt[gf.CENTER_KEY] + pmean


def train_fold(wtr, wva, delta, device):
    model = GFIMG().to(device); opt = torch.optim.Adam(model.parameters(), lr=IMG_LR)
    batches = gf._len_buckets(wtr, IMG_BATCH)
    rng = np.random.RandomState(gf.SEED); border = np.arange(len(batches))
    Ttot = sum(len(w["hgr"]) for w in wtr)
    for ep in range(IMG_EPOCHS):
        model.train(); rng.shuffle(border); tot = 0.0; nb = 0; done = 0; t0 = time.time(); last = t0
        for bi in border:
            batch = [wtr[j] for j in batches[bi]]; bt = gf.collate(batch, device)
            tvt = run_img(model, bt, delta, device)
            loss = (((tvt - bt["tgt"]) ** 2) * bt["mask"]).sum() / bt["mask"].sum().clamp_min(1)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
            tot += float(loss); nb += 1; done += sum(len(w["hgr"]) for w in batch)
            if time.time() - last > 20:
                el = time.time() - t0; eta = el / max(done, 1) * (Ttot - done)
                print(f"      ep{ep+1} {nb}/{len(batches)}b rows{done//1000}k/{Ttot//1000}k "
                      f"rmse~{math.sqrt(tot/max(nb,1)):.2f} {el:.0f}s eta{eta:.0f}s", flush=True); last = time.time()
        print(f"    ep{ep+1}/{IMG_EPOCHS} mse={tot/max(nb,1):.3f} (rmse~{math.sqrt(tot/max(nb,1)):.3f}) {time.time()-t0:.0f}s", flush=True)
    model.eval(); oof = {}
    with torch.no_grad():
        for bidx in gf._len_buckets(wva, IMG_BATCH):
            batch = [wva[j] for j in bidx]; bt = gf.collate(batch, device)
            tvt = run_img(model, bt, delta, device)
            for b, w in enumerate(batch): oof[w["wid"]] = tvt[b, :len(w["hgr"])].cpu().numpy()
    return oof


def main():
    torch.manual_seed(gf.SEED); np.random.seed(gf.SEED)
    wells = gf.load_cache()
    if gf.CENTER_KEY not in wells[0]:
        print(f"★cacheに'{gf.CENTER_KEY}'が無い(古いcache)。EVAL_STRIDE={gf.EVAL_STRIDE}でcache作り直し: python gridfilter_v68.py cache"); return
    delta = torch.arange(-gf.DR_EFF, gf.DR_EFF + 1e-6, gf.DSTEP_EFF, device=DEVICE, dtype=torch.float32)
    print(f"[GF-IMG] wells {len(wells)} CV={gf.CV_MODE} STRUCT={int(gf.USE_STRUCT)}(center={gf.CENTER_KEY}) HEAD={HEAD} "
          f"CH={CH} δ=±{gf.DR_EFF:.0f}/{gf.DSTEP_EFF:.0f}={len(delta)}bin cin={gf.NCH} IMG_BATCH={IMG_BATCH} device={DEVICE}")
    # CV split(gf と同じ leave-field-out / well)
    from sklearn.model_selection import GroupKFold
    if gf.CV_MODE == "field":
        fld = gf.make_fields(wells); nf = min(5, len(set(fld))); assign = np.zeros(len(wells), int)
        for k, (_, va) in enumerate(GroupKFold(nf).split(wells, groups=fld)): assign[va] = k
        folds = list(range(nf)); print(f"[cv] leave-field-out: {len(set(fld))}fields → {nf}-fold")
    else:
        g = np.array([w["wid"] for w in wells]); assign = np.zeros(len(wells), int)
        for k, (_, va) in enumerate(GroupKFold(5).split(wells, groups=g)): assign[va] = k
        folds = list(range(5)); print("[cv] well GroupKFold(5)")
    oof = {}; t0 = time.time()
    for k in folds:
        tr = [wells[i] for i in np.where(assign != k)[0]]; va = [wells[i] for i in np.where(assign == k)[0]]
        print(f"  fold {k}: train {len(tr)} / val {len(va)}", flush=True)
        oof.update(train_fold(tr, va, delta, DEVICE))
    # 評価: overall + center基準 + δ格子oracle
    perw, pool, spool, opool = {}, [], [], []
    for w in wells:
        if w["wid"] not in oof: continue
        p = oof[w["wid"]]; tgt = w["target"]; ctr = w[gf.CENTER_KEY]; n = min(len(p), len(tgt))
        e = p[:n] - tgt[:n]; se = ctr[:n] - tgt[:n]
        oe = (np.clip(np.round((tgt[:n] - ctr[:n]) / gf.DSTEP_EFF), -gf.DR_EFF / gf.DSTEP_EFF, gf.DR_EFF / gf.DSTEP_EFF) * gf.DSTEP_EFF + ctr[:n]) - tgt[:n]
        fin = np.isfinite(e)
        if fin.any():
            perw[w["wid"]] = float(np.sqrt(np.mean(e[fin] ** 2))); pool.append(e[fin]); spool.append(se[fin]); opool.append(oe[fin])
    overall = float(np.sqrt(np.mean(np.concatenate(pool) ** 2)))
    base = float(np.sqrt(np.mean(np.concatenate(spool) ** 2)))
    oracle = float(np.sqrt(np.mean(np.concatenate(opool) ** 2)))
    tag = f"img_{gf.CV_MODE}_{'st' if gf.USE_STRUCT else 'free'}_{HEAD}"
    blab = "struct基準" if gf.USE_STRUCT else "anchor基準(struct-free)"
    print(f"\n[v68 GF-IMG] {tag}  overall RMSE = {overall:.4f}   ({blab} {base:.4f} / δ格子oracle {oracle:.4f})")
    print(f"           坑井別中央 {np.median(list(perw.values())):.4f}  ({len(perw)}坑井, {time.time()-t0:.0f}s)")
    arr = [v for w in wells if w["wid"] in oof for v in oof[w["wid"]]]
    np.save(gf.OUT / f"gf_oof_{tag}.npy", np.array(arr, np.float32))
    (gf.OUT / f"results_{tag}.json").write_text(json.dumps(dict(
        overall=overall, center_base=base, oracle_dgrid=oracle, head=HEAD, struct=gf.USE_STRUCT,
        per_well_median=float(np.median(list(perw.values()))), n_wells=len(perw), cv=gf.CV_MODE,
        cfg=dict(ch=list(CH), dr=gf.DR, dstep=gf.DSTEP, gstep=gf.GSTEP, gr_sig=gf.GR_SIG,
                 img_batch=IMG_BATCH, epochs=IMG_EPOCHS, eval_stride=gf.EVAL_STRIDE)), indent=2))
    print(f"[saved] gf_oof_{tag}.npy / results_{tag}.json")


if __name__ == "__main__":
    main()
