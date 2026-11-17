# train_agg_v54.ipynb を生成するビルダー(1回きり)
import json, io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

def code(src): return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src.splitlines(keepends=True)}
def md(src): return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}

c0 = md("""# v54 train — 候補集合の学習集約(per-seed軌跡 set aggregation)
- 入力: `artifacts_v54/cand_{bank}_{flav}.npz`(per-seed平滑軌跡, 評価行CSV順 — docs/v54.md 整列規約)
- 粒度ノブ(env): `BANKS` / `FLAVORS` / `SEED_STRIDE`(既定2=16seed) / `TOPK`(ww上位k, 0=全部)
- 学習作法: epoch上限なし・early stop(pat12)・**best val loss選択**・ReduceLROnPlateau / fold= sorted[f::5] / まず `FOLDS=0`
- ガード: 集約器は浅く(井戸指紋対策)。fold間分散を必ず見る。良すぎる値(fold0<4.5)はリーク監査。
""")

c1 = code(r'''# ===== 設定 + 候補ロード =====
import os, json, time, gc
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
P0 = Path(r"C:\Users\kosaka256\Documents\rogii_claude")
ART = P0/"v54"/"artifacts_v54"; (ART/"nn54").mkdir(parents=True, exist_ok=True)
TRAIN = P0/"rogii-wellbore-geology-prediction"/"train"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
if DEV == "cuda":
    torch.cuda.set_per_process_memory_fraction(float(os.environ.get("VFRAC", "0.5")))
BANKS = os.environ.get("BANKS", "pf_1,pf_2,pf_3,r0_seed32,r1_seed32,pfA").split(",")
FLAVORS = os.environ.get("FLAVORS", "tw,nbr,self").split(",")
SSTRIDE = int(os.environ.get("SEED_STRIDE", "2"))
TOPK = int(os.environ.get("TOPK", "0"))
FOLDS = [int(x) for x in os.environ.get("FOLDS", "0").split(",")]
ROWSTRIDE = int(os.environ.get("ROWSTRIDE", "4"))     # 学習時の行間引き(評価はフル)
t0 = time.time()
streams = [(b, f) for b in BANKS for f in FLAVORS]
CAND = {}; CWW = {}; wells = None
for b, f in streams:
    d = np.load(ART/f"cand_{b}_{f}.npz")
    ws = sorted(k for k in d.files if not k.endswith("|w"))
    wells = ws if wells is None else [w for w in wells if w in set(ws)]
    for w in ws:
        pr = d[w][::SSTRIDE]; ww = d[f"{w}|w"][::SSTRIDE]
        if TOPK:
            top = np.argsort(-ww)[:TOPK]; pr = pr[top]; ww = ww[top]
        CAND.setdefault(w, []).append(pr); CWW.setdefault(w, []).append(ww / max(ww.sum(), 1e-9))
wells = sorted(wells)
assert len(wells) >= 700, f"候補の共通井戸 {len(wells)} < 700(createが未完/部分キャッシュ疑い)"
SPS = CAND[wells[0]][0].shape[0]
NS = len(streams); NC = NS * SPS
BID = np.repeat([BANKS.index(b) for b, f in streams], SPS)
FID = np.repeat([FLAVORS.index(f) for b, f in streams], SPS)
Y = {}; T_ = {}
for w in wells:
    hw = pd.read_csv(TRAIN/f"{w}__horizontal_well.csv", usecols=["TVT", "TVT_input"])
    ev = hw["TVT_input"].isna().to_numpy()
    Y[w] = hw["TVT"].to_numpy(float)[ev]
    T_[w] = int(ev.sum())
    assert CAND[w][0].shape[1] == T_[w], (w, CAND[w][0].shape, T_[w])
Z = {w: np.concatenate(CAND[w], 0) for w in wells}    # (NC,T) fp16
WWc = {w: np.concatenate(CWW[w], 0).astype(np.float32) for w in wells}
del CAND, CWW; gc.collect()
# chainアンカートークン(v57教訓): v52 GBM OOF(pp後)を絶対値化しeval行CSV順に整列
import pyarrow.parquet as pq
ART52 = P0/"v52"/"artifacts_v52"
tbq = pq.read_table(ART52/"data"/"train.parquet", columns=["id","well","md_since","pf_ancc","last_known_tvt"])
_ids = tbq.column("id").to_pandas().astype(str); _g = tbq.column("well").to_pandas().astype(str).to_numpy()
_mds = np.maximum(tbq.column("md_since").to_numpy(zero_copy_only=False).astype(np.float64), 0.0)
_pfa = tbq.column("pf_ancc").to_numpy(zero_copy_only=False).astype(np.float64)
_lk = tbq.column("last_known_tvt").to_numpy(zero_copy_only=False).astype(np.float64)
del tbq
_oof = np.load(ART52/"gbm_v41_oof.npy").astype(np.float64)
_pp = json.loads((ART52/"pp_params.json").read_text())
_cres = (_oof*(1-_pp["w_pf"]) + (_pfa-_lk)*_pp["w_pf"]) * (1-np.exp(-_mds/_pp["tau"])) * _pp["alpha"]
_rn = _ids.str.rsplit("_", n=1, expand=True)[1].astype(int).to_numpy()
CH = {}
for wn in wells:
    ii = np.where(_g == wn)[0]
    o = np.argsort(_rn[ii], kind="stable")
    ca = (_cres + _lk)[ii[o]]
    CH[wn] = ca[:T_[wn]].astype(np.float32) if len(ca) >= T_[wn] else np.pad(ca, (0, T_[wn]-len(ca)), mode="edge").astype(np.float32)
del _ids, _g, _mds, _pfa, _lk, _oof, _cres, _rn; gc.collect()
# surf文脈(v56): eval行CSV順 [tvt_surf, dist, slope, fmad]
_sn = np.load(P0/"v56"/"artifacts_v56"/"lowo_rows.npz")
SU = {w: (_sn[w][:T_[w]].astype(np.float32) if w in _sn.files else np.zeros((T_[w],4),np.float32)) for w in wells}
# hard重み用: 候補のchainオフセット散布
SPREAD = {w: float(np.std(np.median(Z[w].astype(np.float32) - CH[w][None, :Z[w].shape[1]], 1))) for w in wells}
HARDW = float(os.environ.get("HARDW", "0"))
print(f"wells={len(wells)} streams={NS} seeds/str={SPS} -> 候補{NC}/行 (+chainトークン)  ({time.time()-t0:.0f}s)")
''')

c2 = code(r'''# ===== 基準線(学習なし) =====
def pooled(dh, ws):
    es = [np.asarray(dh[w], np.float64) - Y[w] for w in ws]
    ee = np.concatenate(es); ee = ee[np.isfinite(ee)]
    return float(np.sqrt(np.mean(ee ** 2)))
def stream_mean(w, si):
    zz = Z[w][si*SPS:(si+1)*SPS].astype(np.float64); ww = WWc[w][si*SPS:(si+1)*SPS]
    return (ww[:, None] * zz).sum(0) / max(ww.sum(), 1e-9)
refs = {}
for si, (b, f) in enumerate(streams):
    refs[f"{b}/{f}"] = pooled({w: stream_mean(w, si) for w in wells}, wells)
best_stream = min(refs, key=refs.get); BSI = list(refs).index(best_stream)
dh_mean = {w: (WWc[w][:, None] * Z[w].astype(np.float64)).sum(0) / max(WWc[w].sum(), 1e-9) for w in wells}
dh_med = {w: np.median(Z[w].astype(np.float64), 0) for w in wells}
val0 = wells[FOLDS[0]::5]
print("stream別上位(全井戸):", {k: round(v, 3) for k, v in sorted(refs.items(), key=lambda t: t[1])[:6]})
print(f"best_stream={best_stream} {refs[best_stream]:.3f} / 全候補ww平均 {pooled(dh_mean, wells):.3f} / median {pooled(dh_med, wells):.3f}")
print(f"fold{FOLDS[0]}参照: best_stream {pooled({w: stream_mean(w, BSI) for w in val0}, val0):.3f} / ww平均 {pooled({w: dh_mean[w] for w in val0}, val0):.3f}")
''')

c3 = code(r'''# ===== 集約器(浅いset-attention + 行文脈Conv) =====
DIM = int(os.environ.get("DIM", "32")); TAU = float(os.environ.get("TAU", "1.0"))
class AggNet(nn.Module):
    def __init__(self, nb=len(BANKS) + 1, nf=len(FLAVORS) + 1, dim=DIM):
        super().__init__()
        self.eb = nn.Embedding(nb, dim); self.ef = nn.Embedding(nf, dim)
        self.tok = nn.Sequential(nn.Linear(2, dim), nn.GELU(), nn.Linear(dim, dim))
        self.ctx = nn.Sequential(
            nn.Conv1d(8, dim, 9, padding=4), nn.GELU(),
            nn.Conv1d(dim, dim, 9, padding=4 * 4, dilation=4), nn.GELU(),
            nn.Conv1d(dim, dim, 9, padding=4 * 16, dilation=16))
        self.q = nn.Linear(dim, dim)
        self.u = nn.Linear(dim, 1)
        nn.init.zeros_(self.u.weight); nn.init.zeros_(self.u.bias)
    def forward(self, z, ww, bid, fid, xc):
        med = z.median(1, keepdim=True).values
        dz = z - med
        lw = torch.log(ww.clamp_min(1e-9))[:, :, None].expand_as(dz)
        h = self.tok(torch.stack([dz, lw], -1))
        h = h + self.eb(bid)[None, :, None, :] + self.ef(fid)[None, :, None, :]
        q10, q90 = torch.quantile(z, torch.tensor([0.1, 0.9], device=z.device), dim=1)
        feats = torch.stack([med[:, 0], z.std(1), q90 - q10,
                             (dz.abs() < 2.0).float().mean(1),
                             (ww[:, :, None] * z).sum(1) - med[:, 0],
                             xc[:, 0] - med[:, 0], xc[:, 1] - med[:, 0], xc[:, 2]], 1)
        c = self.ctx(feats).transpose(1, 2)
        score = (h * self.q(c)[:, None]).sum(-1) / (DIM ** 0.5) + self.u(h)[..., 0]
        w = torch.softmax(score / TAU + torch.log(ww.clamp_min(1e-9))[:, :, None], 1)
        return (w * z).sum(1), w
NCT = NC + 1                                             # +chainトークン
BID_T = np.concatenate([BID, [len(BANKS)]]); FID_T = np.concatenate([FID, [len(FLAVORS)]])
def make_batch(ws, stride=1):
    Tm = max((T_[w] + stride - 1) // stride for w in ws); B = len(ws)
    z = torch.zeros(B, NCT, Tm); ww = torch.zeros(B, NCT)
    yv = torch.full((B, Tm), float("nan")); m = torch.zeros(B, Tm)
    xc = torch.zeros(B, 3, Tm)
    for i, w in enumerate(ws):
        zz = torch.from_numpy(Z[w][:, ::stride].astype(np.float32)); t = zz.shape[1]
        z[i, :NC, :t] = zz
        ch = torch.from_numpy(CH[w][::stride].astype(np.float32))[:Tm]
        z[i, NC, :len(ch)] = ch
        ww[i, :NC] = torch.from_numpy(WWc[w]); ww[i, NC] = 1.0 / max(NS, 1)
        yv[i, :t] = torch.from_numpy(np.asarray(Y[w][::stride], np.float32)); m[i, :t] = 1
        su = SU[w][::stride]
        xc[i, 0, :len(ch)] = ch
        xc[i, 1, :len(su)] = torch.from_numpy(su[:, 0])
        xc[i, 2, :len(su)] = torch.from_numpy(np.log1p(np.maximum(su[:, 1], 0)) / 8.0)
    bid = torch.from_numpy(BID_T).long().to(DEV); fid = torch.from_numpy(FID_T).long().to(DEV)
    return z.to(DEV), ww.to(DEV), yv.to(DEV), m.to(DEV), bid, fid, xc.to(DEV)
''')

c4 = code(r'''# ===== 学習(fold別)+ 正直評価 =====
LR = 3e-4; PAT = 12; LRPAT = 4; MAXMIN = float(os.environ.get("MAXMIN", "40"))
BW = int(os.environ.get("BW", "3"))
def batches(ws, rng):
    order = rng.permutation(len(ws)); out = []; cur = []
    for w in [ws[j] for j in order]:
        cur.append(w)
        if len(cur) >= BW: out.append(cur); cur = []
    if cur: out.append(cur)
    return out
RESULTS = {}; OOF = {}
for f in FOLDS:
    val = wells[f::5]; trn = [w for w in wells if w not in set(val)]
    net = AggNet().to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=LRPAT)
    rng = np.random.default_rng(f)
    def val_loss():
        net.eval(); tot = 0.0; n = 0
        with torch.no_grad():
            for ws in batches(val, np.random.default_rng(1)):
                z, ww, yv, m, bid, fid, xc = make_batch(ws, ROWSTRIDE)
                out, _ = net(z, ww, bid, fid, xc)
                ok = m.bool() & torch.isfinite(yv)
                tot += float(F.mse_loss(out[ok], yv[ok])) * int(ok.sum()); n += int(ok.sum())
        return tot / max(n, 1)
    best = float("inf"); best_state = None; bad = 0; ep = 0; t1 = time.time()
    while True:
        ep += 1; net.train(); tl = []
        for ws in batches(trn, rng):
            try:
                z, ww, yv, m, bid, fid, xc = make_batch(ws, ROWSTRIDE)
                out, _ = net(z, ww, bid, fid, xc)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache(); continue
            ok = m.bool() & torch.isfinite(yv)
            if not ok.any(): continue
            if HARDW > 0:
                wgt = torch.tensor([1.0 + HARDW * float(SPREAD[w] > 4.0) for w in ws], device=DEV)[:, None].expand_as(yv)
                se = (out - torch.nan_to_num(yv)) ** 2
                loss = (se[ok] * wgt[ok]).sum() / wgt[ok].sum()
            else:
                loss = F.mse_loss(out[ok], yv[ok])
            if not torch.isfinite(loss): continue
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step(); tl.append(float(loss.detach()))
        vl = val_loss(); sch.step(vl); star = ""
        if vl < best - 1e-4:
            best = vl; best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}; bad = 0; star = " *best"
        else: bad += 1
        print(f"[f{f}] ep{ep} train={np.mean(tl):.3f} val={vl:.3f}{star} (bad={bad}, {time.time()-t1:.0f}s)", flush=True)
        if bad >= PAT or (time.time() - t1) / 60 > MAXMIN:
            print(f"[f{f}] stop @ep{ep} best={best:.3f}"); break
    net.load_state_dict(best_state); net.eval()
    torch.save(net.state_dict(), ART/"nn54"/f"agg_f{f}.pt")
    dh = {}
    with torch.no_grad():
        for w in val:
            z, ww, yv, m, bid, fid, xc = make_batch([w], 1)
            out, _ = net(z, ww, bid, fid, xc)
            dh[w] = out[0, :T_[w]].cpu().numpy()
    OOF.update(dh)
    RESULTS[f] = pooled(dh, val)
    print(f"[foldA{f}] v54 agg RMSE = {RESULTS[f]:.4f}  (fold内基準は上のセル参照)", flush=True)
    del net; gc.collect(); torch.cuda.empty_cache()
print("\n=== v54 集約器:", {k: round(v, 4) for k, v in RESULTS.items()}, "===")
np.savez_compressed(ART/"agg_oof.npz", **OOF)
json.dump({"per_fold": RESULTS, "banks": BANKS, "flavors": FLAVORS, "sstride": SSTRIDE, "topk": TOPK},
          open(ART/"v54_results.json", "w"), indent=1)
''')

nb = {"cells": [c0, c1, c2, c3, c4],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3.14"}},
      "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open(r"c:/Users/kosaka256/Documents/rogii_claude/v54/train_agg_v54.ipynb", "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print("saved v54/train_agg_v54.ipynb")
