# -*- coding: utf-8 -*-
"""v93 L3 image leg trainer -- STRUCT-FREE 2D CNN over evidence images.

A genuinely different VIEW from the per-row GBM (L1) / sequence TCN (L2) /
curve-token L5: each well is a 2D image EV (C=14, nb, ND=181) = candidate
delta-TVT (relative to last_known_tvt) vs measured-depth-bin, with the 14
struct-free trajectories drawn as soft Gaussian ridges. A small 2D CNN reads
the whole image and emits a per-bin delta-TVT residual (nb,), which is expanded
back to per-row via seqfeat's bid mapping. Because it consumes the 2D spatial
pattern rather than per-row / per-token stats, it should decorrelate from the
other legs.

Model:
  - Conv2d stack (Conv2d + GroupNorm + GELU), downsampling ONLY the ND axis
    (181 -> ~12) while keeping the nb axis, then AdaptiveAvgPool over ND and a
    1x1 head -> per-bin scalar. Output = per-bin median(D) + 2*tanh(residual),
    i.e. a bounded residual around the per-bin trajectory median (v62-style anchor;
    head zero-init so it starts at the median).
  - Trained per WELL (773 well-images = fast, NOT 3.78M rows).

CV = GroupKFold(5) by well, IDENTICAL to L1 GBM / L2 TCN / L5:
  well->fold is reconstructed from the SAME per-row split
  `GroupKFold(5).split(np.zeros(N), y, groups=well)` on the parquet row order, so a
  well is in exactly one fold and the per-row image OOF aligns with the other legs
  for stacking.

Loss = weighted Huber on per-bin dtrue (weight = seqfeat wgt). 3 seeds (SEEDS env,
default 0,1,2). Per-epoch progress + ETA; resumable per (seed,fold) via valpred cache.

Output: v93/artifacts_v93/image_oof.npz
  keys = img_s0 / img_s1 / img_s2 / img_mean / y  (per-row, TARGET units = TVT - last_known_tvt).
Models saved to v93/artifacts_v93/image_ckpt/img_s{seed}_f{fold}.pt.

CRITICAL struct-free guarantee: reads ONLY images_v95b.pkl + seqfeat_v95b_ext.pkl (both
built struct-free from the v93 parquet, re-anchored to last_known_tvt) + the parquet
well/target columns for the fold split. Never touches gf/v50b/struct/v38/v41/imagesR/
imagesF/st (explicit assert). cp932-safe ('->' not em-dash).

env:
  SMOKE=1  -> 1 seed(0) / 1 fold(0) / ~40 wells / 2 epochs: confirms build + one
              forward-backward step + per-row OOF shape == N.
  SEEDS="0,1,2"  DIM=32  VFRAC=0.5  MAXMIN=45  MAXEP  BW=6
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
ART = P0 / "v95" / "artifacts_v95b"
PQ = ART / "data" / "train.parquet"
IMG_FP = ART / "images_v95b.pkl"
SF_FP = ART / "seqfeat_v95b_ext.pkl"
CKPT = ART / "image_ckpt"; CKPT.mkdir(parents=True, exist_ok=True)

# --- struct-free assert: only these struct-free sources may be read ----------
_BANNED = ("gf_feature", "v50b", "struct", "v38", "v41", "imagesr", "imagesf",
           "stagea", "nnpf")
for _p in (IMG_FP, SF_FP, PQ):
    assert not any(b in str(_p).lower() for b in _BANNED), f"banned source path: {_p}"

DIM = int(os.environ.get("DIM", "32")); PAT = 12; LRPAT = 4
VFRAC = float(os.environ.get("VFRAC", "0.5"))
MAXMIN = float(os.environ.get("MAXMIN", "45"))
BW = int(os.environ.get("BW", "6"))               # wells per padded batch
N_SPLITS = 5
SMOKE = os.environ.get("SMOKE", "0") == "1"
MAXEP = int(os.environ.get("MAXEP", "2" if SMOKE else "100000"))
SEEDS = [int(s) for s in os.environ.get("SEEDS", "0" if SMOKE else "0,1,2").split(",") if s != ""]
DEV = "cuda" if torch.cuda.is_available() else "cpu"
if DEV == "cuda":
    torch.cuda.set_per_process_memory_fraction(VFRAC)

t0 = time.time()
IM = pickle.loads(IMG_FP.read_bytes())
SF = pickle.loads(SF_FP.read_bytes())
WELLS_ALL = sorted(set(IM.keys()) & set(SF.keys()))
Cimg = IM[WELLS_ALL[0]]["EV"].shape[0]
ND = IM[WELLS_ALL[0]]["EV"].shape[2]
assert Cimg == 20, f"expected 20 image channels (struct-free), got {Cimg}"
print(f"wells={len(WELLS_ALL)} C_img={Cimg} ND={ND} SMOKE={SMOKE} SEEDS={SEEDS} "
      f"({time.time()-t0:.0f}s)", flush=True)


# ======================= 2D CNN =======================
class ImageCNN(nn.Module):
    """(B, C=14, nb, ND) -> per-bin (B, nb). Downsamples ND only; keeps nb."""
    def __init__(self, cin=14, d=DIM):
        super().__init__()

        def blk(i, o, st):
            return nn.Sequential(
                nn.Conv2d(i, o, (3, 7), padding=(1, 3), stride=(1, st)),
                nn.GroupNorm(8, o), nn.GELU())

        self.net = nn.Sequential(
            blk(cin, d, 2),        # ND 181 -> 91
            blk(d, d, 2),          # 91 -> 46
            blk(d, 2 * d, 2),      # 46 -> 23
            blk(2 * d, 2 * d, 2),  # 23 -> 12
        )
        self.pool = nn.AdaptiveAvgPool2d((None, 1))   # collapse ND, keep nb
        self.head = nn.Conv2d(2 * d, 1, 1)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)

    def forward(self, ev, med):
        # ev: (B, C, nb, ND) ; med: (B, nb) per-bin trajectory median (anchor)
        h = self.net(ev)               # (B, 2d, nb, ~12)
        h = self.pool(h)               # (B, 2d, nb, 1)
        r = self.head(h)[:, 0, :, 0]   # (B, nb)
        return med + 2.0 * torch.tanh(r)   # bounded residual around per-bin median


# ======================= batching (pad over nb) =======================
def make_batch(wns):
    nbmax = max(IM[w]["EV"].shape[1] for w in wns); B = len(wns)
    ev = torch.zeros(B, Cimg, nbmax, ND)
    med = torch.zeros(B, nbmax); yv = torch.full((B, nbmax), float("nan"))
    m = torch.zeros(B, nbmax); wg = torch.zeros(B, nbmax)
    for i, w in enumerate(wns):
        e = IM[w]; f = SF[w]; nb = e["EV"].shape[1]
        ev[i, :, :nb] = torch.from_numpy(e["EV"].astype(np.float32))
        D = torch.from_numpy(np.asarray(f["D"], np.float32))     # (nb,14)
        med[i, :nb] = D.median(1).values                         # per-bin median anchor
        yv[i, :nb] = torch.from_numpy(np.asarray(e["dtrue"], np.float32))
        wg[i, :nb] = torch.from_numpy(np.asarray(e["wgt"], np.float32))
        m[i, :nb] = 1
    return (ev.to(DEV), med.to(DEV), yv.to(DEV), m.to(DEV), wg.to(DEV))


def bucket(ws, rng):
    lens = {w: IM[w]["EV"].shape[1] for w in ws}
    order = rng.permutation(len(ws)); ws_ = [ws[j] for j in order]
    wl = [w for i in range(0, len(ws_), 64) for w in sorted(ws_[i:i + 64], key=lambda x: lens[x])]
    out = []; cur = []
    for w in wl:
        t = cur + [w]
        if cur and len(t) > BW:
            out.append(cur); cur = [w]
        else:
            cur = t
    if cur: out.append(cur)
    return out


def run(net, wns, train):
    ev, med, yv, m, wg = make_batch(wns)
    if train:
        ev = ev + 0.02 * torch.randn_like(ev)      # light image noise
    out = net(ev, med)
    ok = m.bool() & torch.isfinite(yv) & (wg > 0)
    if not ok.any(): return None
    wgn = wg / wg[ok].mean()
    loss = F.huber_loss(out, torch.nan_to_num(yv), reduction="none", delta=3.0)
    return (loss * wgn)[ok].mean()


def val_loss(net, val):
    net.eval(); tot = 0.0; n = 0
    with torch.no_grad():
        for wns in bucket(val, np.random.default_rng(1)):
            l = run(net, wns, False)
            if l is None: continue
            tot += float(l) * len(wns); n += len(wns)
    return tot / max(n, 1)


def train_fold(trn, val, seed):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    net = ImageCNN().to(DEV)
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
        el = time.time() - t1
        eta = el / ep * max(0, (PAT - bad))     # rough ETA to early-stop
        print(f"  s{seed} ep{ep} train={np.mean(tl) if tl else float('nan'):.4f} "
              f"val={vl:.4f}{star} (bad={bad}, {el:.0f}s, eta~{eta:.0f}s)", flush=True)
        if bad >= PAT or ep >= MAXEP or el / 60 > MAXMIN:
            print(f"  s{seed} stop @ep{ep} best={best:.4f}", flush=True)
            break
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    return net


def predict_wells(net, wells):
    """val well -> per-row prediction (TARGET units) via bid expansion."""
    out = {}
    with torch.no_grad():
        for w in wells:
            e = IM[w]; f = SF[w]; nb = e["EV"].shape[1]
            ev, med, yv, m, wg = make_batch([w])
            o = net(ev, med)[0, :nb].cpu().numpy().astype(np.float64)   # per-bin
            pred_row = o[np.asarray(e["bid"], np.int64)]                # -> per-row
            out[w] = (np.asarray(e["rows"], np.int64), pred_row)
    return out


# ======================= per-row GroupKFold (L1/L2/L5 identical) =======================
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
    return N, y, fold_of


# ======================= run =======================
if SMOKE:
    # smoke: avoid the full 3.78M-string GroupKFold; simple ~40-well split.
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
    vrows = np.concatenate([preds[w][0] for w in val])
    rm = float(np.sqrt(np.mean((oof[vrows] - y[vrows]) ** 2)))
    print(f"[SMOKE] one seed/fold done. oof.shape={oof.shape} (==N:{oof.shape[0]==N}) "
          f"val-rows={len(vrows)} val rmse(per-row target)={rm:.4f}")
    print("[SMOKE] OK: CNN builds + forward/backward ran + OOF shape correct.")
    sys.exit(0)

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
    for k in range(N_SPLITS):
        cache = CKPT / f"img_s{seed}_f{k}_valpred.npz"
        val = [w for w in WELLS if fold_of[w] == k]
        trn = [w for w in WELLS if fold_of[w] != k]
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
        torch.save(net.state_dict(), CKPT / f"img_s{seed}_f{k}.pt")
        del net; gc.collect()
        if DEV == "cuda": torch.cuda.empty_cache()
    rm = float(np.sqrt(np.mean((oof[filled] - y[filled]) ** 2)))
    seed_oofs[seed] = oof
    print(f"== seed{seed} OOF per-row(target) RMSE = {rm:.4f}  (filled {filled.sum()}/{N})", flush=True)

mean_oof = np.mean([seed_oofs[s] for s in SEEDS], axis=0)
save = {f"img_s{s}": seed_oofs[s].astype(np.float32) for s in SEEDS}
save["img_mean"] = mean_oof.astype(np.float32)
save["y"] = y.astype(np.float32)
np.savez_compressed(ART / "image_oof.npz", **save)
rm_ens = float(np.sqrt(np.mean((mean_oof - y) ** 2)))
print(f"\n=== v93 L3 IMAGE (struct-free 2D CNN leg) ===")
for s in SEEDS:
    print(f"  seed{s:d} OOF RMSE = {np.sqrt(np.mean((seed_oofs[s]-y)**2)):.4f}")
print(f"  {len(SEEDS)}-seed ensemble OOF RMSE = {rm_ens:.4f}")
print(f"  saved {ART / 'image_oof.npz'}  keys={list(save.keys())}")
