# -*- coding: utf-8 -*-
"""v93: extract mdn2 mixture COMPONENTS from the already-trained TCN (NO retraining).

Goal
----
The v93 TCN (mdn2 head) emits 5 outputs per row:
  out[:,0]=mu1, out[:,1]=mu2, out[:,2]=pl_logit (P(mode1)=sigmoid(out2)), out[:,3]=ls1, out[:,4]=ls2.
The saved nn_oof_combo used the mixture MEAN = pl*mu1+(1-pl)*mu2. mu1/mu2 are the two datum
branches (bimodal +-15ft). On ambiguous rows (pl~0.5) the mean smears to the midpoint; a
mode-COMMITTED prediction (pick one branch) fails differently -> decorrelated -> complementary.

This script reproduces the OOF (GroupKFold 5, same as train_nn_v93.py) and, INSTEAD of the mean,
extracts per-row mode params to build "branch-committed" legs, then evaluates whether they help
the {gbm, tcn_mean, l5} blend.

Reuse: preprocessing + SeqNet + forward from submit_v93.py (READ + verbatim). Only differences:
  1) OOF (train.parquet) not test; fold assignment reproduced exactly from train_nn_v93.py.
  2) extract mode params (mu1/mu2/pl/ls1/ls2) instead of only the mean.

Forward path: to reproduce nn_oof_combo EXACTLY (it was built with bf16 autocast in
train_nn_v93._predict_once), the forward here uses bf16 autocast on CUDA as well (per-well, which
is memory-light and numerically equal to the batched training forward since batch samples are
independent). This makes the sanity check (recon mean vs saved nn_oof mean) ~0. Flip TTA is
applied (cfg flip=True) exactly as training: the per-orientation mean is averaged for the sanity
mean; the per-orientation params are averaged for the leg construction.

struct-free: reads only train.parquet + trained combo_mdn2 checkpoints + submit_pack_nn.json + CSVs.
cp932-safe: stdout utf-8, no em-dash. Memory-safe parquet load. float32/bf16 (no fp32 SDPA OOM).

Run (inference, not training):
  .venv/Scripts/python.exe v93/tcn_modes_v93.py
"""
import os, io, sys, json, math, time, gc
try:
    sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
    sys.stderr = io.open(2, "w", encoding="utf-8", closefd=False)
except Exception:
    pass
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import GroupKFold
from scipy.optimize import nnls

PROJ = Path(r"C:\Users\kosaka256\Documents\rogii_claude")
V93 = PROJ / "v93"
ART = V93 / "artifacts_v93"
PARQUET = ART / "data" / "train.parquet"
DATA = PROJ / "rogii-wellbore-geology-prediction"
TRAINDIR = DATA / "train"
SUBMIT_PACK = ART / "submit_pack_nn.json"
NN_DIR = ART / "nn"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_BF16 = (DEVICE == "cuda")   # match train_nn_v93._predict_once (reproduces nn_oof_combo)

N_SPLITS = 5


# =====================================================================
# SeqNet (verbatim from submit_v93.py)
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
                                           dropout_p=0.0)
        return self.proj(o.transpose(1, 2).reshape(B, L, C))


class PrefixEnc(nn.Module):
    def __init__(self, c_in, ctx):
        super().__init__()
        self.c1 = nn.Conv1d(c_in, 64, 5, padding=2)
        self.c2 = nn.Conv1d(64, 64, 5, padding=8, dilation=4)
        self.fc = nn.Linear(128, ctx)
    def forward(self, x, m):
        h = F.gelu(self.c1(x * m)); h = F.gelu(self.c2(h)) * m
        s = m.sum(-1)
        mean = h.sum(-1) / s.clamp(min=1.0)
        mx = (h + (m - 1.0) * 1e4).amax(-1)
        mx = torch.where(s > 0, mx, torch.zeros_like(mx))
        return self.fc(torch.cat([mean, mx], dim=-1)) * (s > 0).float()


class SeqNet(nn.Module):
    def __init__(self, k_in, head, ch, ksize, dilations, drop, use_attn, attn_heads, ch_drop,
                 use_ctx=False, pre_c=4, ctx_dim=64, n_mtp=0):
        super().__init__()
        self.head = head; self.use_ctx = use_ctx
        self.inp = nn.Conv1d(k_in, ch, 1); self.indrop = nn.Dropout(ch_drop)
        if use_ctx:
            self.pre = PrefixEnc(pre_c, ctx_dim); self.ctxp = nn.Linear(ctx_dim, ch)
        self.blocks = nn.ModuleList([Block(ch, d, ksize, drop) for d in dilations])
        self.use_attn = use_attn
        if use_attn:
            self.attn = SelfAttn(ch, attn_heads, drop); self.aln = nn.LayerNorm(ch)
        self.out = nn.Conv1d(ch, 5 if head == "mdn2" else 1, 1)
        self.mtp = nn.Conv1d(ch, n_mtp, 1) if n_mtp else None
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
        return self.out(h), None


# =====================================================================
# Load train.parquet (memory-safe, chunked columns) + preprocess (verbatim submit_v93)
# =====================================================================
def load_train_X(feats_nn, abs2res, med, scale):
    _names = set(pq.read_schema(PARQUET).names)
    assert all(c in _names for c in feats_nn), "features missing from parquet"
    META = ["well", "target", "last_known_tvt", "md_since", "pf_ancc"]
    N = pq.read_metadata(PARQUET).num_rows
    meta_tbl = pq.read_table(PARQUET, columns=META)
    well = meta_tbl.column("well").to_pandas().to_numpy()
    y = meta_tbl.column("target").to_numpy(zero_copy_only=False).astype(np.float64)
    lk = meta_tbl.column("last_known_tvt").to_numpy(zero_copy_only=False).astype(np.float64)
    mds = meta_tbl.column("md_since").to_numpy(zero_copy_only=False).astype(np.float64)
    del meta_tbl; gc.collect()
    X = np.empty((N, len(feats_nn)), np.float32)
    CHUNK = 64
    for s in range(0, len(feats_nn), CHUNK):
        cols = feats_nn[s:s + CHUNK]
        t = pq.read_table(PARQUET, columns=cols)
        for jj, c in enumerate(cols):
            X[:, s + jj] = t.column(jj).to_numpy(zero_copy_only=False).astype(np.float32)
        del t; gc.collect()
    # abs-TVT residualization (subtract lk on abs2res cols) -- verbatim submit_v93
    lkf = lk.astype(np.float32)
    a2r = np.asarray(abs2res, bool)
    X[:, a2r] -= lkf[:, None]
    med = np.asarray(med, np.float32); scale = np.asarray(scale, np.float32)
    X -= med
    X /= scale
    np.nan_to_num(X, copy=False, nan=0.0, posinf=8.0, neginf=-8.0)
    np.clip(X, -8.0, 8.0, out=X)
    return N, well, y, lk, mds, X


def build_prefix(wells_unique, t_scale, gmed, giqr, pre_len):
    """prefix sequences from TRAIN hw CSVs (known-zone tail), same normalization as train_nn."""
    PREF = {}
    for w in wells_unique:
        f = TRAINDIR / f"{w}__horizontal_well.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f, usecols=["MD", "Z", "GR", "TVT_input"])
        kn = df.dropna(subset=["MD", "Z", "GR", "TVT_input"]).tail(pre_len)
        if len(kn) < 8:
            continue
        a = kn.to_numpy(np.float64)                      # [MD, Z, GR, TVT_input]
        gr_n = np.clip((a[:, 2] - gmed) / giqr, -8, 8)
        tvt_rel = (a[:, 3] - a[-1, 3]) / t_scale
        z_rel = (a[:, 1] - a[-1, 1]) / 50.0
        md_rel = (a[:, 0] - a[-1, 0]) / 1000.0
        PREF[w] = np.nan_to_num(np.stack([gr_n, tvt_rel, z_rel, md_rel], axis=1),
                                nan=0.0, posinf=8.0, neginf=-8.0).astype(np.float32)
    return PREF


# =====================================================================
# Main: OOF mode extraction
# =====================================================================
def main():
    t_start = time.time()
    pack = json.loads(SUBMIT_PACK.read_text())
    feats_nn = pack["features"]
    arch = pack["arch"]
    t_mean = float(pack["t_mean"]); t_scale = float(pack["t_scale"])
    gmed = pack["prefix_stats"]["gmed"]; giqr = pack["prefix_stats"]["giqr"]
    use_flip = bool(pack["cfg"]["flip"]); use_prefix = bool(pack["cfg"]["prefix"])
    seeds = pack.get("seeds", [0, 1, 2])
    print(f"[cfg] run={pack.get('run')} seeds={seeds} flip={use_flip} prefix={use_prefix} "
          f"K={len(feats_nn)} t_mean/scale={t_mean:.3f}/{t_scale:.3f} device={DEVICE} bf16={USE_BF16}",
          flush=True)

    N, well, y, lk, mds, X = load_train_X(feats_nn, pack["abs2res"], pack["med"], pack["scale"])
    K_IN = X.shape[1]
    print(f"[data] rows={N} wells={len(pd.unique(well))} K={K_IN} ({time.time()-t_start:.0f}s)", flush=True)

    # fold assignment EXACTLY as train_nn_v93.py
    folds = list(GroupKFold(n_splits=N_SPLITS).split(np.zeros(N), y, groups=well))

    # per-well row order (md_since ascending, stable) -- same as train/submit
    wid_order = pd.unique(well)
    well_rows = {w: np.where(well == w)[0] for w in wid_order}
    for w in wid_order:
        r = well_rows[w]; well_rows[w] = r[np.argsort(mds[r], kind="stable")]

    PREF = build_prefix(wid_order, t_scale, gmed, giqr, arch["pre_len"]) if use_prefix else {}
    if use_prefix:
        print(f"[prefix] {len(PREF)} wells", flush=True)

    def new_model():
        return SeqNet(K_IN, "mdn2", arch["ch"], arch["ksize"], arch["dilations"], arch["drop"],
                      arch["use_attn"], arch["attn_heads"], arch["ch_drop"],
                      use_ctx=use_prefix, pre_c=arch["pre_c"], ctx_dim=arch["ctx_dim"], n_mtp=0).to(DEVICE)

    @torch.no_grad()
    def forward_well(model, r, flip):
        """return raw 5-channel output (5, L) in ORIGINAL row order (flip un-done)."""
        rr = r[::-1] if flip else r
        L = len(rr)
        xb = X[rr][None]                                   # (1,L,K)
        x = torch.from_numpy(np.ascontiguousarray(xb)).to(DEVICE).transpose(1, 2)   # (1,K,L)
        m = torch.ones((1, 1, L), device=DEVICE)
        px = pm = None
        if use_prefix:
            w = well[r[0]]
            a = PREF.get(w)
            if a is None:
                a = np.zeros((1, arch["pre_c"]), np.float32)
            pxb = a[None]
            px = torch.from_numpy(np.ascontiguousarray(pxb)).to(DEVICE).transpose(1, 2)
            pm = torch.ones((1, 1, len(a)), device=DEVICE)
        with torch.autocast(DEVICE, dtype=torch.bfloat16, enabled=USE_BF16):
            o = model(x, m, px, pm)[0]                     # (1,5,L)
        o = o.float()[0].cpu().numpy()                     # (5,L) normalized
        if flip:
            o = o[:, ::-1]                                 # un-flip to original order
        return o                                           # (5, len(r))

    # per-seed OOF param arrays (denormed mu, raw pl/ls)
    seed_mu1 = {s: np.zeros(N, np.float64) for s in seeds}
    seed_mu2 = {s: np.zeros(N, np.float64) for s in seeds}
    seed_pl = {s: np.zeros(N, np.float64) for s in seeds}
    seed_ls1 = {s: np.zeros(N, np.float64) for s in seeds}
    seed_ls2 = {s: np.zeros(N, np.float64) for s in seeds}
    seed_mean = {s: np.zeros(N, np.float64) for s in seeds}   # recon mean (for sanity vs nn_oof)

    for s in seeds:
        for k in range(N_SPLITS):
            fp = NN_DIR / f"combo_mdn2_s{s}_fold{k}.pt"
            if not fp.exists():
                print(f"   WARN missing {fp.name}", flush=True); continue
            model = new_model()
            model.load_state_dict(torch.load(fp, map_location=DEVICE)); model.eval()
            va_rows = folds[k][1]
            va_wells = pd.unique(well[va_rows])
            for w in va_wells:
                r = well_rows[w]
                o = forward_well(model, r, False)          # (5,L)
                if use_flip:
                    o2 = forward_well(model, r, True)
                    # per-orientation mean, then averaged (matches training predict_wells)
                    pl_a = 1.0 / (1.0 + np.exp(-o[2])); pl_b = 1.0 / (1.0 + np.exp(-o2[2]))
                    mu1_a = o[0] * t_scale + t_mean; mu2_a = o[1] * t_scale + t_mean
                    mu1_b = o2[0] * t_scale + t_mean; mu2_b = o2[1] * t_scale + t_mean
                    f_a = pl_a * mu1_a + (1 - pl_a) * mu2_a
                    f_b = pl_b * mu1_b + (1 - pl_b) * mu2_b
                    mean_w = 0.5 * (f_a + f_b)
                    # averaged params for leg construction
                    mu1_w = 0.5 * (mu1_a + mu1_b); mu2_w = 0.5 * (mu2_a + mu2_b)
                    pl_w = 0.5 * (pl_a + pl_b)
                    ls1_w = 0.5 * (o[3] + o2[3]); ls2_w = 0.5 * (o[4] + o2[4])
                else:
                    pl_w = 1.0 / (1.0 + np.exp(-o[2]))
                    mu1_w = o[0] * t_scale + t_mean; mu2_w = o[1] * t_scale + t_mean
                    ls1_w = o[3]; ls2_w = o[4]
                    mean_w = pl_w * mu1_w + (1 - pl_w) * mu2_w
                seed_mu1[s][r] = mu1_w; seed_mu2[s][r] = mu2_w; seed_pl[s][r] = pl_w
                seed_ls1[s][r] = ls1_w; seed_ls2[s][r] = ls2_w; seed_mean[s][r] = mean_w
            del model
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
            print(f"   s{s}_fold{k}: {len(va_wells)} val wells done ({time.time()-t_start:.0f}s)", flush=True)

    # ---- build legs per seed, then 3-seed mean ----
    def leg_committed(s):
        return np.where(seed_pl[s] >= 0.5, seed_mu1[s], seed_mu2[s])
    def leg_sharp(s):
        return np.where(seed_ls1[s] <= seed_ls2[s], seed_mu1[s], seed_mu2[s])

    committed = np.mean([leg_committed(s) for s in seeds], axis=0)
    mode1 = np.mean([seed_mu1[s] for s in seeds], axis=0)
    mode2 = np.mean([seed_mu2[s] for s in seeds], axis=0)
    sharp = np.mean([leg_sharp(s) for s in seeds], axis=0)
    recon_mean = np.mean([seed_mean[s] for s in seeds], axis=0)

    # ---- sanity vs saved nn_oof_combo mean ----
    zc = np.load(ART / "nn_oof_combo.npz")
    nn_saved = np.mean([zc[f"mdn2_s{s}"] for s in seeds], axis=0)
    per_seed_diff = {s: float(np.max(np.abs(seed_mean[s] - zc[f"mdn2_s{s}"]))) for s in seeds}
    max_abs_diff = float(np.max(np.abs(recon_mean - nn_saved)))
    mean_abs_diff = float(np.mean(np.abs(recon_mean - nn_saved)))

    # ---- save ----
    out_fp = ART / "tcn_modes_oof.npz"
    np.savez(out_fp,
             committed=committed, mode1=mode1, mode2=mode2, sharp=sharp, recon_mean=recon_mean,
             y=y,
             **{f"committed_s{s}": leg_committed(s) for s in seeds},
             **{f"mode1_s{s}": seed_mu1[s] for s in seeds},
             **{f"mode2_s{s}": seed_mu2[s] for s in seeds},
             **{f"sharp_s{s}": leg_sharp(s) for s in seeds},
             **{f"pl_s{s}": seed_pl[s] for s in seeds})
    print(f"\n[saved] {out_fp}", flush=True)

    # =====================================================================
    # Blend evaluation
    # =====================================================================
    def rmse(p):
        e = p - y
        return float(np.sqrt(np.mean(e[np.isfinite(e)] ** 2)))

    gbm = np.load(ART / "gbm_l1_oof.npy").astype(np.float64)
    tcn_mean = nn_saved                                    # mean of nn_oof_combo mdn2_s*
    l5 = np.load(ART / "l5_oof.npz")["l5_mean"].astype(np.float64)
    assert gbm.shape[0] == N and l5.shape[0] == N

    print("\n================ SANITY ================", flush=True)
    print(f"recon-mean vs saved nn_oof mean: max|diff|={max_abs_diff:.6f}  mean|diff|={mean_abs_diff:.6f}", flush=True)
    for s in seeds:
        print(f"   per-seed s{s}: max|diff|={per_seed_diff[s]:.6f}", flush=True)

    print("\n================ STANDALONE RMSE (target/residual units) ================", flush=True)
    print(f"   gbm            = {rmse(gbm):.4f}")
    print(f"   tcn_mean       = {rmse(tcn_mean):.4f}")
    print(f"   l5             = {rmse(l5):.4f}")
    print(f"   committed      = {rmse(committed):.4f}")
    print(f"   mode1          = {rmse(mode1):.4f}")
    print(f"   mode2          = {rmse(mode2):.4f}")
    print(f"   sharp          = {rmse(sharp):.4f}")
    print(f"   recon_mean     = {rmse(recon_mean):.4f}  (should ~= tcn_mean)")

    def corr(a, b):
        m = np.isfinite(a) & np.isfinite(b)
        return float(np.corrcoef(a[m], b[m])[0, 1])

    print("\n================ CORRELATION of committed vs existing legs ================", flush=True)
    print(f"   corr(committed, gbm)      = {corr(committed, gbm):.4f}")
    print(f"   corr(committed, tcn_mean) = {corr(committed, tcn_mean):.4f}")
    print(f"   corr(committed, l5)       = {corr(committed, l5):.4f}")
    print(f"   [ref] corr(tcn_mean, gbm) = {corr(tcn_mean, gbm):.4f}")
    print(f"   [ref] corr(tcn_mean, l5)  = {corr(tcn_mean, l5):.4f}")
    print(f"   [ref] corr(gbm, l5)       = {corr(gbm, l5):.4f}")
    print(f"   corr(mode2, gbm/tcn/l5)   = {corr(mode2,gbm):.4f}/{corr(mode2,tcn_mean):.4f}/{corr(mode2,l5):.4f}")
    print(f"   corr(sharp, gbm/tcn/l5)   = {corr(sharp,gbm):.4f}/{corr(sharp,tcn_mean):.4f}/{corr(sharp,l5):.4f}")

    def best_blend(legs, names):
        A = np.column_stack(legs)
        m = np.all(np.isfinite(A), axis=1) & np.isfinite(y)
        w, _ = nnls(A[m], y[m])
        pred = A @ w
        r = rmse(pred)
        wn = w / w.sum() if w.sum() > 0 else w
        return r, w, wn, names

    print("\n================ BLEND (scipy nnls) ================", flush=True)
    r3, w3, wn3, n3 = best_blend([gbm, tcn_mean, l5], ["gbm", "tcn_mean", "l5"])
    print(f"3-way {n3}: RMSE={r3:.4f}")
    print(f"      raw w   = {dict(zip(n3, np.round(w3,4)))}")
    print(f"      norm w  = {dict(zip(n3, np.round(wn3,4)))}")

    for extra_name, extra in [("committed", committed), ("mode2", mode2), ("sharp", sharp),
                              ("mode1", mode1)]:
        r4, w4, wn4, n4 = best_blend([gbm, tcn_mean, l5, extra],
                                     ["gbm", "tcn_mean", "l5", extra_name])
        d = r3 - r4
        print(f"\n4-way +{extra_name}: RMSE={r4:.4f}  (delta vs 3-way = {d:+.4f})")
        print(f"      raw w   = {dict(zip(n4, np.round(w4,4)))}")
        print(f"      norm w  = {dict(zip(n4, np.round(wn4,4)))}")

    # verdict for committed
    r4c, w4c, wn4c, _ = best_blend([gbm, tcn_mean, l5, committed],
                                   ["gbm", "tcn_mean", "l5", "committed"])
    w_committed = float(w4c[3]); improve = r3 - r4c
    print("\n================ VERDICT ================", flush=True)
    nonzero = w_committed > 1e-6
    helps = improve > 1e-4
    if nonzero and helps:
        print(f"YES: committed is COMPLEMENTARY. weight={w_committed:.4f} (norm={wn4c[3]:.4f}), "
              f"3-way {r3:.4f} -> 4-way {r4c:.4f} (improvement {improve:.4f} ft).")
    else:
        print(f"NO: committed does NOT add complementary signal. weight={w_committed:.4f}, "
              f"3-way {r3:.4f} -> 4-way {r4c:.4f} (delta {improve:+.4f} ft).")
    print(f"\n[done] total {time.time()-t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
