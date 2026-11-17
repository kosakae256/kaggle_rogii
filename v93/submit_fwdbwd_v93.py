# -*- coding: utf-8 -*-
"""v93 fwd+backward 提出パイプライン(tabular GBM + TCN + L5)。LOCAL: 可視 test 3井で検証。

段取り:
  1. forward test 生成(submit_v93.run_gen 相当): GRフリー test 錨 -> test sim -> create test.parquet
  2. ★backward test 特徴生成: build_backward_v93.py(SPLIT=test, warm)= test_fwdbwd.parquet(forward pf_* + bpf_*)
  3. ★seqfeat test 生成: build_seqfeat_v93.py(SF_IN=test_fwdbwd, SF_NO_TARGET=1)= seqfeat_test_fwdbwd.pkl
  4. GBM 推論: models_l1_fwdbwd(9var x5fold), features_l1_fwdbwd.json, test_fwdbwd.parquet -> gbm_pred(target単位)
  5. TCN 推論: submit_pack_nn_top192_fwdbwd.json + combo_top192_fwdbwd ckpt(15) -> tcn_pred
  6. L5 推論: l5_fwdbwd_ckpt/*_model.pt(15) + seqfeat_test -> l5_pred(per-bin->per-row, target単位)
  7. blend: OOF(l1_oof_fwdbwd / nn_oof_combo_top192_fwdbwd / l5_fwdbwd_oof)から nnls 重み -> test 3脚へ適用
  8. tvt = last_known_tvt + pred。submission_fwdbwd_v93.csv

struct 禁止 / cp932 安全('->') / TCN SDPA は float32(T4 OOM 回避)。
実行: rogii_claude/.venv/Scripts/python.exe v93/submit_fwdbwd_v93.py
      SKIP_GEN=1 で 1(錨/sim/parquet)を省略 / SKIP_BWD=1 で 2-3(backward/seqfeat)を省略
"""
import os, sys, io, json, glob, subprocess
try:
    sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
    sys.stderr = io.open(2, "w", encoding="utf-8", closefd=False)
except Exception:
    pass
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import nnls
from sklearn.model_selection import GroupKFold

PROJ = Path(r"C:\Users\kosaka256\Documents\rogii_claude")
V93 = PROJ / "v93"; ART = V93 / "artifacts_v93"
DATA = PROJ / "rogii-wellbore-geology-prediction"
TESTDIR = DATA / "test"
PY = str(PROJ / ".venv" / "Scripts" / "python.exe")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TEST_ANCHOR = ART / "grfree_anchor_test.pkl"
TEST_SIM = ART / "sim_grfree_test_v93.pkl"
TEST_PARQUET = ART / "data" / "test.parquet"
SAMPLE = DATA / "sample_submission.csv"
NN_DIR = ART / "nn"
BASE_TRAIN = ART / "data" / "train.parquet"

# ---- VARIANT 切替(env VARIANT): warm(既定, forward終端init/top192 TCN) / anchor(錨終端init/全特徴 TCN) ----
VARIANT = os.environ.get("VARIANT", "warm").lower()
if VARIANT == "anchor":
    VSUF = "_fwdbwd_anchor"; TCN_SUF = "_fwdbwd_anchor"; BWD_INIT = "anchor"
else:
    VSUF = "_fwdbwd"; TCN_SUF = "_top192_fwdbwd"; BWD_INIT = "warm"
TEST_FWDBWD = ART / "data" / f"test{VSUF}.parquet"
SEQFEAT_TEST = ART / f"seqfeat_test{VSUF}.pkl"
FEATS_L1_FB = ART / f"features_l1{VSUF}.json"
MODELS_FB = ART / f"models_l1{VSUF}"
SUBMIT_PACK_FB = ART / f"submit_pack_nn{TCN_SUF}.json"
TCN_CKPT_FMT = f"combo{TCN_SUF}_mdn2_s{{s}}_fold{{k}}.pt"
L5_CKPT = ART / f"l5{VSUF}_ckpt"
OOF_GBM = ART / f"l1_oof{VSUF}.npz"
OOF_TCN = ART / f"nn_oof_combo{TCN_SUF}.npz"
OOF_L5 = ART / f"l5{VSUF}_oof.npz"
SUB_CSV = ART / f"submission{VSUF}_v93.csv"

VARIANTS = ["lightgbm-1", "lightgbm-2", "lightgbm-3", "catboost-1", "catboost-2", "catboost-3",
            "xgboost-1", "xgboost-2", "xgboost-3"]
N_SPLITS = 5
DIM = 64   # L5 の DIM(train_l456 既定)


# =====================================================================
# ステップ1-3: 生成(サブプロセス。env-before-import 保証)
# =====================================================================
def run_gen():
    print("\n==== [1] forward test(錨/sim/create) ====", flush=True)
    if os.environ.get("SKIP_GEN", "0") == "1":
        print("[SKIP_GEN=1] forward test 生成をスキップ", flush=True)
    else:
        if not (TEST_ANCHOR.exists() and os.environ.get("FORCE_ANCHOR", "0") != "1"):
            if subprocess.run([PY, str(V93 / "gen_grfree_test_anchor_v93.py")], env=dict(os.environ)).returncode:
                raise SystemExit("test 錨生成に失敗")
        env = dict(os.environ); env["NN_SPLIT"] = "test"; env["V93_ANCHOR_PKL"] = str(TEST_ANCHOR)
        if subprocess.run([PY, str(V93 / "nn_emission_v93.py")], env=env).returncode:
            raise SystemExit("test sim 生成に失敗")
        env = dict(os.environ); env["SPLIT"] = "test"; env["V93_ANCHOR_PKL"] = str(TEST_ANCHOR)
        env["SIM_PKL"] = str(TEST_SIM); env["OUT_PARQUET"] = str(TEST_PARQUET)
        if subprocess.run([PY, str(V93 / "create_v93.py")], env=env).returncode:
            raise SystemExit("create test.parquet に失敗")
    assert TEST_PARQUET.exists(), f"{TEST_PARQUET} が無い"


def run_backward_and_seqfeat():
    print("\n==== [2-3] backward test + seqfeat test ====", flush=True)
    if os.environ.get("SKIP_BWD", "0") == "1" and TEST_FWDBWD.exists() and SEQFEAT_TEST.exists():
        print("[SKIP_BWD=1] backward/seqfeat をスキップ(既存使用)", flush=True); return
    # backward: SPLIT=test, warm, test 錨/sim, OUT=test_fwdbwd.parquet
    env = dict(os.environ)
    env.update(SPLIT="test", BWD_INIT=BWD_INIT, SKIP_STANDALONE="1",
               V93_ANCHOR_PKL=str(TEST_ANCHOR), V93_FWD_PARQUET=str(TEST_PARQUET),
               SIM_PKL=str(TEST_SIM), OUT_MERGED=str(TEST_FWDBWD), EMIT_VS_FWD="1")
    if subprocess.run([PY, str(V93 / "build_backward_v93.py")], env=env).returncode:
        raise SystemExit("backward test 生成に失敗")
    assert TEST_FWDBWD.exists(), f"{TEST_FWDBWD} が無い"
    # seqfeat: test_fwdbwd を入力, target 無し
    env = dict(os.environ)
    env.update(V93_INPUT="fwdbwd", SF_IN=str(TEST_FWDBWD), SF_OUT=str(SEQFEAT_TEST), SF_NO_TARGET="1")
    if subprocess.run([PY, str(V93 / "build_seqfeat_v93.py")], env=env).returncode:
        raise SystemExit("seqfeat test 生成に失敗")
    assert SEQFEAT_TEST.exists(), f"{SEQFEAT_TEST} が無い"


# =====================================================================
# GBM 推論(fwdbwd)
# =====================================================================
def gbm_predict():
    features = json.loads(FEATS_L1_FB.read_text())
    names = set(pq.read_schema(TEST_FWDBWD).names)
    miss = [c for c in features if c not in names]
    assert not miss, f"test_fwdbwd に GBM 特徴不足: {miss[:8]}"
    tb = pq.read_table(TEST_FWDBWD, columns=features); N = tb.num_rows
    X = np.empty((N, len(features)), np.float32)
    for j in range(len(features)):
        X[:, j] = tb.column(j).to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
    del tb
    print(f"[GBM] test rows={N} feats={len(features)}", flush=True)
    var_preds = []
    for name in VARIANTS:
        fp = [np.asarray(joblib.load(MODELS_FB / name / f"fold{k}.pkl").predict(X), np.float64)
              for k in range(N_SPLITS)]
        var_preds.append(np.mean(fp, axis=0))
    gbm_pred = np.mean(var_preds, axis=0)
    print(f"[GBM] blend(9) mean={gbm_pred.mean():.3f}", flush=True)
    return gbm_pred


# =====================================================================
# TCN(SeqNet mdn2 combo)推論 --- submit_v93 と同一構造(fwdbwd 版パス)
# =====================================================================
class ChanLN(nn.Module):
    def __init__(self, c): super().__init__(); self.ln = nn.LayerNorm(c)
    def forward(self, x): return self.ln(x.transpose(1, 2)).transpose(1, 2)
class Block(nn.Module):
    def __init__(self, c, dil, k, drop):
        super().__init__(); self.conv = nn.Conv1d(c, c, k, padding=dil*(k-1)//2, dilation=dil)
        self.ln = ChanLN(c); self.pw = nn.Conv1d(c, c, 1); self.do = nn.Dropout(drop)
    def forward(self, x, m):
        h = self.conv(x*m); h = F.gelu(h); h = self.ln(h); h = self.do(self.pw(h)); return (x+h)*m
class SelfAttn(nn.Module):
    def __init__(self, ch, heads, drop):
        super().__init__(); self.h = heads; self.dk = ch//heads; self.qkv = nn.Linear(ch, 3*ch); self.proj = nn.Linear(ch, ch)
    def forward(self, x, kmask):
        B, L, C = x.shape; q, k, v = self.qkv(x).chunk(3, -1)
        q = q.view(B, L, self.h, self.dk).transpose(1, 2); k = k.view(B, L, self.h, self.dk).transpose(1, 2)
        v = v.view(B, L, self.h, self.dk).transpose(1, 2)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=kmask[:, None, None, :], dropout_p=0.0)
        return self.proj(o.transpose(1, 2).reshape(B, L, C))
class PrefixEnc(nn.Module):
    def __init__(self, c_in, ctx):
        super().__init__(); self.c1 = nn.Conv1d(c_in, 64, 5, padding=2)
        self.c2 = nn.Conv1d(64, 64, 5, padding=8, dilation=4); self.fc = nn.Linear(128, ctx)
    def forward(self, x, m):
        h = F.gelu(self.c1(x*m)); h = F.gelu(self.c2(h))*m; s = m.sum(-1)
        mean = h.sum(-1)/s.clamp(min=1.0); mx = (h+(m-1.0)*1e4).amax(-1)
        mx = torch.where(s > 0, mx, torch.zeros_like(mx))
        return self.fc(torch.cat([mean, mx], dim=-1))*(s > 0).float()
class SeqNet(nn.Module):
    def __init__(self, k_in, head, ch, ksize, dilations, drop, use_attn, attn_heads, ch_drop,
                 use_ctx=False, pre_c=4, ctx_dim=64, n_mtp=0):
        super().__init__(); self.head = head; self.use_ctx = use_ctx
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
        h = self.inp(self.indrop(x))*m
        if self.use_ctx and px is not None:
            h = (h + self.ctxp(self.pre(px, pm))[:, :, None])*m
        for b in self.blocks: h = b(h, m)
        if self.use_attn:
            q = self.aln(h.transpose(1, 2)); a = self.attn(q, (m[:, 0, :] > 0.5))
            h = (h + a.transpose(1, 2))*m
        return self.out(h), None
def predict_head(out):
    pl = torch.sigmoid(out[:, 2]); return pl*out[:, 0] + (1-pl)*out[:, 1]


def _load_test_X(feats_nn, abs2res, med, scale):
    cols = list(dict.fromkeys(feats_nn + ["well", "id", "last_known_tvt", "md_since"]))
    tb = pq.read_table(TEST_FWDBWD, columns=cols); N = tb.num_rows
    well = tb.column("well").to_pandas().to_numpy(); ids = tb.column("id").to_pandas().to_numpy()
    lk = tb.column("last_known_tvt").to_numpy(zero_copy_only=False).astype(np.float64)
    mds = tb.column("md_since").to_numpy(zero_copy_only=False).astype(np.float64)
    X = np.empty((N, len(feats_nn)), np.float32)
    for j, c in enumerate(feats_nn):
        X[:, j] = tb.column(c).to_numpy(zero_copy_only=False).astype(np.float32)
    del tb
    a2r = np.asarray(abs2res, bool); X[:, a2r] -= lk.astype(np.float32)[:, None]
    X -= np.asarray(med, np.float32); X /= np.asarray(scale, np.float32)
    np.nan_to_num(X, copy=False, nan=0.0, posinf=8.0, neginf=-8.0); np.clip(X, -8.0, 8.0, out=X)
    return N, well, ids, lk, mds, X


def _build_prefix(wells_unique, t_scale, gmed, giqr, pre_len, pre_c):
    PREF = {}
    for w in wells_unique:
        f = TESTDIR / f"{w}__horizontal_well.csv"
        if not f.exists(): continue
        df = pd.read_csv(f, usecols=["MD", "Z", "GR", "TVT_input"])
        kn = df.dropna(subset=["MD", "Z", "GR", "TVT_input"]).tail(pre_len)
        if len(kn) < 8: continue
        a = kn.to_numpy(np.float64)
        gr_n = np.clip((a[:, 2]-gmed)/giqr, -8, 8); tvt_rel = (a[:, 3]-a[-1, 3])/t_scale
        z_rel = (a[:, 1]-a[-1, 1])/50.0; md_rel = (a[:, 0]-a[-1, 0])/1000.0
        PREF[w] = np.nan_to_num(np.stack([gr_n, tvt_rel, z_rel, md_rel], 1), nan=0.0, posinf=8.0, neginf=-8.0).astype(np.float32)
    return PREF


def tcn_predict():
    pack = json.loads(SUBMIT_PACK_FB.read_text())
    feats_nn = pack["features"]; arch = pack["arch"]
    t_mean = float(pack["t_mean"]); t_scale = float(pack["t_scale"])
    gmed = pack["prefix_stats"]["gmed"]; giqr = pack["prefix_stats"]["giqr"]
    use_flip = bool(pack["cfg"]["flip"]); use_prefix = bool(pack["cfg"]["prefix"])
    N, well, ids, lk, mds, X = _load_test_X(feats_nn, pack["abs2res"], pack["med"], pack["scale"])
    K_IN = X.shape[1]
    print(f"[TCN] test rows={N} K={K_IN} flip={use_flip} prefix={use_prefix}", flush=True)
    wid_order = pd.unique(well); well_rows = {w: np.where(well == w)[0] for w in wid_order}
    for w in wid_order:
        r = well_rows[w]; well_rows[w] = r[np.argsort(mds[r], kind="stable")]
    PREF = _build_prefix(wid_order, t_scale, gmed, giqr, arch["pre_len"], arch["pre_c"]) if use_prefix else {}

    def new_model():
        return SeqNet(K_IN, "mdn2", arch["ch"], arch["ksize"], arch["dilations"], arch["drop"],
                      arch["use_attn"], arch["attn_heads"], arch["ch_drop"],
                      use_ctx=use_prefix, pre_c=arch["pre_c"], ctx_dim=arch["ctx_dim"], n_mtp=0).to(DEVICE)

    @torch.no_grad()
    def predict_well(model, r, flip):
        rr = r[::-1] if flip else r; L = len(rr)
        x = torch.from_numpy(np.ascontiguousarray(X[rr][None])).to(DEVICE).transpose(1, 2)
        m = torch.ones((1, 1, L), device=DEVICE); px = pm = None
        if use_prefix:
            a = PREF.get(well[r[0]], np.zeros((1, arch["pre_c"]), np.float32))
            px = torch.from_numpy(np.ascontiguousarray(a[None])).to(DEVICE).transpose(1, 2)
            pm = torch.ones((1, 1, len(a)), device=DEVICE)
        o = model(x, m, px, pm)[0]
        p = predict_head(o.float())[0].cpu().numpy()
        out = np.empty(len(r), np.float64); out[:] = p[::-1] if flip else p; return out

    seeds = pack.get("seeds", [0, 1, 2]); folds = list(range(pack.get("n_splits", N_SPLITS)))
    acc = np.zeros(N, np.float64); nck = 0
    for s in seeds:
        for k in folds:
            fp = NN_DIR / TCN_CKPT_FMT.format(s=s, k=k)
            if not fp.exists():
                print(f"   WARN missing {fp.name}"); continue
            model = new_model(); model.load_state_dict(torch.load(fp, map_location=DEVICE)); model.eval()
            pred_ck = np.zeros(N, np.float64)
            for w in wid_order:
                r = well_rows[w]; p = predict_well(model, r, False)
                if use_flip: p = 0.5*(p + predict_well(model, r, True))
                pred_ck[r] = p
            acc += pred_ck; nck += 1; del model
            if DEVICE == "cuda": torch.cuda.empty_cache()
    nn_pred = (acc/max(nck, 1))*t_scale + t_mean
    print(f"[TCN] {nck}ckpt mean={nn_pred.mean():.3f}", flush=True)
    return nn_pred, ids


# =====================================================================
# L5(曲線トークン)推論 --- train_l456 の L5/make_batch/predict_wells を移植
# =====================================================================
class L5Net(nn.Module):
    def __init__(self, k, d=DIM):
        super().__init__(); self.k = k
        self.enc = nn.Sequential(nn.Conv1d(2, d, 9, padding=4), nn.GELU(),
                                 nn.Conv1d(d, d, 9, padding=16, dilation=4), nn.GELU(),
                                 nn.Conv1d(d, d, 9, padding=64, dilation=16))
        self.emb = nn.Embedding(k, d)
        self.qry = nn.Sequential(nn.Linear(3+4, d), nn.GELU(), nn.Linear(d, d))
        self.res = nn.Linear(d, 1)
    def forward(self, D, S, C, m):
        B, T, k = D.shape
        med = D.median(2, keepdim=True).values
        x = torch.stack([D-med, torch.log(S)], 2).permute(0, 3, 2, 1).reshape(B*k, 2, T)
        hc = self.enc(x).reshape(B, k, -1, T).permute(0, 3, 1, 2)
        hc = hc + self.emb.weight[None, None]
        stats = torch.cat([C, med/10.0, D.std(2, keepdim=True),
                           (D-med).abs().lt(2).float().mean(2, keepdim=True), torch.zeros_like(med)], -1)
        q = self.qry(stats)
        att = (hc @ q[..., None])[..., 0] / (hc.shape[-1]**0.5)
        w = att.softmax(-1); out = (w*D).sum(-1)
        return out + 2.0*torch.tanh(self.res(q)[..., 0])


def l5_predict(N):
    import pickle
    SF = pickle.loads(SEQFEAT_TEST.read_bytes())
    WELLS = sorted(SF.keys()); K = SF[WELLS[0]]["D"].shape[1]
    print(f"[L5] test wells={len(WELLS)} K={K}", flush=True)
    models = sorted(glob.glob(str(L5_CKPT / "l5_s*_f*_model.pt")))
    assert models, f"L5 モデルが無い: {L5_CKPT}"

    def predict_all(net):
        pred = np.full(N, np.nan, np.float64)
        with torch.no_grad():
            for w in WELLS:
                f = SF[w]; T = f["D"].shape[0]
                D = torch.from_numpy(f["D"])[None].to(DEVICE); S = torch.from_numpy(f["S"])[None].to(DEVICE)
                C = torch.from_numpy(f["C"])[None].to(DEVICE); m = torch.ones((1, T), device=DEVICE)
                dh = net(D, S, C, m)[0, :T].cpu().numpy().astype(np.float64)
                pred[f["rows"]] = dh[f["bid"]]
        return pred

    acc = np.zeros(N, np.float64); nm = 0
    for mp in models:
        ck = torch.load(mp, map_location=DEVICE)
        net = L5Net(ck.get("K", K), ck.get("dim", DIM)).to(DEVICE)
        net.load_state_dict(ck["state"]); net.eval()
        acc += predict_all(net); nm += 1; del net
        if DEVICE == "cuda": torch.cuda.empty_cache()
    l5_pred = acc/max(nm, 1)
    print(f"[L5] {nm}model mean={np.nanmean(l5_pred):.3f}", flush=True)
    return l5_pred


# =====================================================================
# blend 重み(OOF から nnls)
# =====================================================================
def blend_weights():
    meta = pq.read_table(BASE_TRAIN, columns=["well", "target"])
    well = meta.column("well").to_pandas().to_numpy()
    y = meta.column("target").to_numpy(zero_copy_only=False).astype(np.float64)
    dg = np.load(OOF_GBM, allow_pickle=True)
    gbm = np.mean([dg[k] for k in dg.files if k not in ("y", "well")], 0)
    dt = np.load(OOF_TCN, allow_pickle=True)
    tcn = np.mean([dt[k] for k in dt.files if k.startswith("mdn2")], 0)
    dl = np.load(OOF_L5, allow_pickle=True); l5 = dl["l5_mean"]
    M = np.stack([gbm, tcn, l5], 1)
    ok = np.all(np.isfinite(M), 1) & np.isfinite(y)
    w, _ = nnls(M[ok], y[ok])
    if w.sum() <= 0: w = np.ones(3)
    w = w/w.sum()
    rms = lambda p: float(np.sqrt(np.mean((p[ok]-y[ok])**2)))
    print(f"[blend] OOF: GBM={rms(gbm):.4f} TCN={rms(tcn):.4f} L5={rms(l5):.4f} "
          f"nnls w=[GBM {w[0]:.2f}/TCN {w[1]:.2f}/L5 {w[2]:.2f}] blend={rms(M@w):.4f}", flush=True)
    return w


def main():
    print(f"\n############ submit VARIANT={VARIANT} (BWD_INIT={BWD_INIT}, TCN={TCN_SUF}) ############", flush=True)
    run_gen()
    run_backward_and_seqfeat()
    meta = pq.read_table(TEST_FWDBWD, columns=["id", "last_known_tvt"])
    ids = meta.column("id").to_pandas().to_numpy()
    lk = meta.column("last_known_tvt").to_numpy(zero_copy_only=False).astype(np.float64)
    N = len(ids)

    gbm_pred = gbm_predict()
    tcn_pred, ids_t = tcn_predict()
    assert np.array_equal(ids_t, ids), "TCN と GBM の id 並び不一致"
    l5_pred = l5_predict(N)

    w = blend_weights()
    l5f = np.where(np.isfinite(l5_pred), l5_pred, 0.0)   # L5 欠損行は 0 寄与(GBM/TCN で埋まる)
    pred = w[0]*gbm_pred + w[1]*tcn_pred + w[2]*l5f
    tvt = lk + pred
    sub = pd.DataFrame({"id": ids, "tvt": tvt.astype(np.float64)})
    out_fp = SUB_CSV
    samp = pd.read_csv(SAMPLE); merged = samp[["id"]].merge(sub, on="id", how="left")
    merged.to_csv(out_fp, index=False)
    tv = merged["tvt"].to_numpy()
    print(f"\n==== submission [{VARIANT} GBM+TCN+L5] -> {out_fp.name} ====", flush=True)
    print(f"rows={len(merged)} sample={len(samp)} match={len(merged)==len(samp)} "
          f"missing={int(merged['tvt'].isna().sum())}", flush=True)
    print(f"tvt finite%={100.0*np.isfinite(tv).mean():.2f} min={np.nanmin(tv):.2f} "
          f"max={np.nanmax(tv):.2f} mean={np.nanmean(tv):.2f}", flush=True)
    print("head:\n" + merged.head(6).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
