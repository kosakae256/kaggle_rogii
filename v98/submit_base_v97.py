# -*- coding: utf-8 -*-
"""v93 推論/提出パイプライン(LOCAL: 可視 test 3井で検証)。

段取り:
  1. GRフリー test 錨生成(gen_grfree_test_anchor_v93.py, sig=9.0 に修正) -> grfree_anchor_test.pkl
  2. V93_ANCHOR_PKL=<test錨> を設定して test sim 生成(nn_emission_v93.py NN_SPLIT=test) -> sim_grfree_test_v93.pkl
  3. create test(create_v93.py SPLIT=test, V93_ANCHOR_PKL=<test錨>) -> data/test.parquet
  4. GBM 推論: 45モデル(9variant×5fold)。fold平均→variant等重み平均 = gbm_pred(target単位=残差)
  5. TCN 推論: submit_pack_nn.json の前処理を train_nn_v93.py と厳密一致で再現。15ckpt(3seed×5fold)平均 = nn_pred
  6. blend: pred = (1-w)*gbm + w*nn (w=submit_pack['w_nn']=0.74)。--gbm-only / SUBMIT_MODE=gbm で gbm 単体も出力
  7. tvt = last_known_tvt + pred。id,tvt を submission_v95.csv に保存(gbm 単体は submission_v95_gbmonly.csv)

struct 禁止: struct/v38/v41/gf は一切読まない。錨は GRフリーのみ。
cp932 安全: em-dash 不使用('->')。TCN の SDPA は float32(T4 math-fallback OOM 回避のため bf16 は使わない)。
実行: rogii_claude/.venv/Scripts/python.exe v93/submit_v93.py            (blend)
      rogii_claude/.venv/Scripts/python.exe v93/submit_v93.py --gbm-only (gbm 単体)
      SKIP_GEN=1 ... で 1-3(錨/sim/parquet 生成)を省略(既存を使う)
"""
import os, sys, io, json, math, glob, pickle, subprocess, argparse
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

PROJ = Path(os.environ.get("ROGII_PROJ", r"C:\Users\kosaka256\Documents\rogii_claude"))
V95 = Path(os.environ.get("ROGII_SCRIPTS", str(PROJ / "v97")))
ART = Path(os.environ.get("ROGII_ART97", str(V95 / "artifacts_v97")))
DATA = Path(os.environ.get("ROGII_DATA", str(PROJ / "rogii-wellbore-geology-prediction")))
TESTDIR = DATA / "test"
PY = os.environ.get("ROGII_PY", str(PROJ / ".venv" / "Scripts" / "python.exe"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TEST_ANCHOR = ART / "grfree_anchor_test.pkl"
TEST_SIM = ART / "sim_grfree_test_v93.pkl"
TEST_PARQUET = ART / "data" / "test.parquet"
FEATS_L1 = ART / "features_l1.json"
SUBMIT_PACK = ART / "submit_pack_nn.json"
MODELS_DIR = ART / "models_l1_fwd"
NN_DIR = ART / "nn"
SAMPLE = DATA / "sample_submission.csv"

VARIANTS = ["lightgbm-1", "lightgbm-2", "lightgbm-3",
            "catboost-1", "catboost-2", "catboost-3",
            "xgboost-1", "xgboost-2", "xgboost-3"]
N_SPLITS = 5


# =====================================================================
# ステップ 1-3: 生成(サブプロセスで env-before-import を保証)
# =====================================================================
def run_gen():
    # 1) test 錨(sig=9.0 修正込み)。既存があれば再学習しない(FORCE_ANCHOR=1 で強制)
    print("\n==== [1/2] GRフリー test 錨 ====", flush=True)
    if TEST_ANCHOR.exists() and os.environ.get("FORCE_ANCHOR", "0") != "1":
        print(f"[skip] {TEST_ANCHOR.name} 既存(FORCE_ANCHOR=1 で再生成)", flush=True)
    else:
        r = subprocess.run([PY, str(V95 / "gen_grfree_test_anchor_v97.py")], env=dict(os.environ))
        if r.returncode != 0:
            raise SystemExit("test 錨生成に失敗")
    # 2) ★v97: NN-emission test-sim 生成(nn_emission_v97 NN_SPLIT=test, 帯中心=test錨tvt)
    TEST_SIM = ART / "sim_grfree_test_v97.pkl"
    print("\n==== [2/3] NN-emission test-sim ====", flush=True)
    if TEST_SIM.exists() and os.environ.get("FORCE_SIM", "0") != "1":
        print(f"[skip] {TEST_SIM.name} 既存(FORCE_SIM=1 で再生成)", flush=True)
    else:
        env = dict(os.environ); env["NN_SPLIT"] = "test"; env["V93_ANCHOR_PKL"] = str(TEST_ANCHOR)
        r = subprocess.run([PY, str(V95 / "nn_emission_v97.py")], env=env)
        if r.returncode != 0:
            raise SystemExit("NN-emission test-sim 生成に失敗")
    # 3) create test parquet(SPLIT=test, V93_ANCHOR_PKL=test錨, SIM_PKL=test-sim ★nn_emission ON)
    print("\n==== [3/3] create test.parquet ====", flush=True)
    env = dict(os.environ)
    env["SPLIT"] = "test"; env["V93_ANCHOR_PKL"] = str(TEST_ANCHOR)
    env["SIM_PKL"] = str(TEST_SIM); env["OUT_PARQUET"] = str(TEST_PARQUET)
    r = subprocess.run([PY, str(V95 / "create_v97.py")], env=env)
    if r.returncode != 0:
        raise SystemExit("create test.parquet に失敗")


# =====================================================================
# GBM 推論
# =====================================================================
def gbm_predict():
    features = json.loads(FEATS_L1.read_text())        # 719列(last_known_tvt 含む, 生値)
    names = set(pq.read_schema(TEST_PARQUET).names)
    miss = [c for c in features if c not in names]
    assert not miss, f"test.parquet に GBM 特徴が不足: {miss[:8]}"
    tb = pq.read_table(TEST_PARQUET, columns=features)
    N = tb.num_rows
    X = np.empty((N, len(features)), np.float32)
    for j, c in enumerate(features):
        X[:, j] = tb.column(j).to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
    del tb
    print(f"[GBM] test rows={N} feats={len(features)}", flush=True)
    var_preds = []
    for name in VARIANTS:
        fold_p = []
        for k in range(N_SPLITS):
            m = joblib.load(MODELS_DIR / name / f"fold{k}.pkl")
            fold_p.append(np.asarray(m.predict(X), np.float64))
            del m
        vp = np.mean(fold_p, axis=0)                    # 5fold 平均
        var_preds.append(vp)
        print(f"   {name:12s} pred[mean/min/max]={vp.mean():.3f}/{vp.min():.3f}/{vp.max():.3f}", flush=True)
    gbm_pred = np.mean(var_preds, axis=0)               # 9variant 等重み平均(l1 OOF と同一)
    print(f"[GBM] blend(9) pred[mean/min/max]={gbm_pred.mean():.3f}/{gbm_pred.min():.3f}/{gbm_pred.max():.3f}", flush=True)
    return gbm_pred


# =====================================================================
# TCN(SeqNet mdn2 combo)推論 --- train_nn_v93.py と厳密一致の前処理/forward
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
        # ★float32 の SDPA(T4 の bf16 math-fallback による B*heads*L^2 実体化 OOM を回避)。
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


def predict_head(out):
    pl = torch.sigmoid(out[:, 2]); return pl * out[:, 0] + (1 - pl) * out[:, 1]   # mdn2 mean


def _load_test_X(feats_nn, abs2res, med, scale):
    """test.parquet を feats_nn 順で読み, abs2res 残差化 -> robust 標準化(train と同一手順)。"""
    cols = list(dict.fromkeys(feats_nn + ["well", "id", "last_known_tvt", "md_since"]))
    tb = pq.read_table(TEST_PARQUET, columns=cols)
    N = tb.num_rows
    well = tb.column("well").to_pandas().to_numpy()
    ids = tb.column("id").to_pandas().to_numpy()
    lk = tb.column("last_known_tvt").to_numpy(zero_copy_only=False).astype(np.float64)
    mds = tb.column("md_since").to_numpy(zero_copy_only=False).astype(np.float64)
    X = np.empty((N, len(feats_nn)), np.float32)
    for j, c in enumerate(feats_nn):
        X[:, j] = tb.column(c).to_numpy(zero_copy_only=False).astype(np.float32)
    del tb
    # abs2res: lk と corr>0.98 だった列は lk を引く(train で確定した bool マスクを適用)
    lkf = lk.astype(np.float32)
    a2r = np.asarray(abs2res, bool)
    X[:, a2r] -= lkf[:, None]
    # robust 標準化(train は subsample 統計。ここは pack の med/scale をそのまま適用)
    med = np.asarray(med, np.float32); scale = np.asarray(scale, np.float32)
    X -= med
    X /= scale
    np.nan_to_num(X, copy=False, nan=0.0, posinf=8.0, neginf=-8.0)
    np.clip(X, -8.0, 8.0, out=X)
    return N, well, ids, lk, mds, X


def _build_prefix(wells_unique, t_scale, gmed, giqr, pre_len, pre_c):
    """test hw CSV から prefix 系列(既知ゾーン tail(PRE_LEN))。train と同一正規化。gmed/giqr は pack から。"""
    PREF = {}
    for w in wells_unique:
        f = TESTDIR / f"{w}__horizontal_well.csv"
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


def tcn_predict():
    pack = json.loads(SUBMIT_PACK.read_text())
    feats_nn = pack["features"]                          # 718(last_known_tvt 除外済)
    arch = pack["arch"]; t_mean = float(pack["t_mean"]); t_scale = float(pack["t_scale"])
    gmed = pack["prefix_stats"]["gmed"]; giqr = pack["prefix_stats"]["giqr"]
    use_flip = bool(pack["cfg"]["flip"]); use_prefix = bool(pack["cfg"]["prefix"])
    N, well, ids, lk, mds, X = _load_test_X(feats_nn, pack["abs2res"], pack["med"], pack["scale"])
    K_IN = X.shape[1]
    print(f"[TCN] test rows={N} K={K_IN} t_mean/scale={t_mean:.3f}/{t_scale:.3f} flip={use_flip} prefix={use_prefix}", flush=True)

    wid_order = pd.unique(well)
    well_rows = {w: np.where(well == w)[0] for w in wid_order}
    for w in wid_order:                                  # md_since 昇順(train と同一)
        r = well_rows[w]; well_rows[w] = r[np.argsort(mds[r], kind="stable")]
    PREF = _build_prefix(wid_order, t_scale, gmed, giqr, arch["pre_len"], arch["pre_c"]) if use_prefix else {}

    def new_model():
        return SeqNet(K_IN, "mdn2", arch["ch"], arch["ksize"], arch["dilations"], arch["drop"],
                      arch["use_attn"], arch["attn_heads"], arch["ch_drop"],
                      use_ctx=use_prefix, pre_c=arch["pre_c"], ctx_dim=arch["ctx_dim"], n_mtp=0).to(DEVICE)

    @torch.no_grad()
    def predict_well(model, r, flip):
        rr = r[::-1] if flip else r
        L = len(rr)
        xb = X[rr][None]                                 # (1,L,K)
        x = torch.from_numpy(np.ascontiguousarray(xb)).to(DEVICE).transpose(1, 2)   # (1,K,L)
        m = torch.ones((1, 1, L), device=DEVICE)
        px = pm = None
        if use_prefix:
            w = well[r[0]]
            a = PREF.get(w)
            if a is None:
                a = np.zeros((1, arch["pre_c"]), np.float32)
            pxb = a[None]                                # (1,Lp,pre_c)
            px = torch.from_numpy(np.ascontiguousarray(pxb)).to(DEVICE).transpose(1, 2)
            pm = torch.ones((1, 1, len(a)), device=DEVICE)
        # ★float32 forward(bf16 autocast は使わない=T4 OOM 回避 & 精度)
        o = model(x, m, px, pm)[0]
        p = predict_head(o.float())[0].cpu().numpy()     # (L,) normalized
        out = np.empty(len(r), np.float64)
        out[:] = p[::-1] if flip else p                  # flip は逆順で埋め戻し
        return out

    seeds = pack.get("seeds", [0, 1, 2]); folds = list(range(pack.get("n_splits", N_SPLITS)))
    ckpts = [(s, k) for s in seeds for k in folds]
    acc = np.zeros(N, np.float64); nck = 0
    for (s, k) in ckpts:
        fp = NN_DIR / f"combo_mdn2_s{s}_fold{k}.pt"
        if not fp.exists():
            print(f"   WARN: missing {fp.name}"); continue
        model = new_model()
        model.load_state_dict(torch.load(fp, map_location=DEVICE)); model.eval()
        pred_ck = np.zeros(N, np.float64)
        for w in wid_order:
            r = well_rows[w]
            p = predict_well(model, r, False)
            if use_flip:
                p = 0.5 * (p + predict_well(model, r, True))
            pred_ck[r] = p
        acc += pred_ck; nck += 1
        del model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        print(f"   ckpt s{s}_fold{k}: raw mean={pred_ck.mean()*t_scale+t_mean:.3f}", flush=True)
    nn_norm = acc / max(nck, 1)
    nn_pred = nn_norm * t_scale + t_mean                 # target 単位(残差)へ復元
    print(f"[TCN] {nck}ckpt 平均 pred[mean/min/max]={nn_pred.mean():.3f}/{nn_pred.min():.3f}/{nn_pred.max():.3f}", flush=True)
    return nn_pred, well, ids, lk


# =====================================================================
# main
# =====================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gbm-only", action="store_true")
    args = ap.parse_args()
    gbm_only = args.gbm_only or os.environ.get("SUBMIT_MODE", "") == "gbm"

    if os.environ.get("SKIP_GEN", "0") != "1":
        run_gen()
    else:
        print("[SKIP_GEN=1] 錨/sim/parquet 生成をスキップ(既存を使用)", flush=True)
    assert TEST_PARQUET.exists(), f"{TEST_PARQUET} が無い"

    gbm_pred = gbm_predict()

    # id/lk の基準は parquet(GBM 読込)。tvt=lk+pred。id 整合のため parquet から直接取る。
    meta = pq.read_table(TEST_PARQUET, columns=["id", "last_known_tvt"])
    ids = meta.column("id").to_pandas().to_numpy()
    lk = meta.column("last_known_tvt").to_numpy(zero_copy_only=False).astype(np.float64)

    if gbm_only:
        pred = gbm_pred
        out_fp = ART / "submission_v95_gbmonly.csv"
        mode = "gbm-only"
    else:
        nn_pred, well_n, ids_n, lk_n = tcn_predict()
        # id 順が GBM と TCN で一致することを確認(両方 parquet 由来なので一致するはず)
        assert np.array_equal(ids_n, ids), "TCN と GBM の id 並びが不一致"
        w = float(pack_w_nn())
        pred = (1.0 - w) * gbm_pred + w * nn_pred
        out_fp = ART / "submission_v95.csv"
        mode = f"blend w_nn={w}"

    tvt = lk + pred
    sub = pd.DataFrame({"id": ids, "tvt": tvt.astype(np.float64)})
    _write_and_report(sub, out_fp, mode)


def pack_w_nn():
    return json.loads(SUBMIT_PACK.read_text())["w_nn"]


def _write_and_report(sub, out_fp, mode):
    # sample_submission の id 順に整列(提出フォーマット厳守)
    samp = pd.read_csv(SAMPLE)
    merged = samp[["id"]].merge(sub, on="id", how="left")
    nmiss = int(merged["tvt"].isna().sum())
    tv = merged["tvt"].to_numpy()
    finite = np.isfinite(tv)
    merged.to_csv(out_fp, index=False)
    print(f"\n==== submission [{mode}] -> {out_fp.name} ====", flush=True)
    print(f"rows={len(merged)}  sample_rows={len(samp)}  match={len(merged)==len(samp)}  missing(id無)= {nmiss}", flush=True)
    print(f"tvt finite%={100.0*finite.mean():.2f}  min={np.nanmin(tv):.2f}  max={np.nanmax(tv):.2f}  mean={np.nanmean(tv):.2f}", flush=True)
    print("head:\n" + merged.head(8).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
