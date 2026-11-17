# -*- coding: utf-8 -*-
"""v96 推論/提出パイプライン(CANDFEAT L5 を加えた 3脚 blend = CV 5.9416)。

構成(v95 からの差分):
  - GBM(9variant x 5fold) / TCN(3seed x 5fold) は submit_v95.py を VERBATIM 再利用(artifacts_v95)。
  - ★L5cf(候補別特徴 CANDFEAT を各候補トークンに loglik/vs_tw の 2役割で付与)を新規追加。
      seqfeat を CANDFEAT=1 NF_MAX=2 で test.parquet から生成 -> l5_cf_ckpt(3seed x 5fold=15)で推論。
  - blend = w_gbm*gbm + w_tcn*tcn + w_l5cf*l5cf(固定重み, 転移安全)。tvt = last_known_tvt + pred。

持ち出しは artifact フォルダのみ:
  artifacts_v95/  = GBM(models_l1_fwd) / TCN(nn) / grfree 錨 / features_l1.json / submit_pack_nn.json
  artifacts_v95b/ = l5_cf_ckpt(L5cf 15ckpt)
  ※ snap 物理脚(-0.027, CV 5.9149)は v89 PF エンジンを test で走らせる必要があり別建て(submit_v96_snap.py)。

struct 禁止: struct/v38/v41/gf は一切読まない。錨は GRフリーのみ。
cp932 安全('->')。TCN の SDPA は float32(T4 OOM 回避)。
実行: .venv/Scripts/python.exe v95/submit_v96.py
      SKIP_GEN=1 で test.parquet 生成をスキップ(既存を使用)
      SKIP_SEQFEAT=1 で L5cf seqfeat 生成をスキップ(既存を使用)
"""
import os, sys, io, json, pickle, subprocess
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

PROJ = Path(os.environ.get("ROGII_PROJ", r"C:\Users\kosaka256\Documents\rogii_claude"))
V95 = Path(os.environ.get("ROGII_SCRIPTS", str(PROJ / "v95")))
ART = Path(os.environ.get("ROGII_ART95", str(V95 / "artifacts_v95")))           # GBM / TCN / 錨
ART_B = Path(os.environ.get("ROGII_ART95B", str(V95 / "artifacts_v95b")))        # L5cf ckpt
PY = os.environ.get("ROGII_PY", str(PROJ / ".venv" / "Scripts" / "python.exe"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TEST_PARQUET = ART / "data" / "test.parquet"
SEQFEAT_TEST = ART_B / "seqfeat_test_cf.pkl"        # ★test の CANDFEAT seqfeat(NF=2)
L5CF_DIR = ART_B / "l5_cf_ckpt"
CONFIG = ART_B / "submit_v96_config.json"
SAMPLE = Path(os.environ.get("ROGII_DATA", str(PROJ / "rogii-wellbore-geology-prediction"))) / "sample_submission.csv"

# submit_v95 を再利用(run_gen / gbm_predict / tcn_predict は完全一致)
sys.path.insert(0, str(V95))
import submit_v95 as s95   # noqa: E402

# L5cf の学習ハイパー(train_l456_v95b の既定と一致させる)
DIM = int(os.environ.get("DIM", "64"))
SEEDS = [0, 1, 2]
N_SPLITS = 5


# =====================================================================
# L5(CANDFEAT)モデル --- train_l456_v95b.py の L5 クラスと VERBATIM
# k(候補数K)/cdim/nf は seqfeat から動的に決める
# =====================================================================
class L5(nn.Module):
    def __init__(self, k, cdim, nf, d=DIM):
        super().__init__()
        self.k = k; self.nf = nf
        self.enc = nn.Sequential(nn.Conv1d(2 + nf, d, 9, padding=4), nn.GELU(),
                                 nn.Conv1d(d, d, 9, padding=4 * 4, dilation=4), nn.GELU(),
                                 nn.Conv1d(d, d, 9, padding=4 * 16, dilation=16))
        self.emb = nn.Embedding(k, d)
        self.qry = nn.Sequential(nn.Linear(cdim + 4, d), nn.GELU(), nn.Linear(d, d))
        self.res = nn.Linear(d, 1)

    def forward(self, D, S, C, m, DX=None):
        B, T, k = D.shape
        med = D.median(2, keepdim=True).values
        x = torch.stack([D - med, torch.log(S)] +
                        ([DX[..., _f] for _f in range(DX.shape[-1])] if DX is not None else []), 2)
        x = x.permute(0, 3, 2, 1).reshape(B * k, 2 + (DX.shape[-1] if DX is not None else 0), T)
        hc = self.enc(x).reshape(B, k, -1, T).permute(0, 3, 1, 2)
        hc = hc + self.emb.weight[None, None]
        stats = torch.cat([C, med / 10.0, D.std(2, keepdim=True),
                           (D - med).abs().lt(2).float().mean(2, keepdim=True),
                           torch.zeros_like(med)], -1)
        q = self.qry(stats)
        att = (hc @ q[..., None])[..., 0] / (hc.shape[-1] ** 0.5)
        w = att.softmax(-1)
        out = (w * D).sum(-1)
        return out + 2.0 * torch.tanh(self.res(q)[..., 0])


# =====================================================================
# L5cf: test seqfeat 生成 + 15ckpt 推論 -> per-row(target 単位=残差)
# =====================================================================
def build_test_seqfeat():
    if SEQFEAT_TEST.exists() and os.environ.get("SKIP_SEQFEAT", "0") == "1":
        print(f"[L5cf] {SEQFEAT_TEST.name} 既存(SKIP_SEQFEAT=1)", flush=True)
        return
    print("\n==== L5cf: test CANDFEAT seqfeat 生成(K=20, NF_MAX=2 loglik/vs_tw) ====", flush=True)
    env = dict(os.environ)
    env["SF_IN"] = str(TEST_PARQUET)
    env["SF_OUT"] = str(SEQFEAT_TEST)
    env["SF_NO_TARGET"] = "1"          # test: target 列なし(dtrue/wgt はダミー)
    env["CANDFEAT"] = "1"
    env["NF_MAX"] = "2"                # ★勝者は 2役割(loglik/vs_tw)。cf.pkl の DX=(nb,K,2) と一致
    # ★K=20 に必須(BASE14 + nbr_gr5/self_graft delta 6ch)。順序も学習時と一致(train seqfeat_v95_cf と D/S/DX 完全一致を確認済)
    env["EXTRA_DELTA"] = ("pf_nbr_gr5_delta_1,pf_nbr_gr5_delta_2,pf_nbr_gr5_delta_3,"
                          "pf_self_graft_delta_1,pf_self_graft_delta_2,pf_self_graft_delta_3")
    env["EXTRA_STD"] = ("pf_nbr_gr5_std_1,pf_nbr_gr5_std_2,pf_nbr_gr5_std_3,"
                        "pf_self_graft_std_1,pf_self_graft_std_2,pf_self_graft_std_3")
    env["V93_INPUT"] = "fwd"           # BASE14(bpf は足さない)。cf.pkl と一致
    r = subprocess.run([PY, str(V95 / "build_seqfeat_v95b.py")], env=env)
    if r.returncode != 0:
        raise SystemExit("L5cf test seqfeat 生成に失敗")
    assert SEQFEAT_TEST.exists(), f"{SEQFEAT_TEST} が生成されなかった"


def l5cf_predict(N, ids_parquet):
    """test seqfeat + l5_cf_ckpt(15) で per-row 予測(TARGET 単位)。ids_parquet 行順に整列した長さ N 配列を返す。"""
    SF = pickle.loads(SEQFEAT_TEST.read_bytes())
    wells = list(SF.keys())
    w0 = SF[wells[0]]
    K = w0["D"].shape[1]
    CDIM = w0["C"].shape[1]
    NF = w0["DX"].shape[2] if "DX" in w0 else 0
    print(f"[L5cf] wells={len(wells)} K={K} CDIM={CDIM} NF={NF} DIM={DIM}", flush=True)
    assert NF == 2, f"NF={NF} != 2(勝者は 2役割)。seqfeat 生成の NF_MAX を確認"

    @torch.no_grad()
    def predict_well(net, w):
        f = SF[w]; T = f["D"].shape[0]
        D = torch.from_numpy(f["D"])[None].to(DEVICE)
        S = torch.from_numpy(f["S"])[None].to(DEVICE)
        C = torch.from_numpy(f["C"])[None].to(DEVICE)
        m = torch.ones((1, T), device=DEVICE)
        DX = torch.from_numpy(f["DX"])[None].to(DEVICE)
        o = net(D, S, C, m, DX)                        # (1,T) per-bin
        dh = o[0].cpu().numpy().astype(np.float64)
        return dh[f["bid"]], f["rows"]                 # per-row(bin->row), parquet 行 index

    acc = np.zeros(N, np.float64); cnt = np.zeros(N, np.float64)
    nck = 0
    for s in SEEDS:
        for k in range(N_SPLITS):
            fp = L5CF_DIR / f"l5_s{s}_f{k}_model.pt"
            if not fp.exists():
                print(f"   WARN: missing {fp.name}"); continue
            net = L5(K, CDIM, NF).to(DEVICE)
            ck = torch.load(fp, map_location=DEVICE)
            net.load_state_dict(ck["state"] if isinstance(ck, dict) and "state" in ck else ck)
            net.eval()
            pred_ck = np.full(N, np.nan)
            for w in wells:
                pr, rows = predict_well(net, w)
                pred_ck[rows] = pr
            fin = np.isfinite(pred_ck)
            acc[fin] += pred_ck[fin]; cnt[fin] += 1
            nck += 1
            del net
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
            print(f"   ckpt s{s}_f{k}: cover%={100.0*fin.mean():.1f} mean={np.nanmean(pred_ck):.3f}", flush=True)
    l5 = np.full(N, np.nan)
    ok = cnt > 0; l5[ok] = acc[ok] / cnt[ok]
    print(f"[L5cf] {nck}ckpt 平均 cover%={100.0*ok.mean():.1f} pred[mean/min/max]="
          f"{np.nanmean(l5):.3f}/{np.nanmin(l5):.3f}/{np.nanmax(l5):.3f}", flush=True)
    return l5


# =====================================================================
# main
# =====================================================================
def main():
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    legs = cfg["blend_legs"]; wts = cfg["blend_weights"]
    assert legs == ["gbm", "tcn", "l5cf"], f"想定外の blend_legs: {legs}"
    w_gbm, w_tcn, w_l5cf = float(wts[0]), float(wts[1]), float(wts[2])
    print(f"[cfg] blend weights gbm={w_gbm:.4f} tcn={w_tcn:.4f} l5cf={w_l5cf:.4f} "
          f"(sum={w_gbm+w_tcn+w_l5cf:.4f}) 目標CV={cfg.get('cv_rmse')}", flush=True)

    # 1) test.parquet 生成(submit_v95 の run_gen: GRフリー錨 -> create SPLIT=test)
    if os.environ.get("SKIP_GEN", "0") != "1":
        s95.run_gen()
    else:
        print("[SKIP_GEN=1] test.parquet 生成をスキップ(既存を使用)", flush=True)
    assert TEST_PARQUET.exists(), f"{TEST_PARQUET} が無い"

    # id/lk の基準は parquet
    meta = pq.read_table(TEST_PARQUET, columns=["id", "last_known_tvt"])
    ids = meta.column("id").to_pandas().to_numpy()
    lk = meta.column("last_known_tvt").to_numpy(zero_copy_only=False).astype(np.float64)
    N = len(ids)

    # 2) GBM(submit_v95 と完全一致)
    gbm_pred = s95.gbm_predict()

    # 3) TCN(submit_v95 と完全一致)。id 並びは parquet 由来で一致
    nn_pred, well_n, ids_n, lk_n = s95.tcn_predict()
    assert np.array_equal(ids_n, ids), "TCN と GBM の id 並びが不一致"

    # 4) L5cf(CANDFEAT)
    build_test_seqfeat()
    l5_pred = l5cf_predict(N, ids)

    # 5) 3脚 blend(固定重み)。L5cf が欠損の行は gbm/tcn の 2脚に再正規化して埋める
    l5f = np.isfinite(l5_pred)
    pred = np.empty(N, np.float64)
    pred[l5f] = w_gbm * gbm_pred[l5f] + w_tcn * nn_pred[l5f] + w_l5cf * l5_pred[l5f]
    if (~l5f).any():
        sw = w_gbm + w_tcn
        pred[~l5f] = (w_gbm / sw) * gbm_pred[~l5f] + (w_tcn / sw) * nn_pred[~l5f]
        print(f"[blend] L5cf 欠損 {int((~l5f).sum())} 行は gbm/tcn 2脚で補完", flush=True)

    tvt = lk + pred
    sub = pd.DataFrame({"id": ids, "tvt": tvt.astype(np.float64)})
    s95._write_and_report(sub, ART_B / "submission_v96.csv",
                          f"3leg gbm={w_gbm:.3f}/tcn={w_tcn:.3f}/l5cf={w_l5cf:.3f}")


if __name__ == "__main__":
    main()
