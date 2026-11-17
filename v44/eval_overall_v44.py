# -*- coding: utf-8 -*-
"""
v44 全eval行 overall 評価(marching推論で元の形に戻す)
=====================================================================================
GeoSteerNet は近傍固定窓(未来H_F=384bin≈768行, typewell±16ft)しか予測できないので、
窓を前進(march)させて全eval帯をカバー: 予測TVTを疑似既知にして次窓を再アンカー。
→ 各evalビン予測 → 生eval行に展開 → overall pooled TVT RMSE(コンペ指標)+ 硬い核。

実行:
  base   : python v44/eval_overall_v44.py            (model_base.pt, in_ch5)
  struct : V44_STRUCT=1 python v44/eval_overall_v44.py (model_struct.pt, in_ch6)
"""
import os, sys, glob, time, math
from pathlib import Path
import numpy as np, pandas as pd, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.config import CFG, _DATA_ROOT
from src import dataset as D
from src.dataset import (resample_typewell_by_step, resample_horizontal_by_step,
                         get_crop_index_and_pad_1d, get_meta_df, KAGGLE_DIR,
                         T_H, T_F, T_TOTAL, H_H, H_F, H_TOTAL, COMPRESSION, USE_STRUCT)
from src.model_sdf_mtp import GeoSteerMTPNet
from sklearn.model_selection import GroupKFold

DEV = "cuda" if torch.cuda.is_available() else "cpu"
IN_CH = 6 if USE_STRUCT else 5
TAG = "struct" if USE_STRUCT else "base"
CKPT = Path(__file__).resolve().parent / "artifacts_v44" / f"model_{TAG}.pt"
NW = int(os.environ.get("N_WELLS", "0"))
print(f"[eval] TAG={TAG} in_ch={IN_CH} ckpt={CKPT.name}")

model = GeoSteerMTPNet(in_ch=IN_CH, base_ch=CFG.SDF_MTP_BASE_CH, K=CFG.SDF_MTP_K).to(DEV)
model.load_state_dict(torch.load(CKPT, map_location=DEV)); model.eval()


def _cp(arr, center, hist, fut):
    i0, i1, pl, pr = get_crop_index_and_pad_1d(len(arr), center, hist, fut)
    return np.pad(arr[i0:i1], (pl, pr))


@torch.no_grad()
def march_predict(wid, split="train"):
    """全evalビンの予測TVT(bin単位)を返す + eval生行の真値/対応。"""
    meta = get_meta_df(); m = meta[meta["sample_id"] == wid].iloc[0]
    t = pd.read_csv(f"{KAGGLE_DIR}/{split}/{wid}__typewell.csv")
    t_tvt, t_gr = resample_typewell_by_step(t, step=m["t_step"], target_step=0.5)
    h = pd.read_csv(f"{KAGGLE_DIR}/{split}/{wid}__horizontal_well.csv")
    h_tvt0, h_tvt1, h_gr0, h_gr1 = resample_horizontal_by_step(h, target_step=COMPRESSION, offset=0)
    K = len(h_tvt0); E = len(h_tvt1)
    if E == 0: return None
    tvt_bins = np.concatenate([h_tvt0, h_tvt1]).astype(float)     # 真値bin(参考)
    gr_bins  = np.concatenate([h_gr0, h_gr1]).astype(float)
    Nb = K + E
    pseudo = tvt_bins.copy(); pseudo[K:] = np.nan                 # 既知=真, 未来=未定
    pred_bins = np.full(E, np.nan)
    # struct bins(future用)
    st_bins = None
    if USE_STRUCT:
        sp = D._get_struct().get(wid)
        if sp is not None:
            st_bins = D._bin_after_like(np.asarray(sp["tvt"], float))  # eval bin整列(offset0)

    front = K
    while front < Nb:
        # ★アンカー = struct(大域安定)。自己回帰の発散を断つ = 物理priorが全eval拡張を可能にする鍵
        if st_bins is not None and (front - K) < len(st_bins):
            anchor = float(st_bins[min(front - K, len(st_bins) - 1)])
        else:
            anchor = pseudo[front - 1]
        if not np.isfinite(anchor): anchor = h_tvt0[-1]
        aidx = int(np.abs(t_tvt - anchor).argmin()); center = aidx + 1
        t_seg_tvt = _cp(t_tvt, center, T_H, T_F); t_seg_gr = _cp(t_gr, center, T_H, T_F)
        t_seg_mask = _cp(np.ones(len(t_tvt)), center, T_H, T_F)
        # h窓: 履歴=front前 H_H, 未来=front以降 H_F
        i0, i1, pl, pr = get_crop_index_and_pad_1d(Nb, front, H_H, H_F)
        h_seg_gr = np.pad(gr_bins[i0:i1], (pl, pr))
        seg_tvt_pseudo = np.pad(pseudo[i0:i1], (pl, pr))          # 履歴の既知/予測TVT
        valid = np.pad(np.ones(i1 - i0), (pl, pr))
        hist_mask = valid.copy(); hist_mask[H_H:] = 0.0
        hist_mask = hist_mask * np.isfinite(seg_tvt_pseudo)
        h_tvt_hist = np.nan_to_num(seg_tvt_pseudo) * hist_mask
        h_mask = valid.copy()
        b = {
            "t_gr": t_seg_gr, "h_gr": h_seg_gr, "t_tvt": t_seg_tvt,
            "h_tvt_history": h_tvt_hist, "h_mask": h_mask,
            "h_history_mask": hist_mask, "t_mask": t_seg_mask,
        }
        if USE_STRUCT:
            tvt_prior = h_tvt_hist.copy()
            if st_bins is not None:
                fb0 = front - K                                   # 未来先頭のeval-bin index
                fut = st_bins[fb0:fb0 + H_F]
                fut = np.pad(fut, (0, H_F - len(fut)), mode="edge") if len(fut) < H_F else fut[:H_F]
                tvt_prior[H_H:] = fut
            b["sdf_prior"] = np.clip((tvt_prior[None, :] - t_seg_tvt[:, None]) / 40.0, -3, 3)
        bt = {k: torch.from_numpy(np.asarray(v, np.float32))[None].to(DEV) for k, v in b.items() if k != "sdf_prior"}
        if USE_STRUCT: bt["sdf_prior"] = torch.from_numpy(b["sdf_prior"].astype(np.float32))[None][None].to(DEV)
        out = model(bt)
        row = out["pred_rows"][0, 0].long().clamp(0, len(t_seg_tvt) - 1).cpu().numpy()
        tvt_win = t_seg_tvt[row]                                  # 未来+履歴の予測TVT(窓内)
        # 未来 H_F bins を格納(front.. )
        for j in range(H_F):
            gb = front + j
            if gb >= Nb: break
            pred_bins[gb - K] = tvt_win[H_H + j]
            pseudo[gb] = tvt_win[H_H + j]                         # 次窓の疑似既知
        front += H_F
    # bin→生eval行(row i → bin i//COMP)
    ps = int(np.flatnonzero(h["TVT_input"].notna().values)[-1])
    true_raw = h["TVT"].to_numpy(float)[ps + 1:]
    idx = np.clip(np.arange(len(true_raw)) // COMPRESSION, 0, E - 1)
    pred_raw = pred_bins[idx]
    return pred_raw, true_raw


meta = get_meta_df()
wells = [w for w in meta.loc[meta.get("split", "train") == "train", "sample_id"].tolist()
         if (Path(_DATA_ROOT) / "train" / f"{w}__horizontal_well.csv").exists()]
if NW: wells = wells[:NW]
_, va_idx = list(GroupKFold(n_splits=5).split(wells, groups=wells))[0]
va_w = [wells[i] for i in va_idx]
try:
    hd = pd.read_csv(_DATA_ROOT.parent / "v32" / "artifacts_v32" / "hardcore_diag.csv") if False else \
         pd.read_csv(Path(r"C:\Users\kosaka256\Documents\rogii_claude\v32\artifacts_v32\hardcore_diag.csv"))
    HARD = set(hd[hd["hard"] == True].iloc[:, 0].astype(str))
except Exception:
    HARD = set()

print(f"[eval] val wells {len(va_w)} で marching 全eval推論 ...")
t0 = time.time(); ea, eh = [], []; near = []
for k, w in enumerate(va_w):
    r = march_predict(w)
    if r is None: continue
    pred, true = r; e = pred - true; e = e[np.isfinite(e)]
    ea.append(e)
    if w in HARD: eh.append(e)
    near.append((pred[:768] - true[:768]))                       # 近傍768行(訓練窓相当)
    if (k + 1) % 20 == 0: print(f"  {k+1}/{len(va_w)} ({time.time()-t0:.0f}s)", flush=True)
A = np.concatenate(ea); Hh = np.concatenate(eh) if eh else np.array([np.nan]); Nn = np.concatenate([x[np.isfinite(x)] for x in near])
print(f"\n=== v44 {TAG} 全eval overall(生行プール, {len(va_w)}val坑井)===")
print(f"  overall(全eval行)  = {np.sqrt(np.mean(A**2)):.4f}")
print(f"  硬い核({len(eh)}坑井) = {np.sqrt(np.nanmean(Hh**2)):.4f}")
print(f"  参考: 近傍768行のみ = {np.sqrt(np.mean(Nn**2)):.4f}(訓練窓RMSEに相当)")
print(f"  ({time.time()-t0:.0f}s)  比較: pf_1≈9.12 / 錨PF≈8.13 / cons≈7.1")
