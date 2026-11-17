# -*- coding: utf-8 -*-
"""
v44 ローカル単GPU 学習/評価ドライバ(GeoSteerNet SDF-CNN)+ struct物理prior ablation
=====================================================================================
- baseline(in_ch=5) vs +struct(in_ch=6, ch5=struct-SDF)を V44_STRUCT で切替。
- engine.py(DDP)は使わず簡易ループ。評価= pred_rows(top1)→ t_tvt[row] で TVT に戻し、
  未来帯(eval)で pooled TVT RMSE(≈コンペ指標。resampled binなので近似)。
実行:
  baseline : FAST_DEBUG=1 python v44/train_local_v44.py
  +struct  : FAST_DEBUG=1 V44_STRUCT=1 python v44/train_local_v44.py
"""
import os, sys, glob, time, math
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.config import CFG, _DATA_ROOT
from src.dataset import GeoSteerDataset, GeoSteerMarchDataset, USE_STRUCT, H_H, get_meta_df
from src.model_sdf_mtp import GeoSteerMTPNet
from sklearn.model_selection import GroupKFold

DEV = "cuda" if torch.cuda.is_available() else "cpu"
IN_CH = 6 if USE_STRUCT else 5
EPOCHS = int(os.environ.get("EPOCHS", "1" if CFG.FAST_DEBUG else "40"))
NW = int(os.environ.get("N_WELLS", "40" if CFG.FAST_DEBUG else "0"))
BATCH = int(os.environ.get("BATCH", "8"))
print(f"[v44] STRUCT={USE_STRUCT} in_ch={IN_CH} device={DEV} epochs={EPOCHS} K={CFG.SDF_MTP_K}")

meta = get_meta_df()
wells = [w for w in meta.loc[meta.get("split", "train") == "train", "sample_id"].tolist()
         if (Path(_DATA_ROOT) / "train" / f"{w}__horizontal_well.csv").exists()]
if NW: wells = wells[:NW]
folds = list(GroupKFold(n_splits=5).split(wells, groups=wells))
tr_idx, va_idx = folds[0]
tr_w = [wells[i] for i in tr_idx]; va_w = [wells[i] for i in va_idx]
print(f"[v44] wells train={len(tr_w)} val={len(va_w)}")

if USE_STRUCT:
    # ★struct錨・多位置(推論 marching と同一レジーム)
    tr_ds = GeoSteerMarchDataset(tr_w, split="train", n_pos=int(os.environ.get("N_POS", "6")))
    va_ds = GeoSteerMarchDataset(va_w, split="train", n_pos=2)
    print(f"[v44] march dataset(struct錨・多位置) train窓={len(tr_ds)} val窓={len(va_ds)}")
else:
    tr_ds = GeoSteerDataset(tr_w, split="train", n_offsets=1)
    va_ds = GeoSteerDataset(va_w, split="train", n_offsets=1)
tr_ld = DataLoader(tr_ds, batch_size=BATCH, shuffle=True, num_workers=0, drop_last=True)
va_ld = DataLoader(va_ds, batch_size=BATCH, shuffle=False, num_workers=0)

model = GeoSteerMTPNet(in_ch=IN_CH, base_ch=CFG.SDF_MTP_BASE_CH, K=CFG.SDF_MTP_K).to(DEV)
opt = torch.optim.AdamW(model.parameters(), lr=CFG.LEARNING_RATE, weight_decay=CFG.WEIGHT_DECAY)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS * max(1, len(tr_ld)), eta_min=CFG.MIN_LR)


def to_dev(b): return {k: v.to(DEV) for k, v in b.items()}


@torch.no_grad()
def evaluate():
    model.eval(); se = 0.0; n = 0
    for b in va_ld:
        b = to_dev(b); out = model(b)
        pr = out["pred_rows"][:, 0, :]                      # (B,H) top1 row
        t_tvt = b["t_tvt"]; h_tvt = b["h_tvt"]; hm = b["h_mask"]
        row = pr.long().clamp(0, t_tvt.shape[1] - 1)
        tvt_pred = torch.gather(t_tvt, 1, row)              # (B,H)
        m = hm.bool().clone(); m[:, :H_H] = False           # 未来帯のみ
        e = (tvt_pred - h_tvt)[m]
        se += float((e ** 2).sum()); n += int(m.sum())
    return math.sqrt(se / max(n, 1))


tag = "struct" if USE_STRUCT else "base"
ARTDIR = Path(__file__).resolve().parent / "artifacts_v44"; ARTDIR.mkdir(exist_ok=True)  # ←out(model出力)と衝突回避
SAVE_EVERY = int(os.environ.get("SAVE_EVERY", "0"))   # >0 で N ep ごとに ep別checkpoint保存

t0 = time.time(); best = (1e18, -1)
for ep in range(EPOCHS):
    model.train(); tl = 0.0; nb = 0
    for b in tr_ld:
        b = to_dev(b); out = model(b); loss = out["loss"]
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.GRAD_CLIP)
        opt.step(); sched.step(); tl += float(loss); nb += 1
    rmse = evaluate()
    tagbest = ""
    if rmse < best[0]:
        best = (rmse, ep)
        torch.save(model.state_dict(), ARTDIR / f"model_{tag}.pt")   # ★best-val を eval_overall が読む名で保存
        tagbest = " ★best"
    if SAVE_EVERY and (ep % SAVE_EVERY == 0 or ep == EPOCHS - 1):
        torch.save(model.state_dict(), ARTDIR / f"model_{tag}_ep{ep}.pt")
    print(f"[v44] ep{ep} loss={tl/max(nb,1):.4f}  val TVT-RMSE(未来帯)={rmse:.4f}  ({time.time()-t0:.0f}s){tagbest}", flush=True)

torch.save(model.state_dict(), ARTDIR / f"model_{tag}_last.pt")       # 最終epも別名で残す
print(f"[v44] best val TVT-RMSE={best[0]:.4f} @ep{best[1]} → model_{tag}.pt(best) / model_{tag}_last.pt(final)")
print("注: val は teacher-forced 窓指標=marching overall の代理。厳密best確認は SAVE_EVERY で ep別保存し eval_overall で再測。")
