"""
v16 NN-PF: パーティクルフィルタ全置換のための共通モジュール.

役割（A/B 両ノートブックで共有）:
  - データ読み込み → 坑井ごとのテンソル化（GR補間, s=TVT+Z, 幾何ベースライン dip 外挿）
  - PyTorch Dataset / collate（可変長パディング, 学習はランダムクロップで augment）
  - GroupKFold(by well) の CV ループ run_cv（モデルは forward(batch)->tvt_pred[B,H] を返すだけ）
  - overall RMSE（評価ゾーン全行プール; Kaggle 指標と一致）

設計の要点（PF を関数として置換するための定式化）:
  - 状態 s = TVT + Z（構造絶対深度, MD に対して滑らか）
  - 幾何ベースライン s_base(MD) = 既知ゾーン末尾の dip を線形外挿
  - モデルは残差 r = s_true - s_base を学習（動的レンジが小さく学習が安定, アンカーが自然に入る）
  - 最終予測 TVT = s_base + r - Z

注意:
  - 学習・実験の「実行」はユーザーが行う（CLAUDE.md 方針）。本ファイルは部品の定義のみ。
  - test には ANCC〜BUDA / TVT 列は無い。ここで使うのは MD,X,Y,Z,GR,TVT_input と typewell のみ（test 互換）。
"""

import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupKFold

# --------------------------------------------------------------------------------------
# 設定
# --------------------------------------------------------------------------------------
DATA_DIR = r"c:\Users\kosaka256\Documents\rogii_claude\rogii-wellbore-geology-prediction"

TARGET_SCALE = 30.0   # 残差 r のスケール（モデルは r/TARGET_SCALE を出力する想定 → 勾配安定）
DIP_FIT_N    = 300    # ベースライン dip フィットに使う既知ゾーン末尾の点数
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------------------
# データ読み込み・坑井テンソル化
# --------------------------------------------------------------------------------------
def list_well_ids(split="train"):
    base = os.path.join(DATA_DIR, split)
    fs = glob.glob(os.path.join(base, "*__horizontal_well.csv"))
    return sorted(os.path.basename(f).split("__")[0] for f in fs)


def load_well_raw(wid, split="train"):
    base = os.path.join(DATA_DIR, split)
    hw = pd.read_csv(os.path.join(base, f"{wid}__horizontal_well.csv"))
    tw = pd.read_csv(os.path.join(base, f"{wid}__typewell.csv"))
    return hw, tw


def _fit_line(x, y):
    """y ~ a*x + b の OLS。点が足りなければ a=0。"""
    if x.size < 2:
        return 0.0, float(y.mean()) if y.size else 0.0
    A = np.vstack([x, np.ones_like(x)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(coef[0]), float(coef[1])


def _standardize(a):
    a = np.asarray(a, dtype=np.float64)
    m = np.nanmean(a)
    s = np.nanstd(a)
    s = s if s > 1e-6 else 1.0
    return ((a - m) / s).astype(np.float32), float(m), float(s)


def build_well(wid, split="train", fit_n=DIP_FIT_N):
    """1坑井 → 学習に必要な numpy 配列をまとめた dict。"""
    hw, tw = load_well_raw(wid, split)

    md  = hw["MD"].to_numpy(np.float64)
    z   = hw["Z"].to_numpy(np.float64)
    gr  = hw["GR"].to_numpy(np.float64)
    tin = hw["TVT_input"].to_numpy(np.float64)
    # GR 欠損は前後補間（v16 の知見: NaN 多発坑井で致命的）
    gr  = pd.Series(gr).interpolate(limit_direction="both").to_numpy()
    if not np.isfinite(gr).all():
        gr = np.nan_to_num(gr, nan=float(np.nanmean(gr)) if np.isfinite(gr).any() else 0.0)

    has_true = "TVT" in hw.columns
    tvt_true = hw["TVT"].to_numpy(np.float64) if has_true else np.full_like(md, np.nan)

    known = np.isfinite(tin)
    evalm = ~known

    # typewell（TVT 昇順, GR 補間）
    tw = tw.sort_values("TVT")
    tw_tvt = tw["TVT"].to_numpy(np.float64)
    tw_gr  = tw["GR"].to_numpy(np.float64)
    tw_gr  = pd.Series(tw_gr).interpolate(limit_direction="both").to_numpy()
    if not np.isfinite(tw_gr).all():
        tw_gr = np.nan_to_num(tw_gr, nan=float(np.nanmean(tw_gr)) if np.isfinite(tw_gr).any() else 0.0)

    # 構造量 s=TVT+Z と幾何ベースライン（既知ゾーン末尾 dip の線形外挿）
    s_known = tin + z
    kmd, ks = md[known], s_known[known]
    if kmd.size >= 2:
        sel = slice(max(0, kmd.size - fit_n), kmd.size)
        a, b = _fit_line(kmd[sel], ks[sel])
    else:
        a, b = 0.0, float(ks.mean()) if ks.size else 0.0
    s_base = a * md + b

    # アンカー（最後の既知点）
    anc = int(np.where(known)[0][-1]) if known.any() else 0

    # 正規化
    gr_n, _, _ = _standardize(gr)
    tw_gr_n, _, _ = _standardize(tw_gr)

    return dict(
        wid=wid,
        md=md.astype(np.float32), z=z.astype(np.float32),
        gr=gr.astype(np.float32), gr_n=gr_n,
        tin=tin.astype(np.float32), tvt_true=tvt_true.astype(np.float32),
        known=known, evalm=evalm,
        tw_tvt=tw_tvt.astype(np.float32), tw_gr=tw_gr.astype(np.float32), tw_gr_n=tw_gr_n,
        s_base=s_base.astype(np.float32), dip=float(a),
        md_anchor=float(md[anc]), z_anchor=float(z[anc]), s_anchor=float(s_base[anc]),
        has_true=has_true,
    )


def build_all(split="train", well_ids=None, verbose=True):
    """全坑井を前処理してメモリ上の list で返す（合計は数十MB 程度なので RAM 安全）。"""
    if well_ids is None:
        well_ids = list_well_ids(split)
    out = []
    for i, wid in enumerate(well_ids):
        try:
            out.append(build_well(wid, split))
        except Exception as e:
            if verbose:
                print(f"[build_all] skip {wid}: {e}")
        if verbose and (i + 1) % 100 == 0:
            print(f"  built {i + 1}/{len(well_ids)}")
    if verbose:
        print(f"[build_all] {len(out)} wells ready (split={split})")
    return out


def target_r_std(wells):
    """学習データから残差 r=s_true-s_base の std を見る（TARGET_SCALE の妥当性確認用）。"""
    rs = []
    for w in wells:
        if not w["has_true"]:
            continue
        m = w["evalm"] & np.isfinite(w["tvt_true"])
        if m.any():
            s_true = w["tvt_true"][m] + w["z"][m]
            rs.append(s_true - w["s_base"][m])
    rs = np.concatenate(rs) if rs else np.array([0.0])
    return float(np.std(rs)), float(np.mean(np.abs(rs)))


# --------------------------------------------------------------------------------------
# Dataset / collate
# --------------------------------------------------------------------------------------
class WellDataset(Dataset):
    """
    mode="train": 評価ゾーンを含むランダムクロップ（augment）。
    mode="eval" : 坑井全長をそのまま（batch_size=1 で検証/推論）。
    """
    def __init__(self, wells, mode="train", crop=768, margin=128, seed=0):
        self.wells = wells
        self.mode = mode
        self.crop = crop
        self.margin = margin
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.wells)

    def _crop_window(self, w):
        L = w["md"].shape[0]
        if self.mode != "train" or L <= self.crop:
            return 0, L
        ev = np.where(w["evalm"])[0]
        if ev.size == 0:
            start = int(self.rng.integers(0, max(1, L - self.crop)))
            return start, start + self.crop
        ev0, ev1 = int(ev[0]), int(ev[-1])
        lo = max(0, ev0 - self.margin)
        hi = max(lo, min(L - self.crop, ev1))         # 窓内に評価ゾーンが入るように
        start = int(self.rng.integers(lo, hi + 1)) if hi > lo else lo
        return start, min(L, start + self.crop)

    def __getitem__(self, i):
        w = self.wells[i]
        s, e = self._crop_window(w)
        sl = slice(s, e)
        md = w["md"][sl]; z = w["z"][sl]
        item = dict(
            wid=w["wid"], idx=i,
            gr_n=torch.from_numpy(w["gr_n"][sl]),
            z_rel=torch.from_numpy((z - w["z_anchor"]).astype(np.float32)),
            md_rel=torch.from_numpy(((md - w["md_anchor"]) / 1000.0).astype(np.float32)),
            sbase=torch.from_numpy(w["s_base"][sl]),
            sbase_rel=torch.from_numpy((w["s_base"][sl] - w["s_anchor"]).astype(np.float32)),
            z=torch.from_numpy(z),
            known=torch.from_numpy(w["known"][sl].astype(np.float32)),
            evalm=torch.from_numpy(w["evalm"][sl]),
            tvt_true=torch.from_numpy(w["tvt_true"][sl]),
            tin=torch.from_numpy(w["tin"][sl]),
            tw_gr_n=torch.from_numpy(w["tw_gr_n"]),
            tw_tvt=torch.from_numpy(w["tw_tvt"]),
            row_offset=s,
        )
        return item


def collate(batch):
    B = len(batch)
    H = max(b["gr_n"].shape[0] for b in batch)
    T = max(b["tw_gr_n"].shape[0] for b in batch)

    def padH(key, fill=0.0, dtype=torch.float32):
        out = torch.full((B, H), fill, dtype=dtype)
        for i, b in enumerate(batch):
            h = b[key].shape[0]
            out[i, :h] = b[key].to(dtype)
        return out

    out = dict(
        gr_n=padH("gr_n"), z_rel=padH("z_rel"), md_rel=padH("md_rel"),
        sbase=padH("sbase"), sbase_rel=padH("sbase_rel"), z=padH("z"),
        known=padH("known"),
        evalm=padH("evalm", dtype=torch.bool),
        tvt_true=padH("tvt_true", fill=float("nan")),
        tin=padH("tin", fill=float("nan")),
    )
    valid = torch.zeros(B, H, dtype=torch.bool)
    for i, b in enumerate(batch):
        valid[i, : b["gr_n"].shape[0]] = True
    out["valid"] = valid

    tw_gr = torch.zeros(B, T, dtype=torch.float32)
    tw_tvt = torch.full((B, T), 1e9, dtype=torch.float32)   # パディングは +inf 相当（searchsorted で末尾へ）
    tw_valid = torch.zeros(B, T, dtype=torch.bool)
    for i, b in enumerate(batch):
        t = b["tw_gr_n"].shape[0]
        tw_gr[i, :t] = b["tw_gr_n"]
        tw_tvt[i, :t] = b["tw_tvt"]
        tw_valid[i, :t] = True
    out["tw_gr_n"] = tw_gr
    out["tw_tvt"] = tw_tvt
    out["tw_valid"] = tw_valid

    out["wid"] = [b["wid"] for b in batch]
    out["idx"] = [b["idx"] for b in batch]
    out["row_offset"] = [b["row_offset"] for b in batch]
    return out


def to_device(batch, device=DEVICE):
    for k, v in batch.items():
        if torch.is_tensor(v):
            batch[k] = v.to(device)
    return batch


# --------------------------------------------------------------------------------------
# 損失・指標
# --------------------------------------------------------------------------------------
def tvt_from_r(r_raw, batch):
    """モデル出力 r_raw[B,H] → TVT 予測。 TVT = s_base + r - Z, r = r_raw*TARGET_SCALE。"""
    return batch["sbase"] + r_raw * TARGET_SCALE - batch["z"]


def masked_loss(tvt_pred, batch, huber_delta=10.0):
    """評価ゾーンのみで Huber 損失（外れ坑井に頑健）。"""
    m = batch["evalm"] & batch["valid"] & torch.isfinite(batch["tvt_true"])
    if m.sum() == 0:
        return tvt_pred.sum() * 0.0
    err = (tvt_pred - torch.nan_to_num(batch["tvt_true"]))[m]
    return torch.nn.functional.huber_loss(err, torch.zeros_like(err), delta=huber_delta)


def overall_rmse(pred_list, true_list):
    """全坑井の評価ゾーンを連結して overall RMSE（Kaggle 指標）。"""
    e = np.concatenate([p - t for p, t in zip(pred_list, true_list)])
    e = e[np.isfinite(e)]
    return float(np.sqrt(np.mean(e ** 2)))


# --------------------------------------------------------------------------------------
# CV ループ（GroupKFold by well）
# --------------------------------------------------------------------------------------
def run_cv(wells, model_factory, name="model", n_splits=5, epochs=20, lr=1e-3,
           batch_size=8, crop=768, weight_decay=1e-4, grad_clip=2.0,
           device=DEVICE, seed=42, verbose=True, return_oof=True):
    """
    model_factory: () -> nn.Module。forward(batch)->r_raw[B,H] を実装すること。
    各 fold で学習 → valid 坑井（全長）で予測 → overall RMSE をプールして報告。
    """
    groups = np.array([w["wid"] for w in wells])
    gkf = GroupKFold(n_splits=n_splits)
    oof = {}   # wid -> (pred_eval, true_eval)
    fold_scores = []

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(np.arange(len(wells)), groups=groups)):
        tr_wells = [wells[i] for i in tr_idx]
        va_wells = [wells[i] for i in va_idx]

        tr_ds = WellDataset(tr_wells, mode="train", crop=crop, seed=seed + fold)
        tr_dl = DataLoader(tr_ds, batch_size=batch_size, shuffle=True,
                           collate_fn=collate, drop_last=False)
        va_ds = WellDataset(va_wells, mode="eval")
        va_dl = DataLoader(va_ds, batch_size=1, shuffle=False, collate_fn=collate)

        torch.manual_seed(seed + fold)
        model = model_factory().to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

        best_rmse, best_state = float("inf"), None
        for ep in range(epochs):
            model.train()
            tot, nb = 0.0, 0
            for batch in tr_dl:
                batch = to_device(batch, device)
                r = model(batch)
                pred = tvt_from_r(r, batch)
                loss = masked_loss(pred, batch)
                opt.zero_grad()
                loss.backward()
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                opt.step()
                tot += float(loss); nb += 1
            sched.step()

            rmse = _evaluate(model, va_dl, device)
            if rmse < best_rmse:
                best_rmse = rmse
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if verbose:
                print(f"[{name}] fold{fold} ep{ep:02d} train_loss={tot/max(nb,1):.4f} val_overall_rmse={rmse:.4f}")

        if best_state is not None:
            model.load_state_dict(best_state)
        # OOF 収集
        _collect_oof(model, va_dl, va_wells, oof, device)
        fold_scores.append(best_rmse)
        if verbose:
            print(f"[{name}] fold{fold} best val_overall_rmse={best_rmse:.4f}")

    # 全 OOF をプールして overall RMSE
    preds = [oof[w["wid"]][0] for w in wells if w["wid"] in oof]
    trues = [oof[w["wid"]][1] for w in wells if w["wid"] in oof]
    cv = overall_rmse(preds, trues)
    if verbose:
        print(f"[{name}] === CV overall RMSE = {cv:.4f}  (fold best mean {np.mean(fold_scores):.4f}) ===")
    return (cv, oof) if return_oof else cv


@torch.no_grad()
def _evaluate(model, va_dl, device):
    model.eval()
    preds, trues = [], []
    for batch in va_dl:
        batch = to_device(batch, device)
        r = model(batch)
        pred = tvt_from_r(r, batch)
        m = batch["evalm"] & batch["valid"] & torch.isfinite(batch["tvt_true"])
        if m.sum() == 0:
            continue
        preds.append(pred[m].cpu().numpy())
        trues.append(batch["tvt_true"][m].cpu().numpy())
    if not preds:
        return float("inf")
    return overall_rmse(preds, trues)


@torch.no_grad()
def _collect_oof(model, va_dl, va_wells, oof, device):
    model.eval()
    for batch in va_dl:
        wid = batch["wid"][0]
        batch = to_device(batch, device)
        r = model(batch)
        pred = tvt_from_r(r, batch)
        m = (batch["evalm"] & batch["valid"] & torch.isfinite(batch["tvt_true"]))[0]
        if m.sum() == 0:
            continue
        oof[wid] = (pred[0][m].cpu().numpy(), batch["tvt_true"][0][m].cpu().numpy())


# --------------------------------------------------------------------------------------
# 共通の小さな部品（A/B で再利用）
# --------------------------------------------------------------------------------------
class GRConv(torch.nn.Module):
    """GR 系列（窓）→ マルチスケール埋め込み。点でなく窓で照合するため（v14 の教訓）。"""
    def __init__(self, out_ch=32, k=7):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv1d(1, out_ch, k, padding=k // 2), torch.nn.GELU(),
            torch.nn.Conv1d(out_ch, out_ch, k, padding=k // 2), torch.nn.GELU(),
            torch.nn.Conv1d(out_ch, out_ch, k, padding=k // 2),
        )

    def forward(self, gr):           # gr [B,L] -> [B,L,C]
        h = self.net(gr.unsqueeze(1))
        return h.transpose(1, 2)


def batched_interp(xp, fp, x):
    """
    バッチ線形補間。 xp:[B,T] 昇順(パディングは大きい値), fp:[B,C,T], x:[B,Q] -> [B,C,Q]
    Model B で typewell 特徴を baseline±offset の TVT に合わせてサンプルするのに使う。
    """
    B, T = xp.shape
    C = fp.shape[1]
    Q = x.shape[1]
    idx = torch.searchsorted(xp, x).clamp(1, T - 1)          # [B,Q]
    x0 = torch.gather(xp, 1, idx - 1)
    x1 = torch.gather(xp, 1, idx)
    w = ((x - x0) / (x1 - x0).clamp_min(1e-6)).clamp(0, 1)   # [B,Q]
    idx_e = idx.unsqueeze(1).expand(B, C, Q)
    f0 = torch.gather(fp, 2, (idx_e - 1))
    f1 = torch.gather(fp, 2, idx_e)
    return f0 + (f1 - f0) * w.unsqueeze(1)
