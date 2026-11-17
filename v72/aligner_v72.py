# -*- coding: utf-8 -*-
"""v72 クロスアテンション・アライナー(PF/tabular/TCN非依存の新定式化)。
   well の生GR(MD) を typewell の生GR(TVT) に、学習クロスアテンションで localize → TVT。
   ・入力: well生GR+geometry+既知prefix / typewell生GR(TVTグリッド=出力位置軸)。struct/train限定列は不使用。
   ・出力: 各well行の typewell位置への分布 a(m,t) → TVT=Σa·TVT_grid。二峰=枝, argmax=commit。
   ・rateを積分しない全域localize=ドリフト構造回避(docs/v72.md)。
   スモーク: 数坑井でforward/backward健全性・prefix較正・eval整合を確認(本学習はユーザー)。
   使い方: python aligner_v72.py --smoke [--wells 6] [--steps 60]"""
import io, sys, argparse, glob, os
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

DATA = Path(r"C:\Users\kosaka256\Documents\rogii_claude\rogii-wellbore-geology-prediction\train")
DEV = "cuda" if torch.cuda.is_available() else "cpu"   # smoke() 内で --cpu 指定時に上書き


# ================= データ =================
def robust_norm(x):
    x = np.asarray(x, np.float64)
    med = np.nanmedian(x); iqr = np.nanpercentile(x, 75) - np.nanpercentile(x, 25)
    return (x - med) / (iqr + 1e-6)


def load_well(wid, stride_w=4, stride_t=2):
    hw = pd.read_csv(DATA / f"{wid}__horizontal_well.csv")
    tw = pd.read_csv(DATA / f"{wid}__typewell.csv")
    # typewell: TVT グリッド(出力位置軸) + 生GR
    tw = tw.sort_values("TVT").iloc[::stride_t].reset_index(drop=True)
    tvt_grid = tw["TVT"].to_numpy(np.float64)
    gr_t = robust_norm(tw["GR"].to_numpy())
    ref = tvt_grid.min(); span = max(tvt_grid.max() - tvt_grid.min(), 1.0)
    tgrid_n = (tvt_grid - ref) / span                     # 0..1(正規化位置)
    # well: 生GR + geometry + prefix
    hw = hw.iloc[::stride_w].reset_index(drop=True)
    MD = hw["MD"].to_numpy(np.float64); GR = hw["GR"].to_numpy(np.float64)
    Z = hw["Z"].to_numpy(np.float64)
    tvt_in = hw["TVT_input"].to_numpy(np.float64)
    tvt_true = hw["TVT"].to_numpy(np.float64) if "TVT" in hw else np.full(len(hw), np.nan)
    gr_w = robust_norm(GR)
    dz = np.gradient(Z, MD); geo = np.stack([
        (Z - np.nanmean(Z)) / (np.nanstd(Z) + 1e-6),
        dz / (np.nanstd(dz) + 1e-6),
        (MD - MD[0]) / 1000.0], 1)
    prefix = np.isfinite(tvt_in)
    ttrue_n = (tvt_true - ref) / span                     # 正規化真値
    return dict(
        gr_w=np.nan_to_num(gr_w).astype(np.float32),
        geo=np.nan_to_num(geo).astype(np.float32),
        gr_t=np.nan_to_num(gr_t).astype(np.float32),
        tgrid_n=tgrid_n.astype(np.float32),
        ttrue_n=ttrue_n.astype(np.float32),
        prefix=prefix, mds=(MD - MD[0]).astype(np.float32),
        ref=ref, span=span, wid=wid)


# ================= モデル =================
class ConvEnc(nn.Module):
    def __init__(self, cin, d, k=5, n=3):
        super().__init__()
        ch = [cin] + [d] * n
        self.blocks = nn.ModuleList([nn.Conv1d(ch[i], ch[i + 1], k, padding=k // 2) for i in range(n)])
        self.norm = nn.ModuleList([nn.GroupNorm(1, d) for _ in range(n)])

    def forward(self, x):                                  # (B,Cin,L)
        for c, nm in zip(self.blocks, self.norm):
            x = F.gelu(nm(c(x)))
        return x                                           # (B,D,L)


class Aligner(nn.Module):
    def __init__(self, d=64, cw=4, ct=2, tf_layers=2, heads=4):
        super().__init__()
        self.wenc = ConvEnc(cw, d)
        self.tenc = ConvEnc(ct, d)
        enc = nn.TransformerEncoderLayer(d, heads, d * 2, batch_first=True, activation="gelu")
        self.wtf = nn.TransformerEncoder(enc, tf_layers)   # well文脈(枝を解く/連続性)
        self.qp = nn.Linear(d, d); self.kp = nn.Linear(d, d)
        self.H = heads; self.dh = d // heads; self.scale = self.dh ** -0.5

    def forward(self, gr_w, geo, gr_t, tgrid_n):
        B = gr_w.shape[0]
        xw = torch.cat([gr_w[:, None], geo.transpose(1, 2)], 1)   # (B,4,L)
        hw = self.wtf(self.wenc(xw).transpose(1, 2))              # (B,L,D)
        xt = torch.stack([gr_t, tgrid_n], 1)                      # (B,2,K)
        ht = self.tenc(xt).transpose(1, 2)                        # (B,K,D)
        L, K = hw.shape[1], ht.shape[1]
        # multi-head cross-attention (well行 → typewell位置)
        q = self.qp(hw).view(B, L, self.H, self.dh).transpose(1, 2)   # (B,H,L,dh)
        k = self.kp(ht).view(B, K, self.H, self.dh).transpose(1, 2)   # (B,H,K,dh)
        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale    # (B,H,L,K)
        ah = F.softmax(scores, dim=-1)                                # (B,H,L,K)
        tg = tgrid_n[:, None, :, None]                               # (B,1,K,1)
        tvt_pred = torch.matmul(ah, tg).mean(1).squeeze(-1)          # (B,L) head平均
        a = ah.mean(1)                                              # (B,L,K) head平均(loss/argmax用)
        return tvt_pred, a


def soft_target(ttrue_n, K, sigma_bins):
    """真の正規化TVT→typewellグリッド上のgaussianソフトラベル(L,K)。範囲外はNaN行→maskで除外。"""
    L = len(ttrue_n)
    pos = np.clip(ttrue_n, 0, 1) * (K - 1)
    idx = np.arange(K)[None, :]
    d = idx - pos[:, None]
    t = np.exp(-0.5 * (d / sigma_bins) ** 2)
    t = t / (t.sum(1, keepdims=True) + 1e-9)
    valid = np.isfinite(ttrue_n) & (ttrue_n >= -0.02) & (ttrue_n <= 1.02)
    return t.astype(np.float32), valid


# ================= スモーク学習 =================
def to_batch(s, sigma_frac=0.006):
    """numpy sample(CPU) → 現DEVのテンソル束(+現sigmaのソフトラベル)。全坑井をGPUに載せないための逐次変換。"""
    d = {k: torch.tensor(np.nan_to_num(s[k]), device=DEV)[None] for k in ("gr_w", "geo", "gr_t", "tgrid_n")}
    K = len(s["tgrid_n"])
    st, valid = soft_target(s["ttrue_n"], K, sigma_bins=max(2.0, K * sigma_frac))
    d["st"] = torch.tensor(st, device=DEV); d["vmask"] = torch.tensor(valid, device=DEV)
    d["ttrue"] = torch.tensor(np.nan_to_num(s["ttrue_n"]), device=DEV)
    d.update(K=K, prefix=s["prefix"], span=s["span"], ref=s["ref"], ttrue_n=s["ttrue_n"], mds=s["mds"], wid=s["wid"])
    return d


def _losses(model, d):
    tvt_pred, a = model(d["gr_w"], d["geo"], d["gr_t"], d["tgrid_n"])
    tp = tvt_pred[0]; vm = d["vmask"]
    L_reg = F.huber_loss(tp[vm], d["ttrue"][vm], delta=0.05)
    loga = torch.log(a[0].clamp_min(1e-8))
    L_align = -(d["st"][vm] * loga[vm]).sum(1).mean()
    d2 = tp[2:] - 2 * tp[1:-1] + tp[:-2]
    L_smooth = (d2 ** 2).mean()
    return L_reg + 0.5 * L_align + 5.0 * L_smooth, L_reg, L_align, L_smooth, tp, a


def smoke(nwells=6, steps=60, d=64, cpu=False):
    global DEV
    if cpu: DEV = "cpu"
    wids = sorted(set(os.path.basename(f).split("__")[0]
                      for f in glob.glob(str(DATA / "*__horizontal_well.csv"))))[:nwells]
    print(f"device={DEV}  wells={len(wids)}  (勾配は毎ステップ全坑井で累積=実学習に近い)", flush=True)
    samples = [to_batch(load_well(w)) for w in wids]
    model = Aligner(d=d).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    print(f"params={sum(p.numel() for p in model.parameters())/1e3:.0f}k", flush=True)

    for step in range(steps):
        opt.zero_grad()
        agg = np.zeros(4)
        for dct in samples:
            loss, lr, la, ls, _, _ = _losses(model, dct)
            (loss / len(samples)).backward()
            agg += [loss.item(), lr.item(), la.item(), ls.item()]
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0); opt.step()
        if step % 10 == 0 or step == steps - 1:
            agg /= len(samples)
            with torch.no_grad():
                rs, ra = [], []
                for dct in samples:
                    _, _, _, _, tp, a = _losses(model, dct)
                    ev = (~dct["prefix"]) & dct["vmask"].cpu().numpy(); span = dct["span"]; K = dct["K"]
                    if not ev.any(): continue
                    pred = tp.cpu().numpy() * span; true = np.nan_to_num(dct["ttrue_n"]) * span
                    amax = a[0].argmax(1).cpu().numpy() / (K - 1) * span
                    rs.append(np.sqrt(np.mean((pred[ev] - true[ev]) ** 2)))
                    ra.append(np.sqrt(np.mean((amax[ev] - true[ev]) ** 2)))
                print(f"  step {step:3d} loss {agg[0]:.3f} (reg {agg[1]:.4f} align {agg[2]:.3f} smooth {agg[3]:.5f}) | "
                      f"eval RMSE soft {np.mean(rs):6.2f}±{np.std(rs):4.1f} / argmax {np.mean(ra):6.2f}ft (in-sample, {len(rs)}井)", flush=True)
    print("\n[smoke] forward/backward 健全。全坑井集約で eval が安定して縮めば設計成立の兆候。本学習(GroupKFold)はユーザー。")


# ================= 本学習(GroupKFold) =================
import pickle


def _load_all(stride_w, stride_t, OUT):
    cache = OUT / f"wellcache_s{stride_w}_{stride_t}.pkl"
    if cache.exists():
        print(f"cache load {cache.name}", flush=True); return pickle.loads(cache.read_bytes())
    wids = sorted(set(os.path.basename(f).split("__")[0]
                      for f in glob.glob(str(DATA / "*__horizontal_well.csv"))))
    samples = []
    for i, w in enumerate(wids):
        try: samples.append(load_well(w, stride_w, stride_t))
        except Exception as e: print(f"  skip {w}: {e}")
        if (i + 1) % 100 == 0: print(f"  loaded {i+1}/{len(wids)}", flush=True)
    cache.write_bytes(pickle.dumps(samples)); print(f"cache saved {cache.name} ({len(samples)}井)", flush=True)
    return samples


def train(folds=5, epochs=12, d=96, heads=4, batch_wells=8, stride_w=3, stride_t=2,
          lr=2e-3, out="artifacts_v72", cpu=False, max_wells=None, fold_only=None, seed=0):
    global DEV
    if cpu: DEV = "cpu"
    OUT = Path(__file__).parent / out; OUT.mkdir(exist_ok=True)
    print(f"=== v72 train === dev={DEV} folds={folds} epochs={epochs} d={d} heads={heads} "
          f"batch_wells={batch_wells} stride_w={stride_w} stride_t={stride_t} lr={lr}", flush=True)
    samples = _load_all(stride_w, stride_t, OUT)
    if max_wells: samples = samples[:max_wells]
    N = len(samples); rng = np.random.RandomState(seed)
    fold_of = np.zeros(N, int); perm = rng.permutation(N)
    for f in range(folds): fold_of[perm[f::folds]] = f
    print(f"wells={N}", flush=True)
    oof_pred, oof_true, oof_md, oof_amax = {}, {}, {}, {}

    for f in range(folds):
        if fold_only is not None and f != fold_only: continue
        tr = np.where(fold_of != f)[0]; va = np.where(fold_of == f)[0]
        model = Aligner(d=d, heads=heads).to(DEV)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
        for ep in range(epochs):
            model.train()
            sig = float(np.interp(ep, [0, max(epochs - 1, 1)], [0.02, 0.004]))  # sigma_frac 広→狭(アテンションを段々鋭く)
            order = rng.permutation(tr); tot = np.zeros(4); nb = 0
            for c in range(0, len(order), batch_wells):
                chunk = order[c:c + batch_wells]; opt.zero_grad()
                for wi in chunk:
                    dct = to_batch(samples[wi], sig)
                    loss, lr_, la, ls, _, _ = _losses(model, dct)
                    (loss / len(chunk)).backward()
                    tot += [loss.item(), lr_.item(), la.item(), ls.item()]; nb += 1
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0); opt.step()
            sched.step(); tot /= max(nb, 1)
            print(f"  fold{f} ep{ep:2d} sig{sig:.3f} loss{tot[0]:.3f} (reg{tot[1]:.4f} align{tot[2]:.3f} smooth{tot[3]:.5f})", flush=True)
        model.eval()
        with torch.no_grad():
            for wi in va:
                s = samples[wi]; dct = to_batch(s, 0.004)
                tp, a = model(dct["gr_w"], dct["geo"], dct["gr_t"], dct["tgrid_n"])
                span, ref, K = s["span"], s["ref"], dct["K"]
                pred = tp[0].cpu().numpy() * span + ref
                amax = a[0].argmax(1).cpu().numpy() / (K - 1) * span + ref
                true = s["ttrue_n"] * span + ref
                ev = (~s["prefix"]) & np.isfinite(true)
                oof_pred[s["wid"]] = pred[ev]; oof_true[s["wid"]] = true[ev]
                oof_md[s["wid"]] = s["mds"][ev]; oof_amax[s["wid"]] = amax[ev]
        torch.save(model.state_dict(), OUT / f"aligner_fold{f}.pt")
        es = np.concatenate([oof_pred[w] - oof_true[w] for w in oof_pred])
        print(f"  [fold{f} done] 累積 pooled RMSE(soft)={np.sqrt(np.mean(es**2)):.4f}", flush=True)

    ws = list(oof_pred)
    e_soft = np.concatenate([oof_pred[w] - oof_true[w] for w in ws])
    e_amax = np.concatenate([oof_amax[w] - oof_true[w] for w in ws])
    print(f"\n=== v72 CV (overall pooled, {len(ws)}井) ===")
    print(f"  soft   RMSE = {np.sqrt(np.mean(e_soft**2)):.4f}")
    print(f"  argmax RMSE = {np.sqrt(np.mean(e_amax**2)):.4f}  (commit)")
    print(f"  (参考: combo 6.42 / naive 15)")
    allmd = np.concatenate([oof_md[w] for w in ws])
    print("  md_since帯別 RMSE(soft) [遠方が平坦ならドリフト回避成功]:")
    qs = np.quantile(allmd, [0, .2, .4, .6, .8, 1.0])
    for i in range(5):
        m = (allmd >= qs[i]) & (allmd <= qs[i + 1])
        if m.any(): print(f"    md {qs[i]:6.0f}-{qs[i+1]:6.0f}: {np.sqrt(np.mean(e_soft[m]**2)):.3f}")
    np.savez(OUT / "oof_v72.npz", **{f"pred_{w}": oof_pred[w] for w in ws},
             **{f"true_{w}": oof_true[w] for w in ws}, **{f"md_{w}": oof_md[w] for w in ws})
    (OUT / "cv_v72.txt").write_text(f"soft {np.sqrt(np.mean(e_soft**2)):.4f} / argmax {np.sqrt(np.mean(e_amax**2)):.4f} / {len(ws)}井\n", encoding="utf-8")
    print(f"[saved] {OUT/'oof_v72.npz'} / cv_v72.txt / aligner_fold*.pt")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="数坑井でパイプライン確認")
    ap.add_argument("--train", action="store_true", help="本学習(GroupKFold, 全773井)")
    ap.add_argument("--wells", type=int, default=6, help="[smoke]坑井数")
    ap.add_argument("--steps", type=int, default=60, help="[smoke]step数")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--d", type=int, default=96, help="モデル次元(大=表現力↑VRAM↑)")
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--batch_wells", type=int, default=8, help="1 optimizer step に累積する坑井数")
    ap.add_argument("--stride_w", type=int, default=3, help="well行間引き(小=高解像度,重い)")
    ap.add_argument("--stride_t", type=int, default=2, help="typewell間引き")
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--max_wells", type=int, default=None, help="デバッグ用に坑井数制限")
    ap.add_argument("--fold_only", type=int, default=None, help="1 fold だけ回す(0..folds-1)")
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()
    if a.train:
        train(folds=a.folds, epochs=a.epochs, d=a.d, heads=a.heads, batch_wells=a.batch_wells,
              stride_w=a.stride_w, stride_t=a.stride_t, lr=a.lr, cpu=a.cpu,
              max_wells=a.max_wells, fold_only=a.fold_only)
    else:
        smoke(a.wells, a.steps, cpu=a.cpu)
