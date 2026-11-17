# -*- coding: utf-8 -*-
"""
v50 Stage C 診断 — 検証器の壁の正体: 情報欠如 か field指紋過学習 か
====================================================================
(v34教訓: 井戸GroupKFoldは同フィールド井戸がtrain側に残るため、学習集約のfield過学習を検出できない)
1) XY をクラスタリングしてフィールドを定義
2) スカラー検証器(C2-v1と同型)を 井戸5fold / leave-field-out の両方で学習・OOF評価
   - 混合の pooled RMSE
   - 介入必要井戸(bestがbaseを2ft以上上回る)での枝ランキング精度
   - base最良井戸での誤介入率
   - 選択的ゲート曲線(確信度τ以上のときだけ argmax 仮説に置換)
実行: python v50/stageC_diag_v50.py
"""
import os, sys, json, pickle
os.environ.setdefault("TAG", "capaug"); os.environ.setdefault("CH", "128")
os.environ.setdefault("DIM", "96"); os.environ.setdefault("DILS", "1,2,4,8,16,32")
os.environ.setdefault("BTAG", "combo"); os.environ.setdefault("REFH", "1")
os.environ.setdefault("DILT", "1,2,4,8,16"); os.environ.setdefault("BCH", "96")
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import stageB_v50 as sb

DEV = sb.DEV
HYPS = pickle.loads((sb.ART / "stageC_hyps.pkl").read_bytes())
data = pickle.loads(sb.sa.CACHE.read_bytes())
WELLS = sorted(HYPS.keys()); K = HYPS[WELLS[0]]["paths"].shape[0]; NF = HYPS[WELLS[0]]["feats"].shape[1]

# ---------- 1) フィールド定義(XY 連結成分, しきい値でクラスタ) ----------
XY = np.array([data[w]["xy"] for w in WELLS])
from scipy.cluster.hierarchy import fcluster, linkage
Z = linkage(XY, method="single")
for th in [5000, 10000, 20000, 40000]:
    lab = fcluster(Z, t=th, criterion="distance")
    sizes = np.bincount(lab)[1:]
    print(f"  th={th}ft: fields={lab.max()}  sizes(top8)={sorted(sizes)[::-1][:8]}")
FTH = float(os.environ.get("FTH", "20000"))
FIELD = fcluster(Z, t=FTH, criterion="distance")
nf_ = FIELD.max()
print(f"採用 th={FTH:g}ft → {nf_} fields")

# ---------- 井戸ごとの教材 ----------
W_feat = {}; W_E = {}; W_W = {}; W_rms = {}
for w in WELLS:
    h = HYPS[w]; e = sb.EVD[w]
    dtr = e["dtrue"]; wgt = e["wgt"]
    ok = np.isfinite(dtr) & (wgt > 0)
    if ok.sum() < 4: continue
    E = (h["paths"][:, ok] - dtr[ok][None, :]).astype(np.float64)
    Wt = wgt[ok].astype(np.float64)
    W_feat[w] = h["feats"]; W_E[w] = E; W_W[w] = Wt
    W_rms[w] = np.sqrt((E ** 2 * Wt[None, :]).sum(1) / Wt.sum())
WL = [w for w in WELLS if w in W_feat]

class V1(nn.Module):
    def __init__(self, nf, h=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(nf, h), nn.GELU(), nn.Linear(h, h), nn.GELU(), nn.Linear(h, 1))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)
    def forward(self, x): return self.net(x)[..., 0]

def train_eval(groups, tag):
    """groups: well→group id。GroupKFold 相当で OOF logits を得る。"""
    gids = np.array([groups[w] for w in WL])
    uniq = np.unique(gids)
    oof_logit = {}
    for gi, gset in enumerate(np.array_split(uniq, min(5, len(uniq)))):
        vmask = np.isin(gids, gset)
        trn = [w for w, vm in zip(WL, vmask) if not vm]
        val = [w for w, vm in zip(WL, vmask) if vm]
        if not val: continue
        torch.manual_seed(gi)
        Xs = torch.tensor(np.stack([W_feat[w] for w in trn]), dtype=torch.float32, device=DEV)
        mu = Xs.mean((0, 1), keepdim=True); sd = Xs.std((0, 1), keepdim=True).clamp_min(1e-6)
        Xn = (Xs - mu) / sd
        rms = torch.tensor(np.stack([W_rms[w] for w in trn]), dtype=torch.float32, device=DEV)
        soft_t = torch.softmax(-(rms - rms.min(1, keepdim=True).values) / 1.5, dim=1)
        Es = [torch.tensor(W_E[w], dtype=torch.float32, device=DEV) for w in trn]
        Ws = [torch.tensor(W_W[w], dtype=torch.float32, device=DEV) for w in trn]
        ver = V1(NF).to(DEV)
        opt = torch.optim.AdamW(ver.parameters(), lr=1e-3, weight_decay=1e-4)
        for ep in range(300):
            lg = ver(Xn); wsoft = torch.softmax(lg, 1)
            ce = -(soft_t * F.log_softmax(lg, 1)).sum(1).mean()
            mse = torch.stack([(((wsoft[i][:, None] * Es[i]).sum(0)) ** 2 * Ws[i]).sum() / Ws[i].sum()
                               for i in range(len(trn))]).mean()
            loss = ce + 0.05 * mse
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        with torch.no_grad():
            Xv = torch.tensor(np.stack([W_feat[w] for w in val]), dtype=torch.float32, device=DEV)
            lgv = ver((Xv - mu) / sd).cpu().numpy()
        for w, l in zip(val, lgv): oof_logit[w] = l
    # ---- 指標 ----
    def pooled(pred_fn):
        num = 0.0; den = 0.0
        for w in WL:
            pw = pred_fn(w)                          # (K,) 重み
            e2 = ((pw[:, None] * W_E[w]).sum(0)) ** 2
            num += (e2 * W_W[w]).sum(); den += W_W[w].sum()
        return float(np.sqrt(num / den))
    base_r = pooled(lambda w: np.eye(K)[0])
    soft_r = pooled(lambda w: np.exp(oof_logit[w]) / np.exp(oof_logit[w]).sum())
    need = [w for w in WL if W_rms[w][0] - W_rms[w].min() > 2.0]     # 介入必要井戸
    safe = [w for w in WL if W_rms[w][0] - W_rms[w].min() < 1.0]     # base最良(≈)井戸
    rank_acc = float(np.mean([np.argmax(oof_logit[w]) == np.argmin(W_rms[w]) for w in need])) if need else float("nan")
    top_not0 = float(np.mean([np.argmax(oof_logit[w]) != 0 for w in safe])) if safe else float("nan")
    print(f"\n[{tag}] base={base_r:.4f}  soft混合={soft_r:.4f}")
    print(f"  介入必要井戸({len(need)}): 枝ランキング正解率={rank_acc:.3f}")
    print(f"  base最良井戸({len(safe)}): 誤介入(top≠base)率={top_not0:.3f}")
    # ---- 選択的ゲート曲線: p_top−p_base > τ のときだけ argmax 仮説へ ----
    print("  選択的ゲート: τ → pooled RMSE / 介入井戸数")
    for tau in [0.1, 0.2, 0.3, 0.5, 0.7]:
        def gate(w):
            p = np.exp(oof_logit[w]); p = p / p.sum()
            j = int(np.argmax(p))
            if j != 0 and p[j] - p[0] > tau:
                return np.eye(K)[j]
            return np.eye(K)[0]
        n_int = sum(1 for w in WL if np.argmax(gate(w)) != 0)
        print(f"    τ={tau:.1f}: {pooled(gate):.4f}  (介入 {n_int}井)")
    return oof_logit

groups_well = {w: i % 5 for i, w in enumerate(WL)}                  # 井戸5fold(ソート順=従来同等)
groups_field = {w: int(FIELD[WELLS.index(w)]) for w in WL}
lg_w = train_eval(groups_well, "井戸5fold")
lg_f = train_eval(groups_field, "leave-field-out")
