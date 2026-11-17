# -*- coding: utf-8 -*-
"""v101 学習 seed ブレンダー(単体PFの 32 seed の凸結合を学習)。後段(L5)とは無関係=単体PF出力のみで評価。
各 seed にスカラー特徴 φ_s(錨距離/尤度/レベル…)→ 共有MLPで logit → 32本 softmax → 凸結合 → 真TVTにfit。
ps_combo(ww·exp(−d²/2τ²) 固定)を学習で超えられるかが第一ゲート。GroupKFold(by well) OOF overall RMSE で判定。
入力は exp_seed_modes_v101.py が吐いた capture キャッシュ(artifacts_v101/cap_{bank}_emit_s32_n773.pkl)。
実行: .venv/Scripts/python.exe v101/train_seed_blender_v101.py   (env: BANKS=pf_1,pf_2,pf_3 TRAJ=pfw EPOCHS=200)
"""
import os, sys, pickle
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "v99"))
import pf_banks_v95 as pf
import torch
import torch.nn as nn

BANKS = os.environ.get("BANKS", "pf_1,pf_2,pf_3").split(",")
TRAJ  = os.environ.get("TRAJ", "pfw")           # pfw=PF自体 / ps=smoothed
EPOCHS = int(os.environ.get("EPOCHS", "200"))
TAUS  = float(os.environ.get("TAUS", "6.0"))    # ps_combo 比較用
NFOLD = 5
DEV = "cuda" if torch.cuda.is_available() else "cpu"
ANCH = pf._ANCHOR
ARTD = os.path.join(os.path.dirname(__file__), "..", "artifacts_v101")


def load_bank(bank):
    p = os.path.join(ARTD, f"cap_{bank}_emit_s32_n773.pkl")
    if not os.path.exists(p):
        print(f"[skip] cache無し: {os.path.basename(p)}"); return None
    with open(p, "rb") as f:
        return pickle.load(f)


def per_seed_features(traj, ww, atvt):
    """φ_s(achievable, 真値不使用)。well内で正規化してバンク/井に不変化。戻り (S,F), base, ps_combo。"""
    S, T = traj.shape
    Lvl = traj.mean(1)                                  # 各seedのレベル
    base = (ww[:, None] * traj).sum(0)                  # baseline(尤度加重平均)
    base_lvl = (ww * Lvl).sum()
    sd = Lvl.std() + 1e-6
    lvl_c = (Lvl - base_lvl) / sd                       # 尤度平均からの相対レベル
    if atvt is not None:
        da = ((traj - atvt[None, :]) ** 2).mean(1)      # 錨距離²
        rda = np.sqrt(da)
        rda_c = (rda - rda.mean()) / (rda.std() + 1e-6)
        a_off = (Lvl - atvt.mean()) / sd                # 錨レベルからのオフセット
        wc = ww * np.exp(-(da - da.min()) / (2 * TAUS ** 2)); wc = wc / max(wc.sum(), 1e-12)
        pscmb = (wc[:, None] * traj).sum(0)             # ps_combo(固定)
    else:
        rda_c = np.zeros(S); a_off = np.zeros(S); pscmb = base
    lw = np.log(ww + 1e-6); lw = (lw - lw.mean()) / (lw.std() + 1e-6)
    # ★per-well 文脈(全seedに broadcast): バンクの「錨を既に使ってるか/雲の広さ」を区別させる
    if atvt is not None:
        b_anchor = float(abs(base_lvl - atvt.mean()))     # baseline が錨からどれだけ離れているか(ft)
    else:
        b_anchor = 0.0
    disp = float(Lvl.std())                               # seed雲の分散(ft)
    ctx = np.tile([b_anchor, disp], (S, 1)).astype(np.float32)
    phi = np.concatenate([np.stack([lvl_c, rda_c, a_off, ww * S, lw], 1).astype(np.float32), ctx], 1)  # (S,7)
    return phi, base, pscmb


# ---- データ構築(全バンク結合) ----
DATA = []   # list of dict(phi, traj, true, base, pscmb, well, bank)
for bank in BANKS:
    cap = load_bank(bank)
    if cap is None:
        continue
    res, trues, keep = cap["res"], cap["trues"], cap["keep"]
    for k in range(len(keep)):
        traj = res[k][TRAJ].astype(np.float64); ww = res[k]["ww"].astype(np.float64)
        ww = ww / max(ww.sum(), 1e-12); true = np.asarray(trues[k], np.float64)
        S, T = traj.shape
        sp = ANCH.get(keep[k]); atvt = None
        if sp is not None and len(np.asarray(sp["tvt"], float)) == T:
            atvt = np.asarray(sp["tvt"], float)
        phi, base, pscmb = per_seed_features(traj, ww, atvt)
        DATA.append(dict(phi=phi, traj=traj.astype(np.float32), true=true.astype(np.float32),
                         base=base.astype(np.float32), pscmb=pscmb.astype(np.float32),
                         well=keep[k], bank=bank))
print(f"[data] {len(DATA)} (bank×well) samples  banks={BANKS}  traj={TRAJ}  dev={DEV}", flush=True)
NF = DATA[0]["phi"].shape[1]


class Scorer(nn.Module):
    def __init__(self, nf, h=16):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(nf, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1))

    def forward(self, phi):                              # phi (S,F) -> weights (S,)
        return torch.softmax(self.net(phi).squeeze(-1), 0)


def overall_rmse(errs):
    e = np.concatenate(errs); e = e[np.isfinite(e)]
    return float(np.sqrt(np.mean(e ** 2)))


# ---- パディングして一括ベクトル化(井ごとループのGPUオーバーヘッド回避) ----
S = DATA[0]["phi"].shape[0]
ROWCAP = int(os.environ.get("ROWCAP", "256"))     # 学習損失用に行を間引く上限(重み推定は行数不変。0=全行)
def pack(idxs, cap=0):
    def rows(i):
        T = len(DATA[i]["true"])
        if cap and T > cap:
            return np.linspace(0, T - 1, cap).round().astype(int)   # 等間隔間引き
        return np.arange(T)
    ridx = [rows(i) for i in idxs]
    Tm = max(len(r) for r in ridx); N = len(idxs)
    PHI = np.zeros((N, S, NF), np.float32); TRA = np.zeros((N, S, Tm), np.float32)
    TRU = np.zeros((N, Tm), np.float32); MSK = np.zeros((N, Tm), np.float32)
    for a, i in enumerate(idxs):
        r = ridx[a]; T = len(r)
        PHI[a] = DATA[i]["phi"]; TRA[a, :, :T] = DATA[i]["traj"][:, r]
        TRU[a, :T] = DATA[i]["true"][r]; MSK[a, :T] = 1.0
    to = lambda x: torch.from_numpy(x).to(DEV)
    return to(PHI), to(TRA), to(TRU), to(MSK)


# ---- GroupKFold by well ----
wells = sorted({d["well"] for d in DATA})
w2f = {w: i % NFOLD for i, w in enumerate(wells)}
oof = [None] * len(DATA)

for fold in range(NFOLD):
    tr = [i for i, d in enumerate(DATA) if w2f[d["well"]] != fold]
    va = [i for i, d in enumerate(DATA) if w2f[d["well"]] == fold]
    model = Scorer(NF).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-5)
    PHI, TRA, TRU, MSK = pack(tr, cap=ROWCAP)
    for ep in range(EPOCHS):
        model.train(); opt.zero_grad()
        logit = model.net(PHI).squeeze(-1)              # (N,S)
        w = torch.softmax(logit, 1)                     # (N,S)
        pred = (w[:, :, None] * TRA).sum(1)             # (N,Tm)
        loss = (((pred - TRU) ** 2) * MSK).sum() / MSK.sum()   # overall pooled MSE
        loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        PHIv, TRAv, _, _ = pack(va, cap=0)
        wv = torch.softmax(model.net(PHIv).squeeze(-1), 1)
        predv = (wv[:, :, None] * TRAv).sum(1).cpu().numpy()
        for a, i in enumerate(va):
            oof[i] = predv[a, :len(DATA[i]["true"])]
    print(f"  fold{fold} done loss={float(loss):.4f} (tr={len(tr)} va={len(va)})", flush=True)

# ---- 集計(全体 + バンク別) ----
def report(idxs, tag):
    eb = [DATA[i]["base"] - DATA[i]["true"] for i in idxs]
    ec = [DATA[i]["pscmb"] - DATA[i]["true"] for i in idxs]
    el = [oof[i] - DATA[i]["true"] for i in idxs]
    rb, rc, rl = overall_rmse(eb), overall_rmse(ec), overall_rmse(el)
    print(f"[{tag:8s}] baseline={rb:.4f}  ps_combo={rc:.4f}({rc-rb:+.3f})  learned_OOF={rl:.4f}({rl-rb:+.3f})  "
          f"[learned vs combo {rl-rc:+.3f}]", flush=True)

print("\n===== overall RMSE (OOF, traj=%s) =====" % TRAJ)
report(range(len(DATA)), "ALL")
for bank in BANKS:
    idxs = [i for i, d in enumerate(DATA) if d["bank"] == bank]
    if idxs:
        report(idxs, bank)
print("\n[done]")
