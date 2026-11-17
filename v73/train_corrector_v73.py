# -*- coding: utf-8 -*-
"""
v73 C2 — 逐次オンライン補正器の学習+OOF評価 (docs/v73.md)
=========================================================
入力: nnpf_v73.py MODE=collect が吐いた rollouts_{tag}.pkl
  {well: {feats(S,T,F=10) f16, e_f(S,T) f16, e_s(S,T) f16, ll(S,), true(T,)}}
やること:
  1) 坑井5-fold(ソート順 index%5 = v50/v51と同じ割当)で GRU補正器を fold-OOF 学習
     入力=innovation特徴系列(因果)、教師= e_s[t-SHIFT](SHIFT=LAG: +L行先まで見てから確定=平滑と同じ合法遅延)
  2) OOF適用: per-seed ê を出し、per-seed補正 → log_likソフトマックスで合成 → overall RMSE を
     gain g∈{0.25,0.5,0.75,1.0} で A/B(g=収縮ガード)。md_since帯別も出す(ドリフト削減の直接指標)。
使い方: TAG=w0.02_s16 EPOCHS=4 python v73/train_corrector_v73.py
注意: RMSE>10等の変な数字はまず実装バグを疑う。
"""
import os, io, sys, time, pickle, json
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
import numpy as np
import torch, torch.nn as nn
from pathlib import Path

P0 = Path(r"C:\Users\kosaka256\Documents\rogii_claude")
ART = P0 / "v73" / "artifacts_v73"
TAG = os.environ.get("TAG", "w0.02_s32")
EPOCHS = int(os.environ.get("EPOCHS", "4"))
HID = int(os.environ.get("HID", "64"))
SHIFT = int(os.environ.get("SHIFT", "16"))       # 教師の遅延(=LAG)。ê_s[t] は特徴 t+SHIFT まで見て確定
BATCH = int(os.environ.get("BATCH", "64"))
LR = float(os.environ.get("LR", "1e-3"))
TCLIP = float(os.environ.get("TCLIP", "40.0"))   # 教師クリップ(tail暴走ガード)
SEED_STRIDE = int(os.environ.get("SEED_STRIDE", "4"))  # 学習はseed間引き(32→8/井)。評価は全seed
BIDIR = int(os.environ.get("BIDIR", "0"))              # 1=双方向GRU(全系列look-ahead=合法, オフライン補正上限)
LS_SCALE = 10.536319667317654                     # r0 likelihood_scale(seed合成温度, nnpf_v73と同一)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
NFEAT = 10


class Corr(nn.Module):
    def __init__(self, f=NFEAT, h=HID, bidir=bool(BIDIR)):
        super().__init__()
        self.gru = nn.GRU(f, h, batch_first=True, bidirectional=bidir)
        ho = h * (2 if bidir else 1)
        self.head = nn.Sequential(nn.Linear(ho, h // 2), nn.GELU(), nn.Linear(h // 2, 1))

    def forward(self, x):                          # (B,T,F) → (B,T)
        o, _ = self.gru(x)
        return self.head(o).squeeze(-1)


def seed_weights(ll):
    if np.isfinite(ll).any():
        mx = np.nanmax(ll[np.isfinite(ll)])
        lkw = np.where(np.isfinite(ll), ll - mx, -np.inf)
        ww = np.exp(lkw / LS_SCALE); s = ww.sum()
        return ww / s if s > 1e-300 else np.full(len(ll), 1.0 / len(ll))
    return np.full(len(ll), 1.0 / len(ll))


def main():
    fp = ART / f"rollouts_{TAG}.pkl"
    R = pickle.loads(fp.read_bytes())
    wells = sorted(R.keys())
    fold_of = {w: i % 5 for i, w in enumerate(wells)}     # v50/v51 と同じ wells[f::5] 割当
    print(f"rollouts {fp.name}: {len(wells)}坑井  SHIFT={SHIFT} EPOCHS={EPOCHS} HID={HID}", flush=True)

    # 特徴正規化(全体で1回。特徴は設計時に概ねスケール済なので mean/std の軽い補正のみ)
    smp = np.concatenate([R[w]["feats"][0, ::7].astype(np.float32) for w in wells[::13]], 0)
    mu = smp.mean(0); sd = smp.std(0) + 1e-3
    print("feat mu:", np.round(mu, 3), "\nfeat sd:", np.round(sd, 3), flush=True)

    # ---------- fold学習 ----------
    for f in range(5):
        ck = ART / f"corr_{TAG}{'_bi' if BIDIR else ''}_f{f}.pt"
        if ck.exists():
            print(f"fold{f}: ckpt exists, skip", flush=True); continue
        trw = [w for w in wells if fold_of[w] != f]
        seqs = []
        for w in trw:
            d = R[w]; S, T, F = d["feats"].shape
            for s in range(0, S, SEED_STRIDE):
                seqs.append((w, s, T))
        model = Corr().to(DEV); opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
        rng = np.random.RandomState(42 + f)
        t0 = time.time()
        for ep in range(EPOCHS):
            rng.shuffle(seqs); tot = 0.0; nb = 0
            for c0 in range(0, len(seqs), BATCH):
                bs = seqs[c0:c0 + BATCH]
                Tmax = max(t for _, _, t in bs)
                X = np.zeros((len(bs), Tmax, NFEAT), np.float32)
                Y = np.zeros((len(bs), Tmax), np.float32); M = np.zeros((len(bs), Tmax), bool)
                for j, (w, s, T) in enumerate(bs):
                    d = R[w]
                    X[j, :T] = (d["feats"][s].astype(np.float32) - mu) / sd
                    es = d["e_s"][s].astype(np.float32)
                    if SHIFT > 0:
                        Y[j, SHIFT:T] = np.clip(es[:T - SHIFT], -TCLIP, TCLIP)
                        M[j, SHIFT:T] = np.isfinite(es[:T - SHIFT])
                    else:
                        Y[j, :T] = np.clip(es, -TCLIP, TCLIP); M[j, :T] = np.isfinite(es)
                xb = torch.from_numpy(X).to(DEV); yb = torch.from_numpy(Y).to(DEV)
                mb = torch.from_numpy(M).to(DEV)
                o = model(xb)
                loss = torch.nn.functional.huber_loss(o[mb], yb[mb], delta=3.0)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0); opt.step()
                tot += loss.item(); nb += 1
            print(f"  fold{f} ep{ep} loss {tot / nb:.4f} ({time.time()-t0:.0f}s)", flush=True)
        torch.save(dict(sd=model.state_dict(), mu=mu, sdv=sd), ck)
        print(f"fold{f} saved {ck.name}", flush=True)

    # ---------- OOF適用+A/B ----------
    models = []
    for f in range(5):
        d = torch.load(ART / f"corr_{TAG}{'_bi' if BIDIR else ''}_f{f}.pt", map_location=DEV, weights_only=False)
        m = Corr().to(DEV); m.load_state_dict(d["sd"]); m.eval()
        models.append((m, d["mu"], d["sdv"]))
    gains = [0.0, 0.25, 0.5, 0.75, 1.0]
    err_by_gain = {g: [] for g in gains}
    md_bins = np.linspace(0, 6000, 7); md_num = {g: np.zeros(6) for g in gains}; md_den = np.zeros(6)
    with torch.no_grad():
        for w in wells:
            d = R[w]; m, mu_, sd_ = models[fold_of[w]]
            S, T, F = d["feats"].shape
            X = (d["feats"].astype(np.float32) - mu_) / sd_
            o = m(torch.from_numpy(X).to(DEV)).cpu().numpy()          # (S,T)
            # ê_s[t] = o[t+SHIFT](遅延整列)。端は最後の値
            eh = np.empty_like(o)
            if SHIFT > 0 and T > SHIFT:
                eh[:, :T - SHIFT] = o[:, SHIFT:]
                eh[:, T - SHIFT:] = o[:, -1][:, None]
            else:
                eh = o.copy()
            ww = seed_weights(d["ll"])
            es = d["e_s"].astype(np.float32)                          # per-seed smoothed誤差
            fin = np.isfinite(es[0])
            mdn = np.arange(T, dtype=np.float32)                      # 行index≒md_since代理(1ft刻み)
            bi = np.clip(np.searchsorted(md_bins, mdn) - 1, 0, 5)
            md_den += np.bincount(bi[fin], minlength=6)
            for g in gains:
                ce = es - g * eh                                      # 補正後 per-seed 誤差
                comb = (ww[:, None] * ce).sum(0)                      # seed合成(=予測合成−真値)
                err_by_gain[g].append(comb[fin])
                md_num[g] += np.bincount(bi[fin], weights=comb[fin] ** 2, minlength=6)
    print("\n=== C2 OOF A/B (overall pooled RMSE) ===")
    res = {}
    for g in gains:
        e = np.concatenate(err_by_gain[g]); r = float(np.sqrt(np.mean(e ** 2)))
        res[f"gain{g}"] = r
        print(f"  gain={g:4.2f}: {r:.4f}" + ("   ← baseline(補正なし)" if g == 0 else ""), flush=True)
    print("\n=== md帯別 RMSE (gain=0 vs best) — 遠方が縮めばドリフト削減 ===")
    best_g = min(gains, key=lambda g: res[f"gain{g}"])
    for k in range(6):
        r0 = np.sqrt(md_num[0.0][k] / max(md_den[k], 1)); rb = np.sqrt(md_num[best_g][k] / max(md_den[k], 1))
        print(f"  行 {int(md_bins[k]):4d}-{int(md_bins[k+1]):4d}: base {r0:6.3f} → g{best_g} {rb:6.3f}")
    (ART / f"corr_eval_{TAG}{'_bi' if BIDIR else ''}.json").write_text(json.dumps(res, indent=1))
    print(f"\n[saved] corr_eval_{TAG}.json / corr_{TAG}_f*.pt")


if __name__ == "__main__":
    main()
