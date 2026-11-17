# -*- coding: utf-8 -*-
"""v104 貪欲候補選択: 103候補の予測(seqfeat D)から、train精度を貪欲最大化する部分集合を選ぶ(Caruana型・置換あり)。
val過適合OK方針。選ばれた候補=keep、それ以外=PRUNE。相関/単体精度も出す。
出力: v104/artifacts_v104/greedy_keep.json(keep名, prune名)。
実行: .venv/Scripts/python.exe v104/greedy_select_v104.py"""
import sys, io, json
try:
    sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
except Exception:
    pass
from pathlib import Path
import numpy as np, pickle

V104 = Path(__file__).resolve().parent
ART = V104 / "artifacts_v104"
SF = ART / "seqfeat_v104_cf.pkl"
CAND = json.load(open(str(SF) + ".cand.json", encoding="utf-8"))
K = len(CAND)

# --- 候補予測 P(N,K) を seqfeat D から復元 + y ---
d = pickle.loads(SF.read_bytes())
y = np.load(ART / "l5_v104baseg-1_oof.npz")["y"]
N = len(y)
P = np.full((N, K), np.nan, np.float64)
for w, f in d.items():
    D = f["D"]; rows = f["rows"]; bid = f["bid"]
    P[rows, :] = D[bid, :]
mask = np.isfinite(y)
print(f"[greedy] N={N} K={K} 有効行={mask.sum()}", flush=True)


def rmse(p):
    m = mask & np.isfinite(p)
    return float(np.sqrt(np.mean((p[m] - y[m]) ** 2)))


# --- 単体精度 ---
solo = np.array([rmse(P[:, k]) for k in range(K)])
order = np.argsort(solo)
print("\n=== 単体精度 上位10 / 下位5 ===")
for k in order[:10]:
    print(f"  {solo[k]:.3f}  {CAND[k]}")
print("  ...")
for k in order[-5:]:
    print(f"  {solo[k]:.3f}  {CAND[k]}")

# --- Caruana 貪欲(置換あり)★matmul高速版: argmin_k sum((r+P_k)^2)= 2 r·P_k + ||P_k||^2 ---
T = int(200)
Pm = P[mask].astype(np.float32)          # (M,K)
ym = y[mask].astype(np.float64)
M = len(ym); norms = (Pm.astype(np.float64) ** 2).sum(0)   # (K,)
S = np.zeros(M, np.float64); sel = []; best_hist = []
for t in range(T):
    n = len(sel)
    r = (S - (n + 1) * ym)
    dot = Pm.T @ r.astype(np.float32)     # (K,) BLAS
    score = 2.0 * dot.astype(np.float64) + norms
    best_k = int(np.argmin(score))
    S += Pm[:, best_k].astype(np.float64); sel.append(best_k)
    ens = S / (n + 1); best_r = float(np.sqrt(np.mean((ens - ym) ** 2)))
    best_hist.append(best_r)
    if t < 5 or (t + 1) % 25 == 0:
        print(f"  iter{t+1}: ens_rmse={best_r:.4f} add={CAND[best_k]}", flush=True)

best_iter = int(np.argmin(best_hist)) + 1
sel_best = sel[:best_iter]
counts = {}
for k in sel_best:
    counts[k] = counts.get(k, 0) + 1
keep_idx = sorted(counts, key=lambda k: -counts[k])
keep = [CAND[k] for k in keep_idx]
prune = [CAND[k] for k in range(K) if k not in set(keep_idx)]
print(f"\n=== Caruana結果: best_iter={best_iter} ens_rmse={min(best_hist):.4f}(単体最良={solo.min():.4f})===")
print(f"  keep {len(keep)}候補 / prune {len(prune)}候補")
print("  keep(選択回数上位):", [f"{CAND[k]}×{counts[k]}" for k in keep_idx[:15]])
print("  prune:", prune[:20], "..." if len(prune) > 20 else "")

# prune を「消せる部分文字列」ではなく完全名で。train_l456 PRUNE は部分一致なので、prune名を完全一致させるため
# 各 prune 名がユニークに一致するようにそのまま渡す(部分一致で他を巻き込まないか検査)。
out = {"keep": keep, "prune": prune, "prune_exact": prune,
       "caruana_ens_rmse": float(min(best_hist)), "solo_best": float(solo.min())}
(ART / "greedy_keep.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"\n[greedy] greedy_keep.json 書き出し(keep={len(keep)})")
