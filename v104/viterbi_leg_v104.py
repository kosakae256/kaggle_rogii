# -*- coding: utf-8 -*-
"""v104 大域アライン脚: 既存 emission sim(GRフリー錨中心 ±45ft/0.5 の GR尤度)に Viterbi 大域MAP + 傾き平滑をかけて δ を出す。
PF(逐次Bayes+resample)とは別アルゴリズム(大域MAP DP)= 脱相関脚。全系列の平滑整合で datum 枝を一貫させる。
学習不要(決定的)。出力 viterbi_oof.npz。env: LAM(傾きペナルティ, 既定0.6) / BAND(遷移帯 bins, 既定5)
実行: .venv/Scripts/python.exe v104/viterbi_leg_v104.py"""
import os, sys, io, time, pickle
try:
    sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
except Exception:
    pass
from pathlib import Path
import numpy as np, pyarrow.parquet as pq

V104 = Path(__file__).resolve().parent
PROJ = V104.parent
ART = V104 / "artifacts_v104"
SIM = pickle.loads((PROJ / "v97" / "artifacts_v97" / "sim_grfree_v97.pkl").read_bytes())
SF = pickle.loads((ART / "seqfeat_v104_cf.pkl").read_bytes())
y_all = np.load(ART / "l5_v104baseg-1_oof.npz")["y"]; N = len(y_all)
# lk(last_known_tvt)を train_combo から global 行整列で
_c = pq.read_table(PROJ / "v100" / "artifacts_v100" / "data" / "train_combo.parquet", columns=["id", "last_known_tvt"])
lk_all = _c.column("last_known_tvt").to_numpy(zero_copy_only=False).astype(np.float64)   # 行順=global
LAM = float(os.environ.get("LAM", "0.6")); BAND = int(os.environ.get("BAND", "5"))
ASIG = float(os.environ.get("ASIG", "9.0"))       # ★錨prior: st(GRフリー錨)へ引き戻す(PF同値 sig~9ft)。生emissionのdrift防止=研究の「制約必須」
OSUF = os.environ.get("VIT_OSUF", "")             # ★変種別に出力分離
BACKWARD = os.environ.get("VIT_BWD", "0") == "1"  # ★逆方向(toe→heel)MAP=誤差蓄積が別=脱相関変種
NB = 181; CENTER = 90; STEP = 0.5
_ANCHOR_PEN = -0.5 * (((np.arange(NB) - CENTER) * STEP) / ASIG) ** 2   # (181,)
print(f"[viterbi] LAM={LAM} BAND={BAND} ASIG={ASIG} wells={len(SIM)}", flush=True)


def viterbi(logE):
    """logE:(n,181) 各行の log尤度。遷移= -LAM*|Δbin|(|Δbin|<=BAND のみ)。MAP path(bin index)を返す。"""
    n, B = logE.shape
    dp = logE[0].copy(); bp = np.zeros((n, B), np.int16)
    offs = np.arange(-BAND, BAND + 1)
    for t in range(1, n):
        prev = dp
        # 各状態 b の best = max_{b' in b-band..b+band} prev[b'] - LAM*|b-b'|
        best = np.full(B, -1e18); arg = np.zeros(B, np.int64)
        for o in offs:
            src = np.clip(np.arange(B) + o, 0, B - 1)
            cand = prev[src] - LAM * abs(o)
            upd = cand > best
            best[upd] = cand[upd]; arg[upd] = src[upd]
        dp = best + logE[t]; bp[t] = arg.astype(np.int16)
    path = np.zeros(n, np.int64); path[-1] = int(np.argmax(dp))
    for t in range(n - 1, 0, -1):
        path[t - 1] = bp[t, path[t]]
    return path


def rmse(p):
    m = np.isfinite(p) & np.isfinite(y_all)
    return float(np.sqrt(np.mean((p[m] - y_all[m]) ** 2)))


def main():
    pred = np.full(N, np.nan)
    t0 = time.time(); nn = 0
    for w, d in SIM.items():
        if w not in SF:
            continue
        sim = np.asarray(d["sim"], np.float64); st = np.asarray(d["st"], np.float64)
        rows = SF[w]["rows"]
        if len(sim) != len(rows):
            continue
        logE = np.log(np.clip(sim, 1e-8, None)) + _ANCHOR_PEN[None, :]   # ★錨prior 付き
        if BACKWARD:
            path = viterbi(logE[::-1])[::-1]      # ★逆方向 MAP
        else:
            path = viterbi(logE)
        tvt = st + (path - CENTER) * STEP
        pred[rows] = tvt - lk_all[rows]           # δ = pred_TVT - lk
        nn += 1
        if nn % 150 == 0:
            print(f"  {nn}井 ({time.time()-t0:.0f}s) 途中OOF={rmse(pred):.4f}", flush=True)
    r = rmse(pred)
    print(f"[viterbi] OOF = {r:.4f}  (L5 5.5549 / TCN 6.01)  cover={100*np.isfinite(pred).mean():.0f}%", flush=True)
    np.savez(ART / f"viterbi{OSUF}_oof.npz", y=y_all, s0=pred)
    # L5相関 + ens
    l5 = np.load(PROJ / "v103/artifacts_v103/l5_v103all_oof.npz")
    import glob
    def l5l(f):
        e = np.load(f); return e["l5_mean"] if "l5_mean" in e.files else np.mean([e[k] for k in e.files if k.startswith("l5_s")], 0)
    L5 = np.nanmean(np.stack([l5l(str(PROJ/"v103/artifacts_v103/l5_v103all_oof.npz"))] +
                             [l5l(f) for f in glob.glob(str(PROJ/"v103/artifacts_v103/l5_v103g*k*_oof.npz"))], 0), 0)
    m = np.isfinite(pred) & np.isfinite(L5) & np.isfinite(y_all)
    c = np.corrcoef(pred[m], L5[m])[0, 1]
    ws = np.linspace(0, 0.6, 61); sc = [rmse(w * pred + (1 - w) * L5) for w in ws]; bi = int(np.argmin(sc))
    print(f"[viterbi] L5相関={c:.4f}  L5grid+viterbi: w={ws[bi]:.2f} -> {sc[bi]:.4f} (Δ{sc[bi]-rmse(L5):+.4f})", flush=True)


if __name__ == "__main__":
    main()
