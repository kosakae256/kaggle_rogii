# -*- coding: utf-8 -*-
"""
v78 P24b — フィールド適応θのsplit-half検証 + field-θ錨pkl生成
==============================================================
①検証: 各適応クラスタ(≥30井)を井戸偶奇で2分し、片側でθ*を選び反対側でLOO評価(交差)。
  選択バイアス抜きの正味利得を測る。
②生成: 合格θでクラスタ別θの錨pkl(k16_anchor_train_ftheta.pkl)を保存(σ(dd)も再較正)。
使い方: python v77/p24b_fieldtheta_val_v77.py
"""
import os, io, sys, glob, pickle, time
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
from pathlib import Path
import numpy as np, pandas as pd

exec(io.open(Path(r"C:\Users\kosaka256\Documents\rogii_claude") / "v77" / "p24_fieldtheta_v77.py",
             encoding="utf-8").read().split("tot_base, tot_best")[0].replace(
    'CLTH = float(os.environ.get("CLTH", "30000"))', 'CLTH = 4000.0'))
# ↑ p24の前半(wells/クラスタ/loo_errs定義)を再利用(CLTH=4000固定)

THETAS_F = np.arange(0, 180, 10.0)

print("\n== split-half検証(θ選択を反対半で評価) ==")
tot_b, tot_v = [], []
theta_final = {}
for c_ in range(cc):
    sub = [w for w in wells if cid[w["wi"]] == c_]
    theta_final[c_] = 118.4
    if len(sub) < 30: continue
    base = loo_errs(118.4, sub)
    if base is None: continue
    rb = float(np.sqrt(np.mean(base ** 2)))
    gains = []
    for par in (0, 1):
        selh = [w for i, w in enumerate(sub) if i % 2 == par]
        evalh = [w for i, w in enumerate(sub) if i % 2 != par]
        bb = loo_errs(118.4, selh)
        if bb is None: continue
        best = (float(np.sqrt(np.mean(bb ** 2))), 118.4)
        for th in THETAS_F:
            e = loo_errs(float(th), selh)
            if e is None: continue
            r = float(np.sqrt(np.mean(e ** 2)))
            if r < best[0]: best = (r, float(th))
        ev_b = loo_errs(118.4, evalh); ev_t = loo_errs(best[1], evalh)
        if ev_b is None or ev_t is None: continue
        gains.append((float(np.sqrt(np.mean(ev_b**2))), float(np.sqrt(np.mean(ev_t**2))), best[1]))
    if not gains: continue
    gb = np.mean([g[0] for g in gains]); gt = np.mean([g[1] for g in gains])
    ths = [g[2] for g in gains]
    ok = gt < gb - 0.02 and (max(ths) - min(ths) <= 30 or len(set(ths)) == 1)
    print(f"cluster{c_}({len(sub)}井): 反対半評価 118.4={gb:.3f} → θ*={ths}={gt:.3f}  {'★採用' if ok else '不採用(不安定/利得なし)'}")
    if ok:
        # 最終θはクラスタ全体LOOで再選択
        best = (rb, 118.4)
        for th in THETAS_F:
            e = loo_errs(float(th), sub)
            if e is None: continue
            r = float(np.sqrt(np.mean(e ** 2)))
            if r < best[0]: best = (r, float(th))
        theta_final[c_] = best[1]

print("\n== field-θ錨pkl生成(場=クエリ側θで全域再構築=座標系一貫) ==")
res_by_bin = [[] for _ in range(NBIN)]
errs_all = []
tmp = {}
def build_field(th):
    f = []
    for wd in wells:
        proj = np.cos(np.radians(wd["az"] - th))
        for j in range(K):
            if abs(proj[j]) > FIELD_MIN_PROJ:
                f.append((wd["mid"][j, 0], wd["mid"][j, 1], wd["c"][j] / proj[j], wd["wi"]))
    return np.array(f)
fields = {th: build_field(th) for th in sorted(set(theta_final.values()))}
for wd in wells:
        th = theta_final[cid[wd["wi"]]]
        fld = fields[th]
        proj = np.cos(np.radians(wd["az"] - th))
        keep = fld[:, 3] != wd["wi"]
        fx, fy, fD = fld[keep, 0], fld[keep, 1], fld[keep, 2]
        Dh = np.zeros(K); dd = np.full(K, 1e6)
        if len(fx) >= KNN_K:
            for j in range(K):
                d2 = (fx - wd["mid"][j, 0]) ** 2 + (fy - wd["mid"][j, 1]) ** 2
                sel = np.argpartition(d2, KNN_K)[:KNN_K]
                wgt = np.exp(np.maximum(-d2[sel] / (2 * KNN_H ** 2), -700))
                Dh[j] = np.sum(wgt * fD[sel]) / np.sum(wgt)
                dd[j] = np.sqrt(np.median(d2[sel]))
        g = wd["ndz"] + (Dh * proj)[wd["segid"]]
        g = np.where(np.abs(proj[wd["segid"]]) < GATE, 0.0, g)
        kstep = KAPPA[np.digitize(dd, KBINS[1:-1])][wd["segid"]]
        pred = wd["ti_s"] + np.cumsum(kstep * g)
        true = wd["ti_s"] + wd["R0"]
        e = pred - true
        dd_row = dd[wd["segid"]]
        for b in range(NBIN):
            mm = (np.digitize(dd_row, KBINS[1:-1]) == b) & np.isfinite(e)
            if mm.any(): res_by_bin[b].extend(e[mm].tolist())
        errs_all.append(e[np.isfinite(e)])
        tmp[wd["wid"]] = dict(tvt=pred.astype(np.float64), dd=dd_row.astype(np.float32), s=wd["s"])
sig_bin = np.array([np.sqrt(np.mean(np.array(r) ** 2)) if len(r) > 100 else 30.0 for r in res_by_bin])
for wid in tmp:
    tmp[wid]["sig"] = sig_bin[np.digitize(tmp[wid]["dd"], KBINS[1:-1])].astype(np.float32)
e = np.concatenate(errs_all)
print(f"field-θ錨 LOO pooled = {np.sqrt(np.mean(e**2)):.4f} (base 9.709)")
print("θ採用クラスタ:", {c: t for c, t in theta_final.items() if t != 118.4})
fp = ART / "k16_anchor_train_ftheta.pkl"
fp.write_bytes(pickle.dumps(tmp, protocol=4))
print(f"[saved] {fp.name} ({len(tmp)}井)")

# blend評価 vs 素PF
bare = pickle.loads((P0 / "v73" / "artifacts_v73" / "seed_dump_bare0.pkl").read_bytes())
LS = 10.536319667317654
em, ep = [], []
for wid, d in bare.items():
    if wid not in tmp: continue
    ll = d["ll"].astype(np.float64)
    ww = np.exp((ll - ll.max()) / LS); ww /= ww.sum()
    epf = (ww[:, None] * d["err"].astype(np.float64)).sum(0)
    tv = d["true"]; a = tmp[wid]["tvt"]
    T = min(len(a), len(tv), len(epf))
    ea = a[:T] - tv[:T]; pw = epf[:T]
    mm = np.isfinite(ea) & np.isfinite(pw)
    em.append(ea[mm]); ep.append(pw[mm])
em = np.concatenate(em); ep = np.concatenate(ep)
sm = np.sqrt(np.mean(em ** 2)); sp = np.sqrt(np.mean(ep ** 2)); rho = np.corrcoef(em, ep)[0, 1]
best = min((float(np.sqrt(np.mean(((1 - w) * ep + w * em) ** 2))), float(w)) for w in np.linspace(0, 1, 101))
print(f"[blend] fθ錨={sm:.4f} ρ={rho:.3f} w={best[1]:.2f} → {best[0]:.4f} (素PF{sp:.3f}比 {best[0]-sp:+.4f}; base錨blend 7.254)")
