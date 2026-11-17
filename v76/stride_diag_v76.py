# -*- coding: utf-8 -*-
"""
v76 Phase D — STRIDE(doc2)のデータ性質クレームを我々のデータ/OOFで追試(docs/v76.md)
D1 formation錨(1.4) / D2 上流累積誤差→下流(2) / D3 難易度予測子(2) / D4 自己一致=情報ゼロ(5)
/ D5 bad-tail坑井比較 / D6 区分線形の区間数+h_len prior fit。
入力: v73/artifacts_v73/seed_dump_champ2.pkl(err(S,T)=ps−true, ll, true) + train CSV(ANCC..BUDA)。
使い方: python v76/stride_diag_v76.py
"""
import os, io, sys, glob, pickle
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr

P0 = Path(r"C:\Users\kosaka256\Documents\rogii_claude")
TRAIN = P0 / "rogii-wellbore-geology-prediction" / "train"
DUMP = P0 / "v73" / "artifacts_v73" / "seed_dump_champ2.pkl"
LS = 10.536319667317654
D = pickle.loads(DUMP.read_bytes())
wells = sorted(D.keys())
print(f"dump: {len(wells)}井 (champ2=structgate+sim, OOF7.107)")


def wsoft(ll):
    m = np.isfinite(ll); mx = np.nanmax(ll[m]) if m.any() else 0.0
    lw = np.where(m, ll - mx, -np.inf)
    ww = np.exp(lw / LS); s = ww.sum()
    return ww / s if s > 1e-300 else np.full(len(ll), 1.0 / len(ll))


# 共通: 井戸ごとの合成誤差列・per-well指標・CSV由来の系列
E = {}; PW = {}
for w in wells:
    d = D[w]; err = d["err"].astype(np.float32); ll = d["ll"]
    ww = wsoft(ll); ec = (ww[:, None] * err).sum(0)
    fin = np.isfinite(ec)
    if fin.sum() < 30: continue
    off = np.nanmedian(err - ec[None, :], axis=1)
    E[w] = dict(ec=ec, fin=fin, seed_spread=float(np.nanstd(off)),
                rmse=float(np.sqrt(np.nanmean(ec[fin] ** 2))), bias=float(np.nanmean(ec[fin])),
                n=int(fin.sum()))
wells = [w for w in wells if w in E]

FORMS = ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]
print("\n=== D1: formation錨(1.4)の再現 ===")
span_list = []; thick_list = []; form_of = {}; grcorr = {}; ratejump = {}; seglens = []; segrms = []
for w in wells:
    hw = pd.read_csv(TRAIN / f"{w}__horizontal_well.csv")
    kn = hw[hw["TVT_input"].notna()]; ev = hw[hw["TVT_input"].isna()]
    tvt = ev["TVT"].to_numpy(float); md = ev["MD"].to_numpy(float); z = ev["Z"].to_numpy(float)
    m = np.isfinite(tvt)
    if m.sum() < 30: continue
    span_list.append(float(np.nanmax(tvt[m]) - np.nanmin(tvt[m])))
    # formation同一性: 各行の геologic位置 = TVT+Z が各formation面のどこにあるか(面はhwの列=行ごとの面深度)
    S = tvt + z
    tops = {f: hw[f].to_numpy(float)[hw.index.get_indexer(ev.index)] for f in FORMS if f in hw.columns}
    if len(tops) == len(FORMS):
        # 行ごとの所属: S が [ANCC, ASTNU) → ANCC 帯 …(深度は負方向なので符号に注意: Z/面は負値)
        arr = np.stack([tops[f] for f in FORMS], 1)   # (T,6) 各面のS(=深度)値
        # 厚み統計(井戸中央値): 隣接面差
        thick_list.append(np.nanmedian(np.abs(np.diff(arr, axis=1)), axis=0))
        # 所属formation(最も近い上側面): index of last top >= S (深度負系なので比較は値で)
        fi = (arr >= S[:, None]).sum(1)               # 粗い層番号
        form_of[w] = fi
    # D3-①: 真面 rate の隣接ジャンプ最大(21行窓の傾き差)
    sr = pd.Series(np.gradient(S, md)).rolling(21, center=True, min_periods=1).median().to_numpy()
    ratejump[w] = float(np.nanmax(np.abs(np.diff(sr[::50])))) if len(sr) > 100 else np.nan
    # D3-②: 既知区間 GR↔typewell 相関
    tw = pd.read_csv(TRAIN / f"{w}__typewell.csv").sort_values("TVT")
    kt = kn["TVT_input"].to_numpy(float); kg = kn["GR"].to_numpy(float)
    mm = np.isfinite(kt) & np.isfinite(kg)
    if mm.sum() > 30:
        eg = np.interp(kt[mm], tw["TVT"].to_numpy(float), tw["GR"].to_numpy(float))
        grcorr[w] = float(np.corrcoef(kg[mm], eg)[0, 1])
    # D6: 区分線形fit(貪欲: 残差>2ftで分割)
    y = S[m]; x = md[m]
    segs = 1; i0 = 0; err2 = []
    for i in range(30, len(x)):
        a, b = np.polyfit(x[i0:i + 1], y[i0:i + 1], 1)
        r = y[i0:i + 1] - (a * x[i0:i + 1] + b)
        if np.abs(r).max() > 2.0:
            segs += 1; err2.extend(r[:-1].tolist()); i0 = i
    seglens.append(segs)

sp = np.array(span_list); th = np.vstack([t for t in thick_list])
print(f"  lateral TVTスパン: 中央 {np.median(sp):.1f}ft (彼: 26ft)  / formation厚 中央 {np.median(th):.0f}ft帯 (彼: 40-116ft)")
# D1-①: formation同一性oracle(井戸×層番号ごとの平均誤差を引く)
se0 = se1 = n0 = 0
for w in wells:
    if w not in form_of: continue
    ec = E[w]["ec"]; fin = E[w]["fin"]; fi = form_of[w][:len(ec)]
    corr = ec.copy()
    for f in np.unique(fi):
        s = fin & (fi == f)
        if s.sum() > 10: corr[s] = ec[s] - np.nanmean(ec[s])
    se0 += np.nansum(ec[fin] ** 2); se1 += np.nansum(corr[fin] ** 2); n0 += fin.sum()
print(f"  formation同一性oracle: {np.sqrt(se0/n0):.3f} → {np.sqrt(se1/n0):.3f} (彼: 10.68→7.36相当の帯)")
bias = np.array([E[w]["bias"] for w in wells if w in form_of])
nseg = np.array([len(np.unique(form_of[w])) for w in wells if w in form_of])
print(f"  lateralが跨ぐ層数: 1層のみ={np.mean(nseg==1)*100:.0f}% (彼: ほぼ跨がない)")

print("\n=== D2: 上流累積誤差→下流RMSE(彼: Spearman0.45) ===")
ups = []; dns = []
for w in wells:
    ec = E[w]["ec"]; fin = E[w]["fin"]; T = len(ec)
    if fin.sum() < 200: continue
    h = T // 2
    e1 = ec[:h][fin[:h]]; e2 = ec[h:][fin[h:]]
    if len(e1) < 50 or len(e2) < 50: continue
    ups.append(abs(np.nanmean(e1))); dns.append(np.sqrt(np.nanmean(e2 ** 2)))
print(f"  前半平均誤差|→後半RMSE: Spearman {spearmanr(ups, dns).statistic:.3f}  (n={len(ups)})")

print("\n=== D3: 難易度予測子 ===")
rj = [(ratejump.get(w, np.nan), E[w]["rmse"]) for w in wells]
rj = [(a, b) for a, b in rj if np.isfinite(a)]
a = np.array([x for x, _ in rj]); b = np.array([y for _, y in rj])
q = np.quantile(a, [0.8]); hi = b[a >= q[0]].mean(); lo = b[a <= np.quantile(a, 0.2)].mean()
print(f"  ①真面rate急変: 荒い1/5={hi:.2f} vs 静か1/5={lo:.2f} (彼: 6.52 vs 4.77) Spearman={spearmanr(a,b).statistic:.3f}(彼0.08)")
gc_ = [(grcorr.get(w, np.nan), E[w]["rmse"]) for w in wells]
gc_ = [(x, y) for x, y in gc_ if np.isfinite(x)]
a2 = np.array([x for x, _ in gc_]); b2 = np.array([y for _, y in gc_])
print(f"  ②既知区間GR↔tw相関→難易度: Spearman={spearmanr(a2,b2).statistic:.3f} (彼: -0.01=予測しない)")

print("\n=== D4: 自己一致(同一証拠=32seedばらつき)は誤差を予測するか(彼: ≈0.00) ===")
ss = np.array([E[w]["seed_spread"] for w in wells]); rr = np.array([E[w]["rmse"] for w in wells])
print(f"  seed間offsetばらつき→井戸RMSE: Spearman={spearmanr(ss, rr).statistic:.3f}")
print(f"  (注: 彼の主張は『符号/方向は分からない』。検出はできる場合がある)")
sb = np.array([abs(E[w]['bias']) for w in wells])
print(f"  seedばらつき→|井戸バイアス|(方向の手掛かり): Spearman={spearmanr(ss, sb).statistic:.3f}")

print("\n=== D5: bad-tail坑井の直接比較(彼の最終値: 40.16/23.05/23.75/33.55) ===")
for w in ["86454a6f", "bb682ebd", "2fd68f7b", "91db7070"]:
    print(f"  {w}: 我々(champ2 PF単体)={E[w]['rmse']:.2f}" if w in E else f"  {w}: dumpに無し")

print("\n=== D6: 区分線形の区間数(彼: 中央~19区間で0.8ft) ===")
print(f"  貪欲PWL(閾2ft): 区間数 中央 {np.median(seglens):.0f} / 平均 {np.mean(seglens):.1f}")
print("\n[saved なし(診断のみ)]")
