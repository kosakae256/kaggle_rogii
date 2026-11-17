# -*- coding: utf-8 -*-
"""
v78準備 — v77の錨/P5/P6出力を「チェーン特徴」形式にエクスポート(train 773井)
出力: v77/artifacts_v77/v78_features_train.pkl
  {wid: dict(anc_tvt, anc_sig, anc_dd, p5a_off, p6e_off, p6e_postd, p5e_off, p6a_off,
             dis_56, dis_5a5e, blend4_tvt)}  … 全てeval行aligned(float32)
- off系は「錨からのオフセット」(pred−anc)。チェーン側はPF系特徴と併置して消費。
- test側は同じ生成器を隠しtest向けに再実行する(K16場=train全体, LOO不要)。
"""
import io, sys, pickle
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
from pathlib import Path
import numpy as np

P0 = Path(r"C:\Users\kosaka256\Documents\rogii_claude")
ART = P0 / "v77" / "artifacts_v77"
anc = pickle.loads((ART / "k16_anchor_train.pkl").read_bytes())
p5a = pickle.loads((ART / "p5_out_dip03_773w.pkl").read_bytes())
p6a = pickle.loads((ART / "p6_out_lp13_773w.pkl").read_bytes())
p5e = pickle.loads((ART / "p5_out_dip03emt05_773w.pkl").read_bytes())
p6e = pickle.loads((ART / "p6_out_lp13emt015_773w.pkl").read_bytes())
ancf = pickle.loads((ART / "k16_anchor_train_ftheta.pkl").read_bytes())
p5f = pickle.loads((ART / "p5_out_dip03fth_773w.pkl").read_bytes())
p6f = pickle.loads((ART / "p6_out_lp13e015fth_773w.pkl").read_bytes())
W = dict(pf=0.52, anc=0.12, p5a=0.19, p6e=0.17)  # 4脚blend重み(773fit)。pf成分はチェーン側で合成

out = {}
for wid in p5a:
    if any(wid not in x for x in (anc, p6a, p5e, p6e, ancf, p5f, p6f)): continue
    A = anc[wid]["tvt"]
    T = min(len(A), len(ancf[wid]["tvt"]), *[len(x[wid]["pred"]) for x in (p5a, p6a, p5e, p6e, p5f, p6f)])
    A = A[:T].astype(np.float32)
    f = dict(anc_tvt=A,
             anc_sig=anc[wid]["sig"][:T].astype(np.float32),
             anc_dd=anc[wid]["dd"][:T].astype(np.float32),
             p5a_off=(np.asarray(p5a[wid]["pred"][:T], np.float32) - A),
             p6a_off=(np.asarray(p6a[wid]["pred"][:T], np.float32) - A),
             p5e_off=(np.asarray(p5e[wid]["pred"][:T], np.float32) - A),
             p6e_off=(np.asarray(p6e[wid]["pred"][:T], np.float32) - A))
    pd_ = p6e[wid].get("postd")
    f["p6e_postd"] = (np.asarray(pd_[:T], np.float32) if pd_ is not None else np.zeros(T, np.float32))
    f["ancf_tvt"] = ancf[wid]["tvt"][:T].astype(np.float32)
    f["p5f_off"] = (np.asarray(p5f[wid]["pred"][:T], np.float32) - A)
    f["p6f_off"] = (np.asarray(p6f[wid]["pred"][:T], np.float32) - A)
    f["dis_56"] = np.abs(f["p5a_off"] - f["p6e_off"])
    f["dis_5a5e"] = np.abs(f["p5a_off"] - f["p5e_off"])
    # blend4のPF以外成分(チェーンでPF予測と合成する場合: pred = wpf·PF + rest)
    rest = 0.13 * f["ancf_tvt"] + 0.18 * (A + f["p5a_off"]) + 0.16 * (A + f["p6f_off"])
    f["blend4_rest_tvt"] = rest.astype(np.float32)
    out[wid] = f
fp = ART / "v78_features_train.pkl"
fp.write_bytes(pickle.dumps(out, protocol=4))
sz = fp.stat().st_size / 1e6
print(f"[saved] {fp.name}  {len(out)}井  {sz:.0f}MB  keys={list(next(iter(out.values())))}")
