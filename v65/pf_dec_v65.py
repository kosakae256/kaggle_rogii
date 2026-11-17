# -*- coding: utf-8 -*-
"""
v65 PF 間引き(MD粗化)ラッパー — 既存 pf_engine_v65.py に一切触れず、
make_inputs の出力(eval行系列 md/z/gr/anc/ancs)を MD=DEC で bin集約して PF を回し、
予測軌跡を**全解像度に補間して採点**する(= 本物の MD-1 メトリクで比較可能)。
=====================================================================
狙い(検証/gr_typewell_md_coarsen.py の結論): 照合の相関利得は MA で既に取れ、PF は生GRを好む。
残る未探索レバーは "decimation そのもの"(ステップ数削減)=
  - PF の overconfidence 緩和(冗長な相関更新を1/DECに)
  - 速度 ~DEC倍
これを RMSE 直で測るのがこのラッパーの目的。current v65 とは別 study/db。

env: DEC(10 固定粗化率) AGG(mean=bin平均 | sub=間引き。md/z/anc/ancs の集約法。GR は常に mean=denoise)
     PHASE(1) 位相アンサンブル数。>1 で bin境界を0..DEC-1にずらした decimated PF を PHASE本平均
              (全行を使い切る+各メンバーは overconfident でない=較正済みフィルタの平均)。コスト≈PHASE/DEC×MD-1。
使い方: tune_pf_dec_v65.py から dec.overall_rmse_on_dec / dec.predict_errs_dec を呼ぶ。
"""
import os, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import pf_engine_v65 as pf

DEC = int(os.environ.get("DEC", "10"))
AGG = os.environ.get("AGG", "mean")     # md/z/anc/ancs: mean or sub
GRAGG = os.environ.get("GRAGG", "mean") # GR: mean(denoise) or sub(位相ごとに別行=disjoint diversity)
PHASE = int(os.environ.get("PHASE", "1"))   # 位相アンサンブル本数(1=単相=現状, DEC=全位相)


def _agg(x, s, how, phase=0):
    """長さ n を s ごとに集約。phase で bin境界を右にずらす(先頭は部分bin)。how=mean or sub(中央点)。"""
    x = np.asarray(x, float); n = len(x)
    cuts = sorted(set([0, n] + list(range((phase % s) or s, n, s))))   # phase=0→[0,s,2s..], phase=p→[0,p,p+s..]
    out = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        seg = x[a:b]
        out.append(float(seg.mean()) if how == "mean" else float(seg[(b - a) // 2]))
    return np.array(out)


def make_inputs_dec(cache, P, phase=0):
    """pf.make_inputs の出力を MD=DEC・指定 phase で集約。full_md/dec_md 付き(補間採点用)。"""
    inp = pf.make_inputs(cache, P)
    md = np.asarray(inp["md"], float)
    if len(md) < DEC * 2:                 # 短い坑井は decimate しない(補間元が足りない)
        di = dict(inp); di["full_md"] = md; di["dec_md"] = md; return di
    di = dict(md=_agg(md, DEC, AGG, phase), z=_agg(inp["z"], DEC, AGG, phase), gr=_agg(inp["gr"], DEC, GRAGG, phase),
              gg=inp["gg"], gmin=inp["gmin"], gst=inp["gst"], gs=inp["gs"],
              ls=inp["ls"], ir=inp["ir"], N=inp["N"],
              anc=_agg(inp["anc"], DEC, "mean", phase), ancs=_agg(inp["ancs"], DEC, "mean", phase))
    di["full_md"] = md; di["dec_md"] = di["md"]
    return di


def _traj_one_phase(P, subset, seed, n_seeds, chunk, phase):
    """1位相の decimated PF → 各坑井の全解像度軌跡 {w: full_traj}。"""
    out = {}
    for c0 in range(0, len(subset), chunk):
        sub = subset[c0:c0 + chunk]
        dinps = [make_inputs_dec(pf.CACHE[w], P, phase) for w in sub]
        preds = pf.run_forward_ens(dinps, P, pf.DEVICE, seed, n_seeds)
        for k, w in enumerate(sub):
            dec_md = dinps[k]["dec_md"]; full_md = dinps[k]["full_md"]; dtraj = preds[k]
            out[w] = dtraj if len(dec_md) == len(full_md) else np.interp(full_md, dec_md, dtraj)
    if pf.DEVICE == "cuda":
        import torch; torch.cuda.synchronize()
    return out


def predict_errs_dec(P, subset, seed, n_seeds, chunk):
    """PHASE 本の位相を平均 → 全解像度の誤差(pred−tvt_true)。PHASE=1 は単相(現状)。"""
    nph = max(1, min(PHASE, DEC))
    acc = None
    for ph in range(nph):
        traj = _traj_one_phase(P, subset, seed, n_seeds, chunk, ph)
        if acc is None:
            acc = {w: traj[w].copy() for w in subset}
        else:
            for w in subset:
                acc[w] += traj[w]
    return {w: acc[w] / nph - pf.CACHE[w]["tvt_true"] for w in subset}


def overall_rmse_on_dec(P, subset, seed, n_seeds, chunk):
    errs = predict_errs_dec(P, subset, seed, n_seeds, chunk)
    e = np.concatenate([errs[w] for w in subset]); e = e[np.isfinite(e)]
    if e.size == 0 or not np.isfinite(e).all():
        return 1e3
    return float(np.sqrt(np.mean(e ** 2)))
