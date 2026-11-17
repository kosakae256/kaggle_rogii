# -*- coding: utf-8 -*-
"""
v89 D1/D2 — 枝証拠の判別力を「学習前に」測る決定的診断(struct-free)。
=================================================================================================
主張(docs/v89.md): 頭打ち(枝判別0.54)の真因は「対抗枝(±1サイクル)を明示的に評価していない」から。
検証: PF(tw)が実際にコミットした軌跡 p を得て、区間ごとに
  - サイクル周期 λ を typewell GR 自己相関から推定(D2)
  - 対抗枝 = p を {−λ, 0, +λ}(必要なら ±2λ)平行移動
  - 各枝の証拠 = tw / self(v88多ソース) の GR-misfit を区間集約
  - argmin-misfit の枝が **真の枝**(=round((true−p)/λ))に一致する率を測る(D1)
比較軸:
  - base = 「常に0(=PFを信じる)」の正解率
  - pick(tw) / pick(tw+self) = 多ソースmisfitで枝を選んだ時の正解率
  - PF正解区間を壊さないか / PF誤り区間を訂正できるか(混同)
→ pick(tw+self) が base と 0.54 を明確に上回れば、**入力定式化(枝証拠)は正しい**=E1へ。
使い方: NW=200 python v89/diag_branch_v89.py
"""
import os, io, sys, time, json
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
from pathlib import Path
import numpy as np

PROJECT = Path(r"C:\Users\kosaka256\Documents\rogii_claude")
sys.path.insert(0, str(PROJECT / "v65")); sys.path.insert(0, str(PROJECT / "v88"))
os.environ["V65_TWSRC"] = "tw+self+nbr"
import pf_engine_v65 as base
import pf_engine_v88 as v88
import warmup_v65 as wu

NW    = int(os.environ.get("NW", "200")) or None
SEED  = int(os.environ.get("SEED", "20251707"))
NSEED = 32
SEG   = int(os.environ.get("SEG", "300"))
CHUNK = int(os.environ.get("CHUNK", "100"))
OUT   = PROJECT / "v89" / "artifacts_v89"; OUT.mkdir(parents=True, exist_ok=True)

P = v88.full_params(dict(wu.R0_PARAMS)); P["use_anchor"] = False; P["n_seeds"] = NSEED
P["tw_source"] = "tw"; P["v88_src"] = "tw+self"; P["v88_beta_self"] = 1.0; P["v88_self_fill"] = 20.0
POW = float(P["gr_power"])
SHIFTS = np.array([-2, -1, 0, 1, 2], float)   # ×λ


def est_lambda(gg, gst):
    """typewell参照格子 gg(TVT等間隔, 刻みgst)の自己相関から周期λ(ft)を推定。8-30ft内の第1ピーク。"""
    x = np.asarray(gg, float); x = x - np.nanmean(x)
    n = len(x)
    if n < int(40 / gst):
        return 16.0
    lo = int(round(8.0 / gst)); hi = min(int(round(30.0 / gst)), n - 1)
    ac = np.array([np.nanmean(x[:n - L] * x[L:]) for L in range(1, hi + 1)])
    ac = ac / (np.nanmax(np.abs(ac)) + 1e-9)
    seg = ac[lo - 1:hi]
    if len(seg) < 3:
        return 16.0
    k = int(np.argmax(seg)) + lo
    return float(np.clip(k * gst, 10.0, 26.0))


def main():
    wells = base.list_wells()
    if NW: wells = wells[:NW]
    cache = {}
    for w in wells:
        c = base.build_cache(w)
        if c is not None and c["tvt_true"] is not None:
            cache[w] = c
    base.CACHE = cache; v88.CACHE = cache
    ALL = [w for w in wells if w in cache]
    print(f"[v89] wells={len(ALL)} seed={SEED} SEG={SEG}", flush=True)

    # tw PF 軌跡 p を得る(base engine, struct-free)
    Pc = base.full_params(dict(wu.R0_PARAMS)); Pc["use_anchor"] = False; Pc["n_seeds"] = NSEED; Pc["tw_source"] = "tw"
    pred = {}
    for c0 in range(0, len(ALL), CHUNK):
        sub = ALL[c0:c0 + CHUNK]
        inps = [base.make_inputs(cache[w], Pc) for w in sub]
        outs = base.run_forward_ens(inps, Pc, base.DEVICE, SEED, NSEED)
        for k, w in enumerate(sub):
            pred[w] = np.asarray(outs[k], float)
    print(f"[v89] tw PF done", flush=True)

    lam_list = []
    # 集計: 各枝法の正解率
    tot = 0; base_ok = 0
    pick = {"tw": 0, "tw+self": 0}
    # 混同: PF正解区間で壊さない率 / PF誤り区間で訂正率
    corr = {"tw": [0, 0], "tw+self": [0, 0]}   # [壊さない(PF正), 訂正(PF誤)]
    n_pfright = 0; n_pfwrong = 0
    for w in ALL:
        c = cache[w]
        x = v88.make_inputs(c, P)
        ref_tw = np.asarray(x["ch_ref"][0], float)
        ref_sf = np.asarray(x["ch_ref"][1], float) if x["ch_ref"].shape[0] > 1 else ref_tw
        msk_sf = np.asarray(x["ch_mask"][1], float) if x["ch_mask"].shape[0] > 1 else np.ones_like(ref_tw)
        obs = np.asarray(x["ch_obs"][0], float)
        sig_tw = float(x["ch_sig"][0]); sig_sf = float(x["ch_sig"][1]) if len(x["ch_sig"]) > 1 else sig_tw
        gmin, gst = x["gmin"], x["gst"]; G = len(ref_tw)
        tvt_g = gmin + gst * np.arange(G)
        lam = est_lambda(ref_tw, gst); lam_list.append(lam)
        p = pred[w]; tt = np.asarray(c["tvt_true"], float)
        z = np.asarray(c["z"], float)
        n = min(len(p), len(tt))
        for s0 in range(0, n - SEG + 1, SEG):
            sl = slice(s0, s0 + SEG)
            pp = p[sl]; tv = tt[sl]
            ok = np.isfinite(pp) & np.isfinite(tv)
            if ok.sum() < SEG // 2:
                continue
            pp = pp[ok]; tv = tv[ok]; ob = obs[sl][ok]
            d = float(np.median(tv - pp))              # datum offset (真−PF)
            true_b = int(np.clip(round(d / lam), -2, 2))
            tot += 1
            if true_b == 0: base_ok += 1
            pf_right = (true_b == 0)
            n_pfright += pf_right; n_pfwrong += (not pf_right)
            # 各枝の証拠
            m_tw = []; m_sf = []
            for sh in SHIFTS:
                hyp = pp + sh * lam
                r_tw = np.interp(hyp, tvt_g, ref_tw)
                r_sf = np.interp(hyp, tvt_g, ref_sf)
                cv = np.interp(hyp, tvt_g, msk_sf)
                d_tw = np.abs((ob - r_tw) / sig_tw) ** POW
                d_sf = np.abs((ob - r_sf) / sig_sf) ** POW
                m_tw.append(np.nanmean(d_tw))
                # self: 被覆のみ、非被覆は中立(=twと同じにして差を作らない)
                m_sf.append(np.nanmean(np.where(cv > 0.5, d_sf, d_tw)))
            m_tw = np.array(m_tw); m_sf = np.array(m_sf)
            for nm, prof in (("tw", m_tw), ("tw+self", m_tw + m_sf)):
                b = int(SHIFTS[int(np.argmin(prof))])
                if b == true_b:
                    pick[nm] += 1
                if pf_right and b == 0:
                    corr[nm][0] += 1
                if (not pf_right) and b == true_b:
                    corr[nm][1] += 1

    print(f"\n=== v89 D2 サイクル周期λ ({len(ALL)}井) 中央{np.median(lam_list):.1f}ft "
          f"[{np.percentile(lam_list,10):.0f},{np.percentile(lam_list,90):.0f}] ===")
    print(f"\n=== v89 D1 枝判別 ({tot}区間, 区間{SEG}行) ===")
    print(f"  base(常に0=PF信頼)   正解率 = {base_ok/tot*100:5.1f}%  (PF正{n_pfright}/誤{n_pfwrong})")
    for nm in ("tw", "tw+self"):
        keep = corr[nm][0] / max(n_pfright, 1)
        fix  = corr[nm][1] / max(n_pfwrong, 1)
        print(f"  pick({nm:8s})        正解率 = {pick[nm]/tot*100:5.1f}%  | PF正を保持 {keep*100:4.1f}% / PF誤を訂正 {fix*100:4.1f}%")
    print(f"\n  ★判定: pick(tw+self) が base と 0.54 を明確に超えれば入力定式化(枝証拠)は正しい → E1へ")
    json.dump(dict(lam_med=float(np.median(lam_list)), tot=tot, base=base_ok/tot,
                   pick_tw=pick["tw"]/tot, pick_twself=pick["tw+self"]/tot),
              io.open(OUT / "diag_branch.json", "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()
