# -*- coding: utf-8 -*-
"""
v88 実験ハーネス — 多チャネル尤度PF(GPU)の対照実験。
============================================================================
規約(ユーザー指示・厳守):
  - **n_seeds=32 固定**。struct錨は使わない(use_anchor=False)。GPU PFのみ。
  - **毎回 r0 対照を同一井戸・同一seed・同一条件で走らせ**、差分と誤差相関で判定。
  - 既定 NW=200(スクリーニング)。NW=0 で全773井。両方の結果を docs/v88.md に記録する。
使い方: NW=200 CFG=e0 python v88/exp_v88.py
"""
import os, io, sys, time, json, pickle
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
from pathlib import Path
import numpy as np

PROJECT = Path(r"C:\Users\kosaka256\Documents\rogii_claude")
sys.path.insert(0, str(PROJECT / "v65")); sys.path.insert(0, str(PROJECT / "v88"))
os.environ["V65_TWSRC"] = "tw+self+nbr"   # ★build_cacheにnbr_pairs/selfを作らせる。実際の使用ソースは v88_src で指定
import pf_engine_v65 as base
import pf_engine_v88 as v88
import warmup_v65 as wu

NW    = int(os.environ.get("NW", "200")) or None
SEED  = int(os.environ.get("SEED", "20251707"))       # 別seed(採用判断用)
NSEED = 32                                             # ★固定(ユーザー指示)
CHUNK = int(os.environ.get("CHUNK", "100"))
CFG   = os.environ.get("CFG", "e0")
OUT   = PROJECT / "v88" / "artifacts_v88"; OUT.mkdir(parents=True, exist_ok=True)

R0 = dict(wu.R0_PARAMS)


def _p(**kw):
    q = v88.full_params(dict(R0)); q["use_anchor"] = False; q["n_seeds"] = NSEED; q["tw_source"] = "tw"
    q.update(kw); return q


# ============================ 実験構成 ============================
def configs(name):
    if name == "e0":      # 回帰確認 + 基準
        return [("v88_raw(=base回帰)", _p())]
    if name == "e2":      # 多チャネル: LF/HF 分離
        C = []
        for wlf in (0.5, 1.0, 2.0, 4.0):
            C.append((f"raw1+lf{wlf}", _p(v88_w_raw=1.0, v88_w_lf=wlf)))
        for wlf in (1.0, 2.0):
            C.append((f"lf{wlf}+hf1(raw off)", _p(v88_w_raw=0.0, v88_w_lf=wlf, v88_w_hf=1.0)))
        return C
    if name == "e2b":     # LF平滑スケールの探索(最良重みで)
        C = []
        for rt, rm in ((2.0, 50.0), (4.0, 100.0), (8.0, 200.0), (4.0, 200.0), (8.0, 100.0)):
            C.append((f"lf(tvt{rt},md{rm})", _p(v88_w_raw=1.0, v88_w_lf=2.0, v88_lf_tvt=rt, v88_lf_md=rm)))
        return C
    if name == "e5":      # ★診断(E1)最良: tw+self 合成尤度。βスイープ + 回帰確認
        C = [("回帰:tw only", _p())]
        for b in (0.1, 0.25, 0.5, 1.0):
            C.append((f"tw+self β{b}", _p(v88_src="tw+self", v88_beta_self=b)))
        return C
    if name == "e5b":     # ★σ_self を tw較正スケールに直した版のβ/σスイープ
        C = []
        for b in (0.1, 0.25, 0.5, 1.0):
            C.append((f"σtw tw+self β{b}", _p(v88_src="tw+self", v88_beta_self=b, v88_self_sig="tw")))
        for sc in (1.5, 2.0):
            C.append((f"σtw×{sc} tw+self β0.5", _p(v88_src="tw+self", v88_beta_self=0.5,
                                                   v88_self_sig="tw", v88_self_sig_scale=sc)))
        return C
    if name == "e5c":     # ★粒子平均センタリング後の微小β探索(v51の W≈0.02 に倣う)
        C = [("回帰:tw only", _p())]
        for b in (0.02, 0.05, 0.075, 0.1, 0.15):
            C.append((f"tw+self β{b}", _p(v88_src="tw+self", v88_beta_self=b)))
        return C
    if name == "e5d":     # 最良β近傍で self被覆しきい値KC / nbr併用 を確認
        C = []
        for kc in (1.0, 2.0, 8.0):
            C.append((f"β0.05 KC{kc}", _p(v88_src="tw+self", v88_beta_self=0.05, v88_conf_kcnt=kc)))
        C.append(("β0.05 +nbr0.05", _p(v88_src="tw+self+nbr", v88_beta_self=0.05, v88_beta_nbr=0.05)))
        C.append(("nbr0.05 のみ", _p(v88_src="tw+nbr", v88_beta_nbr=0.05)))
        return C
    if name == "e5final":  # ★全773井確認: 200井で条件を満たした2構成 + 回帰
        C = [("回帰:tw only", _p())]
        for b in (0.02, 0.1):
            C.append((f"tw+self β{b}", _p(v88_src="tw+self", v88_beta_self=b)))
        return C
    if name == "e6":      # ★self被覆拡張(補間+距離減衰) と 修正版uniqueness
        C = [("回帰:tw only", _p())]
        for f in (2.0, 5.0, 10.0):
            C.append((f"tw+self β0.1 fill{f}", _p(v88_src="tw+self", v88_beta_self=0.1, v88_self_fill=f)))
        C.append(("tw+self β0.25 fill5", _p(v88_src="tw+self", v88_beta_self=0.25, v88_self_fill=5.0)))
        for a in (0.3, 0.6):
            C.append((f"uniq(修正) a{a}", _p(v88_uniq_w=a)))
        return C
    if name == "e7":      # ★self被覆拡張をさらに押す(脱相関脚の強化)
        C = [("回帰:tw only", _p())]
        for f in (10.0, 20.0, 40.0, 80.0):
            C.append((f"β0.1 fill{f:.0f}", _p(v88_src="tw+self", v88_beta_self=0.1, v88_self_fill=f)))
        for b in (0.05, 0.2):
            C.append((f"β{b} fill20", _p(v88_src="tw+self", v88_beta_self=b, v88_self_fill=20.0)))
        return C
    if name == "e7final":  # ★全773井確認: 200井で blend 最良域の3構成
        C = [("回帰:tw only", _p())]
        C.append(("β0.05 fill20", _p(v88_src="tw+self", v88_beta_self=0.05, v88_self_fill=20.0)))
        C.append(("β0.1 fill10", _p(v88_src="tw+self", v88_beta_self=0.1, v88_self_fill=10.0)))
        C.append(("β0.1 fill40", _p(v88_src="tw+self", v88_beta_self=0.1, v88_self_fill=40.0)))
        return C
    if name == "e8":      # ★合成尤度に合わせたPF再チューニング + 文脈項の極小重み
        BASE = dict(v88_src="tw+self", v88_beta_self=0.05, v88_self_fill=20.0)
        C = [("回帰:tw only", _p()), ("基準 β0.05fill20", _p(**BASE))]
        for k, v in (("likelihood_scale", 5.0), ("likelihood_scale", 20.0),
                     ("resamp", 0.10), ("resamp", 0.015),
                     ("gr_sig_mult", 2.0), ("gr_sig_mult", 4.0),
                     ("gr_power", 1.4), ("gr_power", 2.4),
                     ("pn", 0.010), ("n_particles", 800)):
            C.append((f"{k}={v}", _p(**{**BASE, k: v})))
        for wc in (0.02, 0.05, 0.1):
            C.append((f"+ctx k4 s200 w{wc}", _p(**BASE, v88_ctx_k=4, v88_ctx_span=200.0, v88_ctx_w=wc)))
        return C
    if name == "e9":      # ★pn(位置ノイズ)が効いた → 深掘り + 良かった設定の組合せ
        B = dict(v88_src="tw+self", v88_beta_self=0.05, v88_self_fill=20.0)
        C = [("回帰:tw only", _p())]
        for v in (0.008, 0.010, 0.015, 0.020, 0.030):
            C.append((f"pn{v}", _p(**B, pn=v)))
        C.append(("pn0.015+gp1.4", _p(**B, pn=0.015, gr_power=1.4)))
        C.append(("pn0.015+ctx0.05", _p(**B, pn=0.015, v88_ctx_k=4, v88_ctx_span=200.0, v88_ctx_w=0.05)))
        C.append(("pn0.015+N800", _p(**B, pn=0.015, n_particles=800)))
        C.append(("pn0.015 twのみ(self無)", _p(pn=0.015)))
        return C
    if name == "e9final":  # ★全773井: 最良候補の確定
        B = dict(v88_src="tw+self", v88_beta_self=0.05, v88_self_fill=20.0)
        C = [("回帰:tw only", _p())]
        C.append(("★pn0.01", _p(**B, pn=0.010)))
        C.append(("pn0.015+gp1.4", _p(**B, pn=0.015, gr_power=1.4)))
        C.append(("pn0.015+ctx0.05", _p(**B, pn=0.015, v88_ctx_k=4, v88_ctx_span=200.0, v88_ctx_w=0.05)))
        C.append(("対照:pn0.015 self無", _p(pn=0.015)))
        return C
    if name == "e10":     # ★self材料をeval帯に近いprefix末尾に限定(地質連続性は距離減衰)
        B = dict(v88_src="tw+self", v88_beta_self=0.05, v88_self_fill=20.0)
        C = [("回帰:tw only", _p()), ("基準(全prefix)", _p(**B))]
        for t in (500, 1000, 2000, 4000):
            C.append((f"tail{t}", _p(**B, v88_self_tail=t)))
        C.append(("tail1000 β0.1", _p(v88_src="tw+self", v88_beta_self=0.1, v88_self_fill=20.0, v88_self_tail=1000)))
        return C
    if name == "e10final":
        B = dict(v88_src="tw+self", v88_beta_self=0.05, v88_self_fill=20.0)
        return [("回帰:tw only", _p()), ("tail1000", _p(**B, v88_self_tail=1000))]
    if name == "e11":     # ★E2の正しい版: 参照のみTVT平滑・観測は生(領域不整合なし)
        C = [("回帰:tw only", _p())]
        for w in (0.1, 0.25, 0.5, 1.0):
            C.append((f"lfref(4ft) w{w}", _p(v88_w_lfref=w)))
        for t in (8.0, 16.0):
            C.append((f"lfref({t:.0f}ft) w0.25", _p(v88_w_lfref=0.25, v88_lf_tvt=t)))
        C.append(("lfref4 w0.25 + self脚", _p(v88_w_lfref=0.25, v88_src="tw+self",
                                              v88_beta_self=0.05, v88_self_fill=20.0)))
        return C
    if name == "e12":     # ★行ごと情報量重み(GR希少性)
        C = [("回帰:tw only", _p())]
        for g in (0.25, 0.5, 1.0):
            C.append((f"infow {g}", _p(v88_infow=g)))
        C.append(("infow0.5 + self脚", _p(v88_infow=0.5, v88_src="tw+self",
                                          v88_beta_self=0.05, v88_self_fill=20.0)))
        return C
    if name == "e12final":
        B = dict(v88_src="tw+self", v88_beta_self=0.05, v88_self_fill=20.0)
        return [("回帰:tw only", _p()),
                ("infow0.5+self", _p(**B, v88_infow=0.5)),
                ("infow0.25+self", _p(**B, v88_infow=0.25))]
    if name == "eseed":   # ★別seedでの再現性検証(推奨3脚)
        return [("回帰:tw only", _p()),
                ("tw+self β0.1", _p(v88_src="tw+self", v88_beta_self=0.1)),
                ("β0.05 fill20", _p(v88_src="tw+self", v88_beta_self=0.05, v88_self_fill=20.0))]
    if name == "e4":      # ★文脈マッチ(粒子rateでMD窓→TVT窓)
        C = []
        for k, span, wc in ((4, 200.0, 1.0), (8, 400.0, 1.0), (8, 800.0, 1.0),
                            (8, 400.0, 2.0), (8, 400.0, 0.5), (16, 800.0, 1.0)):
            C.append((f"ctx k{k} span{span:.0f} w{wc}", _p(v88_ctx_k=k, v88_ctx_span=span, v88_ctx_w=wc)))
        return C
    if name == "e3":      # ★E3 識別力(distinctiveness)重み: 自己相似な位置では尤度を鈍らせる
        C = [("回帰:tw only", _p())]
        for a in (0.3, 0.6, 1.0):
            C.append((f"uniq a{a}", _p(v88_uniq_w=a)))
        C.append(("uniq a0.6 lag15", _p(v88_uniq_w=0.6, v88_uniq_lag=15.0)))
        C.append(("uniq a0.6 win12", _p(v88_uniq_w=0.6, v88_uniq_win=12.0)))
        C.append(("uniq a0.6 + tw+self β0.1", _p(v88_uniq_w=0.6, v88_src="tw+self", v88_beta_self=0.1)))
        return C
    if name == "e5":      # tw/self/nbr 合成尤度
        C = []
        for s, b in (("tw+nbr", 0.25), ("tw+nbr", 0.5), ("tw+nbr", 1.0),
                     ("tw+self", 0.25), ("tw+self", 0.5)):
            k = dict(v88_src=s)
            k["v88_beta_nbr" if "nbr" in s else "v88_beta_self"] = b
            C.append((f"{s}(β{b})", _p(**k)))
        C.append(("tw+self+nbr(β.5/.5)", _p(v88_src="tw+self+nbr", v88_beta_self=0.5, v88_beta_nbr=0.5)))
        return C
    raise SystemExit(f"unknown CFG={name}")


def main():
    wells = base.list_wells()
    if NW: wells = wells[:NW]
    t0 = time.time()
    cache = {}
    for w in wells:
        c = base.build_cache(w)
        if c is not None and c["tvt_true"] is not None:
            cache[w] = c
    base.CACHE = cache; v88.CACHE = cache
    ALL = [w for w in wells if w in cache]
    print(f"[v88] wells={len(ALL)} seed={SEED} n_seeds={NSEED}(固定) CFG={CFG} cache {time.time()-t0:.0f}s", flush=True)

    # ---- 対照: base エンジン r0(struct-free) ----
    Pc = base.full_params(dict(R0)); Pc["use_anchor"] = False; Pc["n_seeds"] = NSEED
    Pc["tw_source"] = "tw"   # ★対照は素のtw(envに引きずられないよう明示)
    t1 = time.time()
    e_ctl = base.predict_errs(Pc, ALL, SEED, NSEED, CHUNK)
    E_ctl = np.concatenate([e_ctl[w] for w in ALL])
    m0 = np.isfinite(E_ctl)
    rmse_ctl = float(np.sqrt(np.mean(E_ctl[m0] ** 2)))
    print(f"  [対照] r0(base engine, struct-free) = {rmse_ctl:.4f}   ({time.time()-t1:.0f}s)", flush=True)

    rows = []
    store = {"_ctl": E_ctl}
    for nm, P in configs(CFG):
        t2 = time.time()
        try:
            e = v88.predict_errs(P, ALL, SEED, NSEED, CHUNK)
        except Exception as ex:
            print(f"  [{nm}] FAILED {type(ex).__name__}: {ex}", flush=True); continue
        E = np.concatenate([e[w] for w in ALL])
        m = np.isfinite(E) & m0
        r = float(np.sqrt(np.mean(E[m] ** 2)))
        rho = float(np.corrcoef(E[m], E_ctl[m])[0, 1])
        # blend(等重み/最適)も見る: 脱相関脚としての価値
        b50 = float(np.sqrt(np.mean((0.5 * E[m] + 0.5 * E_ctl[m]) ** 2)))
        bw = min(((float(np.sqrt(np.mean(((1 - a) * E_ctl[m] + a * E[m]) ** 2))), a)
                  for a in np.linspace(0, 1, 21)))
        rows.append((nm, r, r - rmse_ctl, rho, b50, bw[0], bw[1]))
        store[nm] = E
        print(f"  [{nm}] RMSE={r:.4f} (Δ{r-rmse_ctl:+.4f})  ρ={rho:.3f}  blend50={b50:.4f} "
              f"best={bw[0]:.4f}@w{bw[1]:.2f}  ({time.time()-t2:.0f}s)", flush=True)

    print(f"\n=== v88 {CFG} 結果 ({len(ALL)}井, seed{SEED}, n_seeds={NSEED}) 対照r0={rmse_ctl:.4f} ===")
    print(f"{'構成':32s} {'RMSE':>8s} {'Δ':>8s} {'ρ':>6s} {'blend50':>8s} {'best':>8s} {'w':>5s}")
    for nm, r, d, rho, b50, bb, bwt in sorted(rows, key=lambda x: x[1]):
        print(f"{nm:32s} {r:8.4f} {d:+8.4f} {rho:6.3f} {b50:8.4f} {bb:8.4f} {bwt:5.2f}")
    pickle.dump(store, open(OUT / f"errs_{CFG}_nw{len(ALL)}.pkl", "wb"))
    print(f"\nsaved errors -> {OUT / f'errs_{CFG}_nw{len(ALL)}.pkl'}")


if __name__ == "__main__":
    main()
