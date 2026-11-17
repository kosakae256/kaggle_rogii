# -*- coding: utf-8 -*-
"""
v84 — 事前区間self-tuning PF(struct不使用, ユーザー発案 2026-07-23)
====================================================================
狙い: ある井戸の事前区間(prefix)の「開始〜MD_end-100ft」だけを既知として、
残り100ftを疑似evalにPFパラメータをOptunaで最適化 → そのパラメータで本番eval推論。
「prefixで自己最適化したパラメータは本番でも良いか?」の転移実験。struct/K16一切不使用。

エンジン: pf_tuning/pf_tuning_pf01.ipynb の struct-free PF(run_pf_lik_ensemble_param)を exec取込。
モード:
  MODE=well  : 井戸ごとに200 trials → 井戸別best params → 本番eval。default比較。
  MODE=group : ランダムN井の疑似eval pooledで共有params最適化(ノイズ減退)。
使い方: MODE=well NW=20 TRIALS=200 python v84/pf_prefix_tune_v84.py
        MODE=group NG=5 GSIZE=10 TRIALS=200 python v84/pf_prefix_tune_v84.py
"""
import os, io, sys, json, time, glob, pickle
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
from pathlib import Path
import numpy as np, pandas as pd

P0 = Path(r"C:\Users\kosaka256\Documents\rogii_claude")
DATA = P0 / "rogii-wellbore-geology-prediction"
ART = P0 / "v84" / "artifacts_v84"; ART.mkdir(parents=True, exist_ok=True)
NB = P0 / "pf_tuning" / "pf_tuning_pf01.ipynb"
_cut_env = os.environ.get("CUT_MD", "100")          # prefix末尾のこのMD(ft)を疑似eval化
CUT_MD = 1e18 if _cut_env.lower() == "max" else float(_cut_env)   # max=先頭錨以外を全て疑似eval
ANCHOR_ROWS = int(os.environ.get("ANCHOR_ROWS", "150"))          # CUT_MD=max時に錨として残す先頭既知行数
TRIALS = int(os.environ.get("TRIALS", "50"))
MODE = os.environ.get("MODE", "well")
NW = int(os.environ.get("NW", "30"))
NG = int(os.environ.get("NG", "5")); GSIZE = int(os.environ.get("GSIZE", "10"))
SEED = int(os.environ.get("SEED", "42"))

# --- ノートのPFセル(c1 imports除く, c3 loading除く, c5 PF実装)を取り込む ---
_nb = json.loads(NB.read_text(encoding="utf-8"))
_src5 = "".join(_nb["cells"][5]["source"])
_ns = dict(np=np, pd=pd, json=json, hashlib=__import__("hashlib"))
from dataclasses import dataclass, asdict, field
_ns.update(dataclass=dataclass, asdict=asdict, field=field)
exec(_src5, _ns)
PFParams = _ns["PFParams"]
run_pf = _ns["run_pf_lik_ensemble_param"]
print(f"PF取込OK (default n_seeds={PFParams().n_seeds})", flush=True)

# ★出発点=既存の強いr0(v73/nnpf_v73 R0, struct不使用)。defベースライン & Optuna初手(enqueue)。
R0 = dict(
    n_particles=400, n_seeds=8, likelihood_scale=10.536319667317654,
    init_pos_std=0.9331541486020234, init_rate_std=0.00047753793163756304,
    mom=0.9997911117425017, vn=0.0006297672017716754, pn=0.005014009140727263,
    rp=0.7627309890952053, rr=0.00045663430474523125, resamp=0.032049849143995945,
    gr_sig_min=11.865793233968713, gr_sig_max=44.98927964612316, gr_sig_def=119.25397464636981,
    gr_sig_mult=3.058318309943017, tvt_clip_margin=150.56111180234504, rate_clip=2.0,
    gr_power=1.7826896851395453, hgr_smooth_r=0, tw_gr_smooth_r=0, jump_prob=0.002, jump_std=5.0,
)
# ★PARAM: ベースパラメータ選択(r0=既定 / p_1..p_5=v69_4 pf_lib)。affineモードで強力パラメータ比較用。
PARAM = os.environ.get("PARAM", "r0")
_PF_FIELDS = set(PFParams().__dict__.keys())
if PARAM != "r0":
    _lib = json.loads((P0 / "v73" / "artifacts_v73" / "pf_lib_v69_4.json").read_text())
    _base = {**R0, **{k: v for k, v in _lib[PARAM].items() if k in _PF_FIELDS}}  # PFParams互換キーのみ
    _base["n_seeds"] = 8
    R0 = _base
    print(f"[v84] ベースパラメータ={PARAM} (n_particles={R0['n_particles']} ls={R0['likelihood_scale']:.2f} mom={R0['mom']:.5f})", flush=True)

# 長さ依存パラメータ: prefix疑似eval(短)では機能せず本番(長)に転移しない → r0固定で探索から外す
FIX_LEN = os.environ.get("FIX_LEN", "1") == "1"    # 1: resamp/rr をr0固定(既定)
FIX_KEYS = {"resamp", "rr"} if FIX_LEN else set()


# 探索空間(r0中心・r0を含むよう範囲拡張, n_seeds=8固定)
def suggest(trial):
    F = lambda n, lo, hi, log=False: (R0[n] if n in FIX_KEYS else trial.suggest_float(n, lo, hi, log=log))
    C = lambda n, ch: (R0[n] if n in FIX_KEYS else trial.suggest_categorical(n, ch))
    return dict(
        n_particles=C("n_particles", [400, 500, 600]), n_seeds=8,
        likelihood_scale=F("likelihood_scale", 9.0, 30.0, True),
        init_pos_std=F("init_pos_std", 0.15, 1.2), init_rate_std=F("init_rate_std", 0.0003, 0.0045),
        mom=F("mom", 0.9980, 0.99985), vn=F("vn", 0.0005, 0.0028, True), pn=F("pn", 0.0025, 0.014, True),
        rp=F("rp", 0.5, 0.82), rr=F("rr", 0.00008, 0.0006, True), resamp=F("resamp", 0.03, 0.14, True),
        gr_sig_min=F("gr_sig_min", 6.0, 13.0), gr_sig_max=F("gr_sig_max", 26.0, 48.0),
        gr_sig_def=F("gr_sig_def", 85.0, 140.0), gr_sig_mult=F("gr_sig_mult", 2.0, 3.4),
        tvt_clip_margin=F("tvt_clip_margin", 80.0, 155.0),
        rate_clip=C("rate_clip", [1.5, 2.0, 2.5, 3.0]), gr_power=F("gr_power", 1.75, 2.7),
        hgr_smooth_r=C("hgr_smooth_r", [0, 2, 3]), tw_gr_smooth_r=C("tw_gr_smooth_r", [0, 1]),
        jump_prob=C("jump_prob", [0.0, 0.002, 0.005]), jump_std=C("jump_std", [5.0, 10.0]),
    )


def _enqueue_r0(st):
    """Optuna初手にr0を投入(固定キーは除く=distributionに無いため)。"""
    st.enqueue_trial({k: v for k, v in R0.items() if k not in FIX_KEYS and k not in ("n_seeds",)})


def load(wid):
    hw = pd.read_csv(DATA / "train" / f"{wid}__horizontal_well.csv").reset_index(drop=True)
    tw = pd.read_csv(DATA / "train" / f"{wid}__typewell.csv").reset_index(drop=True)
    hw["__row_id"] = np.arange(len(hw))
    return hw, tw


def make_cut(hw):
    """prefix末尾CUT_MDを疑似eval化したhw2 と 疑似eval行マスク(true=TVT_input元既知)。
    CUT_MD=max: 先頭ANCHOR_ROWS行のみ既知に残し、prefixの残り全部を疑似eval(錨=先頭側)。"""
    kn = hw["TVT_input"].notna().to_numpy()
    if kn.sum() < 200: return None, None
    md = hw["MD"].to_numpy(float)
    kn_idx = np.where(kn)[0]
    if CUT_MD >= 1e17:                              # max: 先頭ANCHOR_ROWS行を錨、残りprefix全部を疑似eval
        pseudo = kn.copy()
        pseudo[kn_idx[:ANCHOR_ROWS]] = False        # 先頭錨は既知のまま
    else:
        cut = md[kn_idx[-1]] - CUT_MD
        pseudo = kn & (md > cut)
    if pseudo.sum() < 30 or (kn & ~pseudo).sum() < 100: return None, None
    hw2 = hw.copy()
    hw2.loc[pseudo, "TVT_input"] = np.nan          # 疑似eval化(TVT列=真値は残る)
    # 本番evalは落とす(prefix末端まで)=prefixだけで完結させる
    hw2 = hw2.loc[:kn_idx[-1]].copy()
    ps2 = pseudo[:kn_idx[-1] + 1]
    return hw2, ps2


AFFINE = os.environ.get("AFFINE", "0") == "1"       # ★prefixアフィンGR再較正(gain α + offset β)
AMIN, AMAX = 0.3, 3.0


def recalibrate_gr(hw, tw):
    """prefix既知区間で GR_hw ≈ α·GR_tw(TVT_input)+β をロバスト最小二乗fit→全行GRを
    (GR−β)/α で typewellスケールへ写像。α,βは井戸レベル定数=転移する。identity fallback付き。"""
    kn = hw["TVT_input"].notna().to_numpy()
    ttv = tw["TVT"].to_numpy(float); tgr = tw["GR"].to_numpy(float)
    m = np.isfinite(ttv) & np.isfinite(tgr)
    if m.sum() < 8 or kn.sum() < 100: return hw, (1.0, 0.0)
    ttv, tgr = ttv[m], tgr[m]
    order = np.argsort(ttv); ttv, tgr = ttv[order], tgr[order]
    gr = hw["GR"].astype(float).interpolate(limit_direction="both").to_numpy()
    x = np.interp(hw["TVT_input"].to_numpy(float)[kn], ttv, tgr)    # typewell GR at known TVT
    y = gr[kn]
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 100: return hw, (1.0, 0.0)
    x, y = x[ok], y[ok]
    c = np.polyfit(x, y, 1)                          # y = a·x + b
    for _ in range(3):                               # IRLS(Cauchy風)でロバスト化
        r = y - np.polyval(c, x); s = np.median(np.abs(r)) * 1.4826 + 1e-6
        w = 1.0 / (1.0 + (r / (2 * s)) ** 2)
        c = np.polyfit(x, y, 1, w=w)
    a, b = float(c[0]), float(c[1])
    if not np.isfinite(a) or a < AMIN or a > AMAX: return hw, (1.0, 0.0)
    hw2 = hw.copy()
    hw2["GR"] = (gr - b) / a                         # 全行GRをtypewellスケールへ
    return hw2, (a, b)


def rmse_on(hw_run, tw, params, mask, truth):
    if AFFINE:
        hw_run, _ = recalibrate_gr(hw_run, tw)
    pred = run_pf(hw_run, tw, params=params, return_full=True)
    e = pred[mask] - truth[mask]
    e = e[np.isfinite(e)]
    return float(np.sqrt(np.mean(e ** 2))) if len(e) else 1e9


def eval_real(wid, params, ret_err=False, affine=None):
    hw, tw = load(wid)
    ev = hw["TVT_input"].isna().to_numpy() & hw["TVT"].notna().to_numpy()
    use_aff = AFFINE if affine is None else affine
    hw_run, _ = (recalibrate_gr(hw, tw) if use_aff else (hw, (1.0, 0.0)))
    pred = run_pf(hw_run, tw, params=params, return_full=True)
    e = pred[ev] - hw["TVT"].to_numpy(float)[ev]
    e = e[np.isfinite(e)]
    if ret_err: return e
    return float(np.sqrt(np.mean(e ** 2))) if len(e) else np.nan


def _overall(errs):
    e = np.concatenate([x for x in errs if len(x)])
    return float(np.sqrt(np.mean(e ** 2)))


import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
DEF = None  # default params baseline


def tune_well(wid):
    hw, tw = load(wid)
    hw2, ps2 = make_cut(hw)
    if hw2 is None: return None
    truth2 = hw2["TVT"].to_numpy(float)
    def obj(trial):
        p = suggest(trial)
        return rmse_on(hw2, tw, p, ps2, truth2)
    st = optuna.create_study(direction="minimize",
                             sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=20))
    _enqueue_r0(st)                                 # ★初手=r0(強い出発点)
    t0 = time.time()
    st.optimize(obj, n_trials=TRIALS, show_progress_bar=False)
    best = {**{k: R0[k] for k in FIX_KEYS}, **st.best_params}; best["n_seeds"] = 8   # 固定キーはr0で補完
    et = eval_real(wid, best, ret_err=True)
    ed = eval_real(wid, R0, ret_err=True)           # def=r0
    r_pdef = rmse_on(hw2, tw, R0, ps2, truth2)
    return dict(wid=wid, prefix_best=st.best_value, prefix_def=r_pdef,
                real_tuned=float(np.sqrt(np.mean(et**2))), real_def=float(np.sqrt(np.mean(ed**2))),
                err_tuned=et, err_def=ed, sec=time.time() - t0)


def main():
    wids = sorted({os.path.basename(p).split("__")[0] for p in glob.glob(str(DATA / "train" / "*__horizontal_well.csv"))})
    rng = np.random.default_rng(SEED)
    if MODE == "affine":
        # ★アフィンGR再較正 単独効果(Optuna無し, r0固定): 較正あり/なしの本番eval overallを直接比較
        wsel = list(rng.choice(wids, size=min(NW, len(wids)), replace=False))
        eo, ea, ab_list, per = [], [], [], []
        for i, w in enumerate(wsel):
            try:
                e_off = eval_real(w, R0, ret_err=True, affine=False)
                e_on = eval_real(w, R0, ret_err=True, affine=True)
                _, ab = recalibrate_gr(*load(w))
            except Exception as ex:
                print(f"  {w[:8]}: ERROR {type(ex).__name__} {ex}"); continue
            if not len(e_off): continue
            eo.append(e_off); ea.append(e_on); ab_list.append(ab)
            per.append((float(np.sqrt(np.mean(e_off**2))), float(np.sqrt(np.mean(e_on**2)))))
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(wsel)} 暫定 off={_overall(eo):.3f} on={_overall(ea):.3f}", flush=True)
        per = np.array(per); ab = np.array(ab_list)
        print(f"\n=== v84 affine GR再較正 ({len(per)}井, r0固定) ===")
        print(f"  ★本番eval overall(行プール): 較正なし {_overall(eo):.4f} → 較正あり {_overall(ea):.4f}  (Δ{_overall(eo)-_overall(ea):+.4f})")
        print(f"  井戸RMSE平均: {per[:,0].mean():.3f} → {per[:,1].mean():.3f} / 勝率 {(per[:,1]<per[:,0]).mean()*100:.0f}%")
        print(f"  推定α(ゲイン): 中央{np.median(ab[:,0]):.3f} [p10,p90]=[{np.percentile(ab[:,0],10):.2f},{np.percentile(ab[:,0],90):.2f}]  identity化(α=1)された井戸: {(ab[:,0]==1.0).sum()}本")
        print(f"  α×勝敗: |α−1|>0.15の井戸で改善率 {(per[np.abs(ab[:,0]-1)>0.15,1]<per[np.abs(ab[:,0]-1)>0.15,0]).mean()*100:.0f}%")
        # ★較正あり/なしの行別誤差の相関ρ と blend(=第2エンジンになるか)
        Eo = np.concatenate(eo); Ea = np.concatenate(ea)
        n = min(len(Eo), len(Ea))
        rho = float(np.corrcoef(Eo[:n], Ea[:n])[0, 1])
        so, sa = np.sqrt(np.mean(Eo**2)), np.sqrt(np.mean(Ea**2))
        best = min((float(np.sqrt(np.mean(((1-w)*Eo[:n] + w*Ea[:n])**2))), float(w)) for w in np.linspace(0, 1, 101))
        print(f"  ★相関ρ(誤差 なし↔あり) = {rho:.3f}  (低いほど別エンジン=脚価値)")
        print(f"     blend: w_affine={best[1]:.2f} → {best[0]:.4f} (なし{so:.4f}比 {best[0]-so:+.4f})  成立条件ρ<{min(so,sa)/max(so,sa):.3f}")
        pd.DataFrame(dict(wid=wsel[:len(per)], rmse_off=per[:,0], rmse_on=per[:,1],
                          alpha=ab[:,0], beta=ab[:,1])).to_csv(ART / "affine.csv", index=False)
        return
    if MODE == "well":
        wsel = list(rng.choice(wids, size=min(NW, len(wids)), replace=False))
        rows = []
        for i, w in enumerate(wsel):
            r = tune_well(w)
            if r is None: continue
            rows.append(r)
            print(f"  [{i+1}/{len(wsel)}] {w[:8]}: prefix def={r['prefix_def']:.2f}→tuned={r['prefix_best']:.2f}  "
                  f"real def={r['real_def']:.2f}→tuned={r['real_tuned']:.2f}  ({r['sec']:.0f}s)", flush=True)
        o_t = _overall([r["err_tuned"] for r in rows]); o_d = _overall([r["err_def"] for r in rows])
        df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("err_")} for r in rows])
        df.to_csv(ART / "well_tune.csv", index=False)
        print(f"\n=== v84 well-tune ({len(df)}井, {TRIALS}trials, cut={CUT_MD}ft) ===")
        print(f"  prefix疑似eval: default {df.prefix_def.mean():.3f} → self-tuned {df.prefix_best.mean():.3f}(当然改善)")
        print(f"  ★本番eval overall(行プール=Kaggle指標): default {o_d:.4f} → self-tuned {o_t:.4f}  (Δ{o_d-o_t:+.4f})")
        print(f"     井戸RMSE平均: def {df.real_def.mean():.3f}→tuned {df.real_tuned.mean():.3f} / 勝率{(df.real_tuned<df.real_def).mean()*100:.0f}%")
        print(f"     暴発井(tuned>15かつ悪化): {((df.real_tuned>15)&(df.real_tuned>df.real_def)).sum()}本")
        print(f"     相関(prefix改善 vs 本番改善): {np.corrcoef(df.prefix_def-df.prefix_best, df.real_def-df.real_tuned)[0,1]:.3f}")
    else:  # group
        rows = []
        for g in range(NG):
            gw = list(rng.choice(wids, size=GSIZE, replace=False))
            cuts = []
            for w in gw:
                hw, tw = load(w); hw2, ps2 = make_cut(hw)
                if hw2 is None: continue
                cuts.append((w, hw2, tw, ps2, hw2["TVT"].to_numpy(float)))
            if len(cuts) < 1: continue             # GSIZE=1以上どの群サイズでも動く
            def obj(trial):
                p = suggest(trial); es = []
                for w, hw2, tw, ps2, tr in cuts:
                    pred = run_pf(hw2, tw, params=p, return_full=True)
                    e = pred[ps2] - tr[ps2]; es.append(e[np.isfinite(e)])
                e = np.concatenate(es); return float(np.sqrt(np.mean(e ** 2)))
            st = optuna.create_study(direction="minimize",
                                     sampler=optuna.samplers.TPESampler(seed=SEED + g, n_startup_trials=20))
            _enqueue_r0(st)                          # ★初手=r0
            t0 = time.time(); st.optimize(obj, n_trials=TRIALS)
            best = {**{k: R0[k] for k in FIX_KEYS}, **st.best_params}; best["n_seeds"] = 8
            for w, *_ in cuts:
                et = eval_real(w, best, ret_err=True); ed = eval_real(w, R0, ret_err=True)
                rows.append(dict(group=g, wid=w, real_tuned=float(np.sqrt(np.mean(et**2))),
                                 real_def=float(np.sqrt(np.mean(ed**2))), err_tuned=et, err_def=ed))
            print(f"  group{g}({len(cuts)}井): prefix best={st.best_value:.3f} ({time.time()-t0:.0f}s)", flush=True)
        o_t = _overall([r["err_tuned"] for r in rows]); o_d = _overall([r["err_def"] for r in rows])
        df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("err_")} for r in rows])
        df.to_csv(ART / "group_tune.csv", index=False)
        print(f"\n=== v84 group-tune ({NG}群×{GSIZE}井, {TRIALS}trials, cut={CUT_MD}ft) ===")
        print(f"  ★本番eval overall(行プール): default {o_d:.4f} → group-tuned {o_t:.4f}  (Δ{o_d-o_t:+.4f})")
        print(f"     井戸RMSE平均 def{df.real_def.mean():.3f}→tuned{df.real_tuned.mean():.3f} / 勝率{(df.real_tuned<df.real_def).mean()*100:.0f}% / 暴発{((df.real_tuned>15)&(df.real_tuned>df.real_def)).sum()}本")


if __name__ == "__main__":
    main()
