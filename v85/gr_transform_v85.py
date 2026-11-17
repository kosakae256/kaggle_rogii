# -*- coding: utf-8 -*-
"""
v85 — GRの別表現をPFに入力する研究(struct不使用, ユーザー発案 2026-07-24)
========================================================================
前提: 自己相似性(Milankovitch周期)でPFが追えない。GR自体の表現を変えて追いやすくする。
制約(既知の法則): 絶対GRレベルを消す表現(NCC/勾配/rank)は死。低周波+絶対振幅を保つ方向のみ生存。
方針: GR変換T()を hw GR と typewell GR に同一適用 → pf_tuningの struct-free PFに入力 →
      本番eval overall RMSE(r0固定)と、無変換r0との誤差相関ρ(=別エンジンになるか)を測る。
      prefix使用も可(井戸レベルの変換パラメータ推定に使ってよい)。

変換レジストリ(TRANSFORM env):
  none      : 無変換(=r0基準)
  antieq    : ★v85-1 希少性拡張(anti-equalization)。typewell GR分布の低密度域を引き伸ばし=稀な値を識別的に。
  logamp    : 対数圧縮(log(1+g/g0))。高振幅マーカーの相対差を保ちつつ裾を圧縮。
  lowpass   : 低域(MAr)。P6の反エイリアス成分。単体PFへ。
  bandcap   : 生+低域の加算合成(生*w1 + 低域*w2)を1chに畳んだ簡易2ch。
使い方: TRANSFORM=antieq NW=200 python v85/gr_transform_v85.py
"""
import os, io, sys, json, time, glob
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
from pathlib import Path
import numpy as np, pandas as pd

P0 = Path(r"C:\Users\kosaka256\Documents\rogii_claude")
DATA = P0 / "rogii-wellbore-geology-prediction"
ART = P0 / "v85" / "artifacts_v85"; ART.mkdir(parents=True, exist_ok=True)
NB = P0 / "pf_tuning" / "pf_tuning_pf01.ipynb"
TRANSFORM = os.environ.get("TRANSFORM", "antieq")
NW = int(os.environ.get("NW", "200"))
SEED = int(os.environ.get("SEED", "1"))
PARAM = os.environ.get("PARAM", "r0")
# 変換ハイパラ
MAR = int(os.environ.get("MAR", "13"))            # lowpass/bandcap 移動平均窓(行/ft相当)
AQ_STR = float(os.environ.get("AQ_STR", "1.0"))   # antieq強度(0=無変換, 1=フル)
AQ_NB = int(os.environ.get("AQ_NB", "40"))        # antieqヒスト bin数
LOG_G0 = float(os.environ.get("LOG_G0", "40.0"))
BC_W1 = float(os.environ.get("BC_W1", "1.0")); BC_W2 = float(os.environ.get("BC_W2", "1.0"))

# --- struct-free PF取込 ---
_nb = json.loads(NB.read_text(encoding="utf-8"))
_ns = dict(np=np, pd=pd, json=json, hashlib=__import__("hashlib"))
from dataclasses import dataclass, asdict, field
_ns.update(dataclass=dataclass, asdict=asdict, field=field)
exec("".join(_nb["cells"][5]["source"]), _ns)
PFParams = _ns["PFParams"]; run_pf = _ns["run_pf_lik_ensemble_param"]

R0 = dict(
    n_particles=400, n_seeds=8, likelihood_scale=10.536319667317654,
    init_pos_std=0.9331541486020234, init_rate_std=0.00047753793163756304,
    mom=0.9997911117425017, vn=0.0006297672017716754, pn=0.005014009140727263,
    rp=0.7627309890952053, rr=0.00045663430474523125, resamp=0.032049849143995945,
    gr_sig_min=11.865793233968713, gr_sig_max=44.98927964612316, gr_sig_def=119.25397464636981,
    gr_sig_mult=3.058318309943017, tvt_clip_margin=150.56111180234504, rate_clip=2.0,
    gr_power=1.7826896851395453, hgr_smooth_r=0, tw_gr_smooth_r=0, jump_prob=0.002, jump_std=5.0,
)
_PF_FIELDS = set(PFParams().__dict__.keys())
if PARAM != "r0":
    _lib = json.loads((P0 / "v73" / "artifacts_v73" / "pf_lib_v69_4.json").read_text())
    R0 = {**R0, **{k: v for k, v in _lib[PARAM].items() if k in _PF_FIELDS}}; R0["n_seeds"] = 8
print(f"PF取込OK PARAM={PARAM} TRANSFORM={TRANSFORM}", flush=True)


def load(wid):
    hw = pd.read_csv(DATA / "train" / f"{wid}__horizontal_well.csv").reset_index(drop=True)
    tw = pd.read_csv(DATA / "train" / f"{wid}__typewell.csv").reset_index(drop=True)
    hw["__row_id"] = np.arange(len(hw))
    return hw, tw


def _ma(x, r):
    if r <= 1: return np.asarray(x, float)
    return pd.Series(x).rolling(2 * r + 1, center=True, min_periods=1).mean().to_numpy()


def build_transform(tw):
    """typewell GR分布から変換関数 f(g)->g' を構築(hw/tw両方に同一適用)。絶対レベルは保つ。"""
    tgr = tw["GR"].astype(float).to_numpy()
    tgr = tgr[np.isfinite(tgr)]
    if TRANSFORM in ("none", "lowpass", "bandcap"):
        return None
    if TRANSFORM == "logamp":
        return lambda g: np.log1p(np.clip(g, 0, None) / LOG_G0) * LOG_G0   # レベルスケール維持
    if TRANSFORM == "antieq":
        # anti-equalization: 低密度域を引き伸ばす単調変換。f'(x)∝1/pdf(x)。
        lo, hi = np.percentile(tgr, [1, 99])
        edges = np.linspace(lo, hi, AQ_NB + 1)
        cnt, _ = np.histogram(np.clip(tgr, lo, hi), bins=edges)
        dens = cnt / max(cnt.sum(), 1) / (edges[1] - edges[0]) + 1e-9
        inv = 1.0 / dens                                # 低密度=大きいステップ
        inv = inv / inv.mean()                           # 平均スケール1に正規化(絶対レベル保持)
        step = (edges[1] - edges[0]) * inv
        node = np.concatenate([[lo], lo + np.cumsum(step)])  # 累積=変換後座標
        node = (node - node[0]) / (node[-1] - node[0]) * (hi - lo) + lo  # 元レンジへ再スケール
        # AQ_STR=0で恒等, 1でフル
        ident = edges
        node = (1 - AQ_STR) * ident + AQ_STR * node
        return lambda g: np.interp(np.clip(g, lo, hi), edges, node)
    raise ValueError(TRANSFORM)


def apply_transform(hw, tw):
    """hw['GR'] と tw['GR'] を変換して返す(新hw,新tw)。lowpass/bandcapは平滑系。"""
    f = build_transform(tw)
    hw2 = hw.copy(); tw2 = tw.copy()
    hg = hw["GR"].astype(float).interpolate(limit_direction="both").to_numpy()
    tg = tw["GR"].astype(float).to_numpy()
    if TRANSFORM == "lowpass":
        hw2["GR"] = _ma(hg, MAR); tw2["GR"] = _ma(tg, MAR)
    elif TRANSFORM == "bandcap":
        hw2["GR"] = BC_W1 * hg + BC_W2 * _ma(hg, MAR)
        tw2["GR"] = BC_W1 * tg + BC_W2 * _ma(tg, MAR)
    elif f is not None:
        hw2["GR"] = f(hg); tw2["GR"] = f(tg)
    return hw2, tw2


def eval_real(wid, transform=True):
    hw, tw = load(wid)
    ev = hw["TVT_input"].isna().to_numpy() & hw["TVT"].notna().to_numpy()
    hr, tr = (apply_transform(hw, tw) if transform else (hw, tw))
    pred = run_pf(hr, tr, params=R0, return_full=True)
    e = pred[ev] - hw["TVT"].to_numpy(float)[ev]
    return e[np.isfinite(e)]


def _ov(errs): return float(np.sqrt(np.mean(np.concatenate([x for x in errs if len(x)]) ** 2)))


def main():
    wids = sorted({os.path.basename(p).split("__")[0] for p in glob.glob(str(DATA / "train" / "*__horizontal_well.csv"))})
    rng = np.random.default_rng(SEED)
    wsel = list(rng.choice(wids, size=min(NW, len(wids)), replace=False))
    eo, et, per = [], [], []
    t0 = time.time()
    for i, w in enumerate(wsel):
        try:
            e_off = eval_real(w, transform=False)
            e_on = eval_real(w, transform=True)
        except Exception as ex:
            print(f"  {w[:8]}: ERROR {type(ex).__name__} {ex}"); continue
        if not len(e_off) or not len(e_on): continue
        eo.append(e_off); et.append(e_on)
        per.append((float(np.sqrt(np.mean(e_off**2))), float(np.sqrt(np.mean(e_on**2)))))
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(wsel)} 暫定 base={_ov(eo):.3f} {TRANSFORM}={_ov(et):.3f} ({time.time()-t0:.0f}s)", flush=True)
    per = np.array(per)
    Eo = np.concatenate(eo); Et = np.concatenate(et); n = min(len(Eo), len(Et))
    rho = float(np.corrcoef(Eo[:n], Et[:n])[0, 1])
    so, st = np.sqrt(np.mean(Eo**2)), np.sqrt(np.mean(Et**2))
    best = min((float(np.sqrt(np.mean(((1-w)*Eo[:n] + w*Et[:n])**2))), float(w)) for w in np.linspace(0, 1, 101))
    print(f"\n=== v85 GR変換={TRANSFORM} ({len(per)}井, PARAM={PARAM}, r0固定) ===")
    print(f"  ★本番eval overall: 無変換 {so:.4f} → {TRANSFORM} {st:.4f}  (Δ{so-st:+.4f})")
    print(f"  井戸RMSE平均 {per[:,0].mean():.3f}→{per[:,1].mean():.3f} / 勝率 {(per[:,1]<per[:,0]).mean()*100:.0f}%")
    print(f"  ★相関ρ(誤差 無変換↔変換) = {rho:.3f}  blend: w={best[1]:.2f}→{best[0]:.4f}(無変換比{best[0]-so:+.4f}) 成立ρ<{min(so,st)/max(so,st):.3f}")
    pd.DataFrame(dict(wid=wsel[:len(per)], rmse_base=per[:,0], rmse_tf=per[:,1])).to_csv(ART / f"tf_{TRANSFORM}.csv", index=False)


if __name__ == "__main__":
    main()
