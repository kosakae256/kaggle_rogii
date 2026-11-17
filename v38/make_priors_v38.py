# -*- coding: utf-8 -*-
"""
v38 prior銀行: 追加の物理事前分布を {well: {tvt, sig}} pkl として量産
====================================================================
v38(struct=K16 dip場)と同じ流儀で、priorごとに
  tvt(row) = そのpriorの絶対TVT予測(ev行順)
  sig(row) = そのpriorのLOO残差を「信頼度プロキシ」ビンで較正した実誤差
を作る。錨の強さはOptunaに渡さない(σ=実誤差、ANCHOR_MULTだけ外側選択)という原則は共通。

priors:
  hold      : ★last_known TVT を定数保持(傾き0=v28 graft の hold 相当)。σ=外挿距離ビン(奥ほど自動で弱まる)。
              test安全(TVT_input と MD だけ)。V38_HOLD_DECAY>0 で距離比例の追加減衰(奥をさらに弱める手動ノブ)。
  selfdip   : 自坑井の既知帯末尾300行の robust dip で直線外挿(近傍不要=孤立坑井に効く)。σ=外挿距離ビン。
  formation : FormationPlaneKNN の地層面予測(v28 parquet tvtF_ANCC, train=自己除外済=LOO)。σ=spatial_knn_dist ビン。
  dense     : dense ANCC クリギング(v28 parquet tvt_dense_d)。σ=dense_dist ビン。

出力: v38/artifacts_v38/priors/prior_{name}_train.pkl + prior_{name}_sigcurve.json
実行: rogii_claude/.venv/Scripts/python.exe v38/make_priors_v38.py
使い方: set V38_PRIOR=formation → tune_anchor_v38.py(studyは v38_formation_mult{M})
"""
import os, glob, json, pickle
from pathlib import Path
import numpy as np, pandas as pd
import pyarrow.parquet as pq

P = Path(r"C:\Users\kosaka256\Documents\rogii_claude")
TRAIN = P / "rogii-wellbore-geology-prediction" / "train"
OUT = P / "v38" / "artifacts_v38" / "priors"; OUT.mkdir(parents=True, exist_ok=True)
PARQ = P / "v28" / "artifacts_v28" / "data" / "train.parquet"
SIG_MIN, SIG_MAX, NBINS = 3.0, 60.0, 8
TAIL_N = 300


def calib_sigma(proxy, err, nbins=NBINS):
    """proxy(信頼度指標)ビン別の RMS 残差カーブ。"""
    ok = np.isfinite(proxy) & np.isfinite(err)
    pv, ev = proxy[ok], err[ok]
    edges = np.quantile(pv, np.linspace(0, 1, nbins + 1)); edges[0] = -np.inf; edges[-1] = np.inf
    ctr, sg = [], []
    for b in range(nbins):
        m = (pv >= edges[b]) & (pv < edges[b + 1])
        if m.sum() < 50: continue
        ctr.append(float(np.median(pv[m]))); sg.append(float(np.sqrt(np.mean(ev[m] ** 2))))
    ctr = np.array(ctr); sg = np.clip(np.array(sg), SIG_MIN, SIG_MAX)
    return ctr, sg


def save_prior(name, per_well, ctr, sg, extra=None):
    (OUT / f"prior_{name}_train.pkl").write_bytes(pickle.dumps(per_well))
    js = dict(centers=ctr.tolist(), sigs=sg.tolist(), sig_min=SIG_MIN, sig_max=SIG_MAX,
              n_wells=len(per_well), **(extra or {}))
    (OUT / f"prior_{name}_sigcurve.json").write_text(json.dumps(js, indent=1))
    e = np.concatenate([d["_err"] for d in per_well.values()])
    e = e[np.isfinite(e)]
    print(f"[{name:9s}] wells {len(per_well)}  overall LOO RMSE {np.sqrt(np.mean(e**2)):7.3f}  "
          f"sig curve {sg.round(1).tolist()}")
    for d in per_well.values(): d.pop("_err", None)
    (OUT / f"prior_{name}_train.pkl").write_bytes(pickle.dumps(per_well))


# ============ 0) hold(last_known 定数保持 = v28 graft の hold を PF prior 化) ============
#   tvt(ev) = last_known_tvt(傾き0)。σ は「最終既知点からの MD 距離」ビン別 LOO 残差で較正
#   = 近傍(dist小)は強い錨・奥(dist大)ほど σ が大きく=自動的に弱まる。
#   V38_HOLD_DECAY>0 で σ *= (1 + DECAY*dist/1000) を上乗せ(奥をさらに弱める手動調整余地)。
print("building hold ...")
HOLD_DECAY = float(os.environ.get("V38_HOLD_DECAY", "0.0"))
pw = {}; proxy_all, err_all = [], []
for fp in sorted(glob.glob(str(TRAIN / "*__horizontal_well.csv"))):
    wid = os.path.basename(fp).split("__")[0]
    hw = pd.read_csv(fp, usecols=["MD", "TVT", "TVT_input"])
    ti = hw["TVT_input"].to_numpy(float); md = hw["MD"].to_numpy(float); tv = hw["TVT"].to_numpy(float)
    ev = np.where(~np.isfinite(ti))[0]; kn = np.where(np.isfinite(ti))[0]
    if len(ev) == 0 or len(kn) < 10: continue
    last_md, last_tvt = float(md[kn[-1]]), float(ti[kn[-1]])
    dist = md[ev] - last_md
    tvt = np.full(len(ev), last_tvt, dtype=np.float64)   # ★hold: 最終既知TVTを保持(傾き0)
    err = tvt - tv[ev]
    pw[wid] = dict(tvt=tvt, _dist=dist, _err=err.astype(np.float64))
    proxy_all.append(dist); err_all.append(err)
ctr, sg = calib_sigma(np.concatenate(proxy_all), np.concatenate(err_all))
for wid, d in pw.items():
    dd = d.pop("_dist")
    sig = np.clip(np.interp(dd, ctr, sg), SIG_MIN, SIG_MAX)
    if HOLD_DECAY > 0:                                    # 奥ほどさらに弱める手動ノブ
        sig = np.clip(sig * (1.0 + HOLD_DECAY * np.maximum(dd, 0.0) / 1000.0), SIG_MIN, SIG_MAX * 4)
    d["sig"] = sig.astype(np.float64)
save_prior("hold", pw, ctr, sg, dict(proxy="md_since", slope=0, hold_decay=HOLD_DECAY))

# ============ 1) selfdip(CSVから直接) ============
print("building selfdip ...")
pw = {}; proxy_all, err_all = [], []
for fp in sorted(glob.glob(str(TRAIN / "*__horizontal_well.csv"))):
    wid = os.path.basename(fp).split("__")[0]
    hw = pd.read_csv(fp, usecols=["MD", "TVT", "TVT_input"])
    ti = hw["TVT_input"].to_numpy(float); md = hw["MD"].to_numpy(float); tv = hw["TVT"].to_numpy(float)
    ev = np.where(~np.isfinite(ti))[0]
    kn = np.where(np.isfinite(ti))[0]
    if len(ev) == 0 or len(kn) < 10: continue
    t = kn[-TAIL_N:]
    dmd = np.diff(md[t]); dtv = np.diff(ti[t]); m = dmd > 0
    slope = float(np.median(dtv[m] / dmd[m])) if m.sum() >= 3 else 0.0
    last_md, last_tvt = float(md[kn[-1]]), float(ti[kn[-1]])
    dist = md[ev] - last_md
    tvt = last_tvt + slope * dist
    err = tvt - tv[ev]
    pw[wid] = dict(tvt=tvt.astype(np.float64), _dist=dist, _err=err.astype(np.float64))
    proxy_all.append(dist); err_all.append(err)
ctr, sg = calib_sigma(np.concatenate(proxy_all), np.concatenate(err_all))
for wid, d in pw.items():
    d["sig"] = np.clip(np.interp(d.pop("_dist"), ctr, sg), SIG_MIN, SIG_MAX).astype(np.float64)
save_prior("selfdip", pw, ctr, sg, dict(tail_n=TAIL_N, proxy="md_since"))

# ============ 2) formation / dense(v28 parquetから) ============
cols = ["well", "md_since", "last_known_tvt", "target",
        "tvtF_ANCC", "spatial_knn_dist", "tvt_dense_d", "dense_dist"]
t = pq.read_table(PARQ, columns=cols).to_pydict()
gw = np.array(t["well"]); mds = np.asarray(t["md_since"], float)
lk = np.asarray(t["last_known_tvt"], float); tgt = np.asarray(t["target"], float)
true_abs = lk + tgt


def to_abs(col):
    v = np.asarray(col, float)
    return v if abs(np.nanmean(v)) > 1000 else lk + v   # 絶対値かdelta(from-lk)かを自動判定


for name, val_col, prox_col in [("formation", "tvtF_ANCC", "spatial_knn_dist"),
                                ("dense", "tvt_dense_d", "dense_dist")]:
    tvt_abs = to_abs(t[val_col]); proxy = np.asarray(t[prox_col], float)
    err = tvt_abs - true_abs
    ctr, sg = calib_sigma(proxy, err)
    pw = {}
    for w in pd.unique(gw):
        idx = np.where(gw == w)[0]; r = idx[np.argsort(mds[idx], kind="stable")]   # ev行順
        pw[w] = dict(tvt=tvt_abs[r].astype(np.float64),
                     sig=np.clip(np.interp(proxy[r], ctr, sg), SIG_MIN, SIG_MAX).astype(np.float64),
                     _err=err[r])
    save_prior(name, pw, ctr, sg, dict(src=val_col, proxy=prox_col))

print("\ndone. 使い方例(PowerShell):")
print('  $env:V38_PRIOR="hold"; $env:V38_MULT="20"; python v38/tune_anchor_v38.py')
print('  # 奥をさらに弱めて再生成: $env:V38_HOLD_DECAY="1.0"; python v38/make_priors_v38.py')
print('  $env:V38_PRIOR="formation"; $env:V38_MULT="30"; python v38/tune_anchor_v38.py')
