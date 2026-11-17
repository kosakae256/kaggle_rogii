# -*- coding: utf-8 -*-
"""
v42 空間MRF datum同時推定(mean-field変分推論)
=====================================================================================
物理prior=struct(K16 dip場, v41のPFと同じ)を GR非依存の datum投票 に使い、
GRコスト曲線(datum両義=二峰)+ 空間平滑 を結合して、坑井をまたいで datum を同時に解く。

  base    : pf_ancc(実PF軌跡, datum-flip問題を持つ)
  状態    : 坑井ごとのグローバル offset b_w
  データ項: C_w(b) = base+b の GRマッチ費用(z化GR, 二峰)
  struct  : b を struct が示すオフセット s_w=median(struct_tvt-base) へ引く(GR非依存)
  空間項  : 隣接坑井(XY kNN)の b は近い(struct除去後の残差が空間平滑)
  推論    : mean-field  p_w(b)∝exp(-E/T),  b_w=E[b] を反復(posterior mean=RMSE最適)

★過去の地雷回避: 絶対TVT面の空間補間は死ぬ(cross-field 2589ft暴走)。ここは b=struct残差offset
  だけを空間結合する(絶対面ではない)。struct が転移する dip 情報を土台に、残差のみ扱う。

ablation(λ_struct/λ_spatial の on/off)で「空間結合が硬い核を救うか」が同時に出る=診断内蔵。
出力: v42/artifacts_v42/spatial_datum_result.json + v42_spatial.png
実行: .venv/Scripts/python.exe v42/spatial_datum_v42.py(CPU 数分)
"""
import json, pickle, time
from pathlib import Path
import numpy as np, pandas as pd
import pyarrow.parquet as pq
from scipy.spatial import cKDTree

P = Path(r"C:\Users\kosaka256\Documents\rogii_claude")
TRAIN = P / "rogii-wellbore-geology-prediction" / "train"
ART = P / "v42" / "artifacts_v42"; ART.mkdir(parents=True, exist_ok=True)
STRUCT = pickle.loads((P / "v38" / "artifacts_v38" / "struct_prior_train.pkl").read_bytes())

BASE_MODE = "v41_pf"                     # "pf_ancc"(v28=pf_1生PF) / "v41_pf"(v41=錨PF pfA) / "v41_gbm"(強base)
DMAX, DSTEP = 40.0, 1.0                 # offset grid ±40ft(datum gap~15ftの±2倍超)
BGRID = np.arange(-DMAX, DMAX + 1e-6, DSTEP)
KNBR = 8                                 # 空間近傍数
HGR = 2                                  # GR平滑半径
T_MF = 1.0                               # mean-field温度
N_ITER = 30; DAMP = 0.5


def robust_z(x):
    m = np.isfinite(x)
    med = np.nanmedian(x[m]); mad = np.nanmedian(np.abs(x[m] - med)) * 1.4826
    if not np.isfinite(mad) or mad < 1e-6: mad = np.nanstd(x[m]) or 1.0
    return (x - med) / mad


def _smooth(v, r):
    if r <= 0: return v
    return pd.Series(v).rolling(2 * r + 1, center=True, min_periods=1).mean().to_numpy()


# ---------- データ読み込み ----------
print("[v42] parquet ...")
t = pq.read_table(P / "v28" / "artifacts_v28" / "data" / "train.parquet",
                  columns=["well", "target", "last_known_tvt", "pf_ancc", "md_since"]).to_pydict()
gw = np.array(t["well"]); tgt = np.asarray(t["target"], float)
lk = np.asarray(t["last_known_tvt"], float)
mds = np.asarray(t["md_since"], float); true_all = lk + tgt
base_by_well = None
if BASE_MODE == "v41_gbm":
    base_all = lk + np.load(P / "v41" / "artifacts_v41" / "gbm_v41_oof.npy").astype(float)   # 残差→絶対TVT
elif BASE_MODE == "v41_pf":
    # v41 parquet(行順が別)から錨PF pf_ancc を坑井別md順で取得
    t2 = pq.read_table(P / "v41" / "artifacts_v41" / "data" / "train.parquet",
                       columns=["well", "md_since", "pf_ancc"]).to_pydict()
    g2 = np.array(t2["well"]); m2 = np.asarray(t2["md_since"], float); pf2 = np.asarray(t2["pf_ancc"], float)
    base_by_well = {}
    for w in pd.unique(g2):
        ix = np.where(g2 == w)[0]; base_by_well[w] = pf2[ix[np.argsort(m2[ix], kind="stable")]]
    base_all = np.asarray(t["pf_ancc"], float)   # fallback用
else:
    base_all = np.asarray(t["pf_ancc"], float)
if base_by_well is None:
    print(f"[v42] BASE={BASE_MODE}  base overall(全行)={np.sqrt(np.mean((base_all-true_all)**2)):.4f}")
else:
    print(f"[v42] BASE={BASE_MODE}(v41錨PF)  wells={len(base_by_well)}")
wid_order = pd.unique(gw)
rows_of = {w: np.where(gw == w)[0] for w in wid_order}
order_of = {w: rows_of[w][np.argsort(mds[rows_of[w]], kind="stable")] for w in wid_order}
try:
    hd = pd.read_csv(P / "v32" / "artifacts_v32" / "hardcore_diag.csv")
    HARD = set(hd[hd["hard"] == True].iloc[:, 0].astype(str))
except Exception:
    HARD = set()

print("[v42] per-well GRコスト曲線 + struct offset + XY ...")
t0 = time.time()
W = []              # 有効坑井
Cw = {}             # well -> (nb,) GRコスト over BGRID
sw = {}             # struct-preferred offset scalar
sigw = {}           # struct sigma(prior width)
xy = {}             # (x,y)
base_w = {}         # base trajectory(md順)
true_w = {}
for w in wid_order:
    sp = STRUCT.get(w)
    if sp is None: continue
    r = order_of[w]
    TR = true_all[r]
    B = base_by_well.get(w) if base_by_well is not None else base_all[r]
    if B is None or len(B) != len(r): continue
    stv = np.asarray(sp["tvt"], float)
    if len(stv) != len(r): continue
    fp = TRAIN / f"{w}__horizontal_well.csv"; tp = TRAIN / f"{w}__typewell.csv"
    try:
        hw = pd.read_csv(fp, usecols=["X", "Y", "GR", "TVT_input"])
        tw = pd.read_csv(tp, usecols=["TVT", "GR"]).dropna().sort_values("TVT")
    except Exception:
        continue
    if len(tw) < 5: continue
    ev = ~np.isfinite(hw["TVT_input"].to_numpy())
    gr = hw["GR"].to_numpy(float)
    gr_all = pd.Series(gr).interpolate(limit_direction="both").to_numpy()
    gr_ev = _smooth(gr_all, HGR)[ev]
    tw_tvt = tw["TVT"].to_numpy(float); tw_gr = tw["GR"].to_numpy(float)
    z_all = robust_z(np.r_[tw_gr, gr_ev])
    tw_gr_z = z_all[:len(tw_gr)]; gr_ev_z = z_all[len(tw_gr):]
    if len(gr_ev_z) != len(B): continue
    # GRコスト曲線: 各offset b で base+b のGR不一致
    cand = (B[:, None] + BGRID[None, :])                    # (rows, nb)
    eg = np.interp(cand.ravel(), tw_tvt, tw_gr_z).reshape(cand.shape)
    C = np.nanmean((gr_ev_z[:, None] - eg) ** 2, axis=0)    # (nb,)
    W.append(w); Cw[w] = C
    sw[w] = float(np.median(stv - B))                        # struct が示すoffset
    sigw[w] = float(np.clip(np.mean(sp["sig"]), 3.0, 40.0))
    xy[w] = (float(hw["X"].mean()), float(hw["Y"].mean()))
    base_w[w] = B; true_w[w] = TR
print(f"      wells {len(W)} ({time.time()-t0:.0f}s)")

# 空間近傍(自己除外)
XY = np.array([xy[w] for w in W]); wi = {w: i for i, w in enumerate(W)}
tree = cKDTree(XY / (XY.std(0) + 1e-9))
dist, idx = tree.query(XY / (XY.std(0) + 1e-9), k=KNBR + 1)
NBR = {}; NBW = {}
for i, w in enumerate(W):
    jj = [j for j in idx[i] if j != i][:KNBR]; dd = dist[i][1:KNBR + 1]
    ww = np.exp(-(dd / (np.median(dd) + 1e-9)) ** 2)
    NBR[w] = [W[j] for j in jj]; NBW[w] = ww / (ww.sum() + 1e-9)


def infer(lam_struct, lam_spatial, use_gr=True):
    """mean-field で b_w(posterior mean)を返す。"""
    m = {w: (sw[w] if not use_gr else float(BGRID[np.argmin(Cw[w])])) for w in W}
    for _ in range(N_ITER if lam_spatial > 0 else 1):
        m_new = {}
        for w in W:
            E = np.zeros_like(BGRID)
            if use_gr: E = E + Cw[w]
            if lam_struct > 0: E = E + lam_struct * ((BGRID - sw[w]) / sigw[w]) ** 2
            if lam_spatial > 0:
                mn = np.array([m[n] for n in NBR[w]]); wn = NBW[w]
                nb_mean = float((wn * mn).sum())
                E = E + lam_spatial * ((BGRID - nb_mean) / 15.0) ** 2
            p = np.exp(-(E - E.min()) / T_MF); p /= p.sum()
            m_new[w] = float((BGRID * p).sum())
        if lam_spatial > 0:
            m = {w: DAMP * m[w] + (1 - DAMP) * m_new[w] for w in W}
        else:
            m = m_new
    return m


def evalm(m):
    e_all, e_hard = [], []
    for w in W:
        pred = base_w[w] + (m[w] if m is not None else 0.0)
        e = pred - true_w[w]; e = e[np.isfinite(e)]
        e_all.append(e)
        if w in HARD: e_hard.append(e)
    ea = np.concatenate(e_all); eh = np.concatenate(e_hard) if e_hard else np.array([np.nan])
    return float(np.sqrt(np.mean(ea ** 2))), float(np.sqrt(np.nanmean(eh ** 2)))


print("\n=== ablation(base=pf_ancc)===")
configs = [
    ("base(pf_ancc)",        None),
    ("+GR再datum",           dict(lam_struct=0.0, lam_spatial=0.0)),
    ("+struct投票",          dict(lam_struct=0.3, lam_spatial=0.0)),
    ("+spatial(struct off)", dict(lam_struct=0.0, lam_spatial=0.5)),
    ("+struct+spatial",      dict(lam_struct=0.3, lam_spatial=0.5)),
]
res = {}
for name, cfg in configs:
    if cfg is None:
        ov, hd_ = evalm(None)
    else:
        m = infer(cfg["lam_struct"], cfg["lam_spatial"])
        ov, hd_ = evalm(m)
    res[name] = dict(overall=ov, hard=hd_)
    print(f"  {name:22s}: overall={ov:.4f}  hard={hd_:.4f}")

# λ_spatial 小スイープ(struct併用)
print("\n=== λ_spatial スイープ(λ_struct=0.3)===")
sweep = {}
for ls in [0.0, 0.25, 0.5, 1.0, 2.0]:
    m = infer(0.3, ls); ov, hd_ = evalm(m); sweep[ls] = (ov, hd_)
    print(f"  λ_spatial={ls:.2f}: overall={ov:.4f} hard={hd_:.4f}")

(ART / "spatial_datum_result.json").write_text(json.dumps(
    dict(configs=res, sweep={str(k): v for k, v in sweep.items()},
         n_wells=len(W), n_hard=sum(1 for w in W if w in HARD),
         params=dict(DMAX=DMAX, KNBR=KNBR, T_MF=T_MF)), indent=1))

# グラフ
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Meiryo"
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
names = list(res); ov = [res[n]["overall"] for n in names]; hdv = [res[n]["hard"] for n in names]
x = np.arange(len(names))
ax[0].bar(x - 0.2, ov, 0.4, label="overall"); ax[0].bar(x + 0.2, hdv, 0.4, label="硬い核")
ax[0].set_xticks(x); ax[0].set_xticklabels(names, rotation=30, ha="right", fontsize=8)
ax[0].axhline(res["base(pf_ancc)"]["overall"], color="k", ls=":", lw=.7)
ax[0].legend(); ax[0].set_title("v42 空間datum同時推定: ablation")
ls_ = list(sweep); ax[1].plot(ls_, [sweep[l][0] for l in ls_], "o-", label="overall")
ax[1].plot(ls_, [sweep[l][1] for l in ls_], "s-", label="硬い核")
ax[1].set_xlabel("λ_spatial(空間結合強度)"); ax[1].legend()
ax[1].set_title("空間結合を強めると硬い核は縮むか?")
plt.tight_layout(); plt.savefig(ART / "v42_spatial.png", dpi=110)
print(f"\nsaved {ART/'spatial_datum_result.json'} / v42_spatial.png")
