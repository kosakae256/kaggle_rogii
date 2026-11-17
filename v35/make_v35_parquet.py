# -*- coding: utf-8 -*-
"""v35: v28 parquet(557列)に v34 seed dump 由来の「モード構造」特徴 18列を追記する。
=====================================================================================
- 再計算なし: v28/artifacts_v28/data/train.parquet + v34/artifacts_v34/seed_dump_pf{1,2,3}.npz を読むだけ。
- ★参照点は blend ではなく「全軌跡のコンセンサス(576本の行別median)」= モデル非依存。
  blend基準の特徴をGBMに入れると自分の予測を参照する形(v24型スタッキング)になるため。
- 列(prefix m34_, 全float32):
  行レベル(5): m34_cons(コンセンサス軌跡・絶対TVT) / m34_cons_delta(−lk) /
               m34_lo(p10−cons) / m34_hi(p90−cons) / m34_spread(p90−p10)
  井戸レベル(13, 行にブロードキャスト):
    m34_n_modes / m34_minor_mass(トップモード以外の質量計)
    m34_m2_c(挑戦者モード中心, cons基準) / m34_m2_mass / m34_mode_gap
    m34_m2_ext_mass / m34_m2_tw_mass / m34_m2_sm_mass / m34_m2_fw_mass(挑戦者を支持する族別質量)
    m34_m2_xm_self / m34_m2_xm_nbr(挑戦者軌跡のGR採点マージン vs cons)
    m34_m2_jump(挑戦者軌跡のgeo_z最大ジャンプ) / m34_m2_rough(2階差分荒れ, cons比)
出力: v35/artifacts_v35/data/train.parquet
実行: .venv/Scripts/python.exe v35/make_v35_parquet.py(数分・CPUのみ)
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

PROJECT = Path(r"C:\Users\kosaka256\Documents\rogii_claude")
V28_PQ = PROJECT / "v28" / "artifacts_v28" / "data" / "train.parquet"
V34_ART = PROJECT / "v34" / "artifacts_v34"
OUT = PROJECT / "v35" / "artifacts_v35" / "data" / "train.parquet"
OUT.parent.mkdir(parents=True, exist_ok=True)

GAP_LINK = 8.0
SRCS = ("tw", "self", "nbr")
FLTS = (("sm", "ps"), ("fw", "pfw"))

print("[v35] parquet 読み込み ...")
table = pq.read_table(V28_PQ)
g = table.column("well").to_pandas().to_numpy()
lk = table.column("last_known_tvt").to_numpy(zero_copy_only=False).astype(np.float64)
has = table.column("pf_nbr_has").to_numpy(zero_copy_only=False).astype(np.float64)
wid = list(pd.unique(g))
rows_of = {w: np.where(g == w)[0] for w in wid}
nrows = len(g)
print(f"  rows={nrows} wells={len(wid)} cols={table.num_columns}")

DUMPS = {k: np.load(V34_ART / f"seed_dump_pf{k}.npz") for k in (1, 2, 3)}
KEYS = {k: set(DUMPS[k].files) for k in (1, 2, 3)}

ROW_COLS = ["m34_cons", "m34_cons_delta", "m34_lo", "m34_hi", "m34_spread"]
WELL_COLS = ["m34_n_modes", "m34_minor_mass", "m34_m2_c", "m34_m2_mass", "m34_mode_gap",
             "m34_m2_ext_mass", "m34_m2_tw_mass", "m34_m2_sm_mass", "m34_m2_fw_mass",
             "m34_m2_xm_self", "m34_m2_xm_nbr", "m34_m2_jump", "m34_m2_rough"]
COLS = {c: np.full(nrows, np.nan, np.float32) for c in ROW_COLS + WELL_COLS}


def cluster(ds, wts):
    o = np.argsort(ds)
    groups, cur = [], [o[0]]
    for i in o[1:]:
        if ds[i] - ds[cur[-1]] > GAP_LINK:
            groups.append(cur); cur = [i]
        else:
            cur.append(i)
    groups.append(cur)
    out = []
    for m in groups:
        m = np.array(m)
        dd, ww_ = ds[m], wts[m]
        oo = np.argsort(dd); cw = np.cumsum(ww_[oo])
        c = float(dd[oo][min(np.searchsorted(cw, cw[-1] / 2), len(dd) - 1)])
        out.append(dict(c=c, idx=m, mass=float(ww_.sum())))
    return sorted(out, key=lambda m: -m["mass"])


def score_traj(traj, gg, gmin, gst, gs, gr):
    G = len(gg)
    ii = np.clip((np.asarray(traj, float) - gmin) / gst, 0.0, G - 1.000001)
    i0 = np.floor(ii).astype(int); t = ii - i0
    eg = gg[i0] * (1 - t) + gg[np.minimum(i0 + 1, G - 1)] * t
    m = np.isfinite(gr) & np.isfinite(eg)
    if m.sum() < 10:
        return np.nan
    d = (gr[m] - eg[m]) / gs
    return float(np.mean(np.exp(-0.5 * d * d)))


n_done = 0
for w in wid:
    r = rows_of[w]
    nbr_ok = np.nanmean(has[r]) > 0.5
    # ---- 全run の per-seed 軌跡を収集(nbr無し井戸の nbr パスは tw フォールバックなので除外)----
    trajs, wts, meta_run = [], [], []   # meta_run: (k, src, flt) per item
    for k in (1, 2, 3):
        z = DUMPS[k]
        for src in SRCS:
            if src == "nbr" and not nbr_ok:
                continue
            if f"{w}|ps_{src}" not in KEYS[k]:
                continue
            ww_ = z[f"{w}|ww_{src}"]
            for flt, pref in FLTS:
                ps = z[f"{w}|{pref}_{src}"]
                trajs.append(ps); wts.append(ww_); meta_run.append((k, src, flt))
    if not trajs:
        continue
    n_runs = len(trajs)
    T = min(min(t.shape[1] for t in trajs), len(r))
    if T < 5:
        continue
    ALL = np.concatenate([t[:, :T].astype(np.float64) for t in trajs], 0)     # (n_item, T)
    IW = np.concatenate([wv / n_runs for wv in wts])                          # 全体で総和1
    IWRAW = np.concatenate(wts)                                               # run内で総和1(族別質量用)
    RUNID = np.concatenate([np.full(t.shape[0], i) for i, t in enumerate(trajs)])

    cons = np.median(ALL, axis=0)
    p10 = np.percentile(ALL, 10, axis=0); p90 = np.percentile(ALL, 90, axis=0)
    COLS["m34_cons"][r[:T]] = cons
    COLS["m34_cons_delta"][r[:T]] = cons - lk[r[:T]]
    COLS["m34_lo"][r[:T]] = p10 - cons
    COLS["m34_hi"][r[:T]] = p90 - cons
    COLS["m34_spread"][r[:T]] = p90 - p10

    ds = np.nanmedian(ALL - cons[None, :], axis=1)
    fin = np.isfinite(ds)
    if fin.sum() < 2:
        continue
    modes = cluster(ds[fin], IW[fin])
    idx_map = np.where(fin)[0]
    COLS["m34_n_modes"][r] = len(modes)
    COLS["m34_minor_mass"][r] = max(0.0, 1.0 - modes[0]["mass"] / max(IW[fin].sum(), 1e-12))
    if len(modes) >= 2:
        m2 = modes[1]; items2 = idx_map[m2["idx"]]
        COLS["m34_m2_c"][r] = m2["c"]
        COLS["m34_m2_mass"][r] = m2["mass"] / max(IW[fin].sum(), 1e-12)
        COLS["m34_mode_gap"][r] = abs(modes[0]["c"] - m2["c"])
        # 族別質量(run内質量の族平均 — v34の定義と同じ)
        fam = {"ext": [], "tw": [], "sm": [], "fw": []}
        for i_run, (k, src, flt) in enumerate(meta_run):
            in_run = items2[RUNID[items2] == i_run]
            mrun = float(IWRAW[in_run].sum()) if len(in_run) else 0.0
            (fam["ext"] if src in ("self", "nbr") else fam["tw"]).append(mrun)
            fam["sm" if flt == "sm" else "fw"].append(mrun)
        COLS["m34_m2_ext_mass"][r] = float(np.mean(fam["ext"])) if fam["ext"] else 0.0
        COLS["m34_m2_tw_mass"][r] = float(np.mean(fam["tw"])) if fam["tw"] else 0.0
        COLS["m34_m2_sm_mass"][r] = float(np.mean(fam["sm"])) if fam["sm"] else 0.0
        COLS["m34_m2_fw_mass"][r] = float(np.mean(fam["fw"])) if fam["fw"] else 0.0
        # 挑戦者モードの代表軌跡(メンバーのww加重平均)→ GR採点マージン + 物理
        w2 = IW[items2]; w2 = w2 / w2.sum() if w2.sum() > 1e-12 else np.full(len(items2), 1.0 / len(items2))
        traj2 = (w2[:, None] * ALL[items2]).sum(0)
        xs, xn = [], []
        for k in (1, 2, 3):
            z = DUMPS[k]
            if f"{w}|meta" not in KEYS[k]:
                continue
            gmin, gst, gs = z[f"{w}|meta"]; gr = z[f"{w}|gr"].astype(np.float64)[:T]
            for acc, gk in ((xs, "gg_self"), (xn, "gg_nbr")):
                key = f"{w}|{gk}"
                if key not in KEYS[k]:
                    continue
                gg = z[key].astype(np.float64)
                s2 = score_traj(traj2, gg, gmin, gst, gs, gr)
                s0 = score_traj(cons, gg, gmin, gst, gs, gr)
                if np.isfinite(s2) and np.isfinite(s0):
                    acc.append(s2 - s0)
        if xs: COLS["m34_m2_xm_self"][r] = float(np.mean(xs))
        if xn: COLS["m34_m2_xm_nbr"][r] = float(np.mean(xn))
        z1 = DUMPS[1]
        if f"{w}|z" in KEYS[1]:
            zz = z1[f"{w}|z"].astype(np.float64)[:T]
            COLS["m34_m2_jump"][r] = float(np.nanmax(np.abs(np.diff(traj2 + zz))))
        r2 = np.abs(np.diff(traj2, 2)); b2 = np.abs(np.diff(cons, 2))
        COLS["m34_m2_rough"][r] = float((np.nanmean(r2) + 1e-6) / (np.nanmean(b2) + 1e-6))
    n_done += 1
    if n_done % 100 == 0:
        print(f"  {n_done}/{len(wid)} wells ...")

for c in ROW_COLS + WELL_COLS:
    table = table.append_column(c, pa.array(COLS[c], type=pa.float32()))
pq.write_table(table, OUT)
print(f"[v35] saved: {OUT}  rows={nrows} cols={table.num_columns}(+{len(ROW_COLS) + len(WELL_COLS)})")
print(f"  処理井戸 {n_done}/{len(wid)} | 挑戦者モードあり "
      f"{int(np.sum([np.isfinite(COLS['m34_m2_c'][rows_of[w][0]]) for w in wid]))}井戸")
