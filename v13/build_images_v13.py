"""
v13 画像ビルダー(PF再実行なし)―― 保存済みPF波形から U-Net 入力画像を作る
============================================================================

create_pf_images_v13.py が保存した per-seed 生波形(fwd_traj/bwd_traj)から、
バンド幅・ビン幅・チャンネルを **後から変えて** 画像だけ作り直す。**PFは再実行しない**。

入力(各坑井): artifacts_v13/img/{well}.npz の
    fwd_traj(32,T), bwd_traj(128,T), center(T), target_rel(T,train), ev_index(T), last_tvt
出力(各坑井): artifacts_v13/images/{well}.npz の
    img(C,H,T) float16, target_rel(T), mask(T,band依存), center, ev_index, last_tvt, band, binw, H, channels

- 順/逆密度は保存波形から再構成(CSV不要)。
- 密度chは各セットの fwd_traj_{set}/bwd_traj_{set} から再構成。GRコストch(全セット共通1枚)は
  typewell から再計算(PF不要)。GRの平滑/σ は PARAM_SET(=pf_1)の値だけ使う(PF_PARAM_SETS_RAW)。
- target_rel はバンド非依存(=真TVT−center)なので npz の値を流用。mask は新バンドで再計算。

使い方:
  rogii_claude/.venv/Scripts/python.exe v13/build_images_v13.py
  下の CONFIG(BAND/BINW/PF_SETS/KINDS/USE_GRCOST など)を変えて何度でも作り直せる。
"""

import time, glob
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

# ============================== CONFIG ==============================
PROJECT  = Path(r"C:\Users\kosaka256\Documents\rogii_claude")
RAW_DIR  = PROJECT / "v13" / "artifacts_v13" / "img"        # create_pf_images_v13.py の出力(生波形)
OUT_DIR  = PROJECT / "v13" / "artifacts_v13" / "images"     # ここに学習用画像を出す
DATA_DIR = PROJECT / "rogii-wellbore-geology-prediction" / "train"   # grcost 用(typewell)

BAND   = 32.0              # 中心から ±BAND ft
BINW   = 0.5              # ビン幅 ft(1px=0.5ft, H=129)
KDE_SIG_BINS = 1.5         # トラジェクトリ密度のKDE平滑(ビン単位)
# 各PFセットごとに fwd/bwd 密度チャンネルを作る。最後に grcost(全セット共通)を1枚。
PF_SETS  = ["pf_1", "pf_2", "pf_3"]   # 画像化するPFセット(npz に fwd_traj_{set} があるもの)
KINDS    = ["fwd", "bwd"]             # 各セットで作る密度の種類
USE_GRCOST = True                     # GRコストch(全セット共通)を末尾に追加
PARAM_SET = "pf_1"         # grcost の typewell平滑/σ算出に使うPFパラメータ
N_WELLS  = None            # None=全坑井
SAVE_F16 = True
# → チャンネル順: [pf_1 fwd, pf_1 bwd, pf_2 fwd, pf_2 bwd, pf_3 fwd, pf_3 bwd, grcost]
# ===================================================================

H = int(2 * BAND / BINW) + 1
OFFSETS = (np.arange(H) - H // 2) * BINW

PF_PARAM_SETS_RAW = {
    "pf_1": {"gr_sig_def":114.11439615641156,"gr_sig_max":32.79346108797473,"gr_sig_min":7.495879205722816,
             "gr_sig_mult":2.7482077023565075,"hgr_smooth_r":3,"tw_gr_smooth_r":0},
}
def _p_gr(name):
    r = PF_PARAM_SETS_RAW[name]
    return dict(HGR=int(r.get("hgr_smooth_r",0)), TW=int(r.get("tw_gr_smooth_r",0)),
                SIG_MIN=float(r["gr_sig_min"]), SIG_MAX=float(r["gr_sig_max"]),
                SIG_DEF=float(r["gr_sig_def"]), SIG_MULT=float(r["gr_sig_mult"]))


# ----------------------- GR コスト用ヘルパ(PF不要・CSVから)-----------------------
def _smooth_radius_values(vals, fb, r):
    r = int(r); s = pd.Series(vals, dtype="float32").interpolate(limit_direction="both").fillna(float(fb))
    return s.to_numpy(np.float32) if r <= 0 else s.rolling(2*r+1, center=True, min_periods=1).mean().to_numpy(np.float32)

def _grid(tw_tvt, tw_gr, step=0.2):
    tmin=float(tw_tvt.min()); tmax=float(tw_tvt.max()); tvt_g=np.arange(tmin,tmax+step,step)
    return np.interp(tvt_g,tw_tvt,tw_gr).astype(np.float64), float(tmin), float(step)

def gr_ingredients(wid, gr_cfg, ev_index):
    """typewell と horizontal_well から gg(=TVT軸GR), 軸, gs, 観測GR(評価行) を作る(PF不要)。"""
    hw = pd.read_csv(DATA_DIR / f"{wid}__horizontal_well.csv")
    tw = pd.read_csv(DATA_DIR / f"{wid}__typewell.csv").sort_values("TVT")
    tw_tvt = tw["TVT"].to_numpy(float); tw_gr = tw["GR"].to_numpy(float)
    tw_fb = float(np.nanmean(tw_gr)) if np.isfinite(np.nanmean(tw_gr)) else 0.
    tw_gr_pf = _smooth_radius_values(tw_gr, tw_fb, gr_cfg["TW"]).astype(np.float64)
    gg, gmin, gst = _grid(tw_tvt, tw_gr_pf)
    # gs(GRシグマ)
    hgr_known = _smooth_radius_values(hw["GR"].astype(float).to_numpy(), tw_fb, gr_cfg["HGR"])
    kn = hw[hw["TVT_input"].notna()]
    if len(kn) < 20:
        gs = gr_cfg["SIG_DEF"]
    else:
        kpos = hw.index.get_indexer(kn.index)
        resid = hgr_known[kpos] - np.interp(kn["TVT_input"].values, tw_tvt, tw_gr_pf)
        gs = float(np.nanstd(resid))
        if not np.isfinite(gs) or gs <= 0: gs = gr_cfg["SIG_DEF"]
    gs = float(np.clip(gs*gr_cfg["SIG_MULT"], gr_cfg["SIG_MIN"], gr_cfg["SIG_MAX"]))
    gr_full = hw["GR"].astype(float).interpolate(limit_direction="both").fillna(tw_fb).to_numpy(float)
    gr_proc = _smooth_radius_values(gr_full, tw_fb, gr_cfg["HGR"])
    gr_ev = gr_proc[ev_index].astype(np.float32)
    gg_axis = gmin + np.arange(len(gg)) * gst
    return gg.astype(np.float64), gg_axis, gs, gr_ev


# ----------------------- チャンネル構築 -----------------------
def _traj_density(traj, center, T):
    rel = traj.astype(np.float32) - center[None, :]
    idx = np.clip(np.round(rel / BINW).astype(np.int64) + H // 2, 0, H - 1)
    dens = np.zeros((H, T), np.float32); tcol = np.arange(T)
    for s in range(traj.shape[0]):
        np.add.at(dens, (idx[s], tcol), np.float32(1.0))
    dens = gaussian_filter1d(dens, KDE_SIG_BINS, axis=0)
    csum = dens.sum(0, keepdims=True); csum[csum < 1e-12] = 1.0
    return dens / csum

def _gr_cost(center, T, gg, gg_axis, gs, gr_ev):
    tvt_grid = center[:, None] + OFFSETS[None, :]               # (T,H)
    exp_gr = np.interp(tvt_grid.ravel(), gg_axis, gg).reshape(T, H).astype(np.float32)
    cost = np.abs(gr_ev[:, None] - exp_gr).T                    # (H,T)
    return np.clip(cost / (5.0 * gs + 1e-6), 0, 1).astype(np.float32)


def build_one(wid, gr_cfg):
    d = np.load(RAW_DIR / f"{wid}.npz")
    center = d["center"].astype(np.float32); T = len(center)
    chans = []; ch_names = []
    for s in PF_SETS:
        for kind in KINDS:
            key = f"{kind}_traj_{s}"
            if key not in d.files:               # alias fallback(pf_1のみ等)
                key = f"{kind}_traj" if s == PF_SETS[0] else None
            if key is None or key not in d.files:
                continue
            chans.append(_traj_density(d[key].astype(np.float32), center, T))
            ch_names.append(f"{s}_{kind}")
    if USE_GRCOST:
        gg, gg_axis, gs, gr_ev = gr_ingredients(wid, gr_cfg, d["ev_index"].astype(np.int64))
        chans.append(_gr_cost(center, T, gg, gg_axis, gs, gr_ev)); ch_names.append("grcost")
    img = np.stack(chans, 0)                                    # (C,H,T)

    out = dict(img=(img.astype(np.float16) if SAVE_F16 else img.astype(np.float32)),
               center=center, ev_index=d["ev_index"].astype(np.int32), last_tvt=np.float32(d["last_tvt"]),
               band=np.float32(BAND), binw=np.float32(BINW), H=np.int32(H),
               channels=np.array(ch_names))
    in_band = -1.0
    if "target_rel" in d.files:
        tr = d["target_rel"].astype(np.float32)
        out["target_rel"] = tr
        out["mask"] = (np.abs(tr) <= BAND).astype(np.float32)
        in_band = float(out["mask"].mean())
    np.savez_compressed(OUT_DIR / f"{wid}.npz", **out)
    return T, in_band


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    gr_cfg = _p_gr(PARAM_SET)
    wells = sorted(p.stem for p in RAW_DIR.glob("*.npz"))
    if N_WELLS is not None: wells = wells[:N_WELLS]
    n_ch = len(PF_SETS) * len(KINDS) + (1 if USE_GRCOST else 0)
    print(f"raw={RAW_DIR}  -> out={OUT_DIR}")
    print(f"BAND=±{BAND}ft BINW={BINW} H={H}  sets={PF_SETS} kinds={KINDS} grcost={USE_GRCOST} -> {n_ch}ch  wells={len(wells)}")
    t0 = time.perf_counter(); man = []
    for i, wid in enumerate(wells):
        T, ib = build_one(wid, gr_cfg)
        man.append(dict(well=wid, T=T, in_band_frac=ib))
        if (i+1) % 100 == 0 or i+1 == len(wells):
            print(f"  {i+1}/{len(wells)} ({time.perf_counter()-t0:.0f}s)")
    m = pd.DataFrame(man); m.to_csv(OUT_DIR / "manifest.csv", index=False)
    print(f"saved {len(m)} images -> {OUT_DIR}")
    if (m["in_band_frac"] >= 0).any():
        print(f"target in-band 率: mean={m['in_band_frac'].mean():.3f} min={m['in_band_frac'].min():.3f}")

if __name__ == "__main__":
    main()
