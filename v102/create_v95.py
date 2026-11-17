# -*- coding: utf-8 -*-
"""v95 orchestration: train/test.parquet ビルダー(v93 create を 5表現+provenance に拡張)。

v93 → v95 変更:
  - PF表現を tw/self/nbr の3 → tw/self/nbr/nbr_gr5/self_graft の5表現に拡張。
    * nbr_gr5 = 全train井から GR類似 top5(imputers.neighbors_gr_of) を build_inputs_nbr に渡す(refs差替のみ)。
    * self_graft = build_inputs_self_graft(prefix実測+tw較正外挿)。
  - ★NN-emission 無し(w_nn=0・sim非添付)。物理錨(attach_anchor)は温存(pfAのみ)。逆方向PFも無し。
  - ★provenance(各表現の妥当性説明変数)を well単位で計算し features へ: GR一致度(grfit/grcorr)・
    近傍距離/ΔZ/GR類似度・self被覆・self_graft較正係数。
  - PFパラメータは pf_banks_config.json(v93からコピー=同一)。

出力: v95/artifacts_v95/data/{train,test}.parquet
実行: rogii_claude/.venv/Scripts/python.exe v95/create_v95.py   スモーク: NWELLS=6 python v95/create_v95.py
"""
import os, sys, io, glob, time, pickle
try:
    sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
except Exception:
    pass
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pf_banks_v95 as pf
import imputers_v95 as imp
import features_v95 as feats

PROJ = Path(os.environ.get("ROGII_PROJ", r"C:\Users\kosaka256\Documents\rogii_claude"))
DATA_DIR = pf.DATA_DIR
ART = pf.ART
SPLIT = os.environ.get("SPLIT", "train")
IS_TRAIN = (SPLIT == "train")
OUT_PARQUET = (Path(os.environ["OUT_PARQUET"]) if os.environ.get("OUT_PARQUET")
               else ART / "data" / ("train.parquet" if IS_TRAIN else "test.parquet"))
SEED = 4423098
N_SEED = pf.N_SEED
WELL_CHUNK = 60
N_JOBS = 4
NWELLS = int(os.environ["NWELLS"]) if os.environ.get("NWELLS") else None
DEVICE = pf.DEVICE
BANK_ORDER = pf.BANK_ORDER
METHODS = ["tw", "self", "nbr", "nbr_gr5", "self_graft"]
# ★v102: NN-emission を全バンクに(v96/v100両バンク)。SIM_PKL は orchestrator が指定(v97 sim を流用)。
SIM_PKL = (Path(os.environ["SIM_PKL"]) if os.environ.get("SIM_PKL")
           else pf.ART / ("sim_grfree_v97.pkl" if os.environ.get("SPLIT", "train") == "train" else "sim_grfree_test_v97.pkl"))


def _attach_sim(x, wid, SIMD):
    sd = SIMD.get(wid)
    if sd is not None and len(sd.get("st", [])) == len(x["md"]):
        x["_sim"] = sd["sim"]; x["_st"] = sd["st"]
    return x


AFFINE = os.environ.get("AFFINE", "0") == "1"   # ★affine版特徴生成(照合GRをprefix較正)
if AFFINE:
    import affine_v102 as _aff


def _gr_prefix_sm(hw, P):
    """既知prefix の 平滑GR と TVT_input(provenance grfit 用)。"""
    kn = hw[hw["TVT_input"].notna()]
    fb = float(np.nanmean(hw["GR"])) if np.isfinite(np.nanmean(hw["GR"])) else 0.0
    gr_full = hw["GR"].astype(float).interpolate(limit_direction="both").fillna(fb).to_numpy(float)
    gr_sm = pf._smooth_radius_values(gr_full, fb, P["hgr_smooth_r"])
    kpos = hw.index.get_indexer(kn.index)
    return kn["TVT_input"].to_numpy(float), gr_sm[kpos].astype(float)


def _grfit(kn_tvtin, gr_pre, x):
    """参照 gg が prefix GR をどれだけ説明するか: 残差std(grfit)・相関(grcorr)。"""
    if x is None or len(kn_tvtin) < 8:
        return np.nan, np.nan
    gg = x["gg"]; grid = x["gmin"] + np.arange(len(gg)) * x["gst"]
    at = np.interp(kn_tvtin, grid, gg); r = gr_pre - at; m = np.isfinite(r)
    if m.sum() < 8:
        return np.nan, np.nan
    gf = float(np.std(r[m]))
    gc = (float(np.corrcoef(gr_pre[m], at[m])[0, 1])
          if np.std(gr_pre[m]) > 1e-9 and np.std(at[m]) > 1e-9 else np.nan)
    return gf, gc


def build_dataset():
    base = DATA_DIR / SPLIT
    wells = sorted({os.path.basename(q).split("__")[0] for q in glob.glob(str(base / "*__horizontal_well.csv"))})
    if NWELLS is not None:
        wells = wells[:NWELLS]
    P0 = pf.bank_param(BANK_ORDER[0])
    names = []; paths = []
    for w in wells:
        hp = base / f"{w}__horizontal_well.csv"; tp = base / f"{w}__typewell.csv"
        try:
            hw = pd.read_csv(hp); tw = pd.read_csv(tp).sort_values("TVT")
        except Exception:
            continue
        if IS_TRAIN and "TVT" not in hw.columns:
            continue
        if pf.build_smoother_inputs(hw, tw["TVT"].to_numpy(float), tw["GR"].to_numpy(float), P0) is None:
            continue
        names.append(w); paths.append((hp, tp))
    print(f"[{SPLIT}] 使用坑井: {len(names)}  banks={BANK_ORDER} x 5表現={METHODS}  lag={pf.SMOOTH_LAG} n_seed={N_SEED} (★v102: NN-emission ON 全バンク)")

    FI, DI, _tw = imp.build_imputers()
    feats.set_imputers(FI, DI)
    print(f"[imputers] FI={len(FI.df)}井 / DI点={len(DI.ancc)}")

    FWD_ENS = feats.FWD_ENS; FWD_ENS.clear()
    ENS = {m: {} for m in ["self", "nbr", "nbr_gr5", "self_graft"]}   # tw は FWD_ENS
    ZGRAD = {}; NBR_META = {}; NREFS = {}; NREFS_GR5 = {}; PROV = {}

    # ---- 近傍探索(距離 & GR類似) + provenance ----
    print("[nbr] 近傍探索(距離top3 & GR類似top5) + provenance ...")
    t0 = time.perf_counter(); n_nbr = 0; n_gr5 = 0
    _cw, _cxyz = imp._train_centroids_z(); _cmap = {w: i for i, w in enumerate(_cw)}
    for (hp, tp) in paths:
        wid = os.path.basename(str(hp)).split("__")[0]
        hw = pd.read_csv(hp); tw = pd.read_csv(tp).sort_values("TVT")
        tvt_tw = tw["TVT"].to_numpy(float); gr_tw = tw["GR"].to_numpy(float)
        cx, cy = float(hw["X"].mean()), float(hw["Y"].mean()); cz = float(hw["Z"].mean())
        exc = wid if SPLIT == "train" else None
        # 距離近傍
        refw = imp.neighbors_of(cx, cy, exclude_wid=exc)
        NREFS[wid] = [imp.ref_grtvt(r) for r, _ in refw] if len(refw) >= imp.NEED_REFS else []
        nbr_dists = [d for _, d in refw]; nbr_dz = []
        for r, _ in refw:
            if r in _cmap:
                nbr_dz.append(abs(_cxyz[_cmap[r], 2] - cz))
        # GR類似近傍
        kn = hw[hw["TVT_input"].notna()]
        ng = imp.neighbors_gr_of(kn["TVT_input"].to_numpy(float), kn["GR"].to_numpy(float), cx, cy, cz, exclude_wid=exc)
        NREFS_GR5[wid] = [imp.ref_grtvt(w) for w, _, _, _ in ng]
        gr5_sim = [s for _, s, _, _ in ng]; gr5_dist = [d for _, _, d, _ in ng]; gr5_dz = [z for _, _, _, z in ng]
        NBR_META[wid] = dict(has=int(len(NREFS[wid]) > 0), nn=imp.nearest_dist(cx, cy, exclude_wid=exc), nref=len(refw))
        if NREFS[wid]:
            n_nbr += 1
        if NREFS_GR5[wid]:
            n_gr5 += 1
        # provenance: 5表現の gg を P0 で作り grfit
        kt, kg = _gr_prefix_sm(hw, P0)
        x_tw = pf.build_smoother_inputs(hw, tvt_tw, gr_tw, P0)
        x_self = pf.build_inputs_self(hw, tvt_tw, gr_tw, P0)
        x_nbr = pf.build_inputs_nbr(hw, tvt_tw, gr_tw, P0, NREFS[wid])
        x_gr5 = pf.build_inputs_nbr(hw, tvt_tw, gr_tw, P0, NREFS_GR5[wid])
        x_sg = pf.build_inputs_self_graft(hw, tvt_tw, gr_tw, P0)
        prov = {}
        for mm, xx in [("tw", x_tw), ("self", x_self), ("nbr", x_nbr), ("nbr_gr5", x_gr5), ("self_graft", x_sg)]:
            gf, gc = _grfit(kt, kg, xx); prov[mm] = dict(grfit=gf, grcorr=gc)
        _cov = float(x_sg["_graft"]["cover"]) if (x_sg is not None and "_graft" in x_sg) else np.nan
        prov["self"].update(cover=_cov, nprefix=int(len(kt)))     # self被覆=prefixがtw格子を覆う割合(self_graftのcoverと同一計算)
        prov["nbr"].update(nref=len(refw),
                           dist_min=(min(nbr_dists) if nbr_dists else np.nan), dist_mean=(float(np.mean(nbr_dists)) if nbr_dists else np.nan),
                           dz_min=(min(nbr_dz) if nbr_dz else np.nan), dz_mean=(float(np.mean(nbr_dz)) if nbr_dz else np.nan))
        prov["nbr_gr5"].update(nref=len(ng),
                               sim_top1=(gr5_sim[0] if gr5_sim else np.nan), sim_mean=(float(np.mean(gr5_sim)) if gr5_sim else np.nan),
                               dist_min=(min(gr5_dist) if gr5_dist else np.nan), dist_mean=(float(np.mean(gr5_dist)) if gr5_dist else np.nan),
                               dz_min=(min(gr5_dz) if gr5_dz else np.nan), dz_mean=(float(np.mean(gr5_dz)) if gr5_dz else np.nan))
        if x_sg is not None and "_graft" in x_sg:
            prov["self_graft"].update(**x_sg["_graft"])
        PROV[wid] = prov
    print(f"      近傍あり {n_nbr}/{len(names)} / GR類似あり {n_gr5}/{len(names)}  ({time.perf_counter()-t0:.1f}s)")

    # ---- 各バンク × 5表現 smooth-PF(★v102: 全バンク NN-emission ON) ----
    if SIM_PKL.exists():
        SIMD = pickle.loads(SIM_PKL.read_bytes())
        print(f"[v102 emission] sim load: {len(SIMD)}井 <- {SIM_PKL.name}(全バンクemission)")
    else:
        SIMD = {}; print(f"[v102 emission] WARN: {SIM_PKL} 無し -> emission OFF")
    PFCACHE = Path(os.environ.get("PF_CACHE_DIR", str(ART / "pfcache"))); PFCACHE.mkdir(parents=True, exist_ok=True)
    FORCE_PF = os.environ.get("FORCE_PF", "0") == "1"

    def run_method(bank, P, builder, store, m):
        # ★v99 PFキャッシュ: per-(split,表現,bank)。後でバンク/特徴を足しても既存PFは再実行しない
        cache = PFCACHE / f"{SPLIT}_{m}_{bank}.pkl"
        if cache.exists() and not FORCE_PF:
            d = pickle.loads(cache.read_bytes())
            if all(w in d for w in names):                     # 現在の井戸集合を完全にカバーする時のみ流用
                for w in names:
                    store[(w, bank)] = d[w]
                print(f"      [cache] {cache.name} 流用({len(names)}井)")
                return
            print(f"      [cache] {cache.name} 井戸不一致({len(d)}井<必要{len(names)}井) -> 再計算")
        inps = []
        for (hp, tp) in paths:
            wid = os.path.basename(str(hp)).split("__")[0]
            hw = pd.read_csv(hp); tw = pd.read_csv(tp).sort_values("TVT")
            x = builder(hw, tw["TVT"].to_numpy(float), tw["GR"].to_numpy(float), wid)
            if AFFINE:
                x = _aff.apply_affine(pf, x, hw, tw["GR"].to_numpy(float), P)   # ★affine較正
            x = pf.attach_anchor(x, wid, P["_physics"])       # 物理錨
            x = _attach_sim(x, wid, SIMD)                      # ★v102: NN-emission 添付
            inps.append(x)
        outs = pf.run_smoother_ext(inps, P, SEED, N_SEED, WELL_CHUNK, w_nn=P["_w_nn"])   # ★v102: emission ON(バンク別w_nn)
        dd = {}
        for i, w in enumerate(names):
            store[(w, bank)] = outs[i]; dd[w] = outs[i]
        cache.write_bytes(pickle.dumps(dd, protocol=4))       # キャッシュ保存
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    for bank in BANK_ORDER:
        P = pf.bank_param(bank)
        builders = {
            "tw":         lambda hw, tt, tg, wid, _P=P: pf.build_smoother_inputs(hw, tt, tg, _P),
            "self":       lambda hw, tt, tg, wid, _P=P: pf.build_inputs_self(hw, tt, tg, _P),
            "nbr":        lambda hw, tt, tg, wid, _P=P: pf.build_inputs_nbr(hw, tt, tg, _P, NREFS[wid]),
            "nbr_gr5":    lambda hw, tt, tg, wid, _P=P: pf.build_inputs_nbr(hw, tt, tg, _P, NREFS_GR5[wid]),
            "self_graft": lambda hw, tt, tg, wid, _P=P: pf.build_inputs_self_graft(hw, tt, tg, _P),
        }
        for m in METHODS:
            t0 = time.perf_counter(); print(f"[smoothPF-{m}] {bank} ...")
            store = FWD_ENS if m == "tw" else ENS[m]
            run_method(bank, P, builders[m], store, m)
            print(f"      done {time.perf_counter()-t0:.1f}s")

    for (hp, tp) in paths:
        wid = os.path.basename(str(hp)).split("__")[0]
        ZGRAD[wid] = pf.z_gradient_eval(pd.read_csv(hp))

    ENS_ALL = {"self": ENS["self"], "nbr": ENS["nbr"], "nbr_gr5": ENS["nbr_gr5"], "self_graft": ENS["self_graft"]}

    def one(w, hp, tp):
        feats.set_current_well(w)
        df = feats.build_well(str(hp), str(tp), IS_TRAIN)
        if df is None or len(df) == 0:
            return None
        return feats.build_self_nbr_columns(df, w, ENS_ALL, ZGRAD, NBR_META, PROV)

    print("[build] build_well(tw注入) + 5表現/provenance 列 ...")
    t0 = time.perf_counter()
    res = Parallel(n_jobs=N_JOBS, prefer="threads", verbose=5)(
        delayed(one)(names[i], paths[i][0], paths[i][1]) for i in range(len(names)))
    parts = [r for r in res if r is not None]
    print(f"      done {time.perf_counter()-t0:.1f}s  parts={len(parts)}")
    if not parts:
        raise RuntimeError("有効な坑井がありません。")
    return pd.concat(parts, ignore_index=True)


def main():
    print(f"[v95] device={DEVICE} smooth_lag={pf.SMOOTH_LAG} n_seed={N_SEED} chunk={WELL_CHUNK} NWELLS={NWELLS} (NN-emission/逆方向 なし)")
    df = build_dataset()
    if SPLIT == "train" and NWELLS is None and df["well"].nunique() < 700:
        raise SystemExit(f"[GUARD] 出力坑井数 {df['well'].nunique()} 本は異常。")
    if os.environ.get("PF_ONLY_OUT", "0") == "1":         # ★dec10/affine: 共通の非PF特徴は base v102 と同一=出力不要。pf_候補のみ残す
        meta = [c for c in ["id", "well", "target", "last_known_tvt", "md_since"] if c in df.columns]
        keep = meta + [c for c in df.columns if c.startswith("pf_")]
        df = df[keep]
        print(f"[PF_ONLY_OUT] pf_候補のみ出力({len(keep)}列 = meta{len(meta)} + pf_{len(keep)-len(meta)})", flush=True)
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PARQUET, index=False)
    cols = list(df.columns)
    print(f"\nsaved: {OUT_PARQUET}  rows={len(df)} cols={len(cols)} wells={df['well'].nunique()}")
    for pre in ["pf_self", "pf_nbr", "pf_nbr_gr5", "pf_self_graft", "prov_", "pf_"]:
        print(f"  {pre}* = {sum(c.startswith(pre) for c in cols)} 列")


if __name__ == "__main__":
    main()
