# -*- coding: utf-8 -*-
"""
v87 affine コア(create/submit 共有)。
============================================================
v69_4 の PF エンジン名前空間 `eng`(create=import module / submit=埋込globalsのSimpleNamespace)を受け取り、
  - tw/self/nbr の base を作り → 観測GR を affine 較正 → 同一PFパラメータで smooth PF(15本)
  - affine 候補 + 「self/nbr/affine の違い(共通点)」特徴を id 付きで組み立てる
を行う。eng に必要な属性:
  build_smoother_inputs, build_inputs_self, build_inputs_nbr, run_smoother_bank, affine_cal,
  _smooth_radius_values, _attach_anchor, neighbors_of, _ref_grtvt, DEC_BANK, V26_DEVICE,
  PF_BANK, PF_LIB, NEED_REFS
"""
import os, time
import numpy as np
import pandas as pd

AFF_CLIP = (0.3, 3.0)   # affine 傾き α のクリップ(v85 と同一)


def _kgr_ktvt(eng, hw, tw_gr, P):
    kn = hw[hw["TVT_input"].notna()]
    tw_fb = float(np.nanmean(tw_gr)) if np.isfinite(np.nanmean(tw_gr)) else 0.
    gr_full = hw["GR"].astype(float).interpolate(limit_direction="both").fillna(tw_fb).to_numpy(float)
    gr_sm = eng._smooth_radius_values(gr_full, tw_fb, P["hgr_smooth_r"])
    kpos = hw.index.get_indexer(kn.index)
    return gr_sm[kpos].astype(float), kn["TVT_input"].to_numpy(float)


def apply_affine(eng, base, kgr, ktvt, P):
    """参照grid gg に対し観測 base['gr'] を α,β 再較正。kgr≈α·gg(ktvt)+β → gr_aligned=(gr−β)/α。gs も aligned で再計算。"""
    gg = np.asarray(base["gg"], float); gmin = base["gmin"]; gst = base["gst"]
    tvt_grid = gmin + gst * np.arange(len(gg))
    gg_at_k = np.interp(ktvt, tvt_grid, gg)
    a, b = eng.affine_cal(kgr, gg_at_k)
    a = float(np.clip(a, *AFF_CLIP))
    base2 = dict(base)
    base2["gr"] = (np.asarray(base["gr"], float) - b) / a
    if len(kgr) >= 20:
        resid = (kgr - b) / a - gg_at_k
        gs = float(np.nanstd(resid))
        if not np.isfinite(gs) or gs <= 0:
            gs = float(P["gr_sig_def"])
    else:
        gs = float(P["gr_sig_def"])
    base2["gs"] = float(np.clip(gs * P["gr_sig_mult"], P["gr_sig_min"],
                                max(P["gr_sig_max"], P["gr_sig_min"] + 1e-6)))
    return base2, a, b


def compute_nrefs(eng, paths, split):
    NREFS = {}
    for (hp, tp) in paths:
        wid = os.path.basename(str(hp)).split("__")[0]
        hw = pd.read_csv(hp); cx, cy = float(hw["X"].mean()), float(hw["Y"].mean())
        refw = eng.neighbors_of(cx, cy, exclude_wid=(wid if split == "train" else None))
        NREFS[wid] = [eng._ref_grtvt(r) for r, _ in refw] if len(refw) >= eng.NEED_REFS else []
    return NREFS


def _run_source(eng, paths, build_fn, Pkof, SEED_V26, N_SEED, WELL_CHUNK, nrefs=None):
    """build_fn(hw,tw_tvt,tw_gr,P[,refs]) で base→affine→smooth PF。{(wid,nm):dict(mean,std)},{(wid,nm):(a,b)}"""
    ENS = {}; COEF = {}
    for j, lab in enumerate(eng.PF_BANK):
        Pk = Pkof(lab); nm = f"pf_{j + 1}"; t0 = time.perf_counter(); inps = []; keys = []
        for (hp, tp) in paths:
            wid = os.path.basename(str(hp)).split("__")[0]
            hw = pd.read_csv(hp); tw = pd.read_csv(tp).sort_values("TVT")
            tw_tvt = tw["TVT"].to_numpy(float); tw_gr = tw["GR"].to_numpy(float)
            base = build_fn(hw, tw_tvt, tw_gr, Pk, nrefs[wid]) if nrefs is not None else build_fn(hw, tw_tvt, tw_gr, Pk)
            kgr, ktvt = _kgr_ktvt(eng, hw, tw_gr, Pk)
            base2, a, b = apply_affine(eng, base, kgr, ktvt, Pk)
            base2 = eng._attach_anchor(base2, wid, physics=False)   # struct-free(ancs=1e9)
            inps.append(base2); keys.append(wid); COEF[(wid, nm)] = (a, b)
        outs = eng.run_smoother_bank(inps, Pk, eng.DEC_BANK.get(lab, 1), SEED_V26, N_SEED, WELL_CHUNK)
        for i, wid in enumerate(keys):
            ENS[(wid, nm)] = outs[i]
        if eng.V26_DEVICE == "cuda":
            import torch; torch.cuda.empty_cache()
        print(f"      [affine/{build_fn.__name__}] {lab} done {time.perf_counter()-t0:.1f}s", flush=True)
    return ENS, COEF


def run_affine(eng, paths, *, SMOOTH_LAG, N_SEED, SEED_V26, WELL_CHUNK, split):
    """affine 15本(tw/self/nbr × PF_BANK)を回す。"""
    def Pkof(lab): return {**eng.PF_LIB[lab], "smooth_lag": SMOOTH_LAG}
    NREFS = compute_nrefs(eng, paths, split)
    print("[v87] affine-tw ...", flush=True)
    AFF_TW, C_TW = _run_source(eng, paths, eng.build_smoother_inputs, Pkof, SEED_V26, N_SEED, WELL_CHUNK)
    print("[v87] affine-self ...", flush=True)
    AFF_SF, C_SF = _run_source(eng, paths, eng.build_inputs_self, Pkof, SEED_V26, N_SEED, WELL_CHUNK)
    print("[v87] affine-nbr ...", flush=True)
    AFF_NB, C_NB = _run_source(eng, paths, eng.build_inputs_nbr, Pkof, SEED_V26, N_SEED, WELL_CHUNK, nrefs=NREFS)
    return dict(tw=AFF_TW, self=AFF_SF, nbr=AFF_NB), dict(tw=C_TW, self=C_SF, nbr=C_NB)


def assemble_affine_features(eng, paths, AFF, COEF, get_nonaff):
    """affine 候補 + 違い特徴を id 付き DataFrame に。get_nonaff(ens,wid,nm)->array|None(非affine tw/self/nbr平均)。"""
    PF_BANK = eng.PF_BANK; NP = len(PF_BANK)
    AFF_TW, AFF_SF, AFF_NB = AFF["tw"], AFF["self"], AFF["nbr"]
    C_TW, C_SF, C_NB = COEF["tw"], COEF["self"], COEF["nbr"]
    parts = []
    for (hp, tp) in paths:
        wid = os.path.basename(str(hp)).split("__")[0]
        hw = pd.read_csv(hp); ev = hw[hw["TVT_input"].isna()]
        if len(ev) == 0:
            continue
        n = len(ev); ids = [f"{wid}_{i}" for i in ev.index]
        kn = hw[hw["TVT_input"].notna()]; lk = float(kn["TVT_input"].iloc[-1]) if len(kn) else np.nan
        d = {"id": ids}

        def gm(dic, nm):
            r = dic.get((wid, nm)); return np.asarray(r["mean"], float) if r else np.full(n, np.nan)

        def gs(dic, nm):
            r = dic.get((wid, nm)); return np.asarray(r["std"], float) if r else np.full(n, np.nan)

        TWs, SFs, NBs = [], [], []
        for j in range(NP):
            suf = f"_{j + 1}"; nm = f"pf_{j + 1}"
            at, asf, anb = gm(AFF_TW, nm), gm(AFF_SF, nm), gm(AFF_NB, nm)
            TWs.append(at); SFs.append(asf); NBs.append(anb)
            d[f"pf_afftw{suf}"] = at.astype(np.float32); d[f"pf_afftw_std{suf}"] = gs(AFF_TW, nm).astype(np.float32)
            d[f"pf_afftw_delta{suf}"] = (at - lk).astype(np.float32)
            d[f"pf_affself{suf}"] = asf.astype(np.float32); d[f"pf_affself_std{suf}"] = gs(AFF_SF, nm).astype(np.float32)
            d[f"pf_affself_delta{suf}"] = (asf - lk).astype(np.float32)
            d[f"pf_affnbr{suf}"] = anb.astype(np.float32); d[f"pf_affnbr_std{suf}"] = gs(AFF_NB, nm).astype(np.float32)
            d[f"pf_affnbr_delta{suf}"] = (anb - lk).astype(np.float32)
            tw0, sf0, nb0 = get_nonaff("F", wid, nm), get_nonaff("S", wid, nm), get_nonaff("N", wid, nm)
            if tw0 is not None: d[f"pf_afftw_vs_tw{suf}"] = (at - tw0[:n]).astype(np.float32)
            if sf0 is not None: d[f"pf_affself_vs_self{suf}"] = (asf - sf0[:n]).astype(np.float32)
            if nb0 is not None: d[f"pf_affnbr_vs_nbr{suf}"] = (anb - nb0[:n]).astype(np.float32)
            st = np.stack([at, asf, anb], axis=1)
            d[f"pf_afftw_vs_affself{suf}"] = (at - asf).astype(np.float32)
            d[f"pf_afftw_vs_affnbr{suf}"] = (at - anb).astype(np.float32)
            d[f"pf_affself_vs_affnbr{suf}"] = (asf - anb).astype(np.float32)
            d[f"pf_aff_agree_std{suf}"] = np.nanstd(st, axis=1).astype(np.float32)
            d[f"pf_aff_agree_range{suf}"] = (np.nanmax(st, axis=1) - np.nanmin(st, axis=1)).astype(np.float32)
        TW = np.stack(TWs, 1); SF = np.stack(SFs, 1); NB = np.stack(NBs, 1)
        for tag, M in (("afftw", TW), ("affself", SF), ("affnbr", NB)):
            d[f"pf_{tag}_mean"] = np.nanmean(M, 1).astype(np.float32)
            d[f"pf_{tag}_med"] = np.nanmedian(M, 1).astype(np.float32)
            d[f"pf_{tag}_std_between"] = np.nanstd(M, 1).astype(np.float32)
            d[f"pf_{tag}_mean_delta"] = (np.nanmean(M, 1) - lk).astype(np.float32)
        ALL = np.concatenate([TW, SF, NB], axis=1)
        d["pf_aff_all_mean"] = np.nanmean(ALL, 1).astype(np.float32)
        d["pf_aff_all_med"] = np.nanmedian(ALL, 1).astype(np.float32)
        d["pf_aff_all_std_between"] = np.nanstd(ALL, 1).astype(np.float32)
        d["pf_aff_all_range_between"] = (np.nanmax(ALL, 1) - np.nanmin(ALL, 1)).astype(np.float32)
        tw0m = [get_nonaff("F", wid, f"pf_{j+1}") for j in range(NP)]
        if all(x is not None for x in tw0m):
            tw0m = np.stack([x[:n] for x in tw0m], 1)
            d["pf_afftw_mean_vs_tw"] = (np.nanmean(TW, 1) - np.nanmean(tw0m, 1)).astype(np.float32)
        for tag, C in (("tw", C_TW), ("self", C_SF), ("nbr", C_NB)):
            aa = np.array([C.get((wid, f"pf_{j+1}"), (np.nan, np.nan))[0] for j in range(NP)], float)
            bb = np.array([C.get((wid, f"pf_{j+1}"), (np.nan, np.nan))[1] for j in range(NP)], float)
            d[f"aff_a_{tag}"] = np.float32(np.nanmean(aa)); d[f"aff_b_{tag}"] = np.float32(np.nanmean(bb))
        parts.append(pd.DataFrame(d))
    return pd.concat(parts, ignore_index=True)
