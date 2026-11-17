# -*- coding: utf-8 -*-
"""
v89 学習表現emission画像 — v50 StageAの対比学習エンコーダ(ゴースト抑制)をemissionに。
================================================================================================
固定表現(生GR/NCC)でなく **v50の学習表現**(真の対応 vs サイクルゴーストを分離, 判別3.1倍)を使う。
struct/PF/nbr 不使用。flat中心(lk)。foldごとにOOFエンコーダ(capaug_f{fold})でリーク防止。
チャネル: c_learn(学習類似度) c_raw(生尤度=product of experts) c_self(自prefix) 物理(lk/mds/z)
出力: v89/artifacts_v89/learned_img.pkl (v64トレーナ互換, fold=sorted index%5)
使い方: NWELLS= でスモーク / 本番そのまま
"""
import os, sys, time, pickle
import numpy as np, pandas as pd, torch
P0 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(P0, "v50")); sys.path.insert(0, os.path.join(P0, "v64"))
import stageA_v50 as sa
import build_images_v64 as bi
ncc_img, winsum, grid_from_pairs, smooth = bi.ncc_img, bi.winsum, bi.grid_from_pairs, bi.smooth
OFF, ND, NB, STEP, STR, W16, W64, SIG_LK = bi.OFF, bi.ND, bi.NB, bi.STEP, bi.STR, bi.W16, bi.W64, bi.SIG_LK
ART = os.path.join(P0, "v89", "artifacts_v89"); os.makedirs(ART, exist_ok=True)
TRAIN = os.path.join(P0, "rogii-wellbore-geology-prediction", "train")
OUT = os.path.join(ART, os.environ.get("OUT", "learned_img.pkl"))
DEV = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
LOGIT = float(os.environ.get("LOGIT", "8.0"))
PROD = os.environ.get("PROD", "1") == "1"   # ★product of experts(学習×生)チャネル
CENTER_NPZ = os.environ.get("CENTER_NPZ", "")   # ★bootstrap: 前段OOF予測を中心に(空=flat lk)
_CENTER = {}
if CENTER_NPZ:
    for _f in CENTER_NPZ.split(","):
        _z = np.load(_f)
        for _k in _z.files: _CENTER[_k] = _z[_k]
    print(f"[v89] bootstrap center: {len(_CENTER)}井")


CAP_CH, CAP_DIM, CAP_DILS = 128, 96, (1, 2, 4, 8, 16, 32)   # ★capaug/capsyn構成(stageA_ablation.json)
FAMS = os.environ.get("FAMS", "capaug").split(",")   # ★学習表現系統(capaug/capsyn)を複数チャネルに


def load_enc(fam, f):
    ck = torch.load(os.path.join(P0, "v50", "artifacts_v50", f"stageA_enc{fam}_f{f}.pt"), map_location=DEV)
    encA = sa.Enc1D(8, CAP_CH, CAP_DIM, CAP_DILS).to(DEV); encA.load_state_dict(ck["encA"]); encA.eval()
    encB = sa.Enc1D(7, CAP_CH, CAP_DIM, CAP_DILS).to(DEV); encB.load_state_dict(ck["encB"]); encB.eval()
    return encA, encB


def main():
    t0 = time.time()
    ENC = {(fam, f): load_enc(fam, f) for fam in FAMS for f in range(5)}
    wells = sorted(w.split("__")[0] for w in os.listdir(TRAIN) if w.endswith("__horizontal_well.csv"))
    if os.environ.get("NWELLS"): wells = wells[:int(os.environ["NWELLS"])]
    OUTD = {}
    for wi, wn in enumerate(wells):
        try:
            hw = pd.read_csv(os.path.join(TRAIN, f"{wn}__horizontal_well.csv"))
            tw = pd.read_csv(os.path.join(TRAIN, f"{wn}__typewell.csv")).sort_values("TVT")
        except Exception:
            continue
        if "TVT" not in hw.columns or len(tw) < 8: continue
        kn = hw["TVT_input"].notna().to_numpy(); ev = ~kn
        if kn.sum() < 10 or ev.sum() < 100: continue
        tw_tvt = tw["TVT"].to_numpy(float); tw_gr = tw["GR"].to_numpy(float)
        m = np.isfinite(tw_tvt) & np.isfinite(tw_gr); tw_tvt, tw_gr = tw_tvt[m], tw_gr[m]
        if len(tw_tvt) < 8: continue
        gmin, gmax = float(tw_tvt.min()), float(tw_tvt.max())
        gg = np.interp(np.arange(gmin, gmax + STEP, STEP), tw_tvt, tw_gr).astype(np.float32)
        Fh = sa.feats_h(hw["GR"].to_numpy(float), hw["MD"].to_numpy(float), hw["Z"].to_numpy(float))
        Ft = sa.feats_t(gg)
        fold = wi % 5
        a_evs = []                                               # 系統ごとの水平埋め込み
        b_all = {}
        for fam in FAMS:
            encA, encB = ENC[(fam, fold)]                        # ★OOFエンコーダ
            with torch.no_grad():
                a = encA(torch.from_numpy(Fh).to(DEV)).cpu().numpy()
                b_all[fam] = encB(torch.from_numpy(Ft).to(DEV)).cpu().numpy()
            a_evs.append((fam, a[ev]))
        a_ev = a_evs[0][1]                                        # 主系統(delta targetは共通)
        fb = float(np.nanmean(tw_gr))
        gr_all = smooth(hw["GR"].to_numpy(float), fb, 2)
        g = gr_all[ev]; y = hw["TVT"].to_numpy(float)[ev]
        kn_tvt = hw["TVT_input"].to_numpy(float)[kn]; lk = float(kn_tvt[-1])
        z_all = hw["Z"].to_numpy(float); md_all = hw["MD"].to_numpy(float)
        if _CENTER and wn in _CENTER:
            stv = np.asarray(_CENTER[wn], np.float64)            # ★bootstrap: 前段予測を中心
            if len(stv) < len(g): stv = np.pad(stv, (0, len(g) - len(stv)), mode="edge")
        else:
            stv = np.full(len(g), lk, np.float64)                # flat中心
        T = min(len(g), len(stv), len(a_ev)); g = g[:T]; yv = y[:T]; stv = stv[:T]; a_ev = a_ev[:T]
        pos = stv[:, None] + OFF[None, :]                        # (T,ND) 各δのTVT
        # 学習類似度 c_learn[t,δ] = a_ev[t]·b[idx(lk+δ)] を系統ごとに
        idx = np.clip(np.round((pos - gmin) / STEP).astype(int), 0, len(gg) - 1)
        c_learn_fams = []
        for (fam, aev) in a_evs:
            aev = aev[:T]
            bg = b_all[fam][idx]                                 # (T,ND,D)
            cl = np.einsum("td,tnd->tn", aev, bg) * LOGIT
            c_learn_fams.append(1.0 / (1.0 + np.exp(-cl)))       # [0,1]
        c_learn = c_learn_fams[0]                                # 主系統(prod用)
        # 生尤度(product of experts)
        tws = smooth(tw_gr, fb, 2)
        Ttw = np.interp(pos, tw_tvt, tws)
        resid = gr_all[kn] - np.interp(kn_tvt, tw_tvt, tws); sig = float(np.nanstd(resid))
        sig = 15.0 if not np.isfinite(sig) or sig < 3 else max(sig, 5.0)
        c_raw = np.exp(-0.5 * ((g[:, None] - Ttw) / sig) ** 2)
        c_ncc, _ = ncc_img(g, Ttw, W64)
        # self(自prefix)
        gs_, vs_ = grid_from_pairs(kn_tvt, gr_all[kn], pos.min() - 1, pos.max() + 1)
        Tself = np.interp(pos, gs_, np.where(np.isfinite(vs_), vs_, np.interp(gs_, tw_tvt, tws)))
        c_self, _ = ncc_img(g, Tself, W16)
        # 物理
        nb_ = (T + STR - 1) // STR; sel = np.clip(np.arange(nb_) * STR + STR // 2, 0, T - 1)
        c_lk = np.repeat(np.exp(-0.5 * (OFF / SIG_LK) ** 2)[None, :], T, 0)
        md_lk = float(md_all[kn][-1]); mds_row = np.clip((md_all[ev][:T] - md_lk) / 1000, 0, 3) / 3
        c_mds = np.repeat(mds_row[:, None], ND, 1)
        z_lk = float(z_all[kn][-1]); zr = np.clip((z_all[ev][:T] - z_lk) / 50, -3, 3) / 3
        c_z = np.repeat(zr[:, None], ND, 1)
        chs = list(c_learn_fams) + [c_raw, c_ncc, c_self, c_lk, c_mds, c_z]   # ★系統ごとの学習emission
        if PROD:                                 # ★product of experts(v50最強): 学習×生 / 学習×ncc
            chs += [c_learn * c_raw, c_learn * np.clip(c_ncc, 0, 1)]
        EV = np.stack([c[sel] for c in chs], 0).astype(np.float16)
        bid = np.minimum(np.arange(T) // STR, nb_ - 1)
        dtrue = np.full(nb_, np.nan); wgt = np.zeros(nb_); dely = yv - stv
        for bb in range(nb_):
            mm = bid == bb
            if mm.sum(): dtrue[bb] = float(np.nanmedian(dely[mm])); wgt[bb] = float(np.isfinite(dely[mm]).sum())
        OUTD[wn] = dict(EV=EV, dtrue=dtrue.astype(np.float32), wgt=wgt.astype(np.float32),
                        st=stv.astype(np.float32), bid=bid.astype(np.int32), y_ev=yv.astype(np.float32))
        if (wi + 1) % 100 == 0: print(f"  {wi+1}/{len(wells)} ({time.time()-t0:.0f}s)", flush=True)
    pickle.dump(OUTD, open(OUT, "wb"), protocol=4)
    print(f"saved {os.path.basename(OUT)}: {len(OUTD)}井 {len(chs)}ch "
          f"({os.path.getsize(OUT)/(1<<20):.0f}MB, {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
