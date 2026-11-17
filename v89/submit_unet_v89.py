# -*- coding: utf-8 -*-
"""
v89 U-Net 提出 — 学習表現(v50 capaug)×非逐次DecV(ddil)。PF/struct/nbr 不使用。
================================================================================
3段 bootstrap チェーン(各段 5fold アンサンブル):
  flat中心 → ddoof(v89_ddoof_f*) → boot(v89_boot_f*) → boot2(v89_boot2_f*=best 10.88)
各段: fold f ごとに encoder_f で像を作り model_f を回し、fold平均を取る。前段平均を次段の中心にする。
出力予測 = 各evバ行の z = st + dh[bid]  (MODE=pmean 事後平均 / argmax モード)

MODE(env):    pmean(既定, U-Netの標準出力=10.88の実体) | argmax(δ分布のモード)
STAGES(env):  ddoof,boot,boot2 (既定)。短縮可(例 STAGES=ddoof で1段flat)
使い方:
  検証: MODE=pmean VERIFY=30 python v89/submit_unet_v89.py   # train30井でOOF(単fold)再現照合
  提出: python v89/submit_unet_v89.py                         # test全井 → submission.csv
Kaggle: DATA/ART は /kaggle/input を自動検出。encoder(v50 capaug f0-4)と model(v89_*_f0-4)を dataset に。
"""
import os, sys, time, glob, json, pickle, io
try: sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
except Exception: pass
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path

P0 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(P0, "v50")); sys.path.insert(0, os.path.join(P0, "v64"))
import stageA_v50 as sa
import build_images_v64 as bi
ncc_img, winsum, grid_from_pairs, smooth = bi.ncc_img, bi.winsum, bi.grid_from_pairs, bi.smooth
OFF, ND, NB, STEP, STR, W16, W64, SIG_LK = bi.OFF, bi.ND, bi.NB, bi.STEP, bi.STR, bi.W16, bi.W64, bi.SIG_LK

DEV = "cuda" if torch.cuda.is_available() else "cpu"
CAP_CH, CAP_DIM, CAP_DILS = 128, 96, (1, 2, 4, 8, 16, 32)   # v50 capaug/capsyn 構成
FAMS = os.environ.get("FAMS", "capaug").split(",")
LOGIT = float(os.environ.get("LOGIT", "8.0"))
PROD = os.environ.get("PROD", "1") == "1"
MODE = os.environ.get("MODE", "pmean")          # pmean | argmax
STAGES = os.environ.get("STAGES", "ddoof,boot,boot2").split(",")
DILS_DDIL = [(1, 1), (2, 1), (4, 2), (8, 4), (16, 8)]        # ARCH=ddil(全段これ)

# ---- パス検出(Kaggle/ローカル) ----
_C = [Path("/kaggle/input/rogii-v89-artifacts"), Path(os.path.join(P0, "v64", "artifacts_v64"))]
ART = next((p for p in _C if p.exists()), Path(os.path.join(P0, "v64", "artifacts_v64")))
_CE = [Path("/kaggle/input/rogii-v89-artifacts"), Path(os.path.join(P0, "v50", "artifacts_v50"))]
ENCDIR = next((p for p in _CE if (p / "stageA_enccapaug_f0.pt").exists()), Path(os.path.join(P0, "v50", "artifacts_v50")))
def _detect_data():
    for _p in ["/kaggle/input/rogii-wellbore-geology-prediction",
               "/kaggle/input/competitions/rogii-wellbore-geology-prediction",
               os.path.join(P0, "rogii-wellbore-geology-prediction")]:
        if os.path.exists(_p): return Path(_p)
    _h = glob.glob("/kaggle/input/**/sample_submission.csv", recursive=True)
    return Path(os.path.dirname(_h[0]))
DATA = _detect_data()


class DecV(nn.Module):
    """train_unet_v89.py の DecV(ARCH=ddil) と同一。"""
    def __init__(self, cin, ch=128):
        super().__init__()
        self.inp = nn.Conv2d(cin, ch, (5, 5), padding=(2, 2))
        self.blocks = nn.ModuleList(); self.norms = nn.ModuleList()
        for dt, dd in DILS_DDIL:
            self.blocks.append(nn.Sequential(
                nn.Conv2d(ch, ch, (5, 5), padding=(2 * dt, 2 * dd), dilation=(dt, dd)), nn.GELU(),
                nn.Conv2d(ch, ch, 1)))
            self.norms.append(nn.GroupNorm(8, ch))
        self.gru = nn.GRU(ch, ch, batch_first=True, bidirectional=True)
        self.mix = nn.Linear(2 * ch, ch)
        self.out = nn.Conv2d(ch, 1, 1)
        self.ref = nn.Linear(2 * ch, 1)
    def forward(self, x):
        h = self.inp(x)
        for blk, nm in zip(self.blocks, self.norms):
            h = nm(h + blk(h))
        g = h.mean(3).transpose(1, 2)
        g2, _ = self.gru(g)
        h = h + self.mix(g2).transpose(1, 2)[:, :, :, None]
        return self.out(h)[:, 0], 2.0 * torch.tanh(self.ref(g2)[:, :, 0])


def load_enc(fam, f):
    ck = torch.load(ENCDIR / f"stageA_enc{fam}_f{f}.pt", map_location=DEV)
    encA = sa.Enc1D(8, CAP_CH, CAP_DIM, CAP_DILS).to(DEV); encA.load_state_dict(ck["encA"]); encA.eval()
    encB = sa.Enc1D(7, CAP_CH, CAP_DIM, CAP_DILS).to(DEV); encB.load_state_dict(ck["encB"]); encB.eval()
    return encA, encB


def load_model(tag, f):
    p = ART / f"v89_{tag}_f{f}.pt"
    sd = torch.load(p, map_location=DEV)
    cin = sd["inp.weight"].shape[1]
    net = DecV(cin).to(DEV); net.load_state_dict(sd, strict=True); net.eval()
    return net


offs_t = torch.arange(-NB, NB + 1, device=DEV).float() * STEP


def build_ev(hw, tw, center, encA, encB):
    """build_learned_img_v89.py の1井分を再現。center=None→flat lk。
    返り値: EV(CIN,nb_,ND), stv(T,), bid(T,), evidx(T,) 絶対行index, T. Noneで skip。"""
    kn = hw["TVT_input"].notna().to_numpy(); ev = ~kn
    if kn.sum() < 10 or ev.sum() < 100: return None
    tw_tvt = tw["TVT"].to_numpy(float); tw_gr = tw["GR"].to_numpy(float)
    m = np.isfinite(tw_tvt) & np.isfinite(tw_gr); tw_tvt, tw_gr = tw_tvt[m], tw_gr[m]
    if len(tw_tvt) < 8: return None
    gmin, gmax = float(tw_tvt.min()), float(tw_tvt.max())
    gg = np.interp(np.arange(gmin, gmax + STEP, STEP), tw_tvt, tw_gr).astype(np.float32)
    Fh = sa.feats_h(hw["GR"].to_numpy(float), hw["MD"].to_numpy(float), hw["Z"].to_numpy(float))
    Ft = sa.feats_t(gg)
    with torch.no_grad():
        a = encA(torch.from_numpy(Fh).to(DEV)).cpu().numpy()
        b = encB(torch.from_numpy(Ft).to(DEV)).cpu().numpy()
    a_ev = a[ev]
    fb = float(np.nanmean(tw_gr))
    gr_all = smooth(hw["GR"].to_numpy(float), fb, 2)
    g = gr_all[ev]; kn_tvt = hw["TVT_input"].to_numpy(float)[kn]; lk = float(kn_tvt[-1])
    z_all = hw["Z"].to_numpy(float); md_all = hw["MD"].to_numpy(float)
    evidx = np.where(ev)[0]
    if center is not None:
        stv = np.asarray(center, np.float64)
        if len(stv) < len(g): stv = np.pad(stv, (0, len(g) - len(stv)), mode="edge")
    else:
        stv = np.full(len(g), lk, np.float64)
    T = min(len(g), len(stv), len(a_ev)); g = g[:T]; stv = stv[:T]; a_ev = a_ev[:T]; evidx = evidx[:T]
    pos = stv[:, None] + OFF[None, :]
    idx = np.clip(np.round((pos - gmin) / STEP).astype(int), 0, len(gg) - 1)
    bg = b[idx]                                             # (T,ND,D)
    cl = np.einsum("td,tnd->tn", a_ev, bg) * LOGIT
    c_learn = 1.0 / (1.0 + np.exp(-cl))
    tws = smooth(tw_gr, fb, 2)
    Ttw = np.interp(pos, tw_tvt, tws)
    resid = gr_all[kn] - np.interp(kn_tvt, tw_tvt, tws); sig = float(np.nanstd(resid))
    sig = 15.0 if not np.isfinite(sig) or sig < 3 else max(sig, 5.0)
    c_raw = np.exp(-0.5 * ((g[:, None] - Ttw) / sig) ** 2)
    c_ncc, _ = ncc_img(g, Ttw, W64)
    gs_, vs_ = grid_from_pairs(kn_tvt, gr_all[kn], pos.min() - 1, pos.max() + 1)
    Tself = np.interp(pos, gs_, np.where(np.isfinite(vs_), vs_, np.interp(gs_, tw_tvt, tws)))
    c_self, _ = ncc_img(g, Tself, W16)
    nb_ = (T + STR - 1) // STR; sel = np.clip(np.arange(nb_) * STR + STR // 2, 0, T - 1)
    c_lk = np.repeat(np.exp(-0.5 * (OFF / SIG_LK) ** 2)[None, :], T, 0)
    md_lk = float(md_all[kn][-1]); mds_row = np.clip((md_all[ev][:T] - md_lk) / 1000, 0, 3) / 3
    c_mds = np.repeat(mds_row[:, None], ND, 1)
    z_lk = float(z_all[kn][-1]); zr = np.clip((z_all[ev][:T] - z_lk) / 50, -3, 3) / 3
    c_z = np.repeat(zr[:, None], ND, 1)
    chs = [c_learn, c_raw, c_ncc, c_self, c_lk, c_mds, c_z]
    if PROD:
        chs += [c_learn * c_raw, c_learn * np.clip(c_ncc, 0, 1)]
    EV = np.stack([c[sel] for c in chs], 0).astype(np.float32)
    bid = np.minimum(np.arange(T) // STR, nb_ - 1)
    return EV, stv, bid, evidx, T


def run_stage(hw, tw, center, models5, encs5, final=False):
    """1段: 5foldの事後平均で中心用z(pmean)を返す。final段のみ MODE=argmax なら平均分布のargmaxを出力。
    中間段は必ずpmean中心(モデルはpmean OOF中心で学習済のため faithful)。enc/model不整合はNone。"""
    Psum = None; rfsum = None; base = None
    for f in range(5):
        r = build_ev(hw, tw, center, encs5[f][0], encs5[f][1])
        if r is None: return None
        EV, stv, bid, evidx, T = r
        x = torch.from_numpy(EV)[None].to(DEV)
        with torch.no_grad():
            lg, rf = models5[f](x)
            P = torch.softmax(lg[0, :EV.shape[1]], -1)
        Psum = P if Psum is None else Psum + P
        rfv = rf[0, :EV.shape[1]]; rfsum = rfv if rfsum is None else rfsum + rfv
        base = (stv, bid, evidx, T)
    stv, bid, evidx, T = base
    Pavg = Psum / 5.0; rfavg = rfsum / 5.0
    dh_pmean = ((Pavg @ offs_t) + rfavg).cpu().numpy()       # 中心用(常にpmean)
    z_center = stv + np.asarray(dh_pmean, np.float64)[bid]
    if final and MODE == "argmax":
        dh_out = offs_t[Pavg.argmax(-1)].cpu().numpy()       # 平均分布のモード(ensemble argmax)
        z_out = stv + np.asarray(dh_out, np.float64)[bid]
    else:
        z_out = z_center
    return z_out, z_center, evidx, T


def chain_predict(hw, tw, ENC, MODELS):
    center = None; evidx = None; z_out = None
    for i, st in enumerate(STAGES):
        out = run_stage(hw, tw, center, MODELS[st], ENC, final=(i == len(STAGES) - 1))
        if out is None: return None
        z_out, center, evidx, T = out                 # 中心は常にpmean(center)、出力のみfinalでMODE反映
    return z_out, evidx                                 # 最終 z(MODE反映), ev行index


def main():
    t0 = time.time()
    print(f"[v89 submit] DEV={DEV} ART={ART} ENC={ENCDIR} DATA={DATA}")
    print(f"  STAGES={STAGES} MODE={MODE} FAMS={FAMS} PROD={PROD}")
    ENC = {f: load_enc(FAMS[0], f) for f in range(5)}
    MODELS = {st: {f: load_model(st, f) for f in range(5)} for st in STAGES}
    print(f"  loaded encoders + {len(STAGES)}stage×5fold models ({time.time()-t0:.0f}s)")

    VERIFY = int(os.environ.get("VERIFY", "0"))
    if VERIFY > 0:
        # train坑井で OOF(単fold: wi%5==f の val井を model_f で) を再現し boot2 OOF と照合
        oof = dict(np.load(ART / "v89_boot2_f0_oof.npz"))
        for f in range(1, 5): oof.update(dict(np.load(ART / f"v89_boot2_f{f}_oof.npz")))
        TR = os.path.join(str(DATA), "train")
        wells = sorted(w.split("__")[0] for w in os.listdir(TR) if w.endswith("__horizontal_well.csv"))
        diffs = []; checked = 0
        for wi, wn in enumerate(wells):
            if checked >= VERIFY: break
            fold = wi % 5
            if wn not in oof: continue
            try:
                hw = pd.read_csv(os.path.join(TR, f"{wn}__horizontal_well.csv"))
                tw = pd.read_csv(os.path.join(TR, f"{wn}__typewell.csv")).sort_values("TVT")
            except Exception:
                continue
            # 単fold連鎖(OOF再現): 各段 model_fold のみ
            center = None
            ok = True
            for st in STAGES:
                r = build_ev(hw, tw, center, ENC[fold][0], ENC[fold][1])
                if r is None: ok = False; break
                EV, stv, bid, evidx, T = r
                x = torch.from_numpy(EV)[None].to(DEV)
                with torch.no_grad():
                    lg, rf = MODELS[st][fold](x)
                    P = torch.softmax(lg[0, :EV.shape[1]], -1)
                    dh = ((P @ offs_t) + rf[0, :EV.shape[1]]).cpu().numpy()   # OOFはpmean
                center = stv + np.asarray(dh, np.float64)[bid]
            if not ok: continue
            ref = np.asarray(oof[wn], np.float64)[:len(center)]
            d = center[:len(ref)] - ref[:len(center)]
            diffs.append(np.sqrt(np.mean(d[np.isfinite(d)] ** 2))); checked += 1
        diffs = np.array(diffs)
        print(f"\n[VERIFY] {checked}井 単fold連鎖 vs boot2 OOF: |diff|RMSE 中央={np.median(diffs):.4f}ft "
              f"最大={diffs.max():.4f}ft (~0なら連鎖再現OK)")
        return

    # ---- 提出: test全井 5fold平均チェーン ----
    TESTD = os.path.join(str(DATA), "test")
    wids = sorted(set(os.path.basename(f).split("__")[0] for f in glob.glob(os.path.join(TESTD, "*__horizontal_well.csv"))))
    pids = []; pvals = []; nskip = 0
    for k, wid in enumerate(wids):
        try:
            hw = pd.read_csv(os.path.join(TESTD, f"{wid}__horizontal_well.csv"))
            tw = pd.read_csv(os.path.join(TESTD, f"{wid}__typewell.csv")).sort_values("TVT")
        except Exception:
            nskip += 1; continue
        out = chain_predict(hw, tw, ENC, MODELS)
        if out is None: nskip += 1; continue
        z, evidx = out
        pids += [f"{wid}_{int(i)}" for i in evidx]; pvals += list(z)
        if (k + 1) % 25 == 0: print(f"  {k+1}/{len(wids)} ({time.time()-t0:.0f}s)", flush=True)
    pred = pd.DataFrame({"id": pids, "tvt": pvals})
    print(f"predicted {len(pred)} rows / skip={nskip} wells ({time.time()-t0:.0f}s)")
    ss = pd.read_csv(os.path.join(str(DATA), "sample_submission.csv"))
    sub = ss[["id"]].merge(pred, on="id", how="left")
    miss = sub["tvt"].isna()
    if miss.any():
        lkmap = {}
        for wid in set(i.rsplit("_", 1)[0] for i in sub.loc[miss, "id"]):
            h = pd.read_csv(os.path.join(TESTD, f"{wid}__horizontal_well.csv"), usecols=["TVT_input"])
            v = h["TVT_input"].dropna(); lkmap[wid] = float(v.iloc[-1]) if len(v) else 0.0
        sub.loc[miss, "tvt"] = [lkmap[i.rsplit("_", 1)[0]] for i in sub.loc[miss, "id"]]
        print(f"lkフォールバック: {int(miss.sum())} rows")
    sub.to_csv("submission.csv", index=False)
    print(f"submission.csv: {len(sub)} rows ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
