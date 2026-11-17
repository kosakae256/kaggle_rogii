# -*- coding: utf-8 -*-
"""
v89 SDF(iter1) 提出 — 非逐次localizer。最終段を softmax分類 → SDF(変位場)ヘッドに置換した上位互換版。
================================================================================
チェーン: flat中心 → ddoof(softmax) → boot(softmax) → **SDF**(変位投票読み出し)、各段5foldアンサンブル。
  ddoof/boot は中心生成用(pmean)。最終SDF段のみ変位投票 dh = Σ softmax(-|s|/τ)(offs+s) + ref。
SDF iter1 は learned_boot2.pkl(=boot中心)上で学習 → 提出も boot中心の像に SDF を当てる。全773井 overall=11.81。
使い方:
  検証: VERIFY=30 python v89/submit_sdf_v89.py   # 単fold連鎖 vs SDF OOF 照合
  提出: python v89/submit_sdf_v89.py              # test → submission.csv
"""
import os, sys, io, time, glob
try: sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
except Exception: pass
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F

P0 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(P0, "v89"))
import submit_unet_v89 as U        # build_ev, load_enc, DecV(softmax), 定数, offs_t を流用
build_ev, load_enc, DEV, offs_t = U.build_ev, U.load_enc, U.DEV, U.offs_t
NB, ND, STEP = U.NB, U.ND, U.STEP
ART, ENCDIR, DATA = U.ART, U.ENCDIR, U.DATA
FAMS = U.FAMS
DILS_DDIL = U.DILS_DDIL
TAU = float(os.environ.get("TAU", "4.0"))       # 変位投票温度(ft, 学習時と一致)
STAGES = ["ddoof", "boot", "sdf_ddil"]           # ddoof/boot=softmax中心、最終=SDF


class DecVSDF(nn.Module):
    """train_sdf_v89.py の DecV(disp head, AUXCE=0)。out=変位場(ft)。"""
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
        self.disp = nn.Conv2d(ch, 1, 1)
        self.ref = nn.Linear(2 * ch, 1)
    def forward(self, x):
        h = self.inp(x)
        for blk, nm in zip(self.blocks, self.norms):
            h = nm(h + blk(h))
        g = h.mean(3).transpose(1, 2); g2, _ = self.gru(g)
        h = h + self.mix(g2).transpose(1, 2)[:, :, :, None]
        return self.disp(h)[:, 0], 2.0 * torch.tanh(self.ref(g2)[:, :, 0])


def load_sdf(f):
    sd = torch.load(ART / f"v89_sdf_ddil_f{f}.pt", map_location=DEV)
    net = DecVSDF(sd["inp.weight"].shape[1]).to(DEV); net.load_state_dict(sd, strict=True); net.eval()
    return net


def readout_sdf(disp, ref):
    vote = offs_t[None, None, :] + disp
    w = torch.softmax(-disp.abs() / TAU, dim=-1)
    return (w * vote).sum(-1) + ref


def stage_softmax(hw, tw, center, models5, ENC):
    """softmax段(ddoof/boot): 5fold平均pmeanの中心を返す。"""
    Psum = rfsum = base = None
    for f in range(5):
        r = build_ev(hw, tw, center, ENC[f][0], ENC[f][1])
        if r is None: return None
        EV, stv, bid, evidx, T = r
        with torch.no_grad():
            lg, rf = models5[f](torch.from_numpy(EV)[None].to(DEV))
            P = torch.softmax(lg[0, :EV.shape[1]], -1)
        Psum = P if Psum is None else Psum + P
        rfv = rf[0, :EV.shape[1]]; rfsum = rfv if rfsum is None else rfsum + rfv
        base = (stv, bid, evidx)
    stv, bid, evidx = base
    dh = ((Psum / 5.0 @ offs_t) + rfsum / 5.0).cpu().numpy()
    return stv + np.asarray(dh, np.float64)[bid], evidx


def stage_sdf(hw, tw, center, models5, ENC):
    """SDF最終段: 5fold平均の変位場を投票読み出し。"""
    Dsum = rfsum = base = None
    for f in range(5):
        r = build_ev(hw, tw, center, ENC[f][0], ENC[f][1])
        if r is None: return None
        EV, stv, bid, evidx, T = r
        with torch.no_grad():
            disp, rf = models5[f](torch.from_numpy(EV)[None].to(DEV))
        d = disp[0, :EV.shape[1]]; rfv = rf[0, :EV.shape[1]]
        Dsum = d if Dsum is None else Dsum + d
        rfsum = rfv if rfsum is None else rfsum + rfv
        base = (stv, bid, evidx)
    stv, bid, evidx = base
    dh = readout_sdf((Dsum / 5.0)[None], (rfsum / 5.0)[None])[0].cpu().numpy()
    return stv + np.asarray(dh, np.float64)[bid], evidx


def chain(hw, tw, ENC, MODELS):
    center = None; evidx = None
    for st in STAGES:
        fn = stage_sdf if st == "sdf_ddil" else stage_softmax
        out = fn(hw, tw, center, MODELS[st], ENC)
        if out is None: return None
        center, evidx = out
    return center, evidx


def main():
    t0 = time.time()
    print(f"[v89 SDF submit] DEV={DEV} ART={ART} STAGES={STAGES} TAU={TAU}")
    ENC = {f: load_enc(FAMS[0], f) for f in range(5)}
    MODELS = {"ddoof": {f: U.load_model("ddoof", f) for f in range(5)},
              "boot": {f: U.load_model("boot", f) for f in range(5)},
              "sdf_ddil": {f: load_sdf(f) for f in range(5)}}
    print(f"  loaded encoders + ddoof/boot(softmax) + sdf ({time.time()-t0:.0f}s)")

    VERIFY = int(os.environ.get("VERIFY", "0"))
    if VERIFY > 0:
        oof = {}
        for f in range(5): oof.update(dict(np.load(ART / f"v89_sdf_ddil_f{f}_oof.npz")))
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
            center = None; ok = True
            for st in STAGES:
                r = build_ev(hw, tw, center, ENC[fold][0], ENC[fold][1])
                if r is None: ok = False; break
                EV, stv, bid, evidx, T = r
                with torch.no_grad():
                    if st == "sdf_ddil":
                        disp, rf = MODELS[st][fold](torch.from_numpy(EV)[None].to(DEV))
                        dh = readout_sdf(disp[:, :EV.shape[1]], rf[:, :EV.shape[1]])[0].cpu().numpy()
                    else:
                        lg, rf = MODELS[st][fold](torch.from_numpy(EV)[None].to(DEV))
                        P = torch.softmax(lg[0, :EV.shape[1]], -1)
                        dh = ((P @ offs_t) + rf[0, :EV.shape[1]]).cpu().numpy()
                center = stv + np.asarray(dh, np.float64)[bid]
            if not ok: continue
            ref = np.asarray(oof[wn], np.float64)[:len(center)]
            d = center[:len(ref)] - ref[:len(center)]
            diffs.append(np.sqrt(np.mean(d[np.isfinite(d)] ** 2))); checked += 1
        diffs = np.array(diffs)
        print(f"\n[VERIFY] {checked}井 単fold連鎖 vs SDF OOF: |diff|RMSE 中央={np.median(diffs):.4f}ft 最大={diffs.max():.4f}ft (~0で再現OK)")
        return

    TESTD = os.path.join(str(DATA), "test")
    wids = sorted(set(os.path.basename(f).split("__")[0] for f in glob.glob(os.path.join(TESTD, "*__horizontal_well.csv"))))
    pids = []; pvals = []; nskip = 0
    for k, wid in enumerate(wids):
        try:
            hw = pd.read_csv(os.path.join(TESTD, f"{wid}__horizontal_well.csv"))
            tw = pd.read_csv(os.path.join(TESTD, f"{wid}__typewell.csv")).sort_values("TVT")
        except Exception:
            nskip += 1; continue
        out = chain(hw, tw, ENC, MODELS)
        if out is None: nskip += 1; continue
        z, evidx = out
        pids += [f"{wid}_{int(i)}" for i in evidx]; pvals += list(z)
        if (k + 1) % 25 == 0: print(f"  {k+1}/{len(wids)} ({time.time()-t0:.0f}s)", flush=True)
    pred = pd.DataFrame({"id": pids, "tvt": pvals})
    print(f"predicted {len(pred)} rows / skip={nskip} ({time.time()-t0:.0f}s)")
    ss = pd.read_csv(os.path.join(str(DATA), "sample_submission.csv"))
    sub = ss[["id"]].merge(pred, on="id", how="left")
    miss = sub["tvt"].isna()
    if miss.any():
        lkmap = {}
        for wid in set(i.rsplit("_", 1)[0] for i in sub.loc[miss, "id"]):
            v = pd.read_csv(os.path.join(TESTD, f"{wid}__horizontal_well.csv"), usecols=["TVT_input"])["TVT_input"].dropna()
            lkmap[wid] = float(v.iloc[-1]) if len(v) else 0.0
        sub.loc[miss, "tvt"] = [lkmap[i.rsplit("_", 1)[0]] for i in sub.loc[miss, "id"]]
        print(f"lkフォールバック: {int(miss.sum())} rows")
    sub.to_csv("submission.csv", index=False)
    print(f"submission.csv: {len(sub)} rows")


if __name__ == "__main__":
    main()
