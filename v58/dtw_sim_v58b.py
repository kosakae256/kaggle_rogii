# v58b — グリッドFB × NN emission × 物理prior(未測定セル)
#   δ格子(struct±45ft, 0.5ft)上の forward-backward soft事後平均。決定論的(seedなし)。
#   emission logit = β·sim(v50学習類似度, OOF) − LPR·(δ/σs)²  [struct帯prior=v48レシピ]
#   開始 = prefix錨: t=0 に −((δ−d0)/σ0)²/2 を加算(d0 = lk − st(0), v25/v38 ls錨)
#   遷移 = ガウスσδ + ジャンプε(v53 exp02 の高速FBを流用)
#   参照: 同一井戸の PF(pf_1/tw ww平均) / struct(st)
import os, sys, json, time
import numpy as np
import pandas as pd

P0 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(P0, "v51"))
sys.path.insert(0, os.path.join(P0, "v53", "explore"))
import nnpf_v51 as npf
from exp02_pathdecode import fb_smooth                    # (lp, sig_bins, eps, beta) → P
ART = os.path.join(P0, "v58", "artifacts_v58"); os.makedirs(ART, exist_ok=True)
TR = os.path.join(P0, "rogii-wellbore-geology-prediction", "train")
NWELL = int(os.environ.get("NWELL", "40"))
NB = npf.NBAND; ND = 2 * NB + 1; STEP = 0.5
SIG_S = 15.0; SIG0 = 2.0

def main():
    t0 = time.time()
    cand = np.load(os.path.join(P0, "v54", "artifacts_v54", "cand_pf_1_tw.npz"))
    wells_all = sorted(k for k in cand.files if not k.endswith("|w"))
    rng = np.random.default_rng(0)
    wells = [wells_all[i] for i in rng.permutation(len(wells_all))[:NWELL]]
    OFF = (np.arange(ND) - NB) * STEP
    CFG = [(b, s, e, l, a) for b in (2.0, 5.0, 10.0) for s in (1.0, 2.0) for e in (1e-3,)
           for l in (0.0, 0.1) for a in (1,)]
    CFG += [(5.0, 2.0, 1e-3, 0.1, 0)]                     # 錨ablation
    ES = {c: [] for c in CFG}; ES_PF = []; ES_ST = []
    for wi, wn in enumerate(wells):
        sd = npf.SIMD.get(wn)
        if sd is None: continue
        hw = pd.read_csv(os.path.join(TR, f"{wn}__horizontal_well.csv"), usecols=["TVT", "TVT_input"])
        kn = hw["TVT_input"].notna()
        lkv = float(hw.loc[kn, "TVT_input"].iloc[-1])
        ev = (~kn).to_numpy()
        y = hw["TVT"].to_numpy(float)[ev]
        st = np.asarray(sd["st"], np.float64)
        sim = np.asarray(sd["sim"], np.float32).astype(np.float64)
        T = min(len(y), len(st), len(sim))
        finy = np.isfinite(y[:T])
        pf = (cand[f"{wn}|w"][:, None] * cand[wn].astype(np.float64)).sum(0)[:T]
        ES_PF.append((pf - y[:T])[finy]); ES_ST.append((st[:T] - y[:T])[finy])
        simw = np.where(sim[:T] <= -0.999, -0.5, sim[:T])
        d0 = lkv - st[0]
        anch = -0.5 * ((OFF - d0) / SIG0) ** 2
        for (B, S, E, L, A) in CFG:
            em = B * simw - L * (OFF[None, :] / SIG_S) ** 2
            if A: em = em.copy(); em[0] += anch
            lp = em - em.max(1, keepdims=True)
            lp = lp - np.log(np.exp(lp).sum(1, keepdims=True))
            P = fb_smooth(lp, S, E, 1.0)
            zhat = st[:T] + P @ OFF
            ES[(B, S, E, L, A)].append((zhat - y[:T])[finy])
        if (wi + 1) % 10 == 0: print(f"  {wi+1}/{len(wells)} ({time.time()-t0:.0f}s)", flush=True)
    def pooled(es): ee = np.concatenate(es); return float(np.sqrt(np.mean(ee ** 2)))
    print(f"\n参照: PF(pf_1/tw)={pooled(ES_PF):.3f}  struct(st)={pooled(ES_ST):.3f}  (n={len(ES_PF)}井)")
    res = {str(c): pooled(es) for c, es in ES.items()}
    for k, v in sorted(res.items(), key=lambda t: t[1])[:10]:
        print(f"  gridFB (β,σδ,ε,LPR,anchor)={k}: {v:.3f}")
    json.dump(res, open(os.path.join(ART, "dtw_sim_result.json"), "w"), indent=1)
    print(f"done ({time.time()-t0:.0f}s)")

if __name__ == "__main__":
    main()
