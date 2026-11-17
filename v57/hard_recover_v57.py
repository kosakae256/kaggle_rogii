# v57 — hard井戸専用の二段回収(検出→per-seed票モード分解→学習スコアラ→soft事後平均)
#   docs/v57.md 参照。CPUのみ。GroupKFold(5) で検出器/スコアラともOOF。
import os, sys, json, time, glob
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.model_selection import GroupKFold

P0 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART52 = os.path.join(P0, "v52", "artifacts_v52")
ART54 = os.path.join(P0, "v54", "artifacts_v54")
ART56 = os.path.join(P0, "v56", "artifacts_v56")
ART57 = os.path.join(P0, "v57", "artifacts_v57"); os.makedirs(ART57, exist_ok=True)
PQ = os.path.join(ART52, "data", "train.parquet")
TR = os.path.join(P0, "rogii-wellbore-geology-prediction", "train")
GAP = float(os.environ.get("GAP", "3.0"))          # モード分割ギャップ(ft)
MAXM = 4

def main():
    t0 = time.time()
    tb = pq.read_table(PQ, columns=["id", "well", "target", "md_since", "pf_ancc", "last_known_tvt"])
    ids = tb.column("id").to_pandas().astype(str)
    g = tb.column("well").to_pandas().astype(str).to_numpy()
    y_res = tb.column("target").to_numpy(zero_copy_only=False).astype(np.float64)
    mds = np.maximum(tb.column("md_since").to_numpy(zero_copy_only=False).astype(np.float64), 0.0)
    pfa = tb.column("pf_ancc").to_numpy(zero_copy_only=False).astype(np.float64)
    lk = tb.column("last_known_tvt").to_numpy(zero_copy_only=False).astype(np.float64)
    del tb
    oof = np.load(os.path.join(ART52, "gbm_v41_oof.npy")).astype(np.float64)
    pp = json.loads(open(os.path.join(ART52, "pp_params.json")).read())
    chain_res = (oof * (1 - pp["w_pf"]) + (pfa - lk) * pp["w_pf"]) * (1 - np.exp(-mds / pp["tau"])) * pp["alpha"]
    rn = ids.str.rsplit("_", n=1, expand=True)[1].astype(int).to_numpy()
    # 井戸→(CSV順の行index列, chain_abs, y_abs)
    ORD = {}
    for wn in pd.unique(g):
        ii = np.where(g == wn)[0]
        o = np.argsort(rn[ii], kind="stable")
        ORD[wn] = ii[o]
    # v56 surf(eval行CSV順)
    SURF = np.load(os.path.join(ART56, "lowo_rows.npz"))
    nd56 = dict(zip([str(x) for x in np.load(os.path.join(ART56, "lowo_per_well.npz"), allow_pickle=True)["wells"]],
                    np.load(os.path.join(ART56, "lowo_per_well.npz"), allow_pickle=True)["nd"]))
    cfs = sorted(glob.glob(os.path.join(ART54, "cand_*.npz")))
    CANDS = [np.load(f) for f in cfs]
    snames = [os.path.basename(f)[5:-4] for f in cfs]
    FLAV = [s.rsplit("_", 1)[1] for s in snames]
    print(f"streams: {snames}", flush=True)
    wells = sorted(set(g) & set.intersection(*[set(k for k in d.files if not k.endswith("|w")) for d in CANDS]))
    WF = []; MODES = {}; CHAIN = {}; YW = {}
    for wi, wn in enumerate(wells):
        idx = ORD[wn]
        ca = (chain_res[idx] + lk[idx])                    # chain絶対値(eval行CSV順)
        ya = (y_res[idx] + lk[idx])
        T = len(idx)
        zs = []; ws_ = []; fl = []
        for d, f in zip(CANDS, FLAV):
            pr = d[wn].astype(np.float64); ww = d[f"{wn}|w"].astype(np.float64)
            n = min(pr.shape[1], T)
            zs.append(pr[:, :n]); ws_.append(ww); fl += [f] * pr.shape[0]
        n = min(min(z.shape[1] for z in zs), T)
        Zc = np.concatenate([z[:, :n] for z in zs], 0)     # (C,n)
        Wc = np.concatenate(ws_); Wc = Wc / max(Wc.sum(), 1e-9)
        FLc = np.array(fl)
        off = np.median(Zc - ca[None, :n], 1)              # 各候補の井戸オフセット
        so_ = np.argsort(off); s_off = off[so_]
        cuts = np.where(np.diff(s_off) > GAP)[0]
        mid = []; start = 0
        for c in list(cuts) + [len(s_off) - 1]:
            mid.append(so_[start:c + 1]); start = c + 1
        mid = sorted(mid, key=lambda m: -len(m))[:MAXM]
        surf = SURF[wn][:n, 0].astype(np.float64) if wn in SURF.files else None
        surf_off = float(np.median(surf - ca[:n])) if surf is not None else 0.0
        true_off = float(np.median(ya[:n] - ca[:n]))
        mrows = []
        # ★第0モード = chain そのもの(「乗り換えない」選択肢)
        near = np.abs(off) < 2.0
        mrows.append(dict(
            vote=float(near.mean()), wmass=float(Wc[near].sum()),
            vote_nbr=float(((FLc == "nbr") & near).sum() / max((FLc == "nbr").sum(), 1)),
            vote_self=float(((FLc == "self") & near).sum() / max((FLc == "self").sum(), 1)),
            aoff=0.0, off=0.0,
            surf_agree=float(abs(surf_off) <= 2), chain_in=1.0, is_chain=1.0,
            label=0, traj=ca[:n].copy()))
        for mi, mm in enumerate(mid):
            wgt = Wc[mm]
            mtraj = (wgt[:, None] * Zc[mm]).sum(0) / max(wgt.sum(), 1e-9)
            moff = float(np.median(off[mm]))
            mrows.append(dict(
                vote=len(mm) / len(off), wmass=float(Wc[mm].sum()),
                vote_nbr=float((FLc[mm] == "nbr").sum() / max((FLc == "nbr").sum(), 1)),
                vote_self=float((FLc[mm] == "self").sum() / max((FLc == "self").sum(), 1)),
                aoff=abs(moff), off=moff,
                surf_agree=float(np.sign(moff) == np.sign(surf_off)) if abs(surf_off) > 2 else 0.5,
                chain_in=float(abs(moff) < 2.0), is_chain=0.0,
                label=0, traj=mtraj))
        lab = int(np.argmin([abs(r["off"] - true_off) for r in mrows]))
        for mi, r in enumerate(mrows): r["label"] = int(mi == lab)
        MODES[wn] = mrows; CHAIN[wn] = ca[:n]; YW[wn] = ya[:n]
        e = ca[:n] - ya[:n]; e = e[np.isfinite(e)]
        WF.append(dict(w=wn, rmse=float(np.sqrt(np.mean(e ** 2))),
                       nmode=len(mrows), spread=float(np.std(off)),
                       gap12=float(abs(mrows[0]["off"] - mrows[1]["off"])) if len(mrows) > 1 else 0.0,
                       top_vote=mrows[0]["vote"], chain_dev=float(abs(np.median(off))),
                       nd=float(nd56.get(wn, 2000.0)),
                       surf_dev=abs(surf_off)))
        if (wi + 1) % 150 == 0: print(f"  {wi+1}/{len(wells)} ({time.time()-t0:.0f}s)", flush=True)
    W = pd.DataFrame(WF).set_index("w")
    W["hard"] = (W["rmse"] > 10).astype(int)
    print(f"wells={len(W)} hard={int(W['hard'].sum())}  ({time.time()-t0:.0f}s)", flush=True)
    # ---- GroupKFold OOF: 検出器 + モードスコアラ ----
    import lightgbm as lgb
    DETF = ["nmode", "spread", "gap12", "top_vote", "chain_dev", "nd", "surf_dev"]
    MF = ["vote", "wmass", "vote_nbr", "vote_self", "aoff", "surf_agree", "chain_in", "is_chain"]
    wl = list(W.index)
    folds = list(GroupKFold(5).split(np.zeros(len(wl)), groups=np.array(wl)))
    PH = pd.Series(0.0, index=W.index); PM = {}
    from sklearn.metrics import roc_auc_score
    for k, (tr_i, va_i) in enumerate(folds):
        trw = [wl[i] for i in tr_i]; vaw = [wl[i] for i in va_i]
        det = lgb.train(dict(objective="binary", learning_rate=0.05, num_leaves=15, min_data_in_leaf=20,
                             verbose=-1, seed=7), lgb.Dataset(W.loc[trw, DETF], W.loc[trw, "hard"]),
                        num_boost_round=300)
        PH.loc[vaw] = det.predict(W.loc[vaw, DETF])
        Xm, ym = [], []
        for wn in trw:
            if W.loc[wn, "spread"] <= 3.0 or W.loc[wn, "nmode"] < 2: continue   # ★割れ井戸のみで学習
            for r in MODES[wn]:
                Xm.append([r[c] for c in MF]); ym.append(r["label"])
        sc = lgb.train(dict(objective="binary", learning_rate=0.05, num_leaves=15, min_data_in_leaf=40,
                            verbose=-1, seed=7), lgb.Dataset(np.array(Xm), np.array(ym)),
                       num_boost_round=300)
        for wn in vaw:
            X = np.array([[r[c] for c in MF] for r in MODES[wn]])
            p = np.clip(sc.predict(X), 1e-4, 1 - 1e-4)
            z = np.log(p / (1 - p))                          # ★log-odds softmax(較正)
            e_ = np.exp(z - z.max())
            PM[wn] = e_ / e_.sum()
    auc = roc_auc_score(W["hard"], PH)
    print(f"hard検出 OOF AUC = {auc:.3f}", flush=True)
    # モード選択の正答率(hard井戸, OOF)
    accs = []
    for wn in W.index[W["hard"] == 1]:
        lab = int(np.argmax([r["label"] for r in MODES[wn]]))
        accs.append(int(np.argmax(PM[wn]) == lab))
    print(f"hard井戸のモード選択 OOF 正答率 = {np.mean(accs):.2f} (n={len(accs)})", flush=True)
    # ---- 回収ブレンド + 評価 ----
    def overall(pred_map):
        es = [pred_map[wn] - YW[wn] for wn in W.index]
        ee = np.concatenate(es); ee = ee[np.isfinite(ee)]
        return float(np.sqrt(np.mean(ee ** 2)))
    base = overall(CHAIN)
    print(f"baseline chain = {base:.4f}", flush=True)
    def make_pred(wn, LAM, PHT, MARG):
        ph = PH[wn]
        if ph < PHT: return CHAIN[wn]
        pm = PM[wn]
        nc = [i for i in range(len(pm)) if MODES[wn][i]["is_chain"] == 0]
        if not nc: return CHAIN[wn]
        ib = max(nc, key=lambda i: pm[i])
        if pm[ib] - pm[0] < MARG: return CHAIN[wn]           # ★chainをマージンで上回る時だけ発動
        rec = sum(p * r["traj"] for p, r in zip(pm, MODES[wn]))
        w_ = LAM * ph
        return (1 - w_) * CHAIN[wn] + w_ * rec
    results = {}
    for LAM in (0.5, 0.7, 1.0):
        for PHT in (0.3, 0.5):
            for MARG in (0.1, 0.25):
                pred = {wn: make_pred(wn, LAM, PHT, MARG) for wn in W.index}
                r = overall(pred)
                results[f"lam{LAM}_pht{PHT}_m{MARG}"] = r
                print(f"  λ={LAM} pht={PHT} marg={MARG}: {r:.4f} ({r-base:+.4f})", flush=True)
    best = min(results, key=results.get)
    ps = best.split("_"); LAM = float(ps[0][3:]); PHT = float(ps[1][3:]); MARG = float(ps[2][1:])
    eh, ee_ = [], []
    for wn in W.index:
        pr = make_pred(wn, LAM, PHT, MARG)
        e = pr - YW[wn]; e = e[np.isfinite(e)]
        (eh if W.loc[wn, "hard"] else ee_).append((np.sqrt(np.mean((CHAIN[wn]-YW[wn])[np.isfinite(YW[wn])]**2)), np.sqrt(np.mean(e**2))))
    eh = np.array(eh); ee_ = np.array(ee_)
    print(f"\nbest={best} {results[best]:.4f} (Δ{results[best]-base:+.4f})")
    print(f"  hard {len(eh)}井: chain med {np.median(eh[:,0]):.2f} → 回収後 {np.median(eh[:,1]):.2f} (改善井戸 {(eh[:,1]<eh[:,0]-0.1).sum()}/{len(eh)})")
    print(f"  easy {len(ee_)}井: 悪化>0.1ft は {(ee_[:,1]>ee_[:,0]+0.1).sum()}/{len(ee_)}")
    json.dump(dict(base=base, results=results, auc=float(auc), mode_acc=float(np.mean(accs))),
              open(os.path.join(ART57, "hard_recover_result.json"), "w"), indent=1)
    print(f"done ({time.time()-t0:.0f}s)")

if __name__ == "__main__":
    main()
