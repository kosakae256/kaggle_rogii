# 現行チェーン(v52 GBM pp)の困難井戸の特定と原因分類・賞金定量
#   軸: バイアス(枝) / ジッタ(ドリフト) / 空間孤立(v56 nd) / 候補割れ・候補オラクル(v54 cand)
#   出力: クラスタ表 + 賞金表 + hard_wells.png(Pareto/散布) + hard_wells.json
import os, sys, json, time
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

P0 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART52 = os.path.join(P0, "v52", "artifacts_v52")
ART54 = os.path.join(P0, "v54", "artifacts_v54")
ART56 = os.path.join(P0, "v56", "artifacts_v56")
PQ = os.path.join(ART52, "data", "train.parquet")

def main():
    t0 = time.time()
    tb = pq.read_table(PQ, columns=["well", "target", "md_since", "pf_ancc", "last_known_tvt"])
    y = tb.column("target").to_numpy(zero_copy_only=False).astype(np.float64)
    g = tb.column("well").to_pandas().astype(str).to_numpy()
    mds = np.maximum(tb.column("md_since").to_numpy(zero_copy_only=False).astype(np.float64), 0.0)
    pfa = tb.column("pf_ancc").to_numpy(zero_copy_only=False).astype(np.float64)
    lk = tb.column("last_known_tvt").to_numpy(zero_copy_only=False).astype(np.float64)
    del tb
    oof = np.load(os.path.join(ART52, "gbm_v41_oof.npy")).astype(np.float64)
    pp = json.loads(open(os.path.join(ART52, "pp_params.json")).read())
    pred = (oof * (1 - pp["w_pf"]) + (pfa - lk) * pp["w_pf"]) * (1 - np.exp(-mds / pp["tau"])) * pp["alpha"]
    e = pred - y
    fin = np.isfinite(e)
    df = pd.DataFrame({"w": g[fin], "e": e[fin], "mds": mds[fin]})
    gr = df.groupby("w")["e"]
    W = pd.DataFrame({"n": gr.size(), "bias": gr.mean(),
                      "mse": gr.apply(lambda s: float(np.mean(s ** 2)))})
    W["rmse"] = np.sqrt(W["mse"]); W["jit"] = np.sqrt(np.maximum(W["mse"] - W["bias"] ** 2, 0))
    W["Lmds"] = df.groupby("w")["mds"].max()
    total = float(np.sqrt((W["mse"] * W["n"]).sum() / W["n"].sum()))
    print(f"現行 v52 GBM pp overall = {total:.4f}  (井戸 {len(W)})", flush=True)
    # 空間孤立(v56)
    d56 = np.load(os.path.join(ART56, "lowo_per_well.npz"), allow_pickle=True)
    nd = dict(zip([str(x) for x in d56["wells"]], d56["nd"]))
    r56 = dict(zip([str(x) for x in d56["wells"]], d56["rmse"]))
    W["nd"] = [nd.get(w, np.nan) for w in W.index]
    W["r56"] = [r56.get(w, np.nan) for w in W.index]
    # 候補(v54, 存在するストリームのみ)
    import glob
    cfs = sorted(glob.glob(os.path.join(ART54, "cand_*.npz")))
    streams = [os.path.basename(f)[5:-4] for f in cfs]
    print(f"候補ストリーム: {streams}", flush=True)
    CANDS = [np.load(f) for f in cfs]
    # 井戸別: stream-oracle(最良streamのww平均) / seed-oracle(最良1本) / 候補割れ幅
    so, sd, spread = {}, {}, {}
    # 真値(行順=eval行CSV順)は parquet 由来と同一のはずだが、候補は独自整列なのでhwから
    TR = os.path.join(P0, "rogii-wellbore-geology-prediction", "train")
    for wi, w in enumerate(W.index):
        try:
            hw = pd.read_csv(os.path.join(TR, f"{w}__horizontal_well.csv"), usecols=["TVT", "TVT_input"])
        except FileNotFoundError:
            continue
        ev = hw["TVT_input"].isna().to_numpy()
        yv = hw["TVT"].to_numpy(float)[ev]
        finy = np.isfinite(yv)
        best_s = np.inf; best_seed = np.inf; sp = []
        for d in CANDS:
            if w not in d.files: continue
            pr = d[w].astype(np.float64); ww = d[f"{w}|w"]
            n = min(pr.shape[1], len(yv))
            wm = (ww[:, None] * pr[:, :n]).sum(0) / max(ww.sum(), 1e-9)
            best_s = min(best_s, float(np.sqrt(np.mean((wm - yv[:n])[finy[:n]] ** 2))))
            er = pr[:, :n] - yv[None, :n]
            per_seed = np.sqrt(np.nanmean(np.where(finy[None, :n], er, np.nan) ** 2, axis=1))
            best_seed = min(best_seed, float(np.nanmin(per_seed)))
            sp.append(wm)
        so[w] = best_s; sd[w] = best_seed
        spread[w] = float(np.std(np.stack(sp), 0).mean()) if len(sp) >= 2 else np.nan
        if (wi + 1) % 150 == 0: print(f"  cand {wi+1}/{len(W)} ({time.time()-t0:.0f}s)", flush=True)
    W["so"] = [so.get(w, np.nan) for w in W.index]        # stream oracle
    W["sdo"] = [sd.get(w, np.nan) for w in W.index]       # per-seed oracle
    W["spread"] = [spread.get(w, np.nan) for w in W.index]
    # ---- クラスタ ----
    hard = W["rmse"] > 10
    A = hard & (W["bias"].abs() >= 6) & (W["bias"] ** 2 >= 0.5 * W["mse"])      # 枝バイアス
    C = hard & ~A & (W["nd"] > 1500)                                            # 空間孤立
    B = hard & ~A & ~C                                                          # ドリフト/その他
    print(f"\n困難井戸(rmse>10): {hard.sum()}/{len(W)}  RMSE²寄与 {float((W.loc[hard,'mse']*W.loc[hard,'n']).sum()/(W['mse']*W['n']).sum()):.0%}")
    for tag, m in (("A 枝バイアス型", A), ("B ドリフト型", B), ("C 空間孤立型", C)):
        if m.sum() == 0: continue
        sub = W[m]
        print(f"  {tag}: n={m.sum()}  rmse med={sub['rmse'].median():.1f}  |bias|med={sub['bias'].abs().median():.1f} "
              f"jit med={sub['jit'].median():.1f}  nd med={sub['nd'].median():.0f}ft  "
              f"cand-oracle med={sub['so'].median():.1f}  seed-oracle med={sub['sdo'].median():.1f}  spread med={sub['spread'].median():.1f}")
    # ---- 賞金表 ----
    def pooled_replace(mask, new_rmse_col):
        mse2 = W["mse"].copy()
        repl = np.minimum(W.loc[mask, new_rmse_col] ** 2, W.loc[mask, "mse"])
        mse2.loc[mask] = repl
        return float(np.sqrt((mse2 * W["n"]).sum() / W["n"].sum()))
    debias = np.sqrt(np.maximum(W["mse"] - W["bias"] ** 2, 0))
    W["_debias"] = debias
    print(f"\n--- 賞金表(overall {total:.3f} からの到達値) ---")
    print(f"  hard井戸のバイアスだけ全消し           → {pooled_replace(hard, '_debias'):.4f}")
    print(f"  A型のバイアスだけ消し                  → {pooled_replace(A, '_debias'):.4f}")
    print(f"  hard井戸を stream-oracle に置換        → {pooled_replace(hard, 'so'):.4f}")
    print(f"  hard井戸を per-seed oracle に置換      → {pooled_replace(hard, 'sdo'):.4f}")
    print(f"  hard井戸を中央値井戸並み(rmse=4)に     → {float(np.sqrt(((W['mse'].where(~hard, 16.0))*W['n']).sum()/W['n'].sum())):.4f}")
    # 上位20の顔ぶれ
    top = W.sort_values("rmse", ascending=False).head(20)
    print("\nworst20:")
    for w, r in top.iterrows():
        print(f"  {w[:10]} rmse={r['rmse']:6.2f} bias={r['bias']:+6.2f} jit={r['jit']:5.2f} nd={r['nd']:6.0f} "
              f"so={r['so']:5.2f} sdo={r['sdo']:5.2f} spread={r['spread']:5.2f} rows={int(r['n'])}")
    W.drop(columns=["_debias"]).to_csv(os.path.join(ART52, "hard_wells_v52.csv"))
    # ---- グラフ ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "Meiryo"
    fig, ax = plt.subplots(1, 3, figsize=(16.5, 4.6))
    o = W.sort_values("rmse", ascending=False)
    contrib = (o["mse"] * o["n"]).cumsum() / (W["mse"] * W["n"]).sum()
    ax[0].plot(np.arange(1, len(o) + 1), contrib)
    ax[0].axhline(0.5, color="gray", ls="--", lw=0.7); ax[0].set_xlim(0, 120)
    ax[0].set_xlabel("worst側からの井戸数"); ax[0].set_ylabel("RMSE²累積寄与"); ax[0].set_title("誤差のPareto集中")
    cols = np.where(A, "crimson", np.where(C, "darkorange", np.where(B, "royalblue", "lightgray")))
    ax[1].scatter(W["bias"].abs(), W["jit"], s=12, c=cols, alpha=0.7)
    ax[1].set_xlabel("|bias| (ft)"); ax[1].set_ylabel("jitter (ft)")
    ax[1].set_title("A=枝バイアス(赤) B=ドリフト(青) C=孤立(橙)")
    ax[2].scatter(W["spread"], W["rmse"], s=12, alpha=0.6)
    ax[2].set_xlabel("候補割れ幅 spread (ft)"); ax[2].set_ylabel("井戸RMSE (ft)")
    ax[2].set_title("候補割れ vs 現行誤差")
    plt.tight_layout(); plt.savefig(os.path.join(ART52, "hard_wells.png"), dpi=110)
    print(f"\nsaved hard_wells.png / hard_wells_v52.csv ({time.time()-t0:.0f}s)")

if __name__ == "__main__":
    main()
