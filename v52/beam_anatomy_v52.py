# beam search 解剖 — 現行ビーム(6設定, v26/v52系 _beam_jit)の軌跡の性質と敗因を実測
#   (1) 設定別/median合議の RMSE 分布 vs PF(v54 cand pf_1/tw の ww平均)
#   (2) 敗因分解: 井戸平均バイアス(枝コミット) vs 井戸内ジッタ / mds帯別誤差成長
#   (3) 物理無視の定量: corr(Δbeam, ΔZ) vs corr(Δ真値, ΔZ)(beamはZを見ない)
#   出力: 統計表 + beam_anatomy.png(ECDF/散布/事例軌跡, Meiryo)
import os, sys, time, json
import numpy as np
import pandas as pd

P0 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(P0, "v26"))
TRAIN = os.path.join(P0, "rogii-wellbore-geology-prediction", "train")
ART = os.path.join(P0, "v52", "artifacts_v52")
NWELL = int(os.environ.get("NWELL", "80"))

from create_train_parquet_v26 import beam_search, BEAMS   # 同一実装(v52はv26系を内包)

def main():
    t0 = time.time()
    cand = np.load(os.path.join(P0, "v54", "artifacts_v54", "cand_pf_1_tw.npz"))
    wells_all = sorted(k for k in cand.files if not k.endswith("|w"))
    rng = np.random.default_rng(0)
    wells = [wells_all[i] for i in rng.permutation(len(wells_all))[:NWELL]]
    rows = {}
    EB = {t: [] for t in [b[-1] for b in BEAMS]}; EMED = []; EPF = []
    stats = []
    examples = []
    for wi, w in enumerate(wells):
        hw = pd.read_csv(os.path.join(TRAIN, f"{w}__horizontal_well.csv"))
        tw = pd.read_csv(os.path.join(TRAIN, f"{w}__typewell.csv")).sort_values("TVT")
        tw_tvt = tw["TVT"].to_numpy(float); tw_gr = tw["GR"].to_numpy(float)
        fin = np.isfinite(tw_tvt) & np.isfinite(tw_gr); tw_tvt, tw_gr = tw_tvt[fin], tw_gr[fin]
        kn = hw[hw["TVT_input"].notna()]; ev = hw[hw["TVT_input"].isna()]
        if len(kn) < 10 or len(ev) < 100: continue
        last_tvt = float(kn["TVT_input"].iloc[-1])
        gr_full = hw["GR"].astype(float).interpolate(limit_direction="both").fillna(float(np.nanmean(tw_gr)))
        hgr = gr_full.iloc[ev.index[0]:].to_numpy(np.float32)
        y = hw["TVT"].to_numpy(float)[ev.index[0]:]
        z = hw["Z"].to_numpy(float)[ev.index[0]:]
        paths = {}
        for (bs, mc, es, r, tag) in BEAMS:
            paths[tag] = beam_search(hgr, tw_tvt, tw_gr, last_tvt, bs, mc, es, r).astype(np.float64)
        med = np.median(np.stack(list(paths.values())), 0)
        pf = (cand[f"{w}|w"][:, None] * cand[w].astype(np.float64)).sum(0)
        n = min(len(y), len(med), len(pf))
        fin_y = np.isfinite(y[:n])
        for tag in paths: EB[tag].append((paths[tag][:n] - y[:n])[fin_y])
        EMED.append((med[:n] - y[:n])[fin_y]); EPF.append((pf[:n] - y[:n])[fin_y])
        e = (med[:n] - y[:n])[fin_y]
        dz = np.diff(z[:n]); dt = np.diff(y[:n]); db = np.diff(med[:n])
        okd = np.isfinite(dz) & np.isfinite(dt) & np.isfinite(db)
        stats.append(dict(w=w, rmse=float(np.sqrt(np.mean(e**2))), bias=float(np.mean(e)),
                          jit=float(np.std(e - np.mean(e))),
                          c_beam=float(np.corrcoef(db[okd], -dz[okd])[0, 1]) if okd.sum() > 50 else np.nan,
                          c_true=float(np.corrcoef(dt[okd], -dz[okd])[0, 1]) if okd.sum() > 50 else np.nan,
                          step_med_ft=float(np.median(np.abs(db))), twstep=float(np.median(np.diff(tw_tvt)))))
        if len(examples) < 3 and abs(np.mean(e)) > 8:
            examples.append((w, y[:n].copy(), med[:n].copy(), pf[:n].copy(), hw["MD"].to_numpy(float)[ev.index[0]:ev.index[0]+n]))
        if (wi + 1) % 20 == 0: print(f"  {wi+1}/{len(wells)} ({time.time()-t0:.0f}s)", flush=True)
    def pooled(es):
        ee = np.concatenate(es); return float(np.sqrt(np.mean(ee ** 2)))
    print("\n--- 設定別 pooled RMSE(80井) ---")
    for tag in EB: print(f"  beam[{tag:6s}]: {pooled(EB[tag]):7.3f}")
    print(f"  beam median合議 : {pooled(EMED):7.3f}")
    print(f"  PF(pf_1/tw wwm) : {pooled(EPF):7.3f}")
    S = pd.DataFrame(stats)
    print("\n--- 敗因分解(beam median) ---")
    print(f"  井戸|バイアス|>8ft(枝コミット疑い): {(S['bias'].abs()>8).mean():.0%}")
    print(f"  bias^2/(bias^2+jitter^2) 中央値: {np.median(S['bias']**2/(S['bias']**2+S['jit']**2+1e-9)):.2f}")
    print(f"  corr(d_beam, -dZ) med={S['c_beam'].median():.3f}  /  corr(d_true, -dZ) med={S['c_true'].median():.3f}")
    print(f"  1行あたり移動中央値: beam {S['step_med_ft'].median():.3f}ft (twグリッド {S['twstep'].median():.3f}ft/行)")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "Meiryo"
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    for es, lab in ((EMED, "beam med合議"), (EPF, "PF pf_1/tw")):
        rs = np.sort([float(np.sqrt(np.mean(e**2))) for e in es])
        ax[0].plot(rs, np.arange(1, len(rs) + 1) / len(rs), label=lab)
    ax[0].set_xlabel("井戸RMSE (ft)"); ax[0].set_ylabel("ECDF"); ax[0].set_xlim(0, 30); ax[0].legend()
    ax[0].set_title("beam vs PF 井戸RMSE ECDF")
    ax[1].scatter(S["bias"].abs(), S["jit"], s=14, alpha=0.6)
    ax[1].set_xlabel("|井戸バイアス| (ft)"); ax[1].set_ylabel("井戸内ジッタσ (ft)")
    ax[1].set_title("敗因: バイアス(枝) vs ジッタ")
    for (w, yv, mv, pv, mdv) in examples[:1]:
        ax[2].plot(mdv, yv, "k", lw=1.2, label="真値")
        ax[2].plot(mdv, mv, lw=1.0, label="beam med")
        ax[2].plot(mdv, pv, lw=1.0, label="PF")
        ax[2].set_title(f"事例: {w[:10]} (beamバイアス例)"); ax[2].set_xlabel("MD"); ax[2].legend()
    plt.tight_layout()
    fp = os.path.join(ART, "beam_anatomy.png")
    plt.savefig(fp, dpi=110)
    print(f"saved {fp} ({time.time()-t0:.0f}s)")

if __name__ == "__main__":
    main()
