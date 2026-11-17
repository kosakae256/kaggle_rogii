# v61 P2判定材料 — 画像脚の全OOF(5ckpt)を生成し、v52(GBM/TCN/blend, pp後)との
#   相関の fold 安定性・ブレンド寄与(pooled/fold別/median井戸/hard-easy)を測定
import os, sys, json, pickle
import numpy as np
import pandas as pd
import torch

P0 = "."
os.environ.setdefault("EVPKL", "v61/artifacts_v61/imagesR_v61.pkl")
sys.path.insert(0, "v61")
from eval_corr_v61 import UNet2D, EVD, WELLS, DEV, NB, STEP

def main():
    offs_t = torch.arange(-NB, NB + 1, device=DEV).float() * STEP
    EIMG = {}; PIMG = {}
    for f in range(5):
        net = UNet2D().to(DEV)
        net.load_state_dict(torch.load(f"v60/artifacts_v60/v61_img_unet_f{f}.pt", map_location=DEV))
        net.eval()
        with torch.no_grad():
            for w in WELLS[f::5]:
                e = EVD[w]; T = e["EV"].shape[1]
                x = torch.from_numpy(e["EV"].astype(np.float32))[None].to(DEV)
                lg, rf, _ = net(x)
                P = torch.softmax(lg[0, :T], -1)
                dh = ((P @ offs_t) + rf[0, :T]).cpu().numpy()
                z = e["st"] + np.asarray(dh, np.float64)[e["bid"]]
                PIMG[w] = z; EIMG[w] = z - e["y_ev"]
        print(f"fold{f} decoded", flush=True)
    np.savez_compressed("v61/artifacts_v61/img_oof_z.npz", **{w: PIMG[w].astype(np.float32) for w in PIMG})
    import pyarrow.parquet as pq
    ART52 = "v52/artifacts_v52"
    tb = pq.read_table(f"{ART52}/data/train.parquet",
                       columns=["id", "well", "target", "md_since", "pf_ancc", "last_known_tvt"])
    ids = tb.column("id").to_pandas().astype(str)
    g = tb.column("well").to_pandas().astype(str).to_numpy()
    y = tb.column("target").to_numpy(zero_copy_only=False).astype(np.float64)
    mds = np.maximum(tb.column("md_since").to_numpy(zero_copy_only=False).astype(np.float64), 0)
    pfa = tb.column("pf_ancc").to_numpy(zero_copy_only=False).astype(np.float64)
    lk = tb.column("last_known_tvt").to_numpy(zero_copy_only=False).astype(np.float64)
    gbm = np.load(f"{ART52}/gbm_v41_oof.npy").astype(np.float64)
    nn = np.mean([np.load(f"{ART52}/nn_oof_combo.npz")[k] for k in ("mdn2_s0", "mdn2_s1", "mdn2_s2")], 0).astype(np.float64)
    pp = json.loads(open(f"{ART52}/pp_params.json").read())
    W_NN = 0.60
    ebl = (( (1 - W_NN) * gbm + W_NN * nn) * (1 - pp["w_pf"]) + (pfa - lk) * pp["w_pf"]) \
          * (1 - np.exp(-mds / pp["tau"])) * pp["alpha"] - y
    rn = ids.str.rsplit("_", n=1, expand=True)[1].astype(int).to_numpy()
    per = {}
    for w in PIMG:
        ii = np.where(g == w)[0]; o = np.argsort(rn[ii], kind="stable"); iio = ii[o]
        n = min(len(EIMG[w]), len(iio))
        a = EIMG[w][:n]; c = ebl[iio[:n]]
        m = np.isfinite(a) & np.isfinite(c)
        per[w] = (a[m], c[m])
    wl = sorted(per)
    def pooled(es): ee = np.concatenate(es); return float(np.sqrt(np.mean(ee ** 2)))
    print(f"\n単体: img overall = {pooled([per[w][0] for w in wl]):.4f} / v52blend = {pooled([per[w][1] for w in wl]):.4f}")
    # fold別 相関とブレンド
    print("fold別: corr / blend単体 / +img(w0.15) / Δ")
    deltas = []
    for f in range(5):
        ws = [w for w in WELLS[f::5] if w in per]
        A = np.concatenate([per[w][0] for w in ws]); C = np.concatenate([per[w][1] for w in ws])
        rb = np.sqrt(np.mean(C ** 2)); rm = np.sqrt(np.mean((0.85 * C + 0.15 * A) ** 2))
        print(f"  f{f}: corr={np.corrcoef(A, C)[0,1]:.3f}  {rb:.4f} → {rm:.4f}  (Δ{rm-rb:+.4f})")
        deltas.append(rm - rb)
    # overall best w
    A = np.concatenate([per[w][0] for w in wl]); C = np.concatenate([per[w][1] for w in wl])
    best = (0, np.sqrt(np.mean(C ** 2)))
    for wv in np.arange(0.0, 0.41, 0.05):
        r = float(np.sqrt(np.mean(((1 - wv) * C + wv * A) ** 2)))
        if r < best[1]: best = (wv, r)
    print(f"\noverall: blend {np.sqrt(np.mean(C**2)):.4f} → +img best w={best[0]:.2f} {best[1]:.4f} (Δ{best[1]-np.sqrt(np.mean(C**2)):+.4f})")
    print(f"corr(overall) = {np.corrcoef(A, C)[0,1]:.3f}")
    # 井戸別: median改善 / hard-easy
    rows = []
    for w in wl:
        a, c = per[w]
        rb = float(np.sqrt(np.mean(c ** 2))); rm = float(np.sqrt(np.mean((0.85 * c + 0.15 * a) ** 2)))
        rows.append((w, rb, rm))
    R = pd.DataFrame(rows, columns=["w", "bl", "mix"]).set_index("w")
    print(f"井戸med: blend {R['bl'].median():.3f} → mix {R['mix'].median():.3f}")
    print(f"改善井戸: {(R['mix'] < R['bl'] - 0.05).sum()} / 悪化: {(R['mix'] > R['bl'] + 0.05).sum()} / 中立: {((R['mix']-R['bl']).abs()<=0.05).sum()}")
    hard = R["bl"] > 10
    print(f"hard(bl>10, n={int(hard.sum())}): med {R.loc[hard,'bl'].median():.2f} → {R.loc[hard,'mix'].median():.2f}")
    json.dump(dict(img_overall=pooled([per[w][0] for w in wl]), best_w=best[0], mix_overall=best[1],
                   fold_delta=[float(d) for d in deltas]),
              open("v61/artifacts_v61/final_eval.json", "w"), indent=1)

if __name__ == "__main__":
    main()
