# 4member 脚でのスタック重み精密化: 細grid座標降下×複数初期値 + A/B半分割の汎化監査
import os, json
import numpy as np
import pyarrow.parquet as pq
import sys
sys.stdout = open(1, "w", encoding="utf-8", closefd=False)

ART = "v62/artifacts_v62"; ART52 = "v52/artifacts_v52"
LEGS = ["GBM", "TCN", "L3img", "L4", "L5", "L6", "L7"]

tb = pq.read_table(f"{ART52}/data/train.parquet", columns=["id", "well", "target", "md_since", "pf_ancc", "last_known_tvt"])
ids = tb.column("id").to_pandas().astype(str); gw = tb.column("well").to_pandas().astype(str).to_numpy()
y = tb.column("target").to_numpy(zero_copy_only=False).astype(np.float64)
mds = np.maximum(tb.column("md_since").to_numpy(zero_copy_only=False).astype(np.float64), 0)
pfa = tb.column("pf_ancc").to_numpy(zero_copy_only=False).astype(np.float64)
lk = tb.column("last_known_tvt").to_numpy(zero_copy_only=False).astype(np.float64)
gbm = np.load(f"{ART52}/gbm_v41_oof.npy").astype(np.float64)
nn = np.mean([np.load(f"{ART52}/nn_oof_combo.npz")[k] for k in ("mdn2_s0", "mdn2_s1", "mdn2_s2")], 0).astype(np.float64)
pp = json.loads(open(f"{ART52}/pp_params.json").read())
def ppe(r): return (r * (1 - pp["w_pf"]) + (pfa - lk) * pp["w_pf"]) * (1 - np.exp(-mds / pp["tau"])) * pp["alpha"] - y
eg = ppe(gbm); et = ppe(nn)
E = {"L3img": {w: v.astype(np.float64) for w, v in np.load(f"{ART}/v62_oof_err.npz").items()},
     "L4": {w: v.astype(np.float64) for w, v in np.load(f"{ART}/l4_oof3_err.npz").items()},
     "L5": {w: v.astype(np.float64) for w, v in np.load(f"{ART}/l5_oof3_err.npz").items()},
     "L6": {w: v.astype(np.float64) for w, v in np.load(f"{ART}/l6_oof3_err.npz").items()},
     "L7": {w: v.astype(np.float64) for w, v in np.load(f"{ART}/l7_oof3_err.npz").items()}}
WELLS = sorted(E["L5"])
rn = ids.str.rsplit("_", n=1, expand=True)[1].astype(int).to_numpy()
X = {n: {} for n in LEGS}
for w in WELLS:
    ii = np.where(gw == w)[0]; o = np.argsort(rn[ii], kind="stable"); iio = ii[o]
    n0 = min([len(E[n][w]) for n in E] + [len(iio)])
    ms = np.ones(n0, bool)
    for n in E: ms &= np.isfinite(E[n][w][:n0])
    ms &= np.isfinite(eg[iio[:n0]])
    X["GBM"][w] = eg[iio[:n0]][ms]; X["TCN"][w] = et[iio[:n0]][ms]
    for n in E: X[n][w] = E[n][w][:n0][ms]

def pooled(errs): ee = np.concatenate(errs); return float(np.sqrt(np.mean(ee ** 2)))
def sc(wd, ws=WELLS):
    s = sum(wd.values())
    return pooled([sum(wd[n] * X[n][w] for n in LEGS) / s for w in ws])

def cdesc(init, ws=WELLS, grid=0.025, sweeps=6):
    wt = dict(init); cur = sc(wt, ws)
    for _ in range(sweeps):
        moved = False
        for n in LEGS:
            bv, bw = cur, wt[n]
            for v in np.arange(0, 1.0001, grid):
                w2 = dict(wt); w2[n] = float(v)
                if sum(w2.values()) < 1e-9: continue
                r = sc(w2, ws)
                if r < bv - 1e-7: bv, bw = r, float(v)
            if bw != wt[n]: moved = True
            wt[n] = bw; cur = bv
        if not moved: break
    return wt, cur

Z = {n: 0.0 for n in LEGS}
INITS = [
    dict(Z, TCN=0.55, L5=1.0, L3img=0.1, L6=0.05, L7=0.05),   # 旧採用重み
    dict(Z, TCN=0.5, L5=1.0),                                   # 2脚
    dict(Z, **{n: 1.0 / 7 for n in LEGS}),                      # 等重み
    dict(Z, L5=1.0),                                            # L5単独
    dict(Z, TCN=0.5, L5=0.95, L3img=0.05),                      # 今回の粗fit解
    dict(Z, TCN=0.4, L5=0.8, L7=0.3, L3img=0.1),                # L7混ぜ
]
print("=== 4member脚での重み探索(全井戸fit, 参考=in-sample) ===")
print(f"  旧重みそのまま適用: {sc(INITS[0]):.4f} / 2脚: {sc(INITS[1]):.4f}")
best = (None, 9e9)
for i, ini in enumerate(INITS):
    wt, v = cdesc(ini)
    tag = json.dumps({k: round(x, 3) for k, x in wt.items() if x > 0})
    print(f"  init{i}: {v:.4f}  {tag}")
    if v < best[1]: best = (wt, v)
wt_all, v_all = best
print(f"→ 最良: {v_all:.4f}")

print("\n=== A/B半分割監査(半分で重みfit→他半分で評価) ===")
A = WELLS[0::2]; B = WELLS[1::2]
for name, ini in [("旧重み固定", None), ("cdesc", INITS[0])]:
    if ini is None:
        wA = wB = INITS[0]
    else:
        wA, _ = cdesc(ini, A); wB, _ = cdesc(ini, B)
    va = sc(wB, A); vb = sc(wA, B)   # 反対側で評価
    print(f"  {name}: A評価(Bfit) {va:.4f} / B評価(Afit) {vb:.4f} / 合成 {np.sqrt((va**2*sum(len(X['L5'][w]) for w in A)+vb**2*sum(len(X['L5'][w]) for w in B))/sum(len(X['L5'][w]) for w in WELLS)):.4f}")

for f in range(5):
    print(f"  fold{f}: {sc(wt_all, WELLS[f::5]):.4f}")
orc = []
for w in WELLS:
    errs = np.stack([X[n][w] for n in LEGS])
    orc.append(errs[int(np.argmin(np.sqrt(np.mean(errs ** 2, 1))))])
json.dump(dict(weights=wt_all, stack_overall=v_all, pp=pp, seeds=4, seed_tags=["_s0", "_s1", "_s2", ""],
               oracle_well=pooled(orc)),
          open(f"{ART}/stack7_config.json", "w"), indent=1)
print(f"\nsaved stack7_config.json  overall={v_all:.4f} oracle={pooled(orc):.4f}")
print("weights =", json.dumps({k: round(x, 3) for k, x in wt_all.items()}))
