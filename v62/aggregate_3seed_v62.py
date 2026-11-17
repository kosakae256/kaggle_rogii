# L4-L7 の 3seed 平均 OOF を生成 → 脚別 overall → 固定重みスタック再フィット → config v2 保存
import os, json, pickle
import numpy as np
import torch
import sys
sys.stdout = open(1, "w", encoding="utf-8", closefd=False)

ART = "v62/artifacts_v62"
SF = pickle.loads(open(f"{ART}/seqfeat_v62.pkl", "rb").read())
IM = pickle.loads(open(f"{ART}/imagesF_v62.pkl", "rb").read())
WELLS = sorted(set(SF) & set(IM)); K = SF[WELLS[0]]["D"].shape[1]
DEV = "cuda" if torch.cuda.is_available() else "cpu"
src = open("v62/train_l456_v62.py", encoding="utf-8").read()
head = src.split("def make_batch")[0]
head = head.replace('SF = pickle.loads', '_=None #').replace('IM = pickle.loads', '_=None #')
head = head.replace('WELLS = sorted(set(SF) & set(IM))', 'WELLS=[]').replace('K = SF[WELLS[0]]["D"].shape[1]', f'K={K}')
head = head.replace('print(f"wells=', '# print(f"wells=')
g = dict(globals()); exec(head, g)

TAGS = ["_s0", "_s1", "_s2"] + ([""] if os.path.exists(f"{ART}/nn62/v62_l5_f0.pt") else [])

def oof3(arch):
    E = {}
    for f in range(5):
        nets = []
        for tg in TAGS:
            net = {"l4": g["L4"], "l5": g["L5"], "l6": g["L6"], "l7": g["L7"]}[arch]().to(DEV)
            net.load_state_dict(torch.load(f"{ART}/nn62/v62_{arch}{tg}_f{f}.pt", map_location=DEV))
            net.eval(); nets.append(net)
        with torch.no_grad():
            for w in WELLS[f::5]:
                fe = SF[w]; e = IM[w]; T = fe["D"].shape[0]
                D = torch.from_numpy(fe["D"])[None].to(DEV); S = torch.from_numpy(fe["S"])[None].to(DEV)
                C = torch.from_numpy(fe["C"])[None].to(DEV); m = torch.ones(1, T).to(DEV)
                dhs = []
                for net in nets:
                    o_ = net(D, S, C, m)
                    if arch == "l7": o_ = o_[0]
                    dhs.append(o_[0, :T].cpu().numpy())
                dh = np.mean(dhs, 0)
                z = e["st"] + np.asarray(dh, np.float64)[e["bid"]]
                E[w] = z - e["y_ev"]
    np.savez_compressed(f"{ART}/{arch}_oof3_err.npz", **{w: v.astype(np.float32) for w, v in E.items()})
    return E

E3s = {a: oof3(a) for a in ("l4", "l5", "l6", "l7")}
print(f"{len(TAGS)}メンバー平均 OOF 生成完了 (tags={TAGS})")
import pyarrow.parquet as pq
ART52 = "v52/artifacts_v52"
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
d3 = np.load(f"{ART}/v62_oof_err.npz")
E = {"L3img": {w: d3[w].astype(np.float64) for w in d3.files},
     "L4": E3s["l4"], "L5": E3s["l5"], "L6": E3s["l6"], "L7": E3s["l7"]}
rn = ids.str.rsplit("_", n=1, expand=True)[1].astype(int).to_numpy()
LEGS = ["GBM", "TCN", "L3img", "L4", "L5", "L6", "L7"]
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
print("\n脚別 overall(3seed平均版):")
for n in LEGS:
    print(f"  {n:6s}: {pooled([X[n][w] for w in WELLS]):.4f}")
wt = {n: 0.0 for n in LEGS}; wt["TCN"] = 0.5; wt["L5"] = 1.0
def sc(wd, ws=WELLS):
    s = sum(wd.values())
    return pooled([sum(wd[n] * X[n][w] for n in LEGS) / s for w in ws])
cur = sc(wt)
for _ in range(4):
    for n in LEGS:
        bv, bw = cur, wt[n]
        for v in np.arange(0, 1.01, 0.05):
            w2 = dict(wt); w2[n] = float(v)
            if sum(w2.values()) < 1e-9: continue
            r = sc(w2)
            if r < bv - 1e-6: bv, bw = r, float(v)
        wt[n] = bw; cur = bv
print(f"\nスタック(3seed脚): {cur:.4f}  w={json.dumps({k: round(v,2) for k,v in wt.items()})}")
for f in range(5):
    print(f"  fold{f}: {sc(wt, WELLS[f::5]):.4f}")
orc = []
for w in WELLS:
    errs = np.stack([X[n][w] for n in LEGS])
    orc.append(errs[int(np.argmin(np.sqrt(np.mean(errs ** 2, 1))))])
print(f"oracle(井戸別) = {pooled(orc):.4f}")
json.dump(dict(weights=wt, stack_overall=cur, pp=pp, seeds=len(TAGS), seed_tags=TAGS,
               oracle_well=pooled(orc)),
          open(f"{ART}/stack7_config.json", "w"), indent=1)
print(f"saved stack7_config.json (v2, {len(TAGS)}member)")
