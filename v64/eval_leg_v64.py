# -*- coding: utf-8 -*-
# v64 脚評価: 3seed×5fold の 15ckpt を fold毎3seed平均で OOF 推論 → pooled overall / fold別
#   + v63 blend OOF との井戸整合ブレンド寄与測定 + v64_leg_oof.npz 保存
# 実行: CUDA可。VFRAC小さめ(0.5)。
import os, sys, json, time, pickle
import numpy as np

sys.stdout = open(1, "w", encoding="utf-8", closefd=False)
P0 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART = os.path.join(P0, "v64", "artifacts_v64")
import torch, torch.nn as nn, torch.nn.functional as F
DEV = "cuda" if torch.cuda.is_available() else "cpu"
if DEV == "cuda":
    torch.cuda.set_per_process_memory_fraction(float(os.environ.get("VFRAC", "0.5")))
NB = 90; ND = 181; STEP = 0.5; CH = 128; CIN = 9

class UNet2D(nn.Module):
    def __init__(self, cin=CIN, ch=CH):
        super().__init__()
        def blk(ci, co):
            return nn.Sequential(nn.Conv2d(ci, co, 3, padding=1), nn.GELU(),
                                 nn.Conv2d(co, co, 3, padding=1), nn.GELU(), nn.GroupNorm(8, co))
        c1, c2, c3, c4 = ch // 2, ch, ch, ch
        self.e1 = blk(cin, c1); self.e2 = blk(c1, c2); self.e3 = blk(c2, c3); self.bott = blk(c3, c4)
        self.pool = nn.AvgPool2d(2)
        self.u3 = nn.ConvTranspose2d(c4, c3, 2, stride=2); self.d3 = blk(c3 + c3, c3)
        self.u2 = nn.ConvTranspose2d(c3, c2, 2, stride=2); self.d2 = blk(c2 + c2, c2)
        self.u1 = nn.ConvTranspose2d(c2, c1, 2, stride=2); self.d1 = blk(c1 + c1, c1)
        self.gru = nn.GRU(c1, ch, batch_first=True, bidirectional=True)
        self.mix = nn.Linear(2 * ch, c1)
        self.out = nn.Conv2d(c1, 1, 1)
        self.ref = nn.Linear(2 * ch, 1)
    def forward(self, x):
        B, C, T, D = x.shape
        pt = (-T) % 8; pdd = (-D) % 8
        if pt or pdd: x = F.pad(x, (0, pdd, 0, pt))
        h1 = self.e1(x); h2 = self.e2(self.pool(h1)); h3 = self.e3(self.pool(h2))
        hb = self.bott(self.pool(h3))
        d3 = self.d3(torch.cat([self.u3(hb), h3], 1))
        d2 = self.d2(torch.cat([self.u2(d3), h2], 1))
        d1 = self.d1(torch.cat([self.u1(d2), h1], 1))
        d1 = d1[:, :, :T, :D]
        g = d1.mean(3).transpose(1, 2)
        g2, _ = self.gru(g)
        h = d1 + self.mix(g2).transpose(1, 2)[:, :, :, None]
        return self.out(h)[:, 0], 2.0 * torch.tanh(self.ref(g2)[:, :, 0])

t0 = time.time()
EVD = pickle.loads(open(os.path.join(ART, "imagesP_v64.pkl"), "rb").read())
WELLS = sorted(EVD.keys())
offs_t = torch.arange(-NB, NB + 1, device=DEV).float() * STEP
PRED = {}; ERR = {}
fold_rows = []
for f in range(5):
    nets = []
    for tag in ("", "_s1", "_s2"):
        fp = os.path.join(ART, "nn" if False else "", f"v64_unet_f{f}{tag}.pt")
        fp = os.path.join(ART, f"v64_unet_f{f}{tag}.pt")
        net = UNet2D().to(DEV)
        net.load_state_dict(torch.load(fp, map_location=DEV), strict=False)
        net.eval(); nets.append(net)
    es = []
    with torch.no_grad():
        for w in WELLS[f::5]:
            e = EVD[w]; T = e["EV"].shape[1]
            x = torch.from_numpy(e["EV"].astype(np.float32))[None].to(DEV)
            dhs = []
            for net in nets:
                lg, rf = net(x)
                P = torch.softmax(lg[0, :T], -1)
                dhs.append(((P @ offs_t) + rf[0, :T]).cpu().numpy())
            dh = np.mean(dhs, 0)
            z = e["st"].astype(np.float64) + np.asarray(dh, np.float64)[e["bid"]]
            PRED[w] = z.astype(np.float32)
            er = z - e["y_ev"]
            ERR[w] = er.astype(np.float32)
            es.append(er[np.isfinite(er)])
    ee = np.concatenate(es)
    fold_rows.append((f, float(np.sqrt(np.mean(ee ** 2))), len(ee)))
    print(f"fold{f}: 3seed平均 RMSE={fold_rows[-1][1]:.4f} rows={len(ee)} ({time.time()-t0:.0f}s)", flush=True)
allE = np.concatenate([ERR[w][np.isfinite(ERR[w])] for w in WELLS])
pooled = float(np.sqrt(np.mean(allE ** 2)))
print(f"\n=== v64 UNet脚(3seed×5fold平均) pooled overall = {pooled:.4f} ===")
np.savez_compressed(os.path.join(ART, "v64_leg_oof.npz"), **ERR)
np.savez_compressed(os.path.join(ART, "v64_leg_pred.npz"), **PRED)

# ---- v63 blend OOF とのブレンド寄与 ----
import pyarrow.parquet as pq
A63 = os.path.join(P0, "v63", "artifacts_v63")
tb = pq.read_table(os.path.join(A63, "data", "train.parquet"),
                   columns=["id", "well", "target", "md_since", "pf_ancc", "last_known_tvt"])
ids = tb.column("id").to_pandas().astype(str)
gw = tb.column("well").to_pandas().astype(str).to_numpy()
y = tb.column("target").to_numpy(zero_copy_only=False).astype(np.float64)
mds = np.maximum(tb.column("md_since").to_numpy(zero_copy_only=False).astype(np.float64), 0)
pfa = tb.column("pf_ancc").to_numpy(zero_copy_only=False).astype(np.float64)
lk = tb.column("last_known_tvt").to_numpy(zero_copy_only=False).astype(np.float64)
gbm = np.load(os.path.join(A63, "gbm_v41_oof.npy")).astype(np.float64)
nnz = np.load(os.path.join(A63, "nn_oof_combo.npz"))
nnm = np.mean([nnz[k] for k in nnz.files], 0).astype(np.float64)
res = json.loads(open(os.path.join(A63, "results_nn.json"), encoding="utf-8").read())["combo"]
w_nn = float(res["blend_w"])
pp = json.loads(open(os.path.join(A63, "pp_params.json"), encoding="utf-8").read())
blend = w_nn * nnm + (1 - w_nn) * gbm
d63 = (blend * (1 - pp["w_pf"]) + (pfa - lk) * pp["w_pf"]) * (1 - np.exp(-mds / pp["tau"])) * float(pp.get("alpha", 1.0))
z63 = lk + d63; ytrue = lk + y     # 絶対TVT
rn = ids.str.rsplit("_", n=1, expand=True)[1].astype(int).to_numpy()
E63 = []; E64 = []; EB = {}
ws = [w for w in WELLS if w in set(gw)]
for w in ws:
    ii = np.where(gw == w)[0]; o = np.argsort(rn[ii], kind="stable"); iio = ii[o]
    n0 = min(len(iio), len(PRED[w]))
    m = np.isfinite(z63[iio[:n0]]) & np.isfinite(PRED[w][:n0]) & np.isfinite(ytrue[iio[:n0]])
    E63.append((z63[iio[:n0]] - ytrue[iio[:n0]])[m])
    E64.append((PRED[w][:n0].astype(np.float64) - ytrue[iio[:n0]])[m])
A3 = np.concatenate(E63); A4 = np.concatenate(E64)
print(f"整合: {len(ws)}井戸 {len(A3):,}行 | v63blend {np.sqrt(np.mean(A3**2)):.4f} / v64leg {np.sqrt(np.mean(A4**2)):.4f} / corr {np.corrcoef(A3,A4)[0,1]:.3f}")
for wgt in (0.0, 0.05, 0.1, 0.15, 0.2, 0.3):
    e = (1 - wgt) * A3 + wgt * A4
    print(f"  blend w64={wgt:.2f}: {np.sqrt(np.mean(e**2)):.4f}")
