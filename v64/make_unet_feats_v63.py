# -*- coding: utf-8 -*-
# v64 UNet の OOF 出力から v63 チェーン用の特徴量8列を生成(parquet行整列)
#   unet_delta(z−lk) / unet_dh(z−st) / unet_std / unet_ent / unet_pmax /
#   unet_top2gap(第1-第2ピーク距離ft) / unet_p2(第2/第1ピーク比) / unet_vs_pfancc(z−pf_ancc)
# 出力: v63/artifacts_v63/unet_feats_u.npy (N×8, float32, parquet行順) + unet_feats_cols.json
# 実行: CUDA可(~2分)。OOF規律: 各井戸は「その井戸を見ていないfold」の3seed平均事後を使用。
import os, sys, json, time, pickle
import numpy as np

sys.stdout = open(1, "w", encoding="utf-8", closefd=False)
P0 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART = os.path.join(P0, "v64", "artifacts_v64")
A63 = os.path.join(P0, "v63", "artifacts_v63")
import torch, torch.nn as nn, torch.nn.functional as F
DEV = "cuda" if torch.cuda.is_available() else "cpu"
if DEV == "cuda":
    torch.cuda.set_per_process_memory_fraction(float(os.environ.get("VFRAC", "0.5")))
NB = 90; ND = 181; STEP = 0.5; CH = 128; CIN = 9
COLS = ["unet_delta", "unet_dh", "unet_std", "unet_ent", "unet_pmax", "unet_top2gap", "unet_p2", "unet_vs_pfancc"]

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
        gseq = d1.mean(3).transpose(1, 2)
        g2, _ = self.gru(gseq)
        h = d1 + self.mix(g2).transpose(1, 2)[:, :, :, None]
        return self.out(h)[:, 0], 2.0 * torch.tanh(self.ref(g2)[:, :, 0])

t0 = time.time()
EVD = pickle.loads(open(os.path.join(ART, "imagesP_v64.pkl"), "rb").read())
WELLS = sorted(EVD.keys())
offs = (np.arange(ND) - NB) * STEP
offs_t = torch.from_numpy(offs).float().to(DEV)

FE = {}   # well -> (n_rows, 8)
for f in range(5):
    nets = []
    for tag in ("", "_s1", "_s2"):
        net = UNet2D().to(DEV)
        net.load_state_dict(torch.load(os.path.join(ART, f"v64_unet_f{f}{tag}.pt"), map_location=DEV), strict=False)
        net.eval(); nets.append(net)
    with torch.no_grad():
        for w in WELLS[f::5]:
            e = EVD[w]; T = e["EV"].shape[1]
            x = torch.from_numpy(e["EV"].astype(np.float32))[None].to(DEV)
            Ps = []; rfs = []
            for net in nets:
                lg, rf = net(x)
                Ps.append(torch.softmax(lg[0, :T], -1)); rfs.append(rf[0, :T])
            P = torch.stack(Ps).mean(0)                          # (T,181) 3seed平均事後
            rf = torch.stack(rfs).mean(0)
            mu = (P @ offs_t)
            var = (P @ (offs_t ** 2)) - mu ** 2
            std = torch.sqrt(torch.clamp(var, min=0.0))
            ent = -(P * torch.log(P.clamp_min(1e-12))).sum(-1)
            pmax, imax = P.max(-1)
            # 第2ピーク: 第1ピーク±12bin(6ft)をマスクして argmax
            Pm = P.clone()
            ar = torch.arange(ND, device=DEV)[None, :]
            Pm[ (ar - imax[:, None]).abs() <= 12 ] = 0.0
            p2v, i2 = Pm.max(-1)
            top2gap = (i2.float() - imax.float()).abs() * STEP
            p2r = p2v / pmax.clamp_min(1e-9)
            dh = (mu + rf).cpu().numpy().astype(np.float64)
            z = e["st"].astype(np.float64) + dh[e["bid"]]
            n = len(z)
            bidx = e["bid"]
            FE[w] = np.stack([
                z,                                              # 後で lk/pf を引く(parquet側の値で)
                dh[bidx],
                std.cpu().numpy()[bidx],
                ent.cpu().numpy()[bidx],
                pmax.cpu().numpy()[bidx],
                top2gap.cpu().numpy()[bidx],
                p2r.cpu().numpy()[bidx],
                z.copy(),
            ], 1).astype(np.float64)
    print(f"fold{f} done ({time.time()-t0:.0f}s)", flush=True)

# ---- parquet 行整列 → 8列確定(z−lk, z−pf_ancc は parquet の列で計算) ----
import pyarrow.parquet as pq
tb = pq.read_table(os.path.join(A63, "data", "train.parquet"),
                   columns=["id", "well", "last_known_tvt", "pf_ancc"])
ids = tb.column("id").to_pandas().astype(str)
gw = tb.column("well").to_pandas().astype(str).to_numpy()
lk = tb.column("last_known_tvt").to_numpy(zero_copy_only=False).astype(np.float64)
pfa = tb.column("pf_ancc").to_numpy(zero_copy_only=False).astype(np.float64)
rn = ids.str.rsplit("_", n=1, expand=True)[1].astype(int).to_numpy()
N = len(ids)
OUT = np.full((N, len(COLS)), np.nan, np.float32)
hit = 0
for w in FE:
    ii = np.where(gw == w)[0]
    if len(ii) == 0: continue
    o = np.argsort(rn[ii], kind="stable"); iio = ii[o]
    n0 = min(len(iio), len(FE[w]))
    Fw = FE[w][:n0]
    OUT[iio[:n0], 0] = (Fw[:, 0] - lk[iio[:n0]]).astype(np.float32)      # unet_delta
    OUT[iio[:n0], 1] = Fw[:, 1].astype(np.float32)                        # unet_dh
    OUT[iio[:n0], 2:7] = Fw[:, 2:7].astype(np.float32)
    OUT[iio[:n0], 7] = (Fw[:, 7] - pfa[iio[:n0]]).astype(np.float32)      # unet_vs_pfancc
    hit += n0
np.save(os.path.join(A63, "unet_feats_u.npy"), OUT)
json.dump(COLS, open(os.path.join(A63, "unet_feats_cols.json"), "w"))
fin = np.isfinite(OUT[:, 0]).mean()
print(f"\nsaved unet_feats_u.npy: {OUT.shape} 充足率={fin:.3f} ({hit:,}行) cols={COLS}")
print("列統計:")
for j, c in enumerate(COLS):
    v = OUT[:, j][np.isfinite(OUT[:, j])]
    print(f"  {c:16s} med={np.median(v):8.3f} p5={np.percentile(v,5):8.3f} p95={np.percentile(v,95):8.3f}")
