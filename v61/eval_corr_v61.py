# v61 ゲート2: 画像UNet(fold0)の誤差 vs v52 GBM(chain)/v50B の行レベル相関
# env: CKPT(既定 v60/artifacts_v60/v61_img_unet_f0.pt), EVPKL(画像pkl), FOLD=0
import os, sys, json, pickle
import numpy as np
import pandas as pd
import torch

P0 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("EVPKL", os.path.join(P0, "v61", "artifacts_v61", "imagesR_v61.pkl"))
os.environ.setdefault("ARCH", "unet"); os.environ.setdefault("VFRAC", "0.4")
sys.path.insert(0, os.path.join(P0, "v60"))
import importlib.util
spec = importlib.util.spec_from_file_location("tv60", os.path.join(P0, "v60", "train_v60.py"))
# train_v60 は実行スクリプトなので、必要クラスだけ複製せずロードを避ける → 最小限をここで再定義
import torch.nn as nn, torch.nn.functional as F
ND = 181; NB = 90; STEP = 0.5
DEV = "cuda" if torch.cuda.is_available() else "cpu"
if DEV == "cuda": torch.cuda.set_per_process_memory_fraction(0.4)
EVD = pickle.loads(open(os.environ["EVPKL"], "rb").read())
WELLS = sorted(EVD.keys()); CIN = int(EVD[WELLS[0]]["EV"].shape[0])
FOLD = int(os.environ.get("FOLD", "0")); val = WELLS[FOLD::5]
CH = 128

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
        self.out = nn.Conv2d(c1, 1, 1); self.ref = nn.Linear(2 * ch, 1)
        self.sdf = None
    def forward(self, x):
        B, C, T, D = x.shape
        pt = (-T) % 8; pdd = (-D) % 8
        if pt or pdd: x = F.pad(x, (0, pdd, 0, pt))
        h1 = self.e1(x); h2 = self.e2(self.pool(h1)); h3 = self.e3(self.pool(h2)); hb = self.bott(self.pool(h3))
        d3 = self.d3(torch.cat([self.u3(hb), h3], 1))
        d2 = self.d2(torch.cat([self.u2(d3), h2], 1))
        d1 = self.d1(torch.cat([self.u1(d2), h1], 1))[:, :, :T, :D]
        g = d1.mean(3).transpose(1, 2); g2, _ = self.gru(g)
        h = d1 + self.mix(g2).transpose(1, 2)[:, :, :, None]
        return self.out(h)[:, 0], 2.0 * torch.tanh(self.ref(g2)[:, :, 0]), None

def main():
    net = UNet2D().to(DEV)
    ckpt = os.environ.get("CKPT", os.path.join(P0, "v60", "artifacts_v60", "v61_img_unet_f0.pt"))
    net.load_state_dict(torch.load(ckpt, map_location=DEV)); net.eval()
    offs_t = torch.arange(-NB, NB + 1, device=DEV).float() * STEP
    EIMG = {}
    with torch.no_grad():
        for w in val:
            e = EVD[w]; T = e["EV"].shape[1]
            x = torch.from_numpy(e["EV"].astype(np.float32))[None].to(DEV)
            lg, rf, _ = net(x)
            P = torch.softmax(lg[0, :T], -1)
            dh = ((P @ offs_t) + rf[0, :T]).cpu().numpy()
            z = e["st"] + np.asarray(dh, np.float64)[e["bid"]]
            EIMG[w] = z - e["y_ev"]
    # chain誤差(v57方式の整列)
    import pyarrow.parquet as pq
    ART52 = os.path.join(P0, "v52", "artifacts_v52")
    tb = pq.read_table(os.path.join(ART52, "data", "train.parquet"),
                       columns=["id", "well", "target", "md_since", "pf_ancc", "last_known_tvt"])
    ids = tb.column("id").to_pandas().astype(str)
    g = tb.column("well").to_pandas().astype(str).to_numpy()
    y = tb.column("target").to_numpy(zero_copy_only=False).astype(np.float64)
    mds = np.maximum(tb.column("md_since").to_numpy(zero_copy_only=False).astype(np.float64), 0)
    pfa = tb.column("pf_ancc").to_numpy(zero_copy_only=False).astype(np.float64)
    lk = tb.column("last_known_tvt").to_numpy(zero_copy_only=False).astype(np.float64)
    oof = np.load(os.path.join(ART52, "gbm_v41_oof.npy")).astype(np.float64)
    pp = json.loads(open(os.path.join(ART52, "pp_params.json")).read())
    ce = (oof * (1 - pp["w_pf"]) + (pfa - lk) * pp["w_pf"]) * (1 - np.exp(-mds / pp["tau"])) * pp["alpha"] - y
    rn = ids.str.rsplit("_", n=1, expand=True)[1].astype(int).to_numpy()
    b50 = dict(np.load(os.path.join(P0, "v50", "artifacts_v50", "v50_stageBcombo_errs.npz")))
    xi, xc, xb = [], [], []
    for w in val:
        ii = np.where(g == w)[0]
        o = np.argsort(rn[ii], kind="stable")
        cew = ce[ii[o]]
        n = min(len(EIMG[w]), len(cew))
        a = EIMG[w][:n]; c = cew[:n]
        m = np.isfinite(a) & np.isfinite(c)
        xi.append(a[m]); xc.append(c[m])
        if w in b50:
            bb = b50[w].astype(np.float64)
            n2 = min(n, len(bb)); m2 = np.isfinite(a[:n2]) & np.isfinite(bb[:n2])
            xb.append((a[:n2][m2], bb[:n2][m2]))
    A = np.concatenate(xi); C = np.concatenate(xc)
    print(f"fold0 井戸{len(val)}: 画像UNet pooled={np.sqrt(np.mean(A**2)):.4f} / chain pooled={np.sqrt(np.mean(C**2)):.4f}")
    print(f"corr(画像UNet, chain)   = {np.corrcoef(A, C)[0,1]:.3f}   (ゲート: <0.9)")
    Ab = np.concatenate([t[0] for t in xb]); Bb = np.concatenate([t[1] for t in xb])
    print(f"corr(画像UNet, v50B)    = {np.corrcoef(Ab, Bb)[0,1]:.3f}")
    wopt = None; best = 1e9
    for wv in np.arange(0, 0.51, 0.05):
        r = np.sqrt(np.mean(((1 - wv) * C + wv * A) ** 2))
        if r < best: best, wopt = r, wv
    print(f"残差ブレンド(chain×画像): w={wopt:.2f} で {best:.4f} (chain単体 {np.sqrt(np.mean(C**2)):.4f})")

if __name__ == "__main__":
    main()
