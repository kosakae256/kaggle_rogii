# submit_l5_v62.ipynb 生成: submit_tabular_v52 のエンジン(cell0-3)を流用し、
# cell4 を「L5 曲線トークン 5fold平均推論」に差し替え(GBM/pp不使用、L5生出力=CV5.928と同条件)
import json, io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

nb = json.load(open(r"v52/submit_tabular_v52.ipynb", encoding="utf-8"))

cell4 = r'''# ===== 実行: test構造prior → 特徴build → ★L5(曲線トークンsoft選択)5fold平均 =====
t0 = time.time()
v46_build_test_struct(Path(_V26_DATA) / "test")
test_df = build_dataset_v28()
test_df = gf_add_to_testdf(test_df, _V41_STRUCT, Path(_V26_DATA)/"test")
test_df = v50b_add_to_testdf(test_df, Path(_V26_DATA) / "test")
print("test rows:", len(test_df), f"({time.time()-t0:.0f}s)")

# ---- L5 モデル(v62/train_l456_v62.py と同一定義) ----
import torch, torch.nn as nn
_DEV5 = "cuda" if torch.cuda.is_available() else "cpu"
K5 = 16; DIM5 = 64; STR5 = 16
class L5(nn.Module):
    def __init__(self, k=K5, d=DIM5):
        super().__init__()
        self.k = k
        self.enc = nn.Sequential(nn.Conv1d(2, d, 9, padding=4), nn.GELU(),
                                 nn.Conv1d(d, d, 9, padding=4 * 4, dilation=4), nn.GELU(),
                                 nn.Conv1d(d, d, 9, padding=4 * 16, dilation=16))
        self.emb = nn.Embedding(k, d)
        self.qry = nn.Sequential(nn.Linear(3 + 4, d), nn.GELU(), nn.Linear(d, d))
        self.res = nn.Linear(d, 1)
    def forward(self, D, S, C, m):
        B, T, k = D.shape
        med = D.median(2, keepdim=True).values
        x = torch.stack([D - med, torch.log(S)], 2)
        x = x.permute(0, 3, 2, 1).reshape(B * k, 2, T)
        hc = self.enc(x).reshape(B, k, -1, T).permute(0, 3, 1, 2)
        hc = hc + self.emb.weight[None, None]
        stats = torch.cat([C, med / 10.0, D.std(2, keepdim=True),
                           (D - med).abs().lt(2).float().mean(2, keepdim=True),
                           torch.zeros_like(med)], -1)
        q = self.qry(stats)
        att = (hc @ q[..., None])[..., 0] / (hc.shape[-1] ** 0.5)
        w = att.softmax(-1)
        return (w * D).sum(-1) + 2.0 * torch.tanh(self.res(q)[..., 0])

_L5_CANDS = [Path("/kaggle/input/rogii-v62-artifacts/nn62"),
             Path(r"C:\Users\kosaka256\Documents\rogii_claude\v62\artifacts_v62\nn62")]
_L5DIR = next(p for p in _L5_CANDS if p.exists())
NETS5 = []
for f in range(5):
    net = L5().to(_DEV5)
    net.load_state_dict(torch.load(_L5DIR / f"v62_l5_f{f}.pt", map_location=_DEV5)); net.eval()
    NETS5.append(net)
print("L5 folds loaded:", len(NETS5))

# ---- test_df → 井戸別 (D,S,C) 組み立て(v62/build_seqfeat_v62.py と同一手順・同一列順) ----
_B6 = [f"_{k}" for k in range(1, 7)]
_COLS_D = ([f"pf_ancc{s}" for s in _B6] + [f"pf_self_{k}" for k in (1,2,3)] +
           [f"pf_nbr_{k}" for k in (1,2,3)] + ["beam_med_d", "sc_ens_d", "gf_delta", "v50b_delta"])
_COLS_S = ([f"pf_ancc_std{s}" for s in _B6] + [f"pf_self_std_{k}" for k in (1,2,3)] +
           [f"pf_nbr_std_{k}" for k in (1,2,3)] + ["beam_std_d", None, "gf_std", "v50b_std"])
for c in set([c for c in _COLS_D if c] + [c for c in _COLS_S if c]):
    if c not in test_df.columns: test_df[c] = np.nan
_rn = test_df["id"].astype(str).str.rsplit("_", n=1, expand=True)[1].astype(int).to_numpy()
_gw = test_df["well"].astype(str).to_numpy()
preds_id = []; preds_tvt = []
with torch.no_grad():
    for wn in pd.unique(_gw):
        ii = np.where(_gw == wn)[0]
        o = np.argsort(_rn[ii], kind="stable"); iio = ii[o]
        sp = _V41_STRUCT.get(wn)
        n = len(iio)
        if sp is None or len(sp["tvt"]) < n:
            st = np.full(n, float(test_df["last_known_tvt"].to_numpy()[iio[0]]))
        else:
            st = np.asarray(sp["tvt"], float)[:n]
        lkv = test_df["last_known_tvt"].to_numpy(np.float64)[iio]
        mdsv = np.maximum(test_df["md_since"].to_numpy(np.float64)[iio], 0)
        nb_ = (n + STR5 - 1) // STR5
        sel = np.clip(np.arange(nb_) * STR5 + STR5 // 2, 0, n - 1)
        Ds, Ss = [], []
        for cd, cs in zip(_COLS_D, _COLS_S):
            v = test_df[cd].to_numpy(np.float64)[iio]
            if cd in ("beam_med_d", "sc_ens_d", "gf_delta", "v50b_delta"):
                v = v + lkv                                      # delta列 → 絶対値化
            dv = v - st
            sv = (test_df[cs].to_numpy(np.float64)[iio] if cs else np.full(n, 2.5))
            Ds.append(np.clip(np.nan_to_num(dv, nan=0.0), -60, 60)[sel])
            Ss.append(np.clip(np.nan_to_num(sv, nan=3.0), 1.0, 15.0)[sel])
        D = torch.from_numpy(np.stack(Ds, 1).astype(np.float32))[None].to(_DEV5)
        S = torch.from_numpy(np.stack(Ss, 1).astype(np.float32))[None].to(_DEV5)
        bid = np.minimum(np.arange(n) // STR5, nb_ - 1)
        wgt = np.bincount(bid, minlength=nb_).astype(np.float64)
        C = torch.from_numpy(np.stack([mdsv[sel] / 1000.0, (lkv - st)[sel] / 10.0,
                                       np.log1p(wgt)], 1).astype(np.float32))[None].to(_DEV5)
        m = torch.ones(1, nb_).to(_DEV5)
        dh = np.mean([net(D, S, C, m)[0, :nb_].cpu().numpy() for net in NETS5], 0)
        z = st + np.asarray(dh, np.float64)[bid]
        preds_id += list(test_df["id"].to_numpy()[iio]); preds_tvt += list(z)
pred_df = pd.DataFrame({"id": preds_id, "tvt": preds_tvt})
print("predicted rows:", len(pred_df), f"({time.time()-t0:.0f}s)")
'''

nb["cells"][4]["source"] = cell4.splitlines(keepends=True)
import ast
for c in nb["cells"]:
    if c["cell_type"] == "code":
        ast.parse("".join(c["source"]))
json.dump(nb, open(r"v62/submit_l5_v62.ipynb", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("saved v62/submit_l5_v62.ipynb (cells parse OK)")
