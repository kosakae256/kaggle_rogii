# -*- coding: utf-8 -*-
"""v87 train_nn ノートブック生成: v69_4 の train_nn を土台に、
   - cell1: ART→v87(出力+features.json/models/gbm_oof)、PARQUET→v69_4 base(転用)、AFF_PQ追加、
            FORCE_PATTERNS に affine 特徴を追加(NN にも affine を渡す)
   - cell3: base parquet + gf(npy) + affine(v87 parquet) を id整合で三者結合
   - cells 0(md),2,4,5,6,7 は verbatim(ARTがv87なので出力/pack/importanceは自動でv87)
   生成: python v87/_make_train_nn_v87.py → v87/train_nn_v87.ipynb
"""
import json, io
from pathlib import Path

P = Path(r"C:\Users\kosaka256\Documents\rogii_claude")
nb = json.load(io.open(P / "v69" / "train_nn_v69_4.ipynb", encoding="utf-8"))
cells = nb["cells"]

c1 = "".join(cells[1]["source"])
# ART/PARQUET を v87/base に分離、AFF_PQ 追加
c1 = c1.replace(
    'ART     = PROJECT / "v69" / "artifacts_v69_4"\nPARQUET = ART / "data" / "train.parquet"',
    'ART     = PROJECT / "v87" / "artifacts_v87"                       # 出力+features.json/models/gbm_oof(v87 tabular)\n'
    'PARQUET = PROJECT / "v69" / "artifacts_v69_4" / "data" / "train.parquet"   # base(転用)\n'
    'AFF_PQ  = ART / "data" / "train_affine_v87.parquet"               # ★v87 affine(create_affine_v87.py 出力)')
assert "AFF_PQ" in c1, "cell1 ART/PARQUET 置換失敗"
# FORCE_PATTERNS に affine を追加(NN にも affine 候補+違い特徴を強制投入)
c1 = c1.replace(
    '    r"gf_(delta|std|vs_struct)",               # ★v48 grid filter 特徴\n]',
    '    r"gf_(delta|std|vs_struct)",               # ★v48 grid filter 特徴\n'
    '    # ★v87 affine(候補15+affine効果15+一致度+集約+係数)\n'
    '    r"pf_(afftw|affself|affnbr)(_[1-9])?",\n'
    '    r"pf_(afftw_vs_tw|affself_vs_self|affnbr_vs_nbr)(_[1-9])?",\n'
    '    r"pf_(afftw_vs_affself|afftw_vs_affnbr|affself_vs_affnbr)(_[1-9])?",\n'
    '    r"pf_aff_agree_(std|range)(_[1-9])?",\n'
    '    r"pf_(afftw|affself|affnbr)_(mean|med|std_between|mean_delta)",\n'
    '    r"pf_aff_all_(mean|med|std_between|range_between)", r"pf_afftw_mean_vs_tw",\n'
    '    r"aff_(a|b)_(tw|self|nbr)",\n]')
assert "pf_aff_all_" in c1, "cell1 FORCE_PATTERNS 追加失敗"

# cell3 を三者結合版に差し替え
CELL3 = r'''# ===== v87 データ読込(base parquet + gf npy + affine parquet を id整合で三者結合) + 前処理 + prefix =====
import pyarrow.parquet as pq
_pq_names  = set(pq.read_schema(PARQUET).names)
_aff_names = set(pq.read_schema(AFF_PQ).names) if AFF_PQ.exists() else set()
_gf_names  = ["gf_delta", "gf_std", "gf_vs_struct"]
_base_cols = [c for c in feats_nn if c in _pq_names]
_aff_cols  = [c for c in feats_nn if (c in _aff_names) and (c not in _pq_names)]
_gf_cols   = [c for c in feats_nn if (c in _gf_names) and (c not in _pq_names)]
print(f"NN特徴 内訳: base {len(_base_cols)} / affine {len(_aff_cols)} / gf {len(_gf_cols)}")

need = list(dict.fromkeys(["well", "target", "last_known_tvt", "md_since", "pf_ancc", "id"] + _base_cols))
tbl = pq.read_table(PARQUET, columns=need)
N = tbl.num_rows
g      = tbl.column("well").to_pandas().to_numpy()
y      = tbl.column("target").to_numpy(zero_copy_only=False).astype(np.float64)
lk     = tbl.column("last_known_tvt").to_numpy(zero_copy_only=False).astype(np.float64)
mds    = tbl.column("md_since").to_numpy(zero_copy_only=False).astype(np.float64)
pf_abs = tbl.column("pf_ancc").to_numpy(zero_copy_only=False).astype(np.float64)
_bid   = tbl.column("id").to_pandas().to_numpy()
X = np.empty((N, len(feats_nn)), np.float32)
# base
for j, c in enumerate(feats_nn):
    if c in _pq_names:
        X[:, j] = tbl.column(c).to_numpy(zero_copy_only=False).astype(np.float32)
del tbl; gc.collect()
# gf(npy, 行順=base parquet)
if _gf_cols:
    _GF = np.load(GF_NPY)
    for c in _gf_cols:
        X[:, feats_nn.index(c)] = np.nan_to_num(_GF[:N, _gf_names.index(c)], nan=0.0).astype(np.float32)
    del _GF; gc.collect()
# affine(v87 parquet, id整合)
if _aff_cols:
    _at = pq.read_table(AFF_PQ, columns=["id"] + _aff_cols)
    _aid = _at.column("id").to_pandas().to_numpy()
    if len(_aid) == N and np.array_equal(_aid, _bid):
        for c in _aff_cols:
            X[:, feats_nn.index(c)] = _at.column(c).to_numpy(zero_copy_only=False).astype(np.float32)
        print("[v87] affine: 行順一致 → positional")
    else:
        _pos = pd.Series(np.arange(len(_aid)), index=_aid).reindex(_bid).to_numpy()
        _ok = np.isfinite(_pos); _pi = np.where(_ok, _pos, 0).astype(np.int64)
        for c in _aff_cols:
            col = _at.column(c).to_numpy(zero_copy_only=False).astype(np.float32)[_pi]
            col[~_ok] = np.nan; X[:, feats_nn.index(c)] = col
        print(f"[v87] affine: reindex(未マッチ {int((~_ok).sum())})")
    del _at; gc.collect()

folds = list(GroupKFold(n_splits=N_SPLITS).split(np.zeros(N), y, groups=g))   # tabular v87 と同一

samp = np.arange(0, N, max(1, N // 120000))
abs2res = []
for j in range(X.shape[1]):
    v = X[samp, j].astype(np.float64); m = np.isfinite(v)
    ok = m.sum() > 100 and np.nanstd(v[m]) > 1e-9
    c_ = float(np.corrcoef(v[m], lk[samp][m])[0, 1]) if ok else 0.0
    a = bool(c_ > 0.98); abs2res.append(a)
    if a: X[:, j] = X[:, j] - lk.astype(np.float32)
print("絶対TVT→残差化:", int(sum(abs2res)), "列")

med = np.nanmedian(X, axis=0)
q75 = np.nanpercentile(X, 75, axis=0); q25 = np.nanpercentile(X, 25, axis=0)
scl = np.maximum((q75 - q25) / 1.349, 1e-6)
X = (X - med) / scl
X = np.clip(np.nan_to_num(X, nan=0.0, posinf=8.0, neginf=-8.0), -8.0, 8.0).astype(np.float32)

t_mean = float(y.mean()); t_scale = float(y.std())
yn = ((y - t_mean) / t_scale).astype(np.float32)

gbm_oof = np.load(ART / "gbm_v41_oof.npy").astype(np.float64)   # v87 tabular アンサンブルOOF(残差)
assert len(gbm_oof) == N, (len(gbm_oof), N)

wid_order = pd.unique(g)
well_rows = {}
for w in wid_order:
    idx = np.where(g == w)[0]
    well_rows[w] = idx[np.argsort(mds[idx], kind="stable")]
fold_wells = [(pd.unique(g[tr]), pd.unique(g[va])) for tr, va in folds]

# ---- 既知ゾーン(prefix)系列の抽出(キャッシュ) ----
PREF = {}
if CFG["prefix"]:
    pre_fp = ART / "prefix_cache_nn.npz"
    if pre_fp.exists():
        z = np.load(pre_fp, allow_pickle=False)
        PREF = {w: np.nan_to_num(z[w], nan=0.0, posinf=8.0, neginf=-8.0)
                for w in z.files if w in set(wid_order)}
        print("prefix cache load:", len(PREF), "wells")
    else:
        grs = []; raw = {}
        for w in wid_order:
            f = DATA / "train" / f"{w}__horizontal_well.csv"
            if not f.exists(): continue
            df = pd.read_csv(f, usecols=["MD", "Z", "GR", "TVT_input"])
            kn = df.dropna(subset=["MD", "Z", "GR", "TVT_input"]).tail(PRE_LEN)
            if len(kn) < 8: continue
            raw[w] = kn.to_numpy(np.float64); grs.append(raw[w][:, 2])
        allgr = np.concatenate(grs)
        gmed = float(np.nanmedian(allgr)); giqr = float(max(np.nanpercentile(allgr, 75) - np.nanpercentile(allgr, 25), 1e-6))
        for w, a in raw.items():
            gr_n = np.clip((a[:, 2] - gmed) / giqr, -8, 8)
            tvt_rel = (a[:, 3] - a[-1, 3]) / t_scale; z_rel = (a[:, 1] - a[-1, 1]) / 50.0; md_rel = (a[:, 0] - a[-1, 0]) / 1000.0
            PREF[w] = np.nan_to_num(np.stack([gr_n, tvt_rel, z_rel, md_rel], axis=1), nan=0.0, posinf=8.0, neginf=-8.0).astype(np.float32)
        np.savez_compressed(pre_fp, **PREF)
        print("prefix extracted:", len(PREF), "wells | gr med/iqr", round(gmed, 1), round(giqr, 1))

K_IN = X.shape[1]
print("rows", N, "| wells", len(wid_order), "| K", K_IN, "| t_mean/scale", round(t_mean, 3), "/", round(t_scale, 3))
'''

C0 = ("# v87 系列NN — TCN+Attn(v29b combo) を v69_4特徴 + **affine較正PF 15本**に載せ替え\n"
      "base=v69/artifacts_v69_4/data/train.parquet(転用) + gf(v48 npy) + v87 affine parquet を id整合で結合。\n"
      "fold は v87 tabular と完全同一 GroupKFold(5)。gbm_v41_oof.npy(v87) と正直ブレンド。出力=v87/artifacts_v87。")

def mk(src, ct="code"):
    d = {"cell_type": ct, "metadata": {}, "source": src.splitlines(keepends=True)}
    if ct == "code": d["outputs"] = []; d["execution_count"] = None
    return d

new = list(cells)
new[0] = mk(C0, "markdown")
new[1] = mk(c1)
new[3] = mk(CELL3)
out = dict(nb); out["cells"] = new
json.dump(out, io.open(P / "v87" / "train_nn_v87.ipynb", "w", encoding="utf-8"), ensure_ascii=False)
print("saved v87/train_nn_v87.ipynb  cells:", len(new))
