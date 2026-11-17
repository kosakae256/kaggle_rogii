# -*- coding: utf-8 -*-
"""v87 train_tabular ノートブック生成: v69_4 の train_tabular を土台に、cell1(データロード)だけ
   RAM安全版(base+gf+affine を最初から幅広に確保してスロット書込。hstackのピーク回避)へ差し替える。
   入力parquet=v69_4(base) + v87 affine。出力(models/features.json/oof)=v87/artifacts_v87。
   cells 2..5(モデル定義・学習・OOF評価・NNブレンド比較)は verbatim 流用。
   生成: python v87/_make_train_v87.py  → v87/train_tabular_v87.ipynb
"""
import json, io
from pathlib import Path

P = Path(r"C:\Users\kosaka256\Documents\rogii_claude")
src_nb = json.load(io.open(P / "v69" / "train_tabular_v69_4.ipynb", encoding="utf-8"))

CELL1 = r'''# ===== v87: base(v69_4) + gf(v48) + affine(v87) を id整合で"最初から幅広に"確保(hstackのRAMピーク回避) =====
import json, joblib, warnings, gc
from collections import Counter
import numpy as np, pandas as pd
from pathlib import Path
import pyarrow.parquet as pq
from lightgbm import LGBMRegressor, log_evaluation, early_stopping
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import root_mean_squared_error
warnings.filterwarnings("ignore")

PROJECT = Path(r"C:\Users\kosaka256\Documents\rogii_claude")
ART_IN  = PROJECT / "v69" / "artifacts_v69_4"          # ★入力: base parquet(転用)
ART     = PROJECT / "v87" / "artifacts_v87"; ART.mkdir(parents=True, exist_ok=True)   # ★出力: v87
FWD_PQ  = ART_IN / "data" / "train.parquet"
AFF_PQ  = ART / "data" / "train_affine_v87.parquet"     # ★v87 affine(create_affine_v87.py 出力)
USE_BACKWARD = False
MODELS = ART / "models_fwd"; MODELS.mkdir(parents=True, exist_ok=True)
N_SPLITS = 5
ROW_FRAC = 1.0
pp_params = {"alpha": 1.0, "tau": 85, "w_pf": 0.09}

USE_ORIGIN = {"base": True, "pf": True, "beam": True, "self": True, "nbr": True, "zgrad": True}
PREFIXES = [("self", "pf_self"), ("nbr", "pf_nbr"), ("beam", "beam_"), ("pf", "pf_")]
META = {"well", "id", "target", "is_backward"}
def origin_of(c):
    if c == "z_grad": return "zgrad"
    for o, pre in PREFIXES:
        if c.startswith(pre): return o
    return "base"

schema = pq.read_schema(FWD_PQ)
all_feats = [c for c in schema.names if c not in META]
features = [c for c in all_feats if USE_ORIGIN.get(origin_of(c), True)]
print("origin別 全特徴数:", dict(Counter(origin_of(c) for c in all_feats)))

N_FWD = pq.ParquetFile(FWD_PQ).metadata.num_rows

# --- 追加ブロックの列名を先に確定(gf → affine) ---
_GFP = PROJECT / "v48" / "artifacts_v48" / "gf_feature.npy"
gf_cols = ["gf_delta", "gf_std", "gf_vs_struct"] if _GFP.exists() else []
if not gf_cols: print("[warn] gf_feature.npy 無し(先に python v48/gridfilter_feature_v48.py)")
aff_cols = []
if AFF_PQ.exists():
    aff_cols = [c for c in pq.read_schema(AFF_PQ).names if c != "id"]
else:
    print("[warn] train_affine_v87.parquet 無し(先に python v87/create_affine_v87.py)")

feat_all = features + gf_cols + aff_cols
NB = len(features)
print(f"採用 base {NB} + gf {len(gf_cols)} + affine {len(aff_cols)} = {len(feat_all)} 列")
ART.joinpath("features.json").write_text(json.dumps(feat_all))
ART.joinpath("pp_params.json").write_text(json.dumps(pp_params))

# --- 幅広に一括確保(hstack不要=コピーピーク無し) ---
X = np.empty((N_FWD, len(feat_all)), np.float32)

def load_block_into(path, col_list, col_off, nrows):
    """parquet の col_list を X[:, col_off + k] へ列チャンクで書込(欠損列は NaN)。"""
    names = set(pq.read_schema(path).names)
    for j0 in range(0, len(col_list), 24):
        sub = col_list[j0:j0 + 24]; have = [c for c in sub if c in names]
        tb = pq.read_table(path, columns=have) if have else None
        for jj, c in enumerate(sub):
            j = col_off + j0 + jj
            X[:nrows, j] = tb.column(c).to_numpy(zero_copy_only=False).astype(np.float32, copy=False) if c in names else np.nan
        del tb
        if (j0 // 24) % 5 == 4: gc.collect()

# base
load_block_into(FWD_PQ, features, 0, N_FWD)
# gf(v48): np.load、行順=parquet行順(v69_4と同じ)
if gf_cols:
    _GF = np.nan_to_num(np.load(_GFP)[:N_FWD], nan=0.0).astype(np.float32)
    X[:, NB:NB + 3] = _GF; del _GF; gc.collect()
# affine(v87): id整合。行順一致なら positional、ずれてれば reindex
if aff_cols:
    _fid = pq.read_table(FWD_PQ, columns=["id"]).column("id").to_pandas().to_numpy()
    _aid = pq.read_table(AFF_PQ, columns=["id"]).column("id").to_pandas().to_numpy()
    _off = NB + len(gf_cols)
    if len(_aid) == N_FWD and np.array_equal(_aid, _fid):
        load_block_into(AFF_PQ, aff_cols, _off, N_FWD)     # 行順一致=そのまま列チャンク書込
        print("[v87] affine: 行順一致 → positional 書込")
    else:
        _pos = pd.Series(np.arange(len(_aid)), index=_aid).reindex(_fid).to_numpy()  # fwd各行→affine行index
        _ok = np.isfinite(_pos); _posi = np.where(_ok, _pos, 0).astype(np.int64)
        for j0 in range(0, len(aff_cols), 24):
            sub = aff_cols[j0:j0 + 24]; tb = pq.read_table(AFF_PQ, columns=sub)
            for jj, c in enumerate(sub):
                col = tb.column(c).to_numpy(zero_copy_only=False).astype(np.float32)[_posi]
                col[~_ok] = np.nan; X[:, _off + j0 + jj] = col
            del tb
        print(f"[v87] affine: reindex 書込(未マッチ {int((~_ok).sum())} 行=NaN)")
    del _fid, _aid; gc.collect()

# ラベル/群/pp材料
sf = pq.read_table(FWD_PQ, columns=["well", "target", "last_known_tvt", "md_since", "pf_ancc"])
g = sf.column("well").to_pandas().to_numpy()
y = sf.column("target").to_numpy(zero_copy_only=False).astype(np.float32)
lk  = sf.column("last_known_tvt").to_numpy(zero_copy_only=False).astype(np.float64)
mds = sf.column("md_since").to_numpy(zero_copy_only=False).astype(np.float64)
pf_abs = sf.column("pf_ancc").to_numpy(zero_copy_only=False).astype(np.float64)
del sf; gc.collect()
print(f"rows: {N_FWD} | features {len(feat_all)} | wells {pd.unique(g).size} | X {X.nbytes/1e9:.1f} GB")
'''

# --- cell0 md を v87 用に差し替え、cell1 を上記に置換、cells2-5 verbatim ---
CELL0 = ("# v87 tabular学習 — v69_4(base 特徴一式 × PF5バンク tw/self/nbr) + **affine較正PF 15本**追加\n"
         "入力: v69/artifacts_v69_4/data/train.parquet(転用) + v87/artifacts_v87/data/train_affine_v87.parquet(affine)。\n"
         "affine候補(afftw/affself/affnbr)と「self/nbr/affine の違い」特徴を id整合で結合。GroupKFold(5) 順方向。出力=v87/artifacts_v87。")

cells = src_nb["cells"]
def mk(src, ct="code"):
    d = {"cell_type": ct, "metadata": {}, "source": src.splitlines(keepends=True)}
    if ct == "code": d["outputs"] = []; d["execution_count"] = None
    return d

new_cells = [mk(CELL0, "markdown"), mk(CELL1)] + cells[2:]   # cells[0]=md, cells[1]=旧ロード を置換
out = dict(src_nb); out["cells"] = new_cells
json.dump(out, io.open(P / "v87" / "train_tabular_v87.ipynb", "w", encoding="utf-8"), ensure_ascii=False)
print("saved v87/train_tabular_v87.ipynb  cells:", len(new_cells))
