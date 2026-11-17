# -*- coding: utf-8 -*-
"""v93 L1: タブラー GBM バックボーン(最初の v93 CV)。

v52 train_tabular の GBM(LightGBM×3 + CatBoost×3 + XGBoost×3, GroupKFold5)をクリーン再構築。
★struct禁止: 補助 gf_feature(v48, gf_vs_struct=struct依存)/v50b は入れない。v93 parquet単体で学習。
  → GRフリー錨版 parquet の「素の地力」を測る。以後 v62 の別表現スタックのバックボーンになる。
target = TVT − last_known_tvt(=parquet 'target')。OOF pooled RMSE が overall RMSE(Kaggle指標)と一致。
resumable: fold pkl を models_l1_fwd/{name}/fold{k}.pkl に保存。既存はロード。
実行: ./.venv/Scripts/python.exe v93/train_l1_gbm_v93.py   (GPU, 9モデル×5fold=重い。RESUME可)
"""
import io, os, sys, json, gc, warnings
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
warnings.filterwarnings("ignore")
from pathlib import Path
from collections import Counter
import numpy as np, pandas as pd
import pyarrow.parquet as pq
import joblib
from lightgbm import LGBMRegressor, log_evaluation, early_stopping
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import root_mean_squared_error

PROJ = Path(r"C:\Users\kosaka256\Documents\rogii_claude")
ART = PROJ / "v97" / "artifacts_v97"
# ---- 入力切替(env V93_INPUT) ----
#   "fwdbwd"(既定): train_fwdbwd.parquet(forward pf_* + backward bpf_* 30列)。逆方向の転移テスト。
#   "fwd"        : train.parquet(forward-only。既存 baseline と同名出力=上書きしない別系統)。
V93_INPUT = os.environ.get("V93_INPUT", "fwd").lower()   # ★v95: forward-only 既定(逆方向なし)
INPUT_MAP = {                                     # V93_INPUT -> (parquet ファイル名, 出力SUF)
    "fwd":           ("train.parquet",               ""),
    "fwdbwd":        ("train_fwdbwd.parquet",        "_fwdbwd"),
    "fwdbwd_anchor": ("train_fwdbwd_anchor.parquet", "_fwdbwd_anchor"),
}
if V93_INPUT not in INPUT_MAP:
    raise SystemExit(f"[v93 L1] 未知の V93_INPUT={V93_INPUT} (fwd / fwdbwd / fwdbwd_anchor)")
_fn, SUF = INPUT_MAP[V93_INPUT]
PQ = ART / "data" / _fn
# ★実験用 override: GBM_IN で入力parquet, GBM_SUF で出力SUFを差し替え(既存出力を上書きしない)
if os.environ.get("GBM_IN"):
    PQ = Path(os.environ["GBM_IN"]); SUF = os.environ.get("GBM_SUF", SUF)
MODELS = ART / ("models_l1_fwd" if (V93_INPUT == "fwd" and not os.environ.get("GBM_SUF")) else f"models_l1{SUF}")
if not PQ.exists():
    raise SystemExit(f"[v93 L1] 入力 parquet が無い: {PQ}\n  V93_INPUT={V93_INPUT} "
                     f"(fwdbwd* は build_backward_v93.py 完了後に生成される)。forward-only なら V93_INPUT=fwd。")
MODELS.mkdir(parents=True, exist_ok=True)
print(f"[v93 L1] V93_INPUT={V93_INPUT}  入力={PQ.name}  出力SUF='{SUF}'  models={MODELS.name}")
N_SPLITS = 5
META = {"well", "id", "target", "is_backward"}

# ---- 特徴列(v93 parquet 全数値列 − META。★gf/v50b(struct依存/未監査)は入れない) ----
schema = pq.read_schema(PQ)
def origin_of(c):
    if c == "z_grad": return "zgrad"
    for pre, o in [("bpf_self", "bself"), ("bpf_nbr", "bnbr"), ("bpf_", "bpf"),
                   ("pf_self", "self"), ("pf_nbr", "nbr"), ("beam_", "beam"), ("pf_", "pf")]:
        if c.startswith(pre): return o
    return "base"
features = [c for c in schema.names if c not in META]
# ---- メモリ対策: fwdbwd* は X が大きい(~15GB)。GBM のみ bpf の冗長重量列を落とす(既定ON)。
#      parquet と TCN/L5 は full(287列)のまま。GBM_BPF_LEAN=0 で full を使う。
import re as _re
if V93_INPUT != "fwd" and os.environ.get("GBM_BPF_LEAN", "1") == "1":
    def _drop_bpf(c):
        return c.startswith("bpf_") and (
            c.endswith("_rmean21") or c.endswith("_rstd21")
            or c.startswith("bpf_diff_") or c.startswith("bpf_absdiff_") or c.startswith("bpf_dev_"))
    n0 = len(features)
    features = [c for c in features if not _drop_bpf(c)]
    print(f"[GBM lean] bpf 冗長列(rolling/diff/absdiff/dev)を {n0-len(features)} 列ドロップ -> {len(features)} 列")
print("origin別 特徴数:", dict(Counter(origin_of(c) for c in features)))
nb = sum(1 for c in features if c.startswith("bpf_"))
print(f"採用 {len(features)} 列 (struct-free / 補助gf・v50b無し / うち bpf_={nb})")
ART.joinpath(f"features_l1{SUF}.json").write_text(json.dumps(features))

# ---- メモリ安全ロード(X事前確保 → 24列チャンク) ----
N = pq.ParquetFile(PQ).metadata.num_rows
X = np.empty((N, len(features)), np.float32)
names = set(schema.names)
for j0 in range(0, len(features), 24):
    sub = features[j0:j0 + 24]
    tb = pq.read_table(PQ, columns=sub)
    for jj, c in enumerate(sub):
        X[:, j0 + jj] = tb.column(c).to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
    del tb
    if (j0 // 24) % 6 == 5: gc.collect()
meta = pq.read_table(PQ, columns=["well", "target"])
g = meta.column("well").to_pandas().to_numpy()
y = meta.column("target").to_numpy(zero_copy_only=False).astype(np.float32)
del meta; gc.collect()
print(f"rows={N}  features={len(features)}  wells={pd.unique(g).size}  X={X.nbytes/1e9:.1f}GB", flush=True)

# ---- モデル(v52 と同一config) ----
lgb_params = [
    dict(boosting_type="gbdt", num_leaves=255, min_child_samples=15, subsample=0.8, subsample_freq=1,
         colsample_bytree=0.8, reg_lambda=3.0, reg_alpha=0.05, objective="regression", verbose=-1,
         n_jobs=-1, device_type="gpu", gpu_use_dp=False, max_bin=255, learning_rate=0.030, n_estimators=5000, seed=123),
    dict(n_jobs=-1, verbose=-1, reg_alpha=10.788188919840913, subsample=0.47437582748953966, num_leaves=64,
         reg_lambda=95.75401894533888, n_estimators=10000, random_state=0, boosting_type="gbdt",
         learning_rate=0.00934485794382918, colsample_bytree=0.39283351290380497, min_child_weight=0.24081152127177283,
         min_child_samples=40, device="gpu"),
    dict(n_jobs=-1, verbose=-1, reg_alpha=10.788188919840913, subsample=0.47437582748953966, num_leaves=64,
         reg_lambda=95.75401894533888, n_estimators=10000, random_state=29, boosting_type="gbdt",
         learning_rate=0.00934485794382918, colsample_bytree=0.39283351290380497, min_child_weight=0.24081152127177283,
         min_child_samples=40, device="gpu"),
]
cb_params = [
    dict(iterations=8000, depth=7, l2_leaf_reg=2.0, min_data_in_leaf=15, border_count=254, loss_function="RMSE",
         task_type="GPU", devices="0", od_type="Iter", od_wait=300, verbose=0, learning_rate=0.020, random_seed=7),
    dict(iterations=8000, depth=7, l2_leaf_reg=2.0, min_data_in_leaf=15, border_count=254, loss_function="RMSE",
         task_type="GPU", devices="0", od_type="Iter", od_wait=300, verbose=0, learning_rate=0.030, random_seed=123),
    dict(iterations=8000, depth=7, l2_leaf_reg=3.0, min_data_in_leaf=15, border_count=192, loss_function="RMSE",
         task_type="GPU", devices="0", od_type="Iter", od_wait=300, verbose=0, learning_rate=0.010, random_seed=999),
]
xgb_params = [
    dict(n_estimators=10000, learning_rate=0.03, max_depth=6, min_child_weight=10, subsample=0.85, colsample_bytree=0.85,
         reg_alpha=0.1, reg_lambda=5.0, objective="reg:squarederror", eval_metric="rmse", tree_method="hist",
         device="cuda", random_state=0, n_jobs=-1, early_stopping_rounds=250),
    dict(n_estimators=10000, learning_rate=0.02, max_depth=8, min_child_weight=8, subsample=0.80, colsample_bytree=0.80,
         reg_alpha=0.05, reg_lambda=4.0, objective="reg:squarederror", eval_metric="rmse", tree_method="hist",
         device="cuda", random_state=1, n_jobs=-1, early_stopping_rounds=250),
    dict(n_estimators=12000, learning_rate=0.02, max_depth=5, min_child_weight=25, subsample=0.75, colsample_bytree=0.75,
         reg_alpha=0.5, reg_lambda=12.0, objective="reg:squarederror", eval_metric="rmse", tree_method="hist",
         device="cuda", random_state=2, n_jobs=-1, early_stopping_rounds=250),
]
VARIANTS = ([(f"lightgbm-{i+1}", "lgb", p) for i, p in enumerate(lgb_params)] +
            [(f"catboost-{i+1}", "cb", p) for i, p in enumerate(cb_params)] +
            [(f"xgboost-{i+1}", "xgb", p) for i, p in enumerate(xgb_params)])
print("variants:", [v[0] for v in VARIANTS], flush=True)


def fit_one(kind, params, Xtr, ytr, Xva, yva):
    if kind == "lgb":
        m = LGBMRegressor(**params)
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="rmse", callbacks=[early_stopping(250), log_evaluation(0)])
    elif kind == "cb":
        m = CatBoostRegressor(**params)
        m.fit(Xtr, ytr, eval_set=(Xva, yva), early_stopping_rounds=250, use_best_model=True, verbose=0)
    else:
        m = XGBRegressor(**params); m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    return m


gkf = GroupKFold(n_splits=N_SPLITS); folds = list(gkf.split(X, y, groups=g))
oof = {name: np.zeros(len(y), np.float32) for name, _, _ in VARIANTS}
gain_acc = np.zeros(len(features), np.float64); gain_n = 0   # lgb gain importance 蓄積(TCN の TOP_K 用ランキング)
for name, kind, params in VARIANTS:
    d = MODELS / name; d.mkdir(parents=True, exist_ok=True)
    for k, (tr, va) in enumerate(folds):
        fp = d / f"fold{k}.pkl"
        if fp.exists():
            m = joblib.load(fp); tag = "load"
        else:
            m = fit_one(kind, params, X[tr], y[tr], X[va], y[va]); joblib.dump(m, fp); tag = "fit"
        oof[name][va] = m.predict(X[va])
        if kind == "lgb":
            try:
                gain_acc += m.booster_.feature_importance(importance_type="gain"); gain_n += 1
            except Exception as e:
                print(f"  [warn] lgb importance 取得失敗({name} fold{k}): {e}")
        print(f"{name} fold{k} [{tag}]: rmse={root_mean_squared_error(y[va], oof[name][va]):.4f}", flush=True)
        del m; gc.collect()
    print(f"== {name} OOF rmse = {root_mean_squared_error(y, oof[name]):.4f}", flush=True)

# ---- 特徴 importance ランキング(lgb gain 平均, 降順) = TCN/ResNet/tabnn の TOP_K 選抜用 ----
#   ★既存 feat_rank_nn.json(forward, 由来不明)は上書きしない。fwdbwd のみ feat_rank_nn_fwdbwd.json を書く。
if gain_n:                                   # ★v95: fwd(SUF="")でも書く(新規artifactsに既存が無いため)
    order = np.argsort(-gain_acc)
    rank = [features[i] for i in order]
    ART.joinpath(f"feat_rank_nn{SUF}.json").write_text(json.dumps(rank))
    nb_top = sum(1 for c in rank[:192] if c.startswith("bpf_"))
    print(f"saved feat_rank_nn{SUF}.json ({len(rank)}列, lgb gain {gain_n}model平均) "
          f"top192内 bpf_={nb_top}, top8={rank[:8]}", flush=True)

# ---- OOF 保存 + ブレンド(9モデル等重み平均) = v93 L1 CV ----
np.savez_compressed(ART / f"l1_oof{SUF}.npz", **{n: oof[n] for n in oof}, y=y, well=g)
blend = np.mean([oof[n] for n in oof], axis=0).astype(np.float32)
print(f"\n=== v93 L1 GBM (input={V93_INPUT}) ===")
for n in oof:
    print(f"  {n:14s} OOF={root_mean_squared_error(y, oof[n]):.4f}")
print(f"  ★9モデル等重み blend OOF(overall RMSE) = {root_mean_squared_error(y, blend):.4f}")
print(f"  (forward-only baseline: 9モデルblend=6.4954。逆方向で CV が下がるか比較)")
