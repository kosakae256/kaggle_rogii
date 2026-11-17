# v63 の学習ノート2本 → unet特徴入り「u系」ノートを生成(1回実行用)
#   分離方針: models_fwd_u / features_u.json / feat_rank_nn_u.json / gbm_v41_oof_u.npy / RUN=combo_u
#   → 既存ベースライン(6.4467系)の成果物を一切上書きしない
import json, sys
sys.stdout = open(1, "w", encoding="utf-8", closefd=False)

def load(nb): return json.load(open(nb, encoding="utf-8"))
def save(d, nb):
    json.dump(d, open(nb, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("written:", nb)
def rep(d, old, new, expect=None):
    n = 0
    for c in d["cells"]:
        s = "".join(c["source"])
        if old in s:
            n += s.count(old); c["source"] = s.replace(old, new).splitlines(keepends=True)
    assert n > 0, f"not found: {old[:80]!r}"
    if expect is not None: assert n == expect, (n, expect, old[:60])
    return n

# ================= tabular u =================
d = load("v63/train_tabular_v63.ipynb")
rep(d, 'MODELS = ART / ("models_bwd" if USE_BACKWARD else "models_fwd")',
       'MODELS = ART / ("models_bwd_u" if USE_BACKWARD else "models_fwd_u")   # ★u系: unet特徴入り(ベースライン非破壊)')
rep(d, 'ART.joinpath("features.json").write_text(json.dumps(feat_all))',
       'ART.joinpath("features_u.json").write_text(json.dumps(feat_all))')   # 2箇所(初期+gf後)
# gfフックの直後に unet フックを挿入
old_gf_anchor = '''else:
    print("[warn] gf_feature.npy 無し。先に python v48/gridfilter_feature_v48.py を実行")'''
new_gf_anchor = old_gf_anchor + '''

# ★v63u: v64 UNet 特徴(8列, OOF, parquet行整列)を追加
_UNP = PROJECT / "v63" / "artifacts_v63" / "unet_feats_u.npy"
_UN = np.load(_UNP)
assert len(_UN) >= N_FWD, ("unet_feats_u.npy と parquet の行数不一致", len(_UN), N_FWD)
_ucols = json.loads((PROJECT / "v63" / "artifacts_v63" / "unet_feats_cols.json").read_text())
X = np.hstack([X, np.nan_to_num(_UN[:len(X)], nan=0.0).astype(np.float32)])
feat_all = feat_all + _ucols
ART.joinpath("features_u.json").write_text(json.dumps(feat_all))
print("unet特徴 追加:", _ucols)'''
rep(d, old_gf_anchor, new_gf_anchor, expect=1)
rep(d, 'np.save(ART / "gbm_v41_oof.npy", oof_ens)', 'np.save(ART / "gbm_v41_oof_u.npy", oof_ens)')
rep(d, 'print("saved:", ART / "gbm_v41_oof.npy")', 'print("saved:", ART / "gbm_v41_oof_u.npy")')
rep(d, "# ===== v63 tabular(=v47と同一構成, parquetのみ差替: 6バンク普通PF) =====",
       "# ===== v63u tabular = v63 + v64 UNet特徴8列(unet_delta/dh/std/ent/pmax/top2gap/p2/vs_pfancc) =====")
save(d, "v63/train_tabular_v63u.ipynb")

# ================= nn u =================
d = load("v63/train_nn_v63.ipynb")
rep(d, 'RUN = "combo"\nCFG = PRESETS[RUN]',
       'RUN = "combo"\nCFG = PRESETS[RUN]\nRUN = "combo_u"                     # ★u系: ckpt/結果キー分離(unet特徴入り)')
rep(d, 'features_all = json.loads((ART / "features.json").read_text())',
       'features_all = json.loads((ART / "features_u.json").read_text())')
rep(d, 'rank_fp = ART / "feat_rank_nn.json"', 'rank_fp = ART / "feat_rank_nn_u.json"')
rep(d, 'mroot = next((ART / m for m in ["models_fwd", "models_bwd", "models"] if (ART / m).exists()), None)',
       'mroot = next((ART / m for m in ["models_fwd_u", "models_bwd_u"] if (ART / m).exists()), None)')
rep(d, 'r"gf_(delta|std|vs_struct)",               # ★v48 grid filter 特徴',
       'r"gf_(delta|std|vs_struct)",               # ★v48 grid filter 特徴\n    r"unet_.*",                                # ★v64 UNet特徴(8列, 全強制)')
rep(d, 'GF_NPY  = PROJECT / "v48" / "artifacts_v48" / "gf_feature.npy"   # gf 3列は parquet 外(npy)',
       'GF_NPY  = PROJECT / "v48" / "artifacts_v48" / "gf_feature.npy"   # gf 3列は parquet 外(npy)\n'
       'UNET_NPY = PROJECT / "v63" / "artifacts_v63" / "unet_feats_u.npy"   # ★v64 UNet特徴8列(npy)')
old_load = '''_GF = np.load(GF_NPY) if _gf_cols else None
_gf_names = ["gf_delta", "gf_std", "gf_vs_struct"]
for j, c in enumerate(feats_nn):
    if c in _pq_names:
        X[:, j] = tbl.column(c).to_numpy(zero_copy_only=False).astype(np.float32)
    else:
        X[:, j] = np.nan_to_num(_GF[:N, _gf_names.index(c)], nan=0.0).astype(np.float32)'''
new_load = '''_GF = np.load(GF_NPY) if any(c.startswith("gf_") for c in _gf_cols) else None
_gf_names = ["gf_delta", "gf_std", "gf_vs_struct"]
_UN = np.load(UNET_NPY) if any(c.startswith("unet_") for c in _gf_cols) else None
_un_names = json.loads((PROJECT / "v63" / "artifacts_v63" / "unet_feats_cols.json").read_text()) if _UN is not None else []
for j, c in enumerate(feats_nn):
    if c in _pq_names:
        X[:, j] = tbl.column(c).to_numpy(zero_copy_only=False).astype(np.float32)
    elif c.startswith("unet_"):
        X[:, j] = np.nan_to_num(_UN[:N, _un_names.index(c)], nan=0.0).astype(np.float32)
    else:
        X[:, j] = np.nan_to_num(_GF[:N, _gf_names.index(c)], nan=0.0).astype(np.float32)'''
rep(d, old_load, new_load, expect=1)
rep(d, 'gbm_oof = np.load(ART / "gbm_v41_oof.npy").astype(np.float64)',
       'gbm_oof = np.load(ART / "gbm_v41_oof_u.npy").astype(np.float64)')
save(d, "v63/train_nn_v63u.ipynb")

# 構文検証
import ast
for nb in ("v63/train_tabular_v63u.ipynb", "v63/train_nn_v63u.ipynb"):
    dd = load(nb)
    for i, c in enumerate(dd["cells"]):
        if c["cell_type"] == "code":
            ast.parse("".join(c["source"]))
print("all cells parse OK")
