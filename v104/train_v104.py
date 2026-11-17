# -*- coding: utf-8 -*-
"""v104 = v103候補 + sharp/smooth(anchor-free dec10)を「スーパーセットで1回だけ」seqfeat 化。
剪定は L5 学習時に PRUNE(候補名の部分一致除外)で候補軸スライス→seqfeat 再構築不要で高速ループ。
このスクリプトは【seqfeat スーパーセット構築のみ】(重い部分)。L5 剪定ループは run_prune_v104.py。
出力: v104/artifacts_v104/seqfeat_v104_cf.pkl(+ .cand.json 候補名)。
実行: .venv/Scripts/python.exe v104/train_v104.py"""
import os, sys, io, subprocess
try:
    sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
except Exception:
    pass
from pathlib import Path

V104 = Path(__file__).resolve().parent
PROJ = V104.parent
V102 = PROJ / "v102"
V103 = PROJ / "v103"
ART = V104 / "artifacts_v104"
FWD103 = V103 / "artifacts_v103" / "fwd"     # v103 共有特徴(dec10, aff_*)
FWD104 = ART / "fwd"                          # v104 固有(dec10af = sharp/smooth 6表現)
PY = os.environ.get("ROGII_PY", str(PROJ / ".venv" / "Scripts" / "python.exe"))
DATA = str(PROJ / "rogii-wellbore-geology-prediction")
V100_COMBO = str(PROJ / "v100" / "artifacts_v100" / "data" / "train_combo.parquet")
REPS = ("ancc", "self", "nbr", "nbr_gr5", "self_graft")

# --- v103 の候補(v100 combo 51 + dec10 5 + tw-affine 15)---
ED100 = (["pf_nbr_gr5_delta_%d" % k for k in (1, 2, 3)] + ["pf_self_graft_delta_%d" % k for k in (1, 2, 3)]
         + ["v97_pf_%s_delta_%d" % (r, k) for r in REPS for k in (1, 2, 3, 4)]
         + ["v100_pf_%s_delta_%d" % (r, k) for r in REPS for k in (1, 2, 3, 4, 5)])
ES100 = (["pf_nbr_gr5_std_%d" % k for k in (1, 2, 3)] + ["pf_self_graft_std_%d" % k for k in (1, 2, 3)]
         + ["v97_pf_%s_std_%d" % (r, k) for r in REPS for k in (1, 2, 3, 4)]
         + ["v100_pf_%s_std_%d" % (r, k) for r in REPS for k in (1, 2, 3, 4, 5)])
DEC_D = ["dec_%s_delta_1" % t for t in REPS]; DEC_S = ["dec_%s_std_1" % t for t in REPS]
AFF_D = (["aff96_ancc_delta_%d" % k for k in range(1, 7)] + ["aff97_ancc_delta_%d" % k for k in range(1, 5)]
         + ["affpf5_ancc_delta_%d" % k for k in range(1, 6)])
AFF_S = (["aff96_ancc_std_%d" % k for k in range(1, 7)] + ["aff97_ancc_std_%d" % k for k in range(1, 5)]
         + ["affpf5_ancc_std_%d" % k for k in range(1, 6)])
# --- v104 追加: anchor-free dec10 sharp(=1)/smooth(=2)を【6表現】(他PFと同様)---
#     raw dec10 5表現(tw=ancc/self/nbr/nbr_gr5/self_graft)+ affine_tw(affdec_ancc)= 6表現 × 2バンク = 12候補
_DECAF_REPS = ("ancc", "self", "nbr", "nbr_gr5", "self_graft")
SS_D = [f"decaf_{r}_delta_{b}" for r in _DECAF_REPS for b in (1, 2)] + ["affdec_ancc_delta_1", "affdec_ancc_delta_2"]
SS_S = [f"decaf_{r}_std_{b}" for r in _DECAF_REPS for b in (1, 2)] + ["affdec_ancc_std_1", "affdec_ancc_std_2"]

ED = ED100 + DEC_D + AFF_D + SS_D
ES = ES100 + DEC_S + AFF_S + SS_S
_EX103 = ["dec10.parquet", "aff_v96.parquet", "aff_v97.parquet", "aff_pf5.parquet"]   # v103 共有
_EX104 = ["dec10af_train_lean.parquet"]                                                # v104 固有(sharp/smooth 6表現)
EXTRA_PATHS = [FWD103 / f for f in _EX103] + [FWD104 / f for f in _EX104]
SF_EXTRA = ",".join(str(p) for p in EXTRA_PATHS)
SFOUT = os.environ.get("SF_OUT_OVERRIDE", str(ART / "seqfeat_v104_cf.pkl"))   # ★ext版等の別出力に上書き可


def main():
    ART.mkdir(parents=True, exist_ok=True)
    import pyarrow.parquet as pq
    for p in EXTRA_PATHS:
        assert p.exists() and pq.ParquetFile(p).metadata.num_rows > 0, f"★特徴が無い: {p}"
    print(f"[train_v104] seqfeat スーパーセット構築: K={len(ED)}候補(v103 71 + sharp/smooth 6表現×2={len(SS_D)})", flush=True)
    if os.environ.get("SKIP_SEQFEAT", "0") == "1" and Path(SFOUT).exists():
        print("[train_v104] SKIP_SEQFEAT=1 既存流用"); return
    env = dict(os.environ)
    env.update(dict(SF_IN=V100_COMBO, SF_OUT=SFOUT, SF_EXTRA=SF_EXTRA,
                    CANDFEAT="1", NF_MAX="2", V97_NBANK="6", V93_INPUT="fwd",
                    EXTRA_DELTA=",".join(ED), EXTRA_STD=",".join(ES),
                    ROGII_SCRIPTS=str(V102), ROGII_DATA=DATA, PYTHONUNBUFFERED="1"))
    r = subprocess.run([PY, str(V102 / "build_seqfeat_v97.py")], env=env)
    if r.returncode != 0:
        raise SystemExit(f"seqfeat 失敗 ({r.returncode})")
    print(f"\n[train_v104] 完了 -> {SFOUT}(+ .cand.json)。剪定ループは run_prune_v104.py", flush=True)


if __name__ == "__main__":
    main()
