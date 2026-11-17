# -*- coding: utf-8 -*-
"""v103 学習: v100 combo(15バンク)+ v103 の dec10/affine(emission OFF 脱相関候補)を SF_EXTRA で結合 → L5(+TCN)。
今日の発見: emission は多様性を潰し L5 を悪化 → 脱相関の新バンク(dec10=affine+間引き, tw-affine×15)を emission なしで足す。
巨大結合 parquet は作らず、seqfeat が SF_EXTRA(id整列済 parquet)から候補列を結合。出力 l5_v103_oof.npz。
env: NN_SEEDS(既定5) / SKIP_TCN(既定1=まずL5で検証) / SKIP_SEQFEAT
実行: .venv/Scripts/python.exe v103/train_v103.py"""
import os, sys, io, time, subprocess
try:
    sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
except Exception:
    pass
from pathlib import Path

V103 = Path(__file__).resolve().parent
PROJ = V103.parent
V102 = PROJ / "v102"
ART = V103 / "artifacts_v103"
FWD = ART / "fwd"
PY = os.environ.get("ROGII_PY", str(PROJ / ".venv" / "Scripts" / "python.exe"))
DATA = str(PROJ / "rogii-wellbore-geology-prediction")
NN_SEEDS = os.environ.get("NN_SEEDS", "5")
V100_COMBO = str(PROJ / "v100" / "artifacts_v100" / "data" / "train_combo.parquet")
REPS = ("ancc", "self", "nbr", "nbr_gr5", "self_graft")

# --- v100 の base 候補(51 delta, v100 combo 内)---
ED100 = (["pf_nbr_gr5_delta_%d" % k for k in (1, 2, 3)] + ["pf_self_graft_delta_%d" % k for k in (1, 2, 3)]
         + ["v97_pf_%s_delta_%d" % (r, k) for r in REPS for k in (1, 2, 3, 4)]
         + ["v100_pf_%s_delta_%d" % (r, k) for r in REPS for k in (1, 2, 3, 4, 5)])
ES100 = (["pf_nbr_gr5_std_%d" % k for k in (1, 2, 3)] + ["pf_self_graft_std_%d" % k for k in (1, 2, 3)]
         + ["v97_pf_%s_std_%d" % (r, k) for r in REPS for k in (1, 2, 3, 4)]
         + ["v100_pf_%s_std_%d" % (r, k) for r in REPS for k in (1, 2, 3, 4, 5)])

# --- v103 の新候補(dec10=1バンク×5表現, tw-affine=15バンク×tw)。emission OFF ---
DEC_D = ["dec_%s_delta_1" % t for t in REPS]                          # dec10 5表現
DEC_S = ["dec_%s_std_1" % t for t in REPS]
AFF_D = (["aff96_ancc_delta_%d" % k for k in range(1, 7)]             # v96 tw×6
         + ["aff97_ancc_delta_%d" % k for k in range(1, 5)]          # v97 tw×4
         + ["affpf5_ancc_delta_%d" % k for k in range(1, 6)])        # pf5 tw×5
AFF_S = (["aff96_ancc_std_%d" % k for k in range(1, 7)]
         + ["aff97_ancc_std_%d" % k for k in range(1, 5)]
         + ["affpf5_ancc_std_%d" % k for k in range(1, 6)])
V103_SET = os.environ.get("V103_SET", "all").lower()                  # dec(dec10のみ=affine抜き)| all(全部入れ)
if V103_SET == "dec":
    ED = ED100 + DEC_D; ES = ES100 + DEC_S
    _EXTRA_FILES = ["dec10.parquet"]; OSUF = "_v103dec"
else:
    ED = ED100 + DEC_D + AFF_D; ES = ES100 + DEC_S + AFF_S            # 51 + 5 + 15
    _EXTRA_FILES = ["dec10.parquet", "aff_v96.parquet", "aff_v97.parquet", "aff_pf5.parquet"]; OSUF = "_v103all"
SF_EXTRA = ",".join(str(FWD / f) for f in _EXTRA_FILES)
SFOUT = str(ART / f"seqfeat{OSUF}_cf.pkl")


def _run(script, env_extra, tag):
    env = dict(os.environ); env.update(env_extra); env["PYTHONUNBUFFERED"] = "1"
    print(f"\n==== {tag}: {script} ====", flush=True)
    r = subprocess.run([PY, str(V102 / script)], env=env)
    if r.returncode != 0:
        raise SystemExit(f"{tag} 失敗 (code={r.returncode})")


def main():
    t0 = time.time()
    print(f"[train_v103] SET={V103_SET}({'dec10のみ' if V103_SET=='dec' else '全部入れ'}) 追加候補={len(ED)-len(ED100)}本 → L5({NN_SEEDS}seed) emission OFF脱相関", flush=True)
    ART.mkdir(parents=True, exist_ok=True)
    import pyarrow.parquet as pq
    for f in _EXTRA_FILES:
        p = FWD / f
        assert p.exists() and pq.ParquetFile(p).metadata.num_rows > 0, f"★v103生成物が無い: {p}(gen_v103.py 完了待ち)"
    if os.environ.get("SKIP_SEQFEAT", "0") != "1":
        _run("build_seqfeat_v97.py", dict(
            SF_IN=V100_COMBO, SF_OUT=SFOUT, SF_EXTRA=SF_EXTRA,
            CANDFEAT="1", NF_MAX="2", V97_NBANK="6", V93_INPUT="fwd",
            EXTRA_DELTA=",".join(ED), EXTRA_STD=",".join(ES),
            ROGII_SCRIPTS=str(V102), ROGII_DATA=DATA), f"seqfeat[{OSUF}]")
    if os.environ.get("SKIP_TCN", "1") != "1":
        import shutil as _sh
        _fj = ART / "features_l1_fcombo.json"                      # ★TCN の base 特徴(v100)を同梱
        if not _fj.exists():
            _sh.copyfile(PROJ / "v100" / "submit_dataset_v100" / "features_l1_fcombo.json", _fj)
        _extra = ",".join(ED[len(ED100):] + ES[len(ES100):])       # ★dec/aff の予測(delta/std)を TCN 特徴に追加
        _run("train_nn_v98.py", dict(ROGII_ART98=str(ART), V93_INPUT="combo", NN_SEEDS=NN_SEEDS,
             V98_DA="0", X_FP16="1", SF_IN=V100_COMBO, SF_EXTRA=SF_EXTRA, EXTRA_FEATS=_extra,
             ROGII_DATA=DATA), f"TCN({NN_SEEDS}seed, v103特徴 X_FP16)")
    if os.environ.get("SKIP_L5", "0") != "1":
        _run("train_l456_v98.py", dict(
            ROGII_ART98=str(ART), V93_INPUT="combo", ARCH="l5", SF_IN=V100_COMBO,
            SEEDS=",".join(str(i) for i in range(int(NN_SEEDS))), V98_DA="0",
            L5_SF=SFOUT, L5_OSUF=OSUF, ROGII_DATA=DATA), f"L5[{OSUF}]({NN_SEEDS}seed)")
    print(f"\n[train_v103] SET={V103_SET} 完了 ({time.time()-t0:.0f}s) -> l5{OSUF}_oof.npz(v100=5.6684 と比較)", flush=True)


if __name__ == "__main__":
    main()
