# -*- coding: utf-8 -*-
"""v95: GRフリー TEST 錨の生成ラッパ(v93 版の v95 パス対応コピー)。

v92/gen_grfree_anchor.py を GRF_SPLIT=test で実行(TRAINで場/κ/GRUを構築しTEST井を予測)し、
その出力錨を読み込んで sig を定数 9.0 に上書きして保存する。v95 の PF パラメータ/錨は v93 と同一。

env:
  GRFREE_GEN     : gen_grfree_anchor.py のパス(既定 v92/gen_grfree_anchor.py。Kaggle nb では cwd に writefile)
  ROGII_ART95    : 出力先ディレクトリ(既定 v95/artifacts_v95)。grfree_anchor_test.pkl をここに保存
出力: <ROGII_ART95>/grfree_anchor_test.pkl = {wid: {tvt(eval行,), sig(eval行,)=9.0}}
"""
import os, sys, io, pickle, subprocess
try:
    sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
except Exception:
    pass
from pathlib import Path
import numpy as np

PROJ = Path(os.environ.get("ROGII_PROJ", r"C:\Users\kosaka256\Documents\rogii_claude"))
ART = Path(os.environ.get("ROGII_ART95", str(PROJ / "v95" / "artifacts_v95")))
GEN = Path(os.environ.get("GRFREE_GEN", str(PROJ / "v92" / "gen_grfree_anchor.py")))
RAW_OUT = ART / "grfree_anchor_test_raw.pkl"          # gen の生出力(sig=3-seed-std)
FINAL_OUT = ART / "grfree_anchor_test.pkl"            # sig=9.0 に修正後
SIG_CONST = 9.0


def main():
    py = sys.executable
    env = dict(os.environ)
    env["GRF_SPLIT"] = "test"
    env["GRF_ANCHOR_OUT"] = str(RAW_OUT)               # ★必須(未設定だと gen が未定義 P を参照して落ちる)
    ART.mkdir(parents=True, exist_ok=True)
    print(f"[step] run {GEN.name} (GRF_SPLIT=test) -> {RAW_OUT.name}", flush=True)
    r = subprocess.run([py, str(GEN)], env=env)
    if r.returncode != 0:
        raise SystemExit(f"gen_grfree_anchor.py が異常終了 (code={r.returncode})")

    anch = pickle.loads(RAW_OUT.read_bytes())
    print(f"[load] raw test 錨 {len(anch)}井 (sig=3-seed-std を 9.0 へ上書き)", flush=True)
    fixed = {}
    for wid, d in anch.items():
        tvt = np.asarray(d["tvt"], np.float32)
        fixed[wid] = dict(tvt=tvt, sig=np.full(len(tvt), SIG_CONST, np.float32))   # ★sig=9.0 定数
    FINAL_OUT.write_bytes(pickle.dumps(fixed, protocol=4))
    sig_vals = np.unique(np.concatenate([f["sig"] for f in fixed.values()])) if fixed else np.array([])
    print(f"saved: {FINAL_OUT.name}  {len(fixed)}井  sig unique={sig_vals.tolist()}", flush=True)
    return FINAL_OUT


if __name__ == "__main__":
    main()
