# -*- coding: utf-8 -*-
"""v93: GRフリー TEST 錨の生成ラッパ。

v92/gen_grfree_anchor.py を GRF_SPLIT=test で実行(TRAINで場/κ/GRUを構築しTEST井を予測)し、
その出力錨を読み込んで **sig を定数 9.0 に上書き** して保存する。

★sig 修正(重要): train 錨(grfree_anchor_train.pkl)は sig=9.0 定数の bug-fix 版。
  gen_grfree_anchor.py が吐く sig は 15セット(5fold×3seed)の std(~0.5ft)で「3-seed-std」= 誤り。
  module1 の錨尤度は train と同じ sig 前提なので、test も **全井 sig=full(len(tvt), 9.0)** に統一する。

出力: v93/artifacts_v93/grfree_anchor_test.pkl  = {wid: {tvt(eval行,), sig(eval行,)=9.0}}
実行: rogii_claude/.venv/Scripts/python.exe v93/gen_grfree_test_anchor_v93.py
struct 系 artifact は一切読まない(GRフリーのみ)。
"""
import os, sys, io, pickle, subprocess
try:
    sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
except Exception:
    pass
from pathlib import Path
import numpy as np

PROJ = Path(r"C:\Users\kosaka256\Documents\rogii_claude")
ART = PROJ / "v93" / "artifacts_v93"
GEN = PROJ / "v92" / "gen_grfree_anchor.py"
RAW_OUT = ART / "grfree_anchor_test_raw.pkl"          # gen の生出力(sig=3-seed-std)
FINAL_OUT = ART / "grfree_anchor_test.pkl"            # sig=9.0 に修正後
SIG_CONST = 9.0


def main():
    py = str(PROJ / ".venv" / "Scripts" / "python.exe")
    env = dict(os.environ)
    env["GRF_SPLIT"] = "test"
    env["GRF_ANCHOR_OUT"] = str(RAW_OUT)               # ★必須(未設定だと gen が未定義 P を参照して落ちる)
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
    # 確認プリント
    sig_vals = np.unique(np.concatenate([f["sig"] for f in fixed.values()])) if fixed else np.array([])
    print(f"saved: {FINAL_OUT.name}  {len(fixed)}井  sig unique={sig_vals.tolist()}", flush=True)
    for wid, f in fixed.items():
        print(f"   {wid}: eval行={len(f['tvt'])}  tvt[min/max]={f['tvt'].min():.1f}/{f['tvt'].max():.1f}"
              f"  sig={f['sig'][0]:.1f}", flush=True)
    return FINAL_OUT


if __name__ == "__main__":
    main()
