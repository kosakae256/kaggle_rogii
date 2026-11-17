# -*- coding: utf-8 -*-
"""v75_2 リカバリ: train_tabular_v75_2(fold再開) → [成功時のみ] train_nn_v75_2 → CV抽出。
前回の教訓: rcゲート必須(前回はtabular失敗でもnnに進んだ)。cb-2のbad allocationはフレッシュプロセス+
クリーンGPUで再挑戦(fold pkl再開でlgb×3+cb-1は学習スキップ)。再失敗ならcb残りをCPU化する(次の手)。"""
import time, json, subprocess, re, os, io, sys
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)   # ★cp932対策(前回の敗因)
from pathlib import Path

V75 = Path(r"C:\Users\kosaka256\Documents\rogii_claude\v75")
A2 = V75 / "artifacts_v75_2"
PY = r"C:\Users\kosaka256\Documents\rogii_claude\.venv\Scripts\python.exe"
LOG = A2 / "night_v75.log"


def log(m):
    with open(LOG, "a", encoding="utf-8") as f: f.write(m + "\n")
    print(m, flush=True)


def run_nb(nbname):
    nb = json.loads((V75 / nbname).read_text(encoding="utf-8"))
    code = "\n\n".join("".join(c["source"]) for c in nb["cells"] if c.get("cell_type") == "code")
    pyf = A2 / nbname.replace(".ipynb", "_asrun.py"); pyf.write_text(code, encoding="utf-8")
    outlog = A2 / nbname.replace(".ipynb", "_stdout.log")
    log(f"[p2b] === RUN {nbname} === {time.strftime('%H:%M:%S')}")
    env = {**os.environ, "MPLBACKEND": "Agg", "PYTHONIOENCODING": "utf-8"}
    t0 = time.time()
    r = subprocess.run([PY, str(pyf)], cwd=str(V75), capture_output=True,
                       encoding="utf-8", errors="replace", env=env)
    out = (r.stdout or "") + "\n---STDERR---\n" + (r.stderr or "")
    outlog.write_text(out, encoding="utf-8")
    log(f"[p2b] {nbname} exit={r.returncode} ({(time.time()-t0)/60:.1f}min) 末尾:")
    for l in out.splitlines()[-10:]: log("   " + l)
    return r.returncode


def main():
    log(f"\n[p2b] 起動 {time.strftime('%Y-%m-%d %H:%M:%S')} (tabular再開→rcゲート→nn)")
    if (A2 / "gbm_v41_oof.npy").exists():
        rc1 = 0; log("[p2b] gbm_v41_oof.npy あり → tabular完了済み。nnへ直行")
    else:
        rc1 = run_nb("train_tabular_v75_2.ipynb")
        if rc1 != 0 or not (A2 / "gbm_v41_oof.npy").exists():
            log(f"[p2b] ★tabular再失敗(exit={rc1})。nnはスキップ。次の手=catboost残りのCPU化"); return
    rc2 = run_nb("train_nn_v75_2.ipynb")
    log(f"[p2b] done tab={rc1} nn={rc2}")
    # CVサマリ追記
    S = ["=" * 60, "v75_2 リカバリ後 CV", "=" * 60]
    for nb in ["train_tabular_v75_2", "train_nn_v75_2"]:
        lf = A2 / f"{nb}_stdout.log"
        if lf.exists():
            txt = lf.read_text(encoding="utf-8", errors="replace")
            picked = [l.strip() for l in txt.splitlines()
                      if re.search(r'(OOF|oof_pp|combo|blend|rmse=|== )', l) and re.search(r'\d\.\d', l)]
            S += [f"--- {nb} ---"] + ["  " + l for l in picked[-25:]]
    (A2 / "cv_summary_v75_2.txt").write_text("\n".join(S), encoding="utf-8")
    log("\n".join(S))
    log(f"[p2b] 全完了 {time.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
