# -*- coding: utf-8 -*-
"""v79 自動実行: create完了待ち(parquet安定) → make_gf_v79 → train_tabular_v79 → [rcゲート] → train_nn_v79 → CV抽出。
createはユーザーが実行中(別プロセス)。本スクリプトはparquetの出現+サイズ安定で完了を検知して後段を回す。"""
import time, json, subprocess, re, os, io, sys
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
from pathlib import Path

V79 = Path(r"C:\Users\kosaka256\Documents\rogii_claude\v79")
ART = V79 / "artifacts_v79"
PQ = ART / "data" / "train.parquet"
PY = r"C:\Users\kosaka256\Documents\rogii_claude\.venv\Scripts\python.exe"
LOG = ART / "night_v79.log"
ART.mkdir(parents=True, exist_ok=True)


def log(m):
    with open(LOG, "a", encoding="utf-8") as f: f.write(m + "\n")
    print(m, flush=True)


def wait_parquet():
    log(f"[v79] parquet待ち開始 {time.strftime('%H:%M:%S')}")
    last = -1; stable = 0
    while True:
        if PQ.exists():
            sz = PQ.stat().st_size
            if sz == last and sz > 1e8:
                stable += 1
                if stable >= 2:
                    log(f"[v79] parquet確定 {sz/1e6:.0f}MB {time.strftime('%H:%M:%S')}"); return True
            else:
                stable = 0
            last = sz
        time.sleep(60)


def run_py(fp, tag):
    log(f"[v79] === RUN {tag} === {time.strftime('%H:%M:%S')}")
    env = {**os.environ, "MPLBACKEND": "Agg", "PYTHONIOENCODING": "utf-8"}
    t0 = time.time()
    r = subprocess.run([PY, str(fp)], cwd=str(V79), capture_output=True,
                       encoding="utf-8", errors="replace", env=env)
    out = (r.stdout or "") + "\n---STDERR---\n" + (r.stderr or "")
    (ART / f"{tag}_stdout.log").write_text(out, encoding="utf-8")
    log(f"[v79] {tag} exit={r.returncode} ({(time.time()-t0)/60:.1f}min) 末尾:")
    for l in out.splitlines()[-8:]: log("   " + l)
    return r.returncode


def run_nb(nbname):
    nb = json.loads((V79 / nbname).read_text(encoding="utf-8"))
    code = "\n\n".join("".join(c["source"]) for c in nb["cells"] if c.get("cell_type") == "code")
    pyf = ART / nbname.replace(".ipynb", "_asrun.py"); pyf.write_text(code, encoding="utf-8")
    return run_py(pyf, nbname.replace(".ipynb", ""))


def main():
    log(f"\n[v79] 起動 {time.strftime('%Y-%m-%d %H:%M:%S')}")
    wait_parquet()
    rc0 = run_py(V79 / "make_gf_v79.py", "make_gf_v79")
    if rc0 != 0 or not (ART / "gf_feature_v79.npy").exists():
        log(f"[v79] ★make_gf失敗(exit={rc0})。中断"); return
    rc1 = run_nb("train_tabular_v79.ipynb")
    ok_tab = (rc1 == 0) and any(ART.glob("*oof*.npy"))
    if not ok_tab:
        log(f"[v79] ★tabular失敗(exit={rc1}, oof npy={list(map(str, ART.glob('*oof*.npy')))})。nnスキップ"); return
    rc2 = run_nb("train_nn_v79.ipynb")
    log(f"[v79] done gf={rc0} tab={rc1} nn={rc2}")
    S = ["=" * 60, "v79 CVサマリ", "=" * 60]
    for tag in ["train_tabular_v79", "train_nn_v79"]:
        lf = ART / f"{tag}_stdout.log"
        if lf.exists():
            txt = lf.read_text(encoding="utf-8", errors="replace")
            picked = [l.strip() for l in txt.splitlines()
                      if re.search(r'(OOF|oof_pp|combo|blend|rmse=|== )', l) and re.search(r'\d\.\d', l)]
            S += [f"--- {tag} ---"] + ["  " + l for l in picked[-25:]]
    (ART / "cv_summary_v79.txt").write_text("\n".join(S), encoding="utf-8")
    log("\n".join(S))
    log(f"[v79] 全完了 {time.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
