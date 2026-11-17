# -*- coding: utf-8 -*-
"""v82 夜間: train_tabular_v82a → v82b を順に実行しCV行を抽出(v79方式)。"""
import time, json, subprocess, re, os, io, sys
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
from pathlib import Path

V82 = Path(r"C:\Users\kosaka256\Documents\rogii_claude\v82")
PY = r"C:\Users\kosaka256\Documents\rogii_claude\.venv\Scripts\python.exe"
LOG = V82 / "night_v82.log"


def log(m):
    with open(LOG, "a", encoding="utf-8") as f: f.write(m + "\n")
    print(m, flush=True)


def run_nb(nbname, art):
    art.mkdir(parents=True, exist_ok=True)
    nb = json.loads((V82 / nbname).read_text(encoding="utf-8"))
    code = "\n\n".join("".join(c["source"]) for c in nb["cells"] if c.get("cell_type") == "code")
    pyf = art / nbname.replace(".ipynb", "_asrun.py"); pyf.write_text(code, encoding="utf-8")
    log(f"[v82] === RUN {nbname} === {time.strftime('%H:%M:%S')}")
    env = {**os.environ, "MPLBACKEND": "Agg", "PYTHONIOENCODING": "utf-8"}
    t0 = time.time()
    r = subprocess.run([PY, str(pyf)], cwd=str(V82), capture_output=True,
                       encoding="utf-8", errors="replace", env=env)
    out = (r.stdout or "") + "\n---STDERR---\n" + (r.stderr or "")
    (art / f"{nbname}_stdout.log").write_text(out, encoding="utf-8")
    log(f"[v82] {nbname} exit={r.returncode} ({(time.time()-t0)/60:.1f}min)")
    picked = [l.strip() for l in out.splitlines()
              if re.search(r'(OOF|単体\+pp|blend GBM|== )', l) and re.search(r'\d\.\d', l)]
    for l in picked[-12:]: log("   " + l)
    return r.returncode


def main():
    log(f"\n[v82] 起動 {time.strftime('%Y-%m-%d %H:%M:%S')}")
    rc1 = run_nb("train_tabular_v82a.ipynb", V82 / "artifacts_v82a")
    rc2 = run_nb("train_tabular_v82b.ipynb", V82 / "artifacts_v82b") if rc1 == 0 else -1
    log(f"[v82] done a={rc1} b={rc2}  (基準: v79 GBM単体+pp 6.3940 / v69_4系)")


if __name__ == "__main__":
    main()
