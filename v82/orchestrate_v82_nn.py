# -*- coding: utf-8 -*-
"""v82 NN追い掛け: tabular a/b完了(night_v82.logのdone行)を待ち → train_nn_v82a → v82b → CV抽出。"""
import time, json, subprocess, re, os, io, sys
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
from pathlib import Path

V82 = Path(r"C:\Users\kosaka256\Documents\rogii_claude\v82")
PY = r"C:\Users\kosaka256\Documents\rogii_claude\.venv\Scripts\python.exe"
LOG = V82 / "night_v82_nn.log"


def log(m):
    with open(LOG, "a", encoding="utf-8") as f: f.write(m + "\n")
    print(m, flush=True)


def run_nb(nbname, art):
    art.mkdir(parents=True, exist_ok=True)
    nb = json.loads((V82 / nbname).read_text(encoding="utf-8"))
    code = "\n\n".join("".join(c["source"]) for c in nb["cells"] if c.get("cell_type") == "code")
    pyf = art / nbname.replace(".ipynb", "_asrun.py"); pyf.write_text(code, encoding="utf-8")
    log(f"[v82nn] === RUN {nbname} === {time.strftime('%H:%M:%S')}")
    env = {**os.environ, "MPLBACKEND": "Agg", "PYTHONIOENCODING": "utf-8"}
    t0 = time.time()
    r = subprocess.run([PY, str(pyf)], cwd=str(V82), capture_output=True,
                       encoding="utf-8", errors="replace", env=env)
    out = (r.stdout or "") + "\n---STDERR---\n" + (r.stderr or "")
    (art / f"{nbname}_stdout.log").write_text(out, encoding="utf-8")
    log(f"[v82nn] {nbname} exit={r.returncode} ({(time.time()-t0)/60:.1f}min)")
    for l in [l.strip() for l in out.splitlines()
              if re.search(r'(OOF|blend|combo|単体)', l) and re.search(r'\d\.\d', l)][-10:]:
        log("   " + l)
    return r.returncode


def main():
    log(f"\n[v82nn] 起動 {time.strftime('%Y-%m-%d %H:%M:%S')} tabular完了待ち")
    while True:
        t = (V82 / "night_v82.log").read_text(encoding="utf-8") if (V82 / "night_v82.log").exists() else ""
        if "done a=" in t: break
        time.sleep(180)
    log("[v82nn] tabular完了検知 → NN開始")
    rc1 = run_nb("train_nn_v82a.ipynb", V82 / "artifacts_v82a")
    rc2 = run_nb("train_nn_v82b.ipynb", V82 / "artifacts_v82b")
    log(f"[v82nn] done nn_a={rc1} nn_b={rc2} (基準: v79 NN pp6.3232 / combo6.2420)")


if __name__ == "__main__":
    main()
