# -*- coding: utf-8 -*-
"""v75 夜間オーケストレータ(2026-07-21 ユーザー就寝指示):
  Phase1: ユーザー実行中の train_tabular_v75 の完了(artifacts_v75/gbm_v41_oof.npy)を待つ
          → GPU空き待ち → train_nn_v75.ipynb を headless 実行 → CV抽出
  Phase2: create_candidates_v75_2.py → train_tabular_v75_2 → GPU待ち → train_nn_v75_2 → CV抽出
ログ: v75/artifacts_v75_2/night_v75.log / CVサマリ: cv_summary_v75_night.txt"""
import time, json, subprocess, re, os
from pathlib import Path

V75 = Path(r"C:\Users\kosaka256\Documents\rogii_claude\v75")
A1 = V75 / "artifacts_v75"; A2 = V75 / "artifacts_v75_2"; A2.mkdir(parents=True, exist_ok=True)
PY = r"C:\Users\kosaka256\Documents\rogii_claude\.venv\Scripts\python.exe"
LOG = A2 / "night_v75.log"; SUMM = A2 / "cv_summary_v75_night.txt"


def log(m):
    with open(LOG, "a", encoding="utf-8") as f: f.write(m + "\n")
    print(m, flush=True)


def wait_file(fp, grace=120, max_h=14):
    log(f"[orch] 待機: {fp} (max {max_h}h)")
    t0 = time.time()
    while True:
        if Path(fp).exists():
            log(f"[orch] 検出。安定待ち {grace}s"); time.sleep(grace); return True
        if time.time() - t0 > max_h * 3600:
            log("[orch] ★タイムアウト"); return False
        time.sleep(60)


def gpu_free_gb():
    try:
        r = subprocess.run([PY, "-c", "import torch;f,t=torch.cuda.mem_get_info();print(f/1e9)"],
                           capture_output=True, text=True, timeout=120)
        return float(r.stdout.strip().splitlines()[-1])
    except Exception:
        return -1.0


def wait_gpu(min_gb=3.0, max_min=180):
    t0 = time.time()
    while True:
        g = gpu_free_gb()
        log(f"[orch] GPU空き {g:.1f}GB (要{min_gb})")
        if g < 0 or g >= min_gb: return
        if time.time() - t0 > max_min * 60:
            log("[orch] GPU待ちタイムアウト→試行"); return
        time.sleep(300)


def run_nb(nbname, outdir):
    nb = json.loads((V75 / nbname).read_text(encoding="utf-8"))
    code = "\n\n".join("".join(c["source"]) for c in nb["cells"] if c.get("cell_type") == "code")
    pyf = outdir / nbname.replace(".ipynb", "_asrun.py"); pyf.write_text(code, encoding="utf-8")
    outlog = outdir / nbname.replace(".ipynb", "_stdout.log")
    log(f"[orch] === RUN {nbname} === {time.strftime('%H:%M:%S')}")
    env = {**os.environ, "MPLBACKEND": "Agg", "PYTHONIOENCODING": "utf-8"}
    t0 = time.time()
    r = subprocess.run([PY, str(pyf)], cwd=str(V75), capture_output=True,
                       encoding="utf-8", errors="replace", env=env)
    out = (r.stdout or "") + "\n---STDERR---\n" + (r.stderr or "")
    outlog.write_text(out, encoding="utf-8")
    log(f"[orch] {nbname} exit={r.returncode} ({(time.time()-t0)/60:.1f}min) 末尾:")
    for l in out.splitlines()[-10:]: log("   " + l)
    return r.returncode


def run_py(script):
    log(f"[orch] === RUN {script} === {time.strftime('%H:%M:%S')}")
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    t0 = time.time()
    r = subprocess.run([PY, str(V75 / script)], cwd=str(V75), capture_output=True,
                       encoding="utf-8", errors="replace", env=env)
    out = (r.stdout or "") + "\n---STDERR---\n" + (r.stderr or "")
    (A2 / (script + "_stdout.log")).write_text(out, encoding="utf-8")
    log(f"[orch] {script} exit={r.returncode} ({(time.time()-t0)/60:.1f}min) 末尾:")
    for l in out.splitlines()[-10:]: log("   " + l)
    return r.returncode


def extract_cv():
    S = ["=" * 60, "v75 / v75_2 夜間CVサマリ", "=" * 60]
    for outdir, tags in [(A1, [("train_tabular_v75", "V75 TABULAR(ユーザー実行=ログ無し可)"), ("train_nn_v75", "V75 NN")]),
                         (A2, [("train_tabular_v75_2", "V75_2 TABULAR"), ("train_nn_v75_2", "V75_2 NN")])]:
        for nb, tag in tags:
            lf = outdir / f"{nb}_stdout.log"
            S.append(f"\n--- {tag} ---")
            if not lf.exists():
                S.append("  (ログ無し)"); continue
            txt = lf.read_text(encoding="utf-8", errors="replace")
            picked = [l.strip() for l in txt.splitlines()
                      if re.search(r'(OOF|oof|combo|blend|== |rmse|RMSE|saved)', l) and re.search(r'\d\.\d\d', l)]
            S += ["  " + l for l in picked[-30:]] if picked else ["  (CV行抽出できず)"]
        for pk in ["submit_pack_nn.json"]:
            fp = outdir / pk
            if fp.exists():
                try:
                    p = json.loads(fp.read_text())
                    S.append(f"[{outdir.name} NN pack] run={p.get('run')} w_nn={p.get('w_nn')} oof_pp={p.get('oof_blend_pp')}")
                except Exception:
                    pass
    SUMM.write_text("\n".join(S), encoding="utf-8"); log("\n".join(S))


def main():
    log("\n" + "#" * 60 + f"\n[orch] 起動 {time.strftime('%Y-%m-%d %H:%M:%S')}")
    # Phase1: tabular(ユーザー)完了待ち → nn
    if not wait_file(A1 / "gbm_v41_oof.npy"):
        log("[orch] v75 tabular 未完。中止"); return
    wait_gpu()
    rc = run_nb("train_nn_v75.ipynb", A1)
    log(f"[orch] Phase1 done nn_exit={rc}")
    # Phase2: _2 チェーン
    rc0 = run_py("create_candidates_v75_2.py")
    if rc0 == 0 and (A2 / "data" / "train.parquet").exists():
        rc1 = run_nb("train_tabular_v75_2.ipynb", A2)
        wait_gpu()
        rc2 = run_nb("train_nn_v75_2.ipynb", A2)
        log(f"[orch] Phase2 done create=0 tab={rc1} nn={rc2}")
    else:
        log(f"[orch] ★create_v75_2 失敗(exit={rc0})。train系スキップ")
    extract_cv()
    log(f"[orch] 全完了 {time.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
