# -*- coding: utf-8 -*-
"""Kaggle 提出/ノートブック実行の「かかった時間」を監視するツール。

コードコンペの実体:
  - 「提出」= ノートブックを Save & Run All(commit)して **kernel が実行**(これが本体=数時間)。
    → kernel status を queued/running→complete でポーリングし、running→complete の壁時計を計測。
  - その後 competition に submit するとスコアリング(数分)。
    → competition_submissions を監視し、新規提出の出現→COMPLETE と public_score を計測。

使い方(例):
  # ノートブック実行時間を計測(推奨。Save & Run を押したら起動)
  python v99/monitor_kaggle_submit.py --kernel <user>/<notebook-slug>

  # 提出スコアリングも同時に監視(コンペ既定=rogii)
  python v99/monitor_kaggle_submit.py --kernel <user>/<slug> --watch-submission

  # 提出スコアだけ監視(kernel refが分からない時)
  python v99/monitor_kaggle_submit.py --watch-submission

env でも指定可: KAGGLE_KERNEL / KAGGLE_COMPETITION / MON_INTERVAL(秒, 既定30)
ログは v99/kaggle_monitor.log にも追記。
"""
import os, sys, io, time, argparse
from datetime import datetime
try:
    sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
except Exception:
    pass

DONE_KERNEL = {"complete", "error", "cancelacknowledged", "cancelrequested"}
DONE_SUB = {"complete", "error"}
LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kaggle_monitor.log")


def _now():
    return datetime.now().strftime("%H:%M:%S")


def _fmt(sec):
    sec = int(sec); h, r = divmod(sec, 3600); m, s = divmod(r, 60)
    return (f"{h}h{m:02d}m{s:02d}s" if h else (f"{m}m{s:02d}s" if m else f"{s}s"))


def log(msg):
    line = f"[{_now()}] {msg}"
    print(line, flush=True)
    try:
        with io.open(LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def get_api():
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi(); api.authenticate()
    return api


def _status_str(obj):
    """status オブジェクト(enum/str)を小文字文字列へ。'KernelWorkerStatus.RUNNING'→'running'。"""
    s = getattr(obj, "status", obj)
    s = str(s)
    if "." in s:
        s = s.split(".")[-1]
    return s.strip().lower()


def monitor_kernel(api, kernel, interval):
    """kernel の queued→running→complete を計測。running→complete を「実行時間」として報告。"""
    log(f"=== kernel監視 開始: {kernel} (interval={interval}s) ===")
    t_first = time.time()             # 監視開始
    t_running = None                  # running を最初に観測した時刻
    last = None
    while True:
        try:
            st = api.kernels_status(kernel)
            s = _status_str(st)
        except Exception as e:
            log(f"  status取得エラー(継続): {type(e).__name__}: {str(e)[:120]}")
            time.sleep(interval); continue
        if s != last:
            log(f"  status: {last} -> {s}")
            if s == "running" and t_running is None:
                t_running = time.time()
                log(f"  ▶ running 開始を観測(監視開始から {_fmt(t_running - t_first)})")
            last = s
        if s in DONE_KERNEL:
            t_end = time.time()
            run = (t_end - t_running) if t_running else None
            log(f"=== kernel終了: status={s} ===")
            if run is not None:
                log(f"  ★ 実行時間(running→{s}) = {_fmt(run)}")
            log(f"  監視開始→終了(参考) = {_fmt(t_end - t_first)}")
            # 失敗時はログ末尾のヒント
            if s == "error":
                log("  ※ error のため実行時間は途中まで。kernels_logs で原因確認可。")
            return s
        time.sleep(interval)


def _sub_key(s):
    return (str(getattr(s, "date", "")), str(getattr(s, "description", "")), str(getattr(s, "ref", "")))


def monitor_submission(api, competition, interval, baseline_keys):
    """baseline に無い新規提出を検知し、出現→COMPLETE の壁時計と public_score を報告。"""
    log(f"=== 提出監視 開始: {competition} (既存{len(baseline_keys)}件を基準) ===")
    tracked = None; t_seen = None; last = None
    while True:
        try:
            subs = api.competition_submissions(competition)
        except Exception as e:
            log(f"  提出取得エラー(継続): {type(e).__name__}: {str(e)[:120]}")
            time.sleep(interval); continue
        if tracked is None:
            new = [s for s in subs if _sub_key(s) not in baseline_keys]
            if new:
                tracked = sorted(new, key=lambda s: str(getattr(s, "date", "")))[-1]
                t_seen = time.time()
                sub_date = getattr(tracked, "date", None)
                log(f"  ● 新規提出を検知(採点開始): desc='{getattr(tracked,'description','')}' status={_status_str(tracked)}")
                log(f"     提出時刻(server date)={sub_date}  ※採点時間はこの検知→COMPLETEで計測")
        else:
            cur = None
            for s in subs:
                if _sub_key(s) == _sub_key(tracked):
                    cur = s; break
            st = _status_str(cur) if cur is not None else "unknown"
            if st != last:
                log(f"  採点status: {last} -> {st}  (PENDING=隠しtestでノート再実行中)")
                last = st
            if st in DONE_SUB:
                dt = time.time() - t_seen
                pub = getattr(cur, "public_score", None); pri = getattr(cur, "private_score", None)
                sub_date = getattr(tracked, "date", None)
                log(f"=== 採点完了: status={st} ===")
                # ★主計測: 提出時刻(server=UTC)→ 今(UTC)。監視の起動時刻に依存せず正確(観測ラグ±interval)。
                if isinstance(sub_date, datetime):
                    by_date = (datetime.utcnow() - sub_date).total_seconds()
                    log(f"  ★ 採点時間(提出→COMPLETE, server時刻基準) = {_fmt(by_date)}  ※隠しtestでのノート再実行+スコアリング")
                log(f"  (参考) 検知→COMPLETE(監視壁時計) = {_fmt(dt)}")
                log(f"  ★ public_score = {pub}  (private={pri})")
                return st
        time.sleep(interval)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", default=os.environ.get("KAGGLE_KERNEL"),
                    help="監視するノートブック 'user/slug'(実行時間の計測=これが本命)")
    ap.add_argument("--competition", default=os.environ.get("KAGGLE_COMPETITION", "rogii-wellbore-geology-prediction"))
    ap.add_argument("--watch-submission", action="store_true", help="提出スコアリングも監視")
    ap.add_argument("--interval", type=int, default=int(os.environ.get("MON_INTERVAL", "30")))
    args = ap.parse_args()

    if not args.kernel and not args.watch_submission:
        log("エラー: --kernel か --watch-submission のどちらかを指定してください。")
        log("  例) python v99/monitor_kaggle_submit.py --kernel <user>/<slug>")
        sys.exit(2)

    api = get_api()
    log(f"認証OK. competition={args.competition}")

    # 提出監視の基準(監視開始時点の既存提出集合)
    base = set()
    if args.watch_submission:
        try:
            base = { _sub_key(s) for s in api.competition_submissions(args.competition) }
        except Exception as e:
            log(f"基準提出の取得失敗(継続): {e}")

    if args.kernel:
        monitor_kernel(api, args.kernel, args.interval)
    if args.watch_submission:
        monitor_submission(api, args.competition, args.interval, base)
    log("=== 監視終了 ===")


if __name__ == "__main__":
    main()
