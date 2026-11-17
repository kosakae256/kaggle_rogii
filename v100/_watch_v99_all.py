# -*- coding: utf-8 -*-
import io,sys,time
from datetime import datetime,timezone
sys.stdout=io.open(1,"w",encoding="utf-8",closefd=False)
from kaggle.api.kaggle_api_extended import KaggleApi
api=KaggleApi(); api.authenticate()
COMP="rogii-wellbore-geology-prediction"
# 追跡対象=08-02 23:5x台の4提出(seed違い)
TARGETS=["2026-08-02 23:59:33","2026-08-02 23:59:03","2026-08-02 23:58:30","2026-08-02 23:54:02"]
LOG="v100/kaggle_monitor.log"
def log(m):
    line=f"[{datetime.now().strftime('%H:%M:%S')}] {m}"; print(line,flush=True)
    open(LOG,"a",encoding="utf-8").write(line+"\n")
def st(s): return str(getattr(s,'status','')).split('.')[-1].upper()
log("=== v99 seed違い4提出 全監視開始 ===")
done={}; last={}
while len(done)<len(TARGETS):
    try: subs=api.competition_submissions(COMP)
    except Exception as e: log(f"取得err(継続):{str(e)[:80]}"); time.sleep(90); continue
    by={str(s.date)[:19]:s for s in subs}
    for t in TARGETS:
        if t in done: continue
        s=by.get(t)
        if s is None: continue
        cur=st(s); d=s.date.replace(tzinfo=timezone.utc) if s.date.tzinfo is None else s.date
        el=(datetime.now(timezone.utc)-d).total_seconds()/3600
        if last.get(t)!=cur:
            log(f"[{t[11:]}] {last.get(t)}->{cur} (経過{el:.1f}h)"); last[t]=cur
        if cur in ("COMPLETE","ERROR"):
            done[t]=(cur,getattr(s,'public_score',None),el)
            log(f"★[{t[11:]}] {cur} score={getattr(s,'public_score',None)} 採点{el:.1f}h")
    time.sleep(90)
log("=== 全4完了 ===")
scores=[v[1] for v in done.values() if v[1] is not None]
if scores:
    import statistics as st2
    log(f"seed分散: scores={sorted(scores)} min={min(scores)} max={max(scores)} 幅={max(scores)-min(scores):.4f}")
