# -*- coding: utf-8 -*-
import io,sys,time
from datetime import datetime, timezone
sys.stdout=io.open(1,"w",encoding="utf-8",closefd=False)
from kaggle.api.kaggle_api_extended import KaggleApi
api=KaggleApi(); api.authenticate()
COMP="rogii-wellbore-geology-prediction"
TARGET="2026-08-02 23:54:02.643000"   # 追跡する提出(full平滑↓)
LOG="v99/kaggle_monitor.log"
def log(m):
    line=f"[{datetime.now().strftime('%H:%M:%S')}] {m}"
    print(line,flush=True)
    open(LOG,"a",encoding="utf-8").write(line+"\n")
def st_of(s): return str(getattr(s,'status','')).split('.')[-1].upper()
log(f"=== v99採点監視 開始: target提出={TARGET} ===")
last=None
while True:
    try:
        subs=api.competition_submissions(COMP)
    except Exception as e:
        log(f"取得エラー(継続): {type(e).__name__}: {str(e)[:100]}"); time.sleep(60); continue
    cur=None
    for s in subs:
        if str(getattr(s,'date',''))==TARGET: cur=s; break
    if cur is None:
        # dateが変わる可能性に備え最新を掴む
        cur=sorted(subs,key=lambda s:str(getattr(s,'date','')),reverse=True)[0]
    st=st_of(cur)
    d=getattr(cur,'date',None)
    if isinstance(d,datetime): d=d.replace(tzinfo=timezone.utc)
    el=(datetime.now(timezone.utc)-d).total_seconds() if d else 0
    if st!=last:
        log(f"status: {last} -> {st}  (経過 {int(el//3600)}h{int(el%3600//60):02d}m)")
        last=st
    if st in ("COMPLETE","ERROR"):
        h,r=divmod(int(el),3600); m,ss=divmod(r,60)
        log(f"=== 採点完了: status={st} ===")
        log(f"  ★ 採点時間(提出→COMPLETE, server基準) = {h}h{m:02d}m{ss:02d}s")
        log(f"  ★ public_score = {getattr(cur,'public_score',None)} (private={getattr(cur,'private_score',None)})")
        break
    time.sleep(60)
