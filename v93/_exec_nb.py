# -*- coding: utf-8 -*-
"""submit_v93.ipynb を Jupyter 相当で順次 exec(グローバル共有)し、ローカル3井で全実行・バグ検出。"""
import io,sys,json,traceback,time
sys.stdout=io.open(1,"w",encoding="utf-8",closefd=False); sys.stderr=sys.stdout
nb=json.load(open("v93/submit_v93.ipynb",encoding="utf-8"))
ns={"__name__":"__main__"}
for i,c in enumerate(nb["cells"]):
    if c["cell_type"]!="code": continue
    src="".join(c["source"])
    print(f"\n========== exec cell {i} ==========",flush=True); t=time.time()
    try:
        exec(compile(src,f"<cell{i}>","exec"),ns)
    except Exception:
        print(f"!!! cell{i} でエラー:",flush=True); traceback.print_exc(); sys.exit(1)
    print(f"---- cell{i} done ({time.time()-t:.0f}s) ----",flush=True)
print("\n=== 全セル完了 ===",flush=True)
