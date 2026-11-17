# -*- coding: utf-8 -*-
"""モードHMM を K を変えて実行(K=1,7 の比較)。ノートのコードセルを exec。GPU軽量・EPOCHS少なめ。"""
import os, json, matplotlib; matplotlib.use("Agg")
nb=json.load(open("v47/train_mode_hmm_v47.ipynb",encoding="utf-8"))
os.environ["V47_EPOCHS"]=os.environ.get("EP","30")
for Kv in [int(x) for x in os.environ.get("KLIST","1,7").split(",")]:
    os.environ["V47_K"]=str(Kv); g={}
    print(f"\n########## K={Kv} (EPOCHS={os.environ['V47_EPOCHS']}) ##########",flush=True)
    for c in nb["cells"]:
        if c["cell_type"]!="code": continue
        src="".join(c["source"]).replace("plt.show()","pass")
        try: exec(src, g)
        except Exception as e: print(f"  [cell skipped: {type(e).__name__} {e}]",flush=True)
    print(f"########## K={Kv} DONE ##########",flush=True)
