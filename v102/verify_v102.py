# -*- coding: utf-8 -*-
"""v102 検証: 全レグ・全バンクで nn_emission(w_nn>0)と ps_combo(tau>0)が正しく有効かを bank_param で直接確認。
実行: ROGII_ART95=<cfgdir> PFMOD=pf_banks_v95 .venv/Scripts/python.exe v102/verify_v102.py"""
import os, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
pf = __import__(os.environ.get("PFMOD", "pf_banks_v95"))

print(f"--- config: {os.environ.get('ROGII_ART95', os.environ.get('ROGII_ART97'))} (mod={pf.__name__}) ---")
print(f"  PS_COMBO_TAU(global)={pf.PS_COMBO_TAU}  off_banks={pf.PS_COMBO_OFF}  smooth_mode={pf.SMOOTH_MODE_CFG}")
allok = True
for b in pf.BANK_ORDER:
    P = pf.bank_param(b)
    emit = P["_w_nn"] > 0
    psc = P["_ps_combo_tau"] > 0
    phys = P["_physics"]
    # 期待: emission=全バンクON / ps_combo=off_banks以外ON
    exp_psc = b not in pf.PS_COMBO_OFF
    ok = emit and (psc == exp_psc)
    allok &= ok
    print(f"  {b:12s} w_nn={P['_w_nn']:.3f}(emission {'ON ' if emit else 'OFF'}) "
          f"ps_combo_tau={P['_ps_combo_tau']:.1f}({'ON ' if psc else 'off'}) physics={phys}  {'OK' if ok else '★NG'}")
print(f"  => {'全バンク 期待通り' if allok else '★不一致あり'}")
