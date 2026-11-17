# v60/train_v60.py → v64/train_unet_v64.py 生成(1回実行用)
import sys
sys.stdout = open(1, "w", encoding="utf-8", closefd=False)
s = open("v60/train_v60.py", encoding="utf-8").read()

def rep(old, new, n=1):
    global s
    assert s.count(old) == n, (s.count(old), old[:70])
    s = s.replace(old, new)

rep("# v60 trainer — DecoderC系の変種学習(fold0)。v54 create と同居する小スライス運転。\n"
    "#   env: ARCH=base|ddil / SDFH=0|1 / EVPKL(既定=v53 evidenceC) / FOLD=0 / VFRAC=0.25 / TOKMAX=600\n"
    "#        PRE=合成evidenceUで事前学習するptパス(あればロード) / SAVE=保存名\n"
    "#   学習作法: epoch上限なし・early stop(pat12)・best val loss・ReduceLROnPlateau",
    "# v64 trainer — PF-free 2D UNet(9ch GR照合像+物理prior)。Tucker 3位再現路線。\n"
    "#   env: PHASE=fit|pre / ARCH=unet|base|ddil / EVPKL / FOLD / VFRAC / TOKMAX / SEED / SAVE\n"
    "#        PRE=事前学習ckptパス(fit時にロード=T1) / REAL_PKL+STMAP(pre時のdonor-foldリーク除外)\n"
    "#   PHASE=pre: EVPKL=pseudo画像pklで事前学習(val=10%固定, FOLD>=0ならそのfoldのval donor由来pseudoを除外)\n"
    "#   PHASE=fit: 実5-fold(v60と同一作法: early stop pat12 / best val / ReduceLROnPlateau)")

rep('ART = os.path.join(P0, "v60", "artifacts_v60"); os.makedirs(ART, exist_ok=True)\n'
    'EVPKL = os.environ.get("EVPKL", os.path.join(P0, "v53", "artifacts_v53", "evidenceC_v53.pkl"))\n'
    'ARCH = os.environ.get("ARCH", "ddil"); SDFH = os.environ.get("SDFH", "0") == "1"',
    'ART = os.path.join(P0, "v64", "artifacts_v64"); os.makedirs(ART, exist_ok=True)\n'
    'EVPKL = os.environ.get("EVPKL", os.path.join(ART, "imagesP_v64.pkl"))\n'
    'ARCH = os.environ.get("ARCH", "unet"); SDFH = os.environ.get("SDFH", "0") == "1"\n'
    'PHASE = os.environ.get("PHASE", "fit")   # fit=実5fold / pre=pseudo事前学習')

rep('SAVE = os.environ.get("SAVE", f"v60_{ARCH}{\'_sdf\' if SDFH else \'\'}_f{FOLD}")',
    'SAVE = os.environ.get("SAVE", (f"v64_pre_f{FOLD}" if PHASE == "pre" else f"v64_{ARCH}{\'_sdf\' if SDFH else \'\'}_f{FOLD}"))')

rep("val = WELLS[FOLD::5]; trn = [w for w in WELLS if w not in set(val)]",
    '''if PHASE == "pre":
    # 事前学習: pseudo画像pkl。FOLD>=0なら「実foldのval坑井をdonorとするpseudo」を除外(リーク防止)
    pool = list(WELLS)
    if FOLD >= 0:
        rpkl = os.environ.get("REAL_PKL", os.path.join(ART, "imagesP_v64.pkl"))
        smap = os.environ.get("STMAP", "")
        if smap and os.path.exists(rpkl):
            _rw = sorted(pickle.loads(open(rpkl, "rb").read()).keys())
            _valr = set(_rw[FOLD::5])
            _don = {r["id"]: r["donor"] for r in json.loads(open(smap, encoding="utf-8").read())["wells"]}
            pool = [w for w in pool if _don.get(w, "") not in _valr]
            print(f"[pre] donor-fold{FOLD}リーク除外後: {len(pool)}/{len(WELLS)} wells", flush=True)
    val = pool[::10]; trn = [w for w in pool if w not in set(val)]
else:
    val = WELLS[FOLD::5]; trn = [w for w in WELLS if w not in set(val)]''')

rep('print(f"\\n[{SAVE}] fold{FOLD} RMSE = {r:.4f}  (基準 DecoderC f0=5.047)")',
    'print(f"\\n[{SAVE}] {PHASE} fold{FOLD} RMSE = {r:.4f}"\n'
    '      + ("  (pseudo val=T0ゲート)" if PHASE == "pre" else "  (scratch f0基準は初回実測を docs/v64.md に記録)"))')

rep('json.dump(dict(rmse=r, arch=ARCH, sdf=SDFH, best_val=best, epochs=ep),',
    'json.dump(dict(rmse=r, arch=ARCH, sdf=SDFH, best_val=best, epochs=ep, phase=PHASE, pkl=os.path.basename(EVPKL), pre=os.environ.get("PRE", "")),')

open("v64/train_unet_v64.py", "w", encoding="utf-8").write(s)
import ast
ast.parse(s)
print("written v64/train_unet_v64.py, syntax OK")
