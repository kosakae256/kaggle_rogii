# -*- coding: utf-8 -*-
"""v74 Phase I — 物理生成(条件なしガウス面+操舵シミュレータ)による合成井戸 → GRU事前学習
   狙い: GRUの明確なデータ飢餓(773系列でep10過学習)を、物理的に生成した無限の
   (軌道, TVT)ペアで事前学習して解消する。実測統計を焼き込む:
     - 面: 異方性2成分GRF(短波長: σ~12ft/L横400ft(v56ペア差2.5/13/24ftに整合)+
            長波長: σ~25ft/L6000ft=ドリフトスケール)+地域勾配
     - 操舵: z = plan + κ·smoothW(g−plan)(κ~U(0.4,0.9), W~U(200,800), lag~U(30,150)
            = 診断0の実測レンジ)+porpoiseノイズ
     - 幾何: ラテラル方位=θ0=118.4°、井戸間隔~N(500,150)ft(実測: 最近傍median470ft)
   engineチャネルは本物と同一経路(phaseG engine + phaseD.build_seqs を再利用)で生成。
   MODE=gen(合成生成) / pretrain(事前学習) / finetune(実データfine-tune 5fold×3seed)
   env: N_CONST(300), PT_EP(15), FT_EP(10), FT_LR(1e-3), SEEDS(3)
"""
import io, sys, os, json, time
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
os.environ.update(SEG_LEN="0", VEC="1", QW="0", ROBUST="0", STEREO="0",
                  STRIDE="2", NBK="3", DGC="0", INP="1", PREFIX_CTX="0", MULTI="0",
                  ENGINE="phaseG")
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import phaseG_struct2_v74 as G

ART = G.ART
MODE = os.environ.get("MODE", "gen")
N_CONST = int(os.environ.get("N_CONST", "300"))
PT_EP = int(os.environ.get("PT_EP", "15"))
FT_EP = int(os.environ.get("FT_EP", "10"))
FT_LR = float(os.environ.get("FT_LR", "1e-3"))
SEEDS = int(os.environ.get("SEEDS", "3"))
SYNTH_NPZ = os.path.join(ART, "phaseI_synth.npz")
_C0, _S0 = G._C0, G._S0

# κ係数: submit_v74 の5-fold平均(train実データfit)を固定利用
K_AVG = dict(alpha=np.array([0.8108, 0.7240, 0.6174, 0.3772, 0.0248]),
             beta=np.array([0.8112, 0.7220, 0.6148, 0.3650, 0.0608]),
             ap=0.0284, bp=-0.001)

def _dummy_inpz():
    p = os.path.join(ART, "_empty_inp.npz")
    if not os.path.exists(p):
        np.savez(p, _dummy=np.zeros(1))
    return np.load(p)

def grf(rng, na, nc, La, Lc, sig, cell):
    """異方性GRF(スペクトル法, Matérn風 S∝(1+k²L²)^-2)"""
    kx = np.fft.fftfreq(na, d=cell) * 2 * np.pi
    ky = np.fft.rfftfreq(nc, d=cell) * 2 * np.pi
    S = (1.0 + (kx[:, None] * La) ** 2 + (ky[None, :] * Lc) ** 2) ** -2.0
    w = rng.standard_normal((na, nc))
    F = np.fft.rfft2(w) * np.sqrt(S)
    f = np.fft.irfft2(F, s=(na, nc))
    f = f / (f.std() + 1e-12) * sig
    return f

def sample_surface(rng, na=640, nc=320, cell=25.0):
    """g(a,c): a=along(θ0方向), c=across。実測統計を近似。"""
    gs = grf(rng, na, nc, La=1200.0, Lc=400.0, sig=rng.uniform(8, 15), cell=cell)
    gl = grf(rng, na, nc, La=6000.0, Lc=2500.0, sig=rng.uniform(15, 35), cell=cell)
    sa = rng.normal(0, 0.012); sc = rng.normal(0, 0.02)
    A = np.arange(na)[:, None] * cell; C = np.arange(nc)[None, :] * cell
    return gs + gl + sa * A + sc * C, cell, na, nc

def bilerp(S, cell, a, c):
    na, nc = S.shape
    ia = np.clip(a / cell, 0, na - 1.001); ic = np.clip(c / cell, 0, nc - 1.001)
    i0 = ia.astype(int); j0 = ic.astype(int)
    fa = ia - i0; fc = ic - j0
    return (S[i0, j0] * (1 - fa) * (1 - fc) + S[i0 + 1, j0] * fa * (1 - fc) +
            S[i0, j0 + 1] * (1 - fa) * fc + S[i0 + 1, j0 + 1] * fa * fc)

def smooth(x, win):
    return pd.Series(x).rolling(int(win) | 1, center=True, min_periods=1).mean().to_numpy()

def drill(rng, S, cell, na, nc, c_pos, n_rows, is_query):
    """1本掘る: 操舵モデル(診断0実測レンジ)で z を生成、TVT=g−z+b。"""
    a0 = rng.uniform(200, max(300, na * cell - n_rows - 200))
    wander = np.cumsum(rng.normal(0, 0.03, n_rows))
    a = a0 + np.arange(n_rows, dtype=float)
    c = np.clip(c_pos + wander, 100, nc * cell - 100)
    gpath = bilerp(S, cell, a, c)
    # plan: 入口値+計画勾配(地域勾配の誤差つき見積り)
    slope_plan = (gpath[-1] - gpath[0]) / n_rows + rng.normal(0, 0.004)
    plan = gpath[0] + slope_plan * np.arange(n_rows)
    kap = rng.uniform(0.4, 0.9); W = rng.uniform(200, 800); lag = int(rng.uniform(30, 150))
    track = kap * smooth(gpath - plan, W)
    track = np.r_[np.zeros(lag), track[:-lag]] if lag > 0 else track
    z = plan + track + smooth(rng.normal(0, 1.5, n_rows), 60)
    b = rng.uniform(20, 80)
    tvt = gpath - z + b
    x = a * _C0 - c * _S0
    y = a * _S0 + c * _C0
    tin = tvt.copy()
    if is_query:
        n_pre = int(rng.uniform(500, 1500))
        tin[n_pre:] = np.nan
    md = np.arange(n_rows, dtype=float)
    dxr = np.gradient(smooth(x, 101)); dyr = np.gradient(smooth(y, 101))
    return dict(md=md, x=x, y=y, z=z, tvt=tvt, tin=tin,
                kn=np.isfinite(tin), ev=~np.isfinite(tin),
                t0=int(np.where(np.isfinite(tin))[0][-1]),
                dxr=dxr, dyr=dyr, dh=np.maximum(np.hypot(dxr, dyr), 1e-6),
                zs=smooth(z, 201), anc=(tvt + z - b))

def to_gdict(d, wi, fake_tail=0):
    """phaseG エンジン用dictへ変換。fake_tail>0: 既知井に擬似評価尾を作る(build_seqs互換用)"""
    tin = d["tin"].copy()
    if fake_tail > 0:
        tin[-fake_tail:] = np.nan
    fin = np.where(np.isfinite(tin))[0]
    s = int(fin.max()); n = len(d["md"]) - 1 - s
    if n < 2:
        return None
    dX = np.diff(d["x"])[s:]; dY = np.diff(d["y"])[s:]
    Lxy = np.maximum(np.hypot(dX, dY), 1e-3)
    p_row = (dX * _C0 + dY * _S0) / Lxy
    gd = dict(s=s, n=n, dX=dX, dY=dY, ndz=-np.diff(d["z"])[s:], Lxy=Lxy,
              arc=np.cumsum(Lxy), Xl=d["x"][s:], Yl=d["y"][s:],
              anchor=float(tin[s]), perp=np.abs(p_row) < G.PGATE,
              tvt_tail=d["tvt"][s + 1:], ev_tail=~np.isfinite(tin[s + 1:]),
              R0=d["tvt"][s + 1:] - d["tvt"][s], wi=wi)
    gd["seg"] = G.segment_well(gd)
    return gd

def gen():
    import phaseD_seq_v74 as PD
    PD._INPZ = _dummy_inpz()
    t0 = time.time()
    rng = np.random.default_rng(20260721)
    pack = {}; kept = 0
    for ci in range(N_CONST):
        S, cell, na, nc = sample_surface(rng)
        n_don = int(rng.integers(4, 11))
        # across位置: 間隔~N(500,150)
        gaps = np.abs(rng.normal(500, 150, n_don + 1)).clip(250, 1200)
        cs = 400 + np.cumsum(gaps)
        cs = cs[cs < nc * cell - 400]
        if len(cs) < 3:
            continue
        qi = int(rng.integers(0, len(cs)))
        D_syn = {}
        gds = []
        for k, cp in enumerate(cs):
            nrows = int(rng.uniform(3500, 8000))
            d = drill(rng, S, cell, na, nc, cp, nrows, is_query=(k == qi))
            D_syn[f"s{ci}_{k}"] = d
        # donor gdict → mini-field
        for k, (w, d) in enumerate(D_syn.items()):
            gd = to_gdict(d, wi=k, fake_tail=(0 if d["ev"].any() else 60))
            if gd is not None:
                gd["wid"] = w
                gds.append(gd)
        qwid = f"s{ci}_{qi}"
        gq = [g_ for g_ in gds if g_["wid"] == qwid]
        gdon = [g_ for g_ in gds if g_["wid"] != qwid and g_["seg"] is not None
                and g_["seg"][3] is not None]
        if not gq or len(gdon) < 2 or gq[0]["seg"] is None:
            continue
        F = G.build_field(gdon, set(g_["wi"] for g_ in gdon))
        # dump(query + 全wellぶん: build_seqs互換)
        dump = {}
        okc = True
        for g_ in gds:
            if g_["seg"] is None:
                okc = False; break
            pred, rate, ddr, spr = G.predict(F, g_, K_AVG)
            d = D_syn[g_["wid"]]
            n = len(d["md"]); t0w = g_["s"]
            rates_full = np.zeros(n, np.float32); rates_full[t0w + 1:] = rate
            conf_full = np.zeros(n, np.float32)
            conf_full[t0w + 1:] = (np.exp(-np.maximum(ddr - 800.0, 0.0) / 1000.0) *
                                   np.exp(-np.nan_to_num(spr, nan=0.1) / 0.06))
            # build_seqs は t0(tinのNaN境界)基準なので fake_tail 井は t0w と一致するよう tin を合わせる
            if not d["ev"].any():
                d["tin"][t0w + 1:] = np.nan
                d["kn"] = np.isfinite(d["tin"]); d["ev"] = ~d["kn"]
                d["t0"] = t0w
            dump[g_["wid"]] = dict(pred=(pred).astype(np.float32),
                                   rates=rates_full, conf=conf_full)
        if not okc:
            continue
        ws_all = [qwid] + [g_["wid"] for g_ in gds if g_["wid"] != qwid]
        try:
            seqs = PD.build_seqs(D_syn, ws_all, [dump])
        except Exception:
            continue
        X, resid, d7, dtrue, evm, ei = seqs[qwid]
        if evm.sum() < 100:
            continue
        pack[f"c{ci}__X"] = X.astype(np.float16)
        pack[f"c{ci}__r"] = resid.astype(np.float16)
        kept += 1
        if kept % 50 == 0:
            print(f"  gen {kept}  ({time.time()-t0:.0f}s)", flush=True)
    np.savez_compressed(SYNTH_NPZ, **pack)
    print(f"synthetic wells={kept}  C={X.shape[1]}  saved ({time.time()-t0:.0f}s)")

def make_net(C, HID=96):
    import torch.nn as nn
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Linear(C + 1, HID)
            self.gru = nn.GRU(HID, HID, num_layers=2, batch_first=True,
                              bidirectional=True, dropout=0.1)
            self.out = nn.Linear(HID * 2, 1)
        def forward(self, x):
            h = __import__("torch").relu(self.inp(x))
            h, _ = self.gru(h)
            return self.out(h)[..., 0]
    return Net()

def train_loop(net, items, epochs, lr, dev, seed, tag):
    import torch
    import torch.nn as nn
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    rng = np.random.default_rng(seed)
    C = items[0][0].shape[1]
    BS = 12
    t0 = time.time()
    for ep in range(epochs):
        net.train()
        order = rng.permutation(len(items))
        tot, nb = 0.0, 0
        for b0 in range(0, len(order), BS):
            batch = [items[i] for i in order[b0:b0 + BS]]
            L = max(len(b[1]) for b in batch)
            xb = np.zeros((len(batch), L, C + 1), np.float32)
            yb = np.zeros((len(batch), L), np.float32)
            mb = np.zeros((len(batch), L), np.float32)
            for i, (X, yv, fl) in enumerate(batch):
                xb[i, :len(yv), :C] = X; xb[i, :len(yv), C] = fl
                yb[i, :len(yv)] = yv; mb[i, :len(yv)] = 1.0
            xb = torch.from_numpy(xb).to(dev); yb = torch.from_numpy(yb).to(dev)
            mb = torch.from_numpy(mb).to(dev)
            loss = (nn.functional.huber_loss(net(xb), yb, delta=8.0,
                                             reduction="none") * mb).sum() / mb.sum()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            tot += float(loss.detach()); nb += 1
        sch.step()
        if (ep + 1) % 5 == 0 or ep == epochs - 1:
            print(f"{tag} ep{ep+1}  loss={tot/nb:.3f}  ({time.time()-t0:.0f}s)", flush=True)
    return net

def pretrain():
    import torch
    z = np.load(SYNTH_NPZ)
    cids = sorted(set(k.split("__")[0] for k in z.files))
    items = []
    for c in cids:
        X = z[c + "__X"].astype(np.float32); r = z[c + "__r"].astype(np.float32)
        items.append((X, r, 0.0))
        items.append((X[::-1].copy(), (r[::-1] - r[-1]).copy(), 1.0))
    C = items[0][0].shape[1]
    print(f"synthetic items={len(items)}  C={C}")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ps = int(os.environ.get("PT_SEED", "-1"))
    for sd in (range(SEEDS) if ps < 0 else [ps]):
        torch.manual_seed(sd)
        net = make_net(C).to(dev)
        net = train_loop(net, items, PT_EP, 1e-3, dev, sd, f"pre s{sd}")
        torch.save(net.state_dict(), os.path.join(ART, f"phaseI_pre_s{sd}.pt"))
    print("pretrained saved")

def finetune():
    import torch
    import phaseD_seq_v74 as PD
    PD._INPZ = np.load(os.path.join(ART, "inpaint_oof.npz"))
    from sklearn.model_selection import GroupKFold
    t0 = time.time()
    D = PD.F.load()
    ws = sorted(D.keys())
    gz = np.load(os.path.join(ART, "phaseG_dump.npz"))
    dp = {}
    kept = []
    for w in ws:
        if w + "__pred" not in gz.files:
            continue
        d = D[w]; n = len(d["md"]); t0w = d["t0"]
        pred_tail = gz[w + "__pred"]
        if len(pred_tail) != n - t0w - 1:
            continue
        rates_full = np.zeros(n, np.float32); rates_full[t0w + 1:] = gz[w + "__rate"]
        conf_full = np.zeros(n, np.float32)
        ddr = gz[w + "__dd"]; spr = gz[w + "__spread"]
        conf_full[t0w + 1:] = (np.exp(-np.maximum(ddr - 800.0, 0.0) / 1000.0) *
                               np.exp(-np.nan_to_num(spr, nan=0.1) / 0.06))
        dp[w] = dict(pred=pred_tail.astype(np.float32), rates=rates_full, conf=conf_full)
        kept.append(w)
    ws = kept
    seqs = PD.build_seqs(D, ws, [dp])
    if os.environ.get("INP_ZERO", "0") == "1":
        # 提出整合構成: INPチャネル(末尾2ch)を常時ゼロ(test時と同一条件で学習)
        for w in ws:
            seqs[w][0][:, -2:] = 0.0
        print("INP_ZERO: 末尾2chをゼロ化", flush=True)
    C = seqs[ws[0]][0].shape[1]
    print(f"real wells={len(ws)}  C={C}  ({time.time()-t0:.0f}s)", flush=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    gi = np.arange(len(ws))
    if os.environ.get("FOLD_MODE", "well") == "field":
        # leave-field-out: XYクラスタ(連結成分, 5000ft)をfieldとしfield単位で分割
        from scipy.spatial import cKDTree
        cxy = np.array([[np.nanmean(D[w]["x"]), np.nanmean(D[w]["y"])] for w in ws])
        tree = cKDTree(cxy)
        pairs = tree.query_pairs(5000.0)
        parent = list(range(len(ws)))
        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]; a = parent[a]
            return a
        for a, b in pairs:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb
        fid = np.array([find(i) for i in range(len(ws))])
        uniq = {f: i for i, f in enumerate(sorted(set(fid.tolist())))}
        groups = np.array([uniq[f] for f in fid])
        print(f"FOLD_MODE=field: fields={len(uniq)}  最大field={np.bincount(groups).max()}井",
              flush=True)
        folds = list(GroupKFold(5).split(gi, gi, groups))
    else:
        folds = list(GroupKFold(5).split(gi, gi, gi))
    all_es, all_base = [], []
    OOF = {}
    for fold, (tr_i, va_i) in enumerate(folds):
        tr_w = [ws[i] for i in tr_i]; va_w = [ws[i] for i in va_i]
        items = []
        for w in tr_w:
            X, resid, d7, dtrue, evm, ei = seqs[w]
            items.append((X, resid, 0.0))
            items.append((X[::-1].copy(), (resid[::-1] - resid[-1]).copy(), 1.0))
        # 混合DA: 合成井戸を学習バッチに混ぜる(valid=実のみ。v36逆方向DAと同じ立て付け)
        aug = float(os.environ.get("AUG_SYN", "0"))
        if aug > 0:
            zs = np.load(SYNTH_NPZ)
            cids = sorted(set(k.split("__")[0] for k in zs.files))
            rng_a = np.random.default_rng(777 + fold)
            n_add = int(aug * len(items))
            pick = rng_a.choice(len(cids), size=min(n_add, len(cids)), replace=False)
            for ci in pick:
                Xs = zs[cids[ci] + "__X"].astype(np.float32)
                rs = zs[cids[ci] + "__r"].astype(np.float32)
                if os.environ.get("INP_ZERO", "0") == "1":
                    Xs[:, -2:] = 0.0
                items.append((Xs, rs, 0.0))
            print(f"fold{fold}: +合成{len(pick)}本を学習に混合", flush=True)
        preds_sum = {w: None for w in va_w}
        for sd in range(SEEDS):
            torch.manual_seed(sd + 100)
            net = make_net(C).to(dev)
            if os.environ.get("NO_PRETRAIN", "0") != "1":
                sdict = torch.load(os.path.join(ART, f"phaseI_pre_s{sd}.pt"),
                                   map_location=dev)
                net.load_state_dict(sdict)
            net = train_loop(net, items, FT_EP, FT_LR, dev, sd, f"ft f{fold} s{sd}")
            if os.environ.get("SAVE_NETS", "0") == "1":
                mdir = os.path.join(HERE, "models_v74"); os.makedirs(mdir, exist_ok=True)
                torch.save(net.state_dict(),
                           os.path.join(mdir, f"phaseI_ft_f{fold}_s{sd}.pt"))
            net.eval()
            with torch.no_grad():
                for w in va_w:
                    X, resid, d7, dtrue, evm, ei = seqs[w]
                    xb = np.zeros((1, len(resid), C + 1), np.float32)
                    xb[0, :, :C] = X
                    p = net(torch.from_numpy(xb).to(dev))[0].cpu().numpy()
                    preds_sum[w] = p if preds_sum[w] is None else preds_sum[w] + p
        for w in va_w:
            X, resid, d7, dtrue, evm, ei = seqs[w]
            p = preds_sum[w] / SEEDS
            err = ((d7 + p) - dtrue)[evm]
            all_es.append(err)
            all_base.append((d7 - dtrue)[evm])
            OOF[w + "__rows"] = ei[evm].astype(np.int32)
            OOF[w + "__err"] = err.astype(np.float32)
        rm = float(np.sqrt(np.mean(np.concatenate(all_es) ** 2)))
        print(f"fold{fold} 累積 overall={rm:.4f}  ({time.time()-t0:.0f}s)", flush=True)
    ov = float(np.sqrt(np.mean(np.concatenate(all_es) ** 2)))
    base = float(np.sqrt(np.mean(np.concatenate(all_base) ** 2)))
    print(f"\n[Phase I finetune] overall = {ov:.4f}  (engine {base:.4f} / "
          f"事前学習なし基準 9.691)")
    np.savez_compressed(os.path.join(
        ART, os.environ.get("OOF_NAME", "phaseI_oof.npz")), **OOF)
    json.dump(dict(overall=ov, engine=base, pt_ep=PT_EP, ft_ep=FT_EP, ft_lr=FT_LR),
              open(os.path.join(ART, "phaseI_result.json"), "w"))

if __name__ == "__main__":
    if MODE == "gen":
        gen()
    elif MODE == "pretrain":
        pretrain()
    else:
        finetune()
