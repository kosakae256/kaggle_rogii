# -*- coding: utf-8 -*-
"""v104 提出オーケストレータ(Kaggle)。v103 submit を土台に、anchor-free dec10(sharp/smooth)を追加し、
剪定(PRUNE)を候補軸スライスで適用。v104 = v103候補 + sharp/smooth − PRUNE。
流れ:
 (1) v100の15バンク test 特徴生成(=v103 と同一)→ test_combo(15バンク)。
 (2) dec10 + tw-affine を test で生成(=v103)。
 (3) ★dec10af(anchor-free sharp/smooth)を test で生成(emission OFF, tw, lean=sharp/smooth列のみ)。
 (4) 結合 → v104 test_combo。
 (5) L5 推論: seqfeat(v103候補 + sharp/smooth)→ PRUNE で候補軸スライス → v104 L5 ckpt を平均。
 (6) TCN 推論(v104)→ blend(blend_v104.json, expansion 後処理)→ submission。
env: V104_L5_CKPTS(カンマ区切りの ckpt dir 名) / V104_PRUNE(剪定, カンマ区切り部分文字列) / V104_TCN
"""
import os, sys, io, json, pickle, subprocess, glob
try:
    sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False); sys.stderr = io.open(2, "w", encoding="utf-8", closefd=False)
except Exception:
    pass
from pathlib import Path
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq
import torch, torch.nn as nn

WORK = Path(os.environ.get("ROGII_PROJ", "."))
SCR = Path(os.environ.get("ROGII_SCRIPTS", str(WORK)))
DATA = Path(os.environ["ROGII_DATA"])
PY = os.environ.get("ROGII_PY", sys.executable)
CB = Path(os.environ["V100SUB"])            # v100 submit_dataset(configs/錨/diff_cols)
V104DS = Path(os.environ["V104SUB"])        # v104 dataset(v100共有物 + dec10/dec10af config + l5_v104 ckpt + TCN + blend)
V96A = WORK / "v96_art"; V97A = WORK / "v97_art"; V100A = WORK / "v100_art"
FWD = WORK / "fwd"; FWD.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
REPS = ("ancc", "self", "nbr", "nbr_gr5", "self_graft")


def _run(script, env_extra, tag):
    env = dict(os.environ); env.update(env_extra); env["PYTHONUNBUFFERED"] = "1"
    print(f"\n==== {tag}: {script} ====", flush=True)
    r = subprocess.run([PY, str(SCR / script)], env=env)
    if r.returncode != 0:
        raise SystemExit(f"{tag} 失敗 ({r.returncode})")


# ============ (1) v100 の 15バンク(v103 submit と同一機構)============
def gen_v100_15banks():
    import shutil
    for a, src in [(V96A, "v95"), (V97A, "v97"), (V100A, "pf5")]:
        if not a.exists():
            shutil.copytree(V104DS / src, a)
    baseC = dict(ROGII_SCRIPTS=str(SCR), ROGII_DATA=str(DATA), ROGII_PY=PY,
                 GRFREE_GEN=str(SCR / "gen_grfree_anchor.py"), GRFREE_FOLD_ART=str(CB / "grfree_fold_art.pkl"))
    v96pq = V96A / "test.parquet"
    b = dict(baseC, ROGII_ART95=str(V96A), ROGII_ART95B=str(V96A))
    if not (V96A / "grfree_anchor_test.pkl").exists():
        _run("gen_grfree_test_anchor_v95.py", b, "v96錨")
    _run("create_v95.py", dict(b, SPLIT="test", V93_ANCHOR_PKL=str(V96A / "grfree_anchor_test.pkl"), OUT_PARQUET=str(v96pq)), "v96 create")
    v97pq = V97A / "test.parquet"; anc = str(V97A / "grfree_anchor_test.pkl"); sim = str(V97A / "sim_grfree_test_v97.pkl")
    b7 = dict(baseC, ROGII_ART97=str(V97A), ROGII_ART97B=str(V97A))
    if (V96A / "grfree_anchor_test.pkl").exists():
        __import__("shutil").copyfile(V96A / "grfree_anchor_test.pkl", anc)
    else:
        _run("gen_grfree_test_anchor_v97.py", b7, "v97錨")
    _run("nn_emission_v97.py", dict(b7, NN_SPLIT="test", V93_ANCHOR_PKL=anc), "v97 nn-emission")
    _run("create_v97.py", dict(b7, SPLIT="test", V93_ANCHOR_PKL=anc, SIM_PKL=sim, OUT_PARQUET=str(v97pq)), "v97 create")
    v100pq = V100A / "test.parquet"
    _run("create_v95.py", dict(baseC, ROGII_ART95=str(V100A), SPLIT="test",
         V93_ANCHOR_PKL=str(V96A / "grfree_anchor_test.pkl"), OUT_PARQUET=str(v100pq),
         PF_CACHE_DIR=str(V100A / "pfcache")), "v100 create(5バンク)")
    return v96pq, v97pq, v100pq


def build_combo15(v96pq, v97pq, v100pq):
    diff = json.loads((CB / "v97_diff_cols.json").read_text()); diff100 = json.loads((CB / "v100_diff_cols.json").read_text())
    id96 = pq.read_table(v96pq, columns=["id"]).to_pandas()["id"].astype(str).to_numpy()
    assert np.array_equal(id96, pq.read_table(v97pq, columns=["id"]).to_pandas()["id"].astype(str).to_numpy())
    assert np.array_equal(id96, pq.read_table(v100pq, columns=["id"]).to_pandas()["id"].astype(str).to_numpy())
    out = WORK / "data" / "test_combo15.parquet"; (WORK / "data").mkdir(parents=True, exist_ok=True)
    pf96 = pq.ParquetFile(v96pq); pf97 = pq.ParquetFile(v97pq); pf100 = pq.ParquetFile(v100pq); writer = None
    for b96, b97, b100 in zip(pf96.iter_batches(batch_size=300000), pf97.iter_batches(batch_size=300000, columns=diff),
                              pf100.iter_batches(batch_size=300000, columns=diff100)):
        cols = list(b96.columns); names = list(b96.schema.names)
        for j, c in enumerate(diff): cols.append(b97.column(j)); names.append("v97_" + c)
        for j, c in enumerate(diff100): cols.append(b100.column(j)); names.append("v100_" + c)
        tbl = pa.Table.from_arrays(cols, names=names)
        if writer is None: writer = pq.ParquetWriter(out, tbl.schema, compression="zstd")
        writer.write_table(tbl)
    writer.close(); print(f"[combo15] 列={len(pq.read_schema(out).names)}", flush=True); return out


# ============ (2)(3) dec10 + tw-affine + dec10af を test で生成 ============
def gen_v104_fwd():
    V95 = V104DS / "v95"; V97 = V104DS / "v97"; PF5 = V104DS / "pf5"; DEC = V104DS / "dec10"; DECAF = V104DS / "dec10af"
    tanc96 = str(V96A / "grfree_anchor_test.pkl"); tanc97 = str(V97A / "grfree_anchor_test.pkl")
    tasks = [  # (name, LEG, ARTENV, artdir, MODE, PRE, extra, lean_cols or None)
        ("dec10", "v96", "ROGII_ART95", str(DEC), "affine_dec10", "dec", {"DEC": "10", "FWD_METHODS": "tw,self,nbr,nbr_gr5,self_graft", "V93_ANCHOR_PKL": tanc96}, None),
        ("aff_v96", "v96", "ROGII_ART95", str(V95), "affine", "aff96", {"FWD_METHODS": "tw", "V93_ANCHOR_PKL": tanc96}, None),
        ("aff_v97", "v97", "ROGII_ART97", str(V97), "affine", "aff97", {"FWD_METHODS": "tw", "V93_ANCHOR_PKL": tanc97}, None),
        ("aff_pf5", "v100", "ROGII_ART95", str(PF5), "affine", "affpf5", {"FWD_METHODS": "tw", "V93_ANCHOR_PKL": tanc96}, None),
        # ★dec10af: anchor-free(physics_banks=[])、emission OFF、sharp(=1)/smooth(=2)を【6表現】
        #   raw dec10 = 5表現(tw/self/nbr/nbr_gr5/self_graft)+ affine_tw(affine_dec10, tw)= 6表現目
        ("dec10af", "v96", "ROGII_ART95", str(DECAF), "dec10", "decaf",
         {"DEC": "10", "FWD_METHODS": "tw,self,nbr,nbr_gr5,self_graft", "V93_ANCHOR_PKL": tanc96},
         [f"decaf_{r}_{k}_{b}" for r in ("ancc", "self", "nbr", "nbr_gr5", "self_graft") for k in ("delta", "std") for b in (1, 2)]),
        ("dec10af_aff", "v96", "ROGII_ART95", str(DECAF), "affine_dec10", "affdec",
         {"DEC": "10", "FWD_METHODS": "tw", "V93_ANCHOR_PKL": tanc96},
         [f"affdec_ancc_{k}_{b}" for k in ("delta", "std") for b in (1, 2)]),
    ]
    outs = {}
    for (name, leg, artenv, artdir, mode, pre, extra, lean) in tasks:
        outp = FWD / f"{name}_test.parquet"
        if not outp.exists():
            env = {artenv: artdir, "ROGII_DATA": str(DATA), "ROGII_PROJ": str(SCR), "LEG": leg, "MODE": mode,
                   "FWD_SMOOTH_MODE": "full", "FWD_SPLIT": "test", "OUT_PREFIX": pre, "OUT_FWD": str(outp),
                   "STORE_CACHE_DIR": str(FWD / f"cache_{name}"), "PF_NGPU": os.environ.get("PF_NGPU", "1"),
                   "FULL_VRAM_GB": os.environ.get("FULL_VRAM_GB", "8.0")}
            env.update(extra)
            _run("build_forward_v102.py", env, f"v104 test 生成 {name}")
            if lean:                       # ★lean 化: id/well + 必要列のみに絞る(dataset/メモリ節約)
                t = pq.read_table(outp, columns=["id", "well"] + lean)
                pq.write_table(t, outp, compression="zstd")
        outs[name] = (outp, lean)
    return outs


def merge_v104(combo15, fwd_outs):
    base = pq.read_table(combo15); bid = base.column("id").to_pandas().astype(str).to_numpy()
    for name, (p, lean) in fwd_outs.items():
        t = pq.read_table(p); tid = t.column("id").to_pandas().astype(str).to_numpy()
        assert len(tid) == len(bid) and bool((tid == bid).all()), f"{name} id不一致"
        add = lean if lean else [c for c in t.schema.names if c not in ("id", "well")]
        for c in add:
            base = base.append_column(c, t.column(c))
    out = WORK / "data" / "test_combo_v104.parquet"
    pq.write_table(base, out, compression="zstd"); print(f"[merge] v104 test_combo 列={len(base.schema.names)}", flush=True); return out


# ============ (5) L5 推論(v103候補 + sharp/smooth, PRUNE 適用)============
def _mk_l5(k, cdim, nf, dim):
    class L5(nn.Module):
        def __init__(s):
            super().__init__(); s.k = k
            s.enc = nn.Sequential(nn.Conv1d(2 + nf, dim, 9, padding=4), nn.GELU(), nn.Conv1d(dim, dim, 9, padding=16, dilation=4), nn.GELU(), nn.Conv1d(dim, dim, 9, padding=64, dilation=16))
            s.emb = nn.Embedding(k, dim); s.qry = nn.Sequential(nn.Linear(cdim + 4, dim), nn.GELU(), nn.Linear(dim, dim)); s.res = nn.Linear(dim, 1)
        def forward(s, D, S, C, m, DX=None):
            B, T, kk = D.shape; med = D.median(2, keepdim=True).values
            x = torch.stack([D - med, torch.log(S)] + ([DX[..., i] for i in range(DX.shape[-1])] if DX is not None else []), 2)
            x = x.permute(0, 3, 2, 1).reshape(B * kk, 2 + (DX.shape[-1] if DX is not None else 0), T)
            hc = s.enc(x).reshape(B, kk, -1, T).permute(0, 3, 1, 2) + s.emb.weight[None, None]
            stats = torch.cat([C, med / 10.0, D.std(2, keepdim=True), (D - med).abs().lt(2).float().mean(2, keepdim=True), torch.zeros_like(med)], -1)
            q = s.qry(stats); att = (hc @ q[..., None])[..., 0] / (hc.shape[-1] ** 0.5); w = att.softmax(-1)
            return (w * D).sum(-1) + 2.0 * torch.tanh(s.res(q)[..., 0])
    return L5().to(DEVICE)


def l5_v104_predict(pq_path):
    SF = WORK / "seqfeat_test_v104.pkl"
    # v103候補(51 + dec5 + aff15)+ sharp/smooth(2)
    ED = ",".join(["pf_nbr_gr5_delta_%d" % k for k in (1, 2, 3)] + ["pf_self_graft_delta_%d" % k for k in (1, 2, 3)]
                  + ["v97_pf_%s_delta_%d" % (r, k) for r in REPS for k in (1, 2, 3, 4)]
                  + ["v100_pf_%s_delta_%d" % (r, k) for r in REPS for k in (1, 2, 3, 4, 5)]
                  + ["dec_%s_delta_1" % t for t in REPS]
                  + ["aff96_ancc_delta_%d" % k for k in range(1, 7)] + ["aff97_ancc_delta_%d" % k for k in range(1, 5)]
                  + ["affpf5_ancc_delta_%d" % k for k in range(1, 6)]
                  + ["decaf_%s_delta_%d" % (t, b) for t in REPS for b in (1, 2)] + ["affdec_ancc_delta_1", "affdec_ancc_delta_2"])
    ES = ",".join(["pf_nbr_gr5_std_%d" % k for k in (1, 2, 3)] + ["pf_self_graft_std_%d" % k for k in (1, 2, 3)]
                  + ["v97_pf_%s_std_%d" % (r, k) for r in REPS for k in (1, 2, 3, 4)]
                  + ["v100_pf_%s_std_%d" % (r, k) for r in REPS for k in (1, 2, 3, 4, 5)]
                  + ["dec_%s_std_1" % t for t in REPS]
                  + ["aff96_ancc_std_%d" % k for k in range(1, 7)] + ["aff97_ancc_std_%d" % k for k in range(1, 5)]
                  + ["affpf5_ancc_std_%d" % k for k in range(1, 6)]
                  + ["decaf_%s_std_%d" % (t, b) for t in REPS for b in (1, 2)] + ["affdec_ancc_std_1", "affdec_ancc_std_2"])
    _run("build_seqfeat_v97.py", dict(SF_IN=str(pq_path), SF_OUT=str(SF), SF_NO_TARGET="1",
         CANDFEAT="1", NF_MAX="2", V97_NBANK="6", V93_INPUT="fwd", EXTRA_DELTA=ED, EXTRA_STD=ES, ROGII_SCRIPTS=str(SCR)), "v104 seqfeat")
    d = pickle.loads(SF.read_bytes()); wells = list(d.keys()); w0 = d[wells[0]]
    # ★PRUNE: 学習時と同一の候補軸スライス(cand.json の名前で index を決める)
    prune = [x.strip() for x in os.environ.get("V104_PRUNE", "").split(",") if x.strip()]
    if prune:
        names = json.loads((Path(str(SF) + ".cand.json")).read_text(encoding="utf-8"))
        keep = np.asarray([i for i, nm in enumerate(names) if not any(p in nm for p in prune)], dtype=np.int64)
        for w in wells:
            f = d[w]; f["D"] = f["D"][:, keep]; f["S"] = f["S"][:, keep]
            if f.get("DX") is not None: f["DX"] = f["DX"][:, keep]
        print(f"[L5 v104] PRUNE {prune}: 候補 {len(names)}→{len(keep)}", flush=True)
        w0 = d[wells[0]]
    K = w0["D"].shape[1]; CDIM = w0["C"].shape[1]; NF = w0["DX"].shape[2]
    N = int(max(w["rows"].max() for w in d.values())) + 1
    acc = np.zeros(N, np.float64); cnt = np.zeros(N, np.float64)
    ckdirs = [x.strip() for x in os.environ["V104_L5_CKPTS"].split(",") if x.strip()]
    nmodel = 0
    for cd in ckdirs:
        for fp in sorted(glob.glob(str(V104DS / cd / "l5_s*_f*_model.pt"))):
            ck = torch.load(fp, map_location=DEVICE)
            net = _mk_l5(K, CDIM, NF, ck.get("dim", 64) if isinstance(ck, dict) else 64)
            net.load_state_dict(ck["state"] if isinstance(ck, dict) and "state" in ck else ck); net.eval()
            with torch.no_grad():
                for w in wells:
                    f = d[w]; T = f["D"].shape[0]
                    D = torch.from_numpy(f["D"])[None].to(DEVICE); S = torch.from_numpy(f["S"])[None].to(DEVICE)
                    C = torch.from_numpy(f["C"])[None].to(DEVICE); m = torch.ones((1, T), device=DEVICE); DX = torch.from_numpy(f["DX"])[None].to(DEVICE)
                    o = net(D, S, C, m, DX)[0].cpu().numpy().astype(np.float64)
                    acc[f["rows"]] += o[f["bid"]]; cnt[f["rows"]] += 1
            nmodel += 1; del net
            if DEVICE == "cuda": torch.cuda.empty_cache()
    l5 = np.full(N, np.nan); ok = cnt > 0; l5[ok] = acc[ok] / cnt[ok]
    print(f"[L5 v104] cover%={100 * ok.mean():.1f} ({nmodel}model, ckdirs={ckdirs})", flush=True); return l5


def _warm_imp_cache():
    """★train参照構造を1回だけ構築し IMP_CACHE で全 subprocess に共有(setup 重複排除)。"""
    cp = str(WORK / "imp_cache.pkl")
    if not os.path.exists(cp):
        import importlib
        importlib.import_module("imputers_v95").save_cache(cp)
    os.environ["IMP_CACHE"] = cp
    print(f"[submit] IMP_CACHE={cp}(setup 重複排除)", flush=True)


def main():
    _warm_imp_cache()
    v96pq, v97pq, v100pq = gen_v100_15banks()
    c15 = build_combo15(v96pq, v97pq, v100pq)
    fwd = gen_v104_fwd()
    cpq = merge_v104(c15, fwd)
    l5 = l5_v104_predict(cpq)
    meta = pq.read_table(cpq, columns=["id", "last_known_tvt"]); ids = meta.column("id").to_pandas().to_numpy()
    lk = meta.column("last_known_tvt").to_numpy(zero_copy_only=False).astype(np.float64)
    _bl = json.loads((V104DS / "blend_v104.json").read_text()) if (V104DS / "blend_v104.json").exists() else {}
    bwv = _bl.get("bw") or [0.0, 1.0]
    GG = _bl.get("gain_g"); GK = _bl.get("gain_k"); GS = _bl.get("gain_s"); GAIN = float(_bl.get("gain", 1.0))
    tcn = None
    if os.environ.get("V104_TCN", "1") == "1" and (V104DS / "tcn_v104_ckpt").exists():
        tcn = _tcn_v104_predict(cpq, ids)
    pred = l5 if tcn is None else (bwv[0] * tcn + bwv[1] * l5)
    if GG is not None and GK is not None and GS:      # ★後処理 expansion 較正(v100/v103 と同一式)
        pred = float(GG) * pred * (1.0 + float(GK) * np.abs(pred) / float(GS))
    elif GAIN != 1.0:
        pred = GAIN * pred
    fin = np.isfinite(pred)
    if tcn is not None:
        pred[~fin] = tcn[~fin]
    tvt = lk + pred
    sub = pd.DataFrame({"id": ids.astype(str), "tvt": tvt.astype(np.float64)})
    samp = pd.read_csv(DATA / "sample_submission.csv"); merged = samp[["id"]].merge(sub, on="id", how="left")
    out = WORK / "submission_v104.csv"; merged.to_csv(out, index=False)
    print(f"\n==== submission_v104.csv rows={len(merged)} finite%={100 * merged['tvt'].notna().mean():.2f} "
          f"(blend={bwv}, 後処理=({GG},{GK},{GS})|gain{GAIN}, PRUNE={os.environ.get('V104_PRUNE','')}) ====", flush=True)
    print(merged.head(6).to_string(index=False), flush=True)


# ---- TCN(SeqNet)推論(v103 と同一構造)----
class _ChanLN(nn.Module):
    def __init__(s, c): super().__init__(); s.ln = nn.LayerNorm(c)
    def forward(s, x): return s.ln(x.transpose(1, 2)).transpose(1, 2)
class _Block(nn.Module):
    def __init__(s, c, d, k, dr): super().__init__(); s.conv = nn.Conv1d(c, c, k, padding=d*(k-1)//2, dilation=d); s.ln = _ChanLN(c); s.pw = nn.Conv1d(c, c, 1); s.do = nn.Dropout(dr)
    def forward(s, x, m): h = torch.nn.functional.gelu(s.conv(x*m)); h = s.ln(h); h = s.do(s.pw(h)); return (x+h)*m
class _SelfAttn(nn.Module):
    def __init__(s, c, h, dr): super().__init__(); s.h = h; s.dk = c//h; s.qkv = nn.Linear(c, 3*c); s.proj = nn.Linear(c, c)
    def forward(s, x, km):
        B, L, C = x.shape; q, k, v = s.qkv(x).chunk(3, -1)
        q = q.view(B, L, s.h, s.dk).transpose(1, 2); k = k.view(B, L, s.h, s.dk).transpose(1, 2); v = v.view(B, L, s.h, s.dk).transpose(1, 2)
        o = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=km[:, None, None, :], dropout_p=0.0)
        return s.proj(o.transpose(1, 2).reshape(B, L, C))
class _PrefixEnc(nn.Module):
    def __init__(s, ci, ctx): super().__init__(); s.c1 = nn.Conv1d(ci, 64, 5, padding=2); s.c2 = nn.Conv1d(64, 64, 5, padding=8, dilation=4); s.fc = nn.Linear(128, ctx)
    def forward(s, x, m):
        h = torch.nn.functional.gelu(s.c1(x*m)); h = torch.nn.functional.gelu(s.c2(h))*m; ss = m.sum(-1)
        mean = h.sum(-1)/ss.clamp(min=1.0); mx = (h+(m-1.0)*1e4).amax(-1); mx = torch.where(ss > 0, mx, torch.zeros_like(mx))
        return s.fc(torch.cat([mean, mx], -1))*(ss > 0).float()
class _SeqNet(nn.Module):
    def __init__(s, kin, head, ch, ks, dils, dr, ua, ah, cd, uc=False, pc=4, ctx=64, nm=0):
        super().__init__(); s.head = head; s.use_ctx = uc; s.inp = nn.Conv1d(kin, ch, 1); s.indrop = nn.Dropout(cd)
        if uc: s.pre = _PrefixEnc(pc, ctx); s.ctxp = nn.Linear(ctx, ch)
        s.blocks = nn.ModuleList([_Block(ch, d, ks, dr) for d in dils]); s.use_attn = ua
        if ua: s.attn = _SelfAttn(ch, ah, dr); s.aln = nn.LayerNorm(ch)
        s.out = nn.Conv1d(ch, 5 if head == "mdn2" else 1, 1)
    def forward(s, x, m, px=None, pm=None):
        h = s.inp(s.indrop(x))*m
        if s.use_ctx and px is not None: h = (h+s.ctxp(s.pre(px, pm))[:, :, None])*m
        for b in s.blocks: h = b(h, m)
        if s.use_attn:
            q = s.aln(h.transpose(1, 2)); a = s.attn(q, (m[:, 0, :] > 0.5)); h = (h+a.transpose(1, 2))*m
        return s.out(h), None
def _mdn2_mean(o): pl = torch.sigmoid(o[:, 2]); return pl*o[:, 0]+(1-pl)*o[:, 1]


def _tcn_v104_predict(pq_path, ids_ref):
    pack = json.loads((V104DS / "submit_pack_v104.json").read_text())
    feats = pack["features"]; arch = pack["arch"]; t_mean = float(pack["t_mean"]); t_scale = float(pack["t_scale"])
    gmed = pack["prefix_stats"]["gmed"]; giqr = pack["prefix_stats"]["giqr"]
    use_flip = bool(pack["cfg"]["flip"]); use_prefix = bool(pack["cfg"]["prefix"])
    cols = list(dict.fromkeys(feats + ["well", "id", "last_known_tvt", "md_since"]))
    tb = pq.read_table(pq_path, columns=cols); N = tb.num_rows
    well = tb.column("well").to_pandas().to_numpy(); ids = tb.column("id").to_pandas().to_numpy()
    lk = tb.column("last_known_tvt").to_numpy(zero_copy_only=False).astype(np.float64)
    mds = tb.column("md_since").to_numpy(zero_copy_only=False).astype(np.float64)
    X = np.empty((N, len(feats)), np.float32)
    for j, c in enumerate(feats): X[:, j] = tb.column(c).to_numpy(zero_copy_only=False).astype(np.float32)
    del tb
    a2r = np.asarray(pack["abs2res"], bool); X[:, a2r] -= lk.astype(np.float32)[:, None]
    X -= np.asarray(pack["med"], np.float32); X /= np.asarray(pack["scale"], np.float32)
    np.nan_to_num(X, copy=False, nan=0.0, posinf=8.0, neginf=-8.0); np.clip(X, -8, 8, out=X)
    K_IN = X.shape[1]
    wid_order = pd.unique(well); wr = {w: np.where(well == w)[0] for w in wid_order}
    for w in wid_order: wr[w] = wr[w][np.argsort(mds[wr[w]], kind="stable")]
    PREF = {}
    if use_prefix:
        for w in wid_order:
            f = DATA / "test" / f"{w}__horizontal_well.csv"
            if not f.exists(): continue
            df = pd.read_csv(f, usecols=["MD", "Z", "GR", "TVT_input"]).dropna(subset=["MD", "Z", "GR", "TVT_input"]).tail(arch["pre_len"])
            if len(df) < 8: continue
            a = df.to_numpy(np.float64)
            gr_n = np.clip((a[:, 2]-gmed)/giqr, -8, 8); tvt_rel = (a[:, 3]-a[-1, 3])/t_scale; z_rel = (a[:, 1]-a[-1, 1])/50.0; md_rel = (a[:, 0]-a[-1, 0])/1000.0
            PREF[w] = np.nan_to_num(np.stack([gr_n, tvt_rel, z_rel, md_rel], 1), nan=0.0, posinf=8.0, neginf=-8.0).astype(np.float32)
    def new_model(): return _SeqNet(K_IN, "mdn2", arch["ch"], arch["ksize"], arch["dilations"], arch["drop"], arch["use_attn"], arch["attn_heads"], arch["ch_drop"], uc=use_prefix, pc=arch["pre_c"], ctx=arch["ctx_dim"], nm=0).to(DEVICE)
    @torch.no_grad()
    def pw(model, r, flip):
        rr = r[::-1] if flip else r; L = len(rr); x = torch.from_numpy(np.ascontiguousarray(X[rr][None])).to(DEVICE).transpose(1, 2); m = torch.ones((1, 1, L), device=DEVICE); px = pm = None
        if use_prefix:
            a = PREF.get(well[r[0]]); a = a if a is not None else np.zeros((1, arch["pre_c"]), np.float32)
            px = torch.from_numpy(np.ascontiguousarray(a[None])).to(DEVICE).transpose(1, 2); pm = torch.ones((1, 1, len(a)), device=DEVICE)
        o = model(x, m, px, pm)[0]; p = _mdn2_mean(o.float())[0].cpu().numpy(); out = np.empty(len(r), np.float64); out[:] = p[::-1] if flip else p; return out
    acc = np.zeros(N, np.float64); nck = 0
    for s in pack.get("seeds", [0, 1, 2, 3, 4]):
        for k in range(pack.get("n_splits", 5)):
            fp = V104DS / "tcn_v104_ckpt" / f"mdn2_s{s}_fold{k}.pt"
            if not fp.exists(): print(f"  WARN missing {fp.name}"); continue
            model = new_model(); model.load_state_dict(torch.load(fp, map_location=DEVICE)); model.eval()
            pr = np.zeros(N, np.float64)
            for w in wid_order:
                r = wr[w]; p = pw(model, r, False)
                if use_flip: p = 0.5*(p+pw(model, r, True))
                pr[r] = p
            acc += pr; nck += 1; del model
            if DEVICE == "cuda": torch.cuda.empty_cache()
    nn_pred = (acc/max(nck, 1))*t_scale+t_mean
    assert np.array_equal(ids, ids_ref), "TCN と combo の id 不一致"
    print(f"[TCN v104] {nck}ckpt pred[mean]={nn_pred.mean():.3f}", flush=True); return nn_pred


if __name__ == "__main__":
    main()
