# -*- coding: utf-8 -*-
"""v96+v97 特徴結合(combo, CV 5.8881)の提出オーケストレータ。
段取り: (1)v96 test生成 (2)v97 test生成(nn_emission込) (3)id順hstackで combo test.parquet
        (4)TCN_combo 推論 (5)L5_combo 推論 (6)blend(TCN 0.476 + L5 0.524) -> submission.
env(ノートが与える): ROGII_DATA / ROGII_COMBO(combo_dataset root) / ROGII_SCRIPTS / ROGII_PY / ROGII_WORK
GBMは重み0なので不要。self-contained(v95+v97モジュールを併用)。"""
import os, sys, io, json, glob, pickle, subprocess
try:
    sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False); sys.stderr = io.open(2, "w", encoding="utf-8", closefd=False)
except Exception:
    pass
from pathlib import Path
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq
import torch, torch.nn as nn

PROJ = Path(os.environ.get("ROGII_PROJ", r"C:\Users\kosaka256\Documents\rogii_claude"))
SCR = Path(os.environ.get("ROGII_SCRIPTS", str(PROJ / "v97")))         # 全スクリプト置き場
DATA = Path(os.environ.get("ROGII_DATA", str(PROJ / "rogii-wellbore-geology-prediction")))
COMBO = Path(os.environ.get("ROGII_COMBO", str(PROJ / "v97" / "combo_dataset")))
WORK = Path(os.environ.get("ROGII_WORK", str(COMBO)))
PY = os.environ.get("ROGII_PY", str(PROJ / ".venv" / "Scripts" / "python.exe"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
V96A = WORK / "v95"; V97A = WORK / "v97"; CB = COMBO      # ★v98: v96=v95パイプライン / combo成果物はartifacts_v98直下
(V96A / "data").mkdir(parents=True, exist_ok=True); (V97A / "data").mkdir(parents=True, exist_ok=True)
_bwf = COMBO / "blend.json"                               # 学習後に導出した blend重み(無ければ均等)
BW = json.loads(_bwf.read_text())["bw"] if _bwf.exists() else [0.5, 0.5]   # TCN_combo, L5_combo


def _run(script, env_extra, tag):
    env = dict(os.environ); env.update(env_extra); env["PYTHONUNBUFFERED"] = "1"
    print(f"\n==== {tag}: {script} ====", flush=True)
    r = subprocess.run([PY, str(SCR / script)], env=env)
    if r.returncode != 0:
        raise SystemExit(f"{tag} 失敗 (code={r.returncode})")


def gen_v96():
    """v96 test.parquet: grfree錨(v95) -> create_v95 SPLIT=test (nn_emission OFF, 6バンク)。"""
    V96PQ = V96A / "data" / "test.parquet"
    if V96PQ.exists() and os.environ.get("SKIP_GEN", "0") == "1":
        print("[skip] v96 test.parquet 既存"); return V96PQ
    base = dict(ROGII_ART95=str(V96A), ROGII_ART95B=str(V96A), ROGII_SCRIPTS=str(SCR), ROGII_DATA=str(DATA),
                ROGII_PY=PY, GRFREE_GEN=str(SCR / "gen_grfree_anchor.py"),
                GRFREE_FOLD_ART=str(CB / "grfree_fold_art.pkl"))   # ★事前学習済み錨(FOLD_ART)=Kaggleで再学習しない
    if (V96A / "grfree_anchor_test.pkl").exists() and os.environ.get("FORCE_ANCHOR", "0") != "1":
        print("[v96錨] 既存を流用(FORCE_ANCHOR=1で再生成)", flush=True)
    else:
        _run("gen_grfree_test_anchor_v95.py", base, "v96錨")
    _run("create_v95.py", dict(base, SPLIT="test", V93_ANCHOR_PKL=str(V96A / "grfree_anchor_test.pkl"),
                               OUT_PARQUET=str(V96PQ)), "v96 create")
    return V96PQ


def gen_v97():
    """v97 test.parquet: grfree錨(v97) -> nn_emission test-sim -> create_v97 SPLIT=test (4バンク)。"""
    V97PQ = V97A / "data" / "test.parquet"
    if V97PQ.exists() and os.environ.get("SKIP_GEN", "0") == "1":
        print("[skip] v97 test.parquet 既存"); return V97PQ
    base = dict(ROGII_ART97=str(V97A), ROGII_ART97B=str(V97A), ROGII_SCRIPTS=str(SCR), ROGII_DATA=str(DATA),
                ROGII_PY=PY, GRFREE_GEN=str(SCR / "gen_grfree_anchor.py"))
    anc = str(V97A / "grfree_anchor_test.pkl"); sim = str(V97A / "sim_grfree_test_v97.pkl")
    v96anc = V96A / "grfree_anchor_test.pkl"
    if v96anc.exists():                                    # ★v96で生成した錨(grfree sig9.0 同一)を流用=GRU再学習回避
        import shutil; shutil.copyfile(v96anc, anc); print("[v97錨] v96の錨を流用(GRU再学習スキップ)", flush=True)
    else:
        _run("gen_grfree_test_anchor_v97.py", base, "v97錨")
    _run("nn_emission_v97.py", dict(base, NN_SPLIT="test", V93_ANCHOR_PKL=anc), "v97 nn-emission")
    _run("create_v97.py", dict(base, SPLIT="test", V93_ANCHOR_PKL=anc, SIM_PKL=sim,
                               OUT_PARQUET=str(V97PQ)), "v97 create")
    return V97PQ


def build_combo(v96pq, v97pq):
    """id順で v96全 + v97差分(v97_接頭辞) を hstack -> combo test.parquet。"""
    print("\n==== combo test.parquet 構築 ====", flush=True)
    diff = json.loads((CB / "v97_diff_cols.json").read_text())
    id96 = pq.read_table(v96pq, columns=["id"]).to_pandas()["id"].astype(str).to_numpy()
    id97 = pq.read_table(v97pq, columns=["id"]).to_pandas()["id"].astype(str).to_numpy()
    assert np.array_equal(id96, id97), "v96/v97 の id 並びが不一致(test生成の坑井順が違う)"
    out = WORK / "data" / "test_combo.parquet"; (WORK / "data").mkdir(parents=True, exist_ok=True)
    pf96 = pq.ParquetFile(v96pq); pf97 = pq.ParquetFile(v97pq); writer = None
    for b96, b97 in zip(pf96.iter_batches(batch_size=300000), pf97.iter_batches(batch_size=300000, columns=diff)):
        cols = list(b96.columns); names = list(b96.schema.names)
        for j, c in enumerate(diff):
            cols.append(b97.column(j)); names.append("v97_" + c)
        tbl = pa.Table.from_arrays(cols, names=names)
        if writer is None:
            writer = pq.ParquetWriter(out, tbl.schema, compression="zstd")
        writer.write_table(tbl)
    writer.close()
    print(f"[combo] {out.name} 完成 列={len(pq.read_schema(out).names)}", flush=True)
    return out


# ---------------- TCN combo 推論(submit_pack_nn_fcombo + combo_fcombo ckpt) ----------------
class ChanLN(nn.Module):
    def __init__(s, c): super().__init__(); s.ln = nn.LayerNorm(c)
    def forward(s, x): return s.ln(x.transpose(1, 2)).transpose(1, 2)
class Block(nn.Module):
    def __init__(s, c, d, k, dr): super().__init__(); s.conv = nn.Conv1d(c, c, k, padding=d*(k-1)//2, dilation=d); s.ln = ChanLN(c); s.pw = nn.Conv1d(c, c, 1); s.do = nn.Dropout(dr)
    def forward(s, x, m): h = torch.nn.functional.gelu(s.conv(x*m)); h = s.ln(h); h = s.do(s.pw(h)); return (x+h)*m
class SelfAttn(nn.Module):
    def __init__(s, c, h, dr): super().__init__(); s.h = h; s.dk = c//h; s.qkv = nn.Linear(c, 3*c); s.proj = nn.Linear(c, c)
    def forward(s, x, km):
        B, L, C = x.shape; q, k, v = s.qkv(x).chunk(3, -1)
        q = q.view(B, L, s.h, s.dk).transpose(1, 2); k = k.view(B, L, s.h, s.dk).transpose(1, 2); v = v.view(B, L, s.h, s.dk).transpose(1, 2)
        o = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=km[:, None, None, :], dropout_p=0.0)
        return s.proj(o.transpose(1, 2).reshape(B, L, C))
class PrefixEnc(nn.Module):
    def __init__(s, ci, ctx): super().__init__(); s.c1 = nn.Conv1d(ci, 64, 5, padding=2); s.c2 = nn.Conv1d(64, 64, 5, padding=8, dilation=4); s.fc = nn.Linear(128, ctx)
    def forward(s, x, m):
        h = torch.nn.functional.gelu(s.c1(x*m)); h = torch.nn.functional.gelu(s.c2(h))*m; ss = m.sum(-1)
        mean = h.sum(-1)/ss.clamp(min=1.0); mx = (h+(m-1.0)*1e4).amax(-1); mx = torch.where(ss > 0, mx, torch.zeros_like(mx))
        return s.fc(torch.cat([mean, mx], -1))*(ss > 0).float()
class SeqNet(nn.Module):
    def __init__(s, kin, head, ch, ks, dils, dr, ua, ah, cd, uc=False, pc=4, ctx=64, nm=0):
        super().__init__(); s.head = head; s.use_ctx = uc; s.inp = nn.Conv1d(kin, ch, 1); s.indrop = nn.Dropout(cd)
        if uc: s.pre = PrefixEnc(pc, ctx); s.ctxp = nn.Linear(ctx, ch)
        s.blocks = nn.ModuleList([Block(ch, d, ks, dr) for d in dils]); s.use_attn = ua
        if ua: s.attn = SelfAttn(ch, ah, dr); s.aln = nn.LayerNorm(ch)
        s.out = nn.Conv1d(ch, 5 if head == "mdn2" else 1, 1)
    def forward(s, x, m, px=None, pm=None):
        h = s.inp(s.indrop(x))*m
        if s.use_ctx and px is not None: h = (h+s.ctxp(s.pre(px, pm))[:, :, None])*m
        for b in s.blocks: h = b(h, m)
        if s.use_attn:
            q = s.aln(h.transpose(1, 2)); a = s.attn(q, (m[:, 0, :] > 0.5)); h = (h+a.transpose(1, 2))*m
        return s.out(h), None
def _mdn2_mean(o): pl = torch.sigmoid(o[:, 2]); return pl*o[:, 0]+(1-pl)*o[:, 1]


def tcn_combo_predict(pq_path):
    pack = json.loads((CB / "submit_pack_nn_fcombo.json").read_text())
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
    med = np.asarray(pack["med"], np.float32); scale = np.asarray(pack["scale"], np.float32)
    X -= med; X /= scale; np.nan_to_num(X, copy=False, nan=0.0, posinf=8.0, neginf=-8.0); np.clip(X, -8, 8, out=X)
    K_IN = X.shape[1]; print(f"[TCN combo] rows={N} K={K_IN}", flush=True)
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
    def new_model(): return SeqNet(K_IN, "mdn2", arch["ch"], arch["ksize"], arch["dilations"], arch["drop"], arch["use_attn"], arch["attn_heads"], arch["ch_drop"], uc=use_prefix, pc=arch["pre_c"], ctx=arch["ctx_dim"], nm=0).to(DEVICE)
    @torch.no_grad()
    def pw(model, r, flip):
        rr = r[::-1] if flip else r; L = len(rr); x = torch.from_numpy(np.ascontiguousarray(X[rr][None])).to(DEVICE).transpose(1, 2); m = torch.ones((1, 1, L), device=DEVICE); px = pm = None
        if use_prefix:
            a = PREF.get(well[r[0]]); a = a if a is not None else np.zeros((1, arch["pre_c"]), np.float32)
            px = torch.from_numpy(np.ascontiguousarray(a[None])).to(DEVICE).transpose(1, 2); pm = torch.ones((1, 1, len(a)), device=DEVICE)
        o = model(x, m, px, pm)[0]; p = _mdn2_mean(o.float())[0].cpu().numpy(); out = np.empty(len(r), np.float64); out[:] = p[::-1] if flip else p; return out
    acc = np.zeros(N, np.float64); nck = 0
    for s in pack.get("seeds", [0, 1, 2]):
        for k in range(pack.get("n_splits", 5)):
            fp = CB / "nn" / f"combo_fcombo_mdn2_s{s}_fold{k}.pt"
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
    print(f"[TCN combo] {nck}ckpt pred[mean]={nn_pred.mean():.3f}", flush=True)
    return ids, lk, nn_pred


# ---------------- L5 combo 推論(combo seqfeat + l5_fcombo ckpt) ----------------
def l5_combo_predict(pq_path):
    SF = COMBO / "seqfeat_test_combo.pkl"
    ED = ",".join(["pf_nbr_gr5_delta_%d" % k for k in (1, 2, 3)] + ["pf_self_graft_delta_%d" % k for k in (1, 2, 3)]
                  + ["v97_pf_%s_delta_%d" % (r, k) for r in ("ancc", "self", "nbr", "nbr_gr5", "self_graft") for k in (1, 2, 3, 4)])
    ES = ",".join(["pf_nbr_gr5_std_%d" % k for k in (1, 2, 3)] + ["pf_self_graft_std_%d" % k for k in (1, 2, 3)]
                  + ["v97_pf_%s_std_%d" % (r, k) for r in ("ancc", "self", "nbr", "nbr_gr5", "self_graft") for k in (1, 2, 3, 4)])
    _run("build_seqfeat_v97.py", dict(SF_IN=str(pq_path), SF_OUT=str(SF), SF_NO_TARGET="1",
                                      CANDFEAT="1", NF_MAX="2", V97_NBANK="6", V93_INPUT="fwd",
                                      EXTRA_DELTA=ED, EXTRA_STD=ES, ROGII_SCRIPTS=str(SCR)), "combo seqfeat")
    d = pickle.loads(SF.read_bytes()); wells = list(d.keys()); w0 = d[wells[0]]
    K = w0["D"].shape[1]; CDIM = w0["C"].shape[1]; NF = w0["DX"].shape[2]
    N = int(max(w["rows"].max() for w in d.values())) + 1
    acc = np.zeros(N, np.float64); cnt = np.zeros(N, np.float64)
    for s in range(int(os.environ.get("NN_SEEDS", "5"))):   # ★v98: 5seed
        for k in range(5):
            fp = CB / "l5_fcombo_ckpt" / f"l5_s{s}_f{k}_model.pt"
            if not fp.exists(): print(f"  WARN missing {fp.name}"); continue
            ck = torch.load(fp, map_location=DEVICE)
            net = _mk_l5(K, CDIM, NF, ck.get("dim", 64) if isinstance(ck, dict) else 64)
            net.load_state_dict(ck["state"] if isinstance(ck, dict) and "state" in ck else ck); net.eval()
            with torch.no_grad():
                for w in wells:
                    f = d[w]; T = f["D"].shape[0]
                    D = torch.from_numpy(f["D"])[None].to(DEVICE); S = torch.from_numpy(f["S"])[None].to(DEVICE)
                    C = torch.from_numpy(f["C"])[None].to(DEVICE); m = torch.ones((1, T), device=DEVICE); DX = torch.from_numpy(f["DX"])[None].to(DEVICE)
                    o = net(D, S, C, m, DX)[0].cpu().numpy().astype(np.float64); pr = o[f["bid"]]
                    acc[f["rows"]] += pr; cnt[f["rows"]] += 1
            del net
            if DEVICE == "cuda": torch.cuda.empty_cache()
    l5 = np.full(N, np.nan); ok = cnt > 0; l5[ok] = acc[ok]/cnt[ok]
    print(f"[L5 combo] cover%={100*ok.mean():.1f}", flush=True)
    return l5


def _mk_l5(k, cdim, nf, dim):
    """train_l456_v97 の L5 と同一構造(推論用)。"""
    import torch.nn as nn
    class L5(nn.Module):
        def __init__(s):
            super().__init__(); s.k = k
            s.enc = nn.Sequential(nn.Conv1d(2+nf, dim, 9, padding=4), nn.GELU(), nn.Conv1d(dim, dim, 9, padding=16, dilation=4), nn.GELU(), nn.Conv1d(dim, dim, 9, padding=64, dilation=16))
            s.emb = nn.Embedding(k, dim); s.qry = nn.Sequential(nn.Linear(cdim+4, dim), nn.GELU(), nn.Linear(dim, dim)); s.res = nn.Linear(dim, 1)
        def forward(s, D, S, C, m, DX=None):
            B, T, kk = D.shape; med = D.median(2, keepdim=True).values
            x = torch.stack([D-med, torch.log(S)]+([DX[..., i] for i in range(DX.shape[-1])] if DX is not None else []), 2)
            x = x.permute(0, 3, 2, 1).reshape(B*kk, 2+(DX.shape[-1] if DX is not None else 0), T)
            hc = s.enc(x).reshape(B, kk, -1, T).permute(0, 3, 1, 2)+s.emb.weight[None, None]
            stats = torch.cat([C, med/10.0, D.std(2, keepdim=True), (D-med).abs().lt(2).float().mean(2, keepdim=True), torch.zeros_like(med)], -1)
            q = s.qry(stats); att = (hc @ q[..., None])[..., 0]/(hc.shape[-1]**0.5); w = att.softmax(-1)
            return (w*D).sum(-1)+2.0*torch.tanh(s.res(q)[..., 0])
    return L5().to(DEVICE)


def main():
    v96pq = gen_v96(); v97pq = gen_v97()
    cpq = build_combo(v96pq, v97pq)
    ids, lk, tcn = tcn_combo_predict(cpq)
    l5 = l5_combo_predict(cpq)
    # id基準は combo parquet
    meta = pq.read_table(cpq, columns=["id", "last_known_tvt"]); ids2 = meta.column("id").to_pandas().to_numpy()
    lk2 = meta.column("last_known_tvt").to_numpy(zero_copy_only=False).astype(np.float64)
    assert np.array_equal(ids2, ids), "TCN と combo の id 不一致"
    pred = BW[0]*tcn + BW[1]*l5
    fin = np.isfinite(pred); pred[~fin] = tcn[~fin]        # L5欠損は TCN 単独
    tvt = lk2 + pred
    sub = pd.DataFrame({"id": ids2, "tvt": tvt.astype(np.float64)})
    samp = pd.read_csv(DATA / "sample_submission.csv"); merged = samp[["id"]].merge(sub, on="id", how="left")
    out = COMBO / "submission_combo.csv"; merged.to_csv(out, index=False)
    print(f"\n==== submission_combo.csv ==== rows={len(merged)} finite%={100*merged['tvt'].notna().mean():.2f}", flush=True)
    print(merged.head(6).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
