# _build_train_nb.py に v57 教訓(chainトークン/hard重み/surf文脈)を適用するパッチ(1回きり)
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
p = r"c:/Users/kosaka256/Documents/rogii_claude/v54/_build_train_nb.py"
s = open(p, encoding="utf-8").read()

old1 = '''Z = {w: np.concatenate(CAND[w], 0) for w in wells}    # (NC,T) fp16
WWc = {w: np.concatenate(CWW[w], 0).astype(np.float32) for w in wells}
del CAND, CWW; gc.collect()
print(f"wells={len(wells)} streams={NS} seeds/str={SPS} -> 候補{NC}/行  ({time.time()-t0:.0f}s)")'''
new1 = '''Z = {w: np.concatenate(CAND[w], 0) for w in wells}    # (NC,T) fp16
WWc = {w: np.concatenate(CWW[w], 0).astype(np.float32) for w in wells}
del CAND, CWW; gc.collect()
# chainアンカートークン(v57教訓): v52 GBM OOF(pp後)を絶対値化しeval行CSV順に整列
import pyarrow.parquet as pq
ART52 = P0/"v52"/"artifacts_v52"
tbq = pq.read_table(ART52/"data"/"train.parquet", columns=["id","well","md_since","pf_ancc","last_known_tvt"])
_ids = tbq.column("id").to_pandas().astype(str); _g = tbq.column("well").to_pandas().astype(str).to_numpy()
_mds = np.maximum(tbq.column("md_since").to_numpy(zero_copy_only=False).astype(np.float64), 0.0)
_pfa = tbq.column("pf_ancc").to_numpy(zero_copy_only=False).astype(np.float64)
_lk = tbq.column("last_known_tvt").to_numpy(zero_copy_only=False).astype(np.float64)
del tbq
_oof = np.load(ART52/"gbm_v41_oof.npy").astype(np.float64)
_pp = json.loads((ART52/"pp_params.json").read_text())
_cres = (_oof*(1-_pp["w_pf"]) + (_pfa-_lk)*_pp["w_pf"]) * (1-np.exp(-_mds/_pp["tau"])) * _pp["alpha"]
_rn = _ids.str.rsplit("_", n=1, expand=True)[1].astype(int).to_numpy()
CH = {}
for wn in wells:
    ii = np.where(_g == wn)[0]
    o = np.argsort(_rn[ii], kind="stable")
    ca = (_cres + _lk)[ii[o]]
    CH[wn] = ca[:T_[wn]].astype(np.float32) if len(ca) >= T_[wn] else np.pad(ca, (0, T_[wn]-len(ca)), mode="edge").astype(np.float32)
del _ids, _g, _mds, _pfa, _lk, _oof, _cres, _rn; gc.collect()
# surf文脈(v56): eval行CSV順 [tvt_surf, dist, slope, fmad]
_sn = np.load(P0/"v56"/"artifacts_v56"/"lowo_rows.npz")
SU = {w: (_sn[w][:T_[w]].astype(np.float32) if w in _sn.files else np.zeros((T_[w],4),np.float32)) for w in wells}
# hard重み用: 候補のchainオフセット散布
SPREAD = {w: float(np.std(np.median(Z[w].astype(np.float32) - CH[w][None, :Z[w].shape[1]], 1))) for w in wells}
HARDW = float(os.environ.get("HARDW", "0"))
print(f"wells={len(wells)} streams={NS} seeds/str={SPS} -> 候補{NC}/行 (+chainトークン)  ({time.time()-t0:.0f}s)")'''
assert old1 in s, "old1"
s = s.replace(old1, new1)

old3a = '''    def __init__(self, nb=len(BANKS), nf=len(FLAVORS), dim=DIM):'''
new3a = '''    def __init__(self, nb=len(BANKS) + 1, nf=len(FLAVORS) + 1, dim=DIM):'''
assert old3a in s, "old3a"
s = s.replace(old3a, new3a)

old3b = '''            nn.Conv1d(5, dim, 9, padding=4), nn.GELU(),'''
new3b = '''            nn.Conv1d(8, dim, 9, padding=4), nn.GELU(),'''
assert old3b in s, "old3b"
s = s.replace(old3b, new3b)

old3c = '''                             (ww[:, :, None] * z).sum(1) - med[:, 0]], 1)'''
new3c = '''                             (ww[:, :, None] * z).sum(1) - med[:, 0],
                             xc[:, 0] - med[:, 0], xc[:, 1] - med[:, 0], xc[:, 2]], 1)'''
assert old3c in s, "old3c"
s = s.replace(old3c, new3c)

old3d = '''    def forward(self, z, ww, bid, fid):'''
new3d = '''    def forward(self, z, ww, bid, fid, xc):'''
assert old3d in s, "old3d"
s = s.replace(old3d, new3d)

old3e = '''def make_batch(ws, stride=1):
    Tm = max((T_[w] + stride - 1) // stride for w in ws); B = len(ws)
    z = torch.zeros(B, NC, Tm); ww = torch.zeros(B, NC)
    yv = torch.full((B, Tm), float("nan")); m = torch.zeros(B, Tm)
    for i, w in enumerate(ws):
        zz = torch.from_numpy(Z[w][:, ::stride].astype(np.float32)); t = zz.shape[1]
        z[i, :, :t] = zz; ww[i] = torch.from_numpy(WWc[w])
        yv[i, :t] = torch.from_numpy(np.asarray(Y[w][::stride], np.float32)); m[i, :t] = 1
    bid = torch.from_numpy(BID).long().to(DEV); fid = torch.from_numpy(FID).long().to(DEV)
    return z.to(DEV), ww.to(DEV), yv.to(DEV), m.to(DEV), bid, fid'''
new3e = '''NCT = NC + 1                                             # +chainトークン
BID_T = np.concatenate([BID, [len(BANKS)]]); FID_T = np.concatenate([FID, [len(FLAVORS)]])
def make_batch(ws, stride=1):
    Tm = max((T_[w] + stride - 1) // stride for w in ws); B = len(ws)
    z = torch.zeros(B, NCT, Tm); ww = torch.zeros(B, NCT)
    yv = torch.full((B, Tm), float("nan")); m = torch.zeros(B, Tm)
    xc = torch.zeros(B, 3, Tm)
    for i, w in enumerate(ws):
        zz = torch.from_numpy(Z[w][:, ::stride].astype(np.float32)); t = zz.shape[1]
        z[i, :NC, :t] = zz
        ch = torch.from_numpy(CH[w][::stride].astype(np.float32))[:Tm]
        z[i, NC, :len(ch)] = ch
        ww[i, :NC] = torch.from_numpy(WWc[w]); ww[i, NC] = 1.0 / max(NS, 1)
        yv[i, :t] = torch.from_numpy(np.asarray(Y[w][::stride], np.float32)); m[i, :t] = 1
        su = SU[w][::stride]
        xc[i, 0, :len(ch)] = ch
        xc[i, 1, :len(su)] = torch.from_numpy(su[:, 0])
        xc[i, 2, :len(su)] = torch.from_numpy(np.log1p(np.maximum(su[:, 1], 0)) / 8.0)
    bid = torch.from_numpy(BID_T).long().to(DEV); fid = torch.from_numpy(FID_T).long().to(DEV)
    return z.to(DEV), ww.to(DEV), yv.to(DEV), m.to(DEV), bid, fid, xc.to(DEV)'''
assert old3e in s, "old3e"
s = s.replace(old3e, new3e)

old4a = '''                z, ww, yv, m, bid, fid = make_batch(ws, ROWSTRIDE)
                out, _ = net(z, ww, bid, fid)
                ok = m.bool() & torch.isfinite(yv)
                tot += float(F.mse_loss(out[ok], yv[ok])) * int(ok.sum()); n += int(ok.sum())'''
new4a = '''                z, ww, yv, m, bid, fid, xc = make_batch(ws, ROWSTRIDE)
                out, _ = net(z, ww, bid, fid, xc)
                ok = m.bool() & torch.isfinite(yv)
                tot += float(F.mse_loss(out[ok], yv[ok])) * int(ok.sum()); n += int(ok.sum())'''
assert old4a in s, "old4a"
s = s.replace(old4a, new4a)

old4b = '''            try:
                z, ww, yv, m, bid, fid = make_batch(ws, ROWSTRIDE)
                out, _ = net(z, ww, bid, fid)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache(); continue
            ok = m.bool() & torch.isfinite(yv)
            if not ok.any(): continue
            loss = F.mse_loss(out[ok], yv[ok])'''
new4b = '''            try:
                z, ww, yv, m, bid, fid, xc = make_batch(ws, ROWSTRIDE)
                out, _ = net(z, ww, bid, fid, xc)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache(); continue
            ok = m.bool() & torch.isfinite(yv)
            if not ok.any(): continue
            if HARDW > 0:
                wgt = torch.tensor([1.0 + HARDW * float(SPREAD[w] > 4.0) for w in ws], device=DEV)[:, None].expand_as(yv)
                se = (out - torch.nan_to_num(yv)) ** 2
                loss = (se[ok] * wgt[ok]).sum() / wgt[ok].sum()
            else:
                loss = F.mse_loss(out[ok], yv[ok])'''
assert old4b in s, "old4b"
s = s.replace(old4b, new4b)

old4c = '''        for w in val:
            z, ww, yv, m, bid, fid = make_batch([w], 1)
            out, _ = net(z, ww, bid, fid)
            dh[w] = out[0, :T_[w]].cpu().numpy()'''
new4c = '''        for w in val:
            z, ww, yv, m, bid, fid, xc = make_batch([w], 1)
            out, _ = net(z, ww, bid, fid, xc)
            dh[w] = out[0, :T_[w]].cpu().numpy()'''
assert old4c in s, "old4c"
s = s.replace(old4c, new4c)

open(p, "w", encoding="utf-8").write(s)
print("builder patched")
