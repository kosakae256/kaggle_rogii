# -*- coding: utf-8 -*-
"""X_FP16 列融合ロードの数値検証: 元のfloat32正規化 vs 融合float16 を実データ列で突き合わせ。
差は float16 丸め(標準化後[-8,8]で~0.004)のみのはず。abs2res(残差化列)判定も一致確認。"""
import os, json, warnings
import numpy as np, pyarrow.parquet as pq

PQ = r"C:\Users\kosaka256\Documents\rogii_claude\v100\artifacts_v100\data\train_combo.parquet"
FEATS = json.load(open(r"C:\Users\kosaka256\Documents\rogii_claude\v100\artifacts_v100\features_l1_fcombo.json"))
META = ["well", "target", "last_known_tvt", "md_since", "pf_ancc"]

# 検証は先頭200列で(全1207は時間かかるだけ、ロジックは列独立なので十分)
feats = FEATS[:200]

meta = pq.read_table(PQ, columns=META)
lk = meta.column("last_known_tvt").to_numpy(zero_copy_only=False).astype(np.float64)
N = len(lk)
samp = np.arange(0, N, max(1, N // 120000))
lk32 = lk.astype(np.float32); lks = lk[samp]
print(f"N={N}  samp={len(samp)}  検証列={len(feats)}")

# ---- (A) 元パス: f32でロード→残差化→標準化→clip ----
def path_A():
    Xa = np.empty((N, len(feats)), np.float32)
    t = pq.read_table(PQ, columns=feats)
    for j, c in enumerate(feats):
        Xa[:, j] = t.column(j).to_numpy(zero_copy_only=False).astype(np.float32)
    abs2 = []
    for j in range(Xa.shape[1]):
        v = Xa[samp, j].astype(np.float64); m = np.isfinite(v)
        ok = m.sum() > 100 and np.nanstd(v[m]) > 1e-9
        c_ = float(np.corrcoef(v[m], lks[m])[0, 1]) if ok else 0.0
        a = bool(c_ > 0.98); abs2.append(a)
        if a:
            Xa[:, j] = Xa[:, j] - lk32
    Xs = Xa[samp]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        med = np.nanmedian(Xs, axis=0).astype(np.float32)
        q75 = np.nanpercentile(Xs, 75, axis=0).astype(np.float32)
        q25 = np.nanpercentile(Xs, 25, axis=0).astype(np.float32)
    scl = np.maximum((q75 - q25) / 1.349, 1e-6).astype(np.float32)
    Xa -= med; Xa /= scl
    np.nan_to_num(Xa, copy=False, nan=0.0, posinf=8.0, neginf=-8.0)
    np.clip(Xa, -8.0, 8.0, out=Xa)
    return Xa, abs2

# ---- (B) 融合パス: 列ごと f32計算→f16格納(train_nn_v98の_load_fused_fp16と同一ロジック) ----
def path_B():
    Xb = np.empty((N, len(feats)), np.float16)
    abs2 = []; CHUNK = 64
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for s in range(0, len(feats), CHUNK):
            cols = feats[s:s + CHUNK]
            t = pq.read_table(PQ, columns=cols)
            for jj, c in enumerate(cols):
                col = t.column(jj).to_numpy(zero_copy_only=False).astype(np.float32)
                vs = col[samp].astype(np.float64); m = np.isfinite(vs)
                ok = m.sum() > 100 and np.nanstd(vs[m]) > 1e-9
                c_ = float(np.corrcoef(vs[m], lks[m])[0, 1]) if ok else 0.0
                a = bool(c_ > 0.98); abs2.append(a)
                if a:
                    col -= lk32
                cs = col[samp]
                med = np.nanmedian(cs); q75 = np.nanpercentile(cs, 75); q25 = np.nanpercentile(cs, 25)
                scl = max((q75 - q25) / 1.349, 1e-6)
                col -= med; col /= scl
                np.nan_to_num(col, copy=False, nan=0.0, posinf=8.0, neginf=-8.0)
                np.clip(col, -8.0, 8.0, out=col)
                Xb[:, s + jj] = col.astype(np.float16)
    return Xb, abs2

Xa, aA = path_A()
Xb, aB = path_B()

print("abs2res 一致:", aA == aB, f"(A={sum(aA)}列 / B={sum(aB)}列 残差化)")
Xb32 = Xb.astype(np.float32)
d = np.abs(Xa - Xb32)
print(f"max|A-B| = {d.max():.5f}   mean|A-B| = {d.mean():.6f}")
print(f"float16 分解能上限(値8付近)= {np.spacing(np.float16(8)):.5f}")
# 相関(各列がほぼ同一値か)
worst = np.argsort(-d.max(axis=0))[:5]
for j in worst:
    print(f"  col[{j}] {feats[j][:40]:40s} max|d|={d[:,j].max():.5f}  A範囲[{Xa[:,j].min():.2f},{Xa[:,j].max():.2f}]")
ok = (aA == aB) and d.max() < 0.02
print("\n✅ 検証OK(差はfloat16丸めのみ)" if ok else "\n❌ 差が大きい=要調査")
