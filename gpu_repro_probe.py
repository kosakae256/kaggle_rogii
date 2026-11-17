# -*- coding: utf-8 -*-
"""
GPU再現性プローブ(最小・データ不要)
====================================
同じコードを RTX5070(ローカル) と T4(Kaggle) で走らせ、**出力値が一致するか**を見るだけ。
PF と同種の演算(seeded randn → exp → cumsum → searchsorted → gather → 加重和 の T ループ)を回し、
高精度チェックサムを表示する。両環境で数値を比べれば「同一seedで同じ結果が出るか」が分かる。

使い方: 5070 と Kaggle(T4) の両方でこのファイルを実行し、出力の acc / sum / sample を比較する。
実行: python gpu_repro_probe.py
"""

import torch

DEV = "cuda" if torch.cuda.is_available() else "cpu"
print("torch:", torch.__version__, "| cuda:", torch.version.cuda, "| device:", DEV)
if DEV == "cuda":
    print("gpu:", torch.cuda.get_device_name(0))


def probe(seed, B=256, N=500, T=150):
    """PF 縮小版: 伝播→尤度→正規化→系統リサンプリング→加重和。決定論的に checksum を返す。"""
    g = torch.Generator(device=DEV); g.manual_seed(int(seed))
    rn = lambda: torch.randn((B, N), generator=g, device=DEV)
    ar = (torch.arange(N, device=DEV) + 0.5) / N
    pos = rn()
    w = torch.full((B, N), 1.0 / N, device=DEV)
    acc = torch.zeros((), device=DEV, dtype=torch.float64)
    for i in range(T):
        pos = 0.999 * pos + 0.05 * rn()
        d = (pos * 0.1) ** 2
        lk = torch.exp(-0.5 * d).clamp_min(1e-30)
        w = w * lk
        w = w / w.sum(1, keepdim=True).clamp_min(1e-30)
        cumw = torch.cumsum(w, 1)
        idx = torch.searchsorted(cumw, ar.expand(B, N).contiguous()).clamp(0, N - 1)
        pos = torch.gather(pos, 1, idx) + 0.01 * rn()
        w = torch.full((B, N), 1.0 / N, device=DEV)
        acc = acc + (w * pos).sum().double()
    return pos, acc


def summ(pos, acc):
    return (float(acc),
            float(pos.double().sum()),
            float(pos[0, 0]),
            float(pos[-1, -1]))


print("\n=== seed スイープ(同一seedなら両GPUで一致するはず) ===")
first = None
for s in [0, 1, 2]:
    pos, acc = probe(s)
    a, ssum, p00, pll = summ(pos, acc)
    if s == 0:
        first = (pos.clone(), acc.clone())
    print(f"seed={s}: acc={a:.10f}  pos.sum={ssum:.8f}  pos[0,0]={p00:.10f}  pos[-1,-1]={pll:.10f}")

print("\n=== 同一seed(0)を再実行して同一マシン決定性を確認 ===")
pos2, acc2 = probe(0)
maxdiff = float((first[0] - pos2).abs().max())
print(f"acc差={abs(float(first[1]) - float(acc2)):.3e}  pos最大絶対差={maxdiff:.3e}")
print("→ 0なら同一マシンで決定的。両GPUで seed=0 の acc/sum/sample が一致すればクロスGPUでも再現。")
