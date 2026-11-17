# v99 候補（次タスク）— full平滑（系譜ベース全区間スムーサ）で fixed-lag=192 を置換

> ★v98 pipeline 完了後に着手する次タスク。**速くて（lag非依存）精度も上がる可能性**（pf_1 で実証済）。

## 動機（なぜやるか）
- 現行の fixed-lag=192 平滑はコストが **O(r·T·B·L·N)＝Lに線形**。lag192 は lag0(前方フィルタ) の **約8〜13倍**（実測: normal v96 create ≈8h）。原因はリサンプル毎に `buf[B,L,N]` を祖先で並べ替える所（`gather(buf,dim=2)`）。
- **full平滑（全区間・系譜ベース）は O(T·B·N)＝L完全非依存**（前方フィルタ＝lag0 と同オーダー、実質 filter の ~2倍）。
- **pf_1 実証（150井, n_seed32, chunk=1）**: fixed-lag192=7.6539 → **full=7.3064（−0.3475 改善）**。3井smokeでも −0.54。**速くて精度も上**という結果。
  - ※1バンクのみ。他バンク(self/nbr/pfA 等)・全井での確認が必要。datum両義の強い井で退化悪化する可能性は残る([[datum-direction-wall-final]])。

## 手法（実装のやり方）
### アルゴリズム（forward-store + 単一backward祖先sweep）
現 `_smoother_core`(pf_banks_v95/v97) の前方フィルタは**完全にそのまま**。違いは平滑の出し方だけ:

**前方（=lag0と同一コスト）**: 各ステップで
- `pos`（粒子位置, B×N, 並べ替え無しで append）
- リサンプル祖先index `anc_step`（resample井は `idx`、非resampleは恒等 `arange`）

を保存（`pos_hist[T,B,N]` / `anc_hist[T,B,N]`）。窓バッファ `buf[B,L,N]` は**廃止**（gather再索引が消える＝L項消滅）。

**後方（単一sweep, O(T·B·N)）**:
```python
a = arange(N)[None,:].expand(B,N).clone()   # 最終粒子の現在slot
wfin = w                                     # 最終重み(全区間=y_1:T で条件付け)
for i in range(T-1, -1, -1):
    pos_t = gather(pos_hist[i], 1, a)
    pts_full[:, i] = (wfin * pos_t).sum(1) - z[:, i]
    a = gather(anc_hist[i], 1, a)            # 1歩祖先を戻す(O(N))
```
各stepでの祖先整合: リサンプル時 `buf=gather(buf,2,idx)` と等価に、`anc_step[j]=idx[j]`（親slot）。非resampleは恒等。→ 最終粒子の系譜を1回なぞるだけ。

### 実装場所
- 実証コードは **`v98/test_full_smoother_v98.py`** の `dual_core`（forward同一＋fixed-lag/fullを両出力し比較）。ここから full だけ取り出して `run_smoother_ext` の平滑段に差し込む。
- config に `smooth_mode: "fulllin"`（既定 "fixedlag"）を足し、`_smoother_core` を分岐させるのが綺麗。

### メモリ（唯一のコスト）
- `pos_hist[T,B,N]` float32 + `anc_hist[T,B,N]` を持つ → **O(T·B·N) VRAM**（現行 O(L·B·N) 窓より大）。
- chunk=20 だと B=640 で長い井(T~10000)は ~23GB → **12GB GPUではOOM**。対策:
  - **chunk を小さく**（chunk=1〜4）。各井独立なので精度は不変。
  - `anc_hist` を **int16**（N<32767）にして半減。
  - それでも足りなければ **pos_hist を CPU 退避**（step毎に .cpu()、backwardでCPU gather。遅いが確実）。
  - Kaggle 2×T4(16GB) なら chunk やや大きめ可。

## 検証計画（次にやること）
1. `dual_core` を **self/nbr/nbr_gr5/self_graft/pfA/r0/r1 各バンク**で回し、fixed192 vs full の overall RMSE 差を全部出す（150〜全井、n_seed32、chunk=1）。
2. **datum両義の強い長い井（tail）で退化悪化がないか**を井戸別に確認（tail支配なので重要 [[hard-wells-priority]]）。
3. 良ければ create の平滑を full に置換 → **create が桁で高速化**（lag192の8倍→2倍）＝v99 の目玉。
4. 退化が問題なら **適応ラグ**（有効祖先数が閾値を割ったら打ち切り）or **FFBSi**（O(N·M)、遷移密度で退化解消・重いが精度上）を検討。

## 注意 / 位置づけ
- これは **poor man's smoother（系譜ベース）** であり **FFBSm/FFBSi ではない**（遷移密度を使わない安い版）。精度狙いの上位互換は FFBSi。
- fixed-lag=192 も同じ系譜ベース（窓をLで打ち切っただけ）＝ full はその全区間版。両者「遷移密度なし」。
- 特徴として消費されるので、full の「決め打ち性」がモデルの枝選択に効くかは実験次第。
