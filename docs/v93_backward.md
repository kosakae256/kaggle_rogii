# v93 逆方向(backward)特徴 実験

> forward smooth-PF に加えて、eval 区間を終端→手前へ逆走させた backward smooth-PF(lag16, n_seed=32)を
> forward と同格の full-mirror 特徴(bpf_, ~287列)＋backward固有(到達誤差/位置/fwd比較)にして
> L1 GBM / L2 TCN / L5 に投入。init は2通り: **warm**=forward終端 / **anchor**=GRフリー錨終端(独立第2推定)。


## 仮説

- 逆走は forward と誤差の出方が違う(脱相関した第2の推定)ので blend が改善するはず。

- v10 で warm(forward終端init)の逆走は CV改善もLB悪化(=forward答えに条件付けされた準重複ビュー)。

  → **anchor(錨終端init)なら forward に依存しない独立推定**になり、真に効くかを検証する。


## 手法

- backward: `build_backward_v93.py`(BWD_INIT=warm / anchor)。MDミラー反転・per-eval-row 反転・同一PFパラメータ。

- 特徴: forward 派生器(`features_v93`)を bpf_ で再利用(shape/gr一致/consensus/self-nbr) + bpf_arrival_*(到達誤差=heel既知値との差, 非リーク) + bpf_md_from_toe/pos_frac + bpf_vs_fwd(EMIT_VS_FWD=1)。

- 学習: `V93_INPUT=fwdbwd|fwdbwd_anchor` で GBM/TCN/L5(seqfeat は bpf 軌跡ch追加で K=14→26)。CV=GroupKFold5(全脚共通)。


## 結果(overall RMSE = Kaggle指標。target単位)


| 系統 | 脚 | overall | fold0 | fold1 | fold2 | fold3 | fold4 | 備考 |
|---|---|---|---|---|---|---|---|---|
| fwd (baseline) | L1 GBM(9blend) | **6.4954** | 6.1148 | 6.6961 | 5.9390 | 7.0400 | 6.6248 |  |
| fwd (baseline) | L2 TCN(ens) | **6.2915** | 6.0274 | 6.9155 | 5.7789 | 6.2308 | 6.4455 |  |
| fwd (baseline) | L5(mean) | **6.4864** | 6.4792 | 6.6270 | 6.0077 | 6.3817 | 6.9033 |  |
| fwd (baseline) | 等重み blend(3) | **6.1723** | 5.9882 | 6.4696 | 5.6837 | 6.2808 | 6.4045 |  |
| fwd (baseline) | nnls blend(3) | **6.1382** | 5.9872 | 6.4998 | 5.6584 | 6.1299 | 6.3795 | L1 GBM=0.11 / L2 TCN=0.53 / L5=0.36 |
| fwdbwd (warm=forward終端init) | L1 GBM(9blend) | **6.4862** | 6.1167 | 6.8887 | 5.8276 | 7.2520 | 6.2393 |  |
| fwdbwd (warm=forward終端init) | L2 TCN(ens) | **6.2004** | 5.9288 | 6.6958 | 5.7539 | 6.4344 | 6.1425 |  |
| fwdbwd (warm=forward終端init) | L5(mean) | **6.2569** | 6.1889 | 6.7935 | 5.6185 | 6.2417 | 6.3843 |  |
| fwdbwd (warm=forward終端init) | 等重み blend(3) | **6.0862** | 5.8974 | 6.6055 | 5.4878 | 6.3918 | 5.9853 |  |
| fwdbwd (warm=forward終端init) | nnls blend(3) | **6.0079** | 5.8780 | 6.5409 | 5.4326 | 6.1365 | 5.9975 | L1 GBM=0.00 / L2 TCN=0.53 / L5=0.47 |
| fwdbwd_anchor (GRフリー錨終端init) | L1 GBM(9blend) | **6.4873** | 6.1251 | 6.6740 | 6.0116 | 7.2350 | 6.3149 |  |
| fwdbwd_anchor (GRフリー錨終端init) | L2 TCN(ens) | **6.2965** | 5.9768 | 6.7513 | 5.9153 | 6.4756 | 6.3249 |  |
| fwdbwd_anchor (GRフリー錨終端init) | L5(mean) | **6.3993** | 6.2450 | 6.8409 | 5.7943 | 6.6646 | 6.4003 |  |
| fwdbwd_anchor (GRフリー錨終端init) | 等重み blend(3) | **6.1859** | 5.9479 | 6.5470 | 5.7284 | 6.5653 | 6.0963 |  |
| fwdbwd_anchor (GRフリー錨終端init) | nnls blend(3) | **6.1514** | 5.9401 | 6.5705 | 5.7015 | 6.3993 | 6.1058 | L1 GBM=0.07 / L2 TCN=0.52 / L5=0.41 |

## blend(nnls)まとめ

- **fwd (baseline)**: nnls blend overall = **6.1382** (fwd比 +0.0000)
- **fwdbwd (warm=forward終端init)**: nnls blend overall = **6.0079** (fwd比 -0.1302)
- **fwdbwd_anchor (GRフリー錨終端init)**: nnls blend overall = **6.1514** (fwd比 +0.0132)

## bpf vs pf 重複ビュー診断(per-well corr(bpf_ancc_1, pf_ancc_1))

> 0.98+ なら逆走=forwardの重複(CV利得は過学習疑い)。0.6-0.9 なら独立性あり=本物の第2推定。

- fwdbwd (warm): (parquet 未生成 or bpf 無し)
- **fwdbwd_anchor**: n=773井 corr mean=+0.569 p10=-0.081 median=+0.710 p90=+0.942 | corr>0.95 の割合=8.28%

## 所感(自動生成の骨子。後で追記)

- fwdbwd (warm=forward終端init): fwd baseline(6.1382)に対し 改善(-0.1302)。
- fwdbwd_anchor (GRフリー錨終端init): fwd baseline(6.1382)に対し 悪化(+0.0132)。
- ★CV改善は割り引いて見る(v10 で CV改善→LB悪化の前科)。重複corr が高いなら CV利得は過学習疑い。
