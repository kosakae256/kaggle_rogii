# v95 学習結果（fold別・逐次記録）

> v95（5表現 tw/self/nbr/nbr_gr5/self_graft ＋ provenance、NN-emission/逆方向なし、forward-only）の学習結果。
> create_v95 → GBM → seqfeat → TCN → L5 → nnls 3脚 blend。各ステップ完了ごとに追記。指標=overall(行プール)RMSE。
> 作成 2026-07-30。

## 0. データ
- train.parquet: **rows=3,783,989 / cols=845 / wells=773**。target finite=100% std=15.83。
- 新規列: prov_ 33 / pf_nbr_gr5 39 / pf_self_graft 39。
- CV: GroupKFold(5) by well（全脚共通・決定的）。target = TVT − last_known_tvt。

---

## 1. L1 GBM（lightgbm×3 / catboost×3 / xgboost×3, 各5fold）
| variant | fold0 | fold1 | fold2 | fold3 | fold4 | OOF |
|---|---|---|---|---|---|---|
| lightgbm-1 | 6.256 | 6.749 | 6.023 | 6.696 | 7.510 | 6.666 |
| lightgbm-2 | 6.340 | 6.699 | 5.904 | 6.548 | 7.450 | 6.608 |
| lightgbm-3 | 6.351 | 6.588 | 5.918 | 6.584 | 7.501 | 6.608 |
| catboost-1 | 6.550 | 6.872 | 6.016 | 6.742 | 7.765 | 6.812 |
| catboost-2 | 6.473 | 6.802 | 6.016 | 6.746 | 7.773 | 6.786 |
| catboost-3 | 6.532 | 6.804 | 6.015 | 6.747 | 7.888 | 6.825 |
| xgboost-1 | 6.449 | 6.773 | 5.977 | 6.571 | 7.828 | 6.747 |
| xgboost-2 | 6.417 | 6.744 | 5.985 | 6.792 | 7.669 | 6.744 |
| xgboost-3 | 6.566 | 6.732 | 5.966 | 6.631 | 7.750 | 6.753 |

**★GBM 9モデル等重み blend OOF = 6.6529**（v93 GBM 6.4954 より +0.158 悪化。原因=NN-emission削除でPF特徴が弱化。fold2最易/fold4最難）

---

## 2. L2 TCN（mdn2 combo, 3seed×5fold）
| seed | fold0 | fold1 | fold2 | fold3 | fold4 | OOF |
|---|---|---|---|---|---|---|
| s0 | 5.973 | 6.753 | 5.712 | 5.978 | 7.361 | 6.385 |
| s1 | 5.903 | 6.899 | 5.759 | 5.946 | 7.616 | 6.464 |
| s2 | 6.108 | 6.623 | 5.645 | 5.906 | 7.338 | 6.352 |

**★TCN 3seed ensemble OOF = 6.3181**（v93 TCN 6.2915 比 +0.027＝ほぼ互角。追加特徴が系列モデルで効いた）

---

## 3. L5 曲線トークン（3seed×5fold）
| seed | OOF |
|---|---|
| s0 | 7.106 |
| s1 | 7.077 |
| s2 | 7.082 |

**★L5 3seed ensemble OOF = 7.0241**（v93 L5 6.4864 比 +0.538 悪化）
- ★原因: **seqfeatのK=14が pf_ancc/self/nbr_delta の旧3表現のみ＝nbr_gr5/self_graft を含まない**。L5は新表現を使えず、かつNN-emission無しの弱いPF軌跡だけを食う → 最大打撃。
- → 改善: EXTRA_DELTA で nbr_gr5/self_graft の delta を seqfeat に追加して再学習（§3b）。

## 3b. L5 改善（nbr_gr5/self_graft delta 追加, seqfeat K=14→20）
- seqfeat に `pf_nbr_gr5_delta_1..3` / `pf_self_graft_delta_1..3`（＋std）を EXTRA で追加。
- **L5 ext: s0=6.716 s1=6.641 s2=6.758 → 3seed ensemble OOF = 6.6368**（第1形 7.0241 から **−0.39**）。
- ★新表現が L5 でも効いた。以後 v95 の seqfeat は EXTRA 込み(K=20)を標準にする。

### ★改善後 nnls 3脚 blend
| 脚 | 単体OOF |
|---|---|
| GBM | 6.6529 |
| TCN | 6.3181 |
| L5(ext) | 6.6368 |
| **★nnls 3脚 blend** | **6.2036**（重み GBM0.04 / TCN0.63 / L5 0.33） |

**★v95 CV = 6.2036**（v93 forward 6.138 比 **+0.066**＝第1形の+0.133から半減。残差はNN-emission/逆方向の欠如＝Phase2で埋める）

---

## 4. nnls 3脚 blend（GBM / TCN / L5）— 第1形
| 脚 | 単体OOF |
|---|---|
| GBM | 6.6529 |
| TCN | 6.3181 |
| L5 | 7.0241 |
| 等重blend | 6.3634 |
| **★nnls 3脚 blend** | **6.2711**（重み GBM0.12 / TCN0.70 / L5 0.18） |
| (参考)nnls GBM+TCN | 6.3042 |

**★v95 第1形 CV = 6.2711**（v93 forward 6.138 比 **+0.133**、v93最良 6.008 比 +0.263）

---

## 5. 所感
- **v95第1形(6.2711)は v93 forward(6.138) に届かず**＝**NN-emission/逆方向を外した代償**が、5表現＋provenanceの利得を上回った（予想通り）。
- 脚別: **TCN ほぼ互角(+0.027)**＝追加特徴が系列モデルで効いた。**GBM(+0.16)・L5(+0.54)が悪化**。L5はseqfeatが旧3表現のみで新表現を使えていない（要修正）。
- **改善の道**: ①L5 seqfeatに新表現delta追加(§3b) ②Phase2でNN-emission＋逆方向を再追加＝v93水準〜超え狙い。
- ★示唆: **5表現＋provenanceは「全特徴を使うTCN」では機能する**が、GBM/L5では NN-emission喪失を埋めきれない。Phase2(NN-emission復活)が本命。

---

## 6. ★nn_emission版（v95b）— v93超え達成
- **create_v95b（nn_emission forward, w_nn=config値）で PF特徴を強化 → 全脚学習。**
- 脚別: GBM(no-nn)6.653 / **TCN-nn=6.1341**（v95 6.318→−0.18, v93 6.292比−0.16）/ **L5-nn(ext)=6.311**（v93 6.486比−0.18）。
- **★nnls 3脚 blend = 5.9871**（GBM0.06/TCN0.56/L5 0.38）。
  - **v93最良(6.008)を −0.02 超え、v93 forward(6.138)を −0.15 突破**。
  - しかも**逆方向なし・GBMはno-nnのまま**（GBM_v95bは50%で停止中）。nn-GBM＋逆方向で更に下がる余地。
- ＝**v95本領: 5表現＋provenance＋nn_emission で v93超え。第1形6.204→改善L56.018→full-nn 5.987**。
- 枝選択/per-well選択は情報限界＋CV過学習で死（検出0.80/方向0.578/v34は実LB崩壊7.279）。
- 次の脱相関脚候補=**v89 E-UNet**（GR表現U-Net, PF系とρ0.34, PF表現チャネル追加）。
