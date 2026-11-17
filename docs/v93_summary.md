# v93 総括ドキュメント（後で使う参照用）

> v93 で行った全内容を1本に統合。詳細な個別ドキュメントは [v93.md](v93.md)（設計）・[v93_backward.md](v93_backward.md)（fwd/fwdbwd結果）にあり、本書はそれらを含む**全体像＋今セッションの追加実験＋確定した知見**をまとめる。作成 2026-07-29。

---

## 0. タスクと指標（前提）
- ROGII: 水平坑井の評価ゾーンで **TVT（真垂直層厚＝地層内の縦位置）** を予測（回帰）。水平坑井 GR を参照 typewell の GR-vs-TVT に相関づけて位置を当てる。
- 指標 = **全坑井の全行をプールした overall RMSE**（坑井平均ではない）。隠しtest ≈200井。
- 誤差構造の本質: **datum両義性**（Milankovitchサイクル性で GR が±15ft の2位置で等しく合う）＋**区間branch**。行レベル信号はほぼゼロ（[[formulation-diag-segment]]）。

---

## 1. v93 パイプライン全体像

### 1.1 土台
- **v62 の別表現マルチ表現スタッキング**を、**信頼できる物理錨（GRフリー v2, v91/v74）**で再構築したもの。
- GRフリー錨 = `v74/artifacts_v74/grfree_v2_anchor.pkl` = `{wid:{tvt(3seed平均), sig(3seed std)}}`。**CV10.03 / 実LB10.877（転移率1.08＝リークなし）**。struct場（v38/v66）とは別物で**転移が素直**なのが採用理由。

### 1.2 PF 6バンク（v93 の背骨。全行の特徴の柱）
```
banks = ["pf_1","pf_2","pf_3","r0_seed32","r1_seed32","pfA"]   smooth_lag=16
```
- 各バンク × **tw / self / nbr** の3表現（typewell照合 / 自己近傍疑似tw / 空間近傍疑似tw）。
- **NN-emission**（学習GRマッチ, W=0.02, pfA=0.01）を尤度に乗算。★重要: **NN-emission の帯中心＝GRフリー錨**（＝物理錨に張った学習GRマッチ。§6参照）。
- pf_1/2/3・r0・r1 = 錨なし（use_anchor=False）、**pfA = 物理錨バンク**（use_anchor=True）。
- 単体PF RMSE（全井, 目安）: r0≈7.48 / pfA≈7.54 / pf_1≈7.91 / pf_3≈8.09 / r1≈8.82 / pf_2≈9.84。

### 1.3 ★2つのPFエンジン（混同注意＝今セッション最重要の区別）
| | **pf_engine_v93**（tuner用） | **pf_banks_v93**（本番特徴用） |
|---|---|---|
| 種別 | forward PF（単表現・高速・非平滑） | smooth PF（固定ラグ16・tw/self/nbr・派生フル） |
| NN-emission | **なし** | あり（W=0.02） |
| 用途 | optuna調整・脱相関探索 | parquet特徴生成（本番） |
| gs_scale | make_inputsで P["gs_scale"] 乗算 | build_smoother_inputsで env GS_SCALE 乗算 |
- **同じパラメータでも両者は結果が違う**（§5・§6の engine mismatch）。

### 1.4 特徴（features_v93）
- 各バンク47列（bank1で確認）: `pf_ancc/std/delta`, `pf_z/z_delta/vs_z`, `pf_vs_spatial/vs_dense`, `tdpf{-30..30}`(11), `pf_{k}_gr_*`(8, gr残差/best_off), `pf_{k}_rate/acc/rate_abs..`(7, path shape), `pf_self/nbr + vs_tw/vs_nbr/agree`。
- クロスバンク consensus: `pf_ancc_mean/med/std_between`, `pf_dev_mean/med`, `pf_diff_*`, corr summary。
- forward parquet 総列 ≈722（うち pf系 302）。META = {well,id,target,is_backward}。

### 1.5 スタック（L1-L7）と最良blend
- **L1 GBM**（LGB×3+Cat×3+XGB×3, GKF5）/ **L2 TCN**（全特徴, top-k絞らない [[tcn-no-topk]]）/ **L5 曲線トークン**（cross-attention, seqfeat 14ch=軌跡delta+std, 柱）/ L3画像 / L4 / L6 / L7。
- blend = nnls（3脚: GBM/TCN/L5）。**GroupKFold(5) 全脚共通・決定的**。

---

## 2. 主要結果（overall RMSE, target単位）

| 系統 | L1 GBM | L2 TCN | L5 | 等重blend | **nnls blend** |
|---|---|---|---|---|---|
| **fwd（baseline）** | 6.4954 | 6.2915 | 6.4864 | 6.1723 | **6.1382** |
| **fwdbwd warm**（forward終端init） | 6.4862 | 6.2004 | 6.2569 | 6.0862 | **★6.0079** |
| **fwdbwd anchor**（GRフリー錨終端init） | — | — | 6.40 | — | **6.1514** |

- **最良 = fwdbwd warm blend 6.0079**（fwd 6.1382 から −0.13）。nnls重み: GBM 0.00 / TCN 0.53 / L5 0.47。
- 提出物: `v93/submit_fwdbwd_v93.ipynb`（VARIANT="warm"/"anchor" トグル, 既定warm）。

---

## 3. 逆方向（backward）実験
- eval区間を終端→手前へ逆走した smooth PF（lag16, n_seed32）を forward と同格の full-mirror 特徴（bpf_ ~287列）＋backward固有（到達誤差 bpf_arrival_*＝heel既知値との差=非リーク / bpf_md_from_toe / bpf_vs_fwd）に。
- **init 2通り**: warm=forward終端（ls=fwd_far_end+z[-1], ir反転）/ anchor=GRフリー錨終端（独立第2推定, ls=錨tvt[-1]+z[-1]）。
- **結果: warm 圧勝**（blend 6.0079）。anchor は脱相関は強い（corr 0.71）が脚が弱く最下位（6.1514）。
- ★教訓: v10 では warm逆走が「forward答えに条件付け＝準重複ビュー」で LB悪化したが、**v93 の強い土台では warm が最良**（脱相関 corr median 0.79 で genuine, TCN/L5 が改善, GBM横ばい）。バグ修正: anchor init は `pf._ANCHOR[wid]["tvt"][-1]` を直接読む（attach_anchor が physics=False バンクの anc を0埋めするため x["anc"] は使えない→760ft drift）。

---

## 4. 既存6PFの誤差相関構造
- 誤差 e = pred − true のプール相関。**平均 0.725、15ペア中9ペアが >0.7**。
- **pf_1/pf_3/r0 = 冗長クラスタ**（相互 ~0.87）。**pf_2 = 多様だが弱い**（corr 0.54-0.62, RMSE 9.84）。
- ＝「精度が良いPFほど互いに似る（相関高い）」「脱相関したPFは弱い」という**脱相関×精度の緊張**が既存バンクにも現れている。

---

## 5. 脱相関PFチューナ（tune_decorr_v93）と ★engine mismatch の発見
### 5.1 設計
- 既存6PFと**誤差相関 |corr|≤0.7**（別の特徴を持つ）かつ overall RMSE 最小の forward PF を optuna探索。ランダム160井、gs_scale 1.0-2.0、物理anchor on/off選択、ソフト罰則 `rmse+1000·max(0,maxcorr−0.7)`＋feasible-best採用。
- best（PHYS=off）: 160井RMSE 7.65 / **全井9.63 / max_corr 0.698（≤0.7成立）** ＝脱相関だが弱い（pf_2級）。

### 5.2 ★engine mismatch（今セッション最大の発見）
- 同じ tuned パラメータを **smooth PF（pf_banks）で回すと 全井RMSE 8.262 / max_corr 0.759**。
- **forward で測った脱相関（0.698）が smooth では崩れる（0.759>0.7）**。理由: tuner が測っていた脱相関は「**forward候補 vs smooth既存**」で、**forward vs smooth という engine差が脱相関の一部を生成**。新PFも smooth 化（＝他PFと同じ engine）すると既存に似て相関が上がる。
- 皮肉: **smooth化で精度は上がる（9.63→8.26）が脱相関を失う**。
- ★確定則: **脱相関PFは「調整エンジン＝本番エンジン」で測らないと転移しない**。→ v94 は全て forward で統一（§9）。

---

## 6. ★NN-emission と物理錨の関係（gs_scale含む）
- **NN-emission = 「非物理の学習GRマッチ（encoder cosine類似）」を「物理的な錨位置（GRフリー錨）の周り」に配置したもの**。帯"中心"＝錨（物理）、帯"値"＝学習GR（非物理）。
- ★重要: `use_anchor=False`（尤度の錨prior OFF）でも、**NN-emission（W=0.02）は別枠で常時ONでその帯中心が錨** → **錨に縛られる**。既存6PFも同じ錨中心emissionを共有 → **新PFにNN-emission載せると相関が上がる**（0.698→0.759の一因）。
- 完全に物理錨から切り離すには **use_anchor=False かつ w_nn=0（NN-emission OFF）** が必要。
- **gs_scale** = clip後の GRノイズσ乗算（尤度の平滑化＝正則化）。素PF（既に高σ調整済み）では悪化、錨PF（pfA, drift防止）では改善。gs×1.3 実測: pf_1 +0.44 / r0 +0.12（悪化）/ pfA −0.09（改善）。

---

## 7. 7番bank（脱相関PF）を既存チェーンに足す実験
### 7.1 L5（軌跡delta+stdチャネルのみ）
- 7番bank（smooth, tw/self/nbr）を L5 に +3ch（K14→17）: **before 6.4873 → after 6.5222 = +0.035 悪化**。
- ＝**相関済み（0.759）PFは L5 に新情報を足さない**。脱相関が売りだったが smooth で死んだので当然。

### 7.2 tabular GBM（派生フル47列, bank1と完全一致）
- 7番bank を bank1 と**完全同構造の47列**（pf_z/tdpf/vs/dev 全て既存グローバル列＋pf_ancc_7＋CSVグリッドから導出）で生成 → GBM 9モデル before/after。
- lightgbm-1 fold別: fold0 −0.038 / fold1 −0.000 / fold2 **+0.081** / fold3 +0.013 → **OOF 6.5400 → 6.5407 = +0.0007（フラット）**。fold改善と悪化が相殺。
- ★結論: **相関0.759のPFは GBM にも新情報をほぼ足さない**（L5 +0.035 と同傾向）。lightgbm-1フラットを確認した時点で**GBM実験は中止**（残8モデルを回す価値なしと判断, ユーザー指示 2026-07-29）。→ **7番bank（脱相関PFのsmooth化）路線は L5/GBM 両方で無効を確定**。これが v94（最初から corr≤0.7 でゼロ構築）の動機。
- 生成器: `v93/gen_newpf_bank_v93.py`（smooth tw/self/nbr）→ `gen_newpf_full_v93.py`（派生47列）→ `add_newpf_full_v93.py`→`run_gbmexp.py`。

---

## 8. public手法の検証（best public notebook）
- **projection**（区分polyfit dU=(tvt+Z)−anchor vs 正規化MD, IRLS, blend）: **mild（deg5/bw0.3）で −0.035改善**。aggressive（deg3/bw0.75）は +0.13悪化。→ **採用候補（mildのみ）**。
- **gs×1.30**: §6の通り素PF悪化・錨PFのみ改善（既tunedのため利得薄）。
- **SG smoothing / warmup**: ほぼ無効。

---

## 9. v94 への引き継ぎ（本書を使う理由）
- v94 = **完全に作り直す新しいPFバンクをゼロから逐次構築**（v93非参照の独立環境 `v94/`）。
- 設計: forward PF（pf_engine_v94, 非smooth）で調整、600井ランダム（seed毎回変更）、trials200 / n_seeds32固定 / gs_scale 1.0-2.0。
  - **PF#1 = 比較対象なし**（相関制約なし, 純RMSE最小）。**PF#2以降 = 採用済み全v94 PFと corr≤0.7**。
  - **各PF採用後に全773井の誤差を保存**（pf_{i}_errs.pkl）＝比較対象がPFごとに増える。
  - 物理anchor順序 = **off, on, off, on, ...**（PF idx 奇数=off/偶数=on）。
- ★§5の教訓を反映: **全て forward で統一**するので脱相関が apples-to-apples（engine mismatch回避）。
- ファイル: `v94/pf_engine_v94.py`（v93複製・ART=artifacts_v94・v93非参照）, `v94/tune_v94.py`, `v94/artifacts_v94/`（錨pkl複製済）。docs: [v94.md](v94.md)。

---

## 10. 恒常制約（v93/v94共通）
- **★struct全面禁止**（v38/v66 struct場由来はCV→LB転移が悪い, [[no-struct-directive]]）。GRフリー錨（v91）はstruct非該当＝許可。
- CV を信じる（[[cv-lb-policy]]）。overall RMSE で評価（坑井平均禁止）。leave-field-out/厳格clean。
- RAM ≈20GB / VRAM 12GB / chunk~40 / lag≤192 / n_seeds勝手に下げない。ユーザーが書かせた学習コードは勝手に実行しない。

## 11. 主要ファイル索引
- PF: `v93/pf_banks_v93.py`（smooth, 本番）, `v93/pf_engine_v93.py`（forward, tuner）, `pf_banks_config.json`（6バンク）。
- 特徴/学習: `features_v93.py`, `build_seqfeat_v93.py`, `train_l1_gbm_v93.py`（GBM_IN/GBM_SUF override可）, `train_nn_v93.py`, `train_l456_v93.py`（L5, L5_SF/L5_OSUF）。
- backward: `build_backward_v93.py`（BWD_INIT=warm/anchor）, submit `submit_fwdbwd_v93.py/.ipynb`。
- 脱相関/実験: `tune_decorr_v93.py`, `gen_newpf_bank/full_v93.py`, `add_newpf_*`, `run_l5exp2.py`, `run_gbmexp.py`, `test_public_pp*.py`。
- 錨: `grfree_anchor_train.pkl`（artifacts_v93, v74由来）。
