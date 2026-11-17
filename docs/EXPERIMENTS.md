# 実験ログ・知見まとめ(引き継ぎ用・最初に読む)

> これまでの実験の結果と知見を1か所に集約。新しいエージェント／作業者は **最初にこれを読む**。
> 各バージョンの詳細は `docs/vN.md`、LB一覧は `docs/scores.md`、ルールは `docs/COMPETITION_RULES.md`。
> 最終更新: 2026-06-30

## 現状サマリ
- ★**まず [docs/error_anatomy.md](error_anatomy.md) を読む**(外部知見でコンペの誤差構造を解剖した最重要リファレンス)。要点:
  TVT=地層面読み / 誤差=offset+piecewise dip(回収可能)+ **bimodal datum(±15ft, 最悪10%が誤差40%=還元不能)** /
  フォーク群~7.2(=我々)・上位~5.6・smooth oracle~3.0 / **伸びしろは median坑井のGRアライメント** / **cross-fieldのkriging面再構成は袋小路** / seed帯±0.2はノイズ。
- **ベストは依然 v9(Public LB 6.946)= フォーク群の位置**。以降の追加実験は LB を更新できていない。
- タスクの核: 水平坑井の `GR` を typewell の GR プロファイルに対応づけて評価ゾーンの `TVT` を予測。
  v9〜v15 はすべて **Particle Filter(PF)で GR照合→TVT軌跡** を作り、それを特徴量/画像として使う系統。
- **重要な気づき(直近)**: PFのパラメータ自体が(Optunaで)train に過学習しており、PFに依存する追加実験が
  CVで良くてもLBに転移しない。**PFに依存しない定式化への移行**が次の課題(末尾参照)。

## バージョン別サマリ
| Ver | 手法 | CV overall RMSE | LB | 所感 |
|---|---|---|---|---|
| v9 | PF(3セット,32seed)→ GBM(LGBM/Cat/XGB×3)+ apply_pp 補正 | (順方向のみ ≈7.99) | **6.946** | 現ベスト |
| v10 | v9 + **逆方向PF特徴(bpf_)**(cmp_は除外, 順方向GPU/16seed) | (2)順+逆 **7.55**(順のみ7.99) | **7.168** | **CV改善・LB悪化**(CV-LB乖離) |
| v12 | 1D-TCN(v10の全特徴を系列入力) | overall **7.871** / 坑井平均 6.33 | – | 単体はGBM未満。ブレンド素材 |
| v13 | 2D-U-Net(PF順/逆密度+GRコスト画像) | pf_1:8.583 / 3セット7ch:**8.337** | – | 画像のみ→タブ特徴に劣る |
| v14 | 画像の非学習読み出し10種(argmax/DP等) | 全て center(9.28)未満(最良soft-argmax 9.50) | – | 不採用。固定読み出しはPF平均を超えない |
| v15 | CNN系5手法(on-the-fly画像) | m1_unet **8.393**(他も8台) | – | ダメそう(打ち切り) |
| v16 | **PF全置換NN**(案A GRU+attn / 案B 微分可能DP) | – | – | **両案とも不調・採用見送り**(v9超えできず) |
| v17 | 物理候補生成 × CNN選択器 / **kriging prior** | – | – | 候補oracle15(扇カバー不足)。kriging priorは数値修正済(要再検証) |
| v18 | **PF候補(96=3×32seed)の選択を学習** | oracle4.62 | – | oracle低いがシードまぐれ込みで到達難。selectorは平均超えに苦戦 |
| v19 | **残差ブースティングでPFバンク生成**(物理PF,Optuna,再開可) | oracle(7本)5.76 | – | **v9+v19の7本oracle5.76<6.946**。多様性は相補・本物。単純平均8.24は弱い床 |
| v20 | **v10を物理PF×7本(r0〜r3+v9 pf1/2/3)に置換→GBM** + 選択器実験 | OOF≈8.0 | **7.012** | v9 6.946に僅か届かず=PF系統頭打ち。**PF選択は全信号で失敗(直交信号でもmean未満)→選択路線死亡** |

(参考: pf_ancc 単体 overall ≈ 9.19、順方向32本単純平均 ≈ 9.88、center=順方向平均 ≈ 9.28)
(基準: **RMSE≈15 = last_known複製の無意味ライン**。まず15を下回るか→次に v9 6.946 を下回るか。)

## 主要な実験結果(詳細)
### GPU化(v9 PFの高速化)
- CPU(numba) vs GPU(PyTorch, batched)。100坑井で CPU 212s → **GPUバッチ 10.2s = 20.8倍**。精度悪化なし(乱数変動の範囲)。
- 坑井数を増やすほど比が拡大(GPU時間はほぼ最長坑井のステップ数依存)。

### v10 逆方向PF(CV vs LB)
- CV(GroupKFold5 by well, overall): (1)順方向のみ **7.9914** → (2)順+逆 **7.5537**(Δ-0.44)。**逆方向はCVで明確に効く**。
- (3)比較特徴 cmp_ は **有害 → 除外**。
- 提出モデル(順+逆, cmp除外, 3セット, fwd=16/back=8)の 9モデルOOF RMSE = **7.5753**(補正前)。
- **Public LB = 7.168(v9 6.946 より悪化)**。→ 交絡: 順方向PFも GPU/16seed に変更している(v9は32seed)。
  原因仮説: **PFパラメータの過学習**。逆方向はその出力に依存するため転移しない。
- 切り分け中: 「順方向のみ・128seed」版を作成可(`create_train_parquet_v10.py` の `BUILD_BACKWARD=False`)。

### v12 1D-TCN / ブレンド・スタッキング
- TCN単体 overall 7.871(タブ特徴フル使用、PF波形だけではない)。
- **ブレンド**: LGBM(v10,2)×TCN OOF、相関0.95でも best w(LGBM)=0.64 → **7.3924(Δ-0.16)**。系統差で改善。
- スタッキング(TCN予測をLGBM特徴に)も用意(passthrough型)。※ 純粋スタッキングではない点に注意。

### v13 2D-U-Net / 一致度仮説
- U-Net: pf_1 8.583 → 3セット7ch 8.337(PF情報増で改善も、タブ特徴のGBMには届かず)。
- **仮説「順×逆の一致度が高い波形=真」は不成立**: 一致上位5は悪化(10.2〜10.4)、尤度加重平均 pf_ancc(9.19)が最良。
  一致=確信度であって正確さではない。

### v14 画像の非学習読み出し(10種)
- 全手法が center(=順方向PF平均, 9.28)を超えられず。argmax系は飛んで悪い(11+)、**点GRは壊滅的(gr argmax 19.4)**。
- 結論: **固定読み出しはPF平均を超えない。点GRは弱い→窓NCCにすべき。超えるには学習 or GR表現強化が必要**。
  (※ 単体で弱くても「特徴量」としてGBMに足す価値はある、という指摘あり)

### PFチューニング(GPU + 物理項, Optuna)
- 場所: `pf_tuning/pf_tuning_gpu_phys.ipynb`(n_seed=8固定・他Optuna・全well・overall RMSE目的)。
- 物理項 = `lk_phys=exp(-0.5*((rate−dip_prior)/phys_sig)^2)`、rate=d(TVT+Z)/dMD、dip_prior=既知ゾーン回帰(リーク無)。
- best overall RMSE ≈ **9.068**(phys_sig≈0.93=緩い拘束)。物理ON/OFFの公平比較は「物理あり前提で共最適化済み」のため要・別study。

## 重要な知見(横断)
1. **CVとLBが乖離**(v10): trainのoverall CV(難坑井fold4に引っ張られる)は隠しtest約200坑井へ転移しにくい。**矛盾時はLBを信じる**。
2. **PFパラメータが過学習**している可能性が高い(Optunaがtrainに最適化)。これが追加実験がLBで伸びない根本原因の有力仮説。
3. **比較特徴 cmp_ は有害**。逆方向 bpf_ はCVで効くがLBで損。
4. **点GRマッチングは弱い/有害** → **窓付きNCC(セミブランス)** にすべき。
5. **物理拘束(TVT+Z=地層深度が滑らか=dip連続)** が未活用。GRだけでは決まらない。
6. **PF画像のみの学習(v13/v15)は 8.3〜8.6 で頭打ち**。タブ特徴(空間KNN/beam/NCC/GR統計)を持つ GBM/TCN が強い。
7. **ブレンド多様性は効く**(v10×v12=7.39)。系統の違うモデルを混ぜる価値。

## v18-v20 「生成＋選択」路線の総括(2026-06-30, compact直前)
**狙い**: PFを多数(バンク)生成し、「**どのPFの出力を信用するか**」を学習(=選択)して oracle に迫る。
- **PFバンク生成(v19)**: 物理PF(`lk_phys`)を**残差ブースティングでOptuna**(`v19/iterative_pf_boost_v19.py`, 再開可・探索8/評価32seed分離・ラウンドseed+1000)。`pf_bank.json` に各ラウンドbest。
  - 確認(`v19/check_bank_oracle.py`, `INCLUDE_V9=True`): **v19 4本+v9 3本=7本で oracle 5.76 / 壊滅(>10)46本**。**v9とv19は相補**(パラメータ類似でも別坑井をカバー)。**単純平均=8.24(弱い床)**。
  - 未カバー核(oracle>10)狙い撃ち: `v19/tune_uncovered_v19.py`(全PFが>10の坑井だけ再チューニング)。
- **v20**: v10を物理PF×7本に置換(`v20/create_train_parquet_v20.py`, `SELECT_PFS`で選択可・順方向のみ・各PF pf_*特徴)→ GBM(`train_models_v20.ipynb`, 素のLGBM/Cat/XGB・fold毎保存・**ROW_FRAC行間引きでOOM回避**・学習済みskip)→ 提出(`submit_v20.ipynb`, **コード全インライン・modelsのみ外部**)。
  - GBM OOF ≈ **8.0 ≈ v9 OOF 7.99**(→apply_ppで~6.9)。**物理PF＋多PFでも v9 を明確に超えず=PF系統は頭打ち**。

### ★最重要の負の結果: PF選択は「信号不足」で失敗
- `v20/pf_selector_simple_v20.ipynb`(各PFを GR当てはまり+パラメータ+井戸レジームでスコア→最良選択, LightGBM, 5-fold by well):
  **的中率 0.19(≒ランダム1/K)、selector(hard9.3/soft8.9)は平均8.6に負け**。
- 原因: 入れた信号が**全部「GR尤度っぽい」**ものだが、**GR尤度≠真値**(正解候補は consensus から離れた少数派, GRは多義的)。各行の best PF は**散らばりの最小値=ほぼノイズ**で学習不能。
- **oracle(5.42〜6.24)は大半が到達不能**(順序統計のまぐれ)。oracle を反映可能な目標として過信しない。

## ★選択路線は完全に死亡(2026-06-30, signal_compare_v20.py 実行済)
GR尤度と直交する信号を総当たり(`v20/signal_compare_v20.py`)した結果、**全信号が mean-ensemble に負けた**:
```
mean   8.189 | A knowntail 9.754 | B krige 9.651 | C consensus 8.421
D dipdev 11.126 | E rough 9.176 | META(全結合) 8.499 | oracle 5.658(到達不能)
```
- **本命だった A(既知ゾーン答え合わせ)が 9.754 で最悪級**。意味: **既知ゾーンの当てはまり ≠ 評価ゾーンの正しさ**。
  どのPFも既知ゾーンには適合してしまい、既知末尾誤差に識別力が無い(あってもノイズ/逆相関)。難所(dip変化)で割れるPFを既知だけでは見抜けない。
- META(全部LightGBM結合)すら 8.499 > 8.189。**「バンクから正解PFを当てる」は考えうる全信号で否定**。oracle 5.66 は活かせない。
- 結論: **v18-v20「生成＋選択」路線はここで打ち止め**。バンク mean 8.189 は v9 raw 7.99 にすら届かない。

## 次の方向(残る本命): kriging を「選択器」でなく「baseline置換(構造prior)」として使う
B krige が選択器で負けた(9.651)のは枠が悪い。本来の使い道:
- **baseline 差し替え**: 現 baseline = 既知ゾーンの直線dip外挿。これを**近傍坑井から kriging した構造面**に置換。
  近傍坑井の `geo_z=TVT+Z` を空間補間 → **評価ゾーンの geo_z を直接予測** → `TVT = geo_z − Z`。**GR非依存・坑井跨ぎ**。
- これはPFの上に乗せる補正ではなく、**baseline を差し替えて GBM の target(=TVT − baseline)を作り直す**話。フォーラム1位の "LB<7 magic" の核。
- 実装資産: v17 に数値安定版 kriging あり(dx/dy を ls でスケール・trace比例ridge・近傍値域clamp)。`v17/kriging_prior_v17.ipynb`。
- 別ルート(安い): リーク悪用(XR Recovery, 可視test=train重複の同一typewell兄弟)単体 LB 7.295。ブレンド素材として温存。
- 関連メモリ: [[v19-pf-bank-boosting]] [[v18-particle-selector]] [[rmse15-naive-baseline]]。

## 方針更新(error_anatomy 反映, 2026-06-30)
- ★**kriging/cross-well 面再構成は袋小路と判明**([error_anatomy §5](error_anatomy.md)): leave-field-out で 13〜129ft、我々の近傍補間も 2589ft に暴走。**cross-field 構造再構成は捨てる**。
- **本命は median坑井の per-well GRアライメント改善**(7.2→5.6 はここ)。**v21 案A(学習尤度マッチャー)が直球**。+ **ヒール較正**(known ヒールで GR gain/offset をフィットしテールへ持ち越す→datum を ~80% 合法特定)。
- **硬い核(~30件)= bimodal datum + 断層 = 還元不能**。「検出して直す」でなく **posterior mean で hedge**(モードに賭けない)。**PFを多峰のまま保つ(jump-guard/単峰化は逆効果)**。
- 旧アイデア(まだ部分的に有効): `s=TVT+Z` 残差を 窓NCC+dip滑らかさで正則化 → 既知ゾーン自己校正 → TVT状態空間DP。

## 参考ドキュメント
- [analog_competitions.md](analog_competitions.md) … **類似コンペ(Decimeter/Indoor/FORCE/SEG/DTW)上位解法からの移植知見**(deep-research検証済22/3)。FFBSi粒子平滑=最優先、beamコスト項別設計、素朴平均禁(尤度/状態で合成)、grouped-CV-by-well。
- [pf_alternatives.md](pf_alternatives.md) … **PFの派生/近縁アルゴリズム候補(Auxiliary PF/RBPF/GSF/グリッドHMM 等)とGPU適性**。DTW/Viterbi却下・grid系見送りの結論つき。
- [beyond_tabular_proposals.md](beyond_tabular_proposals.md) … **非タブラー推論の提案4件(2026-07-06)**: ①系列NN(=v29実装済) ②GBM→PF閉ループ ③nbrによるdatum双峰解消 ④v28×v27低重みブレンド。
- [pf_bank_aggregation.md](pf_bank_aggregation.md) … **PFバンクを本数無制限で使う表現(2026-07-06, 未実験)**: 2クラスタ統計等の集約特徴 / NNのset-encoder。per-PF別モデル案は過去実験済で却下。_re oracle5.0の割引とseedノイズ診断。
- [error_anatomy.md](error_anatomy.md)(誤差解剖) / [hard_wells.md](hard_wells.md)(硬い核) / [v23.md](v23.md)(grid試行)。
- [v33.md](v33.md) … **選択路線の再訪=「井戸レベルのモード推定」診断(2026-07-06)**: 192本選択でなく2〜3モードの井戸分類に潰す。3ゲート(モードoracle/外部情報AUC>0.6/学習ゲートOOF<6.796)で go/no-go。**結果: AUC0.82/ゲート6.749でPASS、ボトルネック=候補生成(回収26%)**。
- [v34.md](v34.md) … **per-seed候補+クロスソース採点(2026-07-06)**: tw pf_1/2/3のper-seed 32本をdump(平均前を回収, コア無改造)し、候補軌跡をself/nbr gridで採点。v33ゲート再走で回収率26%→?を測る。
- [future_experiments.md](future_experiments.md) … **DA/TTT/合成のバックログ(2026-07-07)**: test既知帯活用(A=test相互の近傍参照プール化/B=正規化再較正/C=教師ありTTT)、マルチカット(GPU 1カット≈4h)、seedジッタ行(ほぼタダ)、typewell合成井戸(おとり無限製造=双峰の教師)、HMM厳密スムーザ移植。

## ファイル早見
- 提出: `v10/submit_v10.ipynb`(Kaggle推論), 学習: `v10/train_models_v10.ipynb`(ローカル), 特徴: `v10/create_train_parquet_v10.py`(`build_v10_dataset`)。
- PF GPU実行: `v9/pf_gpu_run.py`。PFチューニング: `pf_tuning/`。
- 画像系: `v13/`(生成・U-Net), `v14/`(読み出し), `v15/`(CNN5種)。
