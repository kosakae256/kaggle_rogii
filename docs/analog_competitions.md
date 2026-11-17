# 類似コンペからの移植知見（deep-research まとめ）

> 逐次位置推定/信号アライメント系の類似コンペ上位解法から、我々のPFジオステアリングへ移植できる技法を
> deep-research（103エージェント・21ソース・62主張→上位25を3票敵対検証 = 22確定/3棄却）で抽出・検証したもの。
> 出典コンペ: Google Smartphone Decimeter Challenge(GSDC 2021/22), Kaggle Indoor Location & Navigation(2021),
> SEG 2016 Facies, FORCE 2020 Lithology, ほか DTW/well-log correlation。
> 最終更新: 2026-07-03

## 核心（勝ちパターン）
**「強い1ステップ観測モデル（WLS点測位 / WiFi指紋 / GR尤度）を、明示的な逐次デコーダ/平滑で包み、
最後の詰めは"より賢いモデル"でなく"軌跡の後処理"で稼ぐ」**。我々のPF（GR尤度=絶対項＋dip/rate=相対項）は
既にこの型。伸ばすのは後処理・平滑・コスト設計・合成方法・CV。

## 確定した移植技法（confidence: high / 3-0検証）
| # | 技法 | 我々への移植 | 出典 |
|---|---|---|---|
| 0 | **1ステップ観測を逐次平滑/デコーダで包む**（置換・素朴平均でなく） | PFの構造そのもの＝正しい。追うのは平滑/デコーダ側 | chadloh GSDC / ttvand Indoor |
| 1 | **FFBSi（前向き→後ろ向き粒子平滑）**: filter重みを後方遷移密度で再加重し全区間平滑 | **★最優先**。dip外挿・遅延証拠の原理的解。既存前向きPFに後ろ向きパス追加＝v23 Particle Smoother | Särkkä FFBSi (Godsill/Doucet) |
| 2 | **beam探索を"加算コスト=絶対マッチ項＋相対運動(dip)項"**で、**各penalty重みを別々にチューン**（Indoor優勝は~14項） | 我々のbeamを **GR一致項＋dip整合項**に分離し重み最適化。argmax/step でなく beam | ttvand Indoor(combine_predictions_beamsearch) |
| 3 | **snap-to-known-structure** は屋内で最強だが**我々は弱転移**（離散waypoint無し・準リーク依存）。代替=**typewellの地層トップ/マーカーhorizonへ軌跡を引き寄せる** | aug の datum粒子（±層厚再シード）と同系統。マーカーへの soft attract | Navigine / kuto5046 / ttvand |
| 4 | **robust/mode-aware 合成**（hill-climbing選択, seed平均）。**final TVTの素朴平均はダメ**（±15ft双峰を誤midpointに潰す）→ **状態/尤度レベルで合成** | 我々の「バンク平均7.859で悪化」を**理論裏付け**。seed尤度加重(既存)を維持、複数PFは尤度/状態で合成 or hill-climbing選択。素朴平均は禁 | NVIDIA Kaggle playbook / arXiv 2009.07701 |
| 5 | **grouped-CV-by-well**（坑井まるごとhold-out）＝過学習の第一防御。CV構造をデータ構造に合わせる | 私が提案した **out-of-fold調整** を裏付け。v22/v24過学習の正解。目的関数を坑井5分割overallに | SEG(STUART/CRAWFORD blind) / FORCE / arXiv |
| 6 | **標準loss＋強い過学習対策 ＞ 独自loss**（FORCE勝因はCV/正則化、独自penalty行列は選択のみ） | 独自PF lossを凝るより **overall RMSE目的＋強CV/正則化** | FORCE technical retrospective |
| 9 | ~~**DTW でGRログを参照テンプレへ整列**~~ **★このタスクでは死亡(実測済)**: 推論値が RMSE~200 級に吹き飛ぶ坑井が多発し刺さらない(ユーザー実験確認, 2026-07。公開 SCA-U2Net ノートのDTW特徴も CV9.55 で機能せず)。**補完素材としても再提案しない** | — | Silurian well-log DTW論文(移植不成立) |

## 示唆どまり（confidence: medium）
| # | 技法 | 注意 |
|---|---|---|
| 7 | **iterate-then-re-estimate（pseudo-satellite）**: 平滑軌跡を prior として per-step最適化に再投入し反復 | 26位writeup・"could"表現・効果未分離。**試作価値ありだが未実証**。我々=PF+smoother後に平滑軌跡を prior/再initして反復 |
| 8 | **遷移(相対運動)の系統バイアスを回帰補正**してからデコード（kuto5046はIMUデルタを線形回帰補正） | 我々=PFのdip/rateの系統的な over/under-dip を typewell に対し回帰除去 |

## 棄却（過度な期待を削った・信じない）
- 「Kalman+重み付けで42%改善」= 集計値で分離実証なし（0-3）。
- 「Kalman+線形補間が軌跡平滑」= 実は**訓練ラベル補完**の別工程（1-2）。
- 「方向依存の観測重み付けが有効」= ablation無し（1-2）。

## 我々のアクション（優先度順）
1. **Particle Smoother（FFBSi）を out-of-fold で詰める** ← 研究の最優先・v23「あり」と一致。
2. **beamコストを"GR一致＋dip整合"の項別設計**にして重み最適化。
3. **複数PFは素朴平均せず、尤度/状態レベル合成 or hill-climbing選択**（バンク平均の谷を回避）。
4. **out-of-fold(坑井5分割)目的でチューニング**（過学習対策の本筋）。
5. 新レバー: **typewell地層トップへの soft attract**（map-matchingの地質版）、**pseudo-satellite反復**（試作）。

関連: [[error-anatomy-ref]]（双峰datum）, [[pf-alternative-algorithms]]（smoother=あり）, [[v19-pf-bank-boosting]]（平均は谷/過学習）, docs/v23.md（Particle Smoother）。
