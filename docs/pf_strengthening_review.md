# PF強化案（user/rogii_particle_filter_strengthening.md）レビュー

> 作成 2026-08-01。ChatGPT提案12節を、我々の実績(メモリ/docs/コード)と照合して評価。
> 結論を先に: **提案の質は高いが、2大目玉(#1 backward smoothing / #2 look-aheadauxiliary)は共に既試行**。
> 中核の前提(median 6.5 = posterior-mass問題)は、我々が実証した **datum方向壁(情報限界)** と部分的に衝突する。
> ただし **真に未試行で価値がありうる断片**(resample-move / island PF / ESS適応tempering / block likelihood n_eff補正)が残る。

## 0. 提案の中核前提と、我々の実証との衝突
- 提案: 「oracle 2.5 / median 6.5 / ensemble 6.0 → 良い軌道は生成できている。**PF内部(提案・平滑・重み)を直して良modeにposterior massを集めれば median 6.5→5.x**」。
- 我々の実証([[datum-direction-wall-final]] ★★★): **±15ftのdatum枝方向は情報理論的に観測不能**(幾何/GR全表現/運動学の全軸でAUC~0.50)。GRが両位置で等しく合う(Milankovitch alias)ため、**どんな尤度/平滑/重み付けでも正しい枝を選べない**。
- つまり提案は「候補は良い→選択を直す」という**正しい標的**(=我々の[[hard-wells-priority]]/[[v47-selection-ceiling]]と一致)を、**選択=尤度/平滑の改良で解けると仮定**している。我々は選択が**情報限界**であり、かつ後述のとおり**PF内部改良は転移しない**ことを繰り返し観測している。
- ★重要な線引き: 提案の多くは「良modeを**維持**する」施策(mode-preserving等)。我々は既に候補**生成**は強い(oracle 3.3, [[v91-physics-prior]])。壁は候補を**選ぶ**方。維持を強化しても選択壁は動かない。

## 1. 節ごとの照合(既試行/結果 vs 新規)
| # | 提案 | 我々の状況 | 評価 |
|---|---|---|---|
| §1 | 失敗分解の診断(proposal recall / oracle粒子生存 / good-mode mass / ancestor多様性 / 内部oracle差) | 体系的にはやっていない | **★最有用・未実施。まずこれ**(下記) |
| §2 | Block look-ahead **Auxiliary PF**(未来GR区間で提案) | **v23 tune_apf_v23.py で実装済**。[[pf-alternative-algorithms]]「APF=PF中身いじりはダメ」 | 既試行・否定寄り |
| §3 | **Particle Smoothing / FFBS backward** | **v25で実装済**(tune_ffbsi_r0_v25, check_combine_smoother_ffbs)。[[v25-smoother-r0-plan]]: LB6.786だが**beam/hold寄与>平滑**。posterior-mean既定は我々も採用 | 既試行・小利得 |
| §4 | **Likelihood tempering**(ESSで二分探索, Δβ) | tempering自体はv11/v78に存在。**ESS適応で段階導入は未確認** | 一部新規・低コスト |
| §5 | **Resample-move**(ESS低下時MCMC rejuvenation) | **リポジトリに無し=真に未試行** | ★新規 |
| §6 | **Mode-preserving resampling / island PF**(flat/±slope/curve/self/tw/jump島) | 30PFは独立バンク。島間migration型は**未試行**。beam(v25/[[v76-stride-verification]])は近縁(区間beam ρ0.37も単体27.5でblend w=0) | ★新規(但し維持強化=選択壁は残る) |
| §7 | **Block likelihood**(NCC/微分/wavelet/soft-DTW/Student-t/**n_eff補正**) | 学習emission=効いた唯一([[v73-pf-ml-parts]] −1.22, [[v90-emission-pf-sdf]])。multi-metric loss=[[v47]]候補生成の[docs/v96_plan]item7で**未着手**。n_eff補正/Student-tは未明示 | 一部新規・堅牢化として妥当 |
| §8 | **self-template + typewell の α(u) emission混合** | self/nbr/self_graft徹底済。[[v88]]: selfを尤度に入れると第2推定器化(ρ0.79)で有害化も。α(u)適応混合は[docs/v96_plan]item4で未着手 | 一部新規・[[v88]]の教訓に注意 |
| §9 | **Regime mixture dynamics**(flat/smooth/turn/jump switching) | jump/regime=[docs/v96_plan]item6。v91は区分線形。switching-PFとして明示未実装 | 新規寄り |
| §10 | **低次元状態**(block (T,v,a,r) / B-spline control point 逐次) | v91=prefix-dip区分線形。B-spline逐次追加は未実装([docs/v96_plan]) | 新規 |
| §11 | **Rao-Blackwellized PF**(alias/regimeを粒子・offset/slopeをKalman) | **v23 tune_grid_rbpf_v23.py で実装済** | 既試行 |
| §12 | **prefix pseudo-PSでPFハイパー井戸別校正**(global shrinkage) | **Rolling pseudo-PS=[docs/v96_plan]item1で実施→✗不発**(prefix backtestがsuffix難度を予測せず AUC0.488, ρ0.02) | 既試行・**否定** |

## 2. この提案から得るべき最大の価値 = §1の診断
提案自身が言うとおり「診断なしに粒子を増やすと同じ失敗を高コストで繰り返す」。我々にとって診断は**決定的な問い**に答える:

> **良い軌道は「消える」のか(=resampling/退化=提案の#5/#6で直せる)、それとも「最後まで生き残るが重みが低い」のか(=datum情報壁=どんな重み付けでも直らない)?**

- 我々の事前(datum壁)予測: **硬い井では後者**(両枝が生き残り、尤度で分離できない)。→ #2-4,#7の尤度/提案改良は**効かない**。
- しかし**易-中位の一部**では前者(良粒子がalias早期ロックで消える)かもしれない。→ そこだけ #5 resample-move / #6 island が効く**可能性**。
- 診断が両者を**井戸別に分離**する。効く井の割合が分かれば、PF強化の上限が見積もれる(全773井overallで動くか、が[[cv-lb-policy]]の判定基準)。

### 診断で記録する量(§1 + 我々の指標)
proposal recall r_t / oracle粒子生存 / good-mode mass M_t(δ) / ancestor diversity / 内部5推定量(best粒子・max重み・wmean・wmedian・smoothed pmean)の差 / mode消滅率 / horizon別RMSE。既存の per-seed dump([[v34-seed-dump]])・FFBS祖先保存(v25)を流用できる。

## 3. 実験順序の推奨(提案の順序を我々の実績で上書き)
1. **§1診断を先に**(最安・最決定的)。v25のFFBS祖先保存 + v34 seed dumpを使い、硬い井で「消えるvs低重み」を分離。**これで残り全提案の効き代が判明**。
2. 診断が「消える」を示した井が相当数あれば → **#5 resample-move**(真に未試行・低コスト・ESS低下時のみ1-3回)を最優先。次に **#6 island PF**(30バンクを島+migrationに再編)。
3. 「低重み(datum壁)」が支配なら → PF内部改良は打ち切り、**#7 block likelihoodのStudent-t/n_eff補正**だけ堅牢化として試す(overfit注意, [[overfit-policy]])。それ以上は情報限界。
4. **既試行は再走行しない**: #2 APF(v23)/#3 FFBS(v25)/#11 RBPF(v23)/#12 pseudo-PS校正(不発)。過去のスコアと転移結果を先に確認。

## 4. 期待値の較正(正直な線)
- median 6.5→5.x は、**datum壁が支配的なら到達不能**([[datum-direction-wall-final]]: 現実は脱相関脚積み上げで~5.85-5.90)。提案の前提はこの壁を織り込んでいない。
- PF内部改良は**CV→LB転移が悪い**のが我々の一貫観測([[no-struct-directive]]/[[cv-lb-policy]])。採用は必ず**別seed値・全773井overall・fold間分散低**で判定。
- 現実的な上振れ: 診断で「消える」井が一定数あり、resample-move/islandがそこを救えば、**overallで−0.05〜−0.15程度**の可能性(利得はdecorrelated脚として[[v96-submit-notebook]]blendに載る形が本命。単体PFで6.5→5.xは非現実的)。
