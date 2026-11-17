# 誤差の解剖 ―― ROGII の誤差はどこにあるか(外部知見まとめ)

> 出典: Kaggle 公開ノート **"Stop reforking: where the ROGII error actually lives"**(Georgy Mamarin)。
> franticXu / Tabish / pilkwang / souldrive / Tucker らのディスカッション知見を統合したもの。`検証/stop-reforking-where-the-error-actually-lives.ipynb`。
> **この文書はコンペの構造を理解するための最重要リファレンス。v21以降の方針はここに従う。**
> 最終更新: 2026-06-30

## 0. 数字の地図(これが全ての基準)
- carry-last(last_known_tvt 横引き)= **LB 15.883**(=我々の RMSE15 と一致)。
- フォーク群(公開パイプライン)= **~7.2**(canonical = "LB7295 Public Rebuild")。**我々の v9 6.946 / v20 7.0台 はこの群**。
- ボード上位 = **~5.6**(rank1-2)。Tucker(rank2)は**坑井内データのみで ~5** と自己申告。
- **smooth oracle ≈ 3.0**(真値で6次フィット)。= まだ取り切れていない構造が下にある。

→ **15.9 → 5.6 のうち、どこまでが回収可能でどこからが地質(還元不能)か** がこのコンペの本質。

## 1. ★TVTは「地層面を読む」問題(我々の発見と一致)
- **`TVT = ANCC − Z + b_well`**、`b_well` は坑井ごと定数(std 0.0066ft)。`Z`はテール既知・offsetはヒールで固定 → **TVTを予測=ラテラルに沿って地層面を読むこと**。
  - = 我々が見つけた「TVT+Z = 地層 + 定数」と完全同一(既知の community fact だった)。
- **6つの地層列(ANCC〜BUDA)は6面でなく1面+5定数オフセット**(franticXu)。層厚 std < 0.01ft。**typewellのTVT境界から導出**された量で、独立な3D測定面ではない。
  - test には地層列が無い → 再構成が必要。一部の坑井は ANCC が欠損/切れている。
- TVTは**累積垂直距離(TVDのようなもの)**で、単層の厚みではない。

## 2. 誤差の分解: offset + piecewise dip(回収可能部分)
- 各坑井で `ANCC = TVT + Z` を MD の直線でフィットすると、**中央の坑井は数ft以内の直線**(median R²≈0.99)。
- 残りは **dip(傾き)とその折れ点**= Tabish の言う **"parallel jagged lines"**(区分線形)。offsetはヒールで固定、**dipと折れ点を読む**のが仕事。
- **フォーク(~7.2)= offset + 支配的slope を取った状態**(≈ line oracle 6.6)。**上位(~5.6)= dip変化とwiggleまで取っている**。
- **オラクル梯子**(真値で各形状をフィットした到達不能上限, pooled RMSE):
  `flat ~16 / constant-offset ~9 / line ~6.6 / smooth(deg6) ~3.0`。
- **「~10ft の壁」は存在しない**。それは著者の selector(PF+beam)を **leave-whole-field-out** という過酷分割で測った天井であって、タスクの床ではない。**伸びしろは median 坑井**(7.2→5.6 は per-well GRアライメント + dip平滑、cross-field ではない)。

## 3. GRマッチングの天井と「ヒール較正」(重要な合法レバー)
- GRで位置を読むには gain/offset 較正が要るが、**明示的 gain/offset フィットは位置と交絡**。理由: ラテラルは垂直に ~25ft しか動かないのに typewell は数百ft → **データだけから gain/offset を決められない**。
- 合法な flat-surface 推測での較正 → **~8%の坑井しか位置特定できない**(ランダム並み)。
- ★**ヒール較正(known ヒールで GR gain/offset をフィットしテールへ持ち越す)→ ~80% 特定(oracle 82% にほぼ並ぶ)**。GR回転デノイズ(FFT/rolling median)で 84%。
  - = **datum(±15ftのどちらか)は合法にほぼoracle率で特定可能**。ボトルネックはデータでなく較正だった。
  - 残る合法の難問は **dip(per-well shape)**: ヒール直線はテールへ外挿できない(§5)。

### 3.5 GR前処理の実測(自前検証, 2026-07-01): ハイパス/デノイズは【死】、低周波が信号
- `検証/gr_spectrum_analysis.py` / `gr_typewell_correlation.py`(自前)。
- **GRは全坑井きっちり 1.0 ft 刻み** → Nyquist波長=2ft。**回転(~0.01ft/rev)は原理的に不可視**(平均化済)。「センサー回転デノイズ」はこのデータの素の形では扱えない。
- エネルギーは**低周波支配**(水平井: 波長>50ft が 72%、<10ftの wiggle は僅か 10%)。
- ★**真TVTで引いた typewell期待GR と 観測GR の相関 = 生 median r 0.665**。**ハイパスすると単調悪化**(300ft:0.60/100ft:0.49/50ft:0.32/10ft:0.06、改善坑井25%→1.4%)。
  - 核心: **ピアソン相関は offset/gain 不変**(`corr(obs, a·tw+b)=corr(obs,tw)`)。生0.665は既にオフセットを見抜いた値 → ハイパスで消せるのは**本物の低周波共有形状(地層トレンド)だけ** → 下がる。
  - = **低周波は交絡でなく照合の主信号**。ハイパス/detrend/agc/deriv 系は全部 死(§3の「calibration中立」と整合)。
- **含意(頭打ちの根)**: 照合できる情報の大半が**粗いバルク形状**、細 wiggle はほぼ効かない(r≈0.06)。**粗くしか合わせられない=位置が細かく決まらない=双峰/曖昧性(§4)** が原理的に出る。→ 伸ばすのは**GR前処理でなく**、曖昧性を外から縛る側(kriging構造prior / posterior-mean hedge / Particle Smoother)。

## 4. ★★還元不能な尾: bimodal datum(硬い核の正体)
- Eagle Ford は **Milankovitch 周期で律動層理**(石灰岩-泥灰岩 couplet が ~15-25ft で反復, souldrive)。
- 反復ゆえ **水平GRが typewell と「1バンドルずれた2つの層序位置(~±15ft)」で同程度に合う** → datum が**双峰**。真のモードは**ほぼコインフリップ**。
- **最悪の ~10% の坑井が二乗誤差の ~40% を占める**(= 我々の「硬い核 ~30件」「tail支配」と完全一致)。
- 最悪十分位では **oracle 較正でも真TVT近傍の GR-misfit が平坦/わずかに反転**(デコイの方が僅かに合う)→ typewellでは原理的に区別不能。
  - souldrive: 明確な第二極小を持つ ~49% の坑井で、**コスト最小極小が正しい datum なのは 48.8%(コインフリップ)**、極小間マージンは正誤とほぼ無相関(r=0.054)。
- → **GR/PF/NCC のどれでも、この尾は「同点は検出できるが、勝者は選べない」**。= 我々の「選択は信号不足で3連敗」と同一現象。

### 決定理論(尾の正しい扱い): 単一モードに賭けず posterior mean
- 質量 `p` と `1−p` が距離 `d` 離れて存在する坑井で、二乗損失最適は **posterior mean = `p·a + (1−p)·b`**、還元不能リスク `p(1−p)d²`。
- **モードに賭けるのは `¼<p<¾` で midpoint より悪化**(50/50で ~21 vs 15)。posterior mean は全 `p` で最良。
- `p` は **2つの極小の相対深さの2位置softmax**(較正温度が必要)で読める。
- ★**PFを単一モードに collapse させずに走らせると posterior mean を自動で出す** → **blanket jump-guard は逆効果**(第二モードを消す)。

## 5. cross-field は確定した袋小路(我々のkriging案への警告)
- **leave-whole-field-out** で drift slope を安い特徴から予測 → **OOF R² < 0**(全体平均以下)。CNN/GRU/GBM 全滅。
- **cross-well 面再構成(IDW / 平面フィット, leave-field-out)は 13〜129ft で大失敗**。
  - = ★**我々の「近傍坑井から地層/geo_zを補間」案(2589ftに暴走)はこれと同じ**。**cross-field の面再構成は捨てる**。
- ただし注意: これは**最難分割(未知fieldへの外挿)**。コンペは未知fieldへの外挿を要求していない。**within-well の typewell マッチングは転移する**。within-field の学習可能性は否定されていない。

## 6. 3つの「進歩に見える罠」
1. **CV→LB 蜃気楼**: 深いモデルが fold で1桁RMSEでも実走で ~14。CVは見えた坑井の主張、LBは未見坑井。**実testに似た hold-out で検証**。
2. **★seed/refork ばらつき**: PFの再シードだけで **byte同一ノートが 7.168〜7.286**。= 我々の「±0.2はノイズ床」と完全一致。**0.03〜0.2の差で最終提出を選ぶのは運の選択**。差はseed帯を超えてから信じる。
   - **★GPU再現性の実測(2026-07-03, `gpu_repro_probe.py`)**: 乱数(Philox)は **クロスGPU/torchバージョンで一致**するが、**cumsum等のFPリダクションがGPUアーキ(T4 vs RTX5070)で異なり → searchsorted の index が飛び → リサンプリングで別粒子 → 長い逐次PFで軌跡が発散**。同一マシンは決定的(最大差0)。**CPU(`torch.set_num_threads(1)`)なら合計順序固定でクロスマシンほぼ一致**。ただ **32seedアンサンブルで overall は ±0.2 で再現**するので **ビット一致は追わない**(CV↔LBは±0.2で対応)。提出でper-well完全再現が要るときだけ CPU+threads=1 か「坑井ごとseed」。
3. **尾を追う罠**: 双峰坑井でモードに賭ける("ついに解けた"感)は期待誤差を上げる。**誤差が回収可能な median 坑井に労力を使い、尾はhedge**。
- メタ罠: 過保守な分割(leave-field-out)を「タスクの床」と一般化しない。

## 7. 合法性・小レバー
- **look-ahead は合法**(Tiago thread): eval行より先の行を含む全GRトレースを使ってよい → **whole-trace typewell マッチング可**。
- **GR回転デノイズ**: GRセンサは工具回転で回る。FFTに回転周波数が出る → マッチ前にデノイズで整列改善(§3で 80→84%)。
- **leak は無い(著者が撤回)**: 可視 `data/test/` は example、本番は隠しテスト ~200坑井。public LB ≈ その 26%(~50坑井)を honest 採点、private がメダル決定。→ **「XR Recovery=兄弟複製でRMSE≈0」のリーク路線は無効**(我々の COMPETITION_RULES の理解と整合)。

## 8. ★我々(rogii_claude)への含意・方針更新
- **現在地**: v9 6.946 / v20 7.0台 = **フォーク群(offset+line を取得済み)**。下げるには **median坑井の dip/wiggle** を取る = **per-well GRアライメント改善**。
- **v21 案A(学習尤度マッチャー)は方向が正しい**: より良い typewell マッチで median を ~5 へ。狙うのは**粗い位置でなく形(dip/wiggle)**。
- ★**ヒール較正は我々のデータで無価値と実測(2026-07-01, 773坑井, 検証/heel_calibration_viz.ipynb)**: §3の「8%→80%」は弱いflat-fit基準の話。**素のGRマッチが既に87%の坑井で位置特定**でき、full較正でも RMSE同点(10.55=10.55)・位置特定は生87%>full83%で**生が最良**。較正窓を短く(直前N)するほど縦の振れ不足でゲインaが暴走(std 0.13→1.01)し悪化(L50: 位置特定56%)。→ **目盛りずれは主要因でない。較正は打ち切り**。PFで一度19.2→17.9に動いたのはper-well±36の高分散で非ロバスト。
- 含意: **粗い位置(定数offset/datum)は既に易しい(87%特定)。伸びしろは形(dip/wiggle)と双峰tail**。v21案Aの狙いは「どの深さか」でなく「dipの形をどれだけ読めるか」。
- **v21 案B/物理項の見直し**: dip外挿の行き過ぎ抑制は有効だが、**双峰の第二モードを消さないこと**(jump-guard/単峰化は逆効果)。**PFを多峰のまま保ち posterior mean を出す**設計に寄せる。
- **硬い核(~30件)= 双峰datum+断層 = 還元不能**: 「検出して直す」のでなく **posterior mean で hedge**(賭けない)。**cross-fieldのkriging再構成は捨てる**(13-129ft/2589ftで実証)。
- **評価**: seed帯(±0.2)を超える変化のみ信じる。tailの桁変化 と train overall で見る。可能なら **within-field hold-out** を作る。
- **smooth oracle 3.0 が示す通り、median側にまだ ~2ft の回収余地**。ここが現実的な金への道(尾ではなく)。

## 関連
- [[hard-wells-priority]](硬い核=この双峰tail) / [docs/hard_wells.md](hard_wells.md)(該当20坑井) / [docs/v21.md](v21.md)(学習尤度=median改善の手) / [docs/EXPERIMENTS.md](EXPERIMENTS.md)。
