# 夜間自律実験ログ（2026-07-31〜）

> ユーザー就寝中の自律実験。目標=**実RMSE 4.x**（oracleでない）。現状 v95 blend CV 5.987。
> 制約: `tune_v94_bpfeat.py`(GPU使用中, PID11620)を止めない / **GPU控えめ=CPU中心**（GPU 98%飽和中）/
> deep-research→実験→考察→deep-research再実行の反復 / 全記録をここに。

## 出発点の確立事実（v91/session知見）
- **誤差解剖(oracle)**: datum 1.9 + segment branch 4.3 + fine床 0.18。点回帰は床。**回収の的=segment/well枝**。
- **候補生成=天井**: 物理prior oracle 3.3（K4,N10k）。but **選択=壁**（GRマッチ22/U-Net11で候補を選べない）。
- **datum両義性=情報限界**: 方向（±15ftどちら）AUC~0.56。検出(ambiguousか)はAUC0.80だが対称ゆえ点推定改善せず。
- **強baseline法則(4度確認)**: 強blend(5.987)に対しE-UNet/ヒューリスティック/モードゲート/closed-loop全てmarginal(~−0.01〜0.03)。信号枯渇。
- **CV種別**: ★test/trainランダム分布→**well-CV正当・非リーク / field-out誤り**。枝検出AUCは本物の可能性。
- **既存の脱相関blendレシピ**: v90+snap(物理×GR)+v74(GRフリー)=−0.083（固定weight, ρ(snap,v74)=0.328）。
- **モードゲートv95**: 検出AUC0.948, oracle−0.39だが拡散適用死/hard9井−0.017。closed-loop: 脱相関ρ0.82だが特徴cl_res+0.007(冗長)。

## 4.xへの論理
- 点推定は床。**4.xには方向（segment枝）解決が必須**。v91は情報限界と結論したが、
  ①区間構造予測(CRF/Viterbi) ②空間拘束の正しい定式化 ③set予測 が未検証の切り口。
- deep-researchで step-change 方向を探索 → CPU実験 → 反復。

## 実験ログ
（以下、deep-research結果を受けて追記）

---
## deep-research #1 結果 (2026-07-31) — 候補生成が天井/独立軸は空間dipのみ
検証済8finding(25主張3-0). 出典=Petrophysics(Al Ibrahim2022 DTW多解), P-DTW(2025), MDN多解, WTA/MCL(annealed/TimeMCL), soft-DTW/soft-Viterbi(1802.03676), ベイズジオステアリングEnKF/GAN, GSDC2021 factor-graph.

### 3系統(step-change候補)
1. **確率的DTW(SDTW/P-DTW)**: cost行列を確率的再走査→アライメント候補の集合。周期曖昧向け設計。但し**caveat: DTWは単調warp=非単調な水平坑井TVTに直接不適/±15枝を跨いでサンプルするか未検証(最重要検証)**。
2. **MDN+WTA/MCL(MTP loss)**: K頭が別モードに特化(平均崩壊回避)。TimeMCL=系列版。**但しGPU学習要**。
3. **ベイズアンサンブル・ジオステアリング**: 構造prior+wellbore幾何+参照log+観測logの**4源融合**。幾何/dip項がGRを超えて位置拘束。

### ★決定的caveat / open questions
- **RMSE対称両義: set予測は枝コミット前提、hedgeが最適 → 選択器が>50%当てる時のみ改善=0.56 AUC壁。GR単独で壁破れる文献ゼロ。独立軸=空間/dip/近傍のみ**。
- OQ1: 確率的DTWは±15枝を別realizationとしてサンプルするか？(単調warpで1枝内jitterに留まる懸念)
- OQ2: 枝方向AUC0.56を幾何/dip/近傍面拘束"だけ"が超えられるか？tail転移するか？
- OQ3: RMSE対称下でWTA/MCL脚がhedgeを超えるか？(prefix TVT_input+近傍相互をtransductive証拠に枝コミット)
- OQ4: per-segment枝をsoft-Viterbi CRF(end-to-end RMSE)で解けば4.3prize取れるか？or 信号が特徴に存在しないか？

### 我々の既存知見との突合(重要)
- v34モードゲート: 幾何特徴(geo_jump0.86/rough_ratio0.81/dip_absmax0.78)で**モード選択AUC0.948達成済=幾何は方向を部分的に解く**(研究Finding5を我々が既に実証)。
- but モードoracle=5.59(候補プール天井)。**4.xには richer候補(v91物理oracle3.3)が必要**。
- v91: 物理候補oracle3.3だが選択失敗(GR22/U-Net11)。**★未試行=v91物理候補×v34幾何ゲート機構の組合せ**。
- v91既知の反証: 「物理拘束で候補削減は全滅=両義デコイは真の線と同じ幾何柔軟性内に同居」→幾何"plausibility"単独では枝を割れない。研究の独立軸=近傍面だがv91クリギングは470ft間隔で±15ft短波長に不足。

---
## 実験A: per-segment枝Viterbi (CPU, 2026-07-31) — ★oracle3.22実在だがGR emission死
- blend5.9871 / **per-segment枝oracle=3.2164(★4.x射程の天井実在)** / Viterbi(PF支持emission+dip連続)=6.676(+0.69自爆) / per-well多数派=5.982(−0.005)。
- ★**構造(Viterbi)は正しいがemission=PF支持が両義consensusで枝方向を当てられずコミットで悪化**=研究caveat「GR単独で枝方向壁破れず」を我々データで実証。
- 含意: **prize(3.22)は実在→emission(枝方向信号)を独立軸で作れれば4.x到達可能**。次=幾何特徴学習emission(v34モードAUC0.86-0.948の segment版)。

---
## 実験B: per-segment 枝方向 学習分類器 (CPU, 2026-07-31) — ★方向=完全chance
- 幾何(incl/dip_impl/rough)+PF votes+位置特徴でsegment枝を多クラス学習(well-CV)。
- **shifted検出AUC=0.7579(曖昧segmentは検出可)** / **★★枝方向AUC(+vs−)=0.5003=完全chance** / argmax+0.68/Viterbi+0.60悪化。
- ★**幾何+PFでは枝方向ゼロ情報=壁をsegmentレベルで確定**。oracle3.22は方向信号なしでは幻(研究caveat/v91確定)。
- → 残る独立軸=ユーザー案「GR別表現(envelope/位相/微分)」。amplitude+幾何と独立に方向を割れるか(実験C)。

---
## 実験C: GR別表現で枝方向 (CPU, 2026-07-31) — ★GR全表現でも方向信号ゼロ
- shifted segment 29731. GR別表現(gr_env/gr_d1/gr_d2/gr_rough/grs多scale)+offset相関(tdpf/tdsc/tda 77列)+grbest=112特徴で方向(+vs−)をwell-CV学習。
- **方向AUC=0.4710(≒chance以下)** / GR別表現のみ0.4386 / offset相関のみ0.4710。
- ★**GRのどの表現(amplitude/envelope/微分/roughness/offset相関)でも±15ft方向は観測不能**。ユーザー「波形として何かある」仮説は棄却。
- **結論(実験A+B+C): 自井の全情報(GR全表現+幾何)で枝方向=完全にchance=情報限界確定**。oracle3.22は方向信号なしでは到達不能。
- → 4.xの唯一の残路=井戸間独立軸(軌跡オーバーラップ相互拘束/運動学的レート)。deep-research#2待ち。単井路線は枯渇。

## 現時点の総括（4.xの見通し）
- **prize実在**: segment枝oracle3.22 → 枝を解けば4.x射程。
- **壁確定**: 枝方向は自井GR(全表現)+幾何で0.47-0.50=情報限界。研究caveat「GR単独で壁破れず、独立軸は空間のみ」を3実験で実証。
- **残path**: ①井戸間軌跡オーバーラップ(v91は centroid470ftで死、軌跡crossingは未検証) ②運動学的レート(dip×wellbore幾何) ③WTA/MCL set予測(but RMSE対称でcommit問題) — deep-research#2で優先度判定。
- **確実な小利得(4.xと別)**: v91 snap+v74脚(−0.08), closed-loop βblend(−0.01)。4.x未達でも積める。

---
## cross-well軌跡crossing (CPU, 2026-07-31) — ★カバレッジ3%+適用破綻
- 軌跡crossing<150ft: eval行の3%のみ。where近い |真TVT_A−真TVT_B|中央0.8ft/<7.5ft=66%(局所一致は本物)。
- but 井戸レベルdatum補正(近傍真TVT): 非0shift=23井のみ→**適用で7.3→424破綻**(crossingは同一datum保証なし=別地層深度交差)。cross-wellも4.x道でない。

## deep-research #2 結果 (2026-07-31) — ★~6壁は現実、3.22 oracleは部分的に幻
- **唯一のamplitude非依存レバー=運動学的(feature SPACING vs 既知inclination)**。TST forward(Berg&Newson URTEC2013): 測定厚≥TST=水平logは常にsqueeze、feature間隔が(dip−inclination)に紐づく。
- 多井loop closure: 軌跡overlapで大域最小二乗。但し我々coverage3%で限定的。
- prefix錨: 距離で信頼減衰、近prefixは信頼可。
- loss reframing(MDN-MTP/EWTA): 両義性を表現するだけ、mode確率はGR条件付き=独立軸でない=0.58超えない。
- ★**reality check: 真に周期的信号の事後は証明的に~50/50(StratoBayes 50.1/49.9)。オラクル級datum精度の文献ゼロ。~6壁が現実、3.22 oracleは部分的に幻**。
- ＝我々の実験A/B/C(方向0.47-0.50=chance)を文献が裏付け。

---
## 実験D: 運動学的レバー (CPU, 2026-07-31) — ★これも0.50
- rate/acc/inclination/GR間隔(lag/lead)80特徴で枝方向。**方向AUC=0.4964≒chance**。
- 研究#1(amplitude非依存の運動学的feature間隔)も我々データでは方向信号ゼロ。

## ★★★夜間自律実験 最終結論 (2026-07-31)
**RMSE 4.xは到達不能。** 4実験+2 deep-researchが完全一致:
| 情報軸 | 枝方向AUC |
|---|---|
| 幾何+PF (実験B) | 0.5003 |
| GR全表現:振幅/envelope/微分/roughness/offset相関 (実験C) | 0.4710 |
| 運動学的:rate/spacing/inclination (実験D) | 0.4964 |
| cross-well軌跡crossing | coverage3%+適用破綻 |
- **自井の全情報軸(振幅・幾何・運動学)で±15ft枝方向=完全にchance=情報限界**。
- 文献裏付け(DR#2): 真に周期的信号の事後は証明的~50/50(StratoBayes)、オラクル級datum精度の文献ゼロ、~6壁が現実、3.22 oracleは部分的に幻。
- **prize(segment枝oracle3.22)は実在するが、それを取る方向信号がどの軸にも存在しない**=v47/v91/formulation-diagの「datum両義=情報限界」を4軸×文献で最終確定。

## ★到達可能な現実路線(4.xでないが実利)
- **v91 snap+v74脚**(GRベース物理×GRフリーの相互脱相関ρ0.33): v90で固定weight −0.083実証。v95で再評価する価値。
- **closed-loop βblend** −0.0097(今夜確認)。
- **nn-GBM**(GBM_v95b未nn化, ~−0.15期待)/**逆方向PF**(v93 −0.13)。
- これらで v95 5.987 → ~5.85-5.90 が現実的射程。**4.xは情報論的に不可能**。

---
## 実験E: 物理候補×今の特徴で選択 (CPU, ユーザー案) — ★選択の壁 再確認
- v91物理候補 N=3000/K4 (200井): **oracle=3.94(頭上余地−1.5 実在) / GRマッチ選択22.1 / ★学習選択(gr_score/dip/rmse_bl/rough等)=6.45=blend5.44に+1.01悪化**。
- ＝richな今の特徴でも物理候補選択はblendに勝てない(datum両義)。答えは候補内(oracle3.94)だが選択器が届かない=v91/夜間結論の再確認。**物理候補は"脱相関脚"としてのみ価値**(v91 snap −0.066系譜)。

## 実験F: GR井戸間正規化 (CPU, ユーザー案/DR#3推奨) — ★較正ズレ実在, 但し相関不変
- GR較正(250井): **水平GR std = typewell std の0.50倍(振幅半分!)** / 平均比1.03。|平均差|中央8.9/std差14.1=較正ズレ実在。
- 真位置GR照合相関: raw0.676 / z正規化0.676(±0=相関はスケール不変) / 分位match0.647(悪化)。
- ★**較正ズレ(std0.5×)は本物→PF尤度は残差ベースなので正規化(振幅合わせ)で残差照合sharpness改善の余地**。相関テストでは見えない軸。**PF入力GRをtypewellスケール正規化して回す実験が本筋**(GPU空き待ち)。datum壁は破らないがfine精度の小利得期待。

## deep-research #3: ゼロから始めるROGII (2026-07-31) — ★現土台は正しい
- ①定式化=MD逐次ベイズ状態推定(PF, 多峰=複数候補) ②well-CV(井戸全行片側) ③**GR井戸間正規化/detrend(見落とされる土台)** ④DTWアライメントだが大域整合(loop closure, 単一経路信じない) ⑤GBM+工学特徴(FORCE2020表彰台) ⑥候補生成+選択が真の天井(v91一致)。
- ＝我々のPF+well-CV+GBMは正解。未活用レバー=GR正規化(実験Fで較正ズレ実証)。

---
## deep-research #4: PF代替アルゴリズム (2026-07-31) — ★脱相関脚+物理錨synergyで評価
検証済7finding(3-0). ショートリスト(脱相関脚価値順):
1. **グリッド/HMM格子(TVT離散化, max-marginal)**: TRANSFER強/物理高。厳密多峰・GPU超並列・物理錨=遷移bias。研究「PF上でなく主デコーダに」。=v48/v68(8.85実証)。
2. **大域DP整合(N-best DTW+層序cost/grid-HMM)**: 強/高。GR照合を1大域最適化、物理/tie-pointをcost項に。
3. **粒子平滑器(FFBSi/two-filter, dip Rao-Black化)**: 強/中。現fixed-lag PFの上位互換、未来条件付き大域整合、O(N)GPU可、実装最小(backward追加)。
4. **GM-EnKF**: 中-強/高。地質面realizationアンサンブル=錨と自然融合、±15枝明示保持。
- ★**caveat: どれもdatum方向壁(0.5AUC)は破らない**。改善=多峰表現(両枝/hedge活用)+物理prior融合=**脱相関脚**として。単体PF超えでなくblendで効く(ユーザーの読み通り)。
- 推奨優先: ③FFBSi(実装最小・脱相関脚)→①グリッド主デコーダ化+錨強化(v48/v68磨き)→④GM-EnKF(物理synergy最高・新規)。
- 物理錨(GRフリー錨)=①②④の物理prior項にそのまま入る。GPU空き次第の第一候補=③or①。

---
## 実験G: グリッドベイズフィルタ v95 (非PF, 物理項トグル) 2026-07-31
- δ格子(錨からのずれ) forward-backward + GR z正規化emission + dip連続遷移 → posterior-mean δ。完全非PF・決定論。
- **物理あり(grfree錨) RMSE=9.51(30井)** = struct9.72/v48 8.85水準。錨が事後を真位置近傍に集中。
- **物理なし(flat-hold純GR) RMSE=44.4** → ★診断: バグでなく多峰崩壊。**単点GR最良が真δ±8に入る率13.3%=GR単独で位置を87%外す**(真δは窓±70に100%収まるのに)。=「物理錨がなぜ必須か」の定量実証。
- → 物理あり grid filter=非PF脱相関脚候補。全773井でρ(gf,blend)+blend利得を評価中。

## 実験G続: グリッドフィルタ 脱相関脚評価 (全773井)
- 物理あり全773井 **RMSE=10.24**(100秒=速い, 非PF決定論)。|e|中央4.49。
- **★ρ(grid誤差, blend誤差)=0.571=genuinely脱相関**(E-UNet0.70/snap0.62並)。but nnls 4脚=5.9905(weight0.049, 利得ほぼ0)/固定weight全悪化。
- ＝**脱相関だが弱すぎ(10.24 vs PF 8.6)→脚として効かない**(E-UNet同型)。弱い理由=emissionがtw単独(PFはtw/self/nbr多表現+tuned)。
- **強化path=多表現emission化(v68 GF-MR)でtw+self+nbr→8.6級狙い→脱相関脚(ρ0.57)が効く可能性**。特徴価値も検証中。
- 物理なし=44(多峰崩壊, バグでない, GR単独13%)。

## 実験H: 多表現グリッドフィルタ (v68 GF-MR方式) 2026-07-31 — ★多表現naive積は死
- MR版バグ2件修正: ①build_smoother_inputsのgg/grフレーム不整合→生格子で統一 ②錨prior(δを0へ引く物理項)欠落→追加で tw=9.85(元と一致)。
- **多表現 naive積: tw,self,nbr=10.33 / +nbr_gr5=10.62 = tw単独9.85より悪化**。down-weight(REP_W0.2)でも9.88=改善なし。
- ＝**self/nbr表現はgrid emissionに価値を足さない**(tw+錨で飽和/雑音)。積は脆く1つの悪表現が真δを汚す。v68の学習emission(MLP重み付け)のみが多表現の道=大規模GPU建設・datum壁で天井。
- ★**grid filter結論: 単表現tw+物理錨=9.85-10.24が最良。非PF・決定論・脱相関ρ0.57・特徴として−0.029(base)。多表現強化は学習emission必須で費用対効果低**。
- 物理項の実証: 錨prior無し=25/44(多峰崩壊)→錨prior有り=9.85。**「物理項(GRフリー錨prior)がグリッドフィルタを機能させる本体」を定量確認**。

## 実験I: Viterbi(大域DP MAP) 2026-07-31 — ★グリッドと完全冗長
- グリッド機構でMODE=viterbi(max-product+backtrack)。物理あり150井=10.21≈posterior-mean10.24。
- **★ρ(viterbi, pmean)=1.0000=完全同一**。錨prior(sig9)が事後を単峰化→MAP経路=posterior-mean。
- ＝**グリッド系(pmean/Viterbi/MAP)は錨があると全て同一**。collapse方法の違いは無意味。研究「MAP vs max-marginal」の選択は、強い錨下では差が消える。

## ★非PF探索 総括 (2026-07-31)
研究#4ショートリスト(グリッド/Viterbi/FFBSi/GM-EnKF/大域最適化)を評価:
- **グリッドフィルタ(pmean)=非PFで唯一"効いた"**: 物理錨9.85-10.24, ρ0.57脱相関, 特徴−0.029(base)。物理項(錨prior)が本体(無し=多峰崩壊44)。
- **多表現naive積=死**(self/nbr汚染)。学習emission(v68)のみ道だが大規模・datum壁で天井。
- **Viterbi=グリッドと完全冗長(ρ1.0)**。FFBSi=ユーザー既知(精度出るが遅すぎ没)。
- **GM-EnKF/大域最適化=大規模建設で同じdatum壁天井**(EnKFはGR多峰でGaussian collapse、空間融合はv91の粗近傍限界)。
- ★結論: **非PFの実用解=グリッドフィルタ特徴(−0.029)一択**。step-changeでなくmodest脱相関特徴。datum壁が全非PF手法を天井づけ(研究caveat/夜間4実験と一致)。
