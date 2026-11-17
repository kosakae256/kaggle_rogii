# Kaggle ROGII コンペに関する会話記録

- 作成日時: 2026-07-31 08:46 JST
- 形式: Markdown
- 範囲: このチャット内のユーザーとアシスタントの会話

---

## User

kaggleのrogiiコンペについて確認してください。  
この問題に回答する場合、まずどのような方法が考えられますか？

---

## Assistant

## 確認した問題設定

対象は Kaggle の **「ROGII - Wellbore Geology Prediction」**です。

各水平坑井について、Prediction Start（PS）より前の区間では `TVT_input` が既知で、PSより後の長い区間の **TVT（True Vertical Thickness：地層内の深度座標）**を予測します。利用できる主な情報は次のとおりです。

- 水平坑井：`MD`, `X`, `Y`, `Z`, `GR`, `TVT_input`
- 対応するTypewell（垂直参照井）：`TVT`, `GR`, `Geology`
- 目的変数：PS以降の水平坑井の `TVT`
- 評価指標：予測対象全行に対する RMSE

`MD` は坑井に沿った距離、`Z` は空間的な高さ・深さですが、`TVT` は地層を基準にした座標なので、単純に `Z` から決まるわけではありません。Typewellの `GR–TVT` 曲線と、水平坑井で観測したGR波形を対応づけることが中心的な課題です。 ([kaggle.com](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/data?utm_source=chatgpt.com))

## 結論：最初に考えるべき方法

この問題は通常の行単位の表形式回帰というより、**地層座標を復元する時系列・逆問題**として扱うのが適切です。最初の本命は、以下の組み合わせです。

> **既知TVTによるアンカー + GR波形マッチング + 空間的な地層傾斜推定 + 残差GBDT**

### 1. アンカー型ベースライン

まず、PS直前の既知TVTを基準にします。

\[
T_{\mathrm{anchor}} = 	ext{PS直前の最後のTVT}
\]

予測対象はTVTそのものではなく、

\[
\Delta TVT_i = TVT_i - T_{\mathrm{anchor}}
\]

とします。こうすると井戸ごとの絶対的な深度オフセットを除き、PS以降に地層座標がどの程度上下するかだけを学習できます。

最初に作るべきベースラインは次の3つです。

1. **Anchor Hold**  
   PS以降をすべて最後の既知TVTで一定とする。

2. **局所線形外挿**  
   PS直前の数十点から `TVT ~ MD` または `TVT ~ Z` の傾きを推定する。

3. **減衰付き外挿**  
   PS直前の傾きをそのまま延長せず、坑井の先に進むほどゼロ方向に減衰させる。

このデータでは地層が比較的平坦な坑井も多いため、複雑なモデルが単純なAnchor Holdを下回る可能性があります。実際、公開分析でも単純なアンカー予測が強く、無制限の線形外挿は不安定になりやすいと報告されています。 ([kaggle.com](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/703374?utm_source=chatgpt.com))

## 2. GR波形のアラインメント

Typewellには既知の

\[
TVT ightarrow GR_{\mathrm{typewell}}
\]

という参照波形があります。一方、水平坑井では

\[
MD ightarrow GR_{\mathrm{horizontal}}
\]

が観測されています。

したがって、水平坑井のGR波形をTypewellのGR波形のどのTVT位置に対応させるかを推定します。

候補は次のとおりです。

### 相互相関・テンプレートマッチング

短い水平GRウィンドウをTypewell上でスライドさせ、正規化相互相関が最大になるTVT位置を候補にします。

利点は高速で実装が容易なことです。一方、似たGRパターンが複数の深度に存在すると、誤った位置に一致する「alias」が発生します。

### Dynamic Time Warping

水平坑井とTypewellのGR波形の伸縮を許容して対応づけます。地層の厚さや掘削方向によって、同じ地層パターンでも水平軸方向の長さが異なるため、単純な相互相関より適しています。

ただし、通常のDTWをそのまま使うとTVTが不自然に往復する可能性があります。以下の制約が必要です。

- `dTVT/dMD` の上限・下限
- TVTの急激なジャンプに対するペナルティ
- 2階差分による滑らかさ制約
- PS地点の既知TVTへの固定
- GR欠損区間ではマッチングコストを弱める

公開されている問題分析でも、DTW、cross-correlation、Tikhonov正則化付き逆解析が主要候補として挙げられています。 ([storage.googleapis.com](https://storage.googleapis.com/kaggle-forum-message-attachments/3459176/42957/competition_overview.html))

## 3. 状態空間モデル

TVTを坑井に沿って逐次推定する方法です。

状態を例えば

\[
s_i = (TVT_i,\ dTVT_i/dMD)
\]

として、次のようにモデル化します。

\[
TVT_{i+1}=TVT_i+v_i\Delta MD+\epsilon_i
\]

観測尤度には、候補TVTにおけるTypewell GRと水平坑井GRとの差を使います。

### Kalman Filter

GRとTVTの関係を局所線形化する場合に使用できます。高速ですが、GR波形には同じ値を取る深度が複数あるため、多峰性を扱いにくい問題があります。

### Particle Filter / Beam Search

各ステップで複数のTVT候補を保持します。似たGRパターンが複数深度に存在する場合でも、複数の仮説を維持できるため、この問題との相性が良いです。

評価関数の例は、

\[
C =
C_{\mathrm{GR}}
+\lambda_1 C_{\mathrm{slope}}
+\lambda_2 C_{\mathrm{curvature}}
+\lambda_3 C_{\mathrm{spatial}}
\]

です。

公開Notebook・DiscussionでもParticle Filter系のベースラインが試されており、GRを単一の正解位置ではなく、複数の候補を持つ尤度として扱う重要性が指摘されています。 ([kaggle.com](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/707613?utm_source=chatgpt.com))

## 4. 周辺坑井から地層面を推定する

この問題ではGRだけでなく、坑井の `X, Y, Z` も重要です。近隣の学習井から、地層の空間的な傾斜やTVT面を推定できます。

例えば、特定のMD位置または地層境界について、

\[
TVT = f(X,Y)
\]

あるいは、地層の構造座標を

\[
S = TVT + Z
\]

のように定義して、

\[
S=f(X,Y)
\]

を学習します。

候補モデルは以下です。

- k近傍井による距離加重平均
- 局所平面回帰
- RBF補間
- Gaussian Process
- Kriging
- CatBoost / LightGBMによる空間回帰

重要なのは、**距離だけでなく坑井の掘削方位を考慮すること**です。同じ地層傾斜でも、坑井が傾斜方向に掘られているか、走向方向に掘られているかで観測される `dTVT/dMD` が変わります。公式タスク資料でも、近隣井の地質傾斜と坑井方位が有用であることが説明されています。 ([github.com](https://github.com/vamseeachanta/kaggle-rogii-2026/blob/main/docs/task-brief.md?utm_source=chatgpt.com))

## 5. GBDTによる残差学習

LightGBM、CatBoost、XGBoostを単独で行ごとに適用するよりも、物理的・信号処理的な予測を先に作り、その誤差だけを学習する方が安全です。

\[
\widehat{TVT}
=
TVT_{\mathrm{base}}
+
f_{\mathrm{GBDT}}(	ext{features})
\]

`TVT_base` として以下を用意します。

- Anchor Hold
- 局所傾斜予測
- DTW予測
- Particle Filter予測
- 近隣坑井による空間予測

特徴量候補は次のとおりです。

- PSからの距離
- `ΔMD`, `ΔX`, `ΔY`, `ΔZ`
- inclination、azimuth
- `dZ/dMD`、曲率
- GR、GR移動平均、差分、局所分散
- GR欠損マスク、連続欠損長
- Typewellとの最良・第2候補マッチングコスト
- 候補TVT間のばらつき
- 近隣井までの距離
- DTW・PF・空間モデル間の予測差

モデル間の不一致は、不確実性の有力な代理変数になります。公開分析でも「物理的な候補軌道を作成し、GBDTで残差補正する」構成が採用されています。 ([pilkwangkim.github.io](https://pilkwangkim.github.io/posts/ROGII-Target-Free-Stratigraphic-Alignment-for-TVT/))

## 6. ニューラルネット

その後に検討する方法です。

- 1D CNN
- TCN
- BiLSTM / GRU
- Transformer Encoder
- Mixture Density Network
- Neural state-space model

入力系列として、

\[
[GR,\ GR_{\mathrm{mask}},\Delta X,\Delta Y,\Delta Z,
TVT_{\mathrm{input}},	ext{typewell embedding}]
\]

を使い、`ΔTVT` や `dTVT/dMD` を予測します。

ただし、各行を独立サンプルと考えるとデータ数は多く見えますが、独立な単位は坑井です。深層学習では過学習しやすいため、物理制約や補助タスクを入れるべきです。

補助タスクの例：

- 地層ラベル予測
- formation topの位置予測
- TVTの傾き予測
- GR再構成
- DTW候補の選択確率

公開されている方法整理でも、1D CNN、BiLSTM、Transformer、および古典的逆解析を初期値として使うハイブリッド構成が提案されています。 ([storage.googleapis.com](https://storage.googleapis.com/kaggle-forum-message-attachments/3459176/42957/competition_overview.html))

## 最重要：検証設計

このコンペではモデルそのもの以上にCV設計が重要です。

### 行単位のランダム分割は禁止

同じ坑井内の隣接行は非常に強く相関しています。ランダムに行を分けると、検証データのほぼ同じ位置が学習側に入り、スコアが過度に良くなります。

最低限、以下を使用します。

- `GroupKFold(group=well_id)`
- Typewellが共有されている場合は `group=typewell_id`
- より厳密にはXY座標をクラスタリングしたSpatial GroupKFold

さらに、各検証井について実際のPS位置を使って、それ以降のTVTを完全に隠して予測する必要があります。近隣井から特徴を作る際も、検証井の正解TVTを集計に混ぜてはいけません。公開分析でも、井戸単位または空間ブロック単位の検証が推奨されています。 ([storage.googleapis.com](https://storage.googleapis.com/kaggle-forum-message-attachments/3459176/42957/competition_overview.html))

## 推奨する開発順序

実務的には次の順番がよいです。

1. **GroupKFold評価器とAnchor Holdを作成**
2. **PS基準の `ΔTVT` 予測に変更**
3. **局所傾斜・近隣井による空間モデルを追加**
4. **GR相互相関または制約付きDTWを実装**
5. **Particle FilterまたはBeam Searchで軌道を安定化**
6. **各ベース予測を特徴量にしてCatBoostで残差補正**
7. **井戸ごとの信頼度に応じたアンサンブル**
8. **最後にTCNやTransformerを追加**

最初からTransformerだけで解くより、**Anchor／空間地質モデル／GRアラインメントの3系統を別々に作り、OOFで比較する**方が、この問題のどこに予測可能性があるかを把握しやすくなります。

現時点で最も有望な初期構成は、

\[
oxed{
	ext{Anchor}
+	ext{Spatial Dip}
+	ext{DTW or PF}
+	ext{CatBoost Residual}
}
\]

です。これは地質的整合性を保ちながら、モデルが系統的な補正を学習できる構成です。

公開NotebookやDiscussionで新しい手法・スコア更新を定期的に確認する監視も設定できます。

---

## User

わかった  
それはさておきで、予測対象井戸の事前区間が使えるという特性があって、現状それを活かせていません  
どういう方法であれば、活かせると思いますか？

---

## Assistant

はい。既知の事前区間は、単なる追加特徴ではなく、**予測対象井戸について得られる教師付きキャリブレーション区間**として使うべきです。

最も筋がよいのは、次の構造です。

\[
	ext{全井戸共通モデル}
+
	ext{対象井戸の事前区間から推定した井戸固有補正}
\]

## 1. Global model + 井戸固有補正

まず全学習井戸からグローバル予測を作ります。

\[
\widehat{TVT}^{global}_i
=
f(X_i,Y_i,Z_i,MD_i,GR_i,\ldots)
\]

次に、予測対象井戸の事前区間で残差を計算します。

\[
r_i
=
TVT_i-\widehat{TVT}^{global}_i
\]

この残差を、その井戸専用の小さなモデルで学習し、予測区間まで外挿します。

例えば、

\[
r_i
=
b_w
+
a_w(MD_i-MD_{PS})
+
c_w(Z_i-Z_{PS})
\]

として、

\[
\widehat{TVT}_i
=
\widehat{TVT}^{global}_i
+
\widehat b_w
+
\widehat a_w\Delta MD_i
+
\widehat c_w\Delta Z_i
\]

とします。

これは以下を対象井戸ごとに補正できます。

- TVTの絶対オフセット
- 局所的な地層傾斜
- グローバルモデルの系統誤差
- 座標系やTypewellとの差異

最初に試す方法としては、これが最も堅牢です。事前区間が短い場合があるため、通常の線形回帰ではなくRidge回帰などで強く正則化します。

さらに、

\[
	heta_w
=
(b_w,a_w,c_w,\ldots)
\]

を完全に自由に推定するのではなく、全井戸で得たパラメータ分布に縮約する階層モデルにすると安定します。

---

## 2. 疑似PSを使ったprefix条件付き学習

最も重要なのは学習データの作り方です。

各学習井戸について途中に仮想的なPSを設定し、

- 仮想PS以前：TVTを見せる
- 仮想PS以後：TVTを隠す
- 仮想PS以後を予測させる

という学習サンプルを大量に生成します。

つまり、通常の行単位回帰ではなく、

\[
p(TVT_{	ext{future}}
\mid
TVT_{	ext{prefix}},
X_{	ext{all}},
Y_{	ext{all}},
Z_{	ext{all}},
GR_{	ext{all}})
\]

を直接学習します。

これは本番と同じ条件です。

### 入力構造

事前区間では、

\[
[MD,X,Y,Z,GR,TVT,	ext{known mask}]
\]

予測区間では、

\[
[MD,X,Y,Z,GR,0,	ext{unknown mask}]
\]

を入力します。

モデル候補は次のとおりです。

- TCN
- GRU / LSTM
- Transformer Encoder–Decoder
- Temporal Fusion Transformer
- 1D CNN + attention

特に適しているのは、**prefix encoder + future decoder**です。

事前区間から井戸埋め込み \(z_w\) を作り、

\[
z_w
=
Encoder(	ext{prefix})
\]

予測区間では、

\[
\widehat{TVT}_i
=
Decoder(z_w,X_i,Y_i,Z_i,MD_i,GR_i)
\]

とします。

ここで \(z_w\) には、その井戸の地層傾斜、TVTオフセット、GR特性、局所地質などが圧縮されます。

---

## 3. Test-time adaptation

ニューラルネットやGBDTをそのまま固定して使うのではなく、対象井戸の既知区間で一部だけ再学習します。

ただし、モデル全体をfine-tuningすると過学習しやすいため、更新対象は限定します。

適切なのは以下です。

- 最終線形層だけ更新
- offsetとslopeだけ更新
- 小さなadapter層だけ更新
- 井戸固有embeddingだけ最適化
- FiLMのscaleとbiasだけ最適化

例えば、学習済みモデルを

\[
\widehat{TVT}_i=f_\phi(x_i,z_w)
\]

とし、グローバルパラメータ \(\phi\) は固定します。

対象井戸についてのみ、

\[
\widehat z_w
=
rg\min_z
\sum_{i\in prefix}
\left(
TVT_i-f_\phi(x_i,z)
ight)^2
+
\lambda\|z\|^2
\]

を解きます。

その後、同じ \(\widehat z_w\) を予測区間に適用します。

これは「対象井戸の事前区間を使って、対象井戸の潜在的な地質条件を逆推定する」方法です。

---

## 4. 状態空間モデル

事前区間の情報を逐次的に蓄積するなら、状態空間モデルも適しています。

潜在状態を例えば、

\[
s_i=
egin{bmatrix}
TVT_i\
dTVT/dMD_i\
	ext{local dip}_i
\end{bmatrix}
\]

とします。

既知区間ではTVTが観測されるため、Kalman FilterやParticle Filterによって状態を更新できます。

\[
p(s_{PS}\mid TVT_{1:PS},GR_{1:PS},XYZ_{1:PS})
\]

を求め、その事後分布を初期条件として予測区間へ伝播します。

\[
p(s_{PS+1:T}\mid s_{PS},GR_{PS+1:T},XYZ_{PS+1:T})
\]

この方法では、PS直前のTVT値だけでなく、

- TVTの傾き
- 傾きの変化
- 地層面の曲率
- GRとTVTの対応関係
- 推定値の不確実性

を引き継げます。

特に、予測区間でもGRが使用可能なら、GRを観測尤度としてParticle Filterを更新し続ける構成が有効です。

---

## 5. GR–TVT対応を対象井戸内で校正する

事前区間にはGRと正解TVTの両方があります。

したがって、対象井戸について、

\[
GR_{	ext{horizontal}}(TVT)
\]

を部分的に直接観測できます。これを利用して、TypewellのGR曲線との対応を校正できます。

事前区間で推定できるものは次のとおりです。

- Typewellに対するTVTオフセット
- GR信号の振幅差
- GRの平均・分散の違い
- TVT軸方向の伸縮率
- 最適なGR平滑化幅
- Typewellとの相関が高い深度範囲
- 使用すべきTypewell候補

例えば、Typewell側のTVTを

\[
TVT^{type}_i
=
lpha_w+eta_w TVT^{horizontal}_i
\]

と変換し、事前区間で \(lpha_w,eta_w\) を推定します。

その上で、予測区間のGRマッチングやDTWを行います。

重要なのは、**事前区間でマッチング手法自体を井戸ごとに校正してから未来区間に適用すること**です。

---

## 6. 類似井戸検索

事前区間を対象井戸のfingerprintとして使う方法です。

事前区間から以下の特徴を作ります。

- TVTの局所傾き、曲率
- `TVT - Z` の傾向
- `dTVT/dX`, `dTVT/dY`
- GRの平均、分散、自己相関
- GRをTVT軸に変換した波形
- 事前区間の地層ラベル分布
- グローバルモデルの残差系列

この特徴で類似する学習井戸を検索し、その井戸のPS後の挙動を利用します。

\[
\widehat{\Delta TVT}_w(t)
=
\sum_{k\in N(w)}
\omega_k
\Delta TVT_k(\operatorname{warp}_k(t))
\]

単純なkNNだけでなく、

- Siamese network
- contrastive learning
- metric learning
- prefix encoderによる埋め込み検索

も使えます。

地質的に近い井戸が反復的な挙動を持つ場合には強い方法です。

---

## 事前区間から作るべき具体的な特徴量

単に最後のTVTだけを使うのでは不十分です。複数の窓幅で以下を作るのがよいです。

### TVT軌道

- 最後の既知TVT
- 直近10、50、100、500点の `dTVT/dMD`
- Robust linear slope
- 2階差分
- TVTの局所分散
- 最大・最小傾き
- 傾きのトレンド
- PSから各点までの相対TVT

### ジオメトリとの関係

- `dTVT/dX`
- `dTVT/dY`
- `dTVT/dZ`
- `TVT ~ X + Y + Z` の局所回帰係数
- 坑井方位と推定地層傾斜の内積
- `TVT - Z` の局所傾向

### グローバルモデルの誤差

- prefix RMSE
- 平均残差
- 残差傾き
- 残差自己相関
- PS直前残差
- 複数モデルの残差差

### GR関連

- GRとTVTの相互相関
- GRのTVT軸上での勾配
- Typewellとの最大相関位置
- 最大相関と第2候補との差
- DTWコスト
- GR欠損率
- GR振幅の井戸固有補正係数

---

## 推奨する実装順序

現実的には、次の3段階がよいです。

### 第1段階：強いベースライン

\[
oxed{
	ext{Global CatBoost}
+
	ext{prefix残差のRidge補正}
}
\]

補正パラメータは、まず以下だけに絞ります。

\[
r_i=b_w+a_w\Delta MD_i+c_w\Delta Z_i
\]

低コストで、事前区間の効果を直接測れます。

### 第2段階：疑似PS学習

学習井戸を複数の位置で切り、

- prefixの集約特徴
- PS以降の行特徴
- global prediction
- prefix残差モデル

をCatBoostまたはLightGBMに入力します。

1行ごとの入力でも、

\[
	ext{prefix summary features}
\]

を全予測行に付加すれば、かなりの部分を実現できます。

### 第3段階：prefix encoder

TCNやTransformerで既知区間全体をエンコードし、井戸embeddingを作ります。そのembeddingを予測区間の各行に付与します。

---

## 検証上の注意

検証時にも、必ず本番と同じ処理を再現します。

各検証井戸について、

1. 仮想PSを決める
2. PS以前のTVTだけを渡す
3. PS以後のTVTを完全に隠す
4. prefixを使った校正を実行
5. suffixを予測する

という手順にします。

特に、対象井戸の全区間を使ってスケーリングや特徴量生成を行う際は、TVTに関する統計がPS後から混入しないようにします。一方で、予測時に利用可能な `X, Y, Z, MD, GR` は、ルール上使えるなら全区間を使って構いません。

## 最も推奨する形

最初に実装するなら、次です。

\[
oxed{
\widehat{TVT}_{w,i}
=
f_{\mathrm{global}}(x_{w,i})
+
\phi(x_{w,i})^	op\widehat{	heta}_w
}
\]

ここで、

\[
\widehat{	heta}_w
=
rg\min_	heta
\sum_{i\in prefix}
\left[
TVT_{w,i}
-
f_{\mathrm{global}}(x_{w,i})
-
\phi(x_{w,i})^	op	heta
ight]^2
+
\lambda\|	heta\|^2
\]

です。

これは、**対象井戸の事前区間を使った正則化付きtest-time calibration**です。

この方法で効果が確認できた後に、\(	heta_w\) を固定的なoffset・slopeからニューラルな井戸embeddingへ発展させるのが安全です。

---

## User

RMSE4.xに至るための方法は？

---

## User

(4.xは最上位なので、あなたの知識が必要です)

---

## Assistant

RMSE 4.xを狙う場合、前回の「prefix残差をRidge補正する」程度では不足します。必要なのは、**予測対象井戸の既知区間を利用して、その井戸専用に候補軌道の生成・評価・アンサンブルを最適化すること**です。

なお、検索で確認できる公開情報では、2026年7月末時点の公開首位として言及されている値は5.262です。したがって、以下は公開された4.x解法の再現ではなく、4.xに到達するための技術的に妥当な設計仮説です。([kaggle.com](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/718670?utm_source=chatgpt.com))

## 結論

私なら次の構成にします。

\[
oxed{
	ext{Target-well self template}
+
	ext{global trajectory generator}
+
	ext{candidate ranker}
+
	ext{rolling pseudo-PS adaptation}
+
	ext{risk-controlled ensemble}
}
\]

特に重要なのは、**TVTを直接回帰しないこと**です。

この問題を、

1. 正解に近い軌道候補を大量に生成する
2. その中から対象井戸の既知区間を使って正しい候補を選ぶ

という二段階問題に分解します。

公開Discussionでも、500粒子程度あれば検証井戸の90%についてRMSE 4未満の候補が少なくとも複数含まれる一方、問題はその候補を正しく順位づけできないことだと報告されています。つまり、4.xへの主要ボトルネックは候補生成よりも**candidate ranking**である可能性が高いです。([kaggle.com](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/707613?utm_source=chatgpt.com))

---

# 1. 既知区間から「対象井戸専用GRテンプレート」を作る

これが第一の核心です。

通常はTypewellの

\[
GR_{\mathrm{type}}(TVT)
\]

と水平井のGRを比較します。しかし予測対象井戸の既知区間では、

\[
(TVT_i,GR_i)
\]

の対応が直接観測されています。

したがって、対象井戸専用の

\[
p(GR\mid TVT,	ext{direction},	ext{local state})
\]

を構築できます。

主催者資料でも、水平井のPS以前のGRが、TypewellのGRよりも同じ水平井のPS以後のGRに強く対応する場合があり、PS以前の水平井GRを利用することが推奨されています。([github.com](https://github.com/vamseeachanta/kaggle-rogii-2026/blob/main/docs/task-brief.md))

## 作り方

既知区間をTVT軸へ投影し、TVTごとに以下を保存します。

- GR中央値
- GR分散
- GR一次・二次微分
- TVT増加時と減少時の別テンプレート
- 複数回同じTVTを通過した場合の分布
- 局所ウィンドウの波形embedding
- GR欠損率と信頼度

単純には、

\[
\hat g_w(t)
=
rac{
\sum_{i\in prefix}K_h(t-TVT_i)GR_i
}{
\sum_{i\in prefix}K_h(t-TVT_i)
}
\]

というカーネル回帰で作れます。

ただし、GR波形は方向依存・解像度依存があるため、実際には

\[
\hat g_w(t,d)
\]

として、`dTVT/dMD > 0` と `< 0` を分けます。

## Typewellとの統合

prefixで観測済みのTVT範囲では対象井戸テンプレートを優先し、未観測範囲ではTypewellへ滑らかに切り替えます。

\[
g_{\mathrm{ref}}(t)
=
lpha(t)\hat g_w(t)
+
[1-lpha(t)]g_{\mathrm{type}}(t)
\]

\(lpha(t)\) は以下で決めます。

- prefix内の観測密度
- 同じTVTを通過した回数
- GRの再現性
- Typewellとのprefix相関
- TVT範囲外までの距離

4.xを狙うなら、Typewellだけでなく、**test well自身をTypewell化する**必要があります。

---

# 2. 予測対象を低次元の軌道にする

行ごとにTVTを予測すると、ノイズ・ドリフト・不連続が発生します。

TVT軌道を、PS地点で固定された低次元曲線として表します。

\[
T(s)
=
T_{PS}
+
m_0s
+
\sum_{k=1}^{K}eta_kB_k(s)
+
\sum_jJ_jH(s-	au_j)
\]

ここで、

- \(s=MD-MD_{PS}\)
- \(m_0\)：PS地点の初期傾斜
- \(B_k\)：低周波B-spline基底
- \(eta_k\)：緩やかなdip変化
- \(J_j\)：断層・急変候補
- \(	au_j\)：急変位置

です。

通常候補では \(K=3\)〜6程度で十分です。

## 候補ファミリー

少なくとも以下を別々に生成します。

### A. 安全な幾何候補

- constant TVT
- prefix末尾の傾斜継続
- 傾斜減衰
- quadratic trajectory
- cubic spline trajectory

### B. 空間地質候補

周辺井戸から、

- dip magnitude
- dip azimuth
- curvature
- fault probability

を推定します。

主催者資料でも、地質dipは近隣井戸間で似ること、掘削azimuthによって観測されるTVT変化が変わることが明示されています。([github.com](https://github.com/vamseeachanta/kaggle-rogii-2026/blob/main/docs/task-brief.md))

地層面の勾配を \(oldsymbol g=(g_x,g_y)\)、坑井の水平進行方向を \(oldsymbol v=(dX/dMD,dY/dMD)\) とすると、TVT傾斜の空間成分は概ね

\[
rac{dTVT}{dMD}
pprox
rac{dZ}{dMD}
-
oldsymbol g^	opoldsymbol v
\]

で表せます。

これをcandidate priorにします。

### C. GRアラインメント候補

各候補軌道 \(T_j(s)\) について、

\[
GR_{\mathrm{pred},j}(s)
=
g_{\mathrm{ref}}(T_j(s))
\]

を生成し、実際の水平井GRと比較します。

### D. 断層・regime change候補

全井戸を滑らかな軌道だけで扱うと、少数の急変井戸で大きな誤差が出ます。

したがって、

- 断層なし
- 1回のjump
- slope change
- sigmoid型変化

を候補集合へ明示的に追加します。

ただし、断層候補の事前確率は低くします。

---

# 3. Sequential PFではなく、全suffixを使うglobal optimization

Particle Filterを逐次的に動かすだけでは、序盤で誤ったTVTへ移動すると、その誤差が最後まで残ります。実際、逐次モデルが早期にドリフトし、井戸全体で40以上のRMSEを生じる例が報告されています。([kaggle.com](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/723815?utm_source=chatgpt.com))

予測時にはPS以後のGR系列全体が利用可能なので、完全なonline filterである必要はありません。

以下を使います。

- beam search
- dynamic programming
- cross-entropy method
- CMA-ES
- Sobol sampling + local refinement
- particle smoother
- Viterbi decoding
- differentiable spline optimization

目的関数は例えば、

\[
egin{aligned}
L(T)=&
\lambda_{\mathrm{self}}L_{\mathrm{selfGR}}
+
\lambda_{\mathrm{type}}L_{\mathrm{typeGR}}\
&+
\lambda_{\mathrm{spatial}}L_{\mathrm{dip}}
+
\lambda_{\mathrm{smooth}}L_{\mathrm{curvature}}\
&+
\lambda_{\mathrm{prefix}}L_{\mathrm{boundary}}
+
\lambda_{\mathrm{prior}}L_{\mathrm{trajectory}}
\end{aligned}
\]

です。

GR損失は単純MSEではなく、複数の指標を使います。

- 正規化相互相関
- derivative correlation
- multi-scale wavelet correlation
- soft-DTW
- 局所順位相関
- formation transition一致度
- robust Huber loss

候補を最低でも数千本生成し、その後で100〜300本程度に絞ります。

---

# 4. Candidate rankerを学習する

4.xへの最重要部分です。

候補をGR誤差が小さい順に並べても、正解TVT順にはなりません。似たGRパターンが異なるTVTに存在するためです。

そこで、各候補軌道について特徴量を作り、**その候補の将来TVT RMSE**を予測します。

## 学習サンプル

学習井戸ごとに複数の疑似PSを作ります。

疑似PS以前だけを既知とし、疑似PS以後に候補を生成します。

候補 \(j\) について、

\[
y_{wj}
=
RMSE(T_{wj},T_w^{true})
\]

をラベルにします。

## Candidate-level特徴量

### GR整合性

- self-template GR loss
- Typewell GR loss
- 複数scaleの相互相関
- 微分GR correlation
- 最良局所lag
- GR一致区間率
- GR欠損区間率
- self-templateとTypewellの評価差

### 軌道prior

- 初期傾斜
- 平均傾斜
- 最大傾斜
- 曲率
- total variation
- endpoint TVT
- jump数とjump幅
- prefix末尾との接続誤差

### 空間情報

- 近隣井戸dipとの差
- 近隣井戸の予測分散
- 同じTypewellの井戸との差
- azimuth補正後の傾斜差
- 最近傍井戸距離

### 候補集合内の情報

- 候補密度
- 所属clusterのサイズ
- posterior modeまでの距離
- top候補間のTVT分散
- GR score順位
- spatial score順位
- 複数generatorから生成されたか

### prefixから得る井戸信頼度

- TypewellとprefixのGR相関
- self-templateの再現誤差
- prefixのTVT折り返し回数
- prefix長
- pseudo-PS backtest誤差
- 過去区間で最良だったgenerator
- 過去区間におけるrankerのselection regret

## モデル

単純な回帰より、ranking modelが適しています。

- LightGBM LambdaRank
- XGBoost `rank:pairwise`
- CatBoost ranking
- pairwise neural ranker

候補RMSEそのものを予測するモデルと、候補間のpairwise preferenceを学習するモデルをブレンドします。

最終的な重要指標は、モデルRMSEではなく、

\[
	ext{selection regret}
=
RMSE_{\mathrm{selected}}
-
RMSE_{\mathrm{oracle}}
\]

です。

---

# 5. 対象井戸内でrolling pseudo-PSを行う

ここが、既知区間を最も強く活用する方法です。

対象テスト井戸の既知区間の中に、さらに複数の仮想PSを設定します。

例えば既知区間が十分長ければ、

\[
q\in\{40\%,55\%,70\%,85\%\}
\]

地点を疑似PSにします。

各疑似PSについて、

1. \(q\)以前のみを使う
2. \(q\)以後の既知TVTを隠す
3. 本番と同じ候補生成を行う
4. 隠した既知TVTで実際の候補RMSEを計算する
5. その井戸で有効なgenerator・GR尺度・prior weightを評価する

これにより、対象井戸自身で以下を決定できます。

- self-GRとTypewellのどちらを重視するか
- GRをどのscaleで比較するか
- slope extrapolationが有効か
- spatial priorが有効か
- Particle Filterがドリフトしやすいか
- 断層候補を許可するか
- rankerの温度
- top候補を単独選択するか平均するか

## 井戸固有weightの推定

候補scoreを

\[
S_j
=
w_1S_{\mathrm{selfGR},j}
+
w_2S_{\mathrm{typeGR},j}
+
w_3S_{\mathrm{spatial},j}
+
w_4S_{\mathrm{smooth},j}
+
w_5S_{\mathrm{ranker},j}
\]

とし、対象井戸の疑似PSで

\[
\hat{oldsymbol w}_w
=
rg\min_{oldsymbol w}
\sum_q
RMSE_q(oldsymbol w)^2
+
\lambda
\|oldsymbol w-oldsymbol w_{\mathrm{global}}\|^2
\]

を解きます。

これが単なるtest-time fittingと違う点は、**prefixへのin-sample適合ではなく、prefix内の未来予測性能でパラメータを選ぶ**ことです。

私は4.xに必要な最大の差分は、これだと考えます。

---

# 6. 単独選択ではなく、posterior trajectory ensemble

候補rankerが完全でない以上、1本を選ぶと誤選択時の損失が大きくなります。

候補を軌道形状でcluster化します。

- flat cluster
- positive dip cluster
- negative dip cluster
- curved cluster
- fault cluster

各clusterの確率を求め、

\[
\hat T(s)
=
\sum_c p(c\mid D)
\sum_{j\in c}p(j\mid c,D)T_j(s)
\]

とします。

ただし異なるmodeを単純平均すると地質的に存在しない中間軌道になるため、以下を使い分けます。

- 単峰性：posterior mean
- 弱い二峰性：trimmed mean
- 強い二峰性：最良modeを選択
- 低信頼：安全priorとのshrinkage

\[
\hat T_{\mathrm{final}}
=
\gamma_w\hat T_{\mathrm{candidate}}
+
(1-\gamma_w)\hat T_{\mathrm{safe}}
\]

\(\gamma_w\) は対象井戸のrolling pseudo-PS結果から決めます。

---

# 7. 4.xにはcatastrophic well対策が不可欠

RMSE 4.xでは、平均的な井戸を少し改善するより、RMSE 30〜50の井戸をなくす方が重要です。

例えば95%の行がRMSE 3.5でも、残り5%がRMSE 20なら、全体RMSEは約5.7です。

したがって、次のconfidence gateを設けます。

## 低信頼判定

- top候補と2位のscore差が小さい
- candidate trajectoryが複数modeに分離
- rolling pseudo-PSのselection regretが大きい
- self-GR templateが再現しない
- Typewellとprefixの相関が低い
- 最近傍井戸が遠い
- GR欠損が多い
- GR、spatial、prefix slopeの結論が一致しない
- 予測末端で候補分散が急拡大

## 低信頼時の動作

- aggressive GR alignmentを弱める
- jump候補を抑制
- top 1ではなくcluster平均
- prefix slopeを徐々に減衰
- spatial priorへshrink
- 予測距離に応じてconfidenceを低下
- TVT変化量を学習分布のquantile envelope内に制限

公開されているPF系の改善例でも、GR候補へ全面的に置換するのではなく、clipされた小さな補正として利用する方針が示されています。([kaggle.com](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/722041?utm_source=chatgpt.com))

---

# 8. CVの設計

4.xを目指すならCVは以下の二重構造にします。

## Outer CV

検証井戸を完全に学習対象外にします。

- well group split
- Typewell group split
- spatial block split
- coordinate-frame split

複数CVを保持します。

## Inner test-time adaptation

Outer validation井戸については、本番と同じようにprefix TVTを与えます。

そのprefix内でrolling pseudo-PSを行い、井戸固有のweightを学習します。

つまり検証手順自体を、

\[
	ext{unseen well}
ightarrow
	ext{prefix observation}
ightarrow
	ext{self adaptation}
ightarrow
	ext{suffix prediction}
\]

にします。

## 必ず記録する指標

全体RMSEだけでは不足します。

- row-weighted RMSE
- well-median RMSE
- worst 1%、5%、10% RMSE
- oracle@K RMSE
- selected RMSE
- selection regret
- candidate recall at RMSE < 2, 4, 6
- prediction horizon別RMSE
- prefix length別RMSE
- GR reliability別RMSE
- nearest-neighbor distance別RMSE

目標はまず、

\[
RMSE_{\mathrm{oracle@100}}<3
\]

程度まで候補生成を改善し、その後、

\[
	ext{selection regret}<2
\]

を目指すことです。

---

# 実装するモデルの具体形

私なら次の4系統を構築します。

## Model A：Safe spatial trajectory

- prefix末尾のTVT、傾斜、曲率
- 周辺井戸のdip
- well azimuth
- robust spline
- GRは使用しないか弱く使用

役割：低信頼井戸のfallback。

## Model B：Self-GR global aligner

- 対象井戸prefixからGR–TVTテンプレート作成
- suffix全体をglobal spline optimization
- multi-scale soft-DTW
- 方向別GR emission

役割：同じ地層を再通過する井戸。

## Model C：Typewell + spatial aligner

- Typewell GR
- 周辺井戸dip
- formation境界
- global beam search

役割：prefix外のTVTへ進む井戸。

## Model D：Particle/spline proposal generator

- 数千候補を生成
- trajectory rankerで評価
- rolling pseudo-PSでgenerator weightを校正

役割：非線形・多峰性のカバー。

最終予測は、

\[
\hat T
=
w_A T_A
+
w_B T_B
+
w_C T_C
+
w_D T_D
\]

ですが、\(w\)は全井戸共通ではなく、**対象井戸のrolling pseudo-PSで推定**します。

---

# 到達順序

想定される性能段階は次です。数値は保証ではなく、設計上の目安です。

| 段階 | 主な構成 | 想定される壁 |
|---|---|---|
| 7〜9 | PF、Ridge、prefix slope | ドリフトとGR alias |
| 6〜7 | spatial prior、複数candidate | candidate選択 |
| 5〜6 | candidate ranker、global smoothing | test wellごとの差 |
| 4.x | self-GR template、rolling pseudo-PS adaptation、risk controller | catastrophic wellの排除 |

## 一番優先すべき実験

最初に実装すべきものはニューラルネットではありません。

\[
oxed{
	ext{対象井戸内rolling pseudo-PSによるモデル選択}
}
\]

です。

具体的には既存の予測器を5〜10種類用意し、対象井戸のprefix内部で各予測器をbacktestします。そして、その井戸で実際に良かった予測器・ハイパーパラメータだけを本番suffixに適用します。

これだけで、

- flat型井戸
- slope継続型井戸
- GR追跡可能型井戸
- Typewell不一致型井戸
- spatial prior依存型井戸

を対象井戸ごとに分離できます。

4.xに至る解法の中心は、より巨大なモデルではなく、

> **既知区間を使って、対象井戸について「何を信じるべきか」を学習すること**

だと考えます。

---

## User

この会話記録を全てmdふぁいるにしておいてください。
