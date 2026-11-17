現在の数値から見ると、PFを強化する目的はさらに極端な候補を増やすことではなく、良い軌道にposterior massを集めることです。

oracle 2.5：良い軌道を生成する能力はある
median 6.5：生成分布の中心が正解からずれている
ensemble 6.0：PFの尤度・重みが「良い軌道」を十分識別できていない

したがって、目標はoracleを2.5から2.0へ下げることより、

median:6.5→5.x

にすることです。

1. 最初に「PFのどこで失敗しているか」を分解する

OOF井戸で、各ステップまたは各ブロックについて次を記録してください。

Proposal recall

真のTVTが粒子集合の範囲内に存在するか。

r
t
	​

=1[
i
min
	​

∣T
t
(i)
	​

−T
t
true
	​

∣<δ]
Oracle-particle survival

正解に近い粒子が途中まで存在したのに、resamplingで消えていないか。

Good-mode posterior mass
M
t
	​

(δ)=
i
∑
	​

w
t
(i)
	​

1[∣T
t
(i)
	​

−T
t
true
	​

∣<δ]
Ancestor diversity

現在の粒子が何個の異なる祖先から来ているか。

内部oracleとPF出力の差

各PF内部で、

最良粒子
最大重み粒子
weighted mean
weighted median
smoothed posterior mean

を比較します。

結果の解釈は明確です。

正解近傍の粒子が最初から存在しない
→ proposal／transition modelを改善
存在したが途中で消える
→ resampling／tempering／mode維持を改善
最後まで存在するが重みが低い
→ GR likelihoodを改善
高い重みを持つが最終出力が悪い
→ smoothing／posterior summaryを改善

この診断なしに粒子数を増やしても、同じ失敗をより高コストで繰り返す可能性があります。

2. Bootstrap PFから「look-ahead block auxiliary PF」へ変更する

最も優先度が高い改善です。

通常のbootstrap PFは、

dynamicsから粒子を生成
現在点のGRで評価

という順序です。しかしROGIIではGRのaliasが多く、現在点のGRだけでは正しいTVTを識別しにくいはずです。

そこで、

q(T
t
	​

∣T
t−1
	​

,GR
t:t+L
	​

)

のように、未来の短いGR区間を見てから粒子を提案します。

Auxiliary Particle Filterは、観測に適合しそうな祖先を先に評価してから粒子を生成する設計で、bootstrap proposalよりもinformative observationに対応しやすい方法です。

実装方法

まず、水平井のGRウィンドウと参照GRとの類似度マップを作ります。

E(t,u)=similarity(GR
t:t+L
horizontal
	​

,GR
u−Δ:u+Δ
reference
	​

)

各粒子について、transition prior付近にある E(t,u) の上位ピークを探します。

proposalを次の混合分布にします。

q(u)=λq
dynamics
	​

(u)+(1−λ)
m=1
∑
M
	​

π
m
	​

N(u;μ
m
	​

,σ
m
2
	​

)
q
dynamics
	​

：従来のTVT／傾斜モデル
μ
m
	​

：GR類似度の局所ピーク
M：2〜5程度
λ：0.2〜0.6程度からCV

重要なのは、GR最良ピークだけを提案しないことです。alias候補を複数残します。

行単位ではなくブロック単位

1点ずつ更新するのではなく、例えば16〜128行程度を1ブロックとして、

T
t+k
	​

=T
t
	​

+v
t
	​

k+
2
1
	​

a
t
	​

k
2

の軌道を提案します。

そしてブロック全体のGRで評価します。

行単位PFは、一時的にGRが一致しない区間を越えられず、正しいmodeへ移動できないことがあります。ブロックproposalなら、局所的な不一致を許容しながら波形全体を評価できます。

3. FilteringではなくParticle Smoothingを使う

予測時にはPS以後のGRとXYZが全区間利用できます。つまり、リアルタイムfilteringに限定する必要はありません。ROGIIのタスクでは水平井のGR・XYZが予測区間にも提供され、TVTだけがPS以後で未知です。

通常のPFでは、時点 t の推定に 1:t の観測しか使いません。

p(T
t
	​

∣GR
1:t
	​

)

しかし必要なのは、

p(T
t
	​

∣GR
1:T
	​

)

です。

Forward Filtering Backward Simulation

forward PFで粒子と祖先を保存し、最後から逆向きに軌道をsampleします。

p(x
t
(i)
	​

∣x
t+1
	​

,y
1:T
	​

)∝w
t
(i)
	​

p(x
t+1
	​

∣x
t
(i)
	​

)

これにより、予測序盤で誤ったGR aliasへ入っても、後半のGR情報から修正できます。Particle smoothingは過去・現在・未来の観測で潜在軌道を推定する枠組みであり、forward-backward型の線形計算量近似も研究されています。

実装後は、backward trajectoryを例えば100〜500本sampleし、

smoothed posterior mean
smoothed median
smoothed MAP mode
mode別posterior mean

を出力します。

RMSEに対するBayes estimatorはposterior meanなので、最大重み粒子よりも、

T
t
	​

=
k
∑
	​

w
k
	​

T
t
(k)
	​


を基本にすべきです。

ただし明確に離れた複数modeがある場合は、modeごとの平均を保持します。

4. Likelihood temperingで一度のweight collapseを防ぐ

GRの観測尤度が鋭すぎると、1ブロックの更新だけで少数粒子に重みが集中します。

通常の更新、

w
i
′
	​

∝w
i
	​

L
i
	​


を一度に行わず、

w
i
(k)
	​

∝w
i
(k−1)
	​

L
i
Δβ
k
	​

	​


として、

0=β
0
	​

<β
1
	​

<⋯<β
K
	​

=1

と段階的に観測を取り込みます。

各 Δβ
k
	​

 は、更新後のESSが例えば粒子数の50〜70%になるように二分探索します。

Tempered Particle Filteringは、尤度を段階的に導入し、Monte Carlo誤差と粒子退化を抑えるための自己調整型SMCとして提案されています。

ROGIIでは、特に次の区間だけtempering回数を増やす設計がよいです。

GRピークが急峻
複数のTVT候補で似たGRが出る
ESSが突然低下
prefix slopeとGR候補が矛盾
Typewellとself-templateが矛盾
5. Resample-moveで粒子を復活させる

resampling後は同じ粒子の複製が大量にでき、粒子多様性が失われます。Gaussian jitterだけでは、同じmode付近を少し散らすだけです。

resampling後に、posteriorを保存するMCMC moveを入れます。

候補moveは以下です。

TVT全体のvertical shift
slope変更
curvature変更
ブロックcontrol point変更
GR alignment lag変更
jumpの追加・削除
jump位置の移動
smooth regimeとfault regimeの切り替え

例えば、

T
t:t+L
′
	​

=T
t:t+L
	​

+δ
0
	​

+δ
1
	​

s+δ
2
	​

s
2

を提案し、Metropolis-Hastingsで受理します。

α=min(1,
p(T
t:t+L
	​

∣D)q(T
′
∣T)
p(T
t:t+L
′
	​

∣D)q(T∣T
′
)
	​

)

Resample-moveは、逐次importance samplingで生じる粒子退化をMCMC moveによって緩和する方法です。

実務上は、ESS低下時だけ1〜3回moveを実行すれば十分です。

6. Mode-preserving resampling

通常のsystematic resamplingでは、少し重みが劣るmodeが完全に消えます。しかしGR alias問題では、序盤の劣勢modeが後半で正解になることがあります。

粒子を以下でcluster化します。

z
i
	​

=(T
i
	​

,v
i
	​

,T
i
endpoint
	​

,regime)

そしてclusterごとにresamplingします。

最低粒子数を保証する

例えば各有力modeに最低でも、

N
min
	​

=max(10,0.02N)

程度を残します。

残りをmode posteriorに比例して配分します。

N
c
	​

=N
min
	​

+(N−CN
min
	​

)p(c∣D)

これにより、1つのGRピークへの早期collapseを防ぎます。

より実装しやすいのは、30PFを独立モデルとして走らせるのではなく、次のisland PFに変える方法です。

flat island
positive slope island
negative slope island
high-curvature island
self-GR island
Typewell-GR island
fault/jump island

各island内ではresamplingし、一定間隔で一部の粒子だけ交換します。

これは30本を単なる独立出力ではなく、意図的に異なるposterior modeを維持する粒子群として使う設計です。

7. GR likelihoodを作り直す

pointwiseなGR誤差、

L∝exp(−
2σ
2
(GR−
GR
)
2
	​

)

だけでは、おそらく不十分です。

GRの振幅オフセット、スケール差、外れ値、局所的な変形を吸収できないためです。

推奨するblock likelihood
logL=λ
1
	​

S
NCC
	​

+λ
2
	​

S
derivative
	​

+λ
3
	​

S
wavelet
	​

−λ
4
	​

D
amplitude
	​

−λ
5
	​

D
warp
	​

構成要素
z-normalized cross correlation
一階差分GRの相関
複数スケールの平滑化GR相関
wavelet bandごとの相関
soft-DTW
robust amplitude residual
formation transition位置の一致
欠損マスク

ただし各点を独立観測として掛け合わせると、相関したGR値を過剰に数えることになります。

例えば64点のblockでも、実効観測数を、

n
eff
	​

=
1+2∑
k=1
K
	​

ρ
k
	​

n
	​


のように補正し、likelihoodを過度に鋭くしない方がよいです。

Student-tまたはoutlier mixture
p(r)=(1−ϵ)t
ν
	​

(r;0,σ)+ϵU(r)

とすると、局所的なGR異常が軌道全体を破壊しにくくなります。

8. 対象井戸自身のGR emissionを使う

prefixにはTVTとGRの対応が存在します。

(TVT
t
	​

,GR
t
	​

),t<PS

したがって、Typewellだけでなく、

p(GR∣TVT,target well)

を推定できます。

例えば、

g
self
	​

(u)=
∑
t<PS
	​

K
h
	​

(u−TVT
t
	​

)
∑
t<PS
	​

K
h
	​

(u−TVT
t
	​

)GR
t
	​

	​


を作ります。

最終emissionは、

g
ref
	​

(u)=α(u)g
self
	​

(u)+[1−α(u)]g
typewell
	​

(u)

とします。

α(u) は、

prefix内のTVT coverage
同じTVTを複数回通過しているか
self-templateの再現誤差
Typewellとprefix GRの一致度
prefixのTVT範囲からの距離

で決めます。

さらにTypewell GRを対象井戸へ変換します。

GR
horizontal
	​

≈a
w
	​

+b
w
	​

GR
typewell
	​


必要なら単調splineやquantile mappingを使います。

この a
w
	​

,b
w
	​

 をPF状態に含めるより、まずprefixで推定して固定する方が粒子次元を抑えられます。

9. DynamicsをGaussian random walkからregime mixtureへ変える

単一の、

v
t+1
	​

=v
t
	​

+ϵ

では、滑らかなdip変化と断層的な変化を同時に表現しにくいです。

離散regimeを導入します。

r
t
	​

∈{flat,smooth-up,smooth-down,turn,jump}
p(r
t
	​

∣r
t−1
	​

)

を持つSwitching Particle Filterにします。

各regimeでtransitionを変えます。

Smooth
v
t+1
	​

=ρv
t
	​

+ϵ
v
	​

Turn
a
t+1
	​

=ρ
a
	​

a
t
	​

+ϵ
a
	​

Jump
T
t+1
	​

=T
t
	​

+J
t
	​

+v
t
	​

ΔMD

jump regimeには低いprior probabilityを与えます。

重要なのは、jumpをGaussian process noiseの巨大値として表現しないことです。通常区間のノイズを大きくしすぎて、PF全体が不安定になります。

10. 状態を低次元化する

PF状態を各行のTVTにすると、長いsuffixでpath degeneracyが起きやすくなります。

ブロックごとに、

x
b
	​

=(T
b
	​

,v
b
	​

,a
b
	​

,r
b
	​

)

だけを持ち、ブロック内TVTは決定的に生成します。

または、全suffixを少数のB-spline control pointで表します。

T(s)=T
PS
	​

+
k=1
∑
K
	​

β
k
	​

B
k
	​

(s)

粒子は β
k
	​

 を持ちます。

ただし全区間のcontrol pointを一度にsampleすると次元が高くなるため、

逐次的にcontrol pointを追加
blockごとに1〜2個追加
resample-moveで過去control pointも微調整

という方法がよいです。

11. Rao–Blackwellized PF

状態の一部が条件付きで線形Gaussianに近いなら、すべてを粒子化しない方がよいです。

例えば、

非線形・離散：GR alias、formation、jump regime
線形連続：TVT offset、slope、curvature

と分けます。

粒子はalias／regimeだけを持ち、各粒子内部でKalman Filterによりoffset・slopeを更新します。

p(r
1:t
	​

∣y
1:t
	​

)

を粒子で表現し、

p(T
t
	​

,v
t
	​

∣r
1:t
	​

,y
1:t
	​

)

をGaussianで表現します。

Rao–Blackwellizationは、条件付き線形Gaussian成分を解析的に扱うことで、同じ粒子数でMonte Carlo分散を抑えられます。Backward simulationと組み合わせたRao–Blackwellized smootherも提案されています。

12. Test-wellごとにPFハイパーパラメータを校正する

30PFから選択するのではなく、prefix内pseudo-PSを使ってPFそのものを校正します。

最適化対象は、

θ
w
	​

=(σ
v
	​

,σ
a
	​

,σ
GR
	​

,λ
self
	​

,L
lookahead
	​

,ESS
threshold
	​

,p
jump
	​

)

です。

対象井戸の既知区間内に仮想PSを複数置き、

θ
^
w
	​

=arg
θ
min
	​

q
∑
	​

MSE(PF
θ
	​

(prefix
q
	​

),TVT
q:PS
	​

)+λ∥θ−θ
global
	​

∥
2

を解きます。

完全な自由最適化ではなく、グローバル設定へのshrinkageを入れます。

特に対象井戸ごとに変える価値が高いのは、

likelihood temperature
process noise
self-template／Typewellの比率
look-ahead length
jump prior

です。

推奨する完成形

私なら、現在のPFを次の構成へ変更します。

Block Auxiliary PF+Tempered Likelihood+Mode-preserving Resample-Move+Forward-Backward Smoothing
	​


具体的には次です。

1. prefixからGR校正・self-template・初期slope分布を推定
2. suffixを32～64行のblockに分割
3. 各blockでGR similarity mapの上位ピークを計算
4. dynamics proposalとGR-peak proposalの混合から生成
5. block GR likelihoodをtemperingで段階的に導入
6. ESS低下時にmode別resampling
7. slope・shift・curvatureにMCMC rejuvenation
8. 粒子・祖先・regimeを全block保存
9. backward simulationでsmoothed trajectoryを生成
10. mode別posterior meanと全体posterior meanを出力
実験順序

一度に全部変えず、次の順番がよいです。

Forward PF → backward smoother
行単位 → block likelihood
bootstrap proposal → GRピーク混合proposal
通常更新 → ESS制御tempering
通常resampling → mode-preserving resampling
jitter → resample-move
単一dynamics → regime mixture
Typewellのみ → self-template混合
固定パラメータ → prefix pseudo-PS校正

各段階で見るべき値は、最終RMSEだけではありません。

internal oracle
weighted mean RMSE
posterior median RMSE
good-mode mass
true trajectoryの90%区間coverage
ancestor diversity
mode extinction率
horizon別RMSE
最も可能性が高い一手

一つだけ実装するなら、

block look-ahead auxiliary proposal + backward smoothing

です。

現在のPFが正解軌道を生成できているのに、前半のGR aliasで誤った軌道へ固定されているなら、この2つが直接効きます。

30個のPFをそのまま増やすより、各PFを「未来GRを利用して提案し、未来情報で過去軌道を修正できるPF」に変える方が、median 6.5を下げる可能性があります。