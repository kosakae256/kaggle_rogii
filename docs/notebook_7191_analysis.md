# 7.191 ノート解剖 — rogii-lb-7-191-copy-n-upgrade-b31810.ipynb(全部入りパイプライン)

> ユーザー保有の"全部入り"提出ノート(名前はLB7.191, upgrade版)の網羅解析。パイプライン図=[pipeline_7191.svg](pipeline_7191.svg)。
> 最終更新: 2026-07-03

## 全体像(最終フロー)
最終提出 = **`0.3·GBM(ridge) + 0.7·Selector` → ロバスト多項式射影(上書き) → Gold prefix校正 → submission.csv**。
さらに **Grid FB平滑器** が非破壊の候補を吐く(採用は別途)。多段アンサンブル。

## 部品(セル別)
| # | 部品 | セル | 役割 |
|---|---|---|---|
| データ | HW/TW/formation列(ANCC..BUDA, train限定) | — | 入力 |
| PF(selector) | **GPU PF (v20 r0, `_R0_RAW`, use_phys, 128seed, scale{3,5,8,12})** | 5/6 | **CPU PFを置換し selector にだけ供給**(前計算→cache→monkeypatch) |
| PF(GBM) | **CPU PF (run_pf_ancc/run_pf_z)** | 7/28 | **GBM特徴用。未調整のまま**(下記) |
| lik-PF | 尤度加重PF(CPU) | 29 | GBM特徴 + PP |
| Selector | 6分割(n_eval>4840, z_span閾値)→ variant(scale/beam/hold) | 4 | ヒューリスティック本体。v25の値の出所 |
| Beam | 14 configs 平均(DTW的) | 4/28 | selector + GBM特徴 |
| NCC | multi-scale窓相関 8/15/25 | 28 | GBM特徴 |
| 構造 | **FormationPlaneKNN**(近傍坑井のdip面, self除外) / **DenseANCCImputer** | 31 | GBM特徴 + grid平滑のprior |
| build_well | ~200+特徴(PF/beam/NCC/formation/dense/GR統計 + lik-PF) | 32 | GBM入力 |
| GBM | LGBM×n + CatBoost + XGB → **Ridge stack**, GroupKFold(5)OOF | 9-14/33 | 学習系 |
| PP | make_prediction: `w_sub1=0.6·GBM + 0.4·lik-PF` + ランプτ=85 + SG | 15/34 | GBM後処理(773坑井OOF Nelder-Mead調整) |
| Main blend | `0.3·GBM + 0.7·Selector` | 21 | 合成(主役=Selector) |
| 射影 | per-well robust deg4 fit, `0.25·raw + 0.75·fit`(上書き) | 22 | ジッタ除去/誤枝抑制 |
| Grid FB平滑 | linear+構造prior, 前向き後ろ向き, グリッド(決定論・退化なし) | 41 | **非破壊候補のみ**(cell42, w=0.15-0.5) |
| Gold | 可視prefix backtestで候補が勝つ坑井だけ適応α移動(3プロファイル) | 40 | 最終微調整 |

## ★重要な発見(今セッション)

### 1. GPU PF は Selector 専用。GBM の PF は CPU で未調整
- **GPU PF(cell5/6)は `run_pf_lik_ensemble_scales`(=selector)だけ差し替え**。cell6明言「selectorのみ差し替え / run_pf_ancc は現状」。
- **GBMの `run_pf_ancc/run_pf_z`(cell28)は未調整の丸値デフォルト**:
  `ANCC_ALPHA=0.998, ANCC_RN=0.002, ANCC_PN=0.005, RESAMP=0.5, N=600, jump/物理なし`。
  vs r0_seed32(Optuna): `mom=0.99979.., vn=0.00062.., resamp=0.0320..`(15桁端数)。→ **CPU PF は手打ちデフォルト**。
- **非対称**: 作者は selector のPFだけ強化し、GBM側は放置。**GBMのPFを強いsmooth PFに替えるのが未開拓レバー → v26**。

### 2. Grid FB平滑器は単体だとPFを劣化させる(=ノートが非破壊hedgeにしてる理由)
- 検証(check_pipeline_stages, cell41忠実移植): grid平滑器(linear prior)単体は selector予測(~6.7)を大きく上回れず。
  理由: **位置だけのランダムウォーク(dip/rate状態なし)+トゥアンカーなし+鋭いGR尤度 → 誤datumモードにドリフト**。
  PFはdip慣性で耐えるが grid は弱い。→ ノートが grid を w=0.15-0.5 の非破壊候補にしてるのと整合。
- **決定論(seed無し)は長所**(Private安定)。過学習は低い(w_structはCV選択)。**弱点は構造priorの空間転移のみ**。

### 3. 構造prior(FormationPlaneKNN)= 近傍坑井の本当のdip面
- formation列(ANCC..BUDA)は**depth**(検証: 坑井内で U-formation の std≈0.01=ほぼ一定offset)。
- 近傍坑井のX,Y+formation中央値から**平面フィットでdip面**を作り(self除外)、坑井offsetを足して構造TVTに。
- **build_well(GBM特徴, ~30列: tvtF_*/bw_*/form_*/spatial_*)** と **grid平滑のprior(w_struct)** の2箇所で使用。
- test有効(近傍trainのformationを使う→test実在列だけで動く)。ただし**隠しwellがtrainから空間的に遠いと劣化**(knn_dist で監視)。
- ＝研究#8「近傍dip」。v25の弱点(cat4 dip外挿)に直撃しうる**移植候補**だが、単体RMSEは要確認(空間精度依存)。

## 乱数(seed)マップ / パラメータマップ / 過学習評価
- **seed**: PF系は随所 n_seed=128(±0.2床)。cell7/28は bare np.random(順序依存で非再現), cell4/29は明示seed。**grid FB平滑器のみ決定論(seed無)**。gold は cal24/final48(env上書き可)。
- **調整の質**: PP(773 OOF Nelder-Mead)/w_struct(CV)/PFparam(Optuna)=**安全**。selector per-bin(sp45/sel15 vb best)・gold profile・手blend(0.3/0.7, 0.25/0.75)=**LB叩き疑い**。
- **過学習(Public→Private)**: 詳細 [overfit_policy.md](overfit_policy.md)。**主犯候補=selector per-bin値 と gold のマジックナンバー**。安全=PP/w_struct/grid/PFparam。命名"vb/patch"がLB反復調整の匂い。

## v25/v26 への移植の宝(まとめ)
1. **GBMのPFを強いsmooth PFに格上げ**(selectorだけ強化されてる非対称の是正)→ **v26**。
2. **構造prior(近傍dip面)** → v25平滑/GBMの硬い核(cat4)対策。ただし空間転移とOOF実証が条件([[overfit-policy]])。
3. **grid平滑器は丸ごと移植は無駄**(平滑がv25と冗長)。抜くなら**構造prior部分だけ**。
4. **selector の scale/beam/hold は robust PF ~10個で再導出**([[overfit-policy]] の #1)。

関連: [pipeline_7191.svg](pipeline_7191.svg), [overfit_policy.md](overfit_policy.md), [v25.md](v25.md), [v26.md](v26.md), [[overfit-policy]], [[hard-wells-priority]]。
