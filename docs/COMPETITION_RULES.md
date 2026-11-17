# ROGII - Wellbore Geology Prediction — コンペ概要・ルールまとめ

> 新しいエージェント／作業者への引き継ぎ用ドキュメント。コンペの目的・データ・評価・注意点を日本語でまとめる。
> 最終更新: 2026-06-28

---

## 1. コンペの目的（一言で）

水平坑井(horizontal wellbore)に沿った地質を予測する。具体的には、各水平坑井の**評価ゾーン**における **TVT (True Vertical Thickness / 真垂直層厚)** を予測する回帰タスク。

石油・ガス掘削の自動化が背景。掘削中に「ドリルビットが地層のどこにいるか」を推定し、好ましい層に沿って掘り進めるための支援が目的。

---

## 2. 予測対象（ターゲット）

- **ターゲット変数: `TVT`(True Vertical Thickness, ft)**
  - 水平坑井の1ftごとに、手動で解釈された「地質的な位置」を表す値。
  - 各坑井の **評価ゾーン(evaluation zone)** でこの値が隠されており(NaN)、そこを予測する。
- 評価ゾーンの見分け方: 水平坑井CSVの **`TVT_input` 列が NaN になっている行が評価ゾーン**。
  - `TVT_input` は TVT のコピーだが、評価ゾーンだけ NaN にされ「特徴量(入力)」として与えられている。
  - つまり評価ゾーンより手前(坑口側)の TVT は `TVT_input` として既知 → これを手がかりに使える。

---

## 3. 評価指標

- **RMSE (Root Mean Squared Error)**。低いほど良い。
- 提出は評価ゾーンの各行(1ftごと)について TVT を予測。

---

## 4. データ構成

ルートのデータディレクトリ: `rogii-wellbore-geology-prediction/`

```
rogii-wellbore-geology-prediction/
├── train/   各坑井に3ファイル
│   ├── {WELL}__horizontal_well.csv
│   ├── {WELL}__typewell.csv
│   └── {WELL}.png            (坑井経路と地質断面の可視化)
├── test/    各坑井に2ファイル (約200坑井) ※pngなし
│   ├── {WELL}__horizontal_well.csv
│   └── {WELL}__typewell.csv
└── sample_submission.csv
```

- 各坑井は8文字のハッシュで識別 (例: `015fe0d2`)。

### 4.1 `{WELL}__horizontal_well.csv`(水平坑井の軌道・ログ)

| 列 | 意味 | 備考 |
|---|---|---|
| `WELLNAME` | 坑井ID | (説明上の項目。実CSVヘッダにはない場合あり) |
| `MD` | Measured Depth (ft) 坑口からの総延長 | |
| `X` | Easting (ft) 水平面の座標 | |
| `Y` | Northing (ft) 水平面の座標 | |
| `Z` | True Vertical Depth (ft) 海面下の垂直距離 | |
| `ANCC, ASTNU, ASTNL, EGFDU, EGFDL, BUDA` | 各地層フォーメーションの予測深度 | **train のみ** |
| `TVT` | 真垂直層厚 (ft) = **ターゲット** | **train のみ** |
| `GR` | Gamma Ray (API) 自然放射能ログ | train/test 両方 |
| `TVT_input` | TVT のコピー。評価ゾーンは NaN | train/test 両方 |

> ⚠️ **重要**: test の horizontal_well.csv は `MD, X, Y, Z, GR, TVT_input` の6列のみ。
> `ANCC〜BUDA` や `TVT` は test には**存在しない**。train だけにある。

### 4.2 `{WELL}__typewell.csv`(垂直リファレンスログ / タイプウェル)

地質相関(correlation)用の垂直方向の参照ログ。train/test 両方にある。

| 列 | 意味 |
|---|---|
| `TVT` | 垂直深度インデックス (ft)。水平坑井の TVT(地質的位置)に対応する主要な深度基準 |
| `GR` | Gamma Ray (API)。相関に使う垂直方向の GR シグネチャ |
| `Geology` | フォーメーションラベル(例: `EGFDL`, `BUDA`)。カテゴリ。多くは空欄 |

> typewell の `TVT` 軸と GR を「物差し」として、水平坑井で観測した GR パターンを照合し、
> 水平坑井がいま地質のどの深さ(TVT)にいるかを推定する、というのがドメイン上の発想。

### 4.3 提出ファイル `sample_submission.csv`

```
id,tvt
000d7d20_1442,0.0
000d7d20_1443,0.0
...
```

- `id` = `{WELLNAME}_{row_index}`(row_index は horizontal_well.csv の行インデックス)。
- `tvt` = 予測した TVT (ft)。
- 評価ゾーン(`TVT_input` が NaN の行)についてのみ予測行が必要。

---

## 5. ⚠️ 最重要の落とし穴 (test の扱い)

- **ここに見えている `test/` フォルダは、train から数件抜き出したサンプルにすぎない。**
  提出が**本番の隠しテストセットで再実行(rerun)される際**に、これらは実際のテストデータ(約200坑井)に置き換えられる。
- → コードは「test/ にどの坑井が来てもパスが通る」「特徴量は test に存在する列のみから生成できる」形で書くこと。
- → `ANCC〜BUDA` や `TVT` といった train 限定の列に test 推論時に依存しないこと(リーク・実行時エラーの原因)。

---

## 6. その他のルール

- **Working Note Award (任意)**: Public LB でメダルゾーンにいるチームのみ応募可能。ワーキングノート(解法ドキュメント)が審査される。
- スケジュール: 開始から約2ヶ月経過、残り約1ヶ月(2026-06-28 時点)。
- Merger & Entry の締切に注意(チームマージ・エントリー期限)。

---

## 7. このリポジトリの現状(作業の引き継ぎ)

> ※コンペ公式ルールではなく、本ローカル環境での既存作業メモ。

```
rogii_claude/
├── docs/                         このドキュメント
├── rogii-wellbore-geology-prediction/   公式データ (train/ test/ sample_submission)
├── v9/                           現行パイプライン(v9)
│   ├── pf_tuning_create_train_csv_multi_pf_v9.ipynb   特徴量生成(pf trace)
│   └── tuned_train_lightGBM_catboost_fixed_v9.ipynb   モデル学習(LGBM/CatBoost/XGB)
└── artifacts_v9/
    ├── data/train.parquet        生成済み学習データ
    └── models/                   学習済みモデル
        ├── lightgbm-1..3, catboost-1..3, xgboost-1..3 (.pkl)
```

- モデルは LightGBM / CatBoost / XGBoost をそれぞれ3シード(または3分割)で学習しアンサンブルする構成と見られる。

---

## 8. 作業方針メモ

- ユーザーは「基本的には自分で考えて実装したい(学習目的)」。アイデアを求められた時以外は、
  勝手に大きく方針を変えず、依頼ベースで支援する。
- 環境: Windows 11 / PowerShell。一時ファイルはスクラッチパッドへ。
- 推論コードは **test に実在する列のみ** から特徴量を作れることを常に確認する(§5)。
