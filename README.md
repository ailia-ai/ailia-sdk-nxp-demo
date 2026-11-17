# SAM 2.1 Tiny / i.MX95 Demo

ailia Inc.

HDMIに接続したi.MX95で、画像をクリックしてセグメンテーションし、
ailiaとNXP ONNX Runtimeの速度を比較するGTK3アプリです。
SAM 2.1 Tinyを512×512へ変換したモデルを同梱しています。
bedroomの5フレームを、SAM 2.1のメモリを使って追跡できます。

## 実行方式

| 選択肢 | Image Encoder / Memory Attention | その他のネットワーク |
|---|---|---|
| ailia CPU | ailia CPU | ailia CPU |
| ailia NPU | ailia Neutron + CPU fallback | ailia CPU |
| ORT CPU | ONNX Runtime CPU | ONNX Runtime CPU |
| ORT NPU | NeutronExecutionProvider + CPU fallback | ONNX Runtime CPU |

「NPU」は全ノードのNPU実行を意味しません。Prompt Encoder、Mask Decoder、
Memory Encoder、MLPなどはCPUで動作します。4方式で同じモデルを使用します。
公開pip版の **ailia SDK 1.7 (公開版)** はNPU非対応です。
このリポジトリの **ailia SDK 1.7 (NXP NPU対応版)** への差し替えでNPUを利用できます。

## 検証環境・必要なもの

- NXP i.MX95 / Linux aarch64、HDMI、マウス。
- NXP i.MX Distro with Wayland and XWayland `6.18-wrynose`。
- Weston/Wayland、Python 3、NumPy、OpenCV (`cv2`)、GTK3、PyGObject、Cairo。
- ailia Python package `1.7.0.0`、有効なailiaライセンス。
- ORTを使う場合はNXP版ONNX Runtime `1.24.3`。
- NPUを使う場合はBSPのNeutronドライバ、対応ファームウェア、NPUへのアクセス権。

同梱SDKはLinux ARM64専用です。macOS/x86では実行できません。
BSPの共有ライブラリに依存するため、他のYoctoイメージでの動作は保証しません。
同梱バイナリのELF依存シンボルには`GLIBC_2.43`、`GLIBCXX_3.4.32`、
`CXXABI_1.3.15`が含まれます。古いBSPでは差し替えだけでは動作しません。
GTK/OpenCV/Neutron/ORTは対象のNXPイメージから導入してください。
**NXPのORTをPyPIの`onnxruntime`で上書きしないでください。**
PyPI版ではNeutron EPが利用できなくなる可能性があります。

## セットアップ

実機で次を実行します。リポジトリのモデル・画像はGitの通常ファイルです。

```sh
git clone https://github.com/ailia-ai/ailia-sdk-nxp-demo.git
cd ailia-sdk-nxp-demo
sha256sum -c SHA256SUMS

# Python bindingをデモ専用フォルダへインストール（BSPの依存パッケージを維持）
python3 -m pip install --target ./python --no-deps ailia==1.7.0.0

# 同梱NPU対応版をコピーして差し替え
python3 scripts/copy_sdk.py --install-npu --package-dir ./python/ailia
```

`python/`はGit管理対象外です。システムのailiaを上書きしません。
差し替え前にアプリを終了してください。コピー対象は`libailia.so`だけです。
NumPyやOpenCVなどは別途BSPで導入済みであることを確認してください。

pipパッケージからSDKバイナリをコピーする場合は、**差し替え前**に実行します。

```sh
python3 scripts/copy_sdk.py --export-pip --package-dir ./python/ailia
# sdk/from-pip/linux/arm64-v8a/libailia.so にコピー
```

差し替え後に実行すると、コピー元はNPU対応版になります。
通常の仮想環境へpipインストールした場合は、そのPythonを使い、`--package-dir`を省略できます。
詳細は[sdk/README.md](sdk/README.md)を参照してください。

### ライセンス取得

ailiaは商用SDKです。同梱バイナリにもSDKの利用条件が適用されます。
Pythonパッケージは必要に応じて評価ライセンスをオンライン取得します。
初回実行時はネットワーク、DNS、HTTPS接続、正しいシステム日時が必要です。
取得失敗・期限切れ・オフライン利用の場合は、利用条件に合うライセンスを別途用意してください。
Linuxでは実行ユーザーの`~/.shalo/AILIA.lic`などを確認します。
rootと一般ユーザーではライセンスの参照先が異なります。
ライセンスファイルは本リポジトリに含めず、コミットしないでください。
SDKの正式な条件は[同梱ライセンス文書](licenses/LICENSE_AILIA_JA.pdf)を参照してください。

## 起動・操作

Westonと同じユーザーで実行します。

```sh
sh launch.sh                         # 全画面、ailia NPU
sh launch.sh --windowed              # ウィンドウ表示
sh launch.sh --backend ort_cpu       # ORT CPUで開始
sh launch.sh --bedroom               # 5フレーム追跡モード
sh launch.sh --image /path/to/image.jpg
```

ログは`demo.log`へ保存します。`sh start.sh`で直接起動すると標準出力へ出ます。
SSH経由ではWestonの`XDG_RUNTIME_DIR`と`WAYLAND_DISPLAY`を指定してください。
別のPythonパッケージを使う場合は`AILIA_PYTHON_PATH`を設定できます。

- 左クリック：前景点、右クリック：背景点（最大64点）。
- **Undo point / Clear points**：点を戻す／消す。
- **Open image**：実機上の画像を選ぶ。
- **Compare all 4**：同じ入力で4方式を順番に測定する。
- **Bedroom (5 frames)**：最初のフレームへ戻る。対象をクリック後、**Track 5 frames**で追跡する。
- **Exit**：アプリと推論ワーカーを終了する。

任意長動画や途中フレームへの追加クリックには対応していません。
bedroomに戻って最初のフレームで指定し直してください。

### アプリ起動アイコン

```sh
python3 scripts/install_desktop.py
```

実行ユーザーのアプリケーション一覧へ`.desktop`ファイルを配置します。
Westonのデスクトップは通常のファイルアイコンを表示しない場合があります。
Westonパネルへ追加する場合は、使用中の`weston.ini`へ次のブロックを追加し、
`/path/to/ailia-sdk-nxp-demo`を実際の絶対パスに置き換えてください。
既存の設定を丸ごと上書きしないでください。

```ini
[launcher]
icon=/path/to/ailia-sdk-nxp-demo/launcher.png
path=/bin/sh /path/to/ailia-sdk-nxp-demo/launch.sh
```

## 測定値の意味

6 CPUスレッド、プロファイル無効、ailia memory mode **11**で測定します。速度は固定値ではなく実測です。

- 静止画のEncoder特徴量はクリック間で再利用します。
- ClickはPrompt + Decoder + マスク選択。Encoderは含みません。
- ailiaは各モデルをOpenした直後、対応するダミー入力で一度推論します。EncoderだけでなくPrompt、Decoder、動画用モデルも対象です。この時間はモデル準備に含め、測定値から除外します。
- クリック時にはウォームアップを追加しません。ただしクリック数などの動的形状が変わる場合、ランタイム内部の形状準備が発生することがあります。
- 選択中のランタイムとモデルをメモリに保持し、静止画・動画で共通モデルを共有します。ランタイム変更時に旧モデルを解放します。Compare allは選択中の方式を最後に実行して、その方式を保持します。
- 方式を切り替える時だけ推論プロセスを終了・再生成します。NXP ORTのNeutron用CMAメモリはセッション破棄だけでは解放されず、同一プロセスでailia NPUへ戻すと`ailiaCreate -12`になることを実機で確認したためです。連続クリック・同じ方式での再実行時にはプロセスを再生成しません。
- ORTの通常表示はfirst/latestです。公平な比較には**Compare all 4**を使ってください。
- 静止画のCompare allは初期化後3回の中央値です。ailiaはOpen時ウォームアップを使い、ORTのみ比較開始時にウォームアップします。
- 表のSumは推論区間の和で、読み込み・前処理・描画・IPCを含む画面応答時間ではありません。
- 動画は各フレームの実測値で、3回中央値ではありません。ailiaのOpen時ウォームアップは含みません。ORTの初回実行コストは含みます。
- `last_results.json` / `video_results.json`に結果を保存します（Git管理対象外）。
- 再実行・画像変更・ランタイム変更で他方式の測定値を消しません。表は方式ごとの最新測定値を表示し、準備だけでは既存の値を上書きしません。
- `Measurement history`に日時・入力画像付きの履歴を表示します。`benchmark_history.jsonl`へ追記し、アプリ再起動時にも直近200件と最新値を復元します。ファイルには全履歴を保持します。入力や測定方式が異なる値を同条件の結果として比較しないでください。

## モデル・精度上の制約

Image Encoderは512×512の静的INT8 QDQ版です。
Memory Attentionは64個のMatMulをQDQ化し、RoPEの16個の2×2 MatMulはFP32のままです。
入力メモリを4096空間トークン＋16ポインタトークンへ固定し、未使用領域をマスクします。
bedroomの5枚は量子化校正にも使用しており、未知動画の精度検証ではありません。
バックエンドや量子化による出力差があります。表示する予測IoUスコアは実測正解率ではありません。

デモ画像は**Segment Anythingの公式リポジトリのもの**です。
[truck.jpg](https://github.com/facebookresearch/segment-anything/blob/main/notebooks/images/truck.jpg)、
[bedroom](https://github.com/facebookresearch/sam2/tree/main/notebooks/videos/bedroom)を使用します。
画面の緑色マスクは本デモが生成したものです。出典・ライセンスは[NOTICE.md](NOTICE.md)を参照してください。

## 開発・テスト

```sh
python3 -m unittest discover -s tests -v
PYTHONPATH=./python python3 selftest.py --backends ailia_cpu ailia_npu ort_cpu ort_npu
PYTHONPATH=./python python3 video.py --backend ailia_npu --output video_results
```

`selftest.py`は静止画の推論・動的クリック数・出力有限性を検証します。
GTKを使わない単体テストはモデル推論を行いません。
コード整形・静的検査には`ruff format .`、`ruff check .`を使います。

構成：`app.py`（GUI）、`engine.py`（推論とワーカー）、`video.py`（追跡と計時）、
`settings.py`（共通設定・前処理）、`sam2_video_predictor.py`（SAM 2追跡実装）、
`models/`（静止画・動画の全モデル）、`sdk/`（SDK）、`scripts/`（導入補助）。
実験ログ、校正データ、中間モデル、ライセンスキーは含めていません。
