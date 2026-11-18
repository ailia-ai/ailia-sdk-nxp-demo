# SAM 2.1 Tiny / i.MX95 Demo

ailia Inc.

i.MX95のHDMI画面で、クリックによるセグメンテーションとBedroomの5フレーム追跡を行うデモです。
COCO校正済みのSAM2.1 Tiny 512×512モデルを同梱し、ailia／ORTのCPU・NPUを比較できます。

## 動作環境

- i.MX95、HDMI、マウス
- NXP i.MX Distro `6.18-wrynose`（Linux ARM64、Weston/Wayland）
- Python 3、NumPy、OpenCV、GTK3、PyGObject、Cairo
- ailia Python package `1.7.0.0`、有効なailiaライセンス
- NXP ONNX Runtime `1.24.3`、BSP付属のNeutronドライバ

同梱SDKはこのBSP向けです（`GLIBC_2.43`、`GLIBCXX_3.4.32`、`CXXABI_1.3.15`）。
NXP版ORTはPyPI版で上書きせず、そのまま使用してください。

## セットアップ

実機で実行します。

```sh
git clone https://github.com/ailia-ai/ailia-sdk-nxp-demo.git
cd ailia-sdk-nxp-demo
sha256sum -c SHA256SUMS

python3 -m pip install --target ./python --no-deps ailia==1.7.0.0
python3 scripts/copy_sdk.py --install-npu --package-dir ./python/ailia
```

ailia SDK 1.7 (公開版)はNPU非対応です。上記コマンドで同梱のailia SDK 1.7 (NXP NPU対応版)へ差し替えます。
既存アプリを終了してから実行してください。

pip版のバイナリを保存する場合は、差し替え前に次を実行します。

```sh
python3 scripts/copy_sdk.py --export-pip --package-dir ./python/ailia
# 保存先: sdk/from-pip/linux/arm64-v8a/libailia.so
```

### ライセンス

初回の評価ライセンス取得にはインターネット接続と正しいシステム日時が必要です。
オフライン利用や期限切れの場合は、別途有効なライセンスを用意してください。
ライセンスは実行ユーザーごとに管理されます。利用条件は[同梱ライセンス](licenses/LICENSE_AILIA_JA.pdf)を参照してください。

## 起動・操作

Westonと同じユーザーで実行します。

```sh
sh launch.sh                         # 全画面、ailia NPU
sh launch.sh --bedroom               # 5フレーム追跡
sh launch.sh --windowed              # ウィンドウ表示
sh launch.sh --backend ort_cpu       # ORT CPUで開始
sh launch.sh --image /path/to/image.jpg
```

- 左クリック：前景点、右クリック：背景点
- **Undo point / Clear points**：点を戻す／消す
- **Open image**：画像を選択
- **Compare all 4**：4方式を比較
- **Track 5 frames**：最初のフレームで指定した対象を追跡。終了後は1枚目に戻り、再クリックできます。

ランタイムを切り替えるまでモデルを保持し、測定履歴は再実行・再起動後も残ります。
ログは`demo.log`、履歴は`benchmark_history.jsonl`へ保存します。
SSH起動時はWestonの`XDG_RUNTIME_DIR`と`WAYLAND_DISPLAY`を指定してください。

### 起動アイコン

```sh
python3 scripts/install_desktop.py
```

Westonパネルに追加する場合は、`weston.ini`に次を追記します。

```ini
[launcher]
icon=/path/to/ailia-sdk-nxp-demo/launcher.png
path=/bin/sh /path/to/ailia-sdk-nxp-demo/launch.sh
```

## ベンチマーク

2026-11-18、i.MX95、モデル`coco-val2017-v1`。
NXP ORT 1.24.3／ailia SDK 1.7 (NXP NPU対応版)、CPU 6スレッド、プロファイル無効。
ailiaのランタイムは`566f2f71b`（CPU整数MatMulのNEON最適化版）です。
全方式でモデルOpen時に同じダミー入力で1回推論します。その後、Bedroomの5フレームを同じ2点で1周ウォームアップし、続く3回の中央値を測定しました。ウォームアップは測定から除外しています。

| 処理 | 回数／動画 | ORT CPU | ORT NPU※ | ailia CPU | ailia NPU |
|---|---:|---:|---:|---:|---:|
| Image Encoder (ms／回) | 5 | 2,273.1 | 2,194.3 | 2,471.5 | 891.0 |
| Mask Decoder (ms／回) | 5 | 169.6 | 164.8 | 126.8 | 126.9 |
| Memory Attention (ms／回) | 4 | 1,487.2 | 1,484.2 | 1,116.5 | 517.3 |
| Memory Encoder (ms／回) | 5 | 294.3 | 299.9 | 236.1 | 236.4 |
| 5フレーム全体 (秒) | — | 20.14 | 19.75 | 19.12 | 8.75 |

モデル別は1回あたりの平均時間の中央値です。全体時間は画像読み込み・前後処理を含み、モデル読み込み・描画・保存を除きます。
4方式とも同じモデルを使用します。NPU方式ではImage EncoderとMemory AttentionをCPU/NPU混在で実行し、その他はCPU実行です。

※ **ORT NPUは出力異常（最終測定の後続フレームのマスクが0～3画素）があるため、速度参考値です。**

BedroomでのFP32参照に対する平均マスクIoUは、ホストORT CPUで96.07%、実機ailia NPUで95.71%です。
正解マスクに対する精度ではありません。

アプリ内の静止画比較は3回中央値、動画表示は各フレームの実測値です。
ailia・ORTともモデルOpen時に同じダミー入力で1回ウォームアップし、測定から除外します。
表のSumは推論区間の合計で、画面応答時間とは異なります。

## モデル変換

ホストPCで、[ailia-modelsのSAM2](https://github.com/ailia-ai/ailia-models/tree/master/image_segmentation/segment-anything-2)
の公開SAM2.1 Tiny ONNXから生成します。
[COCO 2017 validation画像・アノテーション](https://cocodataset.org/#download)を展開して実行してください。

```sh
python3 -m pip install -r scripts/requirements-calibration.txt
python3 scripts/calibrate_sam2.py \
  --source-dir /path/to/ailia-models/image_segmentation/segment-anything-2 \
  --download-sources \
  --coco-images /path/to/coco/val2017 \
  --coco-annotations /path/to/coco/annotations/instances_val2017.json \
  --output /path/to/sam2-calibrated \
  --workers 2 --threads 2
```

全5,000画像でMin/Max校正し、活性値はtensor-wise UINT8、固定重みはchannel-wise INT8にします。
Memory Attentionには各画像から生成した5フレームの疑似動画を使います。

出力はINT8／FP32混在です。Image EncoderのMatMulは整数演算、ConvはQDQ、
Softmax・LayerNormは入出力のみ量子化します。
Memory AttentionはMatMulを整数化し、RoPEはFP32を維持します。補助モデルもFP32です。

生成モデルは出力先の`models/`、校正記録は`manifest.json`・`ranges_*.json`・`completion.json`に保存します。
同じコマンドで途中再開できます。同梱モデルへのコピーは自動では行いません。

## デモ画像・ライセンス

デモ画像はSegment Anythingの公式リポジトリのものです：
[truck.jpg](https://github.com/facebookresearch/segment-anything/blob/main/notebooks/images/truck.jpg)、
[Bedroom](https://github.com/facebookresearch/sam2/tree/main/notebooks/videos/bedroom)。
出典・ライセンスは[NOTICE.md](NOTICE.md)を参照してください。
