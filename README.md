# SAM 2.1 Tiny and Grounding DINO / i.MX95 Demo

An interactive HDMI demo comparing ailia and NXP ONNX Runtime on CPU and NPU.
Select **SAM 2.1 Tiny** or **Grounding DINO** in the upper-left corner.
SAM segments clicked objects and tracks them across five Bedroom frames; Grounding DINO detects objects described by an English prompt.

## Requirements

- i.MX95, HDMI display, and mouse
- NXP i.MX Distro `6.18-wrynose` (Linux ARM64, Weston/Wayland)
- Python 3, NumPy, OpenCV, GTK3, PyGObject, and Cairo
- ailia Python package `1.7.0.0` and a valid ailia license
- NXP ONNX Runtime `1.24.3` and the Neutron driver supplied with the BSP

The bundled SDK targets this BSP (`GLIBC_2.43`, `GLIBCXX_3.4.32`, and `CXXABI_1.3.15`).
Keep the NXP-provided ORT installation; do not replace it with the PyPI version.

## Setup

Run on the device:

```sh
git clone https://github.com/ailia-ai/ailia-sdk-nxp-demo.git
cd ailia-sdk-nxp-demo
sha256sum -c SHA256SUMS
python3 scripts/download_models.py

python3 -m pip install --target ./python --no-deps ailia==1.7.0.0
python3 scripts/copy_sdk.py --install-npu --package-dir ./python/ailia
```

ailia SDK 1.7 (public release) does not support the NXP NPU. The commands above install the bundled ailia SDK 1.7 (NXP NPU-enabled version) in its place.
Stop running applications before replacing the library.

Converted models are bundled under `models/`. Unmodified SAM auxiliary files are downloaded from ailia-models, verified by SHA-256, and cached locally.
The launcher downloads missing auxiliary files; subsequent launches can run offline.
Keep Grounding DINO's `.onnx` file and all accompanying `.dataNN` files together.

To save the pip-provided binary, run this before replacing it:

```sh
python3 scripts/copy_sdk.py --export-pip --package-dir ./python/ailia
# Saved to: sdk/from-pip/linux/arm64-v8a/libailia.so
```

### License

Obtaining an evaluation license for the first time requires an internet connection and the correct system date and time.
For offline use or an expired license, obtain a valid license separately.
Licenses are managed per user account. See the [bundled license](licenses/LICENSE_AILIA_EN.pdf) for terms of use.

## Launch and Controls

Run as the same user as Weston:

```sh
sh launch.sh                         # Fullscreen, ailia NPU
sh launch.sh --bedroom               # Five-frame tracking
sh launch.sh --windowed              # Windowed mode
sh launch.sh --backend ort_cpu       # Start with ORT CPU
sh launch.sh --image /path/to/image.jpg
sh launch.sh --model grounding --caption "person. truck. car."
```

- **Open image**: select an image
- **Compare all 4**: compare all four backends

Models stay in memory until you switch models or backends. Grounding DINO also reopens when the prompt's token count changes.
Both runtimes perform one excluded dummy inference at model Open. Comparisons report the median of three subsequent runs, excluding loading and drawing.
Benchmark history is kept separately for each model and persists across application restarts.
Logs are saved to `demo.log`, and benchmark history to `benchmark_history.jsonl`.
When launching over SSH, set `XDG_RUNTIME_DIR` and `WAYLAND_DISPLAY` to match the Weston session.

### Launcher Icon

```sh
python3 scripts/install_desktop.py
```

To add a launcher to the Weston panel, append the following to `weston.ini`:

```ini
[launcher]
icon=/path/to/ailia-sdk-nxp-demo/launcher.png
path=/bin/sh /path/to/ailia-sdk-nxp-demo/launch.sh
```

## SAM 2.1 Tiny

### Controls and Models

- Left-click: foreground point; right-click: background point
- **Undo point / Clear points**: undo or clear points
- **Track 5 frames**: track the selected object, then return to frame 1 to accept new clicks

The five bundled 512-pixel models use the name `sam2.1_tiny_<component>_512_<precision>.onnx`.
The INT8 Image Encoder mixes integer MatMul, QDQ Conv, and FP32 operations. Softmax and LayerNorm compute in FP32 with quantized inputs and outputs.
Memory Attention uses integer MatMul with FP32 RoPE. Auxiliary models remain FP32.

### Benchmarks

Measured on 2026-11-18 on i.MX95 with model revision `coco-val2017-v1`.
NXP ORT 1.24.3 / ailia SDK 1.7 (NXP NPU-enabled version), six CPU threads, profiling disabled.
The ailia runtime is `566f2f71b`, with NEON-optimized CPU integer MatMul.
All backends run one inference with identical dummy inputs when opening each model. They then warm up on the complete five-frame Bedroom sequence using the same two points. Results are medians of the next three runs; warm-up is excluded.

| Component | Calls / video | ORT CPU | ORT NPU* | ailia CPU | ailia NPU |
|---|---:|---:|---:|---:|---:|
| Image Encoder (ms / call) | 5 | 2,273.1 | 2,194.3 | 2,471.5 | 891.0 |
| Mask Decoder (ms / call) | 5 | 169.6 | 164.8 | 126.8 | 126.9 |
| Memory Attention (ms / call) | 4 | 1,487.2 | 1,484.2 | 1,116.5 | 517.3 |
| Memory Encoder (ms / call) | 5 | 294.3 | 299.9 | 236.1 | 236.4 |
| Full five-frame sequence (s) | — | 20.14 | 19.75 | 19.12 | 8.75 |

Per-model values are medians of each run's average time per call. Full-sequence timing includes image loading and pre/post-processing, but excludes model loading, rendering, and saving results.
All four backends use the same models. NPU backends run Image Encoder and Memory Attention with mixed CPU/NPU execution; other models run on the CPU.

* **ORT NPU timings are for speed reference only: its outputs are incorrect, with masks containing only 0–3 pixels in subsequent frames of the final measured run.**

Mean mask IoU against the FP32 reference on Bedroom is 96.07% for host ORT CPU and 95.71% for ailia NPU on the device.
These are comparisons with reference outputs, not ground-truth accuracy scores.

In the app, still-image comparisons show medians of three runs; video timings show individual frame measurements.
Both ailia and ORT perform one excluded warm-up inference with the same dummy inputs when opening each model.
The app's Sum column totals inference time, not UI response time.

### Model Conversion

On a host PC, convert the public SAM 2.1 Tiny ONNX models from
[SAM2 in ailia-models](https://github.com/ailia-ai/ailia-models/tree/master/image_segmentation/segment-anything-2).
Extract the [COCO 2017 validation images and annotations](https://cocodataset.org/#download), then run:

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

Min/Max calibration uses all 5,000 images, with tensor-wise UINT8 activations and channel-wise INT8 constant weights.
Memory Attention uses a synthetic five-frame sequence generated from each image.

The resulting models mix INT8 and FP32. In Image Encoder, MatMul uses integer operators, Conv uses QDQ,
and Softmax/LayerNorm quantize only their inputs and outputs.
Memory Attention uses integer MatMul while retaining FP32 RoPE. Auxiliary models also remain FP32.

Generated models are saved under `models/` in the output directory. Calibration records are saved as `manifest.json`, `ranges_*.json`, and `completion.json`.
Run the same command to resume an interrupted conversion. Generated models are not automatically copied over the bundled models.

## Grounding DINO

### Controls and Model

Enter English category names separated by periods, such as `person. truck. car.`, then press **Detect**.
Bounding boxes, labels, and confidence scores appear on the image. **Compare all 4** measures the same image and prompt on each backend.

The Swin-T model uses a 400×599 letterboxed image and up to 256 text tokens.
Its 347 fixed-weight MatMuls use tensor-wise UINT8 activations and channel-wise INT8 weights through standard ONNX `MatMulInteger` nodes. Other operations remain FP32.
The graph is rearranged to use the SDK's existing DeformableAttention fusion; dynamic Attention stays on CPU.
The bundled SDK binary is unchanged. ORT NPU outputs may vary between inferences.

### Benchmarks

To measure all four backends without opening the GUI:

```sh
PYTHONPATH=./python python3 scripts/benchmark_grounding.py --output /tmp/grounding-results.json
```

Measured on i.MX95 on 2026-11-19 with revision `grounding-coco-val2017-v1`,
the unchanged bundled ailia SDK 1.7 (`566f2f71b`), and NXP ORT 1.24.3.
Six CPU threads, profiling disabled, one excluded dummy inference at Open,
then the median of three runs. Input: `images/truck.jpg`, prompt:
`person. truck. car. bicycle. dog.`. Loading and pre/post-processing are excluded.

| Runtime | CPU (ms) | NPU + CPU fallback (ms) |
|---|---:|---:|
| ailia SDK 1.7 | 7,788.1 | 3,924.9 |
| NXP ONNX Runtime 1.24.3 | 7,545.9 | 7,011.9 |

All four runs detected the truck. This is a functional check, not a detection-accuracy benchmark.
Model hashes and calibration provenance are recorded in `models/grounding_manifest.json`.

### Model Conversion

Convert the public `groundingdino_swint_ogc.onnx` from
[Grounding DINO example in ailia-models](https://github.com/ailia-ai/ailia-models/tree/master/object_detection/groundingdino).
Use the same complete COCO val2017 image and annotation directories as SAM:

```sh
curl --fail --location --output groundingdino_swint_ogc.onnx \
  https://storage.googleapis.com/ailia-models/groundingdino/groundingdino_swint_ogc.onnx
python3 -m pip install -r scripts/requirements-calibration.txt
python3 scripts/calibrate_grounding.py \
  --source groundingdino_swint_ogc.onnx \
  --coco-images /path/to/coco/val2017 \
  --coco-annotations /path/to/coco/annotations/instances_val2017.json \
  --output /path/to/grounding-calibrated \
  --workers 3 --threads 3
```

Global Min/Max calibration processes all 5,000 images with the 80 COCO category names as the text prompt.
Image preprocessing and tokenization are shared with the demo. The input model is promoted to FP32 before collecting activation ranges.
The output is `models/groundingdino_swint_ogc_400x599_int8.onnx` with external weight shards.
`manifest.json`, `ranges.json`, and `completion.json` record the calibration inputs, ranges, and artifact hashes.
Rerun the same command to resume an interrupted calibration. Copy the ONNX file and all weight shards together after completion.

## Demo Images and Licenses

Demo images come from the official Segment Anything repositories:
[truck.jpg](https://github.com/facebookresearch/segment-anything/blob/main/notebooks/images/truck.jpg) and
[Bedroom](https://github.com/facebookresearch/sam2/tree/main/notebooks/videos/bedroom).
See [NOTICE.md](NOTICE.md) for attribution and licenses.
