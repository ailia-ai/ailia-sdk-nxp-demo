# SAM 2.1 Tiny / i.MX95 Demo

An interactive segmentation demo for an i.MX95 HDMI display, with click-based segmentation and tracking across five Bedroom frames.
Includes COCO-calibrated SAM 2.1 Tiny 512×512 models for comparing ailia and ORT on CPU and NPU.

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

python3 -m pip install --target ./python --no-deps ailia==1.7.0.0
python3 scripts/copy_sdk.py --install-npu --package-dir ./python/ailia
```

ailia SDK 1.7 (public release) does not support the NXP NPU. The commands above install the bundled ailia SDK 1.7 (NXP NPU-enabled version) in its place.
Stop running applications before replacing the library.

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
```

- Left-click: foreground point; right-click: background point
- **Undo point / Clear points**: undo the last point or clear all points
- **Open image**: select an image
- **Compare all 4**: compare all four backends
- **Track 5 frames**: track the object selected in the first frame. After tracking finishes, the demo returns to the first frame and accepts new clicks.

Models stay in memory until you switch backends. Benchmark history persists across runs and application restarts.
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

## Benchmarks

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

## Model Conversion

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

## Demo Images and Licenses

Demo images come from the official Segment Anything repositories:
[truck.jpg](https://github.com/facebookresearch/segment-anything/blob/main/notebooks/images/truck.jpg) and
[Bedroom](https://github.com/facebookresearch/sam2/tree/main/notebooks/videos/bedroom).
See [NOTICE.md](NOTICE.md) for attribution and licenses.
