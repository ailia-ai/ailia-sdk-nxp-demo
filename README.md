# ailia Vision / i.MX95 Demo

An interactive HDMI demo comparing ailia and NXP ONNX Runtime on CPU and NPU.
Select **SAM 2.1 Tiny**, **Grounding DINO**, **Depth Anything V2**, **YOLOX-S 320**, or **ViT-B/16 INT4** in the upper-left corner.
SAM segments clicked objects and tracks them across five Bedroom frames; Grounding DINO detects objects described by an English prompt.
Depth Anything V2 estimates indoor metric depth.
YOLOX-S detects the 80 COCO object categories without a text prompt.
ViT classifies images into ImageNet categories and displays the top five results.

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

The bundled runtime selects NPU execution from operator types, shapes and
available memory. No NPU tuning environment variables or per-model switches
are required. Benchmark tables below identify their measured runtime.

Converted models are bundled under `models/`. Unmodified SAM auxiliary files and ViT INT4 are downloaded from ailia-models, verified by SHA-256, and cached locally.
The launcher downloads missing public models; subsequent launches can run offline.
Keep Grounding DINO's `.onnx` file and all accompanying `.dataNN` weight files together.
Git LFS is not required. Calibration records for the four calibrated models are
under [`calibration/`](calibration/). Benchmark tables are maintained here;
generated timing logs are not tracked. Tables identify the measured runtime
revision; only the Depth Anything table has been refreshed for `c97d853c4`.

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
sh launch.sh --model grounding --caption "Horse. Clouds. Grasses. Sky. Hill."
sh launch.sh --model depth            # Indoor depth, Bedroom image
sh launch.sh --model yolox            # YOLOX-S 320 object detection
sh launch.sh --model vit              # ViT-B/16 INT4 image classification
sh launch.sh --camera 1               # Start continuous inference with camera 1
sh launch.sh --video-input /tmp/test.avi  # Diagnostic camera replay without USB hardware
```

Use the **Image / Video / Camera** buttons above the preview:

- **Image**: use the model's bundled image or **Open image…**.
- **Video**: SAM 2 only. Click the target in the first Bedroom frame, then
  **Track 5 frames**. Tracking uses memory and returns to frame 1 when finished.
- **Camera**: all five models. USB UVC input automatically starts continuous
  inference, without a Start or Detect step. **Reconnect** retries a missing camera.

USB cameras are detected automatically; use `--camera INDEX` for a specific device.
Only the latest frame is processed, with one inference in flight; models stay loaded
between frames. Live network timings and processing FPS appear in the right panel
without overwriting benchmark history. **Compare current frame** holds one frame
for all four backends, then continuous inference resumes.
SAM applies the clicked foreground/background points independently to each frame;
camera input does not use Bedroom's memory tracking. `--video-input` is a diagnostic
camera substitute that loops at end of file, not a separate video mode for other models.
To test without a camera on the device, run
`PYTHONPATH=python python3 scripts/test_live_video.py`. This creates a temporary
MJPEG video and runs two frames through every model on ailia NPU and ORT CPU.

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

## ViT-B/16 INT4

Uses the public ailia-models `ViT-B_16-224-int4.onnx`, cached as
`models/vit_b16_224_int4.onnx`. Fixed MatMul weights use 4-bit MatMulNBits;
other operations remain floating point. No activation calibration is required.
Input is RGB, resized to 224 × 224 and normalized to [-1, 1].
Select **Classify** for a still image; Camera classifies each incoming frame automatically.
The published graph is unchanged, including its attention-map outputs.

### Benchmark

Same public INT4 model, batch 1, six CPU threads, one excluded dummy inference
at Open, then ten runs without profiling. Network time excludes preprocessing,
classification postprocessing and rendering. Measured on 2026-11-20 with the
bundled ailia runtime `ac22f5086` and `images/vit/input.jpg`.

| Runtime | Median inference | FPS |
| --- | ---: | ---: |
| ailia CPU | 780.6 ms | 1.28 |
| ailia NPU | 329.4 ms | 3.04 |
| NXP ORT 1.24.3 CPU | 677.3 ms | 1.48 |
| NXP ORT 1.24.3 NPU (Neutron EP) | 674.8 ms | 1.48 |

The ORT NPU selection logged a Neutron allocation failure and fell back to CPU
for this graph in the current HDMI environment; it is not a pure NPU measurement.

All four backends classified the example as **Labrador retriever** in all ten runs.

To reproduce each backend measurement:

```sh
PYTHONPATH=python python3 scripts/benchmark_vit.py --backend ailia_npu --output /tmp/vit-ailia-npu.json
```

Other backend choices are `ailia_cpu`, `ort_cpu`, and `ort_npu`.

## YOLOX-S 320

The bundled `models/yolox_s_320_int8.onnx` uses all 5,000 COCO val2017 images
for MinMax calibration. Convolution weights are per-channel INT8; feature maps
are UINT8. Swish uses lookup tables; final detection formatting remains FP32.
ailia NPU automatically uses device requantization for supported operations.
This uses different rounding from the CPU path, so outputs are not bit-identical.

### Benchmark

Same INT8 model, batch 1, six CPU threads, one excluded dummy inference at Open,
then ten runs without profiling. Times exclude preprocessing, NMS and rendering.
Measured on 2026-11-20 with ailia runtime `ac22f5086`, using `images/yolox/input.jpg`.

| Runtime | Median inference | FPS |
| --- | ---: | ---: |
| ailia CPU | 435.4 ms | 2.30 |
| ailia NPU | 114.9 ms | 8.70 |
| NXP ORT 1.24.3 CPU | 794.0 ms | 1.26 |
| NXP ORT 1.24.3 NPU (Neutron EP) | 794.3 ms | 1.26 |

These CPU figures measure the same ConvInteger graph, not an optimized FP32
model. ailia uses tiled NEON INT8 kernels with cached fixed weights on CPU.

### Accuracy

COCO val2017, all 5,000 images, score threshold 0.001, class-agnostic NMS 0.65,
evaluated with the COCO bbox evaluator:

| Model / execution | AP | AP50 | AP75 |
| --- | ---: | ---: | ---: |
| FP32 / host ORT CPU 1.23.2 | 30.42 | 45.79 | 32.05 |
| INT8 / ailia NPU | 29.00 | 44.77 | 30.59 |

AP decreases by 1.42 points. This is the calibration dataset, not an independent
test set. Device requantization changes rounding; the model is not lossless.
The demo uses a higher display threshold of 0.4 and NMS 0.45.

### Model conversion

Run on the host PC with the COCO val2017 images and annotations extracted:

```sh
python3 -m pip install -r scripts/requirements-calibration.txt
curl -L https://storage.googleapis.com/ailia-models/yolox/yolox_s.opt.onnx -o yolox_s.opt.onnx
python3 scripts/calibrate_yolox.py \
  --source yolox_s.opt.onnx \
  --coco-images /path/to/val2017 \
  --coco-annotations /path/to/annotations/instances_val2017.json \
  --output models/yolox_s_320_int8.onnx \
  --report calibration/yolox/report.json
```

The converter requires all 5,000 images and writes the calibration report to
the requested location. To measure one backend on the device:

```sh
PYTHONPATH=./python python3 scripts/benchmark_yolox.py \
  --models models --image images/yolox/input.jpg --backend ailia_npu \
  --output /tmp/yolox_ailia_npu.json
```

For accuracy evaluation, use `scripts/evaluate_yolox.py --annotations ...
--images ... --detections detections.json`. Score saved detections on the host
with the same script's `--score-only --annotations ... --detections detections.json`.

## SAM 2.1 Tiny

### Controls and Models

- Left-click: foreground point; right-click: background point
- **Undo point / Clear points**: undo or clear points
- **Track 5 frames**: track the selected object, then return to frame 1 to accept new clicks

The five bundled 512-pixel models use the name `sam2.1_tiny_<component>_512_<precision>.onnx`.
The INT8 Image Encoder mixes integer MatMul, QDQ Conv, and FP32 operations. Softmax and LayerNorm compute in FP32 with quantized inputs and outputs.
Memory Attention uses integer MatMul with FP32 RoPE. Auxiliary models remain FP32.

### Benchmarks

Measured on 2026-11-20 on i.MX95 with model revision `coco-val2017-v1`.
NXP ORT 1.24.3 / ailia SDK 1.7 (NXP NPU-enabled version), six CPU threads, profiling disabled.
The ailia runtime is `ac22f5086`, with NEON-optimized CPU integer MatMul.
All backends run one inference with identical dummy inputs when opening each model. They then warm up on the complete five-frame Bedroom sequence using the same two points. Results are medians of the next three runs; warm-up is excluded.

```sh
PYTHONPATH=./python python3 scripts/benchmark_sam2.py --backend ailia_npu --output /tmp/sam-results.json
```

| Component | Calls / video | ORT CPU | ORT NPU | ailia CPU | ailia NPU |
|---|---:|---:|---:|---:|---:|
| Image Encoder (ms / call) | 5 | 2,279.8 | 2,273.9 | 2,474.7 | 877.8 |
| Mask Decoder (ms / call) | 5 | 173.1 | 171.5 | 125.8 | 126.9 |
| Memory Attention (ms / call) | 4 | 1,503.1 | 1,502.2 | 1,078.2 | 517.6 |
| Memory Encoder (ms / call) | 5 | 306.5 | 292.0 | 237.4 | 240.7 |
| Full five-frame sequence (s) | — | 20.23 | 20.14 | 18.87 | 8.67 |

Per-model values are medians of each run's average time per call. Full-sequence timing includes image loading and pre/post-processing, but excludes model loading, rendering, and saving results.
All four backends use the same models. NPU mode selects the NPU backend for Image Encoder and Memory Attention; other models use the CPU backend.

Earlier Neutron EP runs produced incorrect tracking masks. These timing measurements are not an accuracy qualification.

Earlier mean mask IoU measurements against the FP32 reference on Bedroom were 96.07% for host ORT CPU and 95.71% for ailia NPU on the device.
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

Enter English category names separated by periods, then press **Detect**.
The bundled example uses `Horse. Clouds. Grasses. Sky. Hill.`.
Bounding boxes, labels, and confidence scores appear on the image. **Compare all 4** measures the same image and prompt on each backend.

The Swin-T model uses a 400×599 letterboxed image and up to 256 text tokens.
Its 347 fixed-weight MatMuls use tensor-wise UINT8 activations and channel-wise INT8 weights through standard ONNX `MatMulInteger` nodes. Other operations remain FP32.
The graph is rearranged to use the SDK's existing DeformableAttention fusion.
Dynamic Attention placement is selected automatically by the runtime.
ORT NPU outputs may vary between inferences.

### Benchmarks

To measure all four backends without opening the GUI:

```sh
PYTHONPATH=./python python3 scripts/benchmark_grounding.py --output /tmp/grounding-results.json
```

Measured on i.MX95 on 2026-11-20 with revision `grounding-coco-val2017-v1`,
the bundled ailia SDK 1.7 (`ac22f5086`), and NXP ORT 1.24.3.
Six CPU threads, profiling disabled, one excluded dummy inference at Open,
then the median of three runs. Input: `images/grounding_dino/demo.jpg`, prompt:
`Horse. Clouds. Grasses. Sky. Hill.`. Loading and pre/post-processing are excluded.

| Runtime | CPU (ms) | NPU (ms) |
|---|---:|---:|
| ailia SDK 1.7 | 8,386.0 | 4,068.0 |
| NXP ONNX Runtime 1.24.3 | 7,553.2 | 7,549.9 |

All four runs produced detections on the horse image. This is a functional check, not a detection-accuracy benchmark.
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

## Depth Anything V2 Metric Depth Small

### Controls

Select **Depth Anything V2** or run `sh launch.sh --model depth`. The indoor
Hypersim Small model starts with the Bedroom image. Open an image, press
**Estimate depth**, or **Compare all 4**. Toggle **Show depth map** to see the
original image; click a pixel to read its estimated distance.
All backends share the selected display range (0–5 m by default; 10 m and
20 m are also available). Dark means near, bright means far. Changing the
display range only changes colors, not the estimated distances.

### Model

Input is RGB, square-resized to 518 × 518 with ImageNet normalization.
COCO val2017 global Min/Max calibration quantizes the 48 fixed-weight MatMuls
to static tensor-wise UINT8 activations and channel-wise INT8 weights, expressed
as standard ONNX QuantizeLinear / MatMulInteger / Cast / Mul nodes.
The 24 dynamic Attention MatMuls explicitly quantize **both** operands with
DynamicQuantizeLinear (tensor-wise UINT8), compute MatMulInteger with both
zero points, then Cast to FLOAT and multiply by the two scales.
ailia NPU executes these integer MatMuls using tiled, uncompressed weights.
Ordinary FP32 MatMul is not silently quantized by the runtime.
Softmax and convolutions remain floating point. The graph is not fully INT8.
The explicit model quantization can change depth estimates;
this demo does not validate absolute distance accuracy.

### Benchmarks

COCO-calibrated model, Bedroom frame `00000.jpg`, 518 × 518, six threads.
Measured on 2026-11-20 with ailia runtime `c97d853c4` and NXP ORT 1.24.3.
One excluded Open-time dummy inference, then the median of three runs;
profiling, loading, preprocessing and drawing are excluded.

| Runtime | Inference (ms) |
|---|---:|
| ailia CPU | 3381.8 |
| ailia NPU | **2693.5** |
| NXP ORT CPU | 3150.0 |
| NXP ORT NPU (Neutron EP) | 3093.7 |

ORT's NPU selection logged a Neutron allocation failure and used CPU fallback.
On this image, ailia CPU and NPU outputs are identical. Against ORT CPU on the
same explicit-quantization model, mean absolute output difference is 2.93 cm
(51.71 cm maximum). Against the previous ailia NPU model, it is 33.83 cm
(135.63 cm maximum): tensor-wise ONNX quantization is not numerically equivalent
to the removed implicit quantizer. These are output comparisons, not ground-truth
depth accuracy measurements. Previous ailia NPU latency was 2626.1 ms on this image.

### Model Conversion

Download the original ailia-models ONNX, then calibrate with **all 5,000 COCO
val2017 images** on the development host:

```sh
python3 -m pip install -r scripts/requirements-calibration.txt
curl -fL https://storage.googleapis.com/ailia-models/depth_anything_v2/depth_anything_v2_metric_hypersim_vits.onnx \
  -o /path/to/depth_anything_v2_metric_hypersim_vits.onnx
python3 scripts/calibrate_depth.py \
  --source /path/to/depth_anything_v2_metric_hypersim_vits.onnx \
  --coco-images /path/to/coco/val2017 \
  --coco-annotations /path/to/coco/annotations/instances_val2017.json \
  --output /path/to/depth-calibrated
```

After `completion.json` confirms 5,000 images, copy
`depth_anything_v2_metric_hypersim_vits_518_int8.onnx` into `models/`.
Restart the demo after replacing the model.
Rerun the same command to resume. Image hashes, ranges and completion checks
are saved in the output folder; no COCO images are bundled with the demo.

To regenerate the model using the already completed COCO calibration ranges:

```sh
python3 scripts/depth_convert.py \
  --source /path/to/depth_anything_v2_metric_hypersim_vits.onnx \
  --ranges calibration/depth_anything_v2/ranges.json \
  --output models/depth_anything_v2_metric_hypersim_vits_518_int8.onnx
```

## Demo Images and Licenses

Grounding DINO uses `images/grounding_dino/demo.jpg` and YOLOX uses
`images/yolox/input.jpg`; ViT uses `images/vit/input.jpg`, copied from their respective ailia-models examples.
Selecting a model loads its example image; **Open Image** selects a custom input.
SAM 2 and Depth Anything use images from the official Segment Anything repositories:
[truck.jpg](https://github.com/facebookresearch/segment-anything/blob/main/notebooks/images/truck.jpg) and
[Bedroom](https://github.com/facebookresearch/sam2/tree/main/notebooks/videos/bedroom).
See [NOTICE.md](NOTICE.md) for attribution and licenses.

## Source Layout

`app.py` is the shared GUI; `runtime.py` manages inference sessions and worker requests.
Model-specific code lives in `scenes/`: `sam2.py`, `sam2_video.py`,
`grounding_dino.py`, `depth_anything_v2.py`, `yolox.py`, and `vit.py`.
`scripts/` contains model conversion and benchmark commands; `images/` contains demo inputs.
