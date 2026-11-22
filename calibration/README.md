# Calibration records

All four bundled models use all 5,000 COCO val2017 images. Conversion commands
are documented in the main README.

| Model | Directory | Records |
| --- | --- | --- |
| SAM 2.1 Tiny | `sam2/` | Image manifest, four range shards, completion report |
| Grounding DINO | `grounding_dino/` | Fixed-weight records; `attention/` contains the additional full-dataset fixed-attention calibration |
| Depth Anything V2 | `depth_anything_v2/` | Fixed-weight records; additional full-dataset passes in `attention/` and `convolution/` |
| YOLOX-S 320 | `yolox/` | Report containing image list, quantization parameters and model hash |

Converter hashes identify the code used at calibration time. The attention
records cover all 5,000 images independently; Grounding DINO uses both the COCO
80-category and default demo captions on every image. Fixed-weight ranges are
unchanged. Grounding's attention completion report identifies its bundled model.
Depth's attention output is the input to the convolution pass; the convolution
completion report identifies the final bundled model. Its 28 Conv layers use
channel-wise 8-bit weights stored as UINT8 with zero point 128 for ORT compatibility.
SAM's completion report uses the original export names; `models/manifest.json`
maps the distributed models and hashes.

COCO images, temporary models, worker logs and benchmark logs are not included.
