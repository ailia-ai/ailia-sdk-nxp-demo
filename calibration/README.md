# Calibration records

All four bundled models use all 5,000 COCO val2017 images. Conversion commands
are documented in the main README.

| Model | Directory | Records |
| --- | --- | --- |
| SAM 2.1 Tiny | `sam2/` | Image manifest, four range shards, completion report |
| Grounding DINO | `grounding_dino/` | Image manifest, merged ranges, completion report |
| Depth Anything V2 | `depth_anything_v2/` | Image manifest, merged ranges, completion report |
| YOLOX-S 320 | `yolox/` | Report containing image list, quantization parameters and model hash |

These are the original calibration records; converter hashes identify the code
used at calibration time. No recalibration or model changes were performed.
SAM's completion report uses the original export names; `models/manifest.json`
maps the distributed models and hashes.

COCO images, temporary models, worker logs and benchmark logs are not included.
