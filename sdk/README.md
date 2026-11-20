# ailia SDK binary

`linux/arm64-v8a/libailia.so`: ailia SDK 1.7 (NXP NPU対応版), Linux ARM64.
Includes NEON INT8 MatMul packing and 4×4/4×8 tiles, plus the batched FP32
2×2 MatMul path used by SAM 2 RoPE.
Also includes spatial ConvInteger, compressed fixed weights, fused UINT8
epilogues/lookup tables, and native integer feature-map operations.
CPU 2D ConvInteger uses bounded spatial tiles, shared NEON SDOT kernels and
cached fixed weights, with exact INT32 accumulation and zero-point correction.
Runtime source revision: `65b87e8b2`, including dynamic-integer fusion and
direct SDOT packing, refactored into shared CPU execution, ordered output
conversion and shape-based dispatch, with the automatic NXP execution policy
(standard ONNX integer-operator backend). NPU placement, weight staging and
device requantization no longer read tuning environment variables. The former
CPU LayerNorm tuning overrides are also removed.
Floating MatMul and fused floating Attention are not implicitly quantized.
Dynamic integer execution requires explicit ONNX quantization operators.
Small or narrow dynamic integer products use fused NEON MatMul/Cast/scaling
with parallel direct RHS packing; sufficiently large products use NPU tiles.
Constant and dynamic quantization coefficients share this execution policy
and CPU kernel, including ordered FLOAT/FP16/UINT8 output conversion.
Standalone CPU MatMulInteger caches packed constant weights and shares the direct
NEON RHS packer; dynamic weights and zero points remain per-inference inputs.
The bundled integer models require this runtime; replace models and library
together. Dedicated static-QDQ MatMul/Conv NPU pattern recognition was removed.
This is not the public pip SDK binary. See `../NOTICE.md` and the SDK terms
in `../licenses/`. No license key, Python binding or Neutron driver is bundled here.

Built/tested on NXP i.MX Distro 6.18-wrynose. Required ELF symbol versions
include GLIBC_2.43, GLIBCXX_3.4.32 and CXXABI_1.3.15. Older BSP images require
a compatible rebuild; replacing the library alone is not sufficient.
Unneeded symbols have been stripped from this distribution copy.

Install the Python binding with pip, then copy this library into it:

```sh
python3 -m pip install --target ./python --no-deps ailia==1.7.0.0
python3 scripts/copy_sdk.py --install-npu --package-dir ./python/ailia
```

To copy **from pip** instead, before installing the NPU replacement:

```sh
python3 scripts/copy_sdk.py --export-pip --package-dir ./python/ailia
# Writes sdk/from-pip/linux/arm64-v8a/libailia.so (ignored by Git).
```

For a normal virtual-environment installation, omit `--package-dir` and use
that environment's Python. `--export-pip` copies the currently installed
library; after replacement it would export the NPU build, not the public one.
The public pip version does not provide the NXP NPU backend.

Run these commands on the i.MX95, with the demo stopped. The copy only
replaces `libailia.so`; other pip-installed files are left untouched.
Do not commit a license file or a copied entire Python environment.
