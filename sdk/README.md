# ailia SDK binary

`linux/arm64-v8a/libailia.so`: ailia SDK 1.7 (NXP NPU対応版), Linux ARM64.
Includes the optimized batched FP32 2×2 MatMul/NEON path used by SAM 2 RoPE.
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
