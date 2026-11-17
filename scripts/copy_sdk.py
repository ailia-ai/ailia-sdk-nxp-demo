"""Copy the NPU library into pip's ailia package, or export pip's library.

Only libailia.so is copied. Python bindings, licenses and optional plug-ins
remain those installed by pip. Stop running inference before installing.
"""

import argparse
import importlib.util
import platform
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIBRARY = Path("linux/arm64-v8a/libailia.so")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--install-npu", action="store_true")
    mode.add_argument("--export-pip", action="store_true")
    parser.add_argument(
        "--package-dir", type=Path, help="ailia package directory (not site-packages)"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "sdk/from-pip")
    args = parser.parse_args()
    if platform.system() != "Linux" or platform.machine() not in ("aarch64", "arm64"):
        parser.error("This library is for Linux ARM64 (i.MX95), not this host.")
    package = args.package_dir
    if package is None:
        spec = importlib.util.find_spec("ailia")
        if spec is None or spec.origin is None:
            parser.error("Install ailia with pip first, or specify --package-dir.")
        package = Path(spec.origin).parent
    if not (package / "__init__.py").is_file():
        parser.error(f"Not an ailia Python package: {package}")
    source, destination = (
        (ROOT / "sdk" / LIBRARY, package / LIBRARY)
        if args.install_npu
        else (package / LIBRARY, args.output / LIBRARY)
    )
    if not source.is_file():
        parser.error(f"Library not found: {source}")
    if source.resolve() == destination.resolve():
        parser.error("Source and destination must differ.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    print(f"Copied {source} -> {destination}")


if __name__ == "__main__":
    main()
