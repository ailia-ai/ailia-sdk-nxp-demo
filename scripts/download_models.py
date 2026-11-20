#!/usr/bin/env python3
"""Download unmodified public models from ailia-models, verifying their hashes."""

import argparse
import hashlib
import json
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(path, url, expected):
    """Verify cached files and replace them atomically only after verification."""
    if path.is_file() and sha256(path) == expected:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as output:
            temporary = Path(output.name)
            with urllib.request.urlopen(url, timeout=60) as response:
                for block in iter(lambda: response.read(1024 * 1024), b""):
                    output.write(block)
        if sha256(temporary) != expected:
            raise ValueError(f"SHA-256 mismatch: {path.name}")
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path, default=ROOT / "models")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "models" / "manifest.json").read_text())
    for name, record in manifest["downloads"].items():
        download(args.models_dir / name, record["url"], record["sha256"])
        print(f"Verified: {name}")


if __name__ == "__main__":
    main()
