"""Public model downloads are verified and safe to retry."""

import hashlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.download_models import download


class DownloadTests(unittest.TestCase):
    def test_download_and_offline_cache(self):
        payload = b"public model"
        expected = hashlib.sha256(payload).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.onnx"
            with patch("urllib.request.urlopen", return_value=io.BytesIO(payload)) as request:
                download(path, "https://example.com/model.onnx", expected)
                download(path, "https://example.com/model.onnx", expected)
                request.assert_called_once()
            self.assertEqual(path.read_bytes(), payload)

    def test_bad_download_preserves_existing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.onnx"
            path.write_bytes(b"existing")
            with patch("urllib.request.urlopen", return_value=io.BytesIO(b"corrupt")):
                with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                    download(path, "https://example.com/model.onnx", "0" * 64)
            self.assertEqual(path.read_bytes(), b"existing")
            self.assertEqual(list(Path(directory).iterdir()), [path])
