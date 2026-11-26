"""Desktop launchers point back to the repository, including paths with spaces."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.install_desktop import desktop_directory, install


class DesktopLauncherTests(unittest.TestCase):
    def test_install_copies_executable_logo_shortcut(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "demo with spaces 100% $test"
            entry, shortcut = install(root, base / "applications", base / "Desktop")
            self.assertEqual(entry.read_bytes(), shortcut.read_bytes())
            text = shortcut.read_text()
            self.assertIn(f"Icon={root}/ailia-logo.png", text)
            self.assertIn('Exec=sh "', text)
            self.assertIn("100%%", text)
            self.assertTrue(shortcut.stat().st_mode & 0o111)
            self.assertEqual(install(root, entry.parent, shortcut.parent), (entry, shortcut))

    def test_desktop_fallback_without_xdg(self):
        with patch("scripts.install_desktop.subprocess.run", side_effect=FileNotFoundError):
            self.assertEqual(desktop_directory(), Path.home() / "Desktop")
