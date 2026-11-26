"""Install a per-user application shortcut without changing Weston settings."""

import argparse
import os
import shutil
import subprocess
from pathlib import Path


def desktop_directory():
    try:
        result = subprocess.run(
            ["xdg-user-dir", "DESKTOP"], capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            return Path(result.stdout.strip())
    except (OSError, subprocess.TimeoutExpired):
        pass
    return Path.home() / "Desktop"


def install(root, applications, desktop):
    applications.mkdir(parents=True, exist_ok=True)
    # Desktop Entry Exec escaping, including reserved percent field codes.
    launch = str(root / "launch.sh").replace("\\", "\\\\").replace('"', '\\"')
    launch = launch.replace("$", "\\$").replace("`", "\\`").replace("%", "%%")
    entry = applications / "ailia-sdk-nxp-demo.desktop"
    entry.write_text(
        "[Desktop Entry]\nType=Application\nName=ailia i.MX95 Demo\n"
        f'Exec=sh "{launch}"\nIcon={root / "ailia-logo.png"}\n'
        "Terminal=false\nCategories=Graphics;Science;\n",
        encoding="utf-8",
    )
    entry.chmod(0o755)
    desktop.mkdir(parents=True, exist_ok=True)
    shortcut = desktop / entry.name
    if shortcut != entry:
        shutil.copy2(entry, shortcut)
    return entry, shortcut


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--desktop-dir", type=Path, help="Override the desktop destination")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    applications = (
        Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share")) / "applications"
    )
    for path in dict.fromkeys(install(root, applications, args.desktop_dir or desktop_directory())):
        print(f"Installed {path}")


if __name__ == "__main__":
    main()
