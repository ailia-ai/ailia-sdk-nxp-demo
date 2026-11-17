"""Install a per-user application shortcut without changing Weston settings."""

import os
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    applications = (
        Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share")) / "applications"
    )
    applications.mkdir(parents=True, exist_ok=True)
    # Desktop Entry Exec escaping, including reserved percent field codes.
    launch = str(root / "launch.sh").replace("\\", "\\\\").replace('"', '\\"')
    launch = launch.replace("$", "\\$").replace("`", "\\`").replace("%", "%%")
    entry = applications / "ailia-sdk-nxp-demo.desktop"
    entry.write_text(
        "[Desktop Entry]\nType=Application\nName=ailia SAM 2.1 Demo\n"
        f'Exec=sh "{launch}"\nIcon={root / "launcher.png"}\n'
        "Terminal=false\nCategories=Graphics;Science;\n",
        encoding="utf-8",
    )
    print(f"Installed {entry}")
    print("Weston panel launchers are configured separately; see README.md.")


if __name__ == "__main__":
    main()
