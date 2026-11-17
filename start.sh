#!/bin/sh
set -eu
DEMO_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
if [ -z "${WAYLAND_DISPLAY:-}" ]; then
    for socket in "$XDG_RUNTIME_DIR"/wayland-*; do
        if [ -S "$socket" ]; then
            export WAYLAND_DISPLAY="$(basename -- "$socket")"
            break
        fi
    done
fi
export GDK_BACKEND=wayland
export PYTHONPATH="${AILIA_PYTHON_PATH:-$DEMO_DIR/python}${PYTHONPATH:+:$PYTHONPATH}"
export NEUTRON_CMA_512SLOTS=1
export AILIA_NEUTRON_QDQ_STREAM_WEIGHTS=1
export AILIA_NEUTRON_DYNAMIC_ATTENTION=1
if [ -z "${WAYLAND_DISPLAY:-}" ]; then
    echo "No Wayland display socket in $XDG_RUNTIME_DIR. Start Weston or set XDG_RUNTIME_DIR and WAYLAND_DISPLAY." >&2
    exit 1
fi
cd "$DEMO_DIR"
exec python3 app.py "$@"
