#!/bin/sh
set -eu
DEMO_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
# A panel double-click must not start two NPU inference workers.
RUNTIME_DIR=${XDG_RUNTIME_DIR:-/run/user/$(id -u)}
exec flock -n "$RUNTIME_DIR/sam-hdmi-demo.lock" sh "$DEMO_DIR/start.sh" "$@" >> "$DEMO_DIR/demo.log" 2>&1
