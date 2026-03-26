#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TARGET_DIR="${1:-$REPO_ROOT/external/ogbench}"
UPSTREAM_URL="${UPSTREAM_URL:-https://github.com/seohongpark/ogbench.git}"

if [ -d "$TARGET_DIR/.git" ]; then
  echo "Updating existing OGBench checkout at $TARGET_DIR"
  git -C "$TARGET_DIR" pull --ff-only
else
  echo "Cloning OGBench into $TARGET_DIR"
  git clone "$UPSTREAM_URL" "$TARGET_DIR"
fi

echo "Upstream OGBench is available at $TARGET_DIR"
echo "Next step:"
echo "  cd \"$TARGET_DIR/impls\""
echo "  pip install -r requirements.txt"

