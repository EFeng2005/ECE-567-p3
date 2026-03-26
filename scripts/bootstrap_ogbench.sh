#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TARGET_DIR="${1:-$REPO_ROOT/external/ogbench}"
UPSTREAM_URL="${UPSTREAM_URL:-https://github.com/seohongpark/ogbench.git}"
PATCH_FILE="${PATCH_FILE:-$REPO_ROOT/patches/ogbench_local.patch}"

if [ -d "$TARGET_DIR/.git" ]; then
  echo "Found existing OGBench checkout at $TARGET_DIR"
  if [ -z "$(git -C "$TARGET_DIR" status --short)" ]; then
    echo "Updating existing OGBench checkout"
    git -C "$TARGET_DIR" pull --ff-only
  else
    echo "Skipping git pull because local changes are present in $TARGET_DIR"
  fi
else
  echo "Cloning OGBench into $TARGET_DIR"
  git clone "$UPSTREAM_URL" "$TARGET_DIR"
fi

if [ -f "$PATCH_FILE" ]; then
  if git -C "$TARGET_DIR" apply --reverse --check "$PATCH_FILE" >/dev/null 2>&1; then
    echo "Local OGBench patch already applied: $PATCH_FILE"
  else
    echo "Applying local OGBench patch: $PATCH_FILE"
    git -C "$TARGET_DIR" apply "$PATCH_FILE"
  fi
fi

echo "Upstream OGBench is available at $TARGET_DIR"
echo "Next step:"
echo "  cd \"$TARGET_DIR/impls\""
echo "  pip install -r requirements.txt"
echo "  python main.py --helpfull | less"
