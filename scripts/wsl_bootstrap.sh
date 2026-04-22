#!/usr/bin/env bash
# Bootstrap upstream OGBench under external/ogbench_full inside WSL.
set -euo pipefail

REPO="/mnt/c/Users/Administrator/Documents/A.Personal-F/NLP/ece567"
TARGET="$REPO/external/ogbench_full"
PATCH="$REPO/patches/ogbench_local.patch"

if [ ! -d "$TARGET/.git" ]; then
  echo "Cloning upstream OGBench into $TARGET"
  git clone https://github.com/seohongpark/ogbench.git "$TARGET"
else
  echo "Using existing checkout at $TARGET"
fi

cd "$TARGET"
if git apply --reverse --check "$PATCH" >/dev/null 2>&1; then
  echo "Patch already applied."
else
  echo "Applying $PATCH"
  git apply "$PATCH"
fi

# Overlay our vendored chiql.py + patched networks.py on top of upstream.
# Anything we explicitly track under external/ogbench/ wins.
echo "Overlaying vendored files from external/ogbench/ onto external/ogbench_full/"
cp -v "$REPO/external/ogbench/impls/agents/chiql.py"     "$TARGET/impls/agents/chiql.py"

echo "DONE"
