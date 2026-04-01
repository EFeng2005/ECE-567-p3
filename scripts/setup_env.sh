#!/usr/bin/env bash
# One-shot environment setup for OGBench on Great Lakes.
# Creates venv, installs dependencies, clones OGBench, applies patch.
#
# Usage:
#   bash scripts/setup_env.sh [VENV_DIR]
#
# Examples:
#   bash scripts/setup_env.sh                                          # default: ./venv
#   bash scripts/setup_env.sh /scratch/$USER/ogbench_venv

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="${1:-$REPO_ROOT/venv}"

echo "===== OGBench Environment Setup ====="
echo "  Repo:  $REPO_ROOT"
echo "  Venv:  $VENV_DIR"
echo ""

# --- Step 1: Load system Python + CUDA ---
module load python/3.10.4 cuda/12.3.0 2>/dev/null || true
echo "[1/4] Modules loaded"

# --- Step 2: Create venv ---
if [ -d "$VENV_DIR" ]; then
  echo "[2/4] Venv already exists at $VENV_DIR, skipping creation"
else
  python3 -m venv "$VENV_DIR"
  echo "[2/4] Venv created"
fi

source "$VENV_DIR/bin/activate"
pip install --upgrade pip setuptools wheel -q

# --- Step 3: Install dependencies ---
echo "[3/4] Installing packages (this may take a few minutes)..."

# OGBench benchmark package
pip install ogbench -q

# JAX with CUDA 12
pip install "jax[cuda12]>=0.4.26" -q

# Training dependencies
pip install "flax>=0.8.4" "distrax>=0.1.5" ml_collections matplotlib moviepy wandb -q

echo "  Installed: ogbench, jax[cuda12], flax, distrax, ml_collections, matplotlib, moviepy, wandb"

# --- Step 4: Clone and patch OGBench ---
echo "[4/4] Setting up OGBench source..."
bash "$REPO_ROOT/scripts/bootstrap_ogbench.sh"

# --- Verify ---
echo ""
echo "===== Verification ====="
python -c "import jax; print(f'  JAX: {jax.__version__}')"
python -c "import ogbench; print('  ogbench: OK')"
python -c "import flax; print(f'  flax: {flax.__version__}')"

echo ""
echo "===== Setup Complete ====="
echo ""
echo "Before submitting jobs, export these environment variables:"
echo ""
echo "  export VENV_PATH=\"$VENV_DIR\""
echo "  export REPO_ROOT=\"$REPO_ROOT\""
echo ""
echo "If using a different SLURM account:"
echo ""
echo "  export ACCOUNT=\"your_account\""
echo ""
echo "Then run:"
echo "  bash scripts/submit_phaseD.sh    # or whichever phase script"
