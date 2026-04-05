#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash setup_uv_pytorch_jupyter.sh [kernel_name] [display_name]
# Example:
#   bash setup_uv_pytorch_jupyter.sh cs336-lm "Python (CS336 LM)"

if ! command -v uv >/dev/null 2>&1; then
  echo "[error] 'uv' is not installed."
  echo "Install first: curl -LsSf https://astral.sh/uv/install.sh | sh"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR="${VENV_DIR:-.venv}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
PROJECT_NAME="$(basename "$SCRIPT_DIR")"
DEFAULT_KERNEL_NAME="${PROJECT_NAME//[^a-zA-Z0-9_-]/-}"

KERNEL_NAME="${1:-${KERNEL_NAME:-$DEFAULT_KERNEL_NAME}}"
DISPLAY_NAME="${2:-${DISPLAY_NAME:-Python ($PROJECT_NAME)}}"

echo "[1/5] Installing Python ${PYTHON_VERSION} via uv (if needed)..."
uv python install "$PYTHON_VERSION"

echo "[2/5] Creating virtual environment at ${VENV_DIR}..."
uv venv "$VENV_DIR" --python "$PYTHON_VERSION"

PYTHON_BIN="$SCRIPT_DIR/$VENV_DIR/bin/python"
if [ ! -x "$PYTHON_BIN" ]; then
  echo "[error] Python executable not found at: $PYTHON_BIN"
  exit 1
fi

echo "[3/5] Upgrading core packaging tools..."
uv pip install --python "$PYTHON_BIN" --upgrade pip setuptools wheel

echo "[4/5] Installing latest PyTorch + Jupyter Notebook + ipykernel..."
uv pip install --python "$PYTHON_BIN" --upgrade torch torchvision torchaudio notebook ipykernel tiktoken

echo "[5/5] Registering Jupyter kernel..."
"$PYTHON_BIN" -m ipykernel install --user --name "$KERNEL_NAME" --display-name "$DISPLAY_NAME"

echo
echo "Done."
echo "- venv: $VENV_DIR"
echo "- kernel name: $KERNEL_NAME"
echo "- display name: $DISPLAY_NAME"
echo
echo "Next:"
echo "  source \"$VENV_DIR/bin/activate\""
echo "  jupyter notebook"
