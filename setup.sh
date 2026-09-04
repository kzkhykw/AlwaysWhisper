#!/bin/sh
# Set up alwayswhisper for local development/use: create a .venv and install
# alwayswhisper into it in editable mode.
#
# Usage:
#   ./setup.sh              # create .venv, install alwayswhisper
#   ./setup.sh large-v3     # also prefetch the "large-v3" model afterward
set -e

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$SCRIPT_DIR"

if command -v python3.12 >/dev/null 2>&1; then
    PYTHON=python3.12
else
    PYTHON=python3
fi

echo "Creating .venv with $PYTHON..."
"$PYTHON" -m venv .venv

echo "Installing alwayswhisper (editable) into .venv..."
.venv/bin/pip install -e .

echo
echo "Done. alwayswhisper is installed in $SCRIPT_DIR/.venv"
echo
echo "Next steps:"
echo "  source .venv/bin/activate"
echo "  alwayswhisper --help"
echo
echo "Models auto-download on first use (to ~/.cache/huggingface/hub)."
echo "To pre-download one now:"
echo "  alwayswhisper prefetch --model large-v3"

if [ -n "$1" ]; then
    echo
    echo "Prefetching model '$1'..."
    .venv/bin/alwayswhisper prefetch --model "$1"
fi
