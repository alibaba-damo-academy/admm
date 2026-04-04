#!/bin/sh
set -e

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

echo "=== Building ADMM Python Package ==="
echo ""

echo "=== Step 1: Building package ==="
pip install . -r requirements.txt
echo "Package built and installed successfully"
echo ""

# Build documentation
echo "=== Step 2: Building documentation ==="
bash docs/build.sh
echo ""
echo "=== Build completed successfully ==="
