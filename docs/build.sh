#!/bin/sh
set -e

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

# Generate RST files from Python modules
python -c "import genrst; genrst.writeRst()"

# Build HTML documentation
echo "=== Building HTML documentation ==="
rm -rf "./_build/html"
python -m sphinx -b dirhtml ./ "./_build/html"
echo "HTML output: $SCRIPT_DIR/_build/html"

# Build PDF documentation
echo ""
echo "=== Building PDF documentation ==="
rm -rf "./_build/latex"
python -m sphinx -b latex ./ "./_build/latex"
cd "./_build/latex"
xelatex -interaction=nonstopmode ADMM.tex > xelatex_pass1.log 2>&1 || true
xelatex -interaction=nonstopmode ADMM.tex > xelatex_pass2.log 2>&1 || true
xelatex -interaction=nonstopmode ADMM.tex > xelatex_pass3.log 2>&1 || true
cd "$SCRIPT_DIR"
if [ -f "./_build/latex/ADMM.pdf" ]; then
    echo "PDF output: $SCRIPT_DIR/_build/latex/ADMM.pdf"
else
    echo "PDF build failed: ADMM.pdf not found"
    exit 1
fi
