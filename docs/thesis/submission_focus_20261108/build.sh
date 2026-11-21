#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
for document in main supplement; do
  pdflatex -no-shell-escape -interaction=nonstopmode -halt-on-error "$document.tex"
  pdflatex -no-shell-escape -interaction=nonstopmode -halt-on-error "$document.tex"
done
