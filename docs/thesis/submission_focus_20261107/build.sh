#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
pdflatex -no-shell-escape -interaction=nonstopmode -halt-on-error main.tex
pdflatex -no-shell-escape -interaction=nonstopmode -halt-on-error main.tex
