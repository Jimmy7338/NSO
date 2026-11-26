#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
NSO_THESIS_PYTHON="${NSO_THESIS_PYTHON:-python3}"
"$NSO_THESIS_PYTHON" render_math.py
if [ "$("$NSO_THESIS_PYTHON" -c 'import json; print(len(json.load(open("math_render_errors.json"))))')" != 0 ]; then exit 1; fi
NSO_THESIS_PANDOC="${NSO_THESIS_PANDOC:-$(command -v pandoc || true)}"
NSO_THESIS_DATA_ARGS=()
if [ -z "$NSO_THESIS_PANDOC" ]; then
  NSO_THESIS_PANDOC=/home/ouc/nso_robot/tools/pandoc_local_20261029/runtime/usr/bin/pandoc
  NSO_THESIS_DATA_ARGS=(--data-dir /home/ouc/nso_robot/tools/pandoc_local_20261029/runtime/usr/share/pandoc/data)
fi
if [ -n "${NSO_THESIS_PANDOC_DATA_DIR:-}" ]; then
  NSO_THESIS_DATA_ARGS=(--data-dir "$NSO_THESIS_PANDOC_DATA_DIR")
fi
"$NSO_THESIS_PANDOC" "${NSO_THESIS_DATA_ARGS[@]}" thesis_render.md -s -c compact.css --metadata pagetitle='毕业论文整合阅读稿' -o thesis.html
"$NSO_THESIS_PYTHON" fit_images.py
NSO_THESIS_LO_PROFILE="$(mktemp -d -t nso-thesis-lo-XXXXXXXX)"
trap 'rm -rf -- "$NSO_THESIS_LO_PROFILE"' EXIT
libreoffice "-env:UserInstallation=file://$NSO_THESIS_LO_PROFILE" --headless --convert-to pdf --outdir . thesis.html
