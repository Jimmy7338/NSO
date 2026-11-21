#!/usr/bin/env bash
set -euo pipefail

here=$(cd -- "$(dirname -- "$0")" && pwd)
destination=${1:?provide a new output directory}
python_exe=${NSO_SCIENCE_PYTHON:-/home/ouc/nso_robot/envs/science312/bin/python}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg

"$python_exe" "$here/evaluate_mechanism.py" "$destination"
"$python_exe" "$here/audit_selection.py" "$destination/endpoint_rows.json" "$destination/selection_audit.json"
"$python_exe" "$here/gate.py" "$destination/endpoint_rows.json" "$destination"
"$python_exe" "$here/diagnose_gate.py" "$destination/endpoint_rows.json" "$destination/diagnosis.json"
"$python_exe" "$here/plot_report.py" "$destination/endpoint_rows.json" "$destination/mechanism_deltas"
"$python_exe" "$here/index_receipts.py" "$destination/original_receipt_index.json"
