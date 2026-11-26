#!/usr/bin/env bash
set -euo pipefail

evaluation_dir=$(cd -- "$(dirname -- "$0")" && pwd)
status_file=${1:?provide the active supervisor status.json}
python_exe=/home/ouc/nso_robot/envs/science312/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

# Run only after the final attempt manifest has been closed by the supervisor.
"$python_exe" - "$status_file" <<'PY'
import json
import sys
import time
from pathlib import Path

path = Path(sys.argv[1])
while True:
    status = json.loads(path.read_text())
    if status['status'] == 'finished':
        assert len(status['jobs_completed']) == 12
        break
    if not Path('/proc', str(status['supervisor_pid'])).exists():
        raise RuntimeError(f'No live scheduler for {path}. Inspect the authoritative run receipts.')
    time.sleep(12)
PY

out="$evaluation_dir/results"
"$python_exe" "$evaluation_dir/evaluate_mechanism.py" "$out"
"$python_exe" "$evaluation_dir/audit_selection.py" "$out/endpoint_rows.json" "$out/selection_audit.json"
"$python_exe" "$evaluation_dir/gate.py" "$out/endpoint_rows.json" "$out"
"$python_exe" "$evaluation_dir/diagnose_gate.py" "$out/endpoint_rows.json" "$out/diagnosis.json"
"$python_exe" "$evaluation_dir/plot_deltas.py" "$out/endpoint_rows.json" "$out/mechanism_deltas"
