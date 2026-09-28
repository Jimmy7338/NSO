#!/usr/bin/env python3
"""Run the bounded P0 contract suites; this does not launch the experiment matrix."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
SUITES = ('tests/test_instance_belief_v40.py', 'tests/test_scene_contract_v40.py',
          'tests/test_replay_export_v40.py')
IMPLEMENTATIONS = ('nso/instance_belief_v40.py', 'nso/scene_contract_v40.py',
                   'scripts/prepare_scene_catalog_v40.py', 'scripts/build_replay_viewer_v40.py',
                   'scripts/verify_saved_instance_interface_v40.py', 'scripts/verify_p0_contracts_v40.py')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def run(output):
    for name in SUITES + IMPLEMENTATIONS:
        if not (ROOT / name).is_file():
            raise FileNotFoundError(name)
    output.mkdir(parents=True, exist_ok=False)
    before = {name: sha(ROOT / name) for name in SUITES + IMPLEMENTATIONS}
    command = [sys.executable, '-B', '-m', 'unittest', 'discover', '-s', 'tests',
               '-p', 'test_*v40.py', '-v']
    environment = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                       MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    started = time.monotonic()
    completed = subprocess.run(command, cwd=ROOT, env=environment, capture_output=True,
                               text=True, timeout=180, check=False)
    log = completed.stdout + completed.stderr
    (output / 'unittest.txt').write_text(log)
    counts = re.findall(r'Ran (\d+) tests?', log)
    after = {name: sha(ROOT / name) for name in before}
    result = dict(status='passed' if completed.returncode == 0 and before == after else 'failed',
                  exit_code=completed.returncode, tests_run=int(counts[-1]) if counts else None,
                  elapsed_seconds=time.monotonic() - started, command=command,
                  source_unchanged_during_tests=before == after,
                  source_sha256=before, log_sha256=sha(output / 'unittest.txt'),
                  new_main_trajectories=0, formal_test_geometry_generated=False,
                  scope='analytic observation/schema/export contracts; no planner efficacy claim')
    write(output / 'result.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result['status'] != 'passed':
        raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'audit_results/v40_contract_gate_20260920')
    run(parser.parse_args().output)
