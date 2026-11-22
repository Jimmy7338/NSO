#!/usr/bin/env python3
"""Advance only from fully recorded development runs to frozen holdout runs."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from study import ROOT, WEIGHTS, freeze_path, read


def wait_for_development(pid):
    receipt = ROOT / 'execution/development/phase_receipt.json'
    supervisor_command = str(ROOT / 'study.py').encode()
    while True:
        try:
            results = read(receipt)
            assert len(results['jobs']) == 20
            if results['status'] != 'completed':
                raise RuntimeError(f'Developer execution needs review: {receipt}')
            return
        except (FileNotFoundError, json.JSONDecodeError):
            cmdline = Path('/proc') / str(pid) / 'cmdline'
            if not cmdline.is_file() or supervisor_command not in cmdline.read_bytes():
                raise RuntimeError(f'Supervisor ended without the complete recorded development matrix: {pid}')
            time.sleep(7)


def call(*args):
    print('Running', *args, flush=True)
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               NUMEXPR_NUM_THREADS='1', PYTHONUNBUFFERED='1')
    subprocess.run([sys.executable, *args], cwd=ROOT, env=env, check=True)


def arms(split):
    return list(dict.fromkeys(row['arm'] for row in read(freeze_path(split))['schedule']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('development_supervisor_pid', type=int)
    options = parser.parse_args()
    wait_for_development(options.development_supervisor_pid)
    assert not WEIGHTS.exists() and not freeze_path('holdout').exists()
    call('measurement/evaluate.py', '--split', 'development', '--arms', *arms('development'),
         '--only-terminal', '--require-finished')
    call('decision/fit_routes.py', 'fit')
    call('study.py', 'fit')
    print('Frozen weights', {key: read(WEIGHTS)[key] for key in ('decision', 'performance')}, flush=True)
    call('study.py', 'seal', 'holdout')
    call('study.py', 'run', 'holdout')
    assert read(ROOT / 'execution/holdout/phase_receipt.json')['status'] == 'completed'
    call('measurement/evaluate.py', '--split', 'holdout', '--arms', *arms('holdout'), '--require-finished')
    call('decision/fit_routes.py', 'evaluate', '--frozen-weights', str(WEIGHTS))
    print('Finished independent terminal and decision evaluation.', flush=True)


if __name__ == '__main__':
    main()
