#!/usr/bin/env python3
"""Execute the fixed remaining T contexts, one fully replayed batch at a time.

This driver never reads scores or changes any acquisition rule. A failure stops
the sequence and leaves every partial artifact in place for a separate review.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def verified(run):
    record = read(run / 'verification.json')
    assert record['status'] == 'passed_full' and record['passed_full'], 'full independent replay required'
    assert record['artifact_manifest_sha256'] == sha(run / 'artifact_hashes.json')
    assert read(run / 'metadata.json')['status'] == 'complete'
    return record


def main(args):
    config = read(ROOT / 'configs/virtual3d/response_v7_pipeline.json')
    first = args.runs / 'T0'
    verified(first)
    if args.audit.exists():
        raise FileExistsError(args.audit)
    args.audit.mkdir(parents=True)
    inventory = read(first / 'metadata.json')['source_sha256']
    inventory = dict(inventory) | {
        'scripts/batch_response_v7_training.py': sha(Path(__file__)),
        'scripts/replay_response_v7.py': sha(ROOT / 'scripts/replay_response_v7.py'),
        'scripts/replay_counterfactual_views.py': sha(ROOT / 'scripts/replay_counterfactual_views.py'),
    }
    report = {'status': 'running', 'fixed_context_order': config['train_contexts'],
              'source_sha256': inventory, 'completed': ['T0'], 'failures': [],
              'scores_read_by_driver': False, 'raw_or_mesh_artifacts_removed': False}
    write(args.audit / 'status.json', report)
    started = time.time()

    def sources_unchanged():
        for name, digest in inventory.items():
            assert sha(ROOT / name) == digest, f'frozen source changed: {name}'

    def command(context, stage, argv):
        sources_unchanged()
        with (args.audit / f'{context}_{stage}.log').open('w') as output:
            result = subprocess.run([sys.executable, *argv], cwd=ROOT, stdout=output,
                                    stderr=subprocess.STDOUT, env=os.environ | {
                                        'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
                                        'PYTHONDONTWRITEBYTECODE': '1'})
        if result.returncode:
            raise RuntimeError(f'{context} {stage} failed; retained log and partial artifacts')

    context = 'T0'
    try:
        sources_unchanged()
        for context in config['train_contexts'][1:]:
            free = shutil.disk_usage(args.runs).free
            # T0 observed size plus 50% contingency, in addition to the reserve.
            first_size = sum(p.stat().st_size for p in first.rglob('*') if p.is_file())
            if free < config['storage']['reserve_bytes'] + 1.5 * first_size:
                raise RuntimeError('insufficient reserve for a complete next context; no candidate skipped')
            prepared, run = args.preparations / context, args.runs / context
            command(context, 'prepare', ['scripts/prepare_response_v7.py', '--context', context,
                                         '--output', str(prepared)])
            p = read(prepared / 'metadata.json')
            assert p['status'] == 'complete' and p['candidate_structure_gate_passed'], 'preparation gate failed'
            command(context, 'execute', ['scripts/eval_response_v7.py', '--prepared', str(prepared),
                                         '--output', str(run)])
            command(context, 'replay', ['scripts/replay_response_v7.py', '--run', str(run), '--workers', '1'])
            verified(run)
            report['completed'].append(context)
            report['elapsed_s'] = time.time() - started
            write(args.audit / 'status.json', report)
            print('completed and independently replayed', context, flush=True)
        sources_unchanged()
        report['status'] = 'complete'
    except Exception as error:
        report['status'] = 'stopped'
        report['failures'].append({'context': context, 'error': f'{type(error).__name__}: {error}'})
        raise
    finally:
        report['elapsed_s'] = time.time() - started
        write(args.audit / 'status.json', report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=lambda x: Path(x).resolve(), required=True)
    parser.add_argument('--preparations', type=lambda x: Path(x).resolve(), required=True)
    parser.add_argument('--audit', type=lambda x: Path(x).resolve(), required=True)
    main(parser.parse_args())
