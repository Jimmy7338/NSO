#!/usr/bin/env python3
"""Replay the two saved V26 G histories through V26 and V27 G runtimes.

This is an offline policy-equivalence audit. It constructs no world, issues no
new physical action, extracts no mesh and evaluates no C/Q/J.
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import argparse
import fcntl
from pathlib import Path
import sys
from types import SimpleNamespace

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.run_observed_autonomous_v26 as collector
from nso.observed_runtime_v26 import ObservedANSRuntimeV26
from nso.observed_runtime_v27 import ObservedANSRuntimeV27
from nso.decision_replay_v13 import load_packet

SOURCE = ROOT/'audit_results/observed_autonomous_v26_20260916'
OUTPUT = ROOT/'audit_results/observed_v27_geometry_reference_20260916'
CASES = (0, 2)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def execute(source, output):
    require(not output.exists(), 'fresh output directory required')
    require(collector.shutil.disk_usage(ROOT).free >= 66*1024**2,
            '2 MiB output plus 64 MiB reserve required')
    collector.verify_inventory(source)
    manifest = collector.read(source/'manifest.json')
    require(manifest['status'] == 'complete', 'completed V26 batch required')
    collector.verify_mapping(manifest['source_sha256'])
    own = [Path(__file__).resolve(), ROOT/'nso/observed_feedback_v27.py',
           ROOT/'nso/observed_planner_v27.py', ROOT/'nso/observed_runtime_v27.py',
           ROOT/'docs/research/V27_AUTONOMOUS_EFFECT_PROTOCOL_20260916.md']
    sources = {str(path.relative_to(ROOT)): collector.sha(path) for path in own}
    inputs = {str((source/'manifest.json').relative_to(ROOT)): collector.sha(source/'manifest.json'),
              str((source/'artifact_hashes.json').relative_to(ROOT)): collector.sha(source/'artifact_hashes.json')}
    output.mkdir()
    collector.write(output, output/'manifest.json', dict(status='running', source=str(source),
        cases=list(CASES), source_sha256=sources, input_sha256=inputs,
        new_worlds=0, new_physical_actions=0, mesh_extractions=0, Q_evaluations=0))
    rows = []
    total_packets = 0
    for index in CASES:
        folder = source/f'case_{index:02d}'
        collector.verify_inventory(folder)
        result = collector.read(folder/'result.json')
        require(result['mode'] == 'G' and result['eligible'], 'eligible V26 G reference required')
        inputs[str((folder/'artifact_hashes.json').relative_to(ROOT))] = collector.sha(folder/'artifact_hashes.json')
        config = SimpleNamespace(**result['public_config'])
        old = ObservedANSRuntimeV26(tuple(result['shape']), config, 400, 'G', 5)
        new = ObservedANSRuntimeV27(tuple(result['shape']), config, 400, 'G', 5)
        old_plan_count = new_plan_count = 0
        for action_id, expected in enumerate(result['trace']):
            packet = load_packet(folder/'packets'/f'{action_id:04d}.npz')
            old.accept(packet)
            new.accept(packet)
            require(old.state.geometry_sha256 == new.state.geometry_sha256 == expected['geometry_sha256'],
                    f'case {index} geometry differs at {action_id}')
            old_action = old.next_action()
            new_action = new.next_action()
            require(old_action == new_action == expected['next_action'],
                    f'case {index} next action differs at {action_id}')
            old_plans = old.plans[old_plan_count:]
            new_plans = new.plans[new_plan_count:]
            require(collector.stable(old_plans) == collector.stable(new_plans),
                    f'case {index} inherited global plan differs at {action_id}')
            old_plan_count, new_plan_count = len(old.plans), len(new.plans)
            require(old.planner.feedback == new.planner.feedback == {},
                    f'case {index} G feedback table must remain empty')
        require(old.summary()['actual_paid_actions'] == new.summary()['actual_paid_actions'] == result['paid_actions'],
                'paid action accounting differs')
        require(old.terminal_reason == new.terminal_reason == result['summary']['terminal_reason'],
                'terminal reason differs')
        total_packets += len(result['trace'])
        rows.append(dict(index=index, assignment=result['assignment'], packets=len(result['trace']),
            paid_actions=result['paid_actions'], planning_states=len(old.plans),
            all_geometry_sha256_equal=True, all_global_plans_equal=True,
            all_next_actions_equal=True, terminal_reason_equal=True,
            v26_feedback_empty=True, v27_feedback_empty=True,
            historical_result_sha256=collector.sha(folder/'result.json'),
            historical_inventory_sha256=collector.sha(folder/'artifact_hashes.json')))
        collector.verify_inventory(folder)
    collector.verify_inventory(source)
    for name, expected in {**manifest['source_sha256'], **sources, **inputs}.items():
        require(collector.sha(ROOT/name) == expected, 'source/input changed during audit: '+name)
    result = dict(status='complete_v26_v27_G_policy_equivalence', cases=rows,
        all_cases_equal=all(all(row[key] for key in ('all_geometry_sha256_equal',
            'all_global_plans_equal', 'all_next_actions_equal', 'terminal_reason_equal',
            'v26_feedback_empty', 'v27_feedback_empty')) for row in rows),
        saved_packets_consumed=total_packets, offline_mapper_integrations_v26=total_packets,
        offline_mapper_integrations_v27=total_packets,
        total_offline_mapper_integrations=2*total_packets,
        new_worlds=0, new_physical_actions=0, new_sensor_packets=0,
        mesh_extractions=0, Q_evaluations=0,
        conclusion='Historical V26 G is an exact deterministic reference for V27 under these saved observations.',
        limitation='Saved-history policy equivalence is not an independent environment or efficacy result.')
    collector.write(output, output/'result.json', result)
    collector.write(output, output/'input_sha256.json', inputs)
    collector.write(output, output/'source_sha256.json', sources)
    collector.write(output, output/'manifest.json', dict(status='complete', source=str(source),
        cases=list(CASES), source_sha256=sources, input_sha256=inputs,
        new_worlds=0, new_physical_actions=0, mesh_extractions=0, Q_evaluations=0))
    collector.seal(output)
    print(result, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    if not args.run:
        parser.error('explicit --run required')
    with (ROOT/'audit_results/.observed_autonomous_v26.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        execute(args.source.resolve(), args.output.resolve())


if __name__ == '__main__':
    main()
