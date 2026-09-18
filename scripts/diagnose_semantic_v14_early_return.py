#!/usr/bin/env python3
"""Observe denial/phase events during an exact replay; never change actions."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nso.runtime_integration import NSORuntimeIntegration
from scripts.collect_semantic_v14_cold_start import execute
from scripts.collect_semantic_gain_v13_history import sha, write


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((args.source / 'manifest.json').read_text())
    if manifest['status'] != 'complete':
        raise ValueError('completed physical history required')
    for name, expected in manifest['source_sha256'].items():
        if sha(Path(name)) != expected:
            raise ValueError(f'frozen source changed: {name}')
    for name, expected in json.loads((args.source / 'artifact_hashes.json').read_text()).items():
        if sha(args.source / name) != expected:
            raise ValueError(f'input changed: {name}')
    original = NSORuntimeIntegration.next_local_action
    trace = []

    def observed_next(runtime, scene_idx):
        state = runtime.states[scene_idx]
        backend = runtime.components._cpu_backend
        offset, before = len(backend.calls), state['phase']
        action = original(runtime, scene_idx)
        events = [c for c in backend.calls[offset:]
                  if c['method'] == 'assess_local_action' and not c['outputs']['allowed']]
        if events or before != state['phase']:
            trace.append(dict(action_id=state['packet'].action_id, phase_before=before,
                phase_after=state['phase'], action_returned=action,
                remaining_budget=backend.scenes[scene_idx]['ledger'].remaining_budget,
                denied_checks=[c['outputs'] for c in events]))
        return action

    NSORuntimeIntegration.next_local_action = observed_next
    try:
        summary = execute(args.source, manifest['protocol'], replay=True)
    finally:
        NSORuntimeIntegration.next_local_action = original
    result = dict(status='passed_exact_replay', observation_only_wrapper=True,
                  trajectory_changed=False, events=trace, history_summary=summary,
                  source_inventory_sha256=sha(args.source / 'artifact_hashes.json'),
                  diagnostic_source_sha256=sha(Path(__file__)))
    write(args.output / 'result.json', result)
    (args.output / 'source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(trace), flush=True)


if __name__ == '__main__':
    main()
