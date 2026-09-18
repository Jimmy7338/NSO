#!/usr/bin/env python3
"""Replay sealed V14 feature bundles without simulator/runtime/evaluator state."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from nso.semantic_opportunities_v14 import route_instance_features


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((args.source / 'manifest.json').read_text())
    if manifest['status'] != 'complete':
        raise ValueError('complete acquisition required')
    for name, expected in manifest['source_sha256'].items():
        if sha(Path(name)) != expected:
            raise ValueError(f'feature acquisition source changed: {name}')
    inventory = json.loads((args.source / 'artifact_hashes.json').read_text())
    for name, expected in inventory.items():
        if sha(args.source / name) != expected:
            raise ValueError(f'feature artifact changed: {name}')
    states, candidates = 0, 0
    for path in sorted(args.source.glob('*/inputs.json')):
        inputs = json.loads(path.read_text())
        with np.load(path.parent / 'observed_map.npz', allow_pickle=False) as data:
            mapper = SimpleNamespace(config=SimpleNamespace(**inputs['config']), belief=data['belief'].copy())
        actual = route_instance_features(mapper, inputs['candidates'], inputs['observed_assets'], inputs['descriptors'])
        expected = json.loads((path.parent / 'features.json').read_text())
        if actual != expected:
            raise ValueError(f'feature replay differs: {path.parent}')
        states += 1
        candidates += len(actual)
    summary = json.loads((args.source / 'summary.json').read_text())
    if (states, candidates) != (summary['states'], summary['candidate_descriptors']) or states == 0:
        raise ValueError('incomplete feature replay')
    result = dict(status='passed', states=states, candidate_descriptors=candidates,
                  exact_feature_replay=True, simulator_constructed=False, future_rewards_read=False,
                  replay_kind='separate process, same feature implementation; BFS checked independently in unit tests',
                  source_inventory_sha256=sha(args.source / 'artifact_hashes.json'),
                  replay_source_sha256=sha(Path(__file__)), semantic_efficacy_proven=False)
    (args.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    (args.output / 'replay_source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(result))


if __name__ == '__main__':
    main()
