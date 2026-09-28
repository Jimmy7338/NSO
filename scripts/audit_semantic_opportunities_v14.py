#!/usr/bin/env python3
"""Restore frozen development prefixes and audit V14 inputs, without rewards."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from time import perf_counter
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np

from nso.cpu_sensor_contract_v10 import digest, json_value
from nso.decision_replay_v13 import decision_state, load_packet
from nso.semantic_opportunities_v14 import (
    COMMON_NAMES, KERNEL_NAMES, observed_descriptors, route_instance_features,
)
from scripts.collect_semantic_gain_v13_continuations import restore
from scripts.collect_semantic_gain_v13_history import sha, write


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    protocol_path = Path('configs/virtual3d/semantic_gain_v13_2_outbound_information_protocol.json')
    protocol = json.loads(protocol_path.read_text())
    sources = sorted({Path(__file__), protocol_path,
                      Path('scripts/collect_semantic_gain_v13_continuations.py'),
                      Path('scripts/collect_semantic_gain_v13_history.py'),
                      Path('tests/virtual3d/test_semantic_opportunities_v14.py'),
                      *[p for base in ('nso', 'env', 'utils') for p in Path(base).rglob('*.py')]})
    frozen = {str(p.relative_to(ROOT) if p.is_absolute() else p): sha(p) for p in sources}
    with zipfile.ZipFile(output / 'sources.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for name in frozen:
            archive.write(name, name)
    manifest = dict(status='running', created_utc=datetime.now(timezone.utc).isoformat(),
                    schema='semantic_opportunities_v14_observed_input_audit/1',
                    source_sha256=frozen, inputs={}, feature_dimension=26,
                    common_feature_names=COMMON_NAMES, instance_kernel_names=KERNEL_NAMES,
                    future_rewards_read=False, model_trained=False,
                    geometry_control='18 common + 8 geometry second moments',
                    semantic_control='18 common + 8 confidence-weighted class/geometry moments')
    write(output / 'manifest.json', manifest)
    rows = []
    try:
        for pair in protocol['eligible_pairs']:
            parent, step = pair['parent'], pair['action_id']
            for arrangement in protocol['arrangements']:
                history = Path(protocol['histories']) / f'{parent}_{arrangement}'
                hp, cp = history / 'manifest.json', history / 'checkpoints.json'
                old = json.loads(hp.read_text())
                for name, expected in old['source_sha256'].items():
                    if sha(Path(name)) != expected:
                        raise ValueError(f'frozen history source changed: {name}')
                checkpoint = next(c for c in json.loads(cp.read_text()) if c['action_id'] == step)
                paths = [history / f'packets/{i:04d}.npz' for i in range(step + 1)]
                manifest['inputs'].update({str(p): sha(p) for p in [hp, cp, *paths]})
                stored = [load_packet(p) for p in paths]
                _, _, runtime = restore(old['protocol'], stored, checkpoint)
                mapper = runtime.states[0]['mapper']
                candidates = checkpoint['candidates']
                if digest(candidates) != checkpoint['candidate_sha256']:
                    raise ValueError('checkpoint candidate seal mismatch')
                before = decision_state(runtime)['sha256']
                started = perf_counter()
                assets, desc = observed_descriptors(runtime, candidates)
                features = route_instance_features(mapper, candidates, assets, desc)
                elapsed = perf_counter() - started
                swapped = deepcopy(desc)
                for d in swapped:
                    for a in d['observed_assets']:
                        a['class_vote'] *= -1
                swapped_features = route_instance_features(mapper, candidates, assets, swapped)
                zero = route_instance_features(mapper, candidates, assets, desc, confidence_scale=0.)
                permutation = list(reversed(range(len(assets))))
                perm_desc = deepcopy(desc)
                for d in perm_desc:
                    d['observed_assets'] = [dict(d['observed_assets'][i], asset_index=j)
                                            for j, i in enumerate(permutation)]
                    for name in ('aperture_factors', 'prefix_aperture_support'):
                        d['observed_aperture_audit'][name] = [d['observed_aperture_audit'][name][i]
                                                            for i in permutation]
                perm_features = route_instance_features(mapper, candidates, [assets[i] for i in permutation], perm_desc)
                for f, x, z, p in zip(features, swapped_features, zero, perm_features):
                    np.testing.assert_array_equal(f['geometry'], x['geometry'])
                    np.testing.assert_array_equal(f['geometry'], z['geometry'])
                    np.testing.assert_array_equal(f['geometry'], p['geometry'])
                    np.testing.assert_array_equal(f['semantic'], p['semantic'])
                    np.testing.assert_array_equal(z['semantic'][18:], np.zeros(8))
                    np.testing.assert_allclose(f['semantic'][18:], -np.asarray(x['semantic'][18:]), atol=0, rtol=0)
                    assert f['second_view_witnesses'] == x['second_view_witnesses']
                assert before == decision_state(runtime)['sha256'], 'feature extraction mutated runtime'
                folder = output / f'{parent}_{arrangement}_step_{step:03d}'
                folder.mkdir()
                # This is a feature-only replay bundle; no simulator truth or rewards.
                fields = ('observed_low', 'observed_high', 'aabb_center', 'front_axis', 'back_axis',
                          'side_axis', 'rear_boundary_xy', 'measured_width_m')
                write(folder / 'inputs.json', dict(config=vars(mapper.config),
                    candidates=candidates, descriptors=desc,
                    observed_assets=[{k: a[k] for k in fields} for a in assets]))
                np.savez_compressed(folder / 'observed_map.npz', belief=mapper.belief)
                write(folder / 'features.json', features)
                row = dict(parent=parent, arrangement=arrangement, action_id=step, candidates=len(features),
                    observed_instances=len(assets), state_sha256=before, candidate_sha256=checkpoint['candidate_sha256'],
                    original_outbound_apertures_all_zero=all(not any(d['observed_aperture_audit']['aperture_factors']) for d in desc),
                    class_sensitive_candidates=sum(any(v != 0 for v in f['semantic'][18:]) for f in features),
                    distinct_semantic_tails=len({tuple(f['semantic'][18:]) for f in features}),
                    candidates_with_second_view_witness=sum(any(w is not None for w in f['second_view_witnesses']) for f in features),
                    extraction_wall_time_s=elapsed, geometry_label_invariance=True,
                    instance_permutation_invariance=True, zero_confidence_tail_zero=True, runtime_unchanged=True)
                rows.append(row)
                write(output / 'partial.json', rows)
                print(json.dumps(row), flush=True)
        for name, expected in frozen.items():
            if sha(Path(name)) != expected:
                raise ValueError(f'source changed: {name}')
        for name, expected in manifest['inputs'].items():
            if sha(Path(name)) != expected:
                raise ValueError(f'input changed: {name}')
        summary = dict(status='passed', states=len(rows), candidate_descriptors=sum(r['candidates'] for r in rows),
                       rows=rows, model_trained=False, information_gate_retested=False,
                       semantic_policy_efficacy_proven=False,
                       boundary='Post-hoc representation audit on existing development histories only.')
        write(output / 'summary.json', summary)
        manifest['status'] = 'complete'
    except Exception as error:
        manifest.update(status='failed', error=repr(error))
        raise
    finally:
        write(output / 'manifest.json', manifest)
        write(output / 'artifact_hashes.json', {str(p.relative_to(output)): sha(p)
              for p in sorted(output.rglob('*')) if p.is_file() and p.name != 'artifact_hashes.json'})


if __name__ == '__main__':
    main()
