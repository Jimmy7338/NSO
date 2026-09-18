#!/usr/bin/env python3
"""Rebuild the existing mapper from sealed packets and audit a 3D observer."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import argparse
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from env.virtual3d_inspection_v4 import InspectionConfigV4
from nso.decision_replay_v13 import array_hash, load_packet
from nso.observed_runtime_mapper_v10 import ObservedRuntimeMapperV10
from nso.surface_feedback_v16 import SurfaceFeedbackV16
from scripts.audit_semantic_v15_feedback_observability import sha, write, require


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    h = json.loads((args.source / 'manifest.json').read_text())
    require(h['status'] == 'complete', 'history is not complete')
    inventory = json.loads((args.source / 'artifact_hashes.json').read_text())
    for name, expected in inventory.items():
        require(sha(args.source / name) == expected, 'history artifact changed')
    names = set(h['source_sha256']) | {'nso/surface_feedback_v16.py',
        'tests/virtual3d/test_surface_feedback_v16.py',
        'scripts/audit_semantic_v15_feedback_observability.py',
        str(Path(__file__).resolve().relative_to(ROOT))}
    for name, expected in h['source_sha256'].items():
        require(sha(ROOT / name) == expected, 'history mapper dependencies changed')
    args.output.mkdir(parents=True, exist_ok=False)
    sources = {n: sha(ROOT / n) for n in sorted(names)}
    with zipfile.ZipFile(args.output / 'sources.zip', 'x', zipfile.ZIP_DEFLATED) as z:
        for name in sorted(names): z.write(ROOT / name, name)
    manifest = dict(status='running', source_sha256=sources, source_history=str(args.source),
        input_inventory_sha256=sha(args.source / 'artifact_hashes.json'),
        new_physical_experiments=0, policy_modified=False, training_allowed=False,
        observer_role='auxiliary observed 3D transitions only; not instance-conditioned prediction')
    write(args.output / 'manifest.json', manifest)
    try:
        tests = subprocess.run([sys.executable, '-m', 'unittest',
            'tests.virtual3d.test_surface_feedback_v16', '-v'], text=True, capture_output=True)
        (args.output / 'tests.txt').write_text(tests.stdout + tests.stderr)
        require(tests.returncode == 0, 'surface feedback tests failed')
        config_doc = json.loads((ROOT / h['protocol']['scene_protocol']).read_text())
        context = next(x for x in config_doc['contexts'] if x['id'] == h['protocol']['context'])
        config = InspectionConfigV4(**{**config_doc['shared_conditions'],
            **{k: v for k, v in context.items() if k not in ('id', 'seed')}})
        summaries = []
        for seed in h['protocol']['structure_seeds']:
            started = perf_counter()
            folder = args.source / f'structure_{seed}'
            decisions = json.loads((folder / 'all_decisions.json').read_text())
            calls = json.loads((folder / 'module_calls.json').read_text())
            by_action = {r['outputs']['action_id']: r['outputs'] for r in calls if r['method'] == 'compute_reward'}
            checkpoints = {}
            for d in decisions: checkpoints.setdefault(d['action_id'], []).append(d['state_before']['evidence'])
            bounds = decisions[0]['bounds']
            mapper = ObservedRuntimeMapperV10((bounds[1], bounds[3]), config)
            events, observer, checks = [], None, 0
            paths = sorted((folder / 'packets').glob('*.npz'))
            require(len(paths) == len(by_action) + 1, 'packet count mismatch')
            for aid, path in enumerate(paths):
                packet = load_packet(path)
                require(packet.action_id == aid, 'packet action sequence differs')
                mapper.update(packet.frame, packet.scan)
                if observer is None:
                    observer = SurfaceFeedbackV16(mapper, action_id=0, coordinate_epoch='v15_ideal_fixed_world')
                else:
                    event = observer.observe(mapper, action_id=aid, coordinate_epoch='v15_ideal_fixed_world')
                    old = by_action[aid]
                    require(event['first_observed_support_count'] == old['new_measured_support_count'],
                            'new support differs from separately recorded IGCR ledger')
                    event['recorded_2d_camera_predicted_cells'] = old['observed_gain_calibration']['camera']['predicted_cells']
                    events.append(event)
                for checkpoint in checkpoints.get(aid, []):
                    arrays = {name: array_hash(value) for name, value in vars(mapper).items()
                              if isinstance(value, np.ndarray)}
                    require(arrays == checkpoint['map_arrays'], 'observed map replay differs')
                    mesh = mapper.mesh()
                    hashes = {name: array_hash(np.asarray(getattr(mesh, name))) for name in
                              ('vertices', 'triangles', 'vertex_colors')}
                    require(hashes == checkpoint['mesh'], 'TSDF reconstruction replay differs')
                    checks += 1
                if aid and aid % 64 == 0:
                    print(f'structure={seed} sensor_replay={aid}', flush=True)
            zero = [e for e in events if e['recorded_2d_camera_predicted_cells'] == 0]
            changed = [e for e in zero if e['first_observed_support_count'] or
                       e['new_directions_on_previously_seen_support'] or
                       e['signed_prior_quality_sum_change'] != 0]
            summary = dict(structure_seed=seed, observed_actions=len(events), map_mesh_checkpoints=checks,
                zero_camera_trials=len(zero), zero_camera_trial_3d_transition_changes=len(changed),
                new_support_at_zero_camera_trials=sum(e['first_observed_support_count'] for e in zero),
                new_directions_at_zero_camera_trials=sum(e['new_directions_on_previously_seen_support'] for e in zero),
                positive_quality_change_actions_at_zero_trials=sum(e['signed_prior_quality_sum_change'] > 0 for e in zero),
                negative_quality_change_actions_at_zero_trials=sum(e['signed_prior_quality_sum_change'] < 0 for e in zero),
                final_observer=observer.snapshot(), elapsed_s=perf_counter() - started)
            write(args.output / f'structure_{seed}_transitions.json', events)
            summaries.append(summary)
            write(args.output / 'partial.json', summaries)
            print(json.dumps(summary), flush=True)
        for name, expected in sources.items():
            require(sha(ROOT / name) == expected, 'source changed during replay')
        write(args.output / 'result.json', dict(status='passed', summaries=summaries,
            tests_passed=7, parent_layouts=1, policy_modified=False,
            new_physical_experiments=0, semantic_efficacy_proven=False,
            instance_conditioned_feedback_validated=False,
            boundary='Existing sensor/map replay validates an auxiliary 3D observation channel. '
                     'Quality changes are signed proxies, not evaluated precision or F1 gains.'))
        manifest['status'] = 'complete'
    except Exception as error:
        manifest.update(status='failed', error=repr(error))
        raise
    finally:
        write(args.output / 'manifest.json', manifest)
        write(args.output / 'artifact_hashes.json', {p.name: sha(p) for p in sorted(args.output.iterdir())
            if p.is_file() and p.name != 'artifact_hashes.json'})


if __name__ == '__main__': main()
