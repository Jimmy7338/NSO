#!/usr/bin/env python3
"""Seal actor-facing cold starts separately from the evaluator's fixed reference."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from scene import BUDGET, DEVELOPMENT, HERE, HOLDOUT, SEED, SOURCE, SPECS


ACTOR_FILES = {'initial_frame.npz', 'initial_scan.npz', 'initial_state.json'}
EVALUATOR_FILES = {'RECEIPT.json', 'cold_start_preflight.json', 'evaluator_truth.npz',
                   'fixed_reference.npz', 'source_mesh.ply'}


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def freeze(paths):
    return {str(path.resolve()): sha(path) for path in sorted(set(paths))}


def write_new(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def room_areas(reachable, spec, resolution):
    cols = (np.arange(reachable.shape[1]) + .5) * resolution
    rows = (reachable.shape[0] - np.arange(reachable.shape[0]) - .5) * resolution
    east = cols[None, :] >= spec['split_x']
    north = rows[:, None] >= spec['split_y']
    groups = {'southwest': ~east & ~north, 'southeast': east & ~north,
              'northwest': ~east & north, 'northeast': east & north}
    counts = {name: int((reachable & mask).sum()) for name, mask in groups.items()}
    if not all(counts.values()) or sum(counts.values()) != reachable.sum():
        raise ValueError('At least one original source room is unreachable')
    return counts


def validate(layout, split, actor_dir, reference_dir):
    if {p.name for p in actor_dir.iterdir()} != ACTOR_FILES:
        raise ValueError(f'Unexpected actor-side input files for {layout}')
    if {p.name for p in reference_dir.iterdir()} != EVALUATOR_FILES:
        raise ValueError(f'Unexpected evaluator-side input files for {layout}')
    initial = read(actor_dir / 'initial_state.json')
    preflight = read(reference_dir / 'cold_start_preflight.json')
    receipt = read(reference_dir / 'RECEIPT.json')
    if (initial['paid_actions'] or initial['policy_select_calls'] or preflight['actor_select_calls']
            or receipt['actor_executed'] or receipt['actor_trajectory_read']
            or initial['stop_budget'] != BUDGET or initial['seed'] != SEED
            or initial['actual_config']['max_steps'] != BUDGET
            or initial['grid_resolution_m'] != .2 or initial['actual_config']['robot_radius_m'] != .2
            or preflight['initial_unknown_reachable_cells'] <= 0):
        raise ValueError(f'Cold-start/reference contract failed for {layout}')
    for filename, expected in (('fixed_reference.npz', receipt['reference_npz_sha256']),
                               ('source_mesh.ply', receipt['source_mesh_sha256'])):
        if sha(reference_dir / filename) != expected:
            raise ValueError(f'Hash differs from reference receipt for {layout}/{filename}')
    dependencies = [Path(dep) for dep in receipt['source_inputs_sha256']]
    for dep in dependencies:
        if sha(dep) != receipt['source_inputs_sha256'][str(dep)]:
            raise ValueError(f'Geometry dependency hash changed: {dep}')
    spec = SPECS[layout]
    with np.load(reference_dir / 'evaluator_truth.npz', allow_pickle=False) as archive:
        reachable = archive['reachable']
    if list(reachable.shape) != initial['shape'] or reachable.sum() != receipt['reachable_cells']:
        raise ValueError(f'Reachable denominator changed for {layout}')
    details = {key: initial[key] for key in ('layout', 'shape', 'anchor', 'seed', 'stop_budget', 'scene_role',
                                             'initial_packet_sha256', 'initial_owner_sha256', 'actual_config')}
    details.update(split=split, topology=spec['topology'], door_edges=spec['room_graph']['edges'],
                   bounds_xy_m=[spec['width_m'], spec['height_m']],
                   threshold_surface_m=.05, reachable_cells=receipt['reachable_cells'],
                   reference_scene_points=receipt['metadata']['all']['visible_samples'],
                   reference_scene_visible_area_m2=receipt['metadata']['all']['estimated_visible_area_m2'],
                   reachable_cells_by_room=room_areas(reachable, spec, initial['grid_resolution_m']),
                   initially_unknown_reachable_cells=preflight['initial_unknown_reachable_cells'])
    return details, dependencies


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('split', choices=('development', 'holdout'))
    args = parser.parse_args()
    layouts = DEVELOPMENT if args.split == 'development' else tuple(HOLDOUT)
    if (HERE.parent / 'runner/runs' / args.split).exists():
        raise ValueError('A split must be sealed before its first actor run')
    actor = [HERE / 'scene.py', SOURCE]
    evaluator = [HERE / cmd for cmd in ('scene.py', 'prepare_scenes.py', 'seal_scenes.py', 'reference_protocol.json')]
    details = []
    for layout in layouts:
        cold = HERE / 'prepared' / layout
        ref = HERE / 'evaluator_references' / layout
        description, dependencies = validate(layout, args.split, cold, ref)
        actor.extend(cold / filename for filename in ACTOR_FILES)
        evaluator.extend(ref / filename for filename in EVALUATOR_FILES)
        evaluator.extend(dependencies)
        details.append(description)
    names = {'split': args.split, 'layouts': details, 'actors_run': 0, 'seed': SEED, 'budget': BUDGET}
    write_new(HERE / f'actor_freeze_{args.split}.json', names | {'status': 'actor_inputs_only',
                                                                'files_sha256': freeze(actor)})
    write_new(HERE / f'evaluator_freeze_{args.split}.json', names | {'status': 'reference_hidden_from_actor',
                                                                    'files_sha256': freeze(evaluator)})
    print(f'{args.split}: {len(details)} actor starts and independently hashed references sealed', flush=True)


if __name__ == '__main__':
    main()
