#!/usr/bin/env python3
"""Hash visible cold inputs and offline-only geometry references in separate ledgers."""

import argparse
import json
from pathlib import Path
import time

from geometry_audit import read, sha
from scene import ANALYSES, BUDGET, HERE, PRIOR_SCENES, SOURCE, SPECS, geometry


SIMULATOR = ANALYSES/'semantic_scene_redesign_20261119'
INPUT_FILES = ('initial_state.json','initial_frame.npz','initial_scan.npz')
REFERENCE_FILES = (
    'RECEIPT.json', 'cold_start_preflight.json', 'evaluator_truth.npz',
    'fixed_reference.npz', 'source_mesh.ply',
)


def fingerprint(paths):
    return {str(path): sha(path) for path in sorted(set(paths))}


def verify(ledger):
    for path, expected in ledger['files_sha256'].items():
        if sha(Path(path)) != expected:
            raise ValueError(f'Frozen scene file changed: {path}')


def seal():
    launch = HERE.parent/'execution'
    if launch.is_dir() and any(path.name in SPECS for path in launch.rglob('*') if path.is_dir()):
        raise ValueError('New-layout actor directories exist before the scene freeze')
    audit = HERE/'formal_geometry_audit.json'
    if read(audit)['policy_calls'] != 0:
        raise ValueError('Scene preparation may not call any planning policy')

    actor = [
        HERE/'scene.py', PRIOR_SCENES/'scene.py', SOURCE, geometry.ARCHIVED_SCENES/'scene.py',
        ANALYSES/'dual_template_exploration_20261119/scene/controlled_world.py',
        SIMULATOR/'scene_adapter/replicacad_world.py', SIMULATOR/'assets/textured_raycast.py',
        SIMULATOR/'assets/replicacad/assemble_apt0.py', SIMULATOR/'assets/replicacad/download_receipt.json',
    ]
    source_only = [
        HERE/'geometry_audit.py', PRIOR_SCENES/'geometry_audit.py', HERE/'prepare_scenes.py',
        HERE/'seal_scenes.py', HERE/'reference_protocol.json', audit,
    ]
    descriptions = []
    for name, spec in SPECS.items():
        prepared = HERE/'prepared'/name
        references = HERE/'evaluator_references'/name
        initial = read(prepared/'initial_state.json')
        receipt = read(references/'RECEIPT.json')
        if (initial['seed'] != spec['seed'] or initial['stop_budget'] != BUDGET or
                initial['paid_actions'] or initial['policy_select_calls']):
            raise ValueError(f'Non-cold or non-matching initial state: {name}')
        if receipt['actor_executed'] or receipt['actor_trajectory_read']:
            raise ValueError(f'Reference contains actor executions: {name}')
        for file, field in (
            ('fixed_reference.npz','reference_npz_sha256'),
            ('source_mesh.ply','source_mesh_sha256'),
        ):
            if sha(references/file) != receipt[field]:
                raise ValueError(f'Changed independent reference file: {name}/{file}')
        for path, expected in receipt['source_inputs_sha256'].items():
            if sha(Path(path)) != expected:
                raise ValueError(f'Changed independent reference source: {path}')
            source_only.append(Path(path))
        actor.extend(prepared/file for file in INPUT_FILES)
        source_only.extend(references/file for file in REFERENCE_FILES)
        descriptions.append({
            field: initial[field] for field in (
                'layout','seed','anchor','shape','stop_budget','initial_packet_sha256',
                'initial_owner_sha256','actual_config',
            )
        } | {'scene_role':'fresh_holdout'})

    native_simulator_cache = fingerprint((HERE/'precomputed_assets').glob('*'))
    if len(native_simulator_cache) != 2*len(SPECS):
        raise ValueError('Unexpected source-world collision cache cardinality')
    common = {
        'split':'holdout', 'layouts':descriptions, 'actors_run':0, 'budget':BUDGET,
        'frozen_unix_s':time.time(),
        'seed_policy':'All strategies in each layout reuse the same cold sensor packet and fixed world seed.',
    }
    actor_hashes = fingerprint(actor)
    if any('evaluator_references' in path or 'precomputed_assets' in path for path in actor_hashes):
        raise ValueError('Hidden geometry was placed in the visible actor-input ledger')
    ledgers = {
        'actor': common | {
            'status':'actor_inputs_only',
            'files_sha256':actor_hashes,
            'simulator_only_collision_cache_sha256':native_simulator_cache,
            'runtime_interface':'Policy consumes native SensorPacket, acquired maps and action feedback.',
        },
        'evaluator': common | {
            'status':'reference_hidden_from_actor',
            'files_sha256':fingerprint(source_only),
        },
    }
    for role, receipt in ledgers.items():
        with (HERE/f'{role}_freeze_holdout.json').open('x') as stream:
            json.dump(receipt,stream,indent=2,allow_nan=False)
            stream.write('\n')
    print('Frozen two new layouts with independent visible-input and offline-reference ledgers', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', action='store_true')
    if parser.parse_args().verify:
        for role in ('actor','evaluator'):
            ledger = read(HERE/f'{role}_freeze_holdout.json')
            verify(ledger)
            if 'simulator_only_collision_cache_sha256' in ledger:
                verify({'files_sha256':ledger['simulator_only_collision_cache_sha256']})
        print('Verified actor, simulator-internal cache, and evaluator source hashes', flush=True)
    else:
        seal()
