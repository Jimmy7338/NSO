#!/usr/bin/env python3
"""Seal first observations and evaluator-only source references separately."""

import argparse
import json
from pathlib import Path

from geometry_audit import read, sha
from scene import ANALYSES, ARCHIVED_SCENES, BUDGET, DEVELOPMENT, HERE, HOLDOUT, SOURCE, SPECS


ACTOR_FILES = ('initial_frame.npz','initial_scan.npz','initial_state.json')
EVALUATION_FILES = ('RECEIPT.json','cold_start_preflight.json','evaluator_truth.npz','fixed_reference.npz','source_mesh.ply')
SIMULATOR = ANALYSES/'semantic_scene_redesign_20261119'


def fingerprint(paths):
    return {str(path):sha(path) for path in sorted(set(paths))}


def write_new(path, report):
    with path.open('x') as stream:
        json.dump(report,stream,indent=2,allow_nan=False)
        stream.write('\n')


def verify(ledger):
    for path, digest in ledger['files_sha256'].items():
        if sha(Path(path)) != digest:
            raise ValueError(f'Previously sealed input changed: {path}')


def seal(split):
    if (HERE.parent/'runner/runs'/split).exists():
        raise ValueError('Seal scene inputs before this split’s first actor run')
    evaluator_paths = [HERE/'geometry_audit.py', HERE/'prepare_scenes.py', HERE/'seal_scenes.py', HERE/'reference_protocol.json']
    if split == 'holdout':
        audit = HERE/'formal_geometry_audit.json'
        if read(audit)['policy_calls'] != 0:
            raise ValueError('Unexpected policy execution before the scene freeze')
        evaluator_paths.append(audit)
    actor_paths = [HERE/'scene.py', SOURCE, ARCHIVED_SCENES/'scene.py',
                   ANALYSES/'dual_template_exploration_20261119/scene/controlled_world.py',
                   SIMULATOR/'scene_adapter/replicacad_world.py', SIMULATOR/'assets/textured_raycast.py',
                   SIMULATOR/'assets/replicacad/assemble_apt0.py', SIMULATOR/'assets/replicacad/download_receipt.json']
    scenes = []
    for layout in DEVELOPMENT if split == 'development' else tuple(HOLDOUT):
        prepared = HERE/'prepared'/layout
        reference = HERE/'evaluator_references'/layout
        initial = read(prepared/'initial_state.json')
        evaluation = read(reference/'RECEIPT.json')
        if initial['paid_actions'] != 0 or initial['policy_select_calls'] != 0 or initial['stop_budget'] != BUDGET:
            raise ValueError('The experiment must start from a zero-action, zero-policy cold frame')
        if evaluation['actor_trajectory_read'] or evaluation['actor_executed']:
            raise ValueError('The source-only reference cannot depend on actor trajectories')
        if sha(reference/'fixed_reference.npz') != evaluation['reference_npz_sha256']:
            raise ValueError(f'Reference sample bytes changed during preparation: {layout}')
        if sha(reference/'source_mesh.ply') != evaluation['source_mesh_sha256']:
            raise ValueError(f'Reference mesh bytes changed during preparation: {layout}')
        for path, digest in evaluation['source_inputs_sha256'].items():
            if sha(Path(path)) != digest:
                raise ValueError(f'Reference source bytes changed: {path}')
            evaluator_paths.append(Path(path))
        actor_paths.extend(prepared/name for name in ACTOR_FILES)
        evaluator_paths.extend(reference/name for name in EVALUATION_FILES)
        desc = {name:initial[name] for name in ('layout','seed','anchor','shape','stop_budget','initial_packet_sha256',
                                                'initial_owner_sha256','actual_config')}
        desc['scene_role'] = 'known_development' if split == 'development' else 'fresh_holdout'
        if split == 'development':
            desc['role_scope_note'] = 'Known development in this study. First-observation and evaluation bytes linked unchanged from weighted archive.'
        elif desc['seed'] != SPECS[layout]['seed']:
            raise ValueError(f'Frozen seed differs from declared scene seed: {layout}')
        scenes.append(desc)
    common = {'split':split,'layouts':scenes,'actors_run':0,'budget':BUDGET,
              'seed_policy':'Each scene declares its fixed seed. All policy arms share each scene seed and first packet.'}
    actor = common | {'status':'actor_inputs_only','files_sha256':fingerprint(actor_paths)}
    if any('evaluator' in Path(path).name for path in actor['files_sha256']):
        raise ValueError('An evaluator reference was placed in the actor ledger')
    write_new(HERE/f'actor_freeze_{split}.json',actor)
    write_new(HERE/f'evaluator_freeze_{split}.json',common | {'status':'reference_hidden_from_actor',
                                                            'files_sha256':fingerprint(evaluator_paths)})
    print(f'{split}: {len(scenes)} zero-action starts and independently hashed references',flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('split',choices=('development','holdout'))
    parser.add_argument('--verify',action='store_true')
    args=parser.parse_args()
    if args.verify:
        for scope in ('actor','evaluator'):
            verify(read(HERE/f'{scope}_freeze_{args.split}.json'))
        print(f'{args.split}: sealed actor and evaluator inputs verified')
    else:
        seal(args.split)
