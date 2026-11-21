"""Seal reference, classifier, source geometry and geometry-only preflight."""

import argparse
import dataclasses
import json
from pathlib import Path
import time

from mechanism_world import ANALYSES, BUDGET, HERE, LAYOUTS, PARENT, SEED, SPECS, api
from prepare import digest, write


def freeze():
    target = HERE/'freeze.json'
    if target.exists():
        raise FileExistsError(target)
    paired = json.loads((HERE/'swap_preflight.json').read_text())
    assert paired['status'] == 'both_frozen_parent_pairs_validated'
    assert not paired['actor_run'] and not paired['scores_opened']
    copied_paths = [HERE / n for n in ('mechanism_world.py','prepare.py','prepare_reference.py',
            'freeze_scene.py','reference_protocol.json','SCENE_MODEL.md','swap_preflight.json')]
    draw_receipts = {}
    for name in LAYOUTS:
        geom = HERE/'prepared'/name
        ref = HERE/'evaluator_references'/name
        prep = json.loads((geom/'receipt.json').read_text())
        surface = json.loads((ref/'RECEIPT.json').read_text())
        assert prep['status'] == 'geometry_and_sensor_preflight_passed'
        assert not prep['actor_run'] and not prep['scored_candidates_opened']
        assert surface['status'] == 'fixed_source_only_surface_reference_prepared' and not surface['actor_trajectory_read']
        assert digest(ref/'fixed_reference.npz') == surface['reference_npz_sha256']
        assert digest(ref/'source_mesh.ply') == surface['source_mesh_sha256']
        copied_paths.extend(sorted(p for p in geom.iterdir() if p.is_file()))
        copied_paths.extend(sorted(p for p in ref.iterdir() if p.is_file()))
        cache_key = prep['geometry']['topology_cache_key']
        for suffix in ('.npz','.json'):
            copied_paths.append(HERE/'precomputed_assets'/(cache_key+suffix))
        draw_receipts[name] = dict(geometry_receipt=str(geom/'receipt.json'),
            evaluator_reference_receipt=str(ref/'RECEIPT.json'), initial_state=str(geom/'initial_state.json'),
            expected_side=SPECS[name]['route_evaluation']['expected_diagnostic_side'])
    copied_paths.extend([PARENT, api.OLD, api.old.PREVIOUS/'scene_adapter/replicacad_world.py',
        api.old.PREVIOUS/'assets/textured_raycast.py', api.old.ASSETS/'assemble_apt0.py',
        api.old.ASSETS/'download_receipt.json',
        ANALYSES/'dual_template_exploration_20261119/audit/surface_metrics.py',
        ANALYSES/'semantic_scene_redesign_20261119/exploration_evaluation/surface_reference.py',
        HERE.parents[1]/'PROTOCOL.json'])
    hashes={str(p.resolve()):digest(p) for p in copied_paths}
    result=dict(status='frozen_before_controlled_policy_outcomes',created_unix_s=time.time(),
        seed=SEED,budget=BUDGET,layout_order=list(LAYOUTS),layout_specs=SPECS,assets=draw_receipts,
        files_sha256=hashes,actor_results_read=False,preflight_candidate_score_calls=0,
        mocap='Exact simulated poses and ideal first-hit class and owner support.',
        permitted_actor_observations=['RGB-D','current first-hit class/visible instance IDs','planar scan','pose'],
        evaluator_only_keys=['layout_specs.route_evaluation','source meshes','evaluator truth grids','fixed source reference'])
    write(target,result)
    verify()
    return result


def verify():
    freeze_path=HERE/'freeze.json'
    row=json.loads(freeze_path.read_text())
    for name, h in row['files_sha256'].items():
        assert digest(name) == h, name
    return row


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--verify',action='store_true')
    row=verify() if p.parse_args().verify else freeze()
    print(json.dumps(dict(status=row['status'],files=len(row['files_sha256']),sha256=digest(HERE/'freeze.json'))))
