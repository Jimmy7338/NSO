#!/usr/bin/env python3
"""Seal all four measured prefixes, shared choices, and structural gates.

No candidate action is executed. Evaluator truth is used only after all choices
have been sealed, for fixed background and budget/area construction diagnostics.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import sys
import time
import zipfile
import numpy as np
from scipy.sparse.csgraph import dijkstra
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from env.virtual3d_competition_v8 import (create_competition_world, collect_competition_prefix,
                                        COMPETITION_ARRANGEMENTS)
from env.virtual3d import camera_pose
from nso.competition_candidates_v8 import candidate_routes
from nso.competition_prior_v8 import score_routes
from nso.route_coverage_v2 import orientation_graph
from scripts.prepare_response_v7 import records_from_prefix, array_hash
from scripts.eval_counterfactual_views import remap, write_json, coverage
from utils.grid_geometry import inflated_obstacles
from utils.cpu_protocol import file_hash
from utils.rgbd_contract import RGBDFrame, PlanarScan
from utils.reconstruction_metrics import ReconstructionEvaluator
from utils.counterfactual_surface_visibility import reference_visible


def read(path):
    return json.loads(Path(path).read_text())


def visit_costs(safe, start_pose, config, region_centers):
    """Exact directed visit-both-return optimum via two possible visit orders.

    All headings in each fixed evaluator-only region are allowed. Shortest
    distances between visits are equivalent to the two-visited-bit graph.
    """
    if not safe[start_pose[:2]]:
        return {'single': [None, None], 'both': None, 'region_state_counts': [0, 0]}
    graph, cells, ids = orientation_graph(safe)
    start = int(ids[start_pose[:2]]) * 4 + start_pose[2]
    state_cells = np.repeat(cells, 4, axis=0)
    xs = (state_cells[:, 1] + .5) * config.resolution_m
    ys = (safe.shape[0] - state_cells[:, 0] - .5) * config.resolution_m
    zones = [np.flatnonzero((np.abs(xs - x) <= .3 + 1e-12) & (ys >= 5.9 - 1e-12) & (ys <= 6.5 + 1e-12))
             for x in region_centers]
    forward = dijkstra(graph, directed=True, indices=start)
    back = dijkstra(graph.T.tocsr(), directed=True, indices=start)
    single = [float(np.min(forward[z] + back[z])) if len(z) else np.inf for z in zones]
    both = np.inf
    for first, second in (zones, zones[::-1]):
        if len(first) and len(second):
            between = dijkstra(graph, directed=True, indices=first)[:, second]
            both = min(both, float(np.min(forward[first, None] + between + back[None, second])))
    return {'single': [int(v) if np.isfinite(v) else None for v in single],
            'both': int(both) if np.isfinite(both) else None,
            'region_state_counts': [len(z) for z in zones]}


def check_pair(first, second):
    for fa, fb, sa, sb in zip(first['frames'], second['frames'], first['scans'], second['scans']):
        for key in ('depth_m', 'intrinsic', 'world_from_camera', 'timestamp_s'):
            np.testing.assert_array_equal(getattr(fa, key), getattr(fb, key))
        for key in sa.__dataclass_fields__:
            np.testing.assert_array_equal(getattr(sa, key), getattr(sb, key))
        np.testing.assert_array_equal(fa.semantic > 0, fb.semantic > 0)
        nonmarker = (fa.semantic == 0) & (fb.semantic == 0)
        np.testing.assert_array_equal(fa.color_rgb[nonmarker], fb.color_rgb[nonmarker])
    if first['records'] != second['records'] or first['routes'] != second['routes']:
        raise ValueError('paired candidate or paid prefix differs')
    if first['geometry_hash'] != second['geometry_hash']:
        raise ValueError('paired nonlabel map differs')
    for name in ('N', 'G', 'O', 'M'):
        np.testing.assert_array_equal(first['predictions']['scores'][name], second['predictions']['scores'][name])
    np.testing.assert_array_equal(first['predictions']['scores']['S'], second['predictions']['scores']['X'])
    return {'raw_nonlabel_prefix_exact': True, 'nonlabel_map_exact': True,
            'shared_candidates_exact': True, 'nonsemantic_scores_exact': True,
            'semantic_pair_swap_exact': True}


def prepare(output):
    if output.exists():
        raise FileExistsError(output)
    protocol_path = ROOT / 'configs/virtual3d/competition_v8_protocol.json'
    protocol = read(protocol_path)
    if shutil.disk_usage(ROOT).free < protocol['storage']['reserve_bytes']:
        raise RuntimeError('insufficient free space above reserve')
    output.mkdir(parents=True)
    names = sorted({str(p.relative_to(ROOT)) for folder in ('env', 'nso', 'utils')
                    for p in (ROOT / folder).glob('*.py')} | {
        'scripts/prepare_competition_v8.py', 'scripts/prepare_response_v7.py',
        'scripts/eval_counterfactual_views.py', 'configs/virtual3d/competition_v8_contexts.json',
        'configs/virtual3d/competition_v8_protocol.json', 'requirements-3d.lock.txt'})
    hashes = {name: file_hash(ROOT / name) for name in names}
    with zipfile.ZipFile(output / 'sources.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for name in names:
            archive.write(ROOT / name, name)
    metadata = {'status': 'running_prefix_preparation', 'source_sha256': hashes,
                'new_candidate_branches': 0, 'role': 'development_positive_control',
                'prediction_inputs_include_truth': False}
    write_json(output / 'metadata.json', metadata)
    started = time.time(); summaries = []; pair_results = {}
    try:
        for context_id in protocol['parent_order']:
            pair = []
            for arrangement in COMPETITION_ARRANGEMENTS:
                if shutil.disk_usage(output).free < protocol['storage']['reserve_bytes']:
                    raise RuntimeError('disk reserve reached')
                world = create_competition_world(context_id, arrangement)
                raw = collect_competition_prefix(world)
                frames = [r['frame'] for r in raw]; scans = [r['scan'] for r in raw]
                records = records_from_prefix(raw, world.shape, world.config)
                folder = output / context_id / arrangement; folder.mkdir(parents=True)
                prefix = folder / 'prefix'; (prefix / 'frames').mkdir(parents=True); (prefix / 'scans').mkdir()
                for i, (frame, scan) in enumerate(zip(frames, scans)):
                    frame.save(prefix / 'frames' / f'{i:04d}.npz'); scan.save(prefix / 'scans' / f'{i:04d}.npz')
                write_json(prefix / 'records.json', records)
                write_json(folder / 'fixture.json', {'context': asdict(world.context), 'arrangement': arrangement,
                                                    'config': asdict(world.config)})
                mapper = remap(frames, scans, records, world.config, world.shape)
                obs = mapper.observation(tuple(records[-1]['position']), records[-1]['heading'], 150, False)
                routes, audit = candidate_routes(mapper, obs)
                prefix_poses = [frame.world_from_camera for frame in frames]
                predictions = score_routes(mapper, routes, prefix_poses) if routes else {'scores': {m: [] for m in ('G','O','N','S','X','M')}, 'selected_candidate_ids': {}}
                write_json(folder / 'candidates.json', routes); write_json(folder / 'candidate_audit.json', audit)
                write_json(folder / 'predictions.json', predictions)
                if routes:
                    for mode, expected in (('shuffled', 'X'), ('absent', 'G')):
                        changed = remap(frames, scans, records, world.config, world.shape, mode)
                        changed_routes, _ = candidate_routes(changed, obs)
                        if changed_routes != routes:
                            raise ValueError('semantic interpretation changed the shared route pool')
                        actual = score_routes(changed, routes, prefix_poses)
                        np.testing.assert_array_equal(actual['scores']['S'], predictions['scores'][expected])
                        del changed
                q = mapper.quality_evidence(max_points=10**9)
                mesh = mapper.mesh()
                geom = array_hash([('belief', mapper.belief), ('camera_seen', mapper.camera_seen),
                                   ('vertices', np.asarray(mesh.vertices)), ('triangles', np.asarray(mesh.triangles)),
                                   ('normals', np.asarray(mesh.vertex_normals))]
                                  + ([] if q is None else [(k, q[k]) for k in sorted(q) if k != 'label']))
                np.savez_compressed(folder / 'prefix_map.npz', belief=mapper.belief, camera_seen=mapper.camera_seen)
                summary = {'context': context_id, 'arrangement': arrangement, 'prefix_paid_actions': 150,
                           'frames': len(frames), 'candidate_count': len(routes), 'candidate_status': audit['status'],
                           'roles': [r['group'] for r in routes], 'costs': [r['cost'] for r in routes],
                           'choices': predictions['selected_candidate_ids'], 'geometry_sha256': geom,
                           'two_marker_supported_assets': len(audit['measured_assets']) == 2
                           and all(a['marked_points'] > 0 for a in audit['measured_assets'])}
                summaries.append(summary)
                pair.append({'frames': frames, 'scans': scans, 'records': records, 'routes': routes,
                             'geometry_hash': geom, 'predictions': predictions})
                print('prefix prepared', summary, flush=True)
                del mapper, mesh, world
            pair_results[context_id] = check_pair(*pair)
            del pair
        # Future outcomes have not been constructed; seal all method choices now.
        seal = {str(p.relative_to(output)): file_hash(p) for context in protocol['parent_order']
                for p in (output / context).rglob('*') if p.is_file()}
        write_json(output / 'pre_evaluator_choices_seal.json', seal)
        write_json(output / 'paired_sensor_audit.json', pair_results)
        structural = []
        for summary in summaries:
            context_id, arrangement = summary['context'], summary['arrangement']
            folder = output / context_id / arrangement
            world = create_competition_world(context_id, arrangement)
            count = protocol['reference']['samples_requested']
            evaluator = ReconstructionEvaluator(world, count=count, seed=2026)
            weight = float(world.mesh.get_surface_area()) / count
            seen = np.zeros(len(evaluator.reference), bool)
            for i in range(151):
                frame = RGBDFrame.load(folder / 'prefix' / 'frames' / f'{i:04d}.npz')
                seen |= reference_visible(evaluator.reference, frame, evaluator.truth,
                                          frame.world_from_camera, world.config.max_depth_m)
            background = evaluator.classes == 1
            background_fraction = float(np.count_nonzero(seen & background) / np.count_nonzero(background))
            middle_x = world.config.width_m / 2
            slots = [(evaluator.classes != 1) & (evaluator.reference[:, 0] < middle_x),
                     (evaluator.classes != 1) & (evaluator.reference[:, 0] >= middle_x)]
            remaining = [float(np.count_nonzero(s & ~seen) * weight) for s in slots]
            records = read(folder / 'prefix' / 'records.json')
            last = records[-1]; start = (*last['position'], last['heading'])
            belief = np.load(folder / 'prefix_map.npz')['belief']
            safe = ~inflated_obstacles(belief != 0, world.config.robot_radius_m / world.config.resolution_m)
            centers = [p[0] for p in world.context.object_front_centers_xy_m]
            known_costs = visit_costs(safe, start, world.config, centers)
            optimistic = visit_costs(np.ones_like(safe), start, world.config, centers)
            rows = {'context': context_id, 'arrangement': arrangement,
                    'observable_background_fraction_actually_seen': background_fraction,
                    'remaining_slot_area_m2': remaining, 'unique_union_area_m2': float(world.mesh.get_surface_area()),
                    'knownsafe_visit_costs': known_costs, 'obstacle_free_visit_costs': optimistic,
                    'reference_points': len(evaluator.reference), 'weight_m2': weight,
                    'coverage_2d': float(np.count_nonzero((belief == 0) & world.reachable) / world.reachable.sum())}
            checks = {'candidate_roles': summary['roles'] == protocol['pre_outcome_structural_gates']['candidate_roles_in_order'],
                      'two_markers': summary['two_marker_supported_assets'],
                      'background_saturation': background_fraction >= .9,
                      'both_assets_have_unseen_surface': all(v > 1e-10 for v in remaining),
                      'one_region_feasible': all(v is not None and v <= 48 for v in known_costs['single']),
                      'two_regions_not_knownsafe_feasible': known_costs['both'] is None or known_costs['both'] > 48}
            rows['checks'] = checks; rows['passed'] = all(checks.values())
            np.savez_compressed(folder / 'reference.npz', points=evaluator.reference, classes=evaluator.classes,
                                weights=np.full(len(evaluator.reference), weight), prefix_seen=seen,
                                west_slot=slots[0], east_slot=slots[1], reachable=world.reachable,
                                vertices=np.asarray(world.mesh.vertices), triangles=np.asarray(world.mesh.triangles))
            write_json(folder / 'structural_audit.json', rows)
            structural.append(rows); print('structure', rows, flush=True)
        for context_id in protocol['parent_order']:
            a, b = [r for r in structural if r['context'] == context_id]
            tolerance = 1e-10 * max(1., abs(a['unique_union_area_m2']), abs(b['unique_union_area_m2']))
            pair_results[context_id]['union_area_equal'] = abs(a['unique_union_area_m2'] - b['unique_union_area_m2']) <= tolerance
        passed = all(r['passed'] for r in structural) and all(p['union_area_equal'] for p in pair_results.values())
        write_json(output / 'structure_summary.json', {'passed': passed, 'histories': structural,
                    'pairs': pair_results, 'prefix_summaries': summaries, 'new_candidate_branches': 0})
        if not all(file_hash(output / name) == digest for name, digest in seal.items()):
            raise RuntimeError('sealed choices or raw prefixes changed during structural evaluation')
        if not all(file_hash(ROOT / name) == digest for name, digest in hashes.items()):
            raise RuntimeError('source dependency changed during preparation')
        metadata.update(status='complete_structural_pass' if passed else 'halted_structural_failure',
                        structural_gate_passed=passed, elapsed_s=time.time() - started)
    except Exception as exc:
        metadata.update(status='failed_preparation', error=f'{type(exc).__name__}: {exc}', elapsed_s=time.time() - started)
        raise
    finally:
        write_json(output / 'metadata.json', metadata)
        write_json(output / 'artifact_hashes.json', {str(p.relative_to(output)): file_hash(p)
                    for p in output.rglob('*') if p.is_file() and p.name != 'artifact_hashes.json'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    prepare(parser.parse_args().output)
