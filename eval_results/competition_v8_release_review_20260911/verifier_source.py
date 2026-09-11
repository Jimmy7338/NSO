#!/usr/bin/env python3
"""Independent additional release gates on a sealed V8 prefix preparation.

No future branch is generated. The frozen mapper and sensor simulator are
reused; grouping, slot assignment, footprint/path checks, and visit-both BFS
are computed here. Truth is consulted only after the choice seal is verified,
for evaluator-side diagnostics; no candidate or score is repaired.
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
from collections import deque
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import traceback
import zipfile


def read(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path, value): Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def next_state(state, action):
    r, c, h = state
    if action == 'left': return r, c, (h - 1) % 4
    if action == 'right': return r, c, (h + 1) % 4
    assert action == 'forward'
    dr, dc = ((-1, 0), (0, 1), (1, 0), (0, -1))[h]
    return r + dr, c + dc, h


def visit_bfs(safe, anchor, config, centers):
    import numpy as np
    yy, xx = np.indices(safe.shape)
    xs, ys = (xx + .5) * config.resolution_m, (safe.shape[0] - yy - .5) * config.resolution_m
    regions = [(np.abs(xs - x) <= .3 + 1e-12) & (ys >= 5.9 - 1e-12) & (ys <= 6.5 + 1e-12) for x in centers]
    counts = [int(np.count_nonzero(r & safe) * 4) for r in regions]
    if not safe[anchor[:2]]:
        return {'single': [None, None], 'both': None, 'region_state_counts': counts}
    flags = regions[0].astype(int) + 2 * regions[1].astype(int)
    initial = (*anchor, int(flags[anchor[:2]]))
    queue, visited = deque([(initial, 0)]), {initial}
    single, both = [None, None], None
    while queue:
        state, distance = queue.popleft()
        pose, bits = state[:3], state[3]
        if pose == anchor:
            for index in range(2):
                if bits & (1 << index) and single[index] is None: single[index] = distance
            if bits == 3:
                both = distance
                break
        for action in ('left', 'right', 'forward'):
            nxt = next_state(pose, action)
            if not (0 <= nxt[0] < safe.shape[0] and 0 <= nxt[1] < safe.shape[1] and safe[nxt[:2]]): continue
            extended = (*nxt, bits | int(flags[nxt[:2]]))
            if extended not in visited:
                visited.add(extended); queue.append((extended, distance + 1))
    return {'single': single, 'both': both, 'region_state_counts': counts}


def groups_from_raw_mapper(mapper, first_pose):
    """Independent geometry grouping and sampled measured-AABB construction."""
    import numpy as np
    from scipy.ndimage import binary_closing, label
    q = mapper.quality_evidence(max_points=10000)
    if q is None: return []
    points = q['point']; shape = mapper.shape
    cells = np.column_stack([shape[0] - 1 - np.floor(points[:, 1] / .2), np.floor(points[:, 0] / .2)]).astype(int)
    inside = ((cells >= 0) & (cells < np.array(shape))).all(1)
    occupied = np.zeros(shape, bool); occupied[tuple(cells[inside].T)] = True
    components, _ = label(binary_closing(occupied, structure=np.ones((3, 3))))
    ids = np.zeros(len(points), int); ids[inside] = components[tuple(cells[inside].T)]
    directions = np.array([[0., 1.], [1., 0.], [0., -1.], [-1., 0.]])
    groups = []
    for group in sorted(set(ids) - {0}):
        indexes = np.flatnonzero(ids == group); cloud = points[indexes]
        raw_low, raw_high = np.quantile(cloud, [.05, .95], axis=0)
        span = raw_high - raw_low
        if len(indexes) < 8 or max(span[:2]) > 2.2 or span[2] < .25: continue
        selection = np.linspace(0, len(indexes) - 1, min(len(indexes), 128), dtype=int)
        sampled = cloud[selection]; low, high = np.quantile(sampled, [.05, .95], axis=0)
        center = (low + high) / 2
        front = directions[np.argmax(directions @ (first_pose[:2, 3] - center[:2]))]
        back = -front; side = np.array([-back[1], back[0]])
        depth = float((high[:2] - low[:2]) @ np.abs(back))
        labels = q['label'][indexes]; marked = (labels == 2) | (labels == 3)
        groups.append({'group': int(group), 'observed_low': low, 'observed_high': high,
                       'aabb_center': center, 'front_axis': front, 'back_axis': back, 'side_axis': side,
                       'rear_boundary_xy': center[:2] + back * depth / 2,
                       'measured_width_m': float((high[:2] - low[:2]) @ np.abs(side)),
                       'measured_depth_m': depth, 'measured_height_m': float(high[2] - low[2]),
                       'marked_points': int(marked.sum()), 'class_vote': float(np.mean(2 * (labels[marked] == 3) - 1)) if marked.any() else 0.,
                       'points': sampled, 'cloud': cloud, 'marked_cloud': cloud[marked]})
    return sorted(groups, key=lambda g: (g['aabb_center'][0], g['aabb_center'][1], g['group']))


def slot_correspondence(groups, objects):
    """Nearest physical AABB memberships are evaluator-only, no repair."""
    import numpy as np
    boxes = np.array([o['box'] for o in objects], float)
    rows = []
    for g in groups:
        cloud = g['cloud']
        delta = np.maximum(np.maximum(boxes[:, None, :3] - cloud[None],
                                      cloud[None] - (boxes[:, :3] + boxes[:, 3:])[:, None]), 0.)
        squared = (delta * delta).sum(2)
        assigned = np.argmin(squared, axis=0)
        indexes, counts = np.unique(assigned, return_counts=True)
        unambiguous = bool(np.all(np.abs(squared[0] - squared[1]) > 1e-12))
        rows.append({'group': g['group'], 'physical_slot_indexes': indexes.tolist(),
                     'nearest_slot_point_counts': counts.tolist(), 'all_points_unambiguous': unambiguous,
                     'marked_points': g['marked_points'], 'mapped_cloud_points': len(cloud),
                     'max_nearest_box_distance_m': float(np.sqrt(squared.min(0)).max()),
                     'single_slot': len(indexes) == 1 and unambiguous,
                     'slot_index': int(indexes[0]) if len(indexes) == 1 else None})
    passed = (len(rows) == 2 and all(r['single_slot'] and r['marked_points'] > 0 for r in rows)
              and {r['slot_index'] for r in rows} == {0, 1})
    return {'passed': passed, 'rule': 'every point in each independently rebuilt accepted geometry group has one strict nearest physical AABB; the two groups occupy different slots; actual markers required',
            'truth_used_after_choice_seal_only': True, 'groups_or_routes_repaired': False, 'rows': rows}


def worker(args):
    import numpy as np
    from scipy.ndimage import distance_transform_edt
    sys.path.insert(0, str(args.snapshot))
    from env.virtual3d_competition_v8 import CompetitionWorldV8, CompetitionContextV8, CompetitionConfigV8
    from env.virtual3d import camera_pose
    from nso.semantic_completion_v3 import SemanticHistoryMapperV3
    from utils.rgbd_contract import RGBDFrame, PlanarScan
    from utils.reconstruction_metrics import ReconstructionEvaluator
    from utils.counterfactual_surface_visibility import reference_visible
    protocol = read(args.snapshot / 'configs/virtual3d/competition_v8_protocol.json')
    original_summary = read(args.run / 'structure_summary.json')
    manifest = read(args.run / 'artifact_hashes.json'); choice_seal = read(args.run / 'pre_evaluator_choices_seal.json')
    for name, digest in choice_seal.items(): assert sha(args.run / name) == digest
    reports, pairs, raw_count = [], {}, 0
    for context_id in protocol['parent_order']:
        pair = []
        for arrangement in protocol['arrangement_order']:
            folder = args.run / context_id / arrangement
            fixture = read(folder / 'fixture.json'); c = CompetitionConfigV8(**fixture['config'])
            context = CompetitionContextV8(**fixture['context'])
            world = CompetitionWorldV8(context, arrangement, c)
            mapper = SemanticHistoryMapperV3(world.shape, c, c.truncation_m)
            records = read(folder / 'prefix/records.json'); assert len(records) == 151
            frames, scans = [], []
            for index, row in enumerate(records):
                if index:
                    assert row['action'] == world.prefix_actions[index - 1]
                    frame, collision, done = world.step(row['action'])
                else: frame, collision, done = world.sense(), False, False
                scan = world.scan()
                saved = RGBDFrame.load(folder / 'prefix/frames' / f'{index:04d}.npz')
                saved_scan = PlanarScan.load(folder / 'prefix/scans' / f'{index:04d}.npz')
                for actual, expected in ((frame, saved), (scan, saved_scan)):
                    for field in fields(type(actual)):
                        np.testing.assert_array_equal(getattr(actual, field.name), getattr(expected, field.name))
                assert not collision and not done and row['position'] == list(world.position) and row['heading'] == world.heading
                assert row['step'] == world.step_count == index
                mapper.update(saved, saved_scan); frames.append(saved); scans.append(saved_scan); raw_count += 1
            with np.load(folder / 'prefix_map.npz') as saved:
                np.testing.assert_array_equal(mapper.belief, saved['belief'])
                np.testing.assert_array_equal(mapper.camera_seen, saved['camera_seen'])
            groups = groups_from_raw_mapper(mapper, frames[0].world_from_camera)
            audit = read(folder / 'candidate_audit.json')
            assert len(groups) == len(audit['measured_assets'])
            for computed, recorded in zip(groups, audit['measured_assets']):
                for key, value in computed.items():
                    if key not in ('points', 'cloud', 'marked_cloud'):
                        np.testing.assert_allclose(value, recorded[key], rtol=0, atol=1e-12, err_msg=key)
            assignment = slot_correspondence(groups, world.objects)
            safe = distance_transform_edt(np.pad(mapper.belief == 0, 1, constant_values=False))[1:-1, 1:-1] > c.robot_radius_m / c.resolution_m + np.sqrt(2) / 2
            assert hashlib.sha256(safe.tobytes()).hexdigest() == audit['safe_sha256']
            anchor = (*world.position, world.heading)
            routes = read(folder / 'candidates.json'); role_rows = []
            assert len({tuple(r['pose']) for r in routes}) == len(routes)
            for i, route in enumerate(routes):
                assert route['candidate_id'] == i and 1 <= route['cost'] <= 48
                assert len(route['actions']) == len(route['states']) - 1 == route['cost']
                assert tuple(route['states'][0]) == tuple(route['states'][-1]) == anchor
                assert all(safe[tuple(state[:2])] for state in route['states'])
                for first, second, action in zip(route['states'], route['states'][1:], route['actions']):
                    assert next_state(tuple(first), action) == tuple(second)
                assert route['states'][route['arrival_action']] == route['pose']
                pose = camera_pose(route['pose'][:2], route['pose'][2], c, world.shape[0])
                row = {'role': route['group'], 'candidate_id': i, 'paid_round_trip_valid': True}
                if route['group'] == 'old_surface_rotation':
                    assert tuple(route['pose'][:2]) == anchor[:2]
                    supports = []
                    for group in groups:
                        local = (group['points'] - pose[:3, 3]) @ pose[:3, :3]
                        z = local[:, 2]; tangent = np.tan(np.deg2rad(c.fov_deg / 2))
                        visible = ((z > .15) & (z <= c.max_depth_m) & (np.abs(local[:, 0]) < z * tangent)
                                   & (np.abs(local[:, 1]) < z * tangent * c.height_px / c.width_px))
                        supports.append(float(visible.mean()))
                    row['observed_point_fov_support'] = max(supports, default=0.)
                    assert row['observed_point_fov_support'] > 0
                if route['asset_index'] is not None:
                    group = groups[route['asset_index']]
                    depth = float((pose[:2, 3] - group['rear_boundary_xy']) @ group['back_axis'])
                    desired_offset = .4 if route['group'].endswith('entry') else .8
                    error = float(np.linalg.norm(pose[:2, 3] - (group['rear_boundary_xy'] + desired_offset * group['back_axis'])))
                    assert depth > .15 and error <= .35
                    row.update(asset_index=route['asset_index'], back_axis_distance_m=depth, target_error_m=error)
                role_rows.append(row)
            for slot in (0, 1):
                entry = [r for r in role_rows if r.get('asset_index') == slot and r['role'].endswith('entry')]
                deep = [r for r in role_rows if r.get('asset_index') == slot and r['role'].endswith('deep')]
                if deep: assert entry and deep[0]['back_axis_distance_m'] >= entry[0]['back_axis_distance_m'] + .2 - 1e-12
            centers = [o['box'][0] + o['box'][3] / 2 for o in world.objects]
            known, optimistic = visit_bfs(safe, anchor, c, centers), visit_bfs(np.ones_like(safe), anchor, c, centers)
            structural = read(folder / 'structural_audit.json')
            assert known == structural['knownsafe_visit_costs'] and optimistic == structural['obstacle_free_visit_costs']
            evaluator = ReconstructionEvaluator(world, count=32000, seed=2026)
            with np.load(folder / 'reference.npz') as saved:
                for name, array in (('points', evaluator.reference), ('classes', evaluator.classes), ('vertices', np.asarray(world.mesh.vertices)), ('triangles', np.asarray(world.mesh.triangles)), ('reachable', world.reachable)):
                    np.testing.assert_array_equal(saved[name], array)
                seen = np.zeros(len(evaluator.reference), bool)
                for frame in frames:
                    seen |= reference_visible(evaluator.reference, frame, evaluator.truth, frame.world_from_camera, c.max_depth_m)
                np.testing.assert_array_equal(saved['prefix_seen'], seen)
                weight = world.mesh.get_surface_area() / 32000
                np.testing.assert_array_equal(saved['weights'], np.full(len(seen), weight))
                background = evaluator.classes == 1
                fraction = float(np.count_nonzero(seen & background) / np.count_nonzero(background))
                slots = [(evaluator.classes != 1) & (evaluator.reference[:, 0] < c.width_m / 2),
                         (evaluator.classes != 1) & (evaluator.reference[:, 0] >= c.width_m / 2)]
                for name, mask in zip(('west_slot', 'east_slot'), slots): np.testing.assert_array_equal(saved[name], mask)
                remaining = [float(np.count_nonzero(mask & ~seen) * weight) for mask in slots]
            np.testing.assert_allclose(fraction, structural['observable_background_fraction_actually_seen'], rtol=0, atol=1e-12)
            np.testing.assert_allclose(remaining, structural['remaining_slot_area_m2'], rtol=0, atol=1e-12)
            checks = {'candidate_roles': [r['group'] for r in routes] == protocol['pre_outcome_structural_gates']['candidate_roles_in_order'],
                      'two_markers': len(groups) == 2 and all(g['marked_points'] > 0 for g in groups),
                      'background_saturation': fraction >= .9, 'both_assets_have_unseen_surface': all(a > 1e-10 for a in remaining),
                      'one_region_feasible': all(v is not None and v <= 48 for v in known['single']),
                      'two_regions_not_knownsafe_feasible': known['both'] is None or known['both'] > 48}
            assert checks == structural['checks'] and all(checks.values()) == structural['passed']
            reports.append({'context': context_id, 'arrangement': arrangement, 'raw_prefix_frames_exact': 151,
                            'independent_geometry_groups_exact': True, 'slot_correspondence': assignment,
                            'role_and_path_checks': role_rows, 'candidate_roles': [r['group'] for r in routes],
                            'missing_roles': audit['missing_roles'], 'structural_checks': checks,
                            'knownsafe_visit_costs': known, 'obstacle_free_visit_costs': optimistic,
                            'background_fraction': fraction, 'remaining_slot_area_m2': remaining})
            pair.append({'occupancy': world.occupancy.copy(), 'blocked': world._blocked.copy(), 'reachable': world.reachable.copy(),
                         'area': float(world.mesh.get_surface_area()), 'frames': frames, 'scans': scans,
                         'belief': mapper.belief.copy(), 'camera_seen': mapper.camera_seen.copy(), 'routes': routes,
                         'predictions': read(folder / 'predictions.json')})
            print('independent release audit', context_id, arrangement, 'candidates', len(routes), 'distinct slots', assignment['passed'], flush=True)
            del mapper, evaluator, world
        first, second = pair
        for key in ('occupancy', 'blocked', 'reachable', 'belief', 'camera_seen'):
            np.testing.assert_array_equal(first[key], second[key])
        assert first['routes'] == second['routes']
        for a, b, x, y in zip(first['frames'], second['frames'], first['scans'], second['scans']):
            for key in ('depth_m', 'intrinsic', 'world_from_camera', 'timestamp_s'): np.testing.assert_array_equal(getattr(a, key), getattr(b, key))
            for field in fields(type(x)): np.testing.assert_array_equal(getattr(x, field.name), getattr(y, field.name))
            np.testing.assert_array_equal(a.semantic > 0, b.semantic > 0)
            np.testing.assert_array_equal(a.color_rgb[a.semantic == 0], b.color_rgb[b.semantic == 0])
            np.testing.assert_array_equal(np.where(a.semantic == 2, 3, np.where(a.semantic == 3, 2, 0)), b.semantic)
        for mode in ('N', 'G', 'O', 'M'):
            np.testing.assert_array_equal(first['predictions']['scores'][mode], second['predictions']['scores'][mode])
        np.testing.assert_array_equal(first['predictions']['scores']['S'], second['predictions']['scores']['X'])
        tolerance = 1e-10 * max(1., abs(first['area']), abs(second['area']))
        assert abs(first['area'] - second['area']) <= tolerance
        pairs[context_id] = {'physical_occupancy_exact': True, 'physical_blocked_exact': True,
                             'physical_reachable_exact': True, 'raw_nonlabel_sensors_exact': True,
                             'actual_marker_swap_exact': True, 'candidate_pool_exact': True,
                             'nonsemantic_scores_exact': True, 'union_area_equal_within_fixed_tolerance': True}
        del pair
    for name, digest in choice_seal.items(): assert sha(args.run / name) == digest
    for name, module in list(sys.modules.items()):
        if name.split('.')[0] in ('env', 'nso', 'utils') and getattr(module, '__file__', None):
            assert Path(module.__file__).resolve().is_relative_to(args.snapshot), name
    structural_pass = all(all(r['structural_checks'].values()) for r in reports)
    assert structural_pass == original_summary['passed']
    extra_pass = all(r['slot_correspondence']['passed'] for r in reports)
    write(args.output / 'summary.json', {'status': 'passed_independent_release_review', 'release_permitted': structural_pass and extra_pass,
          'original_structural_gate_passed': structural_pass, 'additional_slot_and_ground_footprint_gates_passed': extra_pass,
          'raw_prefix_frames_and_scans_checked': raw_count, 'future_branches_generated': 0,
          'mapper_and_simulator_reused_not_independently_implemented': True,
          'candidate_or_prior_repaired': False, 'common_score_formula_recomputed_independently': False,
          'scope': 'independent structural release audit; no semantic utility or new navigation claims',
          'histories': reports, 'pairs': pairs})


def run(args):
    args.run = args.run.resolve(); args.output = args.output.resolve(); args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic(); own_sha = sha(__file__)
    try:
        meta = read(args.run / 'metadata.json'); assert meta['status'] in ('complete_structural_pass', 'halted_structural_failure')
        manifest = read(args.run / 'artifact_hashes.json')
        for name, digest in manifest.items(): assert sha(args.run / name) == digest, name
        with tempfile.TemporaryDirectory(prefix='competition-v8-release-audit-') as directory:
            snapshot = Path(directory) / 'source'; snapshot.mkdir()
            with zipfile.ZipFile(args.run / 'sources.zip') as archive:
                assert set(archive.namelist()) == set(meta['source_sha256'])
                for name, digest in meta['source_sha256'].items():
                    target = (snapshot / name).resolve(); assert target.is_relative_to(snapshot)
                    content = archive.read(name); assert hashlib.sha256(content).hexdigest() == digest
                    target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(content)
            subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker', '--snapshot', str(snapshot),
                            '--run', str(args.run), '--output', str(args.output)], cwd=snapshot, check=True)
        for name, digest in manifest.items(): assert sha(args.run / name) == digest, name
        assert sha(__file__) == own_sha
        summary = read(args.output / 'summary.json')
        summary.update(run=str(args.run), artifact_manifest_sha256=sha(args.run / 'artifact_hashes.json'),
                       choices_seal_sha256=sha(args.run / 'pre_evaluator_choices_seal.json'),
                       source_archive_sha256=sha(args.run / 'sources.zip'), source_sha256=meta['source_sha256'],
                       input_hashes_rechecked_after_review=True, verifier_sha256=own_sha, elapsed_s=time.monotonic() - started)
        write(args.output / 'summary.json', summary)
        write(args.output / 'input_hashes.json', {str(args.run / n): digest for n, digest in manifest.items()})
        print(json.dumps({k: v for k, v in summary.items() if k not in ('histories', 'pairs', 'source_sha256')}, ensure_ascii=False), flush=True)
    except Exception:
        write(args.output / 'failure.json', {'status': 'failed_independent_review', 'traceback': traceback.format_exc()})
        raise
    finally:
        write(args.output / 'artifact_hashes.json', {p.name: sha(p) for p in args.output.iterdir() if p.is_file() and p.name != 'artifact_hashes.json'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True); parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worker', action='store_true'); parser.add_argument('--snapshot', type=Path)
    args = parser.parse_args()
    worker(args) if args.worker else run(args)
