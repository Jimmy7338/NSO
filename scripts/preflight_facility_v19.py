#!/usr/bin/env python3
"""Establish coverage witnesses and conservative near-visit budget exclusions.

This is a truth-owned geometry feasibility calculation, not a policy rollout,
semantic reward, train/test sample, or a claim of complete asset documentation.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from utils.facility_budget_v19 import (advance_pose, conservative_asset_regions,
    exact_region_walk_lower_bound, grid_distances, paired_budget, region_tour_lower_bound,
    scan_known_mask, shortest_actions)
from utils.grid_geometry import DIRECTIONS
from env.virtual3d import camera_pose
from env.canonical_box_scan_v14 import box_union_ranges


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + '\n')


def array_sha(value):
    value = np.ascontiguousarray(value)
    return hashlib.sha256(str(value.dtype).encode() + repr(value.shape).encode() + value.tobytes()).hexdigest()


class ScanAtlas:
    def __init__(self, world):
        self.world = world
        self.cache = {}

    def mask(self, pose):
        pose = tuple(map(int, pose))
        if pose not in self.cache:
            world = self.world
            previous = world.position, world.heading
            try:
                world.position, world.heading = pose[:2], pose[2]
                self.cache[pose] = scan_known_mask(world.scan(), world.shape, world.config.resolution_m)
            finally:
                world.position, world.heading = previous
        return self.cache[pose]

    def follow(self, initial, known, actions):
        state = tuple(initial); result = known.copy()
        for action in actions:
            state = advance_pose(state, action)
            if not self.world.reachable[state[:2]]:
                raise AssertionError('geometry witness left safe floor')
            result |= self.mask(state)
        return state, result


def coverage_witness_candidates(world, target=.8, strides=(5,), return_weights=(0., .4, 1.)):
    """Search only laser-coverage geometry; retain every deterministic trial cost."""
    safe = world.reachable; initial = (*world.start, world.heading)
    atlas = ScanAtlas(world); start_distances = grid_distances(safe, [world.start])
    total = int(safe.sum()); trials = []; routes = []
    def coverage(known):
        return float(np.count_nonzero(known & safe) / total)
    home_cache = {}
    def inexpensive_home(state):
        if state in home_cache:
            return home_cache[state]
        origin = state; route = []
        while state[:2] != world.start:
            r, c, heading = state; choices = []
            for h, (dr, dc) in enumerate(DIRECTIONS):
                nxt = r + dr, c + dc
                if 0 <= nxt[0] < safe.shape[0] and 0 <= nxt[1] < safe.shape[1] and start_distances[nxt] == start_distances[r, c] - 1:
                    delta = (h - heading) % 4
                    turn = [] if delta == 0 else ['right'] if delta == 1 else ['left'] if delta == 3 else ['right', 'right']
                    choices.append((len(turn), h, turn))
            _, h, turn = min(choices)
            for action in turn + ['forward']:
                route.append(action); state = advance_pose(state, action)
        delta = (initial[2] - state[2]) % 4
        route.extend([] if delta == 0 else ['right'] if delta == 1 else ['left'] if delta == 3 else ['right', 'right'])
        home_cache[origin] = route
        return route
    def shorten_with_return(actions):
        state = initial; known = atlas.mask(initial).copy(); best = actions
        for k in range(len(actions) + 1):
            if k:
                state = advance_pose(state, actions[k - 1]); known |= atlas.mask(state)
            home = inexpensive_home(state)
            if k + len(home) >= len(best):
                continue
            _, result = atlas.follow(state, known, home)
            if coverage(result) >= target:
                best = actions[:k] + home
        return best
    for stride in strides:
        rr, cc = np.indices(world.shape)
        cells = [tuple(map(int, x)) for x in np.argwhere(safe & (rr % stride == 0) & (cc % stride == 0))]
        if world.start not in cells:
            cells.append(world.start)
        observations = [(cell, atlas.mask((*cell, 0))) for cell in cells]
        for weight in return_weights:
            state = initial; known = atlas.mask(state).copy(); actions = []
            for iteration in range(64):
                home = shortest_actions(safe, state, world.start, initial[2])
                _, at_home = atlas.follow(state, known, home)
                if coverage(at_home) >= target:
                    actions.extend(home); known = at_home
                    break
                distance = grid_distances(safe, [state[:2]])
                choices = []
                for cell, mask in observations:
                    gain = int(np.count_nonzero(mask & safe & ~known))
                    if gain and distance[cell] >= 0 and cell != state[:2]:
                        score = gain / (int(distance[cell]) + weight * int(start_distances[cell]) + 1)
                        choices.append((score, gain, -int(distance[cell]), cell))
                if not choices:
                    break
                goal = max(choices)[-1]
                route = shortest_actions(safe, state, goal, 0)
                state, known = atlas.follow(state, known, route)
                actions.extend(route)
            unshortened_actions = len(actions)
            actions = shorten_with_return(actions)
            final_state, final_known = atlas.follow(initial, atlas.mask(initial), actions)
            passed = coverage(final_known) >= target and final_state == initial
            trials.append(dict(grid_stride_cells=stride, return_weight=weight, actions=len(actions),
                               unshortened_actions=unshortened_actions,
                               coverage_2d=coverage(final_known), returned=final_state == initial, passed=passed))
            if passed:
                routes.append((len(actions), actions, trials[-1]))
    if not routes:
        return dict(status='failed', reason='no declared geometry witness reached coverage and return', trials=trials), None
    _, actions, chosen = min(routes, key=lambda row: row[0])
    return dict(status='proposed', trials=trials, selected_trial=chosen,
                scan_atlas_pose_count=len(atlas.cache)), actions


def execute_witness(world, actions, target=.8):
    """Validate using actual world.step RGB-D and world.scan, without TSDF."""
    initial = (*world.start, world.heading)
    if (*world.position, world.heading) != initial or world.step_count != 0:
        raise ValueError('coverage witness must start from a fresh world')
    first = world.sense(); first.validate()
    known = scan_known_mask(world.scan(), world.shape, world.config.resolution_m)
    digest = hashlib.sha256(); trace = []
    for action in actions:
        frame, collision, done = world.step(action)
        frame.validate()
        if collision or (done and world.step_count < len(actions)):
            raise ValueError('actual witness collided or exhausted simulator horizon')
        scan = world.scan(); known |= scan_known_mask(scan, world.shape, world.config.resolution_m)
        digest.update(np.ascontiguousarray(scan.ranges_m).tobytes())
        digest.update(np.ascontiguousarray(scan.world_from_laser).tobytes())
        trace.append([*map(int, world.position), int(world.heading)])
    coverage = float(np.count_nonzero(known & world.reachable) / world.reachable.sum())
    returned = (*world.position, world.heading) == initial
    if coverage < target or not returned or world.collisions:
        raise ValueError('actual sensor execution did not validate proposed coverage-return witness')
    return dict(status='passed', paid_actions=len(actions), actions=actions, pose_trace=trace,
                coverage_2d=coverage, returned=returned, collisions=int(world.collisions),
                scan_trace_sha256=digest.hexdigest(), known_mask_sha256=array_sha(known),
                rgbd_frame_validation_count=len(actions) + 1, tsdf_used=False,
                purpose='truth-owned laser coverage feasibility witness; no policy or quality reward')


def all_asset_observation_witness(world, coverage_actions, near_range=1.2):
    """A feasible upper bound with actual visible front observations of all assets.

    The fixed front points constrain only this sufficient witness, never L_all.
    Extra replay of the coverage witness is allowed if this tour alone misses 80%.
    """
    targets = []; near_audit = []; regions = []
    for item in world.objects:
        point = np.asarray(item['front_center'], dtype=float)
        target = world._cell(float(point[0]), float(point[1] - .8))
        if not world.reachable[target]:
            raise ValueError('declared front observation witness is not safe')
        pose = camera_pose(target, 0, world.config, world.shape[0])
        delta = point - pose[:3, 3]; distance = float(np.linalg.norm(delta))
        ray = delta / distance
        hit = float(box_union_ranges(pose[:3, 3], ray[None],
                    np.asarray(world._solid_primitives)[:, :6], world.config.max_depth_m)[0])
        local = pose[:3, :3].T @ delta
        pixel = world.intrinsic @ local
        u, v = pixel[0] / pixel[2], pixel[1] / pixel[2]
        if distance > near_range or abs(hit - distance) > 1e-5 or local[2] <= 0 or not (0 <= u < world.config.width_px and 0 <= v < world.config.height_px):
            raise ValueError('front observation is not close, visible and in the real camera frustum')
        targets.append(target)
        region = np.zeros(world.shape, bool); region[target] = True; regions.append(region)
        near_audit.append(dict(asset_id=int(item['id']), front_point=point.tolist(), target_pose=[*target, 0],
                              physical_range_m=distance, line_of_sight=True, in_camera_frustum=True))
    order = region_tour_lower_bound(world.reachable, world.start, regions)['relaxed_order']
    initial = (*world.start, world.heading); state = initial; actions = []
    for index in order:
        route = shortest_actions(world.reachable, state, targets[index], 0)
        actions.extend(route)
        for action in route:
            state = advance_pose(state, action)
    actions.extend(shortest_actions(world.reachable, state, world.start, initial[2]))
    atlas = ScanAtlas(world); _, known = atlas.follow(initial, atlas.mask(initial), actions)
    appended_coverage = float(np.count_nonzero(known & world.reachable) / world.reachable.sum()) < .8
    if appended_coverage:
        actions.extend(coverage_actions)
    if len(actions) > world.config.max_steps:
        raise ValueError('sufficient close-visit witness exceeds simulator horizon')
    result = execute_witness(world, actions)
    trace = {tuple(initial), *(tuple(pose) for pose in result['pose_trace'])}
    if any(tuple(row['target_pose']) not in trace for row in near_audit):
        raise AssertionError('actual witness did not reach every certified front observation')
    result.update(near_observations=near_audit, near_visit_asset_count=len(near_audit),
                  appended_coverage_witness=appended_coverage,
                  purpose='feasible coverage + all-asset visible <=1.2m front observation + return; not full-quality completion')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--freeze', action='store_true')
    parser.add_argument('--parents', nargs='+', default=['D19-P00', 'D19-P01'])
    parser.add_argument('--stride', type=int, default=5)
    args = parser.parse_args()
    from env.facility_documentation_v19 import FacilityWorldV19
    args.output.mkdir(parents=True, exist_ok=False)
    sources = {str(path.relative_to(ROOT)): sha(path) for folder in ['env', 'utils'] for path in sorted((ROOT / folder).glob('*.py'))}
    for name in ['scripts/preflight_facility_v19.py', 'tests/virtual3d/test_facility_budget_v19.py']:
        if (ROOT / name).exists():
            sources[name] = sha(ROOT / name)
    manifest = dict(status='running', frozen=args.freeze, source_sha256=sources,
                    policy_outcomes=0, semantic_rewards=0, training_allowed=False)
    write(args.output / 'manifest.json', manifest)
    records = []
    try:
        for parent in args.parents:
            variants = []
            for assignment in ['A_complex_B_simple', 'A_simple_B_complex']:
                world = FacilityWorldV19(parent=parent, assignment=assignment)
                bounds = []
                for obj in world.objects:
                    containing = np.asarray(obj.get('near_visit_bounds', obj['evaluation_bounds']))
                    vertices = np.asarray(world.instance_mesh(obj['id']).vertices)
                    if np.any(vertices[:, :2] < containing[0, :2] - 1e-8) or np.any(vertices[:, :2] > containing[1, :2] + 1e-8):
                        raise ValueError('near-visit horizontal bounds do not contain complete asset mesh')
                    bounds.append(containing.tolist())
                regions, service_audit = conservative_asset_regions(world, 1.2)
                relaxation = region_tour_lower_bound(world.reachable, world.start, regions)
                lower = exact_region_walk_lower_bound(world.reachable, world.start, regions)
                if lower['lower_bound_translation_actions'] < relaxation['lower_bound_translation_actions']:
                    raise AssertionError('exact walk cannot be below the region relaxation')
                row = dict(assignment=assignment, config=asdict(world.config),
                    geometry_mesh_sha256=array_sha(np.asarray(world.mesh.vertices)),
                    service_region_contract='safe cells within horizontal 1.2m of containing asset AABB; view/LoS/height relaxed',
                    near_visit_range_m=1.2, service_regions=service_audit,
                    region_relaxation=relaxation, lower_bound=lower)
                attempt, actions = coverage_witness_candidates(world, strides=(args.stride,))
                row['coverage_search'] = attempt
                if actions is None:
                    row['status'] = 'failed'
                else:
                    row['coverage_witness'] = execute_witness(world, actions)
                    fresh = FacilityWorldV19(parent=parent, assignment=assignment)
                    row['sufficient_witness'] = all_asset_observation_witness(fresh, actions)
                    row['status'] = 'complete'
                variants.append(row)
                write(args.output / f'{parent}_{assignment}.json', row)
                print(parent, assignment, 'L_all', lower['lower_bound_translation_actions'],
                      'T_cov', row.get('coverage_witness', {}).get('paid_actions'), flush=True)
            if all(row['status'] == 'complete' for row in variants):
                budget = paired_budget([r['coverage_witness']['paid_actions'] for r in variants],
                                       [r['lower_bound']['lower_bound_translation_actions'] for r in variants])
                budget['sufficient_budget'] = max(r['sufficient_witness']['paid_actions'] for r in variants)
                budget['sufficient_budget_claim'] = 'actual 80% coverage + 6 visible close front observations + return; quality completion not guaranteed'
            else:
                budget = dict(status='failed', reason='coverage witness missing', budget=None)
            records.append(dict(parent=parent, variants=variants, budget=budget))
            write(args.output / f'{parent}.json', records[-1])
        passed = all(row['budget']['status'] == 'passed' for row in records)
        write(args.output / 'result.json', dict(status='passed' if passed else 'failed', parents=records,
            policy_outcomes=0, semantic_rewards=0, training_allowed=False,
            lower_bound_claim='at least one <=1.2m exterior visit per asset, not complete reconstruction',
            source_frozen=args.freeze))
        changed_sources = [name for name, expected in sources.items() if sha(ROOT / name) != expected]
        manifest['source_changes_during_run'] = changed_sources
        if args.freeze:
            if changed_sources:
                raise ValueError('source changed during freeze preflight: ' + ', '.join(changed_sources))
            with zipfile.ZipFile(args.output / 'sources.zip', 'x', zipfile.ZIP_DEFLATED) as archive:
                for name in sources:
                    archive.write(ROOT / name, name)
        manifest['status'] = 'complete' if passed else 'failed'
    except Exception as error:
        manifest.update(status='failed', error=repr(error))
        raise
    finally:
        write(args.output / 'manifest.json', manifest)
        write(args.output / 'artifact_hashes.json', {path.name: sha(path) for path in sorted(args.output.iterdir())
            if path.is_file() and path.name != 'artifact_hashes.json'})


if __name__ == '__main__':
    main()
