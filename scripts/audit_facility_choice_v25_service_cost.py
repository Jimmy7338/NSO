#!/usr/bin/env python3
"""One frozen V25r1 near-observation cost audit; no physical/sensor/TSDF/Q run."""
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import argparse
from collections import deque
import hashlib
import json
import math
from pathlib import Path
import shutil
import signal
import sys
from time import perf_counter
import traceback
import zipfile

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scripts.audit_facility_choice_v25_static_r1 import StaticWorld, projected
from env.facility_choice_v25_r1 import PARENTS_V25, ASSIGNMENTS_V25, VERSION_V25
from env.virtual3d import camera_pose
from utils.grid_geometry import DIRECTIONS

OUTPUT = ROOT/'audit_results/facility_choice_v25_service_cost_20260915'
PROTOCOL = ROOT/'docs/research/V25_SERVICE_COST_PROTOCOL_20260915.md'
R1 = ROOT/'audit_results/facility_choice_v25_static_r1_20260915'
INPUT_SHA = {
    'manifest.json': '3f9394de7c77adab8733d8cd9ee88ac54cd63a9e3d02513f9c2aa416d403cf13',
    'result.json': '7a4acce52e8a5294a8e7d89122d04ef5b8fec5b01d59b1d4aa3d665c1f7019c8',
    'artifact_hashes.json': 'c8db1435b6cf895ef9d44170215c5af856aff9195c44b3c7555f8cd3657c435d',
}
TOTAL_BUDGET = {'D25-P00': 400, 'D25-P01': 440}
REQUIRED = {'A': 3, 'B': 12, 'both': 15}
ACTIONS = ('forward', 'left', 'right')
NEAR_M = 2.0
ACQUISITION_CONTRACT = 'two_views_each_attachment_16px_2m_separation_0p4m_bearing_15deg'
DISK_RESERVE = 64*1024**2
OUTPUT_CAP = 2*1024**2
FAILURE_RESERVE = 32*1024


class CostLimitReached(BaseException):
    pass


def interrupt(number, frame):
    raise CostLimitReached('signal '+str(number))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def capacity(output, amount=0, emergency=False):
    ancestor = output
    while not ancestor.exists():
        ancestor = ancestor.parent
    require(shutil.disk_usage(ancestor).free-amount >= DISK_RESERVE,
            '64 MiB actual free-space reserve would be crossed')
    used = sum(p.stat().st_size for p in output.rglob('*') if p.is_file()) if output.exists() else 0
    limit = OUTPUT_CAP if emergency else OUTPUT_CAP-FAILURE_RESERVE
    require(used+amount+4096 <= limit, '2 MiB cost output cap would be crossed')


def write(output, name, value, emergency=False):
    data = (json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False)+'\n').encode()
    capacity(output, len(data), emergency)
    destination = output/name
    temporary = destination.with_name(destination.name+'.tmp')
    with temporary.open('xb') as stream:
        stream.write(data)
    os.replace(temporary, destination)


def encode(state, shape):
    r, c, h = map(int, state)
    return (r*shape[1]+c)*4+h


def decode(index, shape):
    cell, h = divmod(int(index), 4)
    r, c = divmod(cell, shape[1])
    return [r, c, h]


def make_graph(safe):
    """Every safe cell has all four headings and both legal in-place turns."""
    nr, nc = safe.shape
    neighbors = np.full((nr*nc*4, 3), -1, np.int32)
    for r, c in np.argwhere(safe):
        r, c = int(r), int(c)
        for h in range(4):
            i = encode((r, c, h), safe.shape)
            dr, dc = DIRECTIONS[h]
            rr, cc = r+dr, c+dc
            if 0 <= rr < nr and 0 <= cc < nc and safe[rr, cc]:
                neighbors[i, 0] = encode((rr, cc, h), safe.shape)
            neighbors[i, 1] = encode((r, c, (h-1) % 4), safe.shape)
            neighbors[i, 2] = encode((r, c, (h+1) % 4), safe.shape)
    incoming = np.full_like(neighbors, -1)
    for action in range(3):
        origins = np.flatnonzero(neighbors[:, action] >= 0)
        incoming[neighbors[origins, action], action] = origins
    return dict(safe=safe.copy(), shape=safe.shape, neighbors=neighbors,
                incoming=incoming, safe_pose_count=int(safe.sum())*4)


def validate_route(states, actions, graph):
    require(len(states) == len(actions)+1, 'Route state/action count mismatch')
    for state in states:
        require(len(state) == 3 and state[2] in range(4), 'Invalid heading')
        require(graph['safe'][tuple(state[:2])], 'Route crosses unsafe cell')
    for before, action, after in zip(states, actions, states[1:]):
        require(action in ACTIONS, 'Undeclared action')
        target = graph['neighbors'][encode(before, graph['shape']), ACTIONS.index(action)]
        require(target == encode(after, graph['shape']), 'Invalid route transition')


def shortest_path(graph, start, goal):
    n = len(graph['neighbors'])
    origin, target = encode(start, graph['shape']), encode(goal, graph['shape'])
    previous = np.full(n, -1, np.int32)
    action = np.full(n, -1, np.int8)
    previous[origin] = origin
    queue = deque([origin])
    while queue:
        current = queue.popleft()
        if current == target:
            nodes, actions = [current], []
            while current != origin:
                actions.append(ACTIONS[int(action[current])])
                current = int(previous[current]); nodes.append(current)
            states = [decode(i, graph['shape']) for i in reversed(nodes)]
            actions.reverse(); validate_route(states, actions, graph)
            return dict(states=states, actions=actions, cost=len(actions))
        for ai, following in enumerate(graph['neighbors'][current]):
            following = int(following)
            if following >= 0 and previous[following] < 0:
                previous[following] = current; action[following] = ai
                queue.append(following)
    return None


def near_frustum_possible(world, state, box):
    """Conservative exclusion only; all remaining poses use real union rays."""
    pose = camera_pose(state[:2], state[2], world.config, world.shape[0])
    corners = np.asarray([[box[0]+i*box[3], box[1]+j*box[4], box[2]+k*box[5]]
                          for i in (0, 1) for j in (0, 1) for k in (0, 1)])
    camera = (corners-pose[:3, 3])@pose[:3, :3]
    z = camera[:, 2]
    if z.max() <= .15 or z.min() > NEAR_M:
        return False
    if z.min() <= 0:
        return True
    uvw = camera@world.intrinsic.T
    uv = uvw[:, :2]/uvw[:, 2, None]
    return not (uv[:, 0].max() < 0 or uv[:, 0].min() > world.config.width_px-1
                or uv[:, 1].max() < 0 or uv[:, 1].min() > world.config.height_px-1)


def service_masks(worlds, common_safe):
    """Shared evaluator-side inspection regions from each complex counterpart.

    A bit needs a real nearer first hit that disappears when its one attachment
    is removed. The simple counterpart does not receive fictional attachment
    observations or quality credit. These masks never enter an online policy.
    """
    masks = np.zeros(common_safe.size*4, np.uint8)
    evidence, audits = [], []
    for asset in (0, 1):
        world = next(w for w in worlds if w.objects[asset]['category'] == 3)
        primitives = np.asarray(world._solid_primitives, float)
        indices = np.flatnonzero(primitives[:, 6] == world.objects[asset]['owner']).tolist()[1:]
        require(len(indices) == 2, 'Two unchanged external attachments required')
        reduced = [np.delete(primitives[:, :6], i, axis=0) for i in indices]
        counts = dict(asset_id=asset, complex_counterpart=world.assignment,
                      all_common_safe_poses=int(common_safe.sum())*4,
                      frustum_excluded_poses=0, rendered_full_poses=0,
                      rendered_single_attachment_removals=0, qualifying_poses=0,
                      attachment_qualifying_pose_counts=[0, 0])
        for r, c in np.argwhere(common_safe):
            for heading in range(4):
                state = [int(r), int(c), heading]
                possible = [near_frustum_possible(world, state, primitives[i, :6]) for i in indices]
                if not any(possible):
                    counts['frustum_excluded_poses'] += 1
                    continue
                depth, _ = projected(world, state)
                counts['rendered_full_poses'] += 1
                rows = []
                for local, primitive_id in enumerate(indices):
                    count, interval = 0, None
                    if possible[local]:
                        absent, _ = projected(world, state, reduced[local])
                        counts['rendered_single_attachment_removals'] += 1
                        own = ((depth > .15) & (depth <= NEAR_M)
                               & ((absent == 0) | (depth < absent)))
                        count = int(own.sum())
                        if count:
                            interval = [float(depth[own].min()), float(depth[own].max())]
                            masks[encode(state, common_safe.shape)] |= 1 << (2*asset+local)
                            counts['attachment_qualifying_pose_counts'][local] += 1
                    rows.append(dict(attachment_index=local, primitive_id=primitive_id,
                                     true_unique_near_first_hit_pixels=count, axial_depth_range_m=interval))
                if any(row['true_unique_near_first_hit_pixels'] for row in rows):
                    counts['qualifying_poses'] += 1
                    evidence.append(dict(asset_id=asset, state=state, attachments=rows))
        audits.append(counts)
        print('visibility', world.parent, 'asset', asset, counts['qualifying_poses'], flush=True)
    return masks, evidence, audits


def mask_union(states, masks, shape):
    answer = 0
    for state in states:
        answer |= int(masks[encode(state, shape)])
    return answer


class ServiceReturnSolver:
    """Reverse unit BFS on (pose, acquired bits), with an exact heading goal.

    Costs include every move/turn, permit all routes, credit every paid arrival
    frame, and may finish service while travelling toward home. No dwell or
    long-option commitment is required. The graph itself remains GT-only.
    """
    def __init__(self, graph, masks, required, anchor):
        self.graph, self.global_masks, self.required = graph, masks, required
        bits = [i for i in range(4) if required & (1 << i)]
        self.lookup = np.asarray([sum(bool(m & (1 << bit)) << j for j, bit in enumerate(bits))
                                  for m in range(16)], np.uint8)
        self.observation = self.lookup[masks]
        self.n = len(graph['neighbors'])
        self.full = (1 << len(bits))-1
        self.goal = self.full*self.n+encode(anchor, graph['shape'])
        size = (self.full+1)*self.n
        self.distance = np.full(size, -1, np.int32)
        self.following = np.full(size, -1, np.int32)
        self.action = np.full(size, -1, np.int8)
        self.distance[self.goal] = 0
        queue = deque([self.goal]); visited = 0
        while queue:
            current = queue.popleft(); visited += 1
            mask, pose = divmod(current, self.n)
            observed = int(self.observation[pose])
            if observed & ~mask:
                continue
            base, optional = mask & ~observed, mask & observed
            subset = optional
            while True:
                previous_mask = base | subset
                for ai, predecessor in enumerate(graph['incoming'][pose]):
                    predecessor = int(predecessor)
                    if predecessor < 0 or int(self.observation[predecessor]) & ~previous_mask:
                        continue
                    index = previous_mask*self.n+predecessor
                    if self.distance[index] < 0:
                        self.distance[index] = self.distance[current]+1
                        self.following[index] = current
                        self.action[index] = ai
                        queue.append(index)
                if subset == 0:
                    break
                subset = (subset-1) & optional
        self.visited_states = visited

    def route(self, start, initial_mask):
        pose = encode(start, self.graph['shape'])
        mask = int(self.lookup[initial_mask])
        require(not int(self.observation[pose]) & ~mask,
                'Initial mask must include the already-paid current observation')
        current = mask*self.n+pose
        if self.distance[current] < 0:
            return None
        expected = int(self.distance[current]); states = [list(start)]; actions = []
        acquired = int(initial_mask)
        while current != self.goal:
            following = int(self.following[current]); ai = int(self.action[current])
            require(following >= 0 and ai >= 0, 'Broken reverse-BFS route')
            _, next_pose = divmod(following, self.n)
            state = decode(next_pose, self.graph['shape'])
            actions.append(ACTIONS[ai]); states.append(state)
            acquired |= int(self.global_masks[next_pose]); current = following
        require(len(actions) == expected and (acquired & self.required) == self.required,
                'Service route cost or paid observation union inconsistent')
        validate_route(states, actions, self.graph)
        return dict(states=states, actions=actions, cost=expected,
                    final_template_service_mask=acquired, reverse_bfs_visited_states=self.visited_states)


def fixed_option(route, role, prefix, graph, masks, initial_mask, total_budget):
    base = dict(reachable=route is not None, required_service_mask=REQUIRED[role],
                GT_service_contract_only=True, scripted_development_arm=False,
                cost_bound_only=True, not_for_collection=True,
                unknown_map_runtime=False, actual_execution=False,
                semantic_policy_or_quality_claim=False, total_budget=total_budget,
                paid_prefix_actions=len(prefix['actions']), budget_fits=False,
                prefix_actions=prefix['actions'], prefix_states=prefix['states'])
    if route is None:
        base.update(reason='Service/return route unavailable under the frozen contract',
                    continuation_actions=None, continuation_states=None,
                    return_actions=None, return_states=None, actions=None, states=None,
                    frame_schedule=None, paid_total=None, supplemental_paid_actions=None,
                    first_service_complete_local_action=None)
        return base
    accumulated, complete = int(initial_mask), None
    receipts = []
    for local, state in enumerate(route['states']):
        if local:
            previous = accumulated; accumulated |= int(masks[encode(state, graph['shape'])])
            if accumulated != previous:
                receipts.append(dict(local_action=local, frame_index=len(prefix['actions'])+local,
                                     state=state, newly_acquired_template_bits=accumulated ^ previous,
                                     cumulative_template_mask=accumulated))
        if complete is None and accumulated & REQUIRED[role] == REQUIRED[role]:
            complete = local
    require(complete is not None, 'Completed route lacks an attaining paid service frame')
    continuation = route['actions'][:complete]; returns = route['actions'][complete:]
    actions = prefix['actions']+route['actions']
    states = prefix['states']+route['states'][1:]
    require(prefix['states'][-1] == route['states'][0] == route['states'][-1], 'Anchor mismatch')
    validate_route(states, actions, graph)
    frames = [dict(frame_index=i+1, phase=('prefix' if i < len(prefix['actions']) else
              'continuation' if i < len(prefix['actions'])+complete else 'return'), action=action)
              for i, action in enumerate(actions)]
    base.update(continuation_actions=continuation, continuation_states=route['states'][:complete+1],
                return_actions=returns, return_states=route['states'][complete:], actions=actions, states=states,
                frame_schedule=frames, paid_total=len(actions), supplemental_paid_actions=route['cost'],
                first_service_complete_local_action=complete, budget_fits=len(actions) <= total_budget,
                budget_overrun_actions=max(0, len(actions)-total_budget),
                service_receipts=receipts, final_template_service_mask=accumulated,
                actions_sha256=hashlib.sha256(json.dumps(actions, separators=(',', ':')).encode()).hexdigest(),
                pair_uses_exactly_same_actions_and_states=True, no_truncation=True)
    return base


def acquisition_catalog(worlds, evidence, prefix, common, total_budget):
    """Freeze first valid lexicographic pair before considering route cost.

    This uses the same visibility scan, with a separate multi-view acquisition
    specification. Budget/Q cannot change the pair or thresholds. These exact
    common-safe action strings are only scripted development collection arms.
    """
    catalog = {}
    for asset, role in enumerate(('A', 'B')):
        center = worlds[0].objects[asset]['center'][:2]
        require(center == worlds[1].objects[asset]['center'][:2], 'Shared body centers differ')
        candidates = sorted((row for row in evidence if row['asset_id'] == asset
            and all(a['true_unique_near_first_hit_pixels'] >= 16 for a in row['attachments'])),
            key=lambda row: tuple(row['state']))
        selected = None
        for i, first in enumerate(candidates):
            for second in candidates[i+1:]:
                xy = [((row['state'][1]+.5)*worlds[0].config.resolution_m,
                       (common['shape'][0]-row['state'][0]-.5)*worlds[0].config.resolution_m)
                      for row in (first, second)]
                separation = math.dist(*xy)
                bearings = [math.degrees(math.atan2(y-center[1], x-center[0])) for x, y in xy]
                difference = abs((bearings[1]-bearings[0]+180) % 360-180)
                if separation >= .4-1e-9 and difference >= 15.-1e-9:
                    selected = dict(views=[first, second], xy_m=[list(p) for p in xy],
                        separation_m=separation, horizontal_bearings_deg=bearings,
                        horizontal_bearing_difference_deg=difference)
                    break
            if selected is not None:
                break
        metadata = dict(acquisition_contract=ACQUISITION_CONTRACT,
            minimum_near_pixels_per_attachment_per_view=16, maximum_axial_depth_m=NEAR_M,
            minimum_view_position_separation_m=.4, minimum_horizontal_bearing_difference_deg=15.,
            qualifying_view_count=len(candidates), shared_body_center_xy_m=center,
            pair_selection='first lexicographic (row,col,heading) pair meeting all fixed conditions',
            pair_changed_for_budget_or_Q=False, selected_pair=selected,
            visibility_from_complex_counterpart_only=True,
            simple_counterpart_is_region_observation_not_fictional_attachment_credit=True,
            shape_quality_or_completion_guaranteed=False)
        if selected is None:
            option = fixed_option(None, role, prefix, common,
                                  np.zeros(len(common['neighbors']), np.uint8), 0, total_budget)
            option['reason'] = 'No pair satisfies the fixed multi-view acquisition contract'
            option.update(metadata, cost_bound_only=False, not_for_collection=False,
                          scripted_development_arm=True, acquisition_pair_available=False,
                          selected_view_states=None, arrival_frame_indices=None)
            catalog[role] = option
            continue
        states = [row['state'] for row in selected['views']]
        anchor = prefix['decision_anchor']; alternatives = []
        for order in ((0, 1), (1, 0)):
            goals = [anchor, states[order[0]], states[order[1]], anchor]
            legs = [shortest_path(common, a, b) for a, b in zip(goals, goals[1:])]
            if any(leg is None for leg in legs):
                continue
            route_states, route_actions = [anchor], []
            for leg in legs:
                route_states += leg['states'][1:]; route_actions += leg['actions']
            validate_route(route_states, route_actions, common)
            alternatives.append(dict(states=route_states, actions=route_actions, cost=len(route_actions),
                                     planned_view_order=list(order), leg_costs=[leg['cost'] for leg in legs]))
        alternatives.sort(key=lambda route: (route['cost'],
            tuple(states[route['planned_view_order'][0]]), tuple(states[route['planned_view_order'][1]])))
        route = alternatives[0] if alternatives else None
        view_masks = np.zeros(len(common['neighbors']), np.uint8)
        for view_index, state in enumerate(states):
            view_masks[encode(state, common['shape'])] = 1 << (asset*2+view_index)
        # Acquisition intentionally pays again for each selected view after the
        # common prefix. Neither the initial frame nor prior photos are reused
        # as a free supplemental observation.
        require(int(view_masks[encode(anchor, common['shape'])]) == 0, 'Acquisition view unexpectedly at anchor')
        option = fixed_option(route, role, prefix, common, view_masks, 0, total_budget)
        option.update(metadata, cost_bound_only=False, not_for_collection=False,
                      scripted_development_arm=True, acquisition_pair_available=True,
                      selected_view_states=states,
                      planned_view_order=None if route is None else route['planned_view_order'],
                      planned_leg_costs=None if route is None else route['leg_costs'],
                      compared_visit_order_costs=[dict(order=a['planned_view_order'], cost=a['cost'])
                                                 for a in alternatives])
        if route is not None:
            first_arrivals = [next(i for i, q in enumerate(route['states']) if i > 0 and q == state)
                              for state in states]
            option['arrival_frame_indices'] = [len(prefix['actions'])+i for i in first_arrivals]
            option['actual_first_arrival_view_order'] = sorted(range(2), key=lambda i: first_arrivals[i])
            option['view_receipts'] = option.pop('service_receipts')
            option['required_acquisition_view_mask'] = option.pop('required_service_mask')
            option['final_acquisition_view_mask'] = option.pop('final_template_service_mask')
            option['each_selected_view_has_a_distinct_paid_frame'] = len(set(first_arrivals)) == 2
        else:
            option['arrival_frame_indices'] = None
        catalog[role] = option
    return catalog


def costs_for_graph(graph, masks, anchor, initial_mask, prefix_count, budget):
    solvers, records = {}, {}
    for role, required in REQUIRED.items():
        solver = ServiceReturnSolver(graph, masks, required, anchor)
        route = solver.route(anchor, initial_mask)
        solvers[role] = solver
        records[role] = dict(reachable=route is not None,
                            supplemental_cost=None if route is None else route['cost'],
                            total_paid_cost=None if route is None else prefix_count+route['cost'],
                            budget_fits=False if route is None else prefix_count+route['cost'] <= budget,
                            supplemental_actions=None if route is None else route['actions'],
                            supplemental_states=None if route is None else route['states'],
                            safe_pose_count=graph['safe_pose_count'],
                            reverse_bfs_visited_states=solver.visited_states,
                            optimal_for_declared_service_and_given_GT_graph=True,
                            global_reconstruction_cost_lower_bound_claim=False)
    return solvers, records


def recovery_costs(worlds, common_graph, masks, anchor, initial_mask, prefix_count,
                   budget, prior_parent, solvers, direct):
    contexts = []
    for witness in prior_parent['information']['witnessed_difference']['witnesses']:
        approach = shortest_path(common_graph, anchor, witness['state'])
        require(approach is not None, 'Saved common-safe first-difference witness unreachable')
        differences = []
        for local, state in enumerate(approach['states']):
            images = [projected(w, state)[0] for w in worlds]
            changed = int(np.count_nonzero(images[0] != images[1]))
            if changed:
                differences.append(dict(local_action=local, state=state, differing_pixels=changed))
        require(differences, 'Saved depth-difference witness no longer differs')
        first = differences[0]
        # Stop at the first real disclosure on this approach; never force a
        # geometry method to continue an option after information has arrived.
        k = first['local_action']; information_pose = first['state']
        info_states, info_actions = approach['states'][:k+1], approach['actions'][:k]
        require(k > 0, 'The declared initial anchor already reveals shape')
        acquired = initial_mask | mask_union(info_states[1:], masks, common_graph['shape'])
        cases = []
        for wi, world in enumerate(worlds):
            for role in ('A', 'B'):
                suffix = solvers[wi][role].route(information_pose, acquired)
                baseline = direct[wi][role]['supplemental_cost']
                whole = None if suffix is None else k+suffix['cost']
                if suffix is not None:
                    states = info_states+suffix['states'][1:]
                    actions = info_actions+suffix['actions']
                    validate_route(states, actions, solvers[wi][role].graph)
                    require(states[0] == states[-1] == anchor, 'Recovery must return exact heading')
                    require(baseline is not None and whole >= baseline, 'Conditional route beats its GT optimum')
                else:
                    states = actions = None
                cases.append(dict(assignment=world.assignment, target_role=role,
                    target_is_complex=world.objects['AB'.index(role)]['category'] == 3,
                    reachable=suffix is not None, information_prefix_paid=k,
                    post_information_service_and_return_cost=None if suffix is None else suffix['cost'],
                    whole_supplemental_cost=whole, total_paid_cost=None if whole is None else prefix_count+whole,
                    direct_actual_GT_cost=baseline, extra_vs_direct=None if whole is None else whole-baseline,
                    budget_fits=False if whole is None else prefix_count+whole <= budget,
                    supplemental_actions=actions, supplemental_states=states,
                    information_is_ideal_depth_only=True, perfect_identification_assumed=True,
                    exact_executable_unknown_map_policy_cost=False))
        contexts.append(dict(requested_saved_witness=witness['state'],
            requested_saved_optimistic_distance=witness['optimistic_distance'],
            common_safe_approach_cost=approach['cost'], first_difference=first,
            prefix_before_first_difference_clean_paired=True,
            information_prefix_actions=info_actions, information_prefix_states=info_states,
            cases=cases, conditional_on_this_approach_only=True,
            all_later_information_strategies_optimized=False))
    return contexts


def check_sources(protected, sources):
    for path, expected in {**protected, **sources}.items():
        require(sha(ROOT/path) == expected, 'Frozen source changed: '+path)
    for name, expected in INPUT_SHA.items():
        require(sha(R1/name) == expected, 'Frozen r1 input changed: '+name)


def execute(output, progress):
    for name, expected in INPUT_SHA.items():
        require(sha(R1/name) == expected, 'Frozen r1 input mismatch: '+name)
    for name, expected in read(R1/'artifact_hashes.json').items():
        require(sha(R1/name) == expected, 'Frozen r1 artifact mismatch: '+name)
    prior_manifest, prior = read(R1/'manifest.json'), read(R1/'result.json')
    require(prior_manifest['status'] == 'complete' and prior['finite_static_contract_passed'],
            'A complete passing r1 static source is required')
    protected = {**prior_manifest['protected_source_sha256'], **prior_manifest['source_sha256']}
    require(len(protected) == 153, 'Expected old150 plus r1 three frozen sources')
    paths = [Path(__file__).resolve(), PROTOCOL]
    sources = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    check_sources(protected, sources)
    capacity(output, OUTPUT_CAP-65536)
    output.mkdir(parents=True); progress['created_output'] = True
    write(output, 'manifest.json', dict(status='running', source_sha256=sources,
        protected_source_sha256=protected, input_r1_sha256=INPUT_SHA, world_version=VERSION_V25,
        total_budget=TOTAL_BUDGET, near_axial_depth_m=NEAR_M, minimum_unique_pixels_per_attachment=1,
        actual_acquisition_contract=ACQUISITION_CONTRACT,
        maximum_seconds=600, output_cap_bytes=OUTPUT_CAP, disk_reserve_bytes=DISK_RESERVE,
        repair_count=1, new_geometry_changes=0, fixed_options_are_scripted_development_only=True))
    capacity(output, sum(p.stat().st_size for p in paths)+8192)
    with zipfile.ZipFile(output/'sources.zip', 'x', zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            archive.write(path, str(path.relative_to(ROOT)))
    parents = []
    for parent in PARENTS_V25:
        progress.update(parent=parent, phase='construct_unchanged_r1')
        worlds = [StaticWorld(parent, assignment) for assignment in ASSIGNMENTS_V25]
        prefix = worlds[0].prefix_proposal; anchor = prefix['decision_anchor']
        require(prefix['states'] == worlds[1].prefix_proposal['states'], 'Paired prefixes differ')
        expected_prefix = 234 if parent == 'D25-P00' else 254
        require(len(prefix['actions']) == expected_prefix, 'Declared paid prefix length changed')
        common_safe = worlds[0].reachable & worlds[1].reachable
        common = make_graph(common_safe)
        validate_route(prefix['states'], prefix['actions'], common)
        progress['phase'] = 'near_attachment_visibility'
        masks, evidence, visibility_audit = service_masks(worlds, common_safe)
        initial_mask = mask_union(prefix['states'][1:], masks, common_safe.shape)
        prefix_current = int(masks[encode(anchor, common_safe.shape)])
        require((initial_mask | prefix_current) == initial_mask, 'Final paid prefix frame absent from ledger')
        progress['phase'] = 'common_safe_fixed_options'
        common_solvers, common_costs = costs_for_graph(common, masks, anchor, initial_mask,
                                                      expected_prefix, TOTAL_BUDGET[parent])
        options = {role: fixed_option(common_solvers[role].route(anchor, initial_mask), role, prefix,
                     common, masks, initial_mask, TOTAL_BUDGET[parent]) for role in REQUIRED}
        del common_solvers
        progress['phase'] = 'fixed_multiview_acquisition_catalog'
        catalog = acquisition_catalog(worlds, evidence, prefix, common, TOTAL_BUDGET[parent])
        progress['phase'] = 'actual_GT_graph_optimistic_costs'
        actual_solvers, actual_costs = [], []
        for world in worlds:
            graph = make_graph(world.reachable)
            ss, records = costs_for_graph(graph, masks, anchor, initial_mask,
                                         expected_prefix, TOTAL_BUDGET[parent])
            actual_solvers.append(ss); actual_costs.append(records)
            for role in ('A', 'B'):
                if catalog[role]['reachable']:
                    bound = records[role]['supplemental_cost']
                    require(bound is not None and bound <= catalog[role]['supplemental_paid_actions'],
                            'Optimistic one-pixel contract exceeds its two-view acquisition witness')
        progress['phase'] = 'conditional_first_difference_recovery'
        prior_parent = next(p for p in prior['parents'] if p['parent'] == parent)
        recovery = recovery_costs(worlds, common, masks, anchor, initial_mask,
            expected_prefix, TOTAL_BUDGET[parent], prior_parent, actual_solvers, actual_costs)
        require(all(w.step_count == 0 and w.collisions == 0 for w in worlds), 'Unexpected actual world step')
        entry = dict(parent=parent, total_budget=TOTAL_BUDGET[parent], paid_prefix_actions=expected_prefix,
            remaining_budget=TOTAL_BUDGET[parent]-expected_prefix, anchor=anchor,
            common_safe_cells=int(common_safe.sum()), common_safe_poses=common['safe_pose_count'],
            common_prefix_template_mask=initial_mask, visibility_audit=visibility_audit,
            service_pose_evidence=evidence, service_mask_nonzero_pose_count=int(np.count_nonzero(masks)),
            service_mask_sha256=hashlib.sha256(masks.tobytes()).hexdigest(),
            common_safe_options=options, common_safe_optimal_costs=common_costs, route_catalog=catalog,
            actual_GT_optimistic_costs=[dict(assignment=w.assignment, options=records)
                                      for w, records in zip(worlds, actual_costs)],
            conditional_recovery=recovery, unknown_map_runtime_cost=None,
            actual_scan_pairing_or_collision_feedback_verified=False,
            service_contract_is_not_Q_or_reconstruction_completion=True)
        parents.append(entry)
        write(output, 'partial_result.json', dict(status='partial', parents=parents))
        print(parent, 'common fixed option totals',
              {role: row['paid_total'] for role, row in options.items()}, flush=True)
        print(parent, 'actual acquisition catalog totals',
              {role: row['paid_total'] for role, row in catalog.items()}, flush=True)
        del worlds, actual_solvers, masks, common
    check_sources(protected, sources)
    result = dict(status='complete_static_cost_only', world_version=VERSION_V25, parents=parents,
        primary_service_contract=dict(maximum_axial_depth_m=NEAR_M, minimum_unique_pixels_per_attachment=1,
            two_attachment_bits_per_asset=True, paid_frame_union=True, forced_dwell=False,
            cross_assignment_common_safe_service_poses=True, actual_GT_graphs_may_have_more_transit_cells=True,
            visible_regions_defined_by_complex_counterpart=True, simple_asset_receives_no_fictional_Q=True),
        total_budget_fixed_before_costs_and_Q=TOTAL_BUDGET, extra_budget_margin_actions=0,
        actual_acquisition_contract=ACQUISITION_CONTRACT,
        collector_must_use_route_catalog_only=True,
        new_physical_actions=0, new_sensor_packets=0, new_tsdf_fusions=0, quality_scores_computed=0,
        new_geometry_changes=0, repair_count=1, second_repair_authorized=False,
        semantic_efficacy_proven=False, unknown_map_policy_evaluated=False,
        retrospective_costs_are_not_permission_to_execute_all_options=True,
        first_actual_batch_scope='P00 A/B x two assignments only, separately frozen collector; not run here')
    write(output, 'result.json', result)
    manifest = read(output/'manifest.json'); manifest['status'] = 'complete'
    write(output, 'manifest.json', manifest)
    print(json.dumps(dict(status=result['status'], output=str(output))), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    if not args.run:
        parser.error('Explicit separately authorized --run required')
    output = args.output.resolve()
    require(not output.exists(), 'Existing cost evidence must not be replaced')
    progress = dict(created_output=False, phase='verify_sources')
    started, code = perf_counter(), 0
    for number in (signal.SIGINT, signal.SIGTERM, signal.SIGALRM):
        signal.signal(number, interrupt)
    signal.setitimer(signal.ITIMER_REAL, 600)
    try:
        execute(output, progress)
    except BaseException as error:
        signal.setitimer(signal.ITIMER_REAL, 0)
        code = 124 if isinstance(error, CostLimitReached) else 1
        failure = dict(status='bounded_incomplete' if code == 124 else 'failed', error=repr(error),
            traceback=traceback.format_exc(), progress=progress, process_id=os.getpid(), elapsed_s=perf_counter()-started)
        print(json.dumps(failure, ensure_ascii=False), file=sys.stderr)
        if progress['created_output']:
            try:
                temporary = output/'manifest.json.tmp'
                if temporary.exists():
                    temporary.replace(output/'interrupted_manifest.json.tmp')
                write(output, 'failure.json', failure, emergency=True)
                value = read(output/'manifest.json'); value['status'] = failure['status']
                write(output, 'manifest.json', value, emergency=True)
            except Exception as receipt_error:
                print('Failure receipt error: '+repr(receipt_error), file=sys.stderr)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        if progress['created_output']:
            try:
                write(output, 'timing.json', dict(process_id=os.getpid(), elapsed_s=perf_counter()-started,
                      exit_code=code), emergency=code != 0)
                inventory = {str(p.relative_to(output)): sha(p) for p in sorted(output.rglob('*'))
                             if p.is_file() and p.name != 'artifact_hashes.json'}
                write(output, 'artifact_hashes.json', inventory, emergency=code != 0)
            except Exception as final_error:
                print('Final receipt error: '+repr(final_error), file=sys.stderr)
                code = code or 1
    raise SystemExit(code)


if __name__ == '__main__':
    main()
