"""No-actor scene validation and pre-outcome preparation."""

import argparse
from collections import Counter, deque
import dataclasses
from hashlib import sha256
import json
import os
from pathlib import Path
import time

for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '1'

import numpy as np
import trimesh

from mechanism_world import ANALYSES, BUDGET, HERE, LAYOUTS, PARENT, SEED, SPECS, api, build_scene, cell, make_world
from nso.observed_runtime_mapper_v10 import ObservedRuntimeMapperV10


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def array_digest(values):
    a = np.ascontiguousarray(values)
    hash_ = sha256(str((a.shape, a.dtype)).encode())
    hash_.update(a)
    return hash_.hexdigest()


def write(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')


def declared_route(q, side):
    to_east = 'right' if side == 'right' else 'left'
    to_west = 'left' if side == 'right' else 'right'
    def forward(delta):
        return ['forward'] * round(delta/.2)
    return (forward(q['split_y']-q['start_y']) + [to_east] + forward(q['arm_offset']) + [to_west]
        + forward(q['rejoin_y']-q['split_y']) + [to_west] + forward(q['arm_offset']) + [to_east]
        + forward(q['destination_y']-q['rejoin_y']))


def distances(edges, start):
    queue, costs = deque([start]), {start: 0}
    while queue:
        row, col, h = pose = queue.popleft()
        options = [(row, col, (h-1) % 4), (row, col, (h+1) % 4)]
        if edges[row, col, h]:
            dr, dc = api.old.DIRECTIONS[h]
            options.append((row+dr, col+dc, h))
        for target in options:
            if target not in costs:
                costs[target] = costs[pose] + 1
                queue.append(target)
    return costs


def classify_gate(q, states):
    """Evaluation only. q is a source layout, not an online policy input."""
    row = cell(q['axis_x'], q['route_evaluation']['north_gate_y_m'], q['height_m'])[0]
    for before, after in zip(states, states[1:]):
        if before[0] > row >= after[0]:
            x = (after[1]+.5)*.2
            if x <= q['route_evaluation']['left_limit_x_m']:
                return 'left'
            if x >= q['route_evaluation']['right_limit_x_m']:
                return 'right'
            return None
    return None


def visible_parts(world, frame):
    c = world.config
    view = world._renderer.render(world.intrinsic, frame.world_from_camera,
        c.width_px, c.height_px, max_depth_m=c.max_depth_m, min_depth_m=.15)
    valid = view['depth_m'] > 0
    ids, count = np.unique(view['reference_geometry_id'][valid], return_counts=True)
    return {world._node_for_gid[int(gid)]: int(n) for gid, n in zip(ids,count)
            if int(gid) in world._node_for_gid}


def run_path(world, actions):
    world.position, world.heading = world.start, world.spec['heading']
    world.step_count = world.collisions = world.moves = 0
    states, visibility, merges = [(*world.position, world.heading)], Counter(), []
    for i, action in enumerate(actions,1):
        frame, collision, done = world.step(action)
        assert not collision and not done, (world.layout, i, action)
        state = (*world.position, world.heading)
        states.append(state)
        if world.xy(world.position)[1] + 1e-9 < world.spec['rejoin_y']:
            visibility.update(visible_parts(world, frame))
        else:
            merges.append((i, state))
    return states, visibility, merges[0]


def prepare(layout):
    start = time.monotonic()
    q = SPECS[layout]
    world = make_world(layout)
    assert world.config.max_steps == BUDGET
    truth = world.evaluator_truth()
    frame, scan = world.sense(), world.scan()
    parts_at_start = visible_parts(world,frame)
    owner = world.sense_visible_instances()
    assert np.array_equal(frame.semantic > 0, owner > 0)
    masks = {side: owner == iid for iid, side in ((1, 'left'),(2, 'right'))}
    parts_by_slot = {side: sorted(n.split('/')[1] for n in parts_at_start if n.startswith(side+'/'))
                    for side in ('left', 'right')}
    for side, mask in masks.items():
        assert mask.sum() >= 16, (layout, side, mask.sum())
        assert parts_by_slot[side], (layout, side)
        assert set(parts_by_slot[side]) <= {'plinth','south_side','back','top','north_side'}, (layout,parts_by_slot)
    mapper = ObservedRuntimeMapperV10(world.shape, world.config)
    mapper.update(frame, scan)
    destination_rc = cell(q['axis_x'],q['destination_y'],q['height_m'])
    assert truth['reachable'][destination_rc] and mapper.belief[destination_rc] == -1, (layout,destination_rc)
    no_semantic = dict(depth=array_digest(frame.depth_m), rgb=array_digest(frame.color_rgb),
        support=array_digest(owner), pose=array_digest(frame.world_from_camera),
        lidar=array_digest(scan.ranges_m), pose_lidar=array_digest(scan.world_from_laser),
        observed_belief=array_digest(mapper.belief))

    out = HERE / 'prepared' / layout
    out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/'evaluator_truth.npz',**truth)
    frame.save(out/'initial_frame.npz')
    scan.save(out/'initial_scan.npz')
    np.savez_compressed(out/'initial_instances.npz',visible_instances=owner)
    write(out/'initial_state.json',dict(shape=list(world.shape),anchor=[*world.start, q['heading']],
          actual_config=dataclasses.asdict(world.config), layout=layout, seed=SEED))

    scene, records, metadata = build_scene(layout)
    areas = Counter()
    for node in scene.graph.nodes_geometry:
        transform,name = scene.graph[node]
        part = scene.geometry[name].copy()
        part.apply_transform(transform)
        areas[str(metadata[node]['semantic_class'])] += float(part.area)
    baseline_costs = distances(truth['edge_valid'],(*world.start,q['heading']))
    runs = {}
    gate_rc = cell(q['axis_x'], q['lane_test_y'], q['height_m'])
    for side in ('left','right'):
        actions = declared_route(q, side)
        states, all_hits, first_merge = run_path(world,actions)
        assert states[-1] == (*destination_rc,0), (layout,side,states[-1])
        assert classify_gate(q, states) == side
        gate = next(p for p in states if p[0] == gate_rc[0] and p[2] == 0)
        true_return = distances(truth['edge_valid'],gate)[(*world.start,q['heading'])]
        internal = {n: v for n,v in sorted(all_hits.items()) if '/interior_' in n}
        prefix_only_to_gate = next(i for i,p in enumerate(states) if p == gate)
        runs[side] = dict(geometry_probe_only=True, gate_pose=list(gate),
            gate_outbound_actions=prefix_only_to_gate, gate_shortest_route_actions=baseline_costs[gate],
            gate_shortest_return_actions=true_return, shared_endpoint=list(states[-1]),
            actions=len(actions), states=[list(p) for p in states],
            first_merge_at_action=first_merge[0], planned_nominal_route=actions,
            open_interior_first_hit_pixels_before_rejoin=sum(internal.values()), interior_part_hits=internal)
    expected = q['route_evaluation']['expected_diagnostic_side']
    other = 'right' if expected == 'left' else 'left'
    assert runs[expected]['open_interior_first_hit_pixels_before_rejoin'] > 0, (layout,runs)
    assert runs[other]['open_interior_first_hit_pixels_before_rejoin'] == 0, (layout,runs)
    for key in ('gate_outbound_actions','gate_shortest_return_actions','gate_shortest_route_actions','actions','shared_endpoint'):
        assert runs['left'][key] == runs['right'][key], (layout,key)
    receipt = dict(status='geometry_and_sensor_preflight_passed',layout=layout,parent=q['parent'],
        seed=SEED, budget=BUDGET, geometry=world.geometry_receipt, source_objects=records,
        route_evaluation=q['route_evaluation'], routes=runs,
        initial_support_pixels={side:int(mask.sum()) for side,mask in masks.items()},
        initial_visible_part_names=parts_by_slot, initial_nonsemantic_digests=no_semantic,
        initial_semantic_area_pixels={str(i):int((frame.semantic==i).sum()) for i in (2,3)},
        initial_belief_cell_counts={str(i):int((mapper.belief==i).sum()) for i in (-1,0,1)},
        source_area_m2_by_class=dict(areas), total_source_area_m2=float(sum(areas.values())),
        collision_and_reachability_digests={k:array_digest(v) for k,v in truth.items()},
        shared_destination_initially_unknown=True, elapsed_s=time.monotonic()-start,
        scored_candidates_opened=False, actor_run=False)
    write(out/'receipt.json',receipt)
    return receipt


def pair_check():
    pair_records = {}
    for parent in ('warehouse','gallery'):
        paths = [HERE/'prepared'/f'{parent}_{side}' for side in ('left','right')]
        a,b = (json.loads((p/'receipt.json').read_text()) for p in paths)
        assert a['collision_and_reachability_digests'] == b['collision_and_reachability_digests'], (parent,'geometry')
        assert a['source_area_m2_by_class'].keys() == b['source_area_m2_by_class'].keys()
        for key in a['source_area_m2_by_class']:
            assert abs(a['source_area_m2_by_class'][key]-b['source_area_m2_by_class'][key]) < 1e-9, (parent,key)
        assert a['initial_nonsemantic_digests'] == b['initial_nonsemantic_digests'], (parent,'init sensors')
        fa,fb = (np.load(p/'initial_frame.npz', allow_pickle=False) for p in paths)
        semantic_a,semantic_b = (f['semantic'].copy() for f in (fa,fb))
        assert np.array_equal(np.where(semantic_a==2,3,np.where(semantic_a==3,2,0)),semantic_b), (parent,'label-only')
        pair_records[parent] = dict(physical_occupancy_and_swept_graph_match=True,
            initial_nonsemantic_sensor_bytes_and_belief_match=True, first_hit_labels_match_after_swap=True,
            source_area_identical=True, source_area_m2=a['total_source_area_m2'],
            initial_furniture_support_pixels=a['initial_support_pixels'],
            expected_diagnostic_branch_changes_with_swap=True, shared_unknown_destination=True,
            whole_grid_center_reachable_m2=.04*a['geometry']['reachable_cells'])
    write(HERE/'swap_preflight.json',dict(status='both_frozen_parent_pairs_validated',pairs=pair_records,
           actor_run=False, scores_opened=False))
    return pair_records


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--layout', choices=LAYOUTS)
    parser.add_argument('--check-pairs', action='store_true')
    arguments = parser.parse_args()
    if arguments.check_pairs:
        print(json.dumps(pair_check()))
    elif arguments.layout:
        a=prepare(arguments.layout)
        print(json.dumps({k:a[k] for k in ('layout','status','initial_support_pixels','elapsed_s')}))
    else:
        for layout in LAYOUTS:
            print(prepare(layout)['layout'],flush=True)
        print(json.dumps(pair_check()))
