#!/usr/bin/env python3
"""Audit logged Cycle16 camera poses and updated held-out spatial measurements, without running a simulator."""

from __future__ import annotations

import argparse
import ast
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
STUDIES = HERE.parent
RUN_STUDY = STUDIES / 'same_goal_cold_holdout_20261121'
SCENE = STUDIES / 'same_goal_route_choice_20261121/holdout'
WORLD_SOURCE = STUDIES / 'paired_unseen_gain_20261120/scene/unseen_world.py'
TRANSFORM_SOURCE = (STUDIES / 'reproduction_20261104/simulation_publication_stage/experiments/'
                    'unknown_exploration_20261104/source/nso/cpu_sensor_contract_v10.py')
LAYOUT = 'ordinary_cycle_16'
ARMS = ('fullG', 'fullS')
CLOCKS = (256, 320, 512)
WINDOWS = ((1, 256), (257, 320), (321, 512))
HEADINGS = ('N', 'E', 'S', 'W')
ROOMS = ('A_southwest', 'B_southeast', 'C_northeast', 'D_northwest')
PANELS = ('cycle16_paid256', 'cycle16_paid320', 'cycle16_terminal')
TOL_M = 1e-6


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def source_layout(layout):
    tree = ast.parse((SCENE / 'scene.py').read_text())
    scene_specs = next(node.value for node in tree.body if isinstance(node, ast.Assign)
                       and len(node.targets) == 1 and getattr(node.targets[0], 'id', None) == 'ORDINARY')
    entries = dict(zip(map(ast.literal_eval, scene_specs.keys), scene_specs.values, strict=True))
    spec = entries[layout]
    assert isinstance(spec, ast.Call) and isinstance(spec.func, ast.Name) and spec.func.id == 'dict'
    assert not spec.args
    return {kw.arg: ast.literal_eval(kw.value) for kw in spec.keywords}


def geometry():
    config = source_layout(LAYOUT)
    assert config['topology'] == 'cycle' and all(name in WORLD_SOURCE.read_text() for name in ROOMS)
    source_world_tree = ast.parse(WORLD_SOURCE.read_text())
    gap_function = next(node for node in source_world_tree.body if isinstance(node, ast.FunctionDef)
                        and node.name == 'gap_segments')
    gap = ast.literal_eval(gap_function.args.defaults[0])
    width, height = config['width_m'], config['height_m']
    sx, sy = config['split_x'], config['split_y']
    initial = read(SCENE / 'prepared' / LAYOUT / 'initial_state.json')
    resolution = initial['grid_resolution_m']
    assert initial['shape'] == [round(height / resolution), round(width / resolution)]
    assert 'origin_xy_m: tuple = (0., 0.)' in TRANSFORM_SOURCE.read_text()
    assert '(self.shape[0] - r - .5) * self.resolution_m' in TRANSFORM_SOURCE.read_text()
    midpoints = ((ROOMS[0], ROOMS[1], sx, config['door_bottom_y'], 'x'),
                 (ROOMS[1], ROOMS[2], config['door_right_x'], sy, 'y'),
                 (ROOMS[2], ROOMS[3], sx, config['door_top_y'], 'x'),
                 (ROOMS[3], ROOMS[0], config['door_left_x'], sy, 'y'))
    doorways = []
    for a, b, x, y, axis in midpoints:
        parallel = y if axis == 'x' else x
        doorways.append(dict(rooms=[a, b], wall_normal_axis=axis, center_xy_m=[x, y],
                             clear_interval_along_wall_m=[round(parallel-gap/2, 10), round(parallel+gap/2, 10)]))
    boxes = ((0, sx, 0, sy), (sx, width, 0, sy), (sx, width, sy, height), (0, sx, sy, height))
    return dict(layout=LAYOUT, footprint_xy_m=[width, height], partition_x_m=sx, partition_y_m=sy,
                grid_rows_cols=initial['shape'], grid_resolution_m=resolution, origin_xy_m=[0, 0],
                start_row_col_heading=initial['anchor'], heading_order=list(HEADINGS),
                camera_height_m=initial['actual_config']['camera_height_m'],
                camera_horizontal_fov_deg=initial['actual_config']['fov_deg'],
                source_room_boxes_xy_minmax_m=dict(zip(ROOMS, boxes, strict=True)),
                doorway_clear_width_m=gap, doorways=doorways,
                fixture_centers_xy_m=dict(shelf=config['shelf'][:2], cabinet=config['cabinet'][:2]))


def world_xy(pose, spec):
    r, c = pose[:2]
    rows, cols = spec['grid_rows_cols']
    assert 0 <= r < rows and 0 <= c < cols
    g = spec['grid_resolution_m']
    return round((c + .5) * g, 10), round((rows - r - .5) * g, 10)


def room(x, y, spec):
    width, height = spec['footprint_xy_m']
    sx, sy = spec['partition_x_m'], spec['partition_y_m']
    if not -TOL_M <= x <= width + TOL_M or not -TOL_M <= y <= height + TOL_M:
        return 'outside_source_footprint'
    if abs(x - sx) <= TOL_M or abs(y - sy) <= TOL_M:
        return 'divider_coordinate'
    return ROOMS[(0 if x < sx else 1) if y < sy else (3 if x < sx else 2)]


def transition(before, after, row, spec):
    doorway = next(d for d in spec['doorways'] if set(d['rooms']) == {before, after})
    x, y = row['x_m'], row['y_m']
    parallel = y if doorway['wall_normal_axis'] == 'x' else x
    low, high = doorway['clear_interval_along_wall_m']
    assert low <= parallel <= high and row['action'] == 'forward'
    return dict(paid=row['paid'], from_room=before, to_room=after,
                entry_xy_m=[x, y], observed_heading=row['heading'],
                doorway_center_xy_m=doorway['center_xy_m'])


def sector_row(step, spec, events, mode, own_previous):
    paid = step['step']
    xy = world_xy(step['position'], spec)
    here = room(*xy, spec)
    assert here in ROOMS
    record = events.get(step['selection_call_id'])
    if record is not None:
        assert step['selected_target_pose'] == record['same_global_target']
        target_room = room(*world_xy(record['same_global_target'], spec), spec)
        selected_paid = record['paid_before']
    else:
        target_room = ''
        selected_paid = ''
    pose = (*step['position'], step['heading'])
    return dict(mode=mode, paid=paid, x_m=xy[0], y_m=xy[1], row=pose[0], col=pose[1],
                heading=HEADINGS[pose[2]], heading_id=pose[2], room=here, action=step['action'],
                new_oriented_cell=(pose not in own_previous) and paid > 0,
                phase=step['phase_before_action'], target_selected_paid=selected_paid,
                target_room=target_room, global_call_id=step['selection_call_id'],
                active_local_route=step['local_route_id'], arrived_bound_target=step['matched_target_pose_after_step'],
                observed_packet_sha256=step['packet_sha256'])


def audit_trace(arm, spec, inputs):
    run = RUN_STUDY / 'runner/runs' / LAYOUT / arm
    manifest_path = run / 'manifest.json'
    manifest = read(manifest_path)
    assert manifest['status'] == 'completed_execution' and manifest['paid_actions'] == 512
    assert manifest['returned'] and manifest['collisions'] == 0
    baseline_files = ('steps.json', 'audit.json', 'local_choices.json', 'global_choices.json')
    checkpoints = ('start', 'paid_256', 'paid_320', 'terminal')
    for relative in (*baseline_files, *(f'{c}/{name}' for c in checkpoints
                      for name in ('checkpoint.json','native_mesh.ply','belief.npz'))):
        path = run / relative
        digest = sha(path)
        assert digest == manifest['files_sha256'][relative]
        inputs[str(path)] = digest
    inputs[str(manifest_path)] = sha(manifest_path)
    steps = read(run / 'steps.json')
    assert [row['step'] for row in steps] == list(range(1, 513))
    assert all(row['guard_allowed'] and not row['collision'] for row in steps)
    assert steps[-1]['position'] + [steps[-1]['heading']] == spec['start_row_col_heading']
    planned = read(run / 'local_choices.json')
    events = {event['selection_call_id']: event for event in planned}
    global_events = {event['call_id']: event for event in read(run / 'global_choices.json')}
    assert len(events) == len(planned)
    for event in planned:
        native = global_events[event['selection_call_id']]
        assert native['outputs']['mode'] == 'G' and native['outputs']['selected']['pose'] == event['same_global_target']
    r, c, h = spec['start_row_col_heading']
    start = dict(step=0, position=[r,c], heading=h, action='', phase_before_action='', selection_call_id=None,
                 local_route_id='', matched_target_pose_after_step=None, packet_sha256=manifest['initial_packet_sha256'])
    visited, projected, transitions = set(), [], []
    for step in (start, *steps):
        row = sector_row(step, spec, events, arm, visited)
        visited.add((*step['position'], step['heading']))
        if projected and row['room'] != projected[-1]['room']:
            transitions.append(transition(projected[-1]['room'], row['room'], row, spec))
        projected.append(row)
    counts = []
    for low, high in WINDOWS:
        subset = projected[low:high+1]
        for zone in ROOMS:
            room_rows = [row for row in subset if row['room']==zone]
            headings = Counter(row['heading'] for row in room_rows)
            commands = Counter(row['action'] for row in room_rows)
            counts.append(dict(mode=arm, first_paid=low, last_paid=high, room=zone, acquired_poses=len(room_rows),
                               forward=commands['forward'], turns=commands['left']+commands['right'],
                               new_oriented_cells=sum(row['new_oriented_cell'] for row in room_rows),
                               north=headings['N'], east=headings['E'], south=headings['S'], west=headings['W']))
    audit = read(run / 'audit.json')
    raw_replan = [item for item in audit if item['event']=='exploration_joint_replan_after_rejection']
    decisions = []
    for event in planned:
        if 190 <= event['paid_before'] <= 400:
            reached = next((step['step'] for step in steps if step['selection_call_id']==event['selection_call_id']
                            and step['matched_target_pose_after_step']), None)
            decisions.append(dict(paid_before=event['paid_before'], origin_xy_m=world_xy(event['pose'],spec),
                target_pose=event['same_global_target'], target_xy_m=world_xy(event['same_global_target'],spec),
                route=event['active_local_route_id'], call_id=event['selection_call_id'], bound_arrival_paid=reached))
    return steps, projected, counts, dict(transitions=transitions, checkpoint_poses=[projected[i] for i in CLOCKS],
                                         planning_events_near_comparison_window=decisions,
                                         actual_native_guard_replans=raw_replan)


def published_metrics(inputs):
    measured = RUN_STUDY / 'measurement/results/checkpoints.csv'
    coverage = RUN_STUDY / 'measurement/results/coverage_curve.csv'
    inputs[str(measured)] = sha(measured)
    inputs[str(coverage)] = sha(coverage)
    fields = ('C_free', 'F1_scene_05cm', 'p_scene_05cm', 'r_scene_05cm', 'J',
              'object_net_m2', 'shelf_net_m2', 'cabinet_net_m2')
    checkpoints = {arm: {} for arm in ARMS}
    with measured.open(newline='') as stream:
        for row in csv.DictReader(stream):
            if row['layout']==LAYOUT and row['mode'] in ARMS and row['checkpoint'] in ('paid_256','paid_320','terminal'):
                checkpoints[row['mode']][int(row['actual_paid'])] = {key:float(row[key]) for key in fields}
    assert all(set(x)==set(CLOCKS) for x in checkpoints.values())
    delta = {paid: {key:checkpoints['fullS'][paid][key]-checkpoints['fullG'][paid][key] for key in fields}
             for paid in CLOCKS}
    coverage_256_512 = {arm: {} for arm in ARMS}
    with coverage.open(newline='') as stream:
        for row in csv.DictReader(stream):
            if row['layout']==LAYOUT and row['mode'] in ARMS and 256 <= int(row['paid']) <= 512:
                coverage_256_512[row['mode']][int(row['paid'])] = float(row['C_free'])
    assert all(len(rows)==257 for rows in coverage_256_512.values())
    return dict(checkpoints=checkpoints, S_minus_G=delta,
                min_C_free_after_paid_256={arm:min(values.values()) for arm,values in coverage_256_512.items()})


def reference_sectors(path, spec, inputs):
    path = path.resolve()
    inputs[str(path)] = sha(path)
    rows = []
    with np.load(path, allow_pickle=False) as data:
        for panel, paid in zip(PANELS, CLOCKS, strict=True):
            xyz = data[f'{panel}__reference_xyz']
            status = data[f'{panel}__reference_status']
            weight = data[f'{panel}__reference_weight_m2']
            ids = data[f'{panel}__reference_source_id']
            assert len(xyz)==len(status)==len(ids) and set(np.unique(ids)) <= {0,31,118}
            assert set(np.unique(status)) <= {0,1,2,3}
            # Bits are G=1 and S=2. A sample matched by both is status 3.
            # Grid zones describe reference XY position, never material or camera origin.
            sectors = np.asarray([room(float(x),float(y),spec) for x,y in xyz[:,:2]])
            weight = np.broadcast_to(weight, status.shape)
            for sector in (*ROOMS,'divider_coordinate','outside_source_footprint'):
                for source_id in (0,31,118):
                    mask = (sectors==sector)&(ids==source_id)
                    counts = [int(np.count_nonzero(mask&(status==value))) for value in range(4)]
                    areas = [float(weight[mask&(status==value)].sum()) for value in range(4)]
                    rows.append(dict(paid=paid, geometric_sector=sector, source_instance_id=source_id,
                        reference_count=sum(counts), unmatched_both_count=counts[0], G_only_count=counts[1],
                        S_only_count=counts[2], matched_both_count=counts[3], G_only_m2=areas[1], S_only_m2=areas[2],
                        S_minus_G_matched_m2=areas[2]-areas[1]))
    return rows


def northwest_adjacency(path, spec, trajectory):
    poses = {arm: [row for row in trajectory if row['mode']==arm and 0 < row['paid'] <= 256
                   and row['room'] == 'D_northwest'] for arm in ARMS}
    xy = {arm: np.asarray([(row['x_m'],row['y_m']) for row in rows]) for arm,rows in poses.items()}
    route = {arm: dict(first_paid=rows[0]['paid'], last_paid=rows[-1]['paid'], acquired_poses=len(rows),
                      distinct_cells=len(set(map(tuple,xy[arm]))),
                      horizontal_bounds_m=[xy[arm].min(axis=0).tolist(),xy[arm].max(axis=0).tolist()])
             for arm,rows in poses.items()}
    panels = {}
    with np.load(path,allow_pickle=False) as data:
        for panel,paid in zip(PANELS,CLOCKS,strict=True):
            xyz = data[f'{panel}__reference_xyz']
            status = data[f'{panel}__reference_status']
            ids = data[f'{panel}__reference_source_id']
            planar = np.asarray([room(float(x),float(y),spec)=='D_northwest' for x,y in xyz[:,:2]])
            group = planar & (np.abs(xyz[:,2]) <= TOL_M) & (ids==0)
            positive = group & (status==2)
            negative = group & (status==1)
            weights = np.broadcast_to(data[f'{panel}__reference_weight_m2'],status.shape)
            points = xyz[positive,:2]
            distances = {}
            for arm,coordinates in xy.items():
                nearest = np.linalg.norm(points[:,None] - coordinates[None],axis=-1).min(axis=1)
                distances[arm] = dict(nearest_pose_horizontal_distance_median_m=float(np.median(nearest)),
                                      nearest_pose_horizontal_distance_10_and_90_percentile_m=np.quantile(nearest,[.1,.9]).tolist())
            panels[paid] = dict(S_only_points=int(positive.sum()),G_only_points=int(negative.sum()),
                S_only_area_m2=float(weights[positive].sum()),G_only_area_m2=float(weights[negative].sum()),
                positive_XY_bounds_m=[points.min(axis=0).tolist(),points.max(axis=0).tolist()],
                positive_x_median_m=float(np.median(points[:,0])),
                near_recorded_NW_camera_centers_before_256=distances)
    return dict(geometric_scope='NW source instance ID 0 and absolute z <= 1e-6 m',
                before_256_pose_bounds=route,matched_point_panel_adjacency=panels,
                inference_limit='Planar distances to actual recorded camera positions alone do not establish visibility or view contribution.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spatial-masks', type=Path)
    args = parser.parse_args()
    spec = geometry()
    scene_inputs = (SCENE/'scene.py',WORLD_SOURCE,TRANSFORM_SOURCE,SCENE/'prepared'/LAYOUT/'initial_state.json')
    frozen_world = read(SCENE/'freeze.json')
    assert all(sha(path)==frozen_world['files_sha256'][str(path)] for path in (scene_inputs[0], scene_inputs[1], scene_inputs[3]))
    actor_freeze = RUN_STUDY/'runner/holdout_freeze.json'
    assert sha(TRANSFORM_SOURCE)==read(actor_freeze)['files_sha256'][str(TRANSFORM_SOURCE)]
    inputs = {str(path):sha(path) for path in (*scene_inputs, SCENE/'freeze.json', actor_freeze)}
    raw, csv_rows, sector_rows, qualitative = {}, [], [], {}
    for arm in ARMS:
        raw[arm], projected, counts, qualitative[arm] = audit_trace(arm,spec,inputs)
        csv_rows.extend(projected)
        sector_rows.extend(counts)
    checked = ('action','position','heading','packet_sha256','depth_values_sha256','rgb_values_sha256','scan_ranges_sha256')
    aligned = list(zip(*raw.values(), strict=True))
    first_split = {key:next((a['step'] for a,b in aligned if a[key]!=b[key]),None) for key in checked}
    final_matches = sum(1 for a,b in aligned[490:] if all(a[key]==b[key] for key in checked))
    metrics = published_metrics(inputs)
    adjacency = None
    if args.spatial_masks:
        adjacency = northwest_adjacency(args.spatial_masks,spec,csv_rows)
        area_rows = reference_sectors(args.spatial_masks,spec,inputs)
        write_csv(HERE/'cycle16_reference_sectors.csv',area_rows)
    write_csv(HERE/'cycle16_trajectory.csv',csv_rows)
    write_csv(HERE/'cycle16_camera_poses_by_room.csv',sector_rows)
    facts = dict(status='read_only_geometry_and_execution_audit', geometry=spec,
                 actual_first_difference_paid=first_split, actual_last_22_actions_pose_and_sensor_equal=final_matches,
                 metrics=metrics, actual_navigation=qualitative, northwest_adjacency=adjacency, source_files_sha256=inputs,
                 camera_pose_table='Each paid command has one recorded post-action pose, including stationary turns.',
                 room_attribution='Coordinate-defined room sectors; room labels do not identify camera rays, surfaces or materials.',
                 spatial_mask_source=str(args.spatial_masks.resolve()) if args.spatial_masks else None)
    (HERE/'cycle16_mechanism.json').write_text(json.dumps(facts,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    main()
