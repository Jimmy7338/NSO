#!/usr/bin/env python3
"""Verify floor topology, geometry-only action budgets and archival novelty."""

import argparse
from collections import defaultdict, deque
import hashlib
import json

import numpy as np

from scene import ANALYSES, BUDGET, HERE, HOLDOUT, RESOLUTION_M, SPECS


DIRECTIONS = ((-1,0), (0,1), (1,0), (0,-1))


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def array_sha(array):
    array = np.ascontiguousarray(array, dtype=np.bool_)
    digest = hashlib.sha256(f'{array.shape}\n'.encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def rotations(array):
    for count in range(4):
        yield np.rot90(array, k=count)
        yield np.rot90(array[:,::-1], k=count)


def known_archive_signatures():
    archived = defaultdict(list)
    collections = {'occupancy': 0, 'reachable': 0}
    for path in sorted(ANALYSES.rglob('evaluator_truth.npz')):
        if HERE.parent in path.parents:
            continue
        try:
            with np.load(path,allow_pickle=False) as saved:
                for key in collections:
                    if key in saved and saved[key].ndim == 2:
                        collections[key] += 1
                        archived[key,array_sha(saved[key])].append(str(path))
        except (OSError, ValueError) as error:
            raise ValueError(f'Archived reference cannot be inspected for novelty: {path}') from error
    if min(collections.values()) < 1:
        raise ValueError('At least one archived occupancy/reachable reference is required')
    return archived, collections


def map_rooms(spec, shape):
    rows, cols = shape
    x = ((np.arange(cols)+.5)*RESOLUTION_M)[None,:]
    y = ((rows-.5-np.arange(rows))*RESOLUTION_M)[:,None]
    region = np.full(shape,-1,dtype=np.int16)
    interiors = {}
    for ident,(name,(x0,y0,x1,y1)) in enumerate(spec['rooms'].items()):
        room = (x>=x0)&(x<x1)&(y>=y0)&(y<y1)
        if (region[room] >= 0).any():
            raise ValueError('Analytical room labels overlap on the evaluation grid')
        region[room] = ident
        interiors[name] = (x >= x0+.4)&(x < x1-.4)&(y >= y0+.4)&(y < y1-.4)
    if (region < 0).any():
        raise ValueError('Analytical room labels leave unlabeled source floor')
    return region, interiors, x, y


def swept_entries(room_names, room_labels, reachable, edge_valid):
    result = defaultdict(list)
    for r,c,h in np.argwhere(edge_valid & reachable[:,:,None]):
        dr,dc = DIRECTIONS[h]
        source = room_labels[r,c]
        target = room_labels[r+dr,c+dc]
        if source != target:
            key = tuple(sorted((room_names[source],room_names[target])))
            if room_names[source] == key[0]:
                result[key].append([int(r),int(c),int(h)])
    return result


def heading_distances(edge_valid, origin, reverse=False):
    distance = np.full(edge_valid.shape, -1, dtype=np.int32)
    queue = deque([origin])
    distance[origin] = 0
    row_count,column_count = edge_valid.shape[:2]
    while queue:
        r,c,h = queue.popleft()
        dr,dc = DIRECTIONS[h]
        candidates = [(r,c,(h-1)%4),(r,c,(h+1)%4)]
        if reverse:
            rr,cc = r-dr,c-dc
            if 0 <= rr < row_count and 0 <= cc < column_count and edge_valid[rr,cc,h]:
                candidates.append((rr,cc,h))
        elif edge_valid[r,c,h]:
            candidates.append((r+dr,c+dc,h))
        for state in candidates:
            if distance[state] == -1:
                distance[state] = distance[r,c,h]+1
                queue.append(state)
    return distance


def room_tour(edge_valid, rooms, landmarks, anchor):
    """Compute a source-only tour via fixed interior witness cells."""
    count = len(rooms)
    states = [anchor, *((*landmarks[room],h) for room in rooms for h in range(4))]
    travel = np.array([[distance[state] for state in states]
                       for distance in (heading_distances(edge_valid, state) for state in states)])
    if (travel < 0).any():
        raise ValueError('Geometry-only witness cells must be mutually reachable')
    frontier = {(0,0):(0,())}
    for _ in range(count):
        updated = {}
        for (visited, state),(cost,path) in frontier.items():
            for index in range(count):
                bit = 1<<index
                if visited & bit:
                    continue
                for goal in range(1+4*index,5+4*index):
                    key = (visited|bit,goal)
                    candidate = (cost+int(travel[state,goal]), path+(goal,))
                    updated[key] = min(candidate,updated.get(key,candidate))
        frontier = updated
    cost, sequence = min((paid+int(travel[state,0]), path) for (_,state),(paid,path) in frontier.items())
    return {'actions_including_turns_and_return_heading': cost, 'fits_192_action_cap': cost <= BUDGET,
            'interior_witness_states_rc_heading': [list(states[state]) for state in sequence]}


def check_geometry(layout, archived, archive_counts, seen):
    path = HERE/'evaluator_references'/layout
    spec = SPECS[layout]
    initial = read(HERE/'prepared'/layout/'initial_state.json')
    config = initial['actual_config']
    preliminary = read(path/'cold_start_preflight.json')
    if (initial['paid_actions'] or initial['policy_select_calls'] or initial['seed'] != spec['seed']
            or initial['stop_budget'] != BUDGET or preliminary['initial_unknown_reachable_cells'] <= 0
            or config['resolution_m'] != .2 or config['robot_radius_m'] != .2 or config['max_depth_m'] != 8):
        raise ValueError(f'Start or sensing does not satisfy the declared contract: {layout}')
    truth_path = path/'evaluator_truth.npz'
    with np.load(truth_path,allow_pickle=False) as saved:
        occupancy, reachable, edges = (saved[key] for key in ('occupancy','reachable','edge_valid'))
    anchor = tuple(initial['anchor'])
    names = list(spec['rooms'])
    labels, interiors, x, y = map_rooms(spec, reachable.shape)
    start_room = names[labels[anchor[:2]]]
    actual_edges = swept_entries(names,labels,reachable,edges)
    expected_edges = {door['rooms'] for door in spec['architectural_doors']}
    if set(actual_edges) != expected_edges:
        raise ValueError(f'Expected/mesh door edge disagreement in {layout}: {expected_edges=} {set(actual_edges)=}')
    doors = [door | {'swept_center_crossings_each_way':len(actual_edges[door['rooms']]),
                    'sampled_lateral_crossing_band_m':len(actual_edges[door['rooms']])*RESOLUTION_M,
                    'exact_crossing_states_rc_heading':actual_edges[door['rooms']]}
             for door in spec['architectural_doors']]
    outward = heading_distances(edges, anchor)
    returning = heading_distances(edges, anchor, reverse=True)
    details, landmarks = {}, {}
    for index,(name,rect) in enumerate(spec['rooms'].items()):
        reached = reachable & (labels == index)
        inhabit = reached & interiors[name]
        if not inhabit.any():
            raise ValueError(f'No inside-centre access across swept doors: {layout}/{name}')
        inside = inhabit[:,:,None] & (outward >= 0) & (returning >= 0)
        costs = np.where(inside,outward+returning,np.iinfo(np.int32).max)
        px,py = (rect[0]+rect[2])/2,(rect[1]+rect[3])/2
        to_center = np.where(inhabit,(x-px)**2+(y-py)**2,np.inf)
        landmarks[name] = tuple(int(k) for k in np.unravel_index(to_center.argmin(),reachable.shape))
        details[name] = {'reachable_cells':int(reached.sum()),'reachable_area_m2':float(reached.sum()*RESOLUTION_M**2),
                         'minimum_actions_inside_visit_then_return':int(costs.min()),
                         'interior_witness_cell_rc':landmarks[name]}
    max_round_trip = max(item['minimum_actions_inside_visit_then_return'] for item in details.values())
    if not 0 < max_round_trip <= BUDGET:
        raise ValueError(f'At least one declared room lacks a budget-feasible visit and return: {layout}')
    tour = room_tour(edges,[room for room in names if room!=start_room],landmarks,anchor)
    novelty = {}
    for name,array in (('occupancy',occupancy),('reachable',reachable)):
        signatures = {array_sha(item) for item in rotations(array)}
        matches = sorted({p for signature in signatures for p in archived[name,signature]}
                         | {seen[name,signature] for signature in signatures if (name,signature) in seen})
        if matches:
            raise ValueError(f'Geometry duplicates an earlier reference up to dihedral transforms: {layout}/{name}/{matches}')
        seen[name,array_sha(array)] = layout
        novelty[name] = {'array_sha256':array_sha(array),'other_archives_checked':archive_counts[name], 'matched':matches}
    return {'layout':layout,'topology':spec['topology'],'seed':spec['seed'],'bounds_xy_m':[spec['width_m'],spec['height_m']],
            'start_rc_heading':list(anchor),'start_room':start_room,'room_count':len(details),
            'rooms':details,'doors':doors,'sensor_config':config,'budget_actions':BUDGET,'true_grid_shape':list(reachable.shape),
            'reachable_cells':int(reachable.sum()),'reachable_area_m2':float(reachable.sum()*RESOLUTION_M**2),
            'first_frame_known_unknown_reachable_cells':preliminary['initial_unknown_reachable_cells'],
            'all_individual_room_round_trips_fit_budget':True,'largest_minimum_round_trip_actions':max_round_trip,
            'geometry_only_room_tour':tour,'evaluator_truth_sha256':sha(truth_path),'archive_novelty':novelty}


def evaluate(output):
    archive, counts = known_archive_signatures()
    seen = {}
    details = [check_geometry(layout,archive,counts,seen) for layout in HOLDOUT]
    audit = {'status':'all_formal_geometries_verified_without_actor_execution','policy_calls':0,
             'meaning':'Door verification uses the simulated swept circle. Budget witnesses use full truth after cold-start preparation and never enter policy ranking, mapping, or poses.',
             'scan_range_note':'RGB-D and planar lidar use the same frozen maximum range configured in the source simulator.',
             'layouts':details}
    with output.open('x') as stream:
        json.dump(audit,stream,indent=2,allow_nan=False)
        stream.write('\n')
    print(json.dumps({item['layout']:{'areas_m2':item['reachable_area_m2'],
                     'max_room_return':item['largest_minimum_round_trip_actions'],
                     'witness_tour':item['geometry_only_room_tour']['actions_including_turns_and_return_heading']}
                     for item in details},indent=2),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=type(HERE),default=HERE/'formal_geometry_audit.json')
    evaluate(parser.parse_args().output)
