"""Post-hoc read-only comparison of native proposals with observable crossing support."""

import argparse
import hashlib
import json
from pathlib import Path

import run
import numpy as np
from scipy.sparse.csgraph import dijkstra


def audit(layout):
    # This reconstructs only the zero-action observation, then checks an already
    # committed choice against a frozen evaluator-only classification line.
    old, _ = run.install()
    scene = run.read(run.SCENES / 'prepared' / layout / 'receipt.json')
    choices = run.read(run.HERE / 'layouts' / layout / 'runs/G/attempt_001/choices.json')
    record = choices[0]
    if 'X' not in record['outputs']['scores']:
        return dict(layout=layout, status='first_choice_did_not_trigger')
    plan = record['outputs']
    budget = record['inputs']['remaining_budget']
    world, _, packet, backend, runtime, _, _ = old.bootstrap(layout, 'G')
    assert world.step_count == 0
    from nso_bridge.exploration_joint_policy import inflated_obstacles, orientation_graph

    mapper = runtime.states[0]['mapper']
    resolution = world.config.resolution_m
    safe = ~inflated_obstacles(mapper.belief != 0, world.config.robot_radius_m / resolution)
    assert hashlib.sha256(safe.tobytes()).hexdigest() == plan['candidate_audit']['safe_sha256']
    poses, _, ids = orientation_graph(safe)
    anchor = int(ids[packet.position]) * 4 + packet.heading
    outward = dijkstra(poses, directed=True, indices=anchor)
    return_cost = dijkstra(poses.T.tocsr(), directed=True, indices=anchor)
    rule = scene['route_evaluation']
    row = int(round(world.shape[0] - .5 - rule['north_gate_y_m'] / resolution))

    def side(col):
        x = (col + .5) * resolution
        return ('left' if x <= rule['left_limit_x_m'] else
                'right' if x >= rule['right_limit_x_m'] else None)

    sides = {'left': [], 'right': []}
    for col in range(world.shape[1]):
        label = side(col)
        if label is None or not safe[row, col] or not safe[row + 1, col]:
            continue
        start, end = int(ids[row + 1, col]) * 4, int(ids[row, col]) * 4
        if poses[start, end] and np.isfinite(outward[start]) and np.isfinite(return_cost[end]):
            cost = int(outward[start] + poses[start, end] + return_cost[end])
            if cost <= budget:
                sides[label].append(dict(column=col, outward_cost=int(outward[start] + poses[start, end]),
                                         return_cost=int(return_cost[end]), round_trip=cost))
    routes = []
    for index, route in enumerate(plan['candidates']):
        transitions = zip(route['outbound_states'], route['outbound_states'][1:])
        crossings = [side(after[1]) for before, after in transitions if before[0] > row >= after[0]]
        routes.append(dict(candidate_id=route['candidate_id'], native_role=route['group'],
                           max_north_m=resolution * (world.shape[0] - .5 - min(s[0] for s in route['outbound_states'])),
                           crossing_side=crossings[0] if crossings else None,
                           crossing_occurs=bool(crossings),
                           score_by_mode={mode: plan['scores'][mode][index] for mode in ('N', 'G', 'S', 'X')}))
    return dict(layout=layout, status='cold_start_crossing_analyzed', predecision_authorized_actions=0,
                observed_paid_budget=budget, frozen_geometric_gate=rule,
                sensor_safe_paid_lane_crossings=sides, native_proposals=routes,
                reference_truth_used_to_control_actor=False,
                interpretation='Observed-grid crossing only. The fixed evaluation line is read here after the recorded actor decision.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('layout', choices=run.LAYOUTS)
    args = parser.parse_args()
    report = audit(args.layout)
    print(json.dumps(report, indent=2, allow_nan=False))
