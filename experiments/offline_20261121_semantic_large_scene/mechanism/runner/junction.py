"""Observable test and audit for the one native junction decision."""

import numpy as np


def camera_right_rc(heading):
    return ((0, 1), (1, 0), (0, -1), (-1, 0))[int(heading)]


def lateral_cells(position, origin, heading):
    return sum((int(a) - int(b)) * e for a, b, e in zip(position, origin, camera_right_rc(heading)))


def route_record(route, position, heading):
    side = None
    branch_step = None
    for step, pose in enumerate(route['outbound_states'][1:29], 1):
        distance = lateral_cells(pose[:2], position, heading)
        if abs(distance) >= 3:
            side = 'right' if distance > 0 else 'left'
            branch_step = step
            break
    return dict(candidate_id=int(route['candidate_id']), side=side, first_side_step=branch_step,
                outbound_cost=len(route['outbound_actions']), reserved_return_cost=len(route['return_actions']),
                cost=int(route['cost']))


def observed_gate(packet, anchor, routes):
    frame = packet.frame
    # This test uses foreground presence, independent of its category identity.
    has_facility = (frame.semantic > 0) & (frame.depth_m > 0)
    columns = np.arange(frame.depth_m.shape[1]) - float(frame.intrinsic[0, 2])
    band = .06 * len(columns)
    left = int(np.count_nonzero(has_facility[:, columns <= -band]))
    right = int(np.count_nonzero(has_facility[:, columns >= band]))
    axis_error = lateral_cells(packet.position, anchor[:2], anchor[2])
    aligned = packet.heading == anchor[2] and abs(axis_error) <= 2
    records = [route_record(route, packet.position, packet.heading) for route in routes]
    matched = [[l['candidate_id'], r['candidate_id']]
               for l in records if l['side'] == 'left'
               for r in records if r['side'] == 'right'
               if abs(l['outbound_cost'] - r['outbound_cost']) <= 6
               and abs(l['reserved_return_cost'] - r['reserved_return_cost']) <= 8]
    return dict(qualified=bool(aligned and left >= 16 and right >= 16 and matched),
                paid_before=int(packet.action_id), current_pose=[*map(int, packet.position), int(packet.heading)],
                start_axis_lateral_error_cells=axis_error, start_axis_aligned=bool(aligned),
                current_foreground_pixels={'left': left, 'right': right}, candidates=records,
                matched_left_right_candidates=matched)


def summarize_ranking(records, scores, diagnostics):
    index_for = {row['candidate_id']: n for n, row in enumerate(records)}
    ranks = {}
    for mode in ('G', 'S', 'X'):
        order = sorted(range(len(records)), key=lambda i: (-scores[mode][i], records[i]['cost'],
                                                           records[i]['candidate_id']))
        ranks[mode] = {records[i]['candidate_id']: n for n, i in enumerate(order, 1)}
    output = []
    for record in records:
        cid = record['candidate_id']
        i = index_for[cid]
        output.append(dict(record, common_score=float(scores['N'][i]),
                           scores={m: float(scores[m][i]) for m in ('G', 'S', 'X')},
                           ranks={m: ranks[m][cid] for m in ranks},
                           template_area_m2=diagnostics[i]['template_expected_visible_area_m2']))
    return output
