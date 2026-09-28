"""Compose frozen routes only; no simulator, ray query, sensor or Q access.

After either declared ideal-depth first-difference probe, retrace it with two
half-turns and perform either existing two-view arm from the original anchor.
This is a feasible-route witness, not optimal cost or a noisy online policy.
"""
from pathlib import Path
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'audit_results/facility_choice_v25_service_cost_20260915'
OUTPUT = ROOT / 'audit_results/facility_choice_v25r1_multiview_recovery_composition_20260915'
DELTAS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def walk(start, actions):
    states = [list(start)]
    for action in actions:
        r, c, h = states[-1]
        if action == 'forward':
            dr, dc = DELTAS[h]
            r, c = r + dr, c + dc
        elif action == 'right':
            h = (h + 1) % 4
        elif action == 'left':
            h = (h - 1) % 4
        else:
            raise ValueError('undeclared action')
        states.append([r, c, h])
    return states


def main():
    if OUTPUT.exists():
        raise ValueError('new output required; completed evidence is not overwritten')
    inventory = json.loads((SOURCE / 'artifact_hashes.json').read_text())
    assert all(sha(SOURCE / name) == digest for name, digest in inventory.items())
    manifest = json.loads((SOURCE / 'manifest.json').read_text())
    result = json.loads((SOURCE / 'result.json').read_text())
    assert manifest['status'] == 'complete' and result['status'] == 'complete_static_cost_only'
    sources = {**manifest['source_sha256'], **manifest['protected_source_sha256']}
    assert all(sha(ROOT / name) == digest for name, digest in sources.items())
    rows = []
    for parent in result['parents']:
        anchor = parent['anchor']
        for probe_index, probe in enumerate(parent['conditional_recovery']):
            approach = probe['information_prefix_actions']
            poses = probe['information_prefix_states']
            assert walk(anchor, approach) == poses
            assert len(approach) == probe['first_difference']['local_action']
            inverse = {'forward': 'forward', 'left': 'right', 'right': 'left'}
            back_actions = ['right', 'right'] + [inverse[a] for a in approach[::-1]] + ['right', 'right']
            back_states = walk(poses[-1], back_actions)
            assert back_states[-1] == anchor
            assert back_states[2:-2] == [[r, c, (h + 2) % 4] for r, c, h in poses[::-1]]
            assert {tuple(p[:2]) for p in back_states} == {tuple(p[:2]) for p in poses}
            for arm, route in parent['route_catalog'].items():
                if arm not in ('A', 'B'):
                    continue
                assert route['acquisition_contract'] == result['actual_acquisition_contract']
                assert route['prefix_states'][-1] == anchor
                prefix = route['prefix_actions']
                tail = route['actions'][len(prefix):]
                assert route['states'][len(prefix):] == walk(anchor, tail)
                assert route['states'][-1] == anchor
                actions = prefix + approach + back_actions + tail
                states = walk(anchor, actions)
                assert states[:len(prefix) + 1] == route['prefix_states']
                offset = len(approach) + len(back_actions)
                assert states[len(prefix) + offset:] == route['states'][len(prefix):]
                arrivals = [frame + offset for frame in route['arrival_frame_indices']]
                assert all(states[frame] == pose for frame, pose in zip(arrivals, route['selected_view_states']))
                assert states[-1] == anchor
                assert len(actions) == route['paid_total'] + 2 * len(approach) + 4
                rows.append(dict(parent=parent['parent'], probe_index=probe_index, arm=arm,
                    ideal_probe_state=probe['first_difference']['state'],
                    ideal_probe_differing_pixels=probe['first_difference']['differing_pixels'],
                    ideal_probe_paid=len(approach), retrace_paid=len(back_actions),
                    extra_vs_direct=offset, direct_paid=route['paid_total'], total_paid=len(actions),
                    budget=parent['total_budget'], remaining_budget=parent['total_budget'] - len(actions),
                    budget_fits=len(actions) <= parent['total_budget'],
                    selected_view_states=route['selected_view_states'], arrival_frame_indices=arrivals,
                    exact_anchor_and_heading_returned=True, retrace_only_uses_saved_safe_cells=True,
                    common_across_assignments=True, original_arm_suffix_unchanged=True,
                    actions=actions, states=states))
    assert len(rows) == 8
    report = dict(status='complete', route_witnesses=rows,
        all_witnesses_fit_budget=all(row['budget_fits'] for row in rows),
        construction='paid prefix + declared ideal probe + exact retrace with4 turns + full frozen acquisition suffix',
        source_artifacts_verified=len(inventory), source_files_verified=len(sources),
        source_inventory_sha256=sha(SOURCE / 'artifact_hashes.json'),
        source_result_sha256=sha(SOURCE / 'result.json'), script_sha256=sha(Path(__file__)),
        world_constructions=0, physical_actions=0, new_sensor_packets=0, new_quality_evaluations=0,
        path_searches=0, original_cost_analysis_reexecuted=False,
        perfect_identification_from_saved_ideal_depth_difference_assumed=True,
        circular_heading_independent_footprint_assumption='inherited frozen simulator/cost-graph action contract',
        noisy_recognizer_or_unknown_map_planner_validated=False,
        equal_terminal_Q_after_extra_history_claimed=False,
        scope='GT-feasible composed double-view recovery witnesses; not optimal cost or executed geometry policy')
    payload = (json.dumps(report, separators=(',', ':')) + '\n').encode()
    assert len(payload) < 262144
    if shutil.disk_usage(ROOT).free < 67108864 + 1048576:
        raise OSError('preserve64MiB free plus1MiB analysis reserve')
    OUTPUT.mkdir()
    (OUTPUT / 'result.json').write_bytes(payload)
    (OUTPUT / 'artifact_hashes.json').write_text(json.dumps({'result.json': sha(OUTPUT / 'result.json')}) + '\n')
    for row in rows:
        print(row['parent'], row['probe_index'], row['arm'], row['total_paid'], '/', row['budget'])
    print('8 composed witnesses;0 new physical actions;result', sha(OUTPUT / 'result.json'))


if __name__ == '__main__':
    main()
