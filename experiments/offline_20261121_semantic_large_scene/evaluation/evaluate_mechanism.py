"""Read the controlled runs and classify the selected and executed routes."""

import argparse
import json
from pathlib import Path

from gate import MODES, adjudicate
from measure import measure, read, sha, verified_file

STUDY = Path(__file__).resolve().parents[1]
SCENES = STUDY / 'mechanism/scenes'
RUNNER = STUDY / 'mechanism/runner'


def ref(path, checksum):
    return dict(path=str(path), sha256=checksum)


def gate_crossing(states, rule, grid_shape, cell_m):
    row = int(round(grid_shape[0] - .5 - rule['north_gate_y_m']/cell_m))
    for before, after in zip(states, states[1:]):
        if before[0] > row >= after[0]:
            x = (after[1]+.5)*cell_m
            if x <= rule['left_limit_x_m']:
                return 'left'
            if x >= rule['right_limit_x_m']:
                return 'right'
            return None
    return None


def scene_inputs(layout, runner_freeze):
    prepared = SCENES / 'prepared' / layout
    reference = SCENES / 'evaluator_references' / layout
    data = read(reference / 'RECEIPT.json')
    for path, checksum in data['source_inputs_sha256'].items():
        verified_file(ref(path, checksum))
    assert not data['actor_trajectory_read'] and not data['actor_executed']
    initial_path = RUNNER / 'layouts' / layout / 'prepared/initial_state.json'
    initial = read(verified_file(ref(initial_path, runner_freeze['files_sha256'][str(initial_path)])))
    scene = read(prepared / 'receipt.json')
    assert scene['status'] == 'geometry_and_sensor_preflight_passed'
    assert not scene['scored_candidates_opened'] and not scene['actor_run']
    assert data['layout'] == initial['layout'] == scene['layout'] == layout
    truth_file = prepared / 'evaluator_truth.npz'
    source_meta = dict(anchor=initial['anchor'], initial_packet_sha256=initial['initial_packet_sha256'],
                       reachable=ref(truth_file, data['source_inputs_sha256'][str(truth_file)]),
                       source_mesh=ref(reference / 'source_mesh.ply', data['source_mesh_sha256']),
                       fixed_samples=ref(reference / 'fixed_reference.npz', data['reference_npz_sha256']),
                       source_receipt=ref(reference / 'RECEIPT.json', sha(reference / 'RECEIPT.json')))
    return source_meta, scene, initial


def measure_trial(layout, arm, scene, initial, reference):
    trial = RUNNER / 'layouts' / layout / 'runs' / arm / 'attempt_001'
    rule = scene['route_evaluation']
    row = dict(layout=layout, parent=scene['parent'], placement=layout.rsplit('_', 1)[1], arm=arm,
               informative_side=rule['expected_diagnostic_side'], budget=initial['stop_budget'], path=str(trial))
    if not (trial / 'artifact_manifest.json').is_file():
        status = 'recorded_failure' if (trial / 'failure.json').is_file() else 'missing_or_in_progress'
        return dict(row, failed=True, returned=False, triggered=False, J=None, status=status)
    manifest = read(trial / 'artifact_manifest.json')
    if 'intervention.json' not in manifest['files_sha256']:
        return dict(row, failed=True, returned=False, triggered=False, J=None, status='awaiting_final_intervention_receipt')
    assert manifest['freeze_sha256'] == sha(RUNNER / 'freeze.json')
    result = measure(trial, reference)
    assert result['budget'] == initial['stop_budget']
    receipt = read(trial / 'intervention.json')
    assert not receipt['simulator_hidden_geometry_in_gate_or_scoring']
    plans = read(trial / 'choices.json')
    steps = read(trial / 'steps.json')
    row.update(result, triggered=receipt['status'] == 'triggered', gate_misses=receipt['prior_failed_gate_events'])
    if not row['triggered']:
        assert all(plan['outputs']['mode'] == 'G' for plan in plans)
        return row

    event = receipt['intervention']
    ordinal = event['native_plan_number']
    plan = next(p['outputs'] for p in plans if p['outputs']['plan_number'] == ordinal)
    assert event['mode'] == plan['mode'] == arm
    assert all(p['outputs']['mode'] == 'G' for p in plans if p['outputs']['plan_number'] != ordinal)
    stamp = event['paid_before']
    assert event['packet_sha256'] == (steps[stamp-1]['packet_sha256'] if stamp else initial['initial_packet_sha256'])
    shape = initial['shape']
    cell = initial['grid_resolution_m']
    option = event['chosen_native_option']
    selected = gate_crossing(option['outbound_states'], rule, shape, cell) if option else None
    route_sides = {route['candidate_id']: gate_crossing(route['outbound_states'], rule, shape, cell)
                   for route in plan['candidates']}
    completed_path = [event['current_pose']] + [[*s['position'], s['heading']] for s in steps[stamp:]]
    row.update(trigger_step=stamp, trigger_input_sha256=event['packet_sha256'],
               selected_id=event['chosen_candidate_id'], selected_side=selected,
               selected_side_trigger_proxy=event['selected_side'],
               executed_first_crossing=gate_crossing(completed_path, rule, shape, cell),
               candidate_set_sha256=event['candidate_set_sha256'], score_set_sha256=event['score_set_sha256'],
               route_sides_by_candidate=route_sides, rankings=event['candidate_summaries'],
               shape_priors=event['shape_priors'], remaining_budget=event['remaining_budget'])
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    freeze = read(RUNNER / 'freeze.json')
    for path, checksum in freeze['files_sha256'].items():
        verified_file(ref(path, checksum))
    sfreeze = read(SCENES / 'freeze.json')
    assert sfreeze['status'] == 'frozen_before_controlled_policy_outcomes'
    for path, checksum in sfreeze['files_sha256'].items():
        verified_file(ref(path, checksum))
    rows = []
    for layout in read(RUNNER / 'protocol.json')['layouts']:
        reference, scene, initial = scene_inputs(layout, freeze)
        group = [measure_trial(layout, arm, scene, initial, reference) for arm in MODES]
        triggered = [row for row in group if row['triggered']]
        for key in ('trigger_step', 'trigger_input_sha256', 'candidate_set_sha256', 'score_set_sha256'):
            assert len({row[key] for row in triggered}) <= 1, (layout, key)
        rows.extend(group)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'endpoint_rows.json').write_text(json.dumps(rows, indent=2, allow_nan=False) + '\n')
    finished = all(row['status'] not in ('missing_or_in_progress', 'awaiting_final_intervention_receipt') for row in rows)
    if finished:
        decision = adjudicate(rows, read(STUDY / 'PROTOCOL.json')['controlled_mechanism']['gate_numerical_tie_fraction'])
        print(json.dumps(decision, allow_nan=False))
    else:
        print(json.dumps(dict(status='awaiting_terminal_receipts', measured=sum(row.get('J') is not None for row in rows))))


if __name__ == '__main__':
    main()
