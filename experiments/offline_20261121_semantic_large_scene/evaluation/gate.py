"""Apply the predeclared mechanism gate to evaluator-measured endpoint rows."""

import argparse
import csv
import json
from pathlib import Path
from statistics import mean

PROTOCOL = Path(__file__).resolve().parents[1] / 'PROTOCOL.json'
MODES = ('S', 'G', 'X')
COLUMNS = ('parent', 'layout', 'placement', 'arm', 'status', 'C_free', 'F1_scene_05cm',
           'J', 'actions', 'budget', 'returned', 'collisions', 'stop_reason',
           'triggered', 'trigger_step', 'selected_side', 'informative_side', 'selected_id', 'path')


def adjudicate(rows, eps):
    parent_ids = sorted({row['parent'] for row in rows})
    if len(parent_ids) != 2:
        raise ValueError('The mechanism gate requires two declared parents')
    decisions = {}
    for parent in parent_ids:
        members = [row for row in rows if row['parent'] == parent]
        slots = {(row['placement'], row['arm']): row for row in members}
        places = sorted({row['placement'] for row in members})
        expected_slots = {(place, arm) for place in places for arm in MODES}
        if len(places) != 2 or set(slots) != expected_slots or len(members) != 6:
            raise ValueError(f'Missing or duplicate attempts for parent {parent}')
        for place in places:
            group = [slots[place, arm] for arm in MODES]
            if len({x['initial_packet_sha256'] for x in group if x.get('initial_packet_sha256')}) > 1:
                raise ValueError(f'Initial observation differs for {parent}/{place}')
            if len({x['budget'] for x in group}) != 1:
                raise ValueError(f'Action contract differs for {parent}/{place}')
            if len({x['informative_side'] for x in group}) != 1:
                raise ValueError(f'Geometry classification differs for {parent}/{place}')
            trigger_states = [x['trigger_input_sha256'] for x in group if x.get('triggered')]
            if len(set(trigger_states)) > 1:
                raise ValueError(f'Intervention state is unmatched for {parent}/{place}')
        all_measured = all(row.get('J') is not None for row in members)
        valid = all_measured and all(not row.get('failed') for row in members)
        sem = [slots[place, 'S'] for place in places]
        expected_sides = {row['informative_side'] for row in sem}
        if expected_sides != {'left', 'right'}:
            raise ValueError(f'Informative side did not swap in the scene design for {parent}')
        swaps = all(s['triggered'] and s['selected_side'] == s['informative_side'] for s in sem)
        matched_triggers = all(row.get('triggered', False) for row in members)
        no_return_regression = all(s['returned'] or not any(slots[s['placement'], arm]['returned'] for arm in ('G', 'X'))
                                   for s in sem)
        comparison = {arm: {key: mean(slots[place, arm][key] for place in places)
                            for key in ('J', 'C_free', 'F1_scene_05cm')} for arm in MODES} if all_measured else {}
        deltas = {arm: {key: comparison['S'][key]-comparison[arm][key] for key in comparison['S']}
                  for arm in ('G', 'X')} if all_measured else {}
        wins = valid and all(row['J'] > eps for row in deltas.values())
        decisions[parent] = dict(final_evaluation_valid=valid, S_tracks_informative_side=swaps,
                                 same_trigger_fired_in_all_arms=matched_triggers, improves_J_over_G_and_X=wins,
                                 return_not_worse=no_return_regression, parent_means=comparison,
                                 paired_mean_deltas=deltas,
                                 mechanism_supported=valid and swaps and matched_triggers and wins and no_return_regression)
    supported = sum(row['mechanism_supported'] for row in decisions.values())
    status = {0: 'stop_and_diagnose', 1: 'hold_and_explain_difference', 2: 'release_confirmatory'}[supported]
    return dict(status=status, supported_parents=supported, by_parent=decisions)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('rows', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    rows = json.loads(args.rows.read_text())
    eps = json.loads(PROTOCOL.read_text())['controlled_mechanism']['gate_numerical_tie_fraction']
    decisions = adjudicate(rows, eps)
    args.output.mkdir(exist_ok=True, parents=True)
    (args.output / 'decision.json').write_text(json.dumps(decisions, indent=2, allow_nan=False) + '\n')
    with (args.output / 'mechanism_outcomes.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(decisions, allow_nan=False))


if __name__ == '__main__':
    main()
