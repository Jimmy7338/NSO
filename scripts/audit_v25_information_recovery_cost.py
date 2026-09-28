#!/usr/bin/env python3
"""Reproduce a post-hoc conditional cost diagnosis on frozen safe route cells.

No world, camera, mapper or evaluator is imported. A path is a cost upper-bound
witness under the saved observed-grid contract, not actual task success.
"""
import argparse
import ast
from collections import deque
import hashlib
import json
from pathlib import Path
import shutil

DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))
ROOT = Path('/root/NSO')
READY = ROOT/'audit_results/facility_runtime_v24_readiness_20260915'
WINDOW = ROOT/'audit_results/facility_choice_v24_candidate_window_20260915'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def step(state, action):
    r, c, h = state
    if action == 'left':
        return r, c, (h-1) % 4
    if action == 'right':
        return r, c, (h+1) % 4
    assert action == 'forward', action
    dr, dc = DIRECTIONS[h]
    return r+dr, c+dc, h


def neighbours(state, cells):
    for action in ('left', 'right', 'forward'):
        target = step(state, action)
        if target[:2] in cells:
            yield target, action


def path(start, target, cells):
    start, target = tuple(start), tuple(target)
    previous = {start: None}
    queue = deque([start])
    while queue:
        state = queue.popleft()
        if state == target:
            states, actions = [state], []
            while previous[state] is not None:
                state, action = previous[state]
                states.append(state); actions.append(action)
            return list(reversed(states)), list(reversed(actions))
        for next_state, action in neighbours(state, cells):
            if next_state not in previous:
                previous[next_state] = state, action
                queue.append(next_state)
    return None


def validate(states, actions, cells):
    assert len(states) == len(actions)+1
    assert all(tuple(s[:2]) in cells and s[2] in range(4) for s in states)
    assert all(step(s, a) == tuple(t) for s, a, t in zip(states, actions, states[1:]))


def verified_input(path, hashes):
    root = READY if path.is_relative_to(READY) else WINDOW
    inventory = read(root/'artifact_hashes.json')
    row = inventory[str(path.relative_to(root))]
    expected = row['sha256'] if isinstance(row, dict) else row
    actual = sha(path)
    assert actual == expected, str(path)
    hashes[str(path.relative_to(ROOT))] = actual
    hashes[str((root/'artifact_hashes.json').relative_to(ROOT))] = sha(root/'artifact_hashes.json')
    return read(path)


def run():
    inputs = {}
    # The standard heading/action convention is read and checked, not imported.
    geometry_file = ROOT/'utils/grid_geometry.py'
    tree = ast.parse(geometry_file.read_text())
    constants = [ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == 'DIRECTIONS' for t in n.targets)]
    assert constants == [DIRECTIONS]
    inputs[str(geometry_file.relative_to(ROOT))] = sha(geometry_file)
    window = verified_input(WINDOW/'result.json', inputs)
    assert window['status'] == 'complete'
    output = []
    for parent, index in [('D24-P00', 0), ('D24-P01', 2)]:
        g = verified_input(READY/f'case_{index:02d}_G.json', inputs)
        n = verified_input(READY/f'case_{index:02d}_N.json', inputs)
        other = verified_input(READY/f'case_{index+1:02d}_G.json', inputs)
        assert all(r['status'] == 'complete' for r in (g,n,other))
        candidates = g['selection']['candidates']
        assert candidates == n['selection']['candidates'] == other['selection']['candidates']
        assert all(c['observed_grid_route_consistent'] and not c['errors'] for c in g['candidate_routes'])
        cells = {tuple(s[:2]) for c in candidates for s in c['states']}
        anchor = tuple(g['return_anchor'])
        for c in candidates:
            validate(c['states'], c['actions'], cells)
            assert tuple(c['states'][0]) == tuple(c['states'][-1]) == anchor
            assert c['cost'] == len(c['actions'])
        targets = []
        for asset in g['marked_asset_indices']:
            found = [c for c in candidates if c['group'] == f'asset_{asset}_aperture_view']
            assert len(found) == 1
            targets += found
        assert len(targets) == 2
        contexts = []
        windows = next(p['routes'] for p in window['parents'] if p['parent'] == parent)
        chosen = [('N_first_choice', n['selected'])] + [
            (f'observed_asset_{c["asset_index"]}_aperture', c) for c in targets]
        for role, route in chosen:
            entry = next(r for r in windows if r['role'] == role)
            assert route['candidate_id'] == entry['candidate_id']
            disclosure = entry['first_difference']
            if disclosure is None:
                contexts.append(dict(role=role, first_difference=None, choices=[])); continue
            k = disclosure['local_action_id']
            assert disclosure['on_outbound'] and k <= route['outbound_cost']
            assert route['states'][k] == disclosure['pose']
            assert disclosure['total_action_id'] == g['paid_prefix_actions']+k
            choices = []
            for target in targets:
                outward = path(disclosure['pose'], target['pose'], cells)
                homeward = path(target['pose'], anchor, cells)
                direct = path(anchor, target['pose'], cells)
                if any(p is None for p in (outward,homeward,direct)):
                    choices.append(dict(target_observed_asset_index=target['asset_index'], witness_found=False,
                                        infeasibility_proven=False)); continue
                out_s,out_a = outward; ret_s,ret_a = homeward
                states = [tuple(s) for s in route['states'][:k+1]]+out_s[1:]+ret_s[1:]
                actions = route['actions'][:k]+out_a+ret_a
                validate(states, actions, cells)
                assert states[0] == states[-1] == anchor
                assert states[k+len(out_a)] == tuple(target['pose'])
                whole = len(actions); direct_cost = len(direct[1])+len(ret_a)
                choices.append(dict(target_observed_asset_index=target['asset_index'], witness_found=True,
                    direct_safe_cell_cost=direct_cost, information_prefix_paid=k,
                    post_information_outbound_cost=len(out_a), return_cost=len(ret_a), whole_paid_cost=whole,
                    extra_vs_direct=whole-direct_cost, remaining_diagnostic_budget=g['remaining_diagnostic_budget'],
                    fits_with_5_margin=whole+5 <= g['remaining_diagnostic_budget'],
                    target_pose=target['pose'], target_arrival_index=k+len(out_a), states=states, actions=actions))
            contexts.append(dict(role=role, first_clean_difference_paid=k,
                                 information_pose=disclosure['pose'], choices=choices))
        output.append(dict(parent=parent, known_safe_route_cells=len(cells), contexts=contexts))
    return dict(status='complete', input_sha256=inputs, parents=output,
        scope='conditional cost upper-bound witnesses under the observed-grid route contract',
        assumes_geometric_identification_from_first_clean_difference=True,
        noisy_identification_or_task_success_measured=False, global_optimal_cost_claimed=False,
        actual_candidate_execution=False, new_physical_actions=0, new_sensor_frames=0,
        new_tsdf_fusions=0, semantic_efficacy_proven=False)


def signatures(result):
    return [(p['parent'],c['role'],t['target_observed_asset_index'],t['whole_paid_cost'],
             t['extra_vs_direct'],t['fits_with_5_margin'])
            for p in result['parents'] for c in p['contexts'] for t in c['choices']]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prototype', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), 'Fresh output required'
    assert shutil.disk_usage(args.output.parent).free >= 34*1024**2, '32 MiB reserve required'
    script_hash = sha(Path(__file__).resolve())
    prior = read(args.prototype)
    result = run()
    assert signatures(result) == signatures(prior), 'Reproduction differs from inspected prototype costs'
    result['prototype_reproduction'] = dict(path=str(args.prototype), sha256=sha(args.prototype),
        same_cost_signatures=True, prospective_validation=False,
        note='Prototype input receipt omitted the window result hash; this reproduction explicitly includes and verifies it.')
    result['script_sha256'] = script_hash
    for name, expected in result['input_sha256'].items():
        assert sha(ROOT/name) == expected
    assert sha(Path(__file__).resolve()) == script_hash
    args.output.mkdir()
    data = (json.dumps(result, ensure_ascii=False, indent=2)+'\n').encode()
    assert len(data) < 1024**2
    (args.output/'result.json').write_bytes(data)
    shutil.copyfile(Path(__file__).resolve(), args.output/'audit_script.py')
    (args.output/'artifact_hashes.json').write_text(json.dumps(
        {p.name:sha(p) for p in sorted(args.output.iterdir())}, indent=2)+'\n')
    print(json.dumps(dict(status='complete', output=str(args.output), rows=signatures(result)),ensure_ascii=False))


if __name__ == '__main__':
    main()
