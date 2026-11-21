"""Recompute all native selection rankings and check swapped-view score parity."""

import argparse
import hashlib
import json
from pathlib import Path


def read(p):
    return json.loads(Path(p).read_text())


def canonical(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, allow_nan=False).encode()).hexdigest()


def event(attempt):
    trace = read(attempt / 'steps.json')
    decisions = read(attempt / 'choices.json')
    record = read(attempt / 'intervention.json')['intervention']
    step = len(trace) if record is None else record['paid_before']
    prefix_hash = canonical([s['packet_sha256'] for s in trace[:step]])
    if record is None:
        return dict(prefix_hash=prefix_hash, selected=None, diagnostics=None)

    plan = next(r['outputs'] for r in decisions if r['outputs']['plan_number'] == record['native_plan_number'])
    assert record['score_set_sha256'] == canonical(plan['scores'])
    assert record['candidate_set_sha256'] == canonical(plan['candidates'])
    candidates = plan['candidates']
    scoring = {r['candidate_id']: r for r in record['candidate_summaries']}
    mode_winners = {}
    for mode in ('S','G','X'):
        scores = plan['scores'][mode]
        order = sorted(range(len(scores)), key=lambda i: (-scores[i], candidates[i]['cost'], candidates[i]['candidate_id']))
        for rank, index in enumerate(order, 1):
            row = scoring[candidates[index]['candidate_id']]
            assert row['ranks'][mode] == rank and row['scores'][mode] == scores[index]
        mode_winners[mode] = candidates[order[0]]['candidate_id'] if scores and scores[order[0]] > 0 else None
    assert record['chosen_candidate_id'] == mode_winners[plan['mode']]
    assert all(r['X'] == r['S'][::-1] and r['G'] == [.5,.5] for r in record['shape_priors'])
    return dict(prefix_hash=prefix_hash, selected=record['chosen_candidate_id'],
                diagnostics=dict(winners=mode_winners, scores=plan['scores'],
                                 candidate_set_sha256=record['candidate_set_sha256']))


def audit(rows):
    groups = {}
    for row in rows:
        if row.get('J') is not None:
            groups.setdefault(row['layout'], {})[row['arm']] = event(Path(row['path']))
    parent_parity = {}
    for layout, arms in groups.items():
        assert len({a['prefix_hash'] for a in arms.values()}) == 1, layout
    for parent in ('warehouse', 'gallery'):
        lgroup, rgroup = (groups.get(f'{parent}_{side}', {}) for side in ('left','right'))
        a = lgroup.get('G', {}).get('diagnostics')
        b = rgroup.get('G', {}).get('diagnostics')
        if a is None or b is None:
            parent_parity[parent] = None
            continue
        s, t = a['scores'], b['scores']
        parent_parity[parent] = dict(same_candidates=a['candidate_set_sha256'] == b['candidate_set_sha256'],
                                     same_G=s['G'] == t['G'], same_common=s['N'] == t['N'],
                                     left_S_equals_right_X=s['S'] == t['X'], left_X_equals_right_S=s['X'] == t['S'])
    return dict(per_layout=groups, swapped_parent_parity=parent_parity)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('endpoint_rows', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    result = audit(read(args.endpoint_rows))
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
