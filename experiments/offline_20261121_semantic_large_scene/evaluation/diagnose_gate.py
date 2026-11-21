"""Describe whether the frozen selector had a diagnostic route and ranked it."""

import argparse
import json
from pathlib import Path


def diagnose(rows):
    out = {}
    for layout in dict.fromkeys(row['layout'] for row in rows):
        arms = {r['arm']: r for r in rows if r['layout'] == layout}
        if set(arms) != {'S', 'G', 'X'}:
            raise ValueError(f'Expected S/G/X attempts in {layout}')
        s, g, x = (arms[a] for a in ('S', 'G', 'X'))
        available = sorted(int(cid) for cid, side in s.get('route_sides_by_candidate', {}).items()
                           if side == s['informative_side'])
        qualified = all(r.get('triggered', False) for r in arms.values())
        all_measured = all(r.get('J') is not None for r in arms.values())
        first_ids = {arm: r.get('selected_id') for arm, r in arms.items()}
        contrast = (dict(S_minus_G=s['J']-g['J'], S_minus_X=s['J']-x['J'],
                         cover_S_minus_G=s['C_free']-g['C_free'], F1_S_minus_G=s['F1_scene_05cm']-g['F1_scene_05cm'])
                    if all_measured else None)
        if not qualified:
            reason = 'junction_not_triggered_in_all_arms'
        elif not available:
            reason = 'no_native_candidate_reached_predeclared_diagnostic_gate'
        elif len(set(first_ids.values())) == 1:
            reason = 'diagnostic_route_available_but_category_did_not_change_selection'
        elif s.get('selected_side') != s['informative_side']:
            reason = 'selection_changed_without_choosing_informative_branch'
        elif s.get('executed_first_crossing') != s['informative_side']:
            reason = 'planned_informative_crossing_not_first_executed_crossing'
        elif not all_measured:
            reason = 'terminal_evidence_missing'
        elif contrast['S_minus_G'] <= 1e-9 and contrast['S_minus_X'] <= 1e-9:
            reason = 'changed_informative_route_without_higher_terminal_J'
        else:
            reason = 'see_parent_level_paired_decision'
        out[layout] = dict(reason=reason, informative_side=s['informative_side'], qualifying_native_candidates=available,
                           selected_ids=first_ids, selected_sides={a: r.get('selected_side') for a, r in arms.items()},
                           later_executed_crossings={a: r.get('executed_first_crossing') for a, r in arms.items()},
                           outcomes=contrast, stops={a: r.get('stop_reason') for a, r in arms.items()})
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('rows', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    report = diagnose(json.loads(args.rows.read_text()))
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
