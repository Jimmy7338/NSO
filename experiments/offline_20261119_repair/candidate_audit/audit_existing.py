"""Read existing completed episodes only. No actor/world/source imports."""
from pathlib import Path
import collections
import hashlib
import json
import math

ROOT = Path('/home/ouc/nso_robot/analyses/dual_template_exploration_20261119')
OUT = Path(__file__).resolve().parent
SOURCE = Path('/home/ouc/nso_robot/analyses/reproduction_20261104/simulation_publication_stage/experiments/unknown_exploration_20261104/source')
inputs = {}


def read(path):
    raw = path.read_bytes()
    inputs[str(path)] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)


result = dict(schema='existing_candidate_mechanism_audit_v1', runs_reexecuted=0,
              candidate_or_score_changes=0, rows=[], layouts={}, source_sha256={})
for layout in ['shelf_left', 'shelf_right']:
    run = ROOT / 'layouts' / layout / 'runs' / 'S' / 'attempt_001'
    choices = read(run / 'choices.json')
    gchoices = read(run.parent.parent / 'G' / 'attempt_001' / 'choices.json')
    steps = read(run / 'steps.json')
    templates = read(run / 'template_audit.json')
    assert len(choices) == len(gchoices) == len(templates)
    stats = collections.Counter()
    headings = collections.Counter()
    discretionary_reasons = collections.Counter()
    for choice, gchoice, template in zip(choices, gchoices, templates):
        o = choice['outputs']; a = o['candidate_audit']; scores = o['scores']
        assert scores == gchoice['outputs']['scores']
        assert o['candidates'] == gchoice['outputs']['candidates']
        paid = template['mapper_frames'] - 1
        assert paid == 256 - a['remaining_budget']
        slack = a['remaining_budget'] - a['minimum_current_return_cost']
        count = len(o['candidates'])
        max_unknown = max((v.get('radar_unknown_cells', 0) + v.get('camera_unknown_cells', 0)
                           for v in a['common_role_audit']), default=0)
        all_template_zero = all(abs(v) < 1e-10 for row in o['score_audit']
                                for v in row['template_expected_visible_area_m2'].values())
        semantic_diff = any(abs(s - g) > 1e-10 for s, g in zip(scores['S'], scores['G']))
        stats['plans'] += 1
        stats['single_candidate'] += count == 1
        stats['single_candidate_zero_template'] += count == 1 and all_template_zero
        stats['no_detour_slack'] += slack == 0
        stats['single_with_positive_slack'] += count == 1 and slack > 0
        stats['topology_filter_removed'] += a['candidate_count'] - count
        stats['candidate_cap_states'] += bool(a['omitted_by_cap'])
        stats['candidate_cap_role_omissions'] += len(a['omitted_by_cap'])
        stats['no_positive_coverage_proxy_states'] += max_unknown == 0
        if slack > 0:
            stats['discretionary_states'] += 1
            stats['discretionary_semantic_diff_states'] += semantic_diff
            discretionary_reasons.update(t['reason'] for t in a['target_audit'])
        for candidate in o['candidates']:
            if candidate['group'].startswith('asset_'):
                headings[candidate['pose'][2]] += 1
        selected = o['selected']
        result['rows'].append(dict(layout=layout, paid=paid, budget=a['remaining_budget'],
            return_cost=a['minimum_current_return_cost'], detour_slack=slack,
            feasible_oriented_states=a['feasible_states'], final_candidates=count,
            raw_candidates=a['candidate_count'], max_coverage_proxy_cells=max_unknown,
            all_template_zero=all_template_zero, semantic_score_difference=semantic_diff,
            observed_assets=a['observed_asset_count'], current_pose=a['current_pose'],
            coverage_fraction=steps[paid - 1]['coverage_fraction'] if paid else 582 / 1366,
            selected_group=selected['group'] if selected else None,
            selected_pose=selected['pose'] if selected else None,
            target_reasons=dict(collections.Counter(t['reason'] for t in a['target_audit'])),
            cap_omissions=a['omitted_by_cap']))
    bearings = {}
    # Fixed scene centers are evaluator references only, never input to policy.
    # Count real rendered positive-class observations; repeated frames are not independent views.
    for cls in ['2', '3']:
        x = 2.2 if (layout == 'shelf_left') == (cls == '2') else 7.8
        y = 5.2
        counts = collections.Counter(); first = {}
        for step in steps:
            if step['partial_semantic_pixels'][cls] <= 0:
                continue
            row, col = step['position']
            dx = (col + .5) * .2 - x
            dy = (40 - row - .5) * .2 - y
            angle = (math.atan2(dy, dx) + math.pi / 8) % (2 * math.pi)
            sector = ['E', 'NE', 'N', 'NW', 'W', 'SW', 'S', 'SE'][int(angle / (math.pi / 4))]
            counts[sector] += 1
            first.setdefault(sector, step['step'])
        bearings[cls] = dict(fixed_evaluation_center=[x, y], counts=dict(counts), first_action=first)
    result['layouts'][layout] = dict(counts=dict(stats), object_endpoint_headings=dict(headings),
        discretionary_target_reasons=dict(discretionary_reasons),
        actual_positive_semantic_observation_bearings=bearings)

for rel in ['nso_bridge/exploration_joint_policy.py', 'nso/competition_candidates_v8_1.py',
            'nso/semantic_completion_v3.py', 'nso/camera_joint_planner_v2.py']:
    result['source_sha256'][rel] = hashlib.sha256((SOURCE / rel).read_bytes()).hexdigest()
result['inputs_sha256'] = inputs
result['limits'] = [
    'Two paired layouts of one floorplan, not 51 independent scenes or trials.',
    'Current evidence identifies endpoint proposals and actual rendered class-positive views; it does not establish useful unseen surface at an omitted alternative pose.',
    'no_unused_feasible_aperture_pose pools budget, heading, role reuse and hypothetical aperture tests; the saved audit cannot isolate those subconditions without fixed-state replay.',
    'The >0 slack subset is a post hoc structural diagnosis, not a replacement for all 51 reported planning states.',
    'Candidate endpoints are behind the measured object; route-integrated gain includes all outbound poses and real execution already observed side/front views.',
]
result['script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
out = OUT / 'evidence.json'
with out.open('x') as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
print(json.dumps(dict(output=str(out), layouts=result['layouts']), ensure_ascii=False))
