"""Pure synthetic checks: no run directory, world or future outcome is read."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyze_competition_v8_1 import gate, averages, comparison, audit_outcome, SUMMARY_FIELDS

protocol = json.loads((HERE / 'competition_v9_validation_protocol.json').read_text())
analyzer_ast = ast.parse((HERE / 'analyze_competition_v9.py').read_text())
analyze_body = next(n.body for n in analyzer_ast.body if isinstance(n, ast.FunctionDef) and n.name == 'analyze')
def assignment_index(name):
    return next(i for i, n in enumerate(analyze_body) if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == name for t in n.targets))
original_gates = analyze_body[assignment_index('spec'):assignment_index('oracles')]
extracted_ast = ast.parse((HERE / 'extracted_numeric_gates.py').read_text()).body[0].body[:-1]
assert [ast.dump(n, include_attributes=False) for n in original_gates] == [
    ast.dump(n, include_attributes=False) for n in extracted_ast], 'numeric extraction differs from bound analyzer'
namespace = {'gate': gate, 'PRIMARY': ('new_area_m2', 'final_f1_05cm', 'final_joint_05cm')}
exec(compile((HERE / 'extracted_numeric_gates.py').read_text(), 'extracted_numeric_gates', 'exec'), namespace)
evaluate = namespace['prospective_gates_under_test']
parents = protocol['parent_order']
primary = namespace['PRIMARY']
metrics = (*primary, 'branch_joint_auc_05cm', 'final_coverage_2d')


def base():
    comparisons = {m: {'means': {k: 0.006 for k in metrics},
                      'per_parent': {p: {k: 0.006 for k in metrics} for p in parents}}
                   for m in ('G', 'O', 'N', 'X', 'M')}
    for m in comparisons:
        comparisons[m]['means']['new_area_m2'] = 2.1
    return {'protocol': protocol,
            'rows': [{'collision_count': 0, 'failure': None, 'original_target_reached': True,
                      'returned_to_anchor': True, 'full_original_route_completed': True} for _ in range(48)],
            'metadata': {'new_physical_branches': 48}, 'verification': gate(True),
            'methods': {'total': {m: {'means': {'new_area_m2': 10.}} for m in ('G', 'O', 'N')}},
            'comparisons': comparisons,
            'rate_comparison': {'means': {k: 0.01 for k in metrics}},
            'histories': [{'context': p, 'arrangement': a,
                           'families': {f: {'M_equals_G': True, 'choices': {'S': i + 4}}
                                        for f in ('rate', 'total')}}
                          for p in parents for i, a in enumerate(protocol['arrangement_order'])],
            'selected': {'total': {}, 'rate': {}}, 'parents': parents}


results = []
passed = evaluate(**base())
assert len(passed) == 36 and all(x['passed'] is True for x in passed.values())
results.append({'case': 'all_36_registered_numeric_gates_positive', 'status': 'passed'})


def changed(case, expected_gate, mutate):
    data = base()
    mutate(data)
    gates = evaluate(**data)
    assert gates[expected_gate]['passed'] is False, (case, gates[expected_gate])
    results.append({'case': case, 'expected_failed_gate': expected_gate, 'status': 'passed'})


changed('missing_branch', 'all_48_complete', lambda d: d['rows'].pop())
for key, value in [('collision_count', 1), ('failure', 'collision'), ('original_target_reached', False),
                   ('returned_to_anchor', False), ('full_original_route_completed', False)]:
    changed('safety_' + key, 'all_routes_collision_free_target_and_return',
            lambda d, k=key, v=value: d['rows'][0].update({k: v}))
for other in ('G', 'O', 'N'):
    for metric, value, expected in [('new_area_m2', 1.99, f'S_total_area_over_{other}'),
                                  ('final_f1_05cm', 0.00499, f'S_total_final_f1_05cm_over_{other}'),
                                  ('final_joint_05cm', 0.00499, f'S_total_final_joint_05cm_over_{other}'),
                                  ('branch_joint_auc_05cm', -0.00101, f'S_total_branch_joint_auc_05cm_over_{other}')]:
        changed(other + '_' + metric, expected,
                lambda d, o=other, k=metric, v=value: d['comparisons'][o]['means'].update({k: v}))
    for metric in primary:
        changed(other + '_parent_tie_' + metric, f'all_parents_S_{metric}_over_{other}',
                lambda d, o=other, k=metric: d['comparisons'][o]['per_parent']['Q2'].update({k: 0.}))
for metric in (*primary, 'branch_joint_auc_05cm'):
    value = -0.00101 if metric == 'branch_joint_auc_05cm' else -1e-10
    changed('total_rate_' + metric, 'total_vs_rate_' + metric,
            lambda d, k=metric, v=value: d['rate_comparison']['means'].update({k: v}))
for metric in primary:
    changed('semantic_swap_tie_' + metric, 'correct_S_over_X_all_primary',
            lambda d, k=metric: d['comparisons']['X']['means'].update({k: 0.}))
for family in ('total', 'rate'):
    changed('missing_fallback_' + family, 'missing_M_exact_G_both_families',
            lambda d, f=family: d['histories'][0]['families'][f].update(M_equals_G=False))
changed('coverage_boundary', 'coverage_noninferiority_to_N',
        lambda d: d['comparisons']['N']['means'].update(final_coverage_2d=-0.00101))
for index, parent in enumerate(parents):
    changed('no_choice_exchange_' + parent, parent + '_semantic_choice_exchange',
            lambda d, i=index: d['histories'][2 * i + 1]['families']['total']['choices'].update(S=4))
small = base()
small['methods']['total']['G']['means']['new_area_m2'] = 0.
small['comparisons']['G']['means']['new_area_m2'] = 0.999
assert not evaluate(**small)['S_total_area_over_G']['passed']
small['comparisons']['G']['means']['new_area_m2'] = 1.
assert evaluate(**small)['S_total_area_over_G']['passed']
results.append({'case': 'near_zero_comparator_absolute_gain_exact_boundary', 'status': 'passed'})

# Balanced two-arrangement, four-parent means must coincide with parent means.
rows = []
selected = {}
for index, p in enumerate(parents):
    for ai, a in enumerate(protocol['arrangement_order']):
        values = {key: 10. * index + ai for key in SUMMARY_FIELDS}
        row = dict(values, candidate_id=ai, failure=None, returned_to_anchor=True,
                   collision_count=0, original_target_reached=True, full_original_route_completed=True)
        rows.append(row)
        selected[(p, a)] = {'S': row, 'G': dict(row, **{k: row[k] - (index + 1) for k in SUMMARY_FIELDS})}
assert averages(rows, 8)['means']['new_area_m2'] == 15.5
contrast = comparison(selected, 'G', parents, protocol['arrangement_order'])
assert contrast['means']['new_area_m2'] == 2.5
assert [contrast['per_parent'][p]['new_area_m2'] for p in parents] == [1., 2., 3., 4.]
rows[0]['failure'] = 'simulated failure'
for index, row in enumerate(rows):
    row['f1_gain_05cm'] = index * 0.01
rows[0]['f1_gain_05cm'] = -0.1
assert abs(averages(rows, 8)['means']['f1_gain_05cm'] - 0.0225) < 1e-12
rows[0]['area_per_action'] = None
assert averages(rows, 8)['means']['area_per_action'] is None
assert averages(rows, 8)['defined_counts']['area_per_action'] == 7
results += [{'case': case, 'status': 'passed'} for case in (
    'balanced_parent_aggregation_and_differences', 'negative_failed_row_retained', 'undefined_rate_not_imputed_or_filtered')]

report = {'status': 'passed_pure_synthetic_numeric_gate_review', 'checks': len(results), 'cases': results,
          'gate_inventory': list(passed), 'future_worlds_constructed': 0, 'future_branch_results_read': 0,
          'tests_do_not_prove_real_performance_or_full_integrity': True,
          'reviewed_analyzer_sha256': hashlib.sha256((HERE / 'analyze_competition_v9.py').read_bytes()).hexdigest(),
          'reviewed_protocol_sha256': hashlib.sha256((HERE / 'competition_v9_validation_protocol.json').read_bytes()).hexdigest()}
(HERE / 'numeric_gate_checks.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({k: v for k, v in report.items() if k not in ('cases', 'gate_inventory')}, indent=2))
