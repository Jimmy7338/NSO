"""Exercise the actual per-family selection block with pure synthetic scores."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyze_competition_v8_1 import require, finite

source = (HERE / 'analyze_competition_v9_1.py').read_text()
tree = ast.parse(source)
block = next(n for n in ast.walk(tree) if isinstance(n, ast.For)
             and isinstance(n.target, ast.Tuple)
             and [getattr(x, 'id', '') for x in n.target.elts] == ['family', 'sf', 'cf'])
compiled = compile(ast.fix_missing_locations(ast.Module(body=[block], type_ignores=[])), 'actual_family_selection', 'exec')
protocol = json.loads((HERE / 'competition_v9_1_validation_protocol.json').read_text())
methods = ('G', 'O', 'S', 'N', 'X', 'M')
checks = []


def fixture(parent, profile):
    ids = protocol['candidate_ids_by_parent'][parent]
    costs = [22, 2, 40, 40, 42, 42][:len(ids)]
    rates = [1.] * len(ids)
    if profile == 'last_id_wins':
        rates[-1] = 100.
    totals = [r * c for r, c in zip(rates, costs)]
    prediction = {'rate_scores': {m: rates[:] for m in methods}, 'scores': {m: totals[:] for m in methods},
                  'rate_selected_candidate_ids': {}, 'selected_candidate_ids': {}}
    for field, vector in (('rate_selected_candidate_ids', rates), ('selected_candidate_ids', totals)):
        # Independently rank score ties by declared real costs and then IDs.
        best = sorted(zip(ids, vector, costs), key=lambda x: (-x[1], x[2], x[0]))[0][0]
        prediction[field] = {m: best for m in methods}
    return dict(require=require, finite=finite, METHODS=methods, ids=ids,
                routes=[{'candidate_id': i, 'cost': c} for i, c in zip(ids, costs)],
                prediction=prediction, selected={'rate': {}, 'total': {}},
                parent=parent, arrangement='synthetic', table={i: {'candidate_id': i} for i in ids},
                history={'families': {}})


for parent in protocol['parent_order']:
    for profile in ('ties_cost_then_id', 'last_id_wins'):
        context = fixture(parent, profile)
        exec(compiled, context)
        for family, field in (('total', 'selected_candidate_ids'), ('rate', 'rate_selected_candidate_ids')):
            for mode in methods:
                assert context['selected'][family][(parent, 'synthetic')][mode]['candidate_id'] == context['prediction'][field][mode]
        checks.append({'parent': parent, 'profile': profile, 'candidate_count': len(context['ids']), 'status': 'passed'})
bad = fixture('Q3', 'ties_cost_then_id')
bad['prediction']['scores']['S'] += [9., 9.]
try:
    exec(compiled, bad)
except ValueError as error:
    assert 'invalid score vector' in str(error)
else:
    raise AssertionError('four-route parent accepted leftover six-score vector')
report = {'status': 'passed_actual_variable_pool_selection_block', 'profiles': checks,
          'method_family_choice_checks': 96, 'leftover_six_score_vector_on_four_route_parent_rejected': True,
          'future_worlds_constructed': 0, 'future_branch_outcomes_read': 0,
          'analyzer_sha256': hashlib.sha256(source.encode()).hexdigest()}
(HERE / 'variable_pool_checks.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({k: v for k, v in report.items() if k != 'profiles'}))
