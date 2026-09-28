#!/usr/bin/env python3
"""Independent, read-only V9.1 metric arithmetic audit after complete replay.

No production analyzer/helper, mapper, planner, world or evaluator is imported.
Global P/R and old-support distances come from independently replayed records;
this audit independently checks arithmetic, coverage maps, bitset areas, costs,
selection, aggregation and progression decisions. It does not rerender geometry.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_NAME = 'configs/virtual3d/competition_v9_1_validation_protocol.json'
PROTOCOL_SHA = '8dba14b72f91033fc3ad48b8e23c6c5ca80886f985756ff792fbf9e6be000430'
METHODS = ('G', 'O', 'S', 'N', 'X', 'M')
FAMILIES = ('total', 'rate')
PRIMARY = ('new_area_m2', 'final_f1_05cm', 'final_joint_05cm')
FIELDS = ('new_area_m2', 'area_per_action', 'f1_gain_05cm', 'f1_gain_per_action',
          'coverage_gain_m2', 'branch_joint_auc_02cm', 'branch_joint_auc_05cm',
          'paid_actions', 'planned_actions', 'prefix_paid_actions', 'task_paid_actions',
          'unused_branch_budget_actions', 'background_new_area_m2', 'object_new_area_m2',
          'area_rate_regret', 'final_f1_05cm', 'final_coverage_2d', 'final_joint_05cm',
          'fixed_old_support_error_mean_change_m', 'fixed_old_support_error_p95_change_m')
EXTRA = ('final_covered_area_m2', 'final_area_times_f1_02cm', 'final_area_times_f1_05cm',
         'final_f1_02cm', 'final_precision_02cm', 'final_precision_05cm',
         'final_recall_02cm', 'final_recall_05cm')


def insist(value, message):
    if not value:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            insist(key not in result, 'duplicate JSON key')
            result[key] = value
        return result
    def nonfinite(text):
        raise ValueError('nonfinite JSON literal: ' + text)
    return json.loads(Path(path).read_text(), object_pairs_hook=pairs, parse_constant=nonfinite)


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def mean(values):
    values = list(values)
    return statistics.fmean(values) if values and all(finite(v) for v in values) else None


def difference(a, b):
    return a - b if finite(a) and finite(b) else None


class Auditor:
    def __init__(self):
        self.inputs = {}
        self.numeric_checks = 0
        self.maximum_absolute_error = 0.

    def note(self, path):
        path = Path(path).resolve()
        insist(path.is_relative_to(ROOT), 'input outside NSO workspace')
        current = digest(path)
        insist(str(path) not in self.inputs or self.inputs[str(path)] == current, 'input changed during audit')
        self.inputs[str(path)] = current
        return current

    def read(self, path):
        self.note(path)
        return read_json(path)

    def manifest(self, folder):
        folder = folder.resolve()
        manifest = self.read(folder / 'artifact_hashes.json')
        insist(isinstance(manifest, dict) and manifest, 'missing full manifest')
        for name, expected in manifest.items():
            relative = Path(name)
            insist(not relative.is_absolute() and '..' not in relative.parts, 'unsafe manifest path')
            path = folder / relative
            insist(path.is_file() and self.note(path) == expected, 'sealed file absent or changed: ' + name)
        return manifest

    def equal(self, observed, expected, label):
        if expected is None:
            insist(observed is None, label + ': undefined value was imputed')
        elif isinstance(expected, bool):
            insist(type(observed) is bool and observed == expected, label + ': boolean mismatch')
        elif isinstance(expected, (str, list, dict)):
            insist(observed == expected, label + ': exact value mismatch')
        else:
            insist(finite(observed) and finite(expected), label + ': missing finite scalar')
            error = abs(observed - expected)
            self.maximum_absolute_error = max(self.maximum_absolute_error, error)
            self.numeric_checks += 1
            insist(math.isclose(observed, expected, rel_tol=1e-9, abs_tol=1e-9),
                    f'{label}: {observed!r} versus independently computed {expected!r}')


def held_auc(checkpoints, budget, tag):
    times = [row['action_index'] for row in checkpoints]
    insist(times and times[0] == 0 and 0 <= times[-1] <= budget
            and all(type(t) is int for t in times)
            and all(b > a for a, b in zip(times, times[1:])), 'invalid fixed-window checkpoints')
    values = [row['coverage_2d'] * row[f'f1_{tag}'] for row in checkpoints]
    # Integrate segments directly, rather than using the production integer interpolation.
    area = sum((b - a) * (x + y) * .5 for a, b, x, y in zip(times, times[1:], values, values[1:]))
    return (area + (budget - times[-1]) * values[-1]) / budget


def bitset_areas(reference, visibility):
    weights = reference['weights']
    prefix = reference['prefix_seen']
    insist(np.array_equal(visibility['prefix'], prefix), 'prefix visibility differs')
    union = np.zeros_like(prefix)
    seen = prefix.copy()
    increments = {}
    for name in ('outbound', 'endpoint', 'return'):
        mask = visibility[name]
        insist(mask.dtype == np.bool_ and mask.shape == prefix.shape, 'invalid stage mask')
        union |= mask
        increments[name] = float(weights[mask & ~seen].sum())
        seen |= mask
    insist(np.array_equal(visibility['union'], union), 'saved full union differs')
    new = union & ~prefix
    return {'new_area_m2': float(weights[new].sum()),
            'background_new_area_m2': float(weights[new & (reference['classes'] == 1)].sum()),
            'slot_new_area_m2': [float(weights[new & reference[key]].sum()) for key in ('west_slot', 'east_slot')],
            'slot_remaining_area_m2': [float(weights[~prefix & reference[key]].sum()) for key in ('west_slot', 'east_slot')],
            'stage_new_area_m2': increments}


def pick(ids, vector, routes):
    insist(len(vector) == len(ids) and all(finite(v) for v in vector), 'invalid score vector')
    by_id = {r['candidate_id']: r for r in routes}
    ranked = sorted(zip(ids, vector), key=lambda item: (-item[1], by_id[item[0]]['cost'], item[0]))
    return ranked[0][0]


def aggregate(selected, parents, arrangements, fields):
    parent_means = {p: {k: mean(selected[(p, a)][k] for a in arrangements) for k in fields} for p in parents}
    return {'means': {k: mean(parent_means[p][k] for p in parents) for k in fields},
            'per_parent': parent_means}


def contrast(first, second, parents, arrangements):
    rows = {key: {k: difference(first[key][k], second[key][k]) for k in FIELDS} for key in first}
    return aggregate(rows, parents, arrangements, FIELDS)


def numerical_gates(spec, means, comparisons, versus_rate, histories, rows, parents):
    gates = {'all_44_complete': len(rows) == 44,
             'independent_full_replay': True, 'source_structure_and_release_integrity': True,
             'all_routes_collision_free_target_and_return': all(
                 r['collision_count'] == 0 and r['failure'] is None and r['original_target_reached']
                 and r['returned_to_anchor'] and r['full_original_route_completed'] for r in rows)}
    for other in ('G', 'O', 'N'):
        delta = comparisons[other]['means']
        comparator_area = means['total'][other]['means']['new_area_m2']
        if comparator_area < spec['near_zero_comparator_area_m2']:
            passes = delta['new_area_m2'] >= spec['near_zero_minimum_absolute_area_gain_m2']
        else:
            passes = delta['new_area_m2'] >= comparator_area * spec['min_S_relative_mean_new_area_gain_against_each_total_comparator']
        gates[f'S_total_area_over_{other}'] = passes
        for metric, bound in (
            ('final_f1_05cm', 'min_S_minus_each_total_comparator_mean_global_F1_05m'),
            ('final_joint_05cm', 'min_S_minus_each_total_comparator_mean_C_times_F1_05m'),
            ('branch_joint_auc_05cm', 'minimum_mean_S_minus_each_total_comparator_sparse_joint_auc')):
            gates[f'S_total_{metric}_over_{other}'] = delta[metric] >= spec[bound]
        for metric in PRIMARY:
            gates[f'all_parents_S_{metric}_over_{other}'] = all(
                comparisons[other]['per_parent'][p][metric] > 0 for p in parents)
    for metric, bound in (
        ('new_area_m2', 'minimum_mean_S_total_minus_S_rate_new_area_m2'),
        ('final_f1_05cm', 'minimum_mean_S_total_minus_S_rate_global_F1'),
        ('final_joint_05cm', 'minimum_mean_S_total_minus_S_rate_C_times_F1'),
        ('branch_joint_auc_05cm', 'minimum_mean_S_total_minus_S_rate_sparse_joint_auc')):
        gates[f'total_vs_rate_{metric}'] = versus_rate['means'][metric] >= spec[bound]
    gates['correct_S_over_X_all_primary'] = all(comparisons['X']['means'][k] > 0 for k in PRIMARY)
    gates['missing_M_exact_G_both_families'] = all(h[f]['M_equals_G'] for h in histories.values() for f in FAMILIES)
    gates['coverage_noninferiority_to_N'] = comparisons['N']['means']['final_coverage_2d'] >= spec['minimum_mean_S_minus_N_coverage_2d']
    for p in parents:
        choices = [h['total']['choices']['S'] for (parent, _), h in histories.items() if parent == p]
        gates[p + '_semantic_choice_exchange'] = len(choices) == 2 and choices[0] != choices[1]
    return gates


def audit(args):
    run, analysis, output = args.run.resolve(), args.analysis.resolve(), args.output.resolve()
    insist(not output.exists() and not output.is_relative_to(run) and not output.is_relative_to(analysis),
            'fresh separate output directory required')
    auditor = Auditor()
    # Read no outcome until complete execution and full replay are authoritatively established.
    metadata = auditor.read(run / 'metadata.json')
    insist(metadata['status'] == 'complete_execution_pending_independent_replay'
            and metadata['new_physical_branches'] == 44, 'execution is incomplete; no outcomes read')
    verification = auditor.read(run / 'verification.json')
    insist(verification['status'] == 'passed_full' and verification['passed_full'] is True
            and verification['partial'] is False and verification['max_branches'] is None
            and verification['branches_checked'] == verification['branches_total'] == 44
            and verification['raw_hashes_rechecked_after_replay'] is True,
            'full replay is incomplete; no outcomes read')
    insist((analysis / 'artifact_hashes.json').is_file(), 'complete analyzer artifacts required before outcomes')
    own = digest(__file__)
    manifest = auditor.manifest(run)
    analysis_manifest = auditor.manifest(analysis)
    root_result = auditor.read(analysis / 'summary.json')
    seal = auditor.read(run / 'pre_execution_seal.json')
    auditor.equal(seal['source_sha256'], metadata['source_sha256'], 'execution source seal')
    for key, filename in (('artifact_manifest_sha256', 'artifact_hashes.json'),
                          ('source_archive_sha256', 'sources.zip'), ('pre_execution_seal_sha256', 'pre_execution_seal.json')):
        auditor.equal(verification[key], auditor.note(run / filename), 'replay ' + key)
    auditor.equal(verification['run'], str(run), 'replay run')
    auditor.equal(verification['structural_review_sha256'], seal['structural_review_sha256'], 'replay structure')
    auditor.equal(verification['prepared_manifest_sha256'], seal['prepared_manifest_sha256'], 'replay preparation')
    auditor.equal(verification['verifier_sha256'], metadata['source_sha256']['scripts/replay_competition_v9_1.py'], 'replay implementation')
    with zipfile.ZipFile(run / 'sources.zip') as archive:
        names = archive.namelist()
        insist(len(names) == len(set(names)) and set(names) == set(metadata['source_sha256']), 'archive inventory')
        for name, expected in metadata['source_sha256'].items():
            insist(hashlib.sha256(archive.read(name)).hexdigest() == expected, 'archived source changed')
        insist(metadata['source_sha256'][PROTOCOL_NAME] == PROTOCOL_SHA, 'wrong frozen protocol')
        protocol = json.loads(archive.read(PROTOCOL_NAME))
    auditor.equal(protocol, seal['protocol'], 'protocol copy')
    auditor.equal(root_result['protocol_sha256'], PROTOCOL_SHA, 'analysis protocol')
    auditor.equal(auditor.note(analysis / 'analysis_source.py'), metadata['source_sha256']['scripts/analyze_competition_v9_1.py'], 'analysis implementation')
    auditor.equal(auditor.note(analysis / 'shared_analysis_helpers.py'), metadata['source_sha256']['scripts/analyze_competition_v8_1.py'], 'analysis helper implementation')
    prepared = Path(metadata['prepared']).resolve()
    pmanifest = auditor.manifest(prepared)
    auditor.equal(auditor.note(prepared / 'artifact_hashes.json'), seal['prepared_manifest_sha256'], 'prepared manifest')
    pm = auditor.read(prepared / 'metadata.json')
    insist(pm['status'] == 'complete_structural_pass' and pm['new_candidate_branches'] == 0, 'preparation failed')
    for name, expected in pm['source_sha256'].items():
        auditor.equal(metadata['source_sha256'][name], expected, 'prepared source binding')
    for name, expected in seal['decision_assets'].items():
        insist(manifest.get(name) == pmanifest.get(name) == expected, 'decision asset changed')
    analysis_inputs = auditor.read(analysis / 'input_hashes.json')
    reviews = [Path(name) for name, value in analysis_inputs.items() if value == seal['structural_review_sha256']]
    insist(reviews, 'analysis does not bind a structural review input')
    for review in reviews:
        record = auditor.read(review)
        insist(record['release_allowed'] is True and record['prepared_manifest_sha256'] == seal['prepared_manifest_sha256'], 'invalid structure release')
    for name, expected in seal['guard_verification_sha256'].items():
        insist(auditor.note(ROOT / name) == expected and auditor.read(ROOT / name)['passed_full'] is True, 'guard evidence changed')
    parents, arrangements = protocol['parent_order'], protocol['arrangement_order']
    insist(parents == ['Q0', 'Q1', 'Q2', 'Q3'] and arrangements == ['shelf_west', 'shelf_east'], 'cohort changed')
    insist(protocol['branch_actions'] == 48 and protocol['prefix_paid_actions'] == 150, 'cost contract changed')
    expected_outcomes = {f'{p}/{a}/candidate_{i:03d}/outcome.json' for p in parents for a in arrangements
                         for i in protocol['candidate_ids_by_parent'][p]}
    insist(len(expected_outcomes) == 44 and {n for n in manifest if n.endswith('/outcome.json')} == expected_outcomes, 'incomplete or extra branch set')
    rows, tables, histories = [], {}, {}
    selected = {family: {method: {} for method in METHODS} for family in FAMILIES}
    paired_routes, paired_scores = {}, {}
    for parent in parents:
        ids = protocol['candidate_ids_by_parent'][parent]
        for arrangement in arrangements:
            key = (parent, arrangement)
            folder = run / parent / arrangement
            routes = auditor.read(folder / 'candidates.json')
            prediction = auditor.read(folder / 'predictions.json')
            insist([r['candidate_id'] for r in routes] == ids, 'route inventory/order')
            insist([r['group'] for r in routes] == protocol['candidate_roles_by_parent'][parent], 'route roles')
            paired_routes[key] = routes
            with np.load(folder / 'reference.npz', allow_pickle=False) as saved:
                reference = {name: saved[name].copy() for name in saved.files}
            n = len(reference['points'])
            weights = reference['weights']
            insist(weights.shape == (n,) and np.isfinite(weights).all() and (weights > 0).all(), 'invalid reference weights')
            for name in ('prefix_seen', 'west_slot', 'east_slot'):
                insist(reference[name].dtype == np.bool_ and reference[name].shape == (n,), 'invalid reference masks')
            insist(not np.any(reference['west_slot'] & reference['east_slot'])
                    and np.array_equal(reference['west_slot'] | reference['east_slot'], reference['classes'] != 1), 'slot/background partition')
            triangles = reference['vertices'][reference['triangles']]
            union_area = float((np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1) * .5).sum())
            insist(np.allclose(weights, union_area / 32000, rtol=1e-10, atol=1e-12), 'reference weights renormalized or changed')
            reachable_area = int(reference['reachable'].sum()) * .2 ** 2
            prefix_records = auditor.read(folder / 'prefix/records.json')
            insist([r['step'] for r in prefix_records] == list(range(151)), 'prefix acquisition ledger')
            with np.load(folder / 'prefix_map.npz', allow_pickle=False) as saved:
                prefix_coverage = np.count_nonzero((saved['belief'] != -1) & reference['reachable']) / reference['reachable'].sum()
            table = {}
            for route in routes:
                cid = route['candidate_id']; branch = folder / f'candidate_{cid:03d}'
                row = auditor.read(branch / 'outcome.json')
                actions = auditor.read(branch / 'actions.json')
                checkpoints = auditor.read(branch / 'metrics.json')
                with np.load(branch / 'visibility.npz', allow_pickle=False) as saved:
                    visibility = {name: saved[name].copy() for name in saved.files}
                computed = bitset_areas(reference, visibility)
                for name in ('new_area_m2', 'background_new_area_m2'):
                    auditor.equal(row[name], computed[name], str(branch) + '/' + name)
                for name in ('slot_new_area_m2', 'slot_remaining_area_m2'):
                    for i, value in enumerate(computed[name]):
                        auditor.equal(row[name][i], value, name + str(i))
                for i in range(2):
                    auditor.equal(row['slot_new_fractions'][i], computed['slot_new_area_m2'][i] / computed['slot_remaining_area_m2'][i], 'slot recovery fraction')
                for stage, value in computed['stage_new_area_m2'].items():
                    auditor.equal(row['stage_new_area_m2'][stage], value, 'deduplicated stage area')
                auditor.equal(row['new_area_m2'], sum(computed['slot_new_area_m2']) + computed['background_new_area_m2'], 'area partition')
                paid = len(actions); planned = len(route['actions'])
                insist(0 <= paid <= 48 and 1 <= planned == route['cost'] <= 48, 'paid/planned budget')
                auditor.equal(row['paid_actions'], paid, 'paid action count')
                auditor.equal(row['planned_actions'], planned, 'planned action count')
                auditor.equal(row['prefix_paid_actions'], 150, 'prefix cost')
                auditor.equal(row['task_paid_actions'], 150 + paid, 'total actual task cost')
                insist([a['action_index'] for a in actions] == list(range(1, paid + 1))
                        and all(a['absolute_step'] == 150 + a['action_index'] for a in actions), 'paid timestamp ledger')
                auditor.equal(row['collision_count'], sum(int(a['collision']) for a in actions), 'collision count')
                for field, value in (('context', parent), ('arrangement', arrangement), ('candidate_id', cid), ('group', route['group'])):
                    auditor.equal(row[field], value, 'branch identity/' + field)
                actual_states = [[*a['position'], a['heading']] for a in actions]
                target_actions = [i for i, state in enumerate(actual_states, 1) if state == route['pose']]
                auditor.equal(row['original_target_reached'], bool(target_actions), 'actual target visit')
                auditor.equal(row['actual_arrival_action'], target_actions[0] if target_actions else None, 'actual first arrival')
                if row['returned_to_anchor']:
                    terminal = actual_states[-1] if actual_states else [*prefix_records[-1]['position'], prefix_records[-1]['heading']]
                    insist(terminal == route['states'][0], 'claimed return lost original position/heading')
                if row['full_original_route_completed']:
                    insist(paid == planned and [a['action'] for a in actions] == route['actions']
                            and actual_states == route['states'][1:], 'claimed original route incomplete')
                before, after = row['before'], row['after']
                auditor.equal(before['coverage_2d'], float(prefix_coverage), 'prefix C from observed map')
                with np.load(branch / 'final_map.npz', allow_pickle=False) as saved:
                    final_coverage = np.count_nonzero((saved['belief'] != -1) & reference['reachable']) / reference['reachable'].sum()
                auditor.equal(after['coverage_2d'], float(final_coverage), 'terminal C from observed map')
                insist(checkpoints[0]['action_index'] == 0 and checkpoints[-1]['action_index'] == paid, 'checkpoint terminal cost')
                auditor.equal({k: v for k, v in checkpoints[0].items() if k != 'action_index'}, before, 'initial checkpoint')
                auditor.equal({k: v for k, v in checkpoints[-1].items() if k != 'action_index'}, after, 'terminal checkpoint')
                for state in checkpoints:
                    auditor.equal(state['covered_area_m2'], state['coverage_2d'] * reachable_area, 'covered 2D area')
                    for tag in ('02cm', '05cm'):
                        precision, recall = state['precision_' + tag], state['recall_' + tag]
                        insist(0 <= precision <= 1 and 0 <= recall <= 1, 'invalid precision/recall')
                        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.
                        auditor.equal(state['f1_' + tag], f1, 'F1 harmonic mean')
                        auditor.equal(state['joint_' + tag], state['coverage_2d'] * f1, 'C times F1')
                        auditor.equal(state['area_times_f1_' + tag], state['covered_area_m2'] * f1, 'area times F1')
                for tag in ('02cm', '05cm'):
                    auditor.equal(row['branch_joint_auc_' + tag], held_auc(checkpoints, 48, tag), 'direct-segment 48-action AUC')
                f1_gain = after['f1_05cm'] - before['f1_05cm']
                auditor.equal(row['f1_gain_05cm'], f1_gain, 'F1 gain')
                auditor.equal(row['f1_gain_per_action'], f1_gain / paid if paid else None, 'F1 actual cost rate')
                auditor.equal(row['area_per_action'], computed['new_area_m2'] / paid if paid else None, 'area actual cost rate')
                auditor.equal(row['coverage_gain_m2'], (final_coverage - prefix_coverage) * reachable_area, '2D area gain')
                derived = dict(row)
                derived.update(unused_branch_budget_actions=48 - paid, object_new_area_m2=sum(computed['slot_new_area_m2']),
                               area_rate_regret=None, final_f1_05cm=after['f1_05cm'], final_coverage_2d=after['coverage_2d'],
                               final_joint_05cm=after['joint_05cm'])
                for statistic in ('mean', 'p95'):
                    name = f'fixed_old_support_error_{statistic}_m'
                    derived[f'fixed_old_support_error_{statistic}_change_m'] = difference(after.get(name), before.get(name))
                for field in EXTRA:
                    derived[field] = after[field.removeprefix('final_')]
                table[cid] = derived; rows.append(derived)
            insist(all(row['before'] == table[0]['before'] for row in table.values()), 'different initial reconstruction')
            tables[key] = table; histories[key] = {}; paired_scores[key] = {}
            for family, score_name, choice_name in (('total', 'scores', 'selected_candidate_ids'),
                                                     ('rate', 'rate_scores', 'rate_selected_candidate_ids')):
                scores, choices = prediction[score_name], prediction[choice_name]
                insist(set(scores) == set(choices) == set(METHODS), 'missing method or score family')
                for method in METHODS:
                    choice = pick(ids, scores[method], routes)
                    auditor.equal(choices[method], choice, 'sealed score tie choice')
                    if family == 'total':
                        auditor.equal(scores[method], [v * r['cost'] for v, r in zip(prediction['rate_scores'][method], routes)], 'total=rate*planned cost')
                    selected[family][method][key] = table[choice]
                histories[key][family] = {'choices': choices, 'M_equals_G': scores['M'] == scores['G'] and choices['M'] == choices['G']}
                paired_scores[key][family] = scores
    for parent in parents:
        west, east = [(parent, a) for a in arrangements]
        auditor.equal(paired_routes[west], paired_routes[east], 'paired exact candidate pool')
        for family in FAMILIES:
            a, b = paired_scores[west][family], paired_scores[east][family]
            for method in ('G', 'O', 'N', 'M'):
                auditor.equal(a[method], b[method], 'paired category-independent score')
            auditor.equal(a['S'], b['X'], 'paired S/X swap')
            auditor.equal(a['X'], b['S'], 'paired X/S swap')
    method_results = {f: {m: aggregate(selected[f][m], parents, arrangements, FIELDS + EXTRA) for m in METHODS} for f in FAMILIES}
    for family in FAMILIES:
        for method in METHODS:
            expected = root_result['method_averages'][family][method]
            auditor.equal(expected['histories_available'], 8, 'complete selected histories')
            auditor.equal(expected['histories_expected'], 8, 'expected selected histories')
            for field, value in method_results[family][method]['means'].items():
                auditor.equal(expected['means'][field], value, f'{family}/{method}/mean/{field}')
            for field in FIELDS:
                auditor.equal(expected['defined_counts'][field], sum(finite(r[field]) for r in selected[family][method].values()), 'undefined metric counts')
            chosen = list(selected[family][method].values())
            counters = {'failures': sum(r['failure'] is not None for r in chosen),
                        'not_returned': sum(not r['returned_to_anchor'] for r in chosen),
                        'collisions': sum(r['collision_count'] for r in chosen),
                        'original_targets_missed': sum(not r['original_target_reached'] for r in chosen),
                        'original_routes_incomplete': sum(not r['full_original_route_completed'] for r in chosen)}
            for name, value in counters.items():
                auditor.equal(expected[name], value, 'method failure accounting')
    comparisons = {m: contrast(selected['total']['S'], selected['total'][m], parents, arrangements) for m in ('G', 'O', 'N', 'X', 'M')}
    versus_rate = contrast(selected['total']['S'], selected['rate']['S'], parents, arrangements)
    # Complete-pool upper bounds are diagnostics only: never replace a method,
    # select a new route, remove a parent, or amend the registered gates.
    oracle_histories = {}
    paired_oracles = {}
    for parent in parents:
        ids = protocol['candidate_ids_by_parent'][parent]
        left, right = [(parent, a) for a in arrangements]
        paired_oracles[parent] = {}
        for metric in (*PRIMARY, 'branch_joint_auc_05cm'):
            first = [tables[left][i][metric] for i in ids]
            second = [tables[right][i][metric] for i in ids]
            conditional = (max(first) + max(second)) / 2
            marginal = max((a + b) / 2 for a, b in zip(first, second))
            values = {'conditional_oracle': conditional, 'marginal_oracle': marginal, 'value': conditional - marginal}
            for name, value in values.items():
                auditor.equal(root_result['paired_oracles'][parent][metric][name], value, 'complete paired oracle/' + parent + '/' + metric + '/' + name)
            paired_oracles[parent][metric] = values
        for arrangement in arrangements:
            key = (parent, arrangement)
            top = max(tables[key][i]['final_f1_05cm'] for i in ids)
            best_ids = [i for i in ids if top - tables[key][i]['final_f1_05cm'] <= 1e-10]
            s = selected['total']['S'][key]
            n = selected['total']['N'][key]
            oracle_histories[parent + '/' + arrangement] = {
                'complete_candidate_ids': ids, 'best_F1_candidate_ids': best_ids,
                'oracle_terminal_F1_05cm': top, 'S_total_candidate_id': s['candidate_id'],
                'S_total_F1_regret': top - s['final_f1_05cm'],
                'S_total_is_full_pool_F1_maximizer': s['candidate_id'] in best_ids,
                'N_total_terminal_F1_05cm': n['final_f1_05cm'],
                'oracle_minus_N_total_F1': top - n['final_f1_05cm']}
    oracle_parent_means = {p: {field: mean(oracle_histories[p + '/' + a][field] for a in arrangements)
                              for field in ('oracle_terminal_F1_05cm', 'S_total_F1_regret', 'oracle_minus_N_total_F1')}
                           for p in parents}
    oracle_means = {field: mean(oracle_parent_means[p][field] for p in parents)
                    for field in ('oracle_terminal_F1_05cm', 'S_total_F1_regret', 'oracle_minus_N_total_F1')}
    original_f1_requirement = protocol['progression_necessary_conditions']['min_S_minus_each_total_comparator_mean_global_F1_05m']
    f1_upper_bound = {'histories': oracle_histories, 'per_parent': oracle_parent_means, 'means': oracle_means,
                     'original_mean_F1_margin_required': original_f1_requirement,
                     'oracle_N_margin_shortfall': original_f1_requirement - oracle_means['oracle_minus_N_total_F1'],
                     'complete_pool_any_selector_cannot_meet_N_F1_margin': oracle_means['oracle_minus_N_total_F1'] < original_f1_requirement,
                     'S_is_full_pool_F1_maximizer_in_every_history': all(r['S_total_is_full_pool_F1_maximizer'] for r in oracle_histories.values()),
                     'scope': 'upper bound for selectors of exactly this frozen finite pool under the same equal-history endpoint metric',
                     'diagnostic_only_not_policy_or_new_gate': True, 'original_failed_gates_changed': False}
    for key, values in list(comparisons.items()) + [('rate', versus_rate)]:
        expected = root_result['S_total_vs_S_rate'] if key == 'rate' else root_result['S_total_comparisons'][key]
        for field in FIELDS:
            auditor.equal(expected['means'][field], values['means'][field], 'mean paired difference/' + key + '/' + field)
            for parent in parents:
                auditor.equal(expected['per_parent'][parent][field], values['per_parent'][parent][field], 'parent difference/' + parent + '/' + key + '/' + field)
    gates = numerical_gates(protocol['progression_necessary_conditions'], method_results, comparisons, versus_rate, histories, rows, parents)
    insist(set(root_result['gates']) == set(gates), 'gate inventory differs')
    for name, passed in gates.items():
        auditor.equal(root_result['gates'][name]['passed'], passed, 'progression gate/' + name)
    failed = [name for name, passed in gates.items() if not passed]
    auditor.equal(sorted(root_result['failed_gates']), sorted(failed), 'all failed gates retained')
    auditor.equal(root_result['pending_gates'], [], 'no pending full replay gate')
    auditor.equal(root_result['positive_finite_mechanism_confirmation_allowed'], not failed, 'finite progression conclusion')
    auditor.equal(root_result['paid_branch_actions'], sum(r['paid_actions'] for r in rows), 'all counterfactual paid actions')
    auditor.equal(metadata['paid_actions'], sum(r['paid_actions'] for r in rows), 'acquisition cost total')
    auditor.equal(root_result['unique_prefix_paid_actions'], 1200, 'eight paid prefix costs')
    auditor.equal(root_result['independent_geometry_blocks'], 4, 'independent geometry blocks')
    auditor.equal(root_result['minimum_one_sided_sign_p'], 1 / (2 ** 4), 'four-block significance bound')
    auditor.equal(root_result['all_prefix_coverage_equals_one'], all(r['before']['coverage_2d'] == 1 for r in rows), 'F1/C-F1 redundancy disclosure')
    for name, expected in auditor.inputs.items():
        insist(digest(name) == expected, 'input changed before report')
    insist(digest(__file__) == own, 'independent auditor source changed')
    output.mkdir(parents=True)
    result = {'status': 'passed_independent_metric_arithmetic_and_complete_pool_audit',
              'run': str(run), 'analysis': str(analysis), 'protocol_sha256': PROTOCOL_SHA,
              'branches': len(rows), 'histories': len(histories), 'parent_blocks': 4,
              'numeric_checks': auditor.numeric_checks, 'maximum_absolute_arithmetic_difference': auditor.maximum_absolute_error,
              'all_12_method_means_match': True, 'all_parent_differences_match': True,
              'bitset_area_partitions_match': True, 'direct_segment_48_action_auc_matches': True,
              'all_progression_decisions_match': True, 'method_results': method_results,
              'comparisons': comparisons, 'total_vs_rate': versus_rate, 'gates': gates, 'failed_gates': failed,
              'paired_oracles': paired_oracles, 'complete_pool_F1_upper_bound': f1_upper_bound,
              'positive_finite_mechanism_confirmation_allowed': not failed,
              'independent_world_or_geometry_reconstruction_performed': False,
              'production_analyzer_helper_mapper_planner_imported': False,
              'global_precision_recall_and_old_support_distances_source': 'previously completed independently replayed records; arithmetic checked here',
              'input_files_modified': False, 'auditor_sha256': own}
    for name, value in (('verification.json', result), ('input_hashes.json', auditor.inputs)):
        (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    (output / 'auditor_source.py').write_bytes(Path(__file__).read_bytes())
    (output / 'REPORT.md').write_text(
        '# V9.1 独立结果算术核验\n\n'
        f'通过：44 条路线、8 个 history、4 个 parent，12 组方法均值及全部 parent 差值与主分析一致。共 {auditor.numeric_checks} 项数值比较，最大绝对误差 {auditor.maximum_absolute_error:.3g}。\n\n'
        '独立重算完整 reference 上的去重可见面积、背景与两对象分割、实际成本、二维覆盖、F1/联合指标公式、固定 48 动作分段积分、选点与配对等权统计。所有实际失败与负变化保留。\n\n'
        f'未通过的真实 progression 门：{", ".join(failed) if failed else "无"}。算术核验通过不改变任何效果门的成败。\n\n'
        f"完整候选池的终点 F1 上界对 N 的均值差为 {oracle_means['oracle_minus_N_total_F1']:.12f}，原要求 {original_f1_requirement:.3f}；S 的平均全池 F1 regret 为 {oracle_means['S_total_F1_regret']:.12f}。这项上界只诊断本固定池的能力，既不是在线策略，也不修改原门槛。\n\n"
        '未导入主分析器/helper、mapper、planner 或 world；未重新计算几何。全局 precision/recall 与旧支撑误差取自已经全量独立回放的原记录，本次检查其后续公式和统计。\n')
    (output / 'artifact_hashes.json').write_text(json.dumps({p.name: digest(p) for p in output.iterdir() if p.is_file()}, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('status', 'branches', 'numeric_checks', 'maximum_absolute_arithmetic_difference', 'failed_gates')}))


def self_test():
    points = [{'action_index': 0, 'coverage_2d': 1., 'f1_05cm': .2},
              {'action_index': 8, 'coverage_2d': 1., 'f1_05cm': .6}]
    insist(abs(held_auc(points, 48, '05cm') - 17 / 30) < 1e-12, 'held-tail integration')
    insist(abs(held_auc(points[:1], 48, '05cm') - .2) < 1e-12, 'zero-paid constant trace')
    insist(mean([None, 1.]) is None and abs(mean([-.3, .1]) + .1) < 1e-12, 'null and negative aggregation')
    insist(pick([0, 1, 2, 3], [1., 1., 1., 1.], [{'candidate_id': i, 'cost': c} for i, c in enumerate([22, 2, 40, 40])]) == 1, 'cost tie rule')
    reference = {'weights': np.array([.5, 1., 2., 3.]), 'prefix_seen': np.array([True, False, False, False]),
                 'classes': np.array([1, 1, 2, 3]), 'west_slot': np.array([False, False, True, False]),
                 'east_slot': np.array([False, False, False, True])}
    masks = {'prefix': reference['prefix_seen'], 'outbound': np.array([True, True, True, False]),
             'endpoint': np.array([False, True, True, True]), 'return': np.array([True, False, True, True]),
             'union': np.ones(4, bool)}
    result = bitset_areas(reference, masks)
    insist(result['new_area_m2'] == 6. and result['background_new_area_m2'] == 1.
            and result['slot_new_area_m2'] == [2., 3.] and result['stage_new_area_m2'] == {'outbound': 3., 'endpoint': 3., 'return': 0.}, 'unique area partition')
    print(json.dumps({'status': 'passed_pure_self_test', 'future_outcome_files_read': 0, 'worlds_constructed': 0}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path)
    parser.add_argument('--analysis', type=Path)
    parser.add_argument('--output', type=Path, default=ROOT / 'audit_results/competition_v9_1_independent_metrics_20260911')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        insist(args.run is not None and args.analysis is not None, '--run and --analysis are required')
        audit(args)
