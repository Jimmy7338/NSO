#!/usr/bin/env python3
"""Read-only analysis of the sealed V8.1 competition experiment.

Never captures observations, fits a model, changes a choice, or imports a
planner/evaluator. Output must be separate from the acquisition directory.
All protocol gates use the frozen contract; oracle tables are posthoc only.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import zipfile

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = 'configs/virtual3d/competition_v8_1_protocol.json'
PROTOCOL_SHA256 = '878e22e32dbf6b9b566653e1e4d0696ad965640dac2c74e4330f25889166ee76'
METHODS = ('G', 'O', 'S', 'N', 'X', 'M')
TOL = 1e-9
SUMMARY_FIELDS = ('new_area_m2', 'area_per_action', 'f1_gain_05cm',
                  'f1_gain_per_action', 'coverage_gain_m2',
                  'branch_joint_auc_02cm', 'branch_joint_auc_05cm',
                  'paid_actions', 'planned_actions', 'prefix_paid_actions',
                  'task_paid_actions', 'unused_branch_budget_actions', 'background_new_area_m2',
                  'object_new_area_m2', 'area_rate_regret',
                  'final_f1_05cm', 'final_coverage_2d', 'final_joint_05cm',
                  'fixed_old_support_error_mean_change_m',
                  'fixed_old_support_error_p95_change_m')


class IntegrityError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise IntegrityError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def finite(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value))


def near(actual, expected, name):
    require(finite(actual) and finite(expected)
            and math.isclose(actual, expected, rel_tol=1e-9, abs_tol=TOL),
            f'{name}: {actual!r} != {expected!r}')


def mean_complete(values):
    values = list(values)
    return float(np.mean(values)) if values and all(finite(v) for v in values) else None


def minus(first, second):
    return first - second if finite(first) and finite(second) else None


def gate(value, **details):
    return {'status': 'pending' if value is None else 'passed' if value else 'failed',
            'passed': value, **details}


class Inputs:
    def __init__(self):
        self.hashes = {}
        self.manifests = {}

    def note(self, path):
        path = Path(path).resolve()
        digest = sha(path)
        previous = self.hashes.get(str(path))
        require(previous is None or previous == digest, f'input changed: {path}')
        self.hashes[str(path)] = digest
        return digest

    def read(self, path):
        self.note(path)
        return json.loads(Path(path).read_text())

    def manifest(self, folder):
        folder = Path(folder).resolve()
        key = str(folder)
        if key in self.manifests:
            return self.manifests[key]
        manifest = self.read(folder / 'artifact_hashes.json')
        require(isinstance(manifest, dict) and bool(manifest), 'empty artifact manifest')
        for name, digest in manifest.items():
            p = folder / name
            require(not Path(name).is_absolute() and p.resolve().is_relative_to(folder),
                    f'unsafe artifact path: {name}')
            require(p.is_file(), f'missing sealed artifact: {p}')
            require(self.note(p) == digest, f'artifact SHA mismatch: {p}')
        self.manifests[key] = manifest
        return manifest

    def declared_read(self, folder, relative):
        manifest = self.manifest(folder)
        require(relative in manifest, f'file absent from acquisition seal: {relative}')
        return self.read(Path(folder) / relative)


def archived_contract(inputs, run, metadata):
    require('sources.zip' in inputs.manifest(run), 'source archive not sealed')
    declared = metadata.get('source_sha256')
    require(isinstance(declared, dict) and bool(declared), 'missing source SHA inventory')
    with zipfile.ZipFile(run / 'sources.zip') as archive:
        names = archive.namelist()
        require(len(names) == len(set(names)), 'duplicate archive names')
        for name, digest in declared.items():
            require(name in names, f'source dependency missing in archive: {name}')
            require(hashlib.sha256(archive.read(name)).hexdigest() == digest,
                    f'archived source hash mismatch: {name}')
        require(declared.get(PROTOCOL_PATH) == PROTOCOL_SHA256,
                'run does not bind the frozen V8.1 protocol bytes')
        protocol = json.loads(archive.read(PROTOCOL_PATH))
    require(protocol['schema_version'] == 'competition_v8_1_protocol/1', 'wrong protocol')
    rev = protocol['revision_provenance']
    for key in ('immutable_scorer', 'immutable_scoring_geometry_module'):
        require(declared.get(rev[key]) == rev[key + '_sha256'],
                f'frozen scoring component changed: {rev[key]}')
    return protocol


def bound_preparation(inputs, run, metadata, protocol, explicit=None):
    field = next((k for k in ('preparation', 'prepared', 'preparation_path')
                  if isinstance(metadata.get(k), str)), None)
    declared_path = metadata.get(field) if field else None
    if explicit is not None:
        prepared = explicit.resolve()
        if declared_path:
            p = Path(declared_path)
            require(prepared == (p if p.is_absolute() else ROOT / p).resolve(),
                    '--prepared differs from run metadata')
    elif declared_path:
        p = Path(declared_path)
        prepared = (p if p.is_absolute() else ROOT / p).resolve()
    else:
        return None, gate(None, reason='run has no preparation path binding')
    inputs.manifest(prepared)
    digest = inputs.note(prepared / 'artifact_hashes.json')
    execution_seal = inputs.declared_read(run, 'pre_execution_seal.json')
    declared_digest = next((metadata[k] for k in (
        'preparation_manifest_sha256', 'prepared_manifest_sha256',
        'preparation_artifact_manifest_sha256') if k in metadata),
        execution_seal.get('prepared_manifest_sha256'))
    require(declared_digest == digest, 'preparation manifest binding missing or changed')
    pm = inputs.declared_read(prepared, 'metadata.json')
    require(pm.get('status') == 'complete_structural_pass'
            and pm.get('structural_gate_passed') is True,
            'preparation did not pass all four structural histories')
    require(pm.get('source_sha256', {}).get(PROTOCOL_PATH) == PROTOCOL_SHA256,
            'preparation uses a different protocol')
    require(pm.get('new_candidate_branches') == 0 and pm.get('new_sensor_captures') == 0,
            'V8.1 preparation included new physical capture or future branches')
    structure = inputs.declared_read(prepared, 'structure_summary.json')
    expected = {(p, a) for p in protocol['parent_order'] for a in protocol['arrangement_order']}
    got = {(h['context'], h['arrangement']) for h in structure.get('histories', [])}
    require(structure.get('passed') is True and got == expected
            and all(h.get('passed') is True for h in structure['histories']),
            'preparation structural inventory incomplete')
    seal = inputs.declared_read(prepared, 'pre_evaluator_choices_seal.json')
    for parent, arrangement in expected:
        for name in ('candidates.json', 'predictions.json', 'reference.npz'):
            relative = f'{parent}/{arrangement}/{name}'
            require(relative in seal, f'pre-outcome seal missing {relative}')
            require(seal[relative] == inputs.note(prepared / relative), 'pre-outcome seal changed')
            require(relative in inputs.manifest(run), f'run missing frozen {relative}')
            require(inputs.note(run / relative) == seal[relative],
                    f'acquisition changed prepared artifact {relative}')
    return prepared, gate(True, preparation=str(prepared), manifest_sha256=digest,
                          histories=4, frozen_candidates_predictions_reference_exact=True)


def replay_status(inputs, run, metadata, expected):
    path = run / 'verification.json'
    if not path.exists():
        return gate(None, reason='independent verification.json absent')
    row = inputs.read(path)
    checks = {
        'status_passed_full': row.get('status') == 'passed_full',
        'passed_full_boolean': row.get('passed_full') is True,
        'not_partial': row.get('partial') is False,
        'no_branch_limit': 'max_branches' in row and row['max_branches'] is None,
        'checked_all_branches': row.get('branches_checked') == expected,
        'total_exact': row.get('branches_total') == expected,
        'run_binding': row.get('run') == str(run.resolve()),
        'manifest_binding': row.get('artifact_manifest_sha256') == inputs.note(run / 'artifact_hashes.json'),
        'source_archive_binding': row.get('source_archive_sha256') == inputs.note(run / 'sources.zip'),
        'source_inventory_binding': row.get('source_sha256') == metadata.get('source_sha256'),
        'raw_rechecked_after': row.get('raw_hashes_rechecked_after_replay') is True,
    }
    complete = all(checks.values())
    return gate(True if complete else False if row.get('status') == 'failed' else None,
                report_sha256=inputs.note(path), declared_status=row.get('status'), checks=checks,
                reason=None if complete else 'verification missing strict full-run proof or failed')


def load_reference(inputs, folder):
    path = folder / 'reference.npz'; inputs.note(path)
    with np.load(path, allow_pickle=False) as saved:
        needed = ('points', 'classes', 'weights', 'prefix_seen', 'west_slot', 'east_slot', 'reachable')
        require(all(n in saved for n in needed), f'missing reference arrays: {path}')
        ref = {name: saved[name].copy() for name in needed}
    n = len(ref['points'])
    for name in needed[1:-1]:
        require(ref[name].shape == (n,), f'invalid reference shape: {name}')
    require(np.isfinite(ref['weights']).all() and (ref['weights'] > 0).all(), 'invalid weights')
    for name in ('prefix_seen', 'west_slot', 'east_slot', 'reachable'):
        require(ref[name].dtype == np.bool_, f'nonboolean reference array: {name}')
    require(not np.any(ref['west_slot'] & ref['east_slot']), 'overlapping slot masks')
    require(np.array_equal(ref['west_slot'] | ref['east_slot'], ref['classes'] != 1),
            'object slots do not partition nonbackground reference')
    ref['remaining'] = [float(ref['weights'][ref[k] & ~ref['prefix_seen']].sum())
                        for k in ('west_slot', 'east_slot')]
    return ref


def audit_outcome(row, route, parent, arrangement, ref, protocol):
    cid = route['candidate_id']; label = f'{parent}/{arrangement}/{cid}'
    for key, value in (('context', parent), ('arrangement', arrangement),
                       ('candidate_id', cid), ('group', route['group'])):
        require(row.get(key) == value, f'{label}: {key} mismatch')
    paid, planned = row.get('paid_actions'), row.get('planned_actions')
    require(type(paid) is int and 0 <= paid <= protocol['branch_actions'], f'{label}: invalid paid cost')
    require(type(planned) is int and planned == route['cost'] == len(route['actions']),
            f'{label}: planned cost mismatch')
    require(1 <= planned <= protocol['branch_actions'], f'{label}: planned budget exceeded')
    require(row.get('prefix_paid_actions') == protocol['prefix_paid_actions'], f'{label}: prefix cost')
    require(row.get('task_paid_actions') == protocol['prefix_paid_actions'] + paid, f'{label}: task cost')
    for key in ('original_target_reached', 'full_original_route_completed', 'returned_to_anchor'):
        require(type(row.get(key)) is bool, f'{label}: missing boolean {key}')
    require(type(row.get('collision_count')) is int and row['collision_count'] >= 0, f'{label}: collisions')
    require(row.get('failure') is None or isinstance(row['failure'], str), f'{label}: failure type')
    if row['full_original_route_completed']:
        require(paid == planned and row['original_target_reached'] and row['returned_to_anchor'],
                f'{label}: inconsistent complete-route flags')
    area = row.get('new_area_m2')
    require(finite(area) and area >= 0, f'{label}: invalid new area')
    slots, remaining, fractions = (row.get(k) for k in
                                  ('slot_new_area_m2', 'slot_remaining_area_m2', 'slot_new_fractions'))
    require(all(isinstance(v, list) and len(v) == 2 for v in (slots, remaining, fractions)),
            f'{label}: slot inventory')
    for i in range(2):
        require(finite(slots[i]) and slots[i] >= 0, f'{label}: slot area')
        near(remaining[i], ref['remaining'][i], f'{label}: remaining[{i}]')
        require(remaining[i] > 0 and slots[i] <= remaining[i] + TOL, f'{label}: slot area bounds')
        near(fractions[i], slots[i] / remaining[i], f'{label}: slot fraction[{i}]')
    bg = row.get('background_new_area_m2')
    require(finite(bg) and bg >= 0, f'{label}: background area')
    near(area, sum(slots) + bg, f'{label}: area partition')
    require(area <= float(ref['weights'][~ref['prefix_seen']].sum()) + TOL,
            f'{label}: new area exceeds unseen reference')
    before, after = row['before'], row['after']
    for state in (before, after):
        require(finite(state.get('coverage_2d')) and 0 <= state['coverage_2d'] <= 1, f'{label}: coverage')
        for tag in ('02cm', '05cm'):
            for metric in ('f1', 'precision', 'recall'):
                value = state.get(f'{metric}_{tag}')
                require(finite(value) and -TOL <= value <= 1 + TOL, f'{label}: {metric}_{tag}')
            near(state[f'joint_{tag}'], state['coverage_2d'] * state[f'f1_{tag}'], f'{label}: joint')
            require(finite(row.get(f'branch_joint_auc_{tag}')), f'{label}: missing branch AUC')
    near(row['f1_gain_05cm'], after['f1_05cm'] - before['f1_05cm'], f'{label}: F1 difference')
    near(row['coverage_gain_m2'], (after['coverage_2d'] - before['coverage_2d'])
         * int(ref['reachable'].sum()) * protocol['sensors']['resolution_m'] ** 2, f'{label}: coverage area')
    if paid:
        near(row['area_per_action'], area / paid, f'{label}: area rate')
        near(row['f1_gain_per_action'], row['f1_gain_05cm'] / paid, f'{label}: F1 rate')
    else:
        require(row.get('area_per_action') is None and row.get('f1_gain_per_action') is None,
                f'{label}: zero-paid rates must remain null')
        near(area, 0., f'{label}: area without paid observation')
    result = dict(row)
    result.update(object_new_area_m2=float(sum(slots)),
                  unused_branch_budget_actions=protocol['branch_actions'] - paid,
                  object_area_per_action=sum(slots) / paid if paid else None,
                  background_area_per_action=bg / paid if paid else None,
                  background_fraction_of_new_area=bg / area if area > 0 else None,
                  final_f1_05cm=after['f1_05cm'], final_coverage_2d=after['coverage_2d'],
                  final_joint_05cm=after['joint_05cm'], area_rate_regret=None,
                  fixed_old_support_error_mean_change_m=minus(
                      after.get('fixed_old_support_error_mean_m'), before.get('fixed_old_support_error_mean_m')),
                  fixed_old_support_error_p95_change_m=minus(
                      after.get('fixed_old_support_error_p95_m'), before.get('fixed_old_support_error_p95_m')),
                  common_success_sensitivity_eligible=(row['failure'] is None
                      and row['full_original_route_completed'] and row['returned_to_anchor']
                      and row['collision_count'] == 0))
    return result


def audit_saved_evidence(inputs, run, relative, row, ref, protocol):
    """Recompute saved bitset unions and sparse AUC; does not rerender rays."""
    folder = run / relative
    manifest = inputs.manifest(run)
    for name in ('visibility.npz', 'metrics.json', 'actions.json'):
        require(relative + '/' + name in manifest, f'missing sealed branch evidence: {relative}/{name}')
    with np.load(folder / 'visibility.npz', allow_pickle=False) as saved:
        masks = {k: saved[k].copy() for k in ('prefix', 'outbound', 'endpoint', 'return', 'union')}
    for name, mask in masks.items():
        require(mask.dtype == np.bool_ and mask.shape == ref['prefix_seen'].shape,
                f'{relative}: invalid visibility mask {name}')
    require(np.array_equal(masks['prefix'], ref['prefix_seen']), f'{relative}: prefix visibility changed')
    union = masks['outbound'] | masks['endpoint'] | masks['return']
    require(np.array_equal(union, masks['union']), f'{relative}: saved union mismatch')
    new = union & ~ref['prefix_seen']; weights = ref['weights']
    near(row['new_area_m2'], float(weights[new].sum()), f'{relative}: bitset new area')
    near(row['background_new_area_m2'], float(weights[new & (ref['classes'] == 1)].sum()),
         f'{relative}: bitset background area')
    for i, name in enumerate(('west_slot', 'east_slot')):
        near(row['slot_new_area_m2'][i], float(weights[new & ref[name]].sum()), f'{relative}: bitset slot area')
    seen = ref['prefix_seen'].copy(); stage_sum = 0.
    for name in ('outbound', 'endpoint', 'return'):
        increment = float(weights[masks[name] & ~seen].sum()); seen |= masks[name]; stage_sum += increment
        near(row['stage_new_area_m2'][name], increment, f'{relative}: deduplicated {name} area')
    near(stage_sum, row['new_area_m2'], f'{relative}: stage partition')
    metrics = inputs.read(folder / 'metrics.json'); actions = inputs.read(folder / 'actions.json')
    require(len(actions) == row['paid_actions'], f'{relative}: action count mismatch')
    require([a['action_index'] for a in actions] == list(range(1, row['paid_actions'] + 1)),
            f'{relative}: noncontinuous paid action IDs')
    require(all(a['absolute_step'] == row['prefix_paid_actions'] + a['action_index'] for a in actions),
            f'{relative}: absolute action index mismatch')
    require(sum(bool(a['collision']) for a in actions) == row['collision_count'], f'{relative}: collision count')
    times = [m['action_index'] for m in metrics]
    require(times and times[0] == 0 and times[-1] == row['paid_actions']
            and all(b > a for a, b in zip(times, times[1:])), f'{relative}: invalid AUC checkpoints')
    require({k: v for k, v in metrics[-1].items() if k != 'action_index'} == row['after'],
            f'{relative}: final checkpoint mismatch')
    require({k: v for k, v in metrics[0].items() if k != 'action_index'} == row['before'],
            f'{relative}: prefix checkpoint mismatch')
    for tag in ('02cm', '05cm'):
        trace = np.interp(np.arange(protocol['branch_actions'] + 1), times,
                          [m[f'joint_{tag}'] for m in metrics])
        auc = float(np.sum((trace[:-1] + trace[1:]) / 2) / protocol['branch_actions'])
        near(row[f'branch_joint_auc_{tag}'], auc, f'{relative}: independently summed sparse AUC')
    return {'saved_visibility_partition_recomputed': True, 'sparse_auc_recomputed': True,
            'saved_action_count_checked': len(actions), 'raw_ray_or_TSDF_replay': False}


def paired_oracle(first, second, ids, metric, tolerance):
    if not ids or any(i not in first or i not in second or not finite(first[i].get(metric))
                      or not finite(second[i].get(metric)) for i in ids):
        return {'evaluable': False, 'metric': metric, 'candidate_ids': list(ids),
                'reason': 'empty, missing or undefined utility; no imputation', 'value': None}
    a = np.array([first[i][metric] for i in ids]); b = np.array([second[i][metric] for i in ids])
    marginal = (a + b) / 2
    conditional = float((a.max() + b.max()) / 2)
    value = conditional - float(marginal.max())
    require(value >= -TOL, 'negative oracle value beyond arithmetic tolerance')
    arg = lambda values: [i for i, v in zip(ids, values) if float(values.max() - v) <= tolerance]
    aa, ab = arg(a), arg(b)
    return {'evaluable': True, 'metric': metric, 'candidate_ids': list(ids),
            'world_values': [a.tolist(), b.tolist()], 'marginal_values': marginal.tolist(),
            'conditional_oracle': conditional, 'marginal_oracle': float(marginal.max()),
            'value': value, 'world_maximizer_sets': [aa, ab],
            'maximizer_intersection': sorted(set(aa) & set(ab)),
            'marginal_maximizer_set': arg(marginal), 'tie_absolute_tolerance': tolerance,
            'world_maximizers_details': [[{
                'candidate_id': i, 'group': table[i]['group'], 'failure': table[i]['failure'],
                'paid_actions': table[i]['paid_actions'], 'planned_actions': table[i]['planned_actions'],
                'returned_to_anchor': table[i]['returned_to_anchor'],
                'background_fraction_of_new_area': table[i]['background_fraction_of_new_area']}
                for i in choices] for table, choices in ((first, aa), (second, ab))],
            'is_deployed_method': False}


def averages(rows, expected_count):
    rows = list(rows)
    return {'histories_available': len(rows), 'histories_expected': expected_count,
            'means': {k: mean_complete(r.get(k) for r in rows) if len(rows) == expected_count else None
                      for k in SUMMARY_FIELDS},
            'defined_counts': {k: sum(finite(r.get(k)) for r in rows) for k in SUMMARY_FIELDS},
            'failures': sum(r['failure'] is not None for r in rows),
            'not_returned': sum(not r['returned_to_anchor'] for r in rows),
            'collisions': sum(r['collision_count'] for r in rows),
            'original_targets_missed': sum(not r['original_target_reached'] for r in rows),
            'original_routes_incomplete': sum(not r['full_original_route_completed'] for r in rows)}


def comparison(selected, other, parents, arrangements):
    pairs = []
    for parent in parents:
        for arrangement in arrangements:
            rows = selected.get((parent, arrangement), {})
            if 'S' not in rows or other not in rows:
                continue
            s, c = rows['S'], rows[other]
            pairs.append({'context': parent, 'arrangement': arrangement,
                          'S_candidate_id': s['candidate_id'], 'other_candidate_id': c['candidate_id'],
                          **{k: minus(s.get(k), c.get(k)) for k in SUMMARY_FIELDS}})
    count = len(parents) * len(arrangements)
    return {'comparator': other, 'pairs': pairs,
            'means': {k: mean_complete(r[k] for r in pairs) if len(pairs) == count else None
                      for k in SUMMARY_FIELDS},
            'per_parent': {p: {k: mean_complete(r[k] for r in pairs if r['context'] == p)
                              if sum(r['context'] == p for r in pairs) == len(arrangements) else None
                              for k in SUMMARY_FIELDS} for p in parents}}


def protocol_gates(protocol, histories, branches, selected, methods, comparisons,
                   oracles, complete, defined, structure, verification, release_bound):
    spec = protocol['progression_necessary_conditions']; parents = protocol['parent_order']
    gates = {'all_24_complete': gate(complete),
             'all_24_primary_utilities_defined': gate(defined if complete else None),
             'structural_preparation_and_choice_binding': structure,
             'pre_execution_release_binding': release_bound,
             'independent_full_replay': verification}
    gates['physical_safety'] = gate(
        all(r['collision_count'] == 0 and r['returned_to_anchor'] for r in branches) if complete else None,
        collisions=sum(r['collision_count'] for r in branches),
        not_returned=sum(not r['returned_to_anchor'] for r in branches),
        original_targets_missed=sum(not r['original_target_reached'] for r in branches),
        original_routes_incomplete=sum(not r['full_original_route_completed'] for r in branches))
    exclusive = protocol['post_branch_exclusivity_gate']; violations = []
    for row in branches:
        if row['group'] in ('west_deep', 'east_deep'):
            target = 0 if row['group'] == 'west_deep' else 1
            if row['slot_new_fractions'][1-target] > exclusive['each_declared_deep_route_other_slot_new_fraction_at_most']:
                violations.append({'context': row['context'], 'arrangement': row['arrangement'],
                    'candidate_id': row['candidate_id'], 'reason': 'other_slot_leakage',
                    'actual': row['slot_new_fractions'][1-target]})
            if row['slot_new_area_m2'][target] <= exclusive['each_declared_deep_route_target_slot_new_area_m2_greater_than']:
                violations.append({'context': row['context'], 'arrangement': row['arrangement'],
                    'candidate_id': row['candidate_id'], 'reason': 'no_actual_target_evidence',
                    'actual': row['slot_new_area_m2'][target]})
        if all(f >= exclusive['no_candidate_may_recover_both_slots_fraction_at_least']
               for f in row['slot_new_fractions']):
            violations.append({'context': row['context'], 'arrangement': row['arrangement'],
                'candidate_id': row['candidate_id'], 'reason': 'both_slots_almost_exhausted'})
    gates['actual_sensor_exclusivity'] = gate(not violations if complete else None, violations=violations)
    for p in parents:
        oracle = oracles[p]['complete_pool']['area_per_action']
        value = (oracle['value'] > spec['per_parent_primary_oracle_must_exceed']
                 and not oracle['maximizer_intersection']) if defined and oracle['evaluable'] else None
        gates[f'{p}_positive_primary_information_value'] = gate(value, oracle=oracle)
    for other in ('G', 'O', 'N'):
        sm, cm = methods['S']['means']['area_per_action'], methods[other]['means']['area_per_action']
        delta = comparisons[other]['means']['area_per_action']
        relative = sm / cm - 1 if finite(sm) and finite(cm) and cm > 0 else None
        if not defined or not finite(delta) or not finite(cm):
            passes = None; rule = 'undefined'
        elif cm < spec['near_zero_comparator_threshold']:
            passes = delta >= spec['near_zero_minimum_absolute_gain_m2_per_action']; rule = 'absolute'
        else:
            passes = relative >= spec['minimum_relative_gain']; rule = 'relative'
        parent_deltas = {p: comparisons[other]['per_parent'][p]['area_per_action'] for p in parents}
        if passes is not None:
            passes = passes and all(finite(v) and v > 0 for v in parent_deltas.values())
        gates[f'S_area_rate_over_{other}'] = gate(passes, mean_delta=delta,
            relative_gain=relative, rule=rule, per_parent_delta=parent_deltas)
    for other in ('G', 'O'):
        value = comparisons[other]['means']['f1_gain_05cm']
        gates[f'S_F1_noninferiority_to_{other}'] = gate(
            value >= spec['min_S_minus_G_O_mean_f1_gain_difference'] if finite(value) else None,
            mean_delta=value)
    f1 = comparisons['N']['means']['f1_gain_per_action']
    f1parents = {p: comparisons['N']['per_parent'][p]['f1_gain_per_action'] for p in parents}
    gates['S_net_F1_per_action_over_N'] = gate(
        f1 > 0 and all(finite(v) and v >= 0 for v in f1parents.values()) if finite(f1) else None,
        mean_delta=f1, per_parent_delta=f1parents)
    cov = comparisons['N']['means']['coverage_gain_m2']
    gates['S_coverage_noninferiority_to_N'] = gate(
        cov >= -spec['max_S_coverage_mean_loss_against_N_m2_per_branch'] if finite(cov) else None,
        mean_delta_m2=cov)
    xdelta = comparisons['X']['means']['area_per_action']
    gates['correct_S_over_swapped_X'] = gate(xdelta > 0 if defined and finite(xdelta) else None,
                                           mean_delta=xdelta)
    gates['missing_M_exactly_G'] = gate(all(h['M_equals_G'] for h in histories)
                                      if len(histories) == 4 else None)
    for p in parents:
        hs = [h for h in histories if h['context'] == p]
        choices = [h['selected_candidate_ids']['S'] for h in hs]
        gates[f'{p}_S_choice_changes_with_H_slot'] = gate(
            len(set(choices)) == 2 if len(hs) == 2 else None, choices=choices,
            note='choice change required; target roles and actual utility separately reported')
    return gates


def release_proof(inputs, run, metadata, prepared, protocol):
    """Require a runner-declared pre-execution seal digest, not file mtime."""
    if prepared is None:
        return gate(None, reason='preparation binding absent')
    saved = inputs.declared_read(run, 'pre_execution_seal.json')
    require(saved.get('protocol') == protocol, 'executor changed frozen protocol')
    require(saved.get('physical_branch_limit') == protocol['physical_branch_limit'], 'execution seal branch limit')
    require(saved.get('source_sha256') == metadata.get('source_sha256'), 'execution seal source inventory changed')
    require(saved.get('prepared_manifest_sha256') == inputs.note(prepared / 'artifact_hashes.json'),
            'execution seal preparation changed')
    decision_assets = saved.get('decision_assets')
    require(isinstance(decision_assets, dict) and bool(decision_assets), 'empty pre-execution seal')
    original = inputs.declared_read(prepared, 'pre_evaluator_choices_seal.json')
    require(decision_assets == original, 'executor decision inventory differs from all-four-history preparation seal')
    for relative, digest in decision_assets.items():
        require(relative in inputs.manifest(run) and inputs.note(run / relative) == digest,
                f'pre-execution decision asset changed: {relative}')
    review_sha = saved.get('structural_review_sha256')
    require(isinstance(review_sha, str) and len(review_sha) == 64, 'missing independent structural review digest')
    return gate(True, pre_execution_seal_sha256=inputs.note(run / 'pre_execution_seal.json'),
                prepared_choice_seal_sha256=inputs.note(prepared / 'pre_evaluator_choices_seal.json'),
                independent_structural_review_sha256=review_sha,
                scope='frozen runner receipt bound to full prepared decision seal; structural review content not reopened here; full physical replay remains separate')


def analyze(run, output, prepared_override=None):
    run, output = run.resolve(), output.resolve()
    require(not output.is_relative_to(run) and not run.is_relative_to(output),
            'analysis output must be separate from the acquisition tree')
    require(not output.exists(), 'output exists; use a new independent analysis directory')
    output.mkdir(parents=True)
    inputs = Inputs(); result = None; code = 0
    try:
        manifest = inputs.manifest(run)
        metadata = inputs.declared_read(run, 'metadata.json')
        protocol = archived_contract(inputs, run, metadata)
        prepared, structure = bound_preparation(inputs, run, metadata, protocol, prepared_override)
        verification = replay_status(inputs, run, metadata, protocol['physical_branch_limit'])
        release = release_proof(inputs, run, metadata, prepared, protocol)
        parents, arrangements, ids = (protocol[k] for k in ('parent_order', 'arrangement_order', 'candidate_order'))
        rows = []; histories = []; tables = {}; selected = {}; missing = []; evidence_audits = []
        for parent in parents:
            for arrangement in arrangements:
                rel = f'{parent}/{arrangement}'; folder = run / rel
                routes = inputs.declared_read(run, rel + '/candidates.json')
                prediction = inputs.declared_read(run, rel + '/predictions.json')
                require([r['candidate_id'] for r in routes] == ids, f'{rel}: six candidate IDs/order')
                require([r['group'] for r in routes] == protocol['pre_outcome_structural_gates']['candidate_roles_in_order'],
                        f'{rel}: role order changed')
                scores, choices = prediction['scores'], prediction['selected_candidate_ids']
                require(set(scores) == set(METHODS) and set(choices) == set(METHODS), f'{rel}: scorer inventory')
                for mode in METHODS:
                    require(len(scores[mode]) == len(ids) and all(finite(s) for s in scores[mode]),
                            f'{rel}: invalid score vector {mode}')
                    pick = min(range(len(ids)), key=lambda i: (-scores[mode][i], routes[i]['cost'], ids[i]))
                    require(choices[mode] == ids[pick], f'{rel}: sealed choice disagrees with tie rule: {mode}')
                ref = load_reference(inputs, folder)
                table = {}
                for route in routes:
                    relative = f'{rel}/candidate_{route["candidate_id"]:03d}/outcome.json'
                    if relative not in manifest:
                        missing.append(relative); continue
                    outcome = inputs.declared_read(run, relative)
                    row = audit_outcome(outcome, route, parent, arrangement, ref, protocol)
                    evidence_audits.append({'branch': relative.rsplit('/', 1)[0],
                        **audit_saved_evidence(inputs, run, relative.rsplit('/', 1)[0], row, ref, protocol)})
                    table[route['candidate_id']] = row; rows.append(row)
                if table:
                    require(all(r['before'] == next(iter(table.values()))['before'] for r in table.values()),
                            f'{rel}: branch prefix reconstruction metrics differ')
                if len(table) == len(ids) and all(finite(r['area_per_action']) for r in table.values()):
                    top = max(r['area_per_action'] for r in table.values())
                    for row in table.values():
                        row['area_rate_regret'] = top - row['area_per_action']
                tables[(parent, arrangement)] = table
                selected[(parent, arrangement)] = {m: table[choices[m]] for m in METHODS if choices[m] in table}
                histories.append({'context': parent, 'arrangement': arrangement,
                    'selected_candidate_ids': choices, 'scores': scores,
                    'M_equals_G': scores['M'] == scores['G'] and choices['M'] == choices['G'],
                    'G_equals_O': scores['G'] == scores['O'] and choices['G'] == choices['O'],
                    'scored_candidates': len(routes), 'outcomes_available': len(table),
                    'undefined_rate_candidate_ids': [i for i, r in table.items() if r['area_per_action'] is None],
                    'reference_remaining_slot_area_m2': ref['remaining'],
                    'reference_sha256': inputs.note(folder / 'reference.npz'),
                    'selected_roles': {m: routes[ids.index(choices[m])]['group'] for m in METHODS}})
        expected_paths = {f'{p}/{a}/candidate_{i:03d}/outcome.json' for p in parents for a in arrangements for i in ids}
        actual_paths = {name for name in manifest if name.endswith('/outcome.json')}
        require(not (actual_paths - expected_paths), 'unscheduled outcomes in primary acquisition seal')
        execution_complete = metadata.get('status') in ('complete', 'complete_execution_pending_independent_replay')
        complete = (len(rows) == protocol['physical_branch_limit'] and not missing and execution_complete
                    and metadata.get('new_physical_branches') == protocol['physical_branch_limit'])
        defined = complete and all(finite(r['area_per_action']) for r in rows)
        pair_checks = {}
        for parent in parents:
            hs = [h for h in histories if h['context'] == parent]; a, b = hs
            ra = inputs.declared_read(run, f'{parent}/{arrangements[0]}/candidates.json')
            rb = inputs.declared_read(run, f'{parent}/{arrangements[1]}/candidates.json')
            require(ra == rb, f'{parent}: paired routes differ')
            checks = {m: a['scores'][m] == b['scores'][m] for m in ('G', 'O', 'N', 'M')}
            checks.update(S_X_swap_exact=(a['scores']['S'] == b['scores']['X'] and a['scores']['X'] == b['scores']['S']))
            require(all(checks.values()), f'{parent}: paired nonsemantic/semantic score invariant broken')
            pair_checks[parent] = checks
        oracle_metrics = ('area_per_action', 'object_area_per_action', 'background_area_per_action',
                          'f1_gain_05cm', 'f1_gain_per_action', 'branch_joint_auc_05cm')
        oracles = {}
        for parent in parents:
            a, b = (tables[(parent, arrangement)] for arrangement in arrangements)
            success_ids = [i for i in ids if i in a and i in b and a[i]['common_success_sensitivity_eligible']
                           and b[i]['common_success_sensitivity_eligible']]
            full = {metric: paired_oracle(a, b, ids, metric, protocol['posthoc_oracle']['tie_absolute_tolerance'])
                    for metric in oracle_metrics}
            # A partial cohort or even one zero-paid outcome invalidates the whole primary oracle gate.
            if not defined:
                full['area_per_action'].update(evaluable=False, value=None,
                    reason='the complete 24-branch primary table is unavailable or contains undefined zero-paid utility')
            oracles[parent] = {'complete_pool': full, 'world_order': arrangements,
                'common_success_sensitivity': {metric: paired_oracle(a, b, success_ids, metric,
                    protocol['posthoc_oracle']['tie_absolute_tolerance']) for metric in oracle_metrics},
                'sensitivity_candidate_ids': success_ids, 'sensitivity_replaces_primary': False}
        methods = {m: averages([d[m] for d in selected.values() if m in d], 4) for m in METHODS}
        comparisons = {m: comparison(selected, m, parents, arrangements) for m in ('G', 'O', 'N', 'X', 'M')}
        gates = protocol_gates(protocol, histories, rows, selected, methods, comparisons, oracles,
                               complete, defined, structure, verification, release)
        positive = all(g['passed'] is True for g in gates.values())
        failed = [k for k, g in gates.items() if g['passed'] is False]
        pending = [k for k, g in gates.items() if g['passed'] is None]
        status = ('passed_development_necessary_conditions' if positive else
                  'failed_development_gates' if failed else 'pending_required_evidence')
        choices_rows = [{'context': p, 'arrangement': a, 'method': m, **row}
                        for (p, a), modes in selected.items() for m, row in modes.items()]
        result = {'schema_version': 'competition_v8_1_analysis/1', 'status': status,
            'run': str(run), 'acquisition_status': metadata.get('status'),
            'scope': 'two-parent synthetic asset-prior positive control; no training or calibration',
            'expected_branches': 24, 'available_branches': len(rows), 'missing_outcomes': missing,
            'complete_primary_table': complete, 'primary_utilities_all_defined': defined,
            'zero_paid_branches': [{k: r[k] for k in ('context', 'arrangement', 'candidate_id', 'terminal_reason')}
                                   for r in rows if r['paid_actions'] == 0],
            'physical_failures': [{k: r[k] for k in ('context', 'arrangement', 'candidate_id', 'failure',
                'collision_count', 'returned_to_anchor', 'original_target_reached', 'full_original_route_completed',
                'paid_actions', 'planned_actions', 'terminal_reason')} for r in rows
                if r['failure'] is not None or r['collision_count'] or not r['returned_to_anchor']
                or not r['full_original_route_completed']],
            'histories': histories, 'paired_score_checks': pair_checks,
            'saved_branch_evidence_audits': evidence_audits,
            'method_averages': methods, 'S_comparisons': comparisons, 'paired_oracles': oracles,
            'localized_fine_class_area_rate_differences': {
                'S_minus_O': comparisons['O']['means']['area_per_action'],
                'S_minus_X': comparisons['X']['means']['area_per_action'],
                'does_not_override_other_gates_or_replay': True},
            'gates': gates, 'failed_gates': failed, 'pending_gates': pending,
            'positive_mechanism_claim_allowed': positive,
            'old_V8_structural_failure_preserved': True, 'whole_system_advantage_proven': False,
            'independent_generalization_confirmation': False, 'mainstream_superiority_proven': False,
            'calibrated_or_learned_semantic_response': False,
            'branch_joint_auc_limit': '48 paid-action sparse-checkpoint interpolation with terminal hold; no continued replanning; prefix150 excluded from this local AUC',
            'prefix_cost_limit': 'all selections pay the same actual prefix150; task cost=150+paid; four physical histories, not 24 independent prefixes',
            'sampling_limit': 'two development parents and four swapped histories; 24 alternatives are dependent counterfactuals',
            'already_saturated_2D_coverage': all(r['before']['coverage_2d'] == 1. for r in rows) if complete else None,
            'oracle_limit': 'GT posthoc finite-pool value of category information; never a deployed algorithm or sample-selection rule',
            'arithmetic_verification_scope': 'sealed outcomes, score choices, reference denominators, saved visibility unions/stages and sparse AUC arithmetic; no independent RGBD projection or TSDF replay performed by this analyzer',
            'protocol_sha256': PROTOCOL_SHA256}
        write_json(output / 'protocol.json', protocol)
        write_json(output / 'branches.json', rows)
        write_json(output / 'choices.json', choices_rows)
        write_csv(output / 'branches.csv', rows)
        write_csv(output / 'choices.csv', choices_rows)
    except Exception as exc:
        code = 1
        result = {'schema_version': 'competition_v8_1_analysis/1', 'status': 'analysis_integrity_failure',
                  'run': str(run), 'error': f'{type(exc).__name__}: {exc}',
                  'positive_mechanism_claim_allowed': False, 'whole_system_advantage_proven': False}
    # Recheck every input observed by this analyzer before publishing output.
    changed = [path for path, digest in inputs.hashes.items() if not Path(path).is_file() or sha(path) != digest]
    if changed:
        code = 1; result.update(status='inputs_changed_during_analysis', changed_inputs=changed,
                                positive_mechanism_claim_allowed=False)
    source = Path(__file__).resolve()
    shutil.copyfile(source, output / 'analyze_competition_v8_1.py')
    result['provenance'] = {'analyzer_sha256': sha(source), 'input_files_checked': len(inputs.hashes),
                            'numpy_version': np.__version__, 'source_snapshot': 'analyze_competition_v8_1.py',
                            'acquisition_files_modified': False}
    write_json(output / 'input_hashes.json', inputs.hashes)
    write_json(output / 'summary.json', result)
    (output / 'REPORT.md').write_text(chinese_report(result))
    write_json(output / 'artifact_hashes.json', {str(p.relative_to(output)): sha(p)
               for p in sorted(output.rglob('*')) if p.is_file() and p.name != 'artifact_hashes.json'})
    print(json.dumps({k: result.get(k) for k in ('status', 'available_branches',
        'positive_mechanism_claim_allowed', 'failed_gates', 'pending_gates', 'error')}, ensure_ascii=False))
    return code


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def write_csv(path, rows):
    if not rows:
        Path(path).write_text(''); return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, keys); writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False, allow_nan=False)
                             if isinstance(v, (dict, list)) else v for k, v in row.items()})


def fmt(value, digits=6):
    return f'{value:.{digits}f}' if finite(value) else '不可判定'


def chinese_report(result):
    lines = ['# V8.1 双对象竞争实验：完整分支与协议判定', '',
             f'分析状态：`{result["status"]}`。']
    if 'error' in result:
        return '\n'.join(lines + ['', f'输入完整性检查未通过：{result["error"]}',
                                  '没有据此生成正向效果结论。原采集数据未修改。', ''])
    lines += ['', f'预定24条，实际读取{result["available_branches"]}条；零付费分支'
              f'{len(result["zero_paid_branches"])}条。缺失或零付费主效用不补成0。',
              '本批只有两个开发父上下文、四个交换世界；24条是共享历史的反事实候选，不是24个独立样本。', '',
              '|方法|全场新增m²|面积／动作|终点F1@5cm|F1增量|二维新增m²|稀疏J-AUC@5cm|实际动作|未用48步预算|150+实际动作|',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for mode in METHODS:
        r = result['method_averages'][mode]['means']
        keys = ('new_area_m2', 'area_per_action', 'final_f1_05cm', 'f1_gain_05cm',
                'coverage_gain_m2', 'branch_joint_auc_05cm', 'paid_actions',
                'unused_branch_budget_actions', 'task_paid_actions')
        lines.append('|' + mode + '|' + '|'.join(fmt(r[k]) for k in keys) + '|')
    lines += ['', '均值按四history等权；任何必要值缺失即标不可判定，定义数量另见JSON。'
              'G/O/S/N共享基础项；S是显式资产先验，X只交换解释，M回退G。', '',
              '|父上下文|全场面积率信息oracle增量|两世界最优集合|交集|共同成功子集oracle（仅敏感性）|',
              '|---|---:|---|---|---:|']
    for parent, record in result['paired_oracles'].items():
        o = record['complete_pool']['area_per_action']; s = record['common_success_sensitivity']['area_per_action']
        lines.append(f'|{parent}|{fmt(o.get("value"))}|{o.get("world_maximizer_sets", "不可判定")}'
                     f'|{o.get("maximizer_intersection", "不可判定")}|{fmt(s.get("value"))}|')
    lines += ['', 'oracle是事后知道收益表的类别条件最优决策相对共同最优决策的差，'
              '不是S成绩，也不允许用来删上下文或选新算法。成功共同候选敏感性表不替换原24条主表。', '',
              '|S相对对照|面积率均值差|F1增量均值差|F1／动作均值差|二维新增均值差|',
              '|---|---:|---:|---:|---:|']
    for mode, r in result['S_comparisons'].items():
        lines.append('|' + mode + '|' + '|'.join(fmt(r['means'][k]) for k in
                     ('area_per_action', 'f1_gain_05cm', 'f1_gain_per_action', 'coverage_gain_m2')) + '|')
    lines += ['', '|固定门槛|判定|', '|---|---|']
    labels = {'passed': '通过', 'failed': '未通过', 'pending': '待证据／不可判定'}
    for name, entry in result['gates'].items():
        lines.append(f'|{name}|{labels[entry["status"]]}|')
    lines += ['', '跨对象实际可见性不能由路径区距离代替：', '',
              '|history／候选|未满足的实际传感器排他条件|实际量|', '|---|---|---:|']
    reasons = {'other_slot_leakage': '非目标剩余面积获取超过10%',
               'no_actual_target_evidence': '目标未取得新增证据',
               'both_slots_almost_exhausted': '同时取得两槽位各≥90%剩余面积'}
    violations = result['gates']['actual_sensor_exclusivity']['violations']
    for v in violations:
        lines.append(f'|{v["context"]}/{v["arrangement"]}/{v["candidate_id"]}'
                     f'|{reasons[v["reason"]]}|{fmt(v.get("actual")) if "actual" in v else "两边均过线"}|')
    if not violations:
        lines.append('|—|当前读取分支未记录排他违规，完整性门槛另判|—|')
    lines += ['', f'记录到失败、未返程或未完成原路线的分支{len(result["physical_failures"])}条；'
              '安全返回但没有到原目标也单独保留。完整明细、原计划与实际成本、两槽位新增比例见branches.csv，'
              '所有评分器选中的失败情况见choices.csv和JSON，不能把短失败分支的高面积率称为成功观察。', '',
              '独立回放状态：' + labels[result['gates']['independent_full_replay']['status']] + '。'
              '本分析只核封存、选点与算术，没有重新投影RGB-D或融合TSDF；完整passed_full回放证明缺失时，不允许正向机制结论。', '',
              '实际可见新面积不等于准确重建面积，F1改变也不单独证明定位或同支持精度提高。'
              'J-AUC为共同48动作窗口的稀疏检查点插值，短路线后保持终点，不含剩余预算继续规划。'
              '共同前缀实际支付150动作，局部AUC不含它；任务成本必须显示150+paid。', '',
              ('全部必要门槛通过，仅支持另立新T/C与完整闭环验证计划。' if result['positive_mechanism_claim_allowed']
               else '本批尚未满足全部开发机制门槛，不能声称语义或完整方案优势已获证明。'),
              '原V8结构失败保留；本批不训练、不校准、不打开保留C/E，不证明自然语义泛化、完整ANS四模块协同或主流领先。', '']
    if result.get('already_saturated_2D_coverage'):
        lines += ['四历史分支起点的二维覆盖C已经为1，因此这里的C×F1变化主要反映F1，'
                  '不能当作从未知环境完成二维覆盖的证据。', '']
    lines += ['面积率A/L与固定48动作后的F1／J-AUC是不同目标：短路线可以提高A/L，'
              '同时留下更多未用预算并降低终点完整性。不能用效率优势覆盖终点F1门槛失败。'
              '另一方面，一般收益表即使不满足严格对象排他，也可能有正的信息价值：'
              '只要两世界最优候选集合不相交，有限同池的条件oracle增量即可为正。'
              '这条一般命题不改本批已冻结的10%排他门槛，也不允许事后换目标或重选候选。', '']
    return '\n'.join(lines)


def self_test():
    """Synthetic mathematical edge cases; never opens experiment directories."""
    def row(value, fail=False):
        return {'area_per_action': value, 'group': 'test', 'failure': 'synthetic' if fail else None,
                'paid_actions': 2, 'planned_actions': 2, 'returned_to_anchor': not fail,
                'background_fraction_of_new_area': 0.0}
    a = {0: row(2.), 1: row(1.)}; b = {0: row(1.), 1: row(2.)}
    o = paired_oracle(a, b, [0, 1], 'area_per_action', 1e-10)
    near(o['value'], .5, 'oracle swapped optimum'); require(o['maximizer_intersection'] == [], 'oracle intersection')
    b[0] = row(2.)
    near(paired_oracle(a, b, [0, 1], 'area_per_action', 1e-10)['value'], 0., 'oracle common optimum')
    b[0] = row(None)
    require(not paired_oracle(a, b, [0, 1], 'area_per_action', 1e-10)['evaluable'], 'null utility imputed')
    require(not paired_oracle(a, b, [], 'area_per_action', 1e-10)['evaluable'], 'empty subset oracle')
    require(mean_complete([1., None]) is None, 'missing mean imputed')
    require(gate(None)['status'] == 'pending' and gate(False)['status'] == 'failed', 'tri-state gates')
    require(not finite(float('nan')) and not finite(True), 'invalid scalar accepted')
    print(json.dumps({'status': 'synthetic_self_test_passed', 'future_data_opened': False}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--prepared', type=Path, help='optional; must match acquisition binding')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        if args.run is None or args.output is None:
            parser.error('--run and --output are required')
        sys.exit(analyze(args.run, args.output, args.prepared))
