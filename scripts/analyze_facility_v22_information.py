#!/usr/bin/env python3
"""Verify and analyze the completed early-coverage V22 information screen.

Read-only saved evidence analysis. Does not import a simulator or execute a
policy. Fixed observed-vote rules, a restricted hindsight information ceiling,
and actual full-task eligibility are reported separately. Aliases and reference
samplings never become additional independent physical experiments.
"""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import analyze_facility_v20_probe as old
from scripts import analyze_facility_v21_joint as joint
from scripts.analyze_facility_v20_probe import require, read, sha, digest, close, inside, verify_inventory, write_csv
from scripts.analyze_facility_v21_joint import atomic_json, finite, metric_check, roi_steps

DEFAULT_SOURCE = ROOT / 'eval_results/facility_v22_early_coverage_20260915'
DEFAULT_OUTPUT = ROOT / 'audit_results/facility_v22_information_analysis_20260915'
PARENTS = ('D19-P00', 'D19-P01')
ASSIGNMENTS = ('A_complex_B_simple', 'A_simple_B_complex')
OPTIONS = ('N', 'A', 'B')
REFERENCES = (2026, 2027, 2028)
PATH_FIELDS = ('pose', 'states', 'actions', 'outbound_states', 'outbound_actions',
    'return_states', 'return_actions', 'outbound_cost', 'return_cost', 'cost',
    'arrival_action', 'return_anchor')
NOTES = [
    'Ten unique physical trajectories represent twelve declared arms in two parents and two assignments. P00 N=B is a pre-outcome exact-path alias, not an extra experiment.',
    'The actual initial selection is STGHP/select_initial_coverage_v22. Its preceding select_topo_target is an unexecuted common N proposal. All continuation selections use N.',
    'Primary reward is true evaluation C2D times equal-six-facility external F1@5cm, reference 2026. Missing facilities score zero. References 2027/2028 are sensitivity checks on the same trajectories.',
    'All failures are retained. Qualified endpoint comparisons require C2D>=0.8, return, zero collisions, no terminal failure and primitive budget compliance.',
    'The fixed S_complex rule selects the observed A/B role with canonical class_vote<0; flipped selects class_vote>0. Neither rule reads assignment names. Rules query predeclared complete physical-arm returns, not a deployed dynamic or learned semantic policy.',
    'A fixed rule is task-qualified only when both selected assignment endpoints qualify. Its raw means remain visible even when qualification fails.',
    'Restricted oracle information value compares a single fixed option across assignments against an observed-vote-group-conditioned hindsight option. Both use exactly the same options that qualify in both assignments. No common option makes this comparison unavailable.',
    'The restricted oracle is an upper bound within that common feasible option pool only. A fixed semantic rule selecting options outside that pool is separately reported and cannot be bounded by it.',
    'The development information gate requires strictly more than 5% relative primary reward improvement in each of the two parents; exact 5% is excluded. It does not establish learned, dynamic semantic, strong-G or full-architecture efficacy.',
    'Observed labels are artificial RGB markers. Shared initial nonsemantic sensor identity is checked; equal initial inputs do not imply equal future structure or sensor histories.',
    'C_ROI is known free-or-occupied cells over a fixed public ROI. It is neither true reachable-floor coverage nor a certified bound. No posthoc correction or threshold adjustment is applied.',
    'Coverage is recorded every paid action. Quality is recorded every 20 actions and at termination. Plot connections do not create extra measured checkpoints or per-view causal gains.',
]


def physical_path(selected):
    path = {key: selected[key] for key in PATH_FIELDS}
    require(path['cost'] == path['outbound_cost'] + path['return_cost'], 'Initial total path cost differs')
    for name, cost in (('actions', 'cost'), ('outbound_actions', 'outbound_cost'), ('return_actions', 'return_cost')):
        require(len(path[name]) == path[cost], 'Initial paid path length differs: ' + name)
    return path


def strict_five_percent(relative):
    return relative is not None and relative > .05 and not math.isclose(relative, .05, rel_tol=0, abs_tol=1e-12)


def difference(value, baseline):
    if value is None or baseline is None:
        return dict(available=False, absolute=None, percentage_points=None, relative=None, strict_over_five_percent=False)
    delta = value-baseline
    relative = delta/baseline if baseline > 0 else None
    return dict(available=True, absolute=delta, percentage_points=100*delta, relative=relative,
        relative_unavailable_reason='nonpositive_baseline' if relative is None else None,
        strict_over_five_percent=strict_five_percent(relative))


def observed_rule(votes, sign):
    """A fixed rule on measured votes; assignment identifiers are not inputs."""
    require(set(votes) == {'A', 'B'}, 'Two observed A/B votes required')
    values = {key: finite(value, 'observed class vote') for key, value in votes.items()}
    matches = [key for key in ('A', 'B') if values[key]*sign > 0]
    return matches[0] if len(matches) == 1 else None


def parent_information(rows, reference_seed):
    """Six declarations, two assignments; aliases collapse by physical path.

    This function reads no result file and can be checked with synthetic rows.
    Every row includes its actual, unfiltered qualification flag and C/Q/J.
    """
    require(len(rows) == 6, 'Each parent needs all six N/A/B declarations')
    by = {(row['assignment'], row['first_option']): row for row in rows}
    require(set(by) == set(itertools.product(ASSIGNMENTS, OPTIONS)), 'Missing/duplicate declared arm')
    require(len({row['parent'] for row in rows}) == 1, 'Mixed parents')
    parent = rows[0]['parent']
    votes = {}
    for assignment in ASSIGNMENTS:
        values = [by[assignment, name]['observed_category_votes'] for name in OPTIONS]
        require(all(value == values[0] for value in values), 'Initial observed votes differ across arms')
        votes[assignment] = values[0]
        for vote in values[0].values(): finite(vote, 'observed vote')
    # The same fixed geometric option must exist in both assignments. Partition
    # aliases by the complete path, never by observed reward or endpoint mesh.
    canonical, aliases, representatives = {}, {}, []
    for option in OPTIONS:
        keys = tuple(by[a, option]['initial_path_sha256'] for a in ASSIGNMENTS)
        require(keys[0] == keys[1], 'Initial option path differs across paired nonsemantic inputs')
        previous = next((name for name in representatives if
            by[ASSIGNMENTS[0], name]['initial_path_sha256'] == keys[0]), None)
        name = option if previous is None else previous
        if previous is None: representatives.append(name); aliases[name] = []
        aliases[name].append(option); canonical[option] = name
        if previous is not None:
            for assignment in ASSIGNMENTS:
                x, y = by[assignment, option], by[assignment, name]
                require(x['unique_case_index'] == y['unique_case_index'], 'Same-path alias ran as an extra sample')
                require(all(x[k] == y[k] for k in ('C', 'Q', 'J', 'eligible')), 'Alias endpoint differs')
    common = [name for name in representatives if all(by[a, name]['eligible'] for a in ASSIGNMENTS)]
    mean = lambda name, assignments: sum(by[a, name]['J'] for a in assignments)/len(assignments)
    fixed = min(common, key=lambda name: (-mean(name, ASSIGNMENTS), OPTIONS.index(name))) if common else None
    blind = mean(fixed, ASSIGNMENTS) if fixed is not None else None
    groups = defaultdict(list)
    for assignment in ASSIGNMENTS:
        groups[tuple(votes[assignment][name] for name in ('A', 'B'))].append(assignment)
    group_results = []
    for cue, assignments in groups.items():
        chosen = min(common, key=lambda name: (-mean(name, assignments), OPTIONS.index(name))) if common else None
        group_results.append(dict(observed_votes=dict(zip(('A', 'B'), cue)), assignments=assignments,
            chosen_canonical_option=chosen, choice_aliases=[] if chosen is None else aliases[chosen],
            group_value=None if chosen is None else mean(chosen, assignments),
            weight=len(assignments)/len(ASSIGNMENTS)))
    conditional = sum(x['group_value']*x['weight'] for x in group_results) if common else None
    info = difference(conditional, blind)
    policies = {}
    def policy(name, choices):
        selected = [by[a, choices[a]] for a in ASSIGNMENTS if choices[a] is not None]
        available = len(selected) == len(ASSIGNMENTS)
        qualified = available and all(row['eligible'] for row in selected)
        raw = {key: sum(row[key] for row in selected)/2 if available else None for key in ('C', 'Q', 'J')}
        policies[name] = dict(choices=choices, available=available,
            unavailable_reason=None if available else 'observed_rule_ambiguous_or_no_common_feasible_option',
            all_assignments_eligible=qualified, choices_in_common_feasible_pool=available and all(canonical[choices[a]] in common for a in ASSIGNMENTS),
            raw_mean=raw, qualified_mean=raw if qualified else dict(C=None, Q=None, J=None),
            selected_cases=[{key: row[key] for key in ('assignment', 'first_option', 'unique_case_index', 'C', 'Q', 'J', 'eligible')} for row in selected])
    policy('N', {a: 'N' for a in ASSIGNMENTS})
    policy('S_complex', {a: observed_rule(votes[a], -1) for a in ASSIGNMENTS})
    policy('flipped_simple', {a: observed_rule(votes[a], 1) for a in ASSIGNMENTS})
    policy('best_fixed_nonsemantic', {a: fixed for a in ASSIGNMENTS})
    comparisons = {}
    for baseline in ('N', 'best_fixed_nonsemantic', 'flipped_simple'):
        comparisons['S_complex_vs_' + baseline] = dict(
            raw=difference(policies['S_complex']['raw_mean']['J'], policies[baseline]['raw_mean']['J']),
            qualified=difference(policies['S_complex']['qualified_mean']['J'], policies[baseline]['qualified_mean']['J']))
    d = [by[a, 'A']['J']-by[a, 'B']['J'] for a in ASSIGNMENTS]
    all_ab_qualified = all(by[a, name]['eligible'] for a in ASSIGNMENTS for name in ('A', 'B'))
    utility = lambda a, name: by[a, name]['J'] if by[a, name]['eligible'] else 0.
    ud = [utility(a, 'A')-utility(a, 'B') for a in ASSIGNMENTS]
    u_fixed = max(sum(utility(a, name) for a in ASSIGNMENTS)/2 for name in representatives)
    u_conditional = sum(max(sum(utility(a, name) for a in assignments)/len(assignments)
        for name in representatives)*len(assignments)/2 for assignments in groups.values())
    return dict(parent=parent, reference_seed=reference_seed, canonical_options=representatives,
        aliases=aliases, common_feasible_options=common, available=bool(common),
        unavailable_reason=None if common else 'no_option_eligible_in_both_assignments',
        fixed_choice=fixed, fixed_value=blind, conditional_value=conditional,
        observed_groups=group_results, information_value=info,
        restricted_oracle_scope='Both fixed and conditional choices use exactly the both-assignment-qualified canonical pool.',
        fixed_rules=policies, fixed_rule_comparisons=comparisons,
        route_assignment_interaction=dict(A_minus_B_by_assignment=dict(zip(ASSIGNMENTS, d)),
            raw_difference_of_differences=d[0]-d[1], all_AB_endpoints_eligible=all_ab_qualified,
            qualified_difference_of_differences=d[0]-d[1] if all_ab_qualified else None,
            eligibility_zero_U={a: {name: utility(a, name) for name in ('A', 'B')} for a in ASSIGNMENTS},
            U_difference_of_differences=ud[0]-ud[1], causal_semantic_effect_claimed=False),
        failure_zero_boundary_only=dict(fixed_value=u_fixed, observed_conditioned_value=u_conditional,
            difference=difference(u_conditional, u_fixed), used_for_primary_gate=False,
            scope='All canonical choices; ineligible reward replaced by zero only for this secondary boundary diagnostic.'),
        all_declared_endpoints_retained=True)


def verify_source(source):
    manifest = read(source/'manifest.json')
    require(manifest.get('status') == 'complete', 'Formal analysis requires a complete frozen V22 batch')
    require(manifest['version'] == 'facility-early-coverage-v22-information-screen-1' and manifest['mode'] == manifest['continuation_mode'] == 'N', 'Expected early V22 coverage screen')
    config = manifest['config']
    expected = dict(parents=list(PARENTS), assignments=list(ASSIGNMENTS), first_options=list(OPTIONS),
        budgets=dict(zip(PARENTS, (160, 184))), coverage_minimum=.8,
        reference_seeds=list(REFERENCES), primary_reference_seed=2026,
        task_asset_count=6, quality_curve_action_stride=20, coverage_trace_action_stride=1,
        sensor_model='iid_025px', noise_seed=1901, first_choice_action_id=0, initial_history_actions=0,
        continuation_mode='N', expected_declared_cases=12, expected_unique_cases=10,
        physical_contract=dict(max_depth_m=5., voxel_m=.04, truncation_m=.12))
    for name, value in expected.items(): require(config[name] == value, 'Frozen screen contract changed: ' + name)
    rule = config['fixed_semantic_rule']
    require(rule['input'] == 'actual initial v22_first_choice.observed_targets canonical class_vote'
        and rule['aligned'] == 'choose the unique A/B role with negative class_vote (complex prior)'
        and rule['flipped'] == 'choose the unique A/B role with positive class_vote'
        and rule['fitted'] is False and rule['deployed_online'] is False, 'Pre-outcome fixed semantic rule changed')
    proxy = config['planning_coverage_proxy']
    require(proxy['kind'] == 'known_cells_over_fixed_public_roi' and proxy['target_fraction'] == .8
        and proxy['evaluation_truth_used'] is False and proxy['true_coverage_guaranteed'] is False, 'Observed ROI proxy contract changed')
    for name in ('training_allowed', 'semantic_policy_deployed', 'strong_G_comparison_performed',
                 'full_architecture_efficacy_proven', 'old_N_metric_reuse_allowed', 'aliases_are_independent_experiments'):
        require(manifest[name] is False and config[name] is False, 'Unexpected efficacy/reuse flag: ' + name)
    require(manifest['semantic_information_test'] is True, 'Expected information screen')
    cases, declared = manifest['cases'], manifest['declared_cases']
    require(len(cases) == 10 and [c['index'] for c in cases] == list(range(10)), 'Ten unique physical cases required')
    require(len(declared) == 12 and [d['declared_index'] for d in declared] == list(range(12)), 'Twelve declarations required')
    require({(d['parent'], d['assignment'], d['first_option']) for d in declared} == set(itertools.product(PARENTS, ASSIGNMENTS, OPTIONS)), 'Declared Cartesian matrix differs')
    require(manifest['declared_to_unique'] == {str(d['declared_index']): d['unique_case_index'] for d in declared}, 'Explicit alias mapping differs')
    for row in [*cases, *declared]:
        require(row['mode'] == 'N' and row['budget'] == config['budgets'][row['parent']], 'Mode or budget differs')
        path = physical_path(row['initial_selected'])
        require(path == row['initial_path'] and digest(path) == row['initial_path_sha256'], 'Initial physical path/hash differs')
        choice = row['v22_first_choice']; selected = row['initial_selected']
        require(row['initial_candidate_pool_sha256'] == choice['common_pool_sha256'], 'Declared common pool identity differs')
        require(choice['requested_option'] == selected['v22_first_option'] == row['first_option'], 'Prepared first option differs')
        require(choice['chosen_candidate_id'] == selected['candidate_id'] == choice['option_map'][row['first_option']], 'Chosen initial candidate differs')
        require(choice['action_id'] == 0 and choice['initial_observation_only'] and not choice['class_used_for_ranking'] and not choice['evaluation_truth_used'] and not choice['semantic_policy_deployed'], 'Initial information boundary differs')
        targets = choice['observed_targets']
        require(len(targets) == 2 and [t['role'] for t in targets] == ['A', 'B'] and all(t['marked_points'] > 0 for t in targets), 'Both initial marked labels must be observed')
        require(row['observed_category_votes'] == {t['role']: t['class_vote'] for t in targets}, 'Observed class vote association differs')
        require(observed_rule(row['observed_category_votes'], -1) is not None and observed_rule(row['observed_category_votes'], 1) is not None, 'Fixed class rules require opposite nonzero observed votes')
        require(row['fixed_semantic_rule_choice'] == observed_rule(row['observed_category_votes'], -1)
            and row['flipped_semantic_rule_choice'] == observed_rule(row['observed_category_votes'], 1), 'Predeclared fixed rule differs from observed votes')
    for d in declared:
        case = cases[d['unique_case_index']]
        for key in ('parent', 'assignment', 'initial_path', 'initial_path_sha256', 'initial_packet_sha256', 'initial_nonsemantic_sha256', 'observed_category_votes'):
            require(d[key] == case[key], 'Alias changed physical condition: ' + key)
        expected_alias = None if d['first_option'] == case['first_option'] else case['first_option']
        require(d['alias_of_first_option'] == expected_alias, 'Alias representative differs')
    for parent in PARENTS:
        subset = [d for d in declared if d['parent'] == parent]
        require(len({d['initial_nonsemantic_sha256'] for d in subset}) == len({d['initial_candidate_pool_sha256'] for d in subset}) == 1, 'Paired initial nonsemantic input/pool differs')
        for assignment in ASSIGNMENTS:
            lookup = {d['first_option']: d for d in subset if d['assignment'] == assignment}
            paths = {k: d['initial_path_sha256'] for k, d in lookup.items()}
            ids = {k: d['unique_case_index'] for k, d in lookup.items()}
            require(len(set(paths.values())) == len(set(ids.values())) == (2 if parent == PARENTS[0] else 3), 'Unique path count differs')
            if parent == PARENTS[0]: require(ids['N'] == ids['B'] != ids['A'], 'Predeclared P00 N=B alias differs')
        for option in OPTIONS:
            require(len({d['initial_path_sha256'] for d in subset if d['first_option'] == option}) == 1, 'Fixed route differs across assignments')
    artifacts = verify_inventory(source)
    for case in cases:
        folder = source/f'case_{case["index"]:02d}'; verify_inventory(folder)
        receipt = read(folder/'verification.json')
        require(receipt['status'] == 'passed' and receipt['independent_process'] is True and receipt['fresh_world_sensor_action_metric_replay'] is True and receipt['physical_process_id'] != receipt['replay_process_id'], 'Every unique trajectory requires a fresh-process replay')
    frozen = manifest['source_sha256']
    require('docs/research/V22_EARLY_COVERAGE_PROTOCOL_20260915.md' in frozen, 'Pre-outcome analysis protocol was not frozen with physical sources')
    with zipfile.ZipFile(source/'sources.zip') as archive:
        require(set(archive.namelist()) == set(frozen), 'Frozen source inventory differs')
        for name, wanted in frozen.items(): require(hashlib.sha256(archive.read(name)).hexdigest() == wanted, 'Frozen source changed: ' + name)
        require(json.loads(archive.read('configs/virtual3d/facility_early_coverage_v22_probe.json')) == config, 'Archived config differs from declaration')
    reference = Path(manifest['reference_source_root'])
    require(sha(reference/'manifest.json') == manifest['reference_source_manifest_sha256'] and sha(reference/'artifact_hashes.json') == manifest['reference_source_inventory_sha256'], 'Reference source provenance differs')
    expected_refs = {f'references/{p}_{a}_{s}.npz' for p, a, s in itertools.product(PARENTS, ASSIGNMENTS, REFERENCES)}
    require(set(manifest['reference_sha256']) == expected_refs, 'Twelve fixed reference caches required')
    for name, wanted in manifest['reference_sha256'].items(): require(sha(inside(reference, name)) == wanted, 'Reference cache changed: ' + name)
    return manifest, dict(artifacts_verified=artifacts, unique_fresh_process_replays_verified=10,
        frozen_source_files_verified=len(frozen), reference_caches_verified=12,
        source_manifest_sha256=sha(source/'manifest.json'), source_inventory_sha256=sha(source/'artifact_hashes.json'),
        source_archive_is_authoritative=True)


def initial_nonsemantic_hash(path):
    import numpy as np
    names = dict(depth='frame__depth_m', scan='scan__ranges_m', intrinsic='frame__intrinsic',
        world_from_camera='frame__world_from_camera', world_from_laser='scan__world_from_laser',
        camera_timestamp='frame__timestamp_s', laser_timestamp='scan__timestamp_s',
        angle_min='scan__angle_min_rad', angle_increment='scan__angle_increment_rad', range_max='scan__range_max_m')
    with np.load(path, allow_pickle=False) as packet:
        return digest({key: packet[name].tolist() for key, name in names.items()})


def analyze_case(source, case):
    folder = source/f'case_{case["index"]:02d}'
    result = read(folder/'result.json'); calls = read(folder/'module_calls.json.gz')
    audit = read(folder/'runtime_audit.json.gz'); replay = read(folder/'verification.json')
    timing = read(folder/'timing.json'); n = result['paid_actions']
    base = {k: case[k] for k in ('index', 'parent', 'assignment', 'first_option', 'budget')}
    require(result['case'] == case and result['first_option'] == case['first_option'] and result['continuation_mode'] == 'N', 'Physical case declaration differs')
    require(result['semantic_information_test'] and not result['trained'] and not result['semantic_policy_deployed'] and not result['strong_G_comparison_performed'], 'Physical scope flags differ')
    require(digest([{k: v for k, v in c.items() if k != 'elapsed_s'} for c in calls]) == result['module_calls_sha256'], 'Module log digest differs')
    require(digest(audit) == result['runtime_audit_sha256'], 'Runtime log digest differs')
    require([c['call_id'] for c in calls] == list(range(1, len(calls)+1)), 'Module call IDs differ')
    for call in calls:
        require(digest(call['inputs']) == call['input_sha256'] and digest(call['outputs']) == call['output_sha256'], 'Module I/O digest differs')
        require(call['truth_used'] is False and call['trained'] is False, 'Module used truth/training')
    require({c['module'] for c in calls} == {'OV-SDF', 'STGHP', 'RPN-UQ', 'IGCR'}, 'Four module interfaces missing')
    require(len(result['actions']) == n == replay['actions'] == result['termination']['final_action_id'], 'Physical/replay/terminal counts differ')
    require(replay['physical_process_id'] == timing['process_id'], 'Physical process identity differs')
    require(result['primitive_budget_compliant'] == (n <= case['budget']), 'Primitive budget flag differs')
    require(sorted(p.name for p in (folder/'packets').glob('*.npz')) == [f'{i:04d}.npz' for i in range(n+1)], 'Raw packet sequence missing/duplicated')
    require(initial_nonsemantic_hash(folder/'packets/0000.npz') == case['initial_nonsemantic_sha256'], 'Measured initial nonsemantic identity differs')
    import numpy as np
    with np.load(folder/'final_mesh.npz', allow_pickle=False) as mesh:
        require(set(mesh.files) == set(result['final_mesh_sha256']), 'Mesh field inventory differs')
        for key, wanted in result['final_mesh_sha256'].items():
            arr = np.ascontiguousarray(mesh[key])
            require(hashlib.sha256(f'{arr.dtype.str}:{arr.shape}:'.encode()+arr.tobytes()).hexdigest() == wanted, 'Final mesh array differs')
    trace = result['coverage_trace']
    require([r['action_id'] for r in trace] == list(range(n+1)), 'Coverage trace must include every paid action')
    for row in trace: require(0 <= finite(row['coverage_2d'], 'coverage trace') <= 1, 'Invalid coverage trace')
    steps = [{**base, **row} for row in roi_steps(calls, trace, n)]
    firsts = [c for c in calls if c['method'] == 'select_initial_coverage_v22']
    require(len(firsts) == 1 and firsts[0]['action_id'] == 0, 'One actual action-zero V22 selection required')
    first = firsts[0]; selected = first['outputs']['selected']; choice = first['outputs']['v22_first_choice']
    require(selected == case['initial_selected'] and choice == case['v22_first_choice'], 'Actual first selection differs from preparation')
    require(selected['selection_call_id'] == first['call_id'] and physical_path(selected) == case['initial_path'], 'First executed selection binding differs')
    proposal_id = first['inputs']['nominal_selection_call_id']
    require(0 < proposal_id < first['call_id'], 'Nominal proposal must precede actual selection')
    proposal = calls[proposal_id-1]
    require(proposal['method'] == 'select_topo_target' and proposal['action_id'] == 0 and proposal['outputs']['selected']['candidate_id'] == choice['nominal_N_candidate_id'], 'Nominal N proposal differs')
    require(choice['common_pool_sha256'] == digest(first['outputs']['candidates']) == digest(proposal['outputs']['candidates']), 'Actual first choice used a different candidate pool')
    require(not choice['intermediate_proposal_executed'] and choice['common_continuation'] == 'V21_N', 'Intervention boundary changed')
    options = {}; selection_rows = []; actual_selections = []
    for call in calls:
        out = call['outputs']; method = call['method']; action_id = call['action_id']
        if method not in ('select_topo_target', 'select_initial_coverage_v22'): continue
        if call['call_id'] == proposal_id: continue
        require(out['mode'] == 'N', 'Continuation mode differs')
        if method == 'select_topo_target':
            require(action_id > 0 and out['effective_objective'] == 'N', 'Post-initial continuation must be common N')
        else:
            require(out['effective_objective'] == 'declared_initial_coverage_option', 'Actual first objective differs')
        option = out['selected']; actual_selections.append(option)
        if option is not None:
            require(option['selection_call_id'] == call['call_id'] and option['option_id'] not in options, 'Selected option binding duplicated')
            require(option['v21_coverage_intent'] is True and option['group'].startswith('coverage_'), 'V22 selected a noncoverage role')
            options[option['option_id']] = dict(option=option, selected_action_id=action_id,
                paid_action_ids=[], arrival_action_ids=[], refresh_cancel_action_ids=[], safety_denial_action_ids=[])
        state = out['coverage_state']; current = steps[action_id]
        for key in ('known_cells', 'task_cells', 'deficit_cells'): require(state[key] == current[key], 'Selection ROI count differs')
        close(state['C_plan'], current['C_ROI'], 'Selection ROI ratio')
        selection_rows.append({**base, 'action_id': action_id, 'selection_call_id': call['call_id'],
            'method': method, 'option_id': None if option is None else option['option_id'],
            'candidate_id': None if option is None else option['candidate_id'],
            'target': None if option is None else option['pose'], 'effective_objective': out['effective_objective'],
            'C_ROI': current['C_ROI'], 'coverage_2d': current['coverage_2d']})
    require([row['option'] for row in audit if row['event'] == 'global_choice'] == actual_selections, 'Runtime installed proposal rather than actual selection')
    refreshes = [c['outputs'] for c in calls if c['method'] == 'refresh_observed_route_v21']
    require([{k: v for k, v in x.items() if k != 'event'} for x in audit if x['event'] == 'v21_route_update'] == refreshes, 'Module/runtime refresh differs')
    for row in refreshes:
        require(row['option_id'] in options, 'Refresh references an unknown option')
        if not row['target_retained']: options[row['option_id']]['refresh_cancel_action_ids'].append(row['action_id'])
    for call in calls:
        if call['method'] == 'assess_local_action' and not call['outputs']['allowed'] and call.get('executed_option_id') in options:
            options[call['executed_option_id']]['safety_denial_action_ids'].append(call['action_id'])
    authorizations = [row for row in audit if row['event'] == 'paid_action_authorized']
    observations = [row for row in audit if row['event'] == 'observation']
    require([row['next_action_id'] for row in authorizations] == [row['action_id'] for row in observations] == list(range(1, n+1)), 'Paid authorization/observation sequence differs')
    paid_counts = Counter()
    for i, (auth, obs, action) in enumerate(zip(authorizations, observations, result['actions']), 1):
        assessment = auth['assessment']; oid = auth['option_id']; item = options.get(oid)
        require(assessment['allowed'] and assessment['action'] == action['action'] and assessment['remaining_budget'] == case['budget']-i+1, 'Paid authorization differs')
        require(obs['packet_sha256'] == action['packet_sha256'] and obs['collision'] == action['collision'], 'Paid observed packet differs')
        if not action['collision']: require(assessment['next_pose'] == action['pose'], 'Paid pose differs from authorization')
        require(assessment['reserved_return_cost'] is not None and assessment['reserved_return_cost'] <= case['budget']-i, 'Paid return reserve violated')
        require(oid is None or item is not None, 'Paid action refers to unexecuted proposal/unknown option')
        if item is not None:
            require(auth['selection_call_id'] == item['option']['selection_call_id'], 'Paid selection-call binding differs')
            item['paid_action_ids'].append(i)
        if obs['arrived']:
            require(item is not None and auth['phase'] == 'outbound' and not action['collision'] and action['pose'] == item['option']['pose'], 'Measured arrival differs')
            item['arrival_action_ids'].append(i)
        kind = 'return' if auth['phase'] == 'return' else 'coverage_outbound'
        paid_counts[kind] += 1
        steps[i].update(action=action['action'], phase=auth['phase'], option_id=oid, arrived=obs['arrived'], collision=action['collision'])
    collisions = sum(int(x['collision']) for x in result['actions'])
    require(collisions == result['collisions'], 'Collision count differs')
    first_item = options[selected['option_id']]
    recorded = result['initial_option_outcome']
    actual_paid = [a['next_action_id'] for a in authorizations if a['option_id'] == selected['option_id'] and a['phase'] == 'outbound']
    cancel_events = [row for row in audit if row.get('option_id') == selected['option_id'] and
        ((row.get('event') == 'v21_route_update' and not row.get('target_retained', True)) or 'denied' in row.get('event', ''))]
    expected_outcome = dict(option_id=selected['option_id'], declared_first_option=case['first_option'],
        initial_target=selected['pose'], paid_outbound_action_ids=actual_paid, paid_outbound_actions=len(actual_paid),
        arrival_action_ids=first_item['arrival_action_ids'], measured_arrived=bool(first_item['arrival_action_ids']),
        cancellation_or_denial_events=cancel_events, explicit_cancel_or_denial=bool(cancel_events),
        target_pose_ever_observed=any(a['pose'] == selected['pose'] for a in result['actions']),
        target_pose_visit_is_not_attributed_quality_gain=True)
    require(recorded == expected_outcome, 'Recorded first-option outcome differs')
    first_outcome = {**base, **recorded, 'selection_call_id': first['call_id'],
        'refresh_cancel_action_ids': first_item['refresh_cancel_action_ids'],
        'safety_denial_action_ids_from_bound_module_calls': first_item['safety_denial_action_ids'],
        'explicit_cancel_or_safety_denial_including_module_calls': bool(cancel_events or first_item['safety_denial_action_ids']),
        'observed_targets': choice['observed_targets'], 'causal_quality_gain': None,
        'causal_quality_gain_reason': 'Shared TSDF checkpoints cannot attribute gain to one selected view.'}
    term = result['termination']; endpoints = []
    require(set(result['before']) == set(result['after']) == {str(s) for s in REFERENCES}, 'Three reference results required')
    for seed in REFERENCES:
        before, after = result['before'][str(seed)], result['after'][str(seed)]
        metric_check(before); metric_check(after)
        close(before['coverage_2d'], trace[0]['coverage_2d'], 'Initial coverage')
        close(after['coverage_2d'], trace[-1]['coverage_2d'], 'Terminal coverage')
        require(after['returned'] == term['returned_to_anchor'] and after['failed'] == term['failed'] and after['collisions'] == collisions, 'Eligibility term fields differ')
        eligible = after['eligible'] and result['primitive_budget_compliant']
        endpoints.append({**base, 'reference_seed': seed, 'C': after['coverage_2d'],
            'Q': after['05cm']['external_macro_f1'], 'J': after['05cm']['joint_external'],
            'U_eligibility_boundary': after['05cm']['joint_external'] if eligible else 0.,
            'eligible': eligible, 'evaluator_eligible': after['eligible'], 'returned': after['returned'],
            'failed': term['failed'], 'collisions': collisions, 'paid_actions': n,
            'primitive_budget_compliant': result['primitive_budget_compliant'],
            'termination_reason': term['reason'], 'missing_asset_count': after['missing_asset_count']})
    quality = []; curve = result['quality_curve']
    require(result['curve_action_stride'] == 20 and result['coverage_trace_action_stride'] == 1, 'Curve stride differs')
    require([row['action_id'] for row in curve] == sorted(set([*range(0, n+1, 20), n])), 'Quality checkpoints missing/duplicated')
    for row in curve:
        metric_check(row['metrics']); m = row['metrics']; i = row['action_id']
        close(m['coverage_2d'], trace[i]['coverage_2d'], 'Checkpoint coverage')
        quality.append({**base, 'action_id': i, 'reference_seed': 2026, 'C': m['coverage_2d'],
            'Q': m['05cm']['external_macro_f1'], 'J': m['05cm']['joint_external'],
            'eligible': m['eligible'], 'missing_asset_count': m['missing_asset_count']})
    require(curve[0]['metrics'] == result['before']['2026'] and curve[-1]['metrics'] == result['after']['2026'], 'Curve endpoints differ')
    hits = [row['action_id'] for row in trace if row['coverage_2d'] >= .8]
    require(result['first_evaluator_coverage_gate_action'] == (min(hits) if hits else None), 'First true coverage crossing differs')
    primary = endpoints[0]
    summary = {**base, 'paid_actions': n, 'C': primary['C'], 'Q': primary['Q'], 'J': primary['J'],
        'eligible': primary['eligible'], 'returned': term['returned_to_anchor'], 'failed': term['failed'],
        'collisions': collisions, 'primitive_budget_compliant': result['primitive_budget_compliant'],
        'termination_reason': term['reason'], 'first_evaluator_coverage_gate_action': min(hits) if hits else None,
        'final_C_ROI': steps[-1]['C_ROI'], 'actual_global_decisions': len(selection_rows),
        'unexecuted_initial_proposals': 1, 'paid_action_counts': dict(paid_counts),
        'first_option_outcome': first_outcome, 'trajectory_sha256': digest(result['actions'])}
    return dict(summary=summary, endpoints=endpoints, steps=steps, quality=quality,
        selections=selection_rows, initial_outcome=first_outcome, actions=result['actions'], result=result)


def first_difference(left, right, fields):
    for i, (x, y) in enumerate(zip(left, right), 1):
        if any(x[k] != y[k] for k in fields): return i
    return None


def divergence_rows(manifest, data):
    by = {(d['parent'], d['assignment'], d['first_option']): d for d in manifest['declared_cases']}
    rows = []
    for parent, assignment in itertools.product(PARENTS, ASSIGNMENTS):
        for a, b in itertools.combinations(OPTIONS, 2):
            da, db = by[parent, assignment, a], by[parent, assignment, b]
            left, right = data[da['unique_case_index']]['actions'], data[db['unique_case_index']]['actions']
            rows.append(dict(parent=parent, assignment=assignment, left_option=a, right_option=b,
                left_unique_case_index=da['unique_case_index'], right_unique_case_index=db['unique_case_index'],
                exact_path_alias=da['unique_case_index'] == db['unique_case_index'],
                first_paid_command_difference=first_difference(left, right, ('action',)),
                first_measured_pose_difference=first_difference(left, right, ('pose',)),
                first_command_or_pose_difference=first_difference(left, right, ('action', 'pose')),
                shared_compared_paid_length=min(len(left), len(right)),
                different_termination_length=len(left) != len(right),
                left_paid_actions=len(left), right_paid_actions=len(right)))
    return rows


def paired_revelation(source, manifest, data):
    """Inspect this batch's saved sensors only; no reuse of old N's window."""
    import numpy as np
    by = {(d['parent'], d['assignment'], d['first_option']): d for d in manifest['declared_cases']}
    fields = ('frame__depth_m', 'scan__ranges_m', 'frame__intrinsic', 'frame__world_from_camera',
              'scan__world_from_laser', 'scan__range_max_m', 'scan__angle_min_rad', 'scan__angle_increment_rad')
    result = []
    for parent in PARENTS:
        done = set()
        for option in OPTIONS:
            pair = [by[parent, a, option] for a in ASSIGNMENTS]
            path = pair[0]['initial_path_sha256']
            if path in done: continue
            done.add(path)
            indices = [d['unique_case_index'] for d in pair]
            acts = [data[i]['actions'] for i in indices]
            control_difference = first_difference(*acts, ('action', 'pose'))
            limit = min(len(a) for a in acts); first = None; different_fields = []
            for action_id in range(limit+1):
                paths = [source/f'case_{i:02d}/packets/{action_id:04d}.npz' for i in indices]
                with np.load(paths[0], allow_pickle=False) as left, np.load(paths[1], allow_pickle=False) as right:
                    different_fields = [name for name in fields if not np.array_equal(left[name], right[name])]
                if different_fields: first = action_id; break
            aliases = [name for name in OPTIONS if by[parent, ASSIGNMENTS[0], name]['initial_path_sha256'] == path]
            result.append(dict(parent=parent, canonical_first_option=option, aliases=aliases,
                unique_case_indices=indices, first_nonsemantic_sensor_difference_action=first,
                fields_different_at_first=different_fields,
                initial_nonsemantic_identical=first != 0,
                first_command_or_pose_difference=control_difference,
                shared_motion_through_first_sensor_difference=first is not None and (control_difference is None or control_difference > first),
                matched_sensor_prefix_last_action=limit if first is None else first-1,
                compared_through_action=limit if first is None else first,
                censoring='No difference within common recorded action length' if first is None else None,
                source_is_current_V22_saved_trajectory=True, old_N_revelation_window_reused=False,
                interpretation='A sensor difference after motion divergence is confounded by viewpoint; this is a boundary diagnostic, not semantic gain.'))
    return result


def validate_aggregate(source, manifest, data):
    aggregate = read(source/'result.json')
    require(aggregate['status'] == 'complete' and aggregate['unique_physical_trajectories'] == 10
        and aggregate['declared_option_cases'] == 12 and aggregate['independent_process_replays'] == 10, 'Complete aggregate cardinalities differ')
    require(len(aggregate['cases']) == 10 and len(aggregate['declared_cases']) == 12, 'Aggregate omitted failures/aliases')
    require(aggregate['declared_to_unique'] == manifest['declared_to_unique'], 'Aggregate alias map differs')
    for row, item in zip(aggregate['cases'], data):
        summary = item['summary']; primary = item['endpoints'][0]
        fields = dict(index=summary['index'], parent=summary['parent'], assignment=summary['assignment'],
            first_option=summary['first_option'], continuation_mode='N', budget=summary['budget'],
            paid_actions=summary['paid_actions'], coverage_2d=summary['C'], external_macro_f1=summary['Q'],
            joint_external=summary['J'], eligible=primary['evaluator_eligible'], returned_to_anchor=summary['returned'],
            collisions=summary['collisions'], failed=summary['failed'], primitive_budget_compliant=summary['primitive_budget_compliant'],
            initial_option_outcome=item['result']['initial_option_outcome'], replay_status='passed', replay_actions=summary['paid_actions'])
        require(row == fields, 'Aggregate unique endpoint differs')
    for declared, row in zip(manifest['declared_cases'], aggregate['declared_cases']):
        require(row == {**declared, 'endpoint': aggregate['cases'][declared['unique_case_index']]}, 'Aggregate declaration endpoint differs')
    paid = sum(d['summary']['paid_actions'] for d in data)
    require(aggregate['total_physical_actions'] == aggregate['total_replay_actions'] == paid, 'Aggregate paid action count differs')
    eligible = sum(d['endpoints'][0]['evaluator_eligible'] for d in data)
    require(aggregate['eligible_unique_count'] == eligible and aggregate['common_online_coverage_passed'] == all(d['summary']['eligible'] for d in data), 'Aggregate qualification count differs')
    require(aggregate['eligible_declared_count'] == sum(data[d['unique_case_index']]['endpoints'][0]['evaluator_eligible'] for d in manifest['declared_cases']), 'Aggregate alias qualification count differs')


def make_plots(output, data, declared):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = dict(N='#444444', A='#0072B2', B='#D55E00')
    for filename, key, label in [('coverage_curves.png', 'coverage_2d', 'True reachable-floor coverage C2D'),
                                  ('exterior_quality_curves.png', 'Q', 'Equal-six external F1 at 5 cm')]:
        fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharey=True)
        for ax, (parent, assignment) in zip(axes.flat, itertools.product(PARENTS, ASSIGNMENTS)):
            ds = [d for d in declared if d['parent'] == parent and d['assignment'] == assignment]
            done = set()
            for d in ds:
                i = d['unique_case_index']
                if i in done: continue
                done.add(i); item = data[i]
                aliases = '='.join(x['first_option'] for x in ds if x['unique_case_index'] == i)
                rows = item['steps'] if key == 'coverage_2d' else item['quality']
                ax.plot([r['action_id'] for r in rows], [r[key] for r in rows], color=colors[d['first_option']],
                    label=aliases+(' [ineligible]' if not item['summary']['eligible'] else ''),
                    marker=None if key == 'coverage_2d' else '.', linewidth=1.4)
            if key == 'coverage_2d': ax.axhline(.8, color='#888888', linestyle='--', linewidth=.8)
            ax.set(title=parent+' / '+assignment.replace('_', ' '), xlabel='Paid actions', ylabel=label, ylim=(0, 1))
            ax.grid(alpha=.2); ax.legend(fontsize=8)
        fig.suptitle('V22 early original coverage route; common N continuation — all cases retained', fontsize=11)
        fig.tight_layout(); fig.savefig(output/filename, dpi=135); plt.close(fig)


def self_test():
    def matrix():
        rows = []
        for ai, assignment in enumerate(ASSIGNMENTS):
            for oi, option in enumerate(OPTIONS):
                value = {'N': .5, 'A': .8 if ai == 0 else .4, 'B': .4 if ai == 0 else .8}[option]
                rows.append(dict(parent=PARENTS[1], assignment=assignment, first_option=option,
                    unique_case_index=ai*3+oi, initial_path_sha256=option, observed_category_votes=dict(A=-1 if ai == 0 else 1, B=1 if ai == 0 else -1),
                    C=1., Q=value, J=value, eligible=True))
        return rows
    rows = matrix(); p = parent_information(rows, 2026)
    close(p['fixed_value'], .6, 'Synthetic best fixed'); close(p['conditional_value'], .8, 'Synthetic conditional')
    require(p['information_value']['strict_over_five_percent'] and p['fixed_rules']['S_complex']['all_assignments_eligible'], 'Visible information/fixed rule failed')
    require(p['fixed_rules']['S_complex']['choices'] == dict(zip(ASSIGNMENTS, ('A', 'B'))), 'Rule used wrong vote sign')
    same = deepcopy(rows)
    for row in same: row['observed_category_votes'] = dict(A=-1, B=1)
    require(abs(parent_information(same, 2026)['information_value']['absolute']) < 1e-12, 'Invisible assignments gave information value')
    failed = deepcopy(rows); failed[4]['eligible'] = False
    p = parent_information(failed, 2026)
    require('A' not in p['common_feasible_options'] and all(g['chosen_canonical_option'] != 'A' for g in p['observed_groups']), 'One-assignment failed arm entered restricted oracle')
    require(p['fixed_rules']['S_complex']['all_assignments_eligible'] and not p['fixed_rules']['S_complex']['choices_in_common_feasible_pool'], 'Outside-pool fixed rule qualification lost')
    empty = deepcopy(rows)
    for row in empty: row['eligible'] = row['assignment'] == ASSIGNMENTS[0]
    p = parent_information(empty, 2026)
    require(not p['available'] and p['information_value']['relative'] is None, 'No common pool should be unavailable')
    zeros = deepcopy(rows)
    for row in zeros: row['Q'] = row['J'] = 0.
    require(parent_information(zeros, 2026)['information_value']['relative'] is None, 'Zero denominator became a finite improvement')
    aliases = deepcopy(rows)
    for ai in (0, 1):
        aliases[ai*3+2].update({k: aliases[ai*3][k] for k in ('unique_case_index', 'initial_path_sha256', 'C', 'Q', 'J', 'eligible')})
    require(parent_information(aliases, 2026)['canonical_options'] == ['N', 'A'], 'N=B alias counted as a third experiment')
    require(not strict_five_percent(.05) and strict_five_percent(.051), 'Strict 5% boundary failed')
    try: parent_information(rows[:-1], 2026)
    except ValueError: pass
    else: raise AssertionError('Missing declaration silently accepted')
    print('Nine synthetic behavior checks passed; no experiment or evidence read/written.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--self-test', action='store_true', help='Synthetic behavior only; no evidence reads/writes')
    args = parser.parse_args()
    if args.self_test: self_test(); return
    source, output = args.source.resolve(), args.output.resolve()
    require(not output.is_relative_to(source) and not source.is_relative_to(output), 'Output must be separate from evidence')
    source_paths = [Path(__file__).resolve(), Path(old.__file__).resolve(), Path(joint.__file__).resolve()]
    analysis_hashes = {str(p.relative_to(ROOT)): sha(p) for p in source_paths}
    manifest, checks = verify_source(source)
    data = [analyze_case(source, case) for case in manifest['cases']]
    validate_aggregate(source, manifest, data)
    declared_rows = []
    for d in manifest['declared_cases']:
        for endpoint in data[d['unique_case_index']]['endpoints']:
            declared_rows.append({**endpoint, 'first_option': d['first_option'], 'unique_case_index': d['unique_case_index'],
                'declared_index': d['declared_index'], 'alias_of_first_option': d['alias_of_first_option'],
                'initial_path_sha256': d['initial_path_sha256'], 'observed_category_votes': d['observed_category_votes'],
                'fixed_semantic_rule_choice': observed_rule(d['observed_category_votes'], -1),
                'flipped_semantic_rule_choice': observed_rule(d['observed_category_votes'], 1)})
    information = [parent_information([r for r in declared_rows if r['parent'] == parent and r['reference_seed'] == seed], seed)
        for seed in REFERENCES for parent in PARENTS]
    primary = [row for row in information if row['reference_seed'] == 2026]
    divergence = divergence_rows(manifest, data); revelation = paired_revelation(source, manifest, data)
    gates = dict(restricted_oracle_both_parents_strict_over_five_percent=all(p['information_value']['strict_over_five_percent'] for p in primary),
        fixed_S_vs_N_both_parents_qualified_strict_over_five_percent=all(p['fixed_rule_comparisons']['S_complex_vs_N']['qualified']['strict_over_five_percent'] for p in primary),
        fixed_S_vs_best_fixed_both_parents_qualified_strict_over_five_percent=all(p['fixed_rule_comparisons']['S_complex_vs_best_fixed_nonsemantic']['qualified']['strict_over_five_percent'] for p in primary),
        fixed_S_vs_flipped_both_parents_qualified_strict_over_five_percent=all(p['fixed_rule_comparisons']['S_complex_vs_flipped_simple']['qualified']['strict_over_five_percent'] for p in primary),
        all_four_N_endpoints_eligible=all(p['fixed_rules']['N']['all_assignments_eligible'] for p in primary),
        all_four_fixed_S_endpoints_eligible=all(p['fixed_rules']['S_complex']['all_assignments_eligible'] for p in primary),
        thresholds_changed_after_outcomes=False, strong_G_or_learned_semantic_efficacy_gate=False)
    summary = dict(status='complete', source_root=str(source), checks=checks,
        unique_physical_trajectories=10, declared_option_cases=12, independent_process_replays=10,
        independent_parent_layouts=2, reference_sensitivities_per_trajectory=3,
        total_physical_actions=sum(d['summary']['paid_actions'] for d in data),
        eligible_unique_count=sum(d['summary']['eligible'] for d in data),
        all_cases_retained=True, unique_cases=[d['summary'] for d in data],
        primary_information=primary, reference_sensitivities=information, development_gates=gates,
        initial_action_divergence=divergence, current_saved_history_revelation=revelation,
        semantic_information_screen=True, fixed_semantic_rule_is_predeclared_matrix_query=True,
        semantic_policy_deployed=False, trained=False, semantic_efficacy_proven=False,
        semantic_information_gain_proven=False, joint_planning_efficacy_proven=False,
        strong_G_comparison_performed=False, full_architecture_efficacy_proven=False,
        aliases_are_independent_experiments=False, reference_sampling_repeats_are_independent_experiments=False,
        physical_experiments_started_by_analyzer=False, interpretation=NOTES)
    require(all(sha(ROOT/name) == wanted for name, wanted in analysis_hashes.items()), 'Analysis dependency changed during execution')
    compact = dict(status='complete', unique_physical_trajectories=10, eligible_unique_count=summary['eligible_unique_count'], development_gates=gates)
    if args.verify_only:
        print(json.dumps({**compact, 'artifacts_written': False}, indent=2)); return
    output.mkdir(parents=True, exist_ok=False)
    try:
        write_csv(output/'unique_endpoints.csv', [r for d in data for r in d['endpoints']])
        write_csv(output/'declared_endpoints.csv', declared_rows)
        write_csv(output/'coverage_curves.csv', [r for d in data for r in d['steps']])
        write_csv(output/'quality_checkpoints.csv', [r for d in data for r in d['quality']])
        write_csv(output/'actual_selections.csv', [r for d in data for r in d['selections']])
        write_csv(output/'initial_option_outcomes.csv', [d['initial_outcome'] for d in data])
        write_csv(output/'initial_action_divergence.csv', divergence)
        write_csv(output/'paired_sensor_revelation.csv', revelation)
        make_plots(output, data, manifest['declared_cases'])
        with zipfile.ZipFile(output/'sources.zip', 'x', zipfile.ZIP_DEFLATED) as archive:
            for path in source_paths: archive.write(path, str(path.relative_to(ROOT)))
        atomic_json(output/'source_sha256.json', analysis_hashes)
        atomic_json(output/'result.json', summary)
        atomic_json(output/'input_provenance.json', dict(source_root=str(source),
            manifest_sha256=sha(source/'manifest.json'), inventory_sha256=sha(source/'artifact_hashes.json'),
            aggregate_sha256=sha(source/'result.json'), frozen_sources=manifest['source_sha256'],
            reference_source_root=manifest['reference_source_root'], reference_sha256=manifest['reference_sha256'],
            unique_case_inventory_sha256={str(c['index']): sha(source/f'case_{c["index"]:02d}/artifact_hashes.json') for c in manifest['cases']},
            checks=checks, no_simulation_or_policy_execution=True))
        atomic_json(output/'artifact_hashes.json', {str(p.relative_to(output)): sha(p) for p in sorted(output.rglob('*')) if p.is_file() and p.name != 'artifact_hashes.json'})
        verify_inventory(output)
    except Exception as error:
        atomic_json(output/'analysis_failure.json', dict(status='failed', error=repr(error))); raise
    print(json.dumps({**compact, 'output': str(output)}, indent=2))


if __name__ == '__main__':
    try: main()
    except (ValueError, FileNotFoundError) as error:
        print('Analysis refused: '+str(error), file=sys.stderr); raise SystemExit(2)
