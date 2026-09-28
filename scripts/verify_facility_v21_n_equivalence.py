#!/usr/bin/env python3
"""Verify V21 N decisions on the four completed V20 saved sensor histories.

No world, evaluator, sense() or step() is instantiated/executed. The next
saved packet is loaded only after V21 has selected and authorized its action.
This is saved-observation runtime equivalence, not a new physical experiment.
The output directory must be new; failed evidence is retained without retry.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
import argparse
from collections import Counter
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter
from types import SimpleNamespace
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DEFAULT_SOURCE = ROOT / 'eval_results/facility_v20_coverage_probe_20260915'
DEFAULT_OUTPUT = ROOT / 'audit_results/facility_v21_n_equivalence_20260915'
CONFIG = ROOT / 'configs/virtual3d/facility_joint_budget_v21_probe.json'
METHODS = {'coverage_bootstrap_v20': 'coverage_bootstrap', 'coverage_bootstrap_v21': 'coverage_bootstrap',
           'observe_unique_coverage_v20': 'observe_unique_coverage', 'observe_unique_coverage_v21': 'observe_unique_coverage',
           'refresh_observed_route_v20': 'refresh_observed_route', 'refresh_observed_route_v21': 'refresh_observed_route'}
VERSION_KEYS = {'v20_coverage_intent': 'versioned_coverage_intent', 'v21_coverage_intent': 'versioned_coverage_intent',
                'v20_coverage': 'versioned_coverage', 'v21_coverage': 'versioned_coverage'}
BOOT_SHARED = ('action_id', 'observed_actions', 'scan_observations', 'prefix_actions', 'history_prefixes',
    'radar_confirmation_count', 'confirmed_hit_cells', 'confirmed_hit_current_belief_conflicts',
    'union_yield_mean', 'union_yield', 'union_success', 'union_failure', 'safe_map_permission_unchanged',
    'rate_is_confidence_bound')
OBS_EXTRA = ('known_lost_sha256', 'net_known_gain_cells', 'known_cells_before', 'known_cells_after',
             'navigation_counts_are_diagnostic_only', 'coverage_budget_unit')
PREFIX_DIAGNOSTICS = ('credited_safe_gain', 'net_known_gain', 'credited_known_gain', 'known_lost')
NORMALIZATION = {
    'module_envelope': 'Verify raw input/output digests, then omit elapsed_s and those derived digests; retain all other envelope fields.',
    'version_names': {'methods': METHODS, 'keys': VERSION_KEYS,
        'refresh_audit_events': ['v20_route_update', 'v21_route_update'],
        'selection_planner_revision': ['v20_common_coverage_budget', 'v21_fixed_roi_known_budget']},
    'bootstrap': {'input_extra_v21': {'coverage_proxy': 'fixed_task_roi_known_fraction'},
        'budget_snapshot_diagnostics_allowed_to_differ': True, 'shared_snapshot_fields_compared': BOOT_SHARED},
    'selection_outputs_allowed_to_differ': ['coverage_state', 'quality_budget_admitted_candidates', 'scores.G',
                                          'score_audit[*].versioned_coverage.coverage_budget'],
    'refresh_outputs_allowed_to_differ': ['coverage_budget'],
    'observation_ledger_extra_diagnostics_v21': OBS_EXTRA,
    'closed_prefix_budget_diagnostics_allowed_to_differ': PREFIX_DIAGNOSTICS,
    'never_omitted': ['all candidate geometry/paths/costs', 'scores.N', 'selected target/option ID/heading',
        'guard inputs and outputs including belief hashes and return reserve', 'actual feedback',
        'map versions', 'coverage prediction masks/counts', 'refresh retention/cancellation/lengths',
        'runtime authorization/observation records', 'termination', 'final mesh arrays'],
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    path = Path(path)
    if path.suffix == '.gz':
        with gzip.open(path, 'rt', encoding='utf-8') as stream:
            return json.load(stream)
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def dump(path, value):
    data = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n'
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('x', encoding='utf-8') as stream:
        stream.write(data); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def seal(folder):
    dump(folder / 'artifact_hashes.json', {str(p.relative_to(folder)): sha(p)
        for p in sorted(folder.rglob('*')) if p.is_file() and p.name != 'artifact_hashes.json'})


def verify_inventory(folder):
    inventory = read(folder / 'artifact_hashes.json')
    actual = {str(p.relative_to(folder)) for p in folder.rglob('*') if p.is_file() and p.name != 'artifact_hashes.json'}
    require(set(inventory) == actual, 'Artifact inventory differs: ' + str(folder))
    for name, expected in inventory.items():
        path = (folder / name).resolve()
        require(path.is_relative_to(folder.resolve()) and sha(path) == expected, 'Artifact changed: ' + str(path))
    return inventory


def renamed(value):
    """Only the two explicit metadata keys change; route group names remain exact."""
    if isinstance(value, list):
        return [renamed(v) for v in value]
    if isinstance(value, dict):
        result = {}
        for k, v in value.items():
            key = VERSION_KEYS.get(k, k)
            require(key not in result, 'Version normalization key collision')
            result[key] = renamed(v)
        return result
    return value


def normalized_call(call):
    require(digest(call['inputs']) == call['input_sha256'] and digest(call['outputs']) == call['output_sha256'],
            'Raw module input/output digest differs')
    row = renamed({k: deepcopy(v) for k, v in call.items() if k not in ('elapsed_s', 'input_sha256', 'output_sha256')})
    method = METHODS.get(row['method'], row['method']); row['method'] = method
    output = row['outputs']
    if method == 'coverage_bootstrap':
        if 'coverage_proxy' in row['inputs']:
            require(row['inputs'].pop('coverage_proxy') == 'fixed_task_roi_known_fraction', 'Unexpected V21 bootstrap proxy')
        row['outputs'] = {k: output[k] for k in BOOT_SHARED}
    elif method == 'observe_unique_coverage':
        for key in OBS_EXTRA:
            output.pop(key, None)
        if 'closed_prefix' in output:
            for key in PREFIX_DIAGNOSTICS:
                output['closed_prefix'].pop(key, None)
    elif method == 'select_topo_target':
        require(output['mode'] == output['effective_objective'] == 'N', 'N equivalence changed objective')
        require(output.pop('planner_revision') in ('v20_common_coverage_budget', 'v21_fixed_roi_known_budget'), 'Unexpected planner revision')
        output.pop('coverage_state'); output.pop('quality_budget_admitted_candidates'); output['scores'].pop('G')
        for score in output['score_audit']:
            score['versioned_coverage'].pop('coverage_budget')
    elif method == 'refresh_observed_route':
        output.pop('coverage_budget')
    return row


def normalized_audit(row):
    row = renamed(deepcopy(row))
    if row['event'] in ('v20_route_update', 'v21_route_update'):
        row['event'] = 'versioned_route_update'; row.pop('coverage_budget')
    return row


def first_difference(a, b, path='$'):
    if type(a) is not type(b):
        return dict(path=path, expected_type=type(a).__name__, actual_type=type(b).__name__)
    if isinstance(a, dict):
        if set(a) != set(b):
            return dict(path=path, missing=sorted(set(a) - set(b)), extra=sorted(set(b) - set(a)))
        for key in a:
            difference = first_difference(a[key], b[key], path + '.' + key)
            if difference is not None:
                return difference
    elif isinstance(a, list):
        if len(a) != len(b):
            return dict(path=path, expected_length=len(a), actual_length=len(b))
        for i, (av, bv) in enumerate(zip(a, b)):
            difference = first_difference(av, bv, path + f'[{i}]')
            if difference is not None:
                return difference
    elif a != b:
        return dict(path=path, expected=a, actual=b)
    return None


def same(a, b, where):
    difference = first_difference(a, b)
    if difference is not None:
        raise ValueError(json.dumps(dict(where=where, difference=difference,
            expected_sha256=digest(a), actual_sha256=digest(b)), sort_keys=True))


class CallComparator:
    def __init__(self, calls, audit):
        self.calls = calls; self.audit = audit; self.call_index = 0; self.audit_index = 0
        self.normalized_calls = []; self.normalized_audits = []
        self.decisions = []; self.refreshes = []; self.guard_beliefs = []; self.methods = Counter()

    def compare_new(self, backend, runtime, stage):
        while self.call_index < len(backend.calls):
            index = self.call_index
            require(index < len(self.calls), f'Extra V21 module call at {stage}')
            expected = normalized_call(self.calls[index]); actual = normalized_call(backend.calls[index])
            same(expected, actual, f'{stage}: module call {index + 1}')
            self.normalized_calls.append(actual); self.call_index += 1; self.methods[actual['method']] += 1
            output = actual['outputs']
            if actual['method'] == 'select_topo_target':
                self.decisions.append(dict(action_id=actual['action_id'], call_id=actual['call_id'],
                    candidate_count=len(output['candidates']), candidate_geometry_paths_costs_sha256=digest(output['candidates']),
                    candidate_generation_audit_sha256=digest(output['candidate_audit']),
                    N_scores_sha256=digest(output['scores']['N']), score_audit_except_budget_sha256=digest(output['score_audit']),
                    selected_target_sha256=digest(output['selected']),
                    selected_pose=None if output['selected'] is None else output['selected']['pose'],
                    selected_option_id=None if output['selected'] is None else output['selected']['option_id'],
                    all_fields_exact_after_declared_budget_exclusions=True))
            elif actual['method'] == 'refresh_observed_route':
                self.refreshes.append(dict(output, comparison_sha256=digest(output)))
            elif actual['method'] == 'assess_local_action':
                belief = digest(backend.scenes[0]['mapper'].belief.tolist())
                require(actual['inputs']['belief_sha256'] == belief, 'Current map differs from guard input hash')
                self.guard_beliefs.append(dict(action_id=actual['action_id'], call_id=actual['call_id'],
                    belief_sha256=belief, allowed=output['allowed'], authorized_action=output['action'],
                    guard_input_sha256=digest(actual['inputs']), guard_output_sha256=digest(output)))
        while self.audit_index < len(runtime.audit):
            index = self.audit_index
            require(index < len(self.audit), f'Extra V21 runtime audit row at {stage}')
            expected = normalized_audit(self.audit[index]); actual = normalized_audit(runtime.audit[index])
            same(expected, actual, f'{stage}: runtime audit {index}')
            self.normalized_audits.append(actual); self.audit_index += 1

    def finish(self):
        require(self.call_index == len(self.calls) and self.audit_index == len(self.audit), 'Missing final module/audit records')


def public_configuration(reference):
    import numpy as np
    # NPZ arrays are lazy. Index ONLY metadata, never points, faces, instances,
    # reachable masks, visibility, evaluator references or other hidden arrays.
    with np.load(reference, allow_pickle=False) as cache:
        metadata = json.loads(str(cache['metadata'].item()))
    config = metadata['signature_payload']['config']
    return config, SimpleNamespace(**config)


def initialize_runtime(first, config, shape, case, protocol):
    from nso.cpu_sensor_contract_v10 import GridTransform
    from nso.facility_runtime_v21 import FacilityRuntimeV21, facility_components_v21
    args = SimpleNamespace(nso_backend='cpu_v10', eval=True, train_global=False,
        use_open_vocab_semantic=True, use_topo_graph=True, use_rpn_uq=True, use_igcr=True,
        cpu_score_mode='N', cpu_disable_feedback=False,
        cpu_max_candidates=5, cpu_coverage_slots=4, cpu_planner_revision='v10_3_1',
        cpu_semantic_source_schema='inspection_v4', cpu_measured_novelty_floor=.25,
        cpu_task_asset_count=protocol['task_asset_count'],
        cpu_v20_replan_interval=protocol['replan_interval_actions'],
        cpu_v20_coverage_slots=protocol['coverage_candidate_slots'])
    components = facility_components_v21(args, shape)
    runtime = FacilityRuntimeV21(components, 1, shape)
    runtime.start_sensor_episode(0, config=config,
        transform=GridTransform(shape, config.resolution_m), packets=[first],
        total_budget=case['budget'], return_anchor=(*first.position, first.heading))
    return runtime


def verify_case(source, manifest, case, protocol, progress):
    import numpy as np
    from nso.decision_replay_v13 import load_packet, array_hash
    folder = source / f'case_{case["index"]:02d}'
    inventory = verify_inventory(folder); result = read(folder / 'result.json')
    calls = read(folder / 'module_calls.json.gz'); audit = read(folder / 'runtime_audit.json.gz')
    same(result['case'], case, 'Prepared historical case identity')
    require(digest([{k: v for k, v in row.items() if k != 'elapsed_s'} for row in calls]) == result['module_calls_sha256'], 'Historical module digest differs')
    require(digest(audit) == result['runtime_audit_sha256'], 'Historical runtime digest differs')
    physical = read(folder / 'timing.json')['process_id']; original_replay = read(folder / 'verification.json')
    require(os.getpid() not in (physical, original_replay['replay_process_id']), 'Verification process must be independent of original execution/replay')
    name = f'references/{case["parent"]}_{case["assignment"]}_2026.npz'
    reference = Path(manifest['reference_source_root']) / name
    require(sha(reference) == manifest['reference_sha256'][name], 'Reference metadata source hash differs')
    public, config = public_configuration(reference)
    same({key: getattr(config, key) for key in protocol['physical_contract']}, protocol['physical_contract'], 'Public physical contract')
    shape = (round(config.height_m / config.resolution_m), round(config.width_m / config.resolution_m))
    first = load_packet(folder / 'packets/0000.npz')
    require(first.action_id == 0 and first.action is None and first.sha256() == case['initial_packet_sha256'], 'Saved initial packet differs')
    runtime = initialize_runtime(first, config, shape, case, protocol)
    backend = runtime.components._cpu_backend; state = runtime.states[0]; mapper = state['mapper']
    comparison = CallComparator(calls, audit)
    comparison.compare_new(backend, runtime, 'frame 0 bootstrap')
    actual_actions = []; steps = []; packet_receipts = [{'action_id': 0, 'packet_sha256': first.sha256()}]
    progress.update(case_index=case['index'], matched_paid_actions=0, matched_module_calls=comparison.call_index,
                    matched_audit_rows=comparison.audit_index)
    for action_id, expected_action in enumerate(result['actions'], 1):
        # No next packet is loaded or supplied before this decision is made.
        before_belief = mapper.belief.copy(); before_hash = digest(before_belief.tolist())
        action = runtime.next_local_action(0)
        comparison.compare_new(backend, runtime, f'action {action_id} decision/authorization')
        require(action == expected_action['action'], f'Chosen action differs at {action_id}')
        pending = state['pending']; require(pending is not None and pending['allowed'], 'Chosen action lacks authorization')
        same(pending['next_pose'], expected_action['pose'], f'Authorized pose at action {action_id}')
        require(pending['remaining_budget'] == case['budget'] - action_id + 1 and
                pending['reserved_return_cost'] is not None and pending['reserved_return_cost'] <= case['budget'] - action_id,
                'Return reserve or remaining paid budget differs')
        pending_receipt = deepcopy(pending)
        packet = load_packet(folder / f'packets/{action_id:04d}.npz')
        require(packet.action_id == action_id and packet.sha256() == expected_action['packet_sha256'] and packet.action == action,
                f'Raw packet identity/action differs at {action_id}')
        actual = dict(action=action, pose=[*packet.position, packet.heading], collision=packet.collision, packet_sha256=packet.sha256())
        same(expected_action, actual, f'Raw action/pose/collision receipt {action_id}')
        runtime.observe(0, action_id, None, None, None, sensor_packet=packet)
        comparison.compare_new(backend, runtime, f'action {action_id} observed')
        # Extra V21 budget-ledger diagnostics are checked against this real map,
        # though their definitions intentionally differ from the V20 budget.
        ledger = backend.scenes[0]['coverage_v21']; event = ledger.events[-1]
        roi = ledger.task_mask; known_before = (before_belief != -1) & roi; known_after = (mapper.belief != -1) & roi
        lost = known_before & ~known_after
        require(event['known_cells_before'] == int(known_before.sum()) and event['known_cells_after'] == int(known_after.sum()), 'V21 known counts differ from observed belief')
        require(event['known_lost_sha256'] == hashlib.sha256(np.ascontiguousarray(lost).tobytes()).hexdigest() and event['net_known_gain_cells'] == int(known_after.sum()) - int(known_before.sum()), 'V21 net known/loss diagnostics differ')
        actual_actions.append(actual); packet_receipts.append(dict(action_id=action_id, packet_sha256=packet.sha256()))
        steps.append(dict(action_id=action_id, pose=actual['pose'], action=action,
            before_belief_sha256=before_hash, after_belief_sha256=digest(mapper.belief.tolist()),
            authorization_sha256=digest(pending_receipt), reserved_return_cost=pending_receipt['reserved_return_cost']))
        progress.update(matched_paid_actions=action_id, matched_module_calls=comparison.call_index,
                        matched_audit_rows=comparison.audit_index)
        if action_id % 50 == 0:
            print(f'V21 N saved-history case={case["index"]} matched_actions={action_id}', flush=True)
    require(runtime.next_local_action(0) is None, 'V21 requests an extra action after historical endpoint')
    comparison.compare_new(backend, runtime, 'terminal decision'); comparison.finish()
    require(state['closed'], 'V21 failed to terminate at the historical endpoint')
    same(state['termination'], result['termination'], 'Termination including safety and return heading')
    same(actual_actions, result['actions'], 'Entire chosen and consumed action trajectory')
    mesh = mapper.mesh()
    mesh_hashes = {key: array_hash(np.asarray(getattr(mesh, key))) for key in ('vertices', 'triangles', 'vertex_colors')}
    same(result['final_mesh_sha256'], mesh_hashes, 'Final TSDF mesh arrays')
    require(len(actual_actions) == result['paid_actions'], 'Saved action count differs')
    auth_count = sum(row['event'] == 'paid_action_authorized' for row in runtime.audit)
    require(auth_count == len(actual_actions), 'One authorization per consumed paid observation required')
    provenance = dict(raw_source_root=str(folder), source_case_inventory_sha256=sha(folder / 'artifact_hashes.json'),
        source_case_artifact_sha256=inventory, reference_metadata_file=str(reference), reference_file_sha256=sha(reference),
        reference_member_read='metadata only', reference_config_sha256=digest(public), public_config=public,
        hidden_reference_arrays_loaded=False, original_physical_process_id=physical,
        original_replay=original_replay, saved_history_verifier_process_id=os.getpid())
    return dict(status='passed', case=case, mode='N', paid_actions=len(actual_actions), raw_packets_consumed=len(packet_receipts),
        module_calls_compared=comparison.call_index, runtime_audit_rows_compared=comparison.audit_index,
        module_method_counts=dict(comparison.methods), decisions_compared=len(comparison.decisions),
        candidate_rows_compared=sum(x['candidate_count'] for x in comparison.decisions),
        route_refreshes_compared=len(comparison.refreshes), guard_assessments_compared=len(comparison.guard_beliefs),
        paid_authorizations_compared=auth_count, termination=state['termination'],
        trajectory_sha256=digest(actual_actions), original_trajectory_sha256=digest(result['actions']),
        raw_packet_sequence_sha256=digest(packet_receipts), pre_and_post_belief_receipts_sha256=digest(steps),
        normalized_module_calls_sha256=digest(comparison.normalized_calls),
        normalized_runtime_audit_sha256=digest(comparison.normalized_audits), final_mesh_sha256=mesh_hashes,
        decisions=comparison.decisions, route_refreshes=comparison.refreshes, guard_belief_receipts=comparison.guard_beliefs,
        action_state_receipts=steps, provenance=provenance,
        runtime_decisions_on_saved_history_exact=True, fresh_sensor_simulation=False,
        evaluator_executed=False, future_packet_supplied_to_planner=False, independent_process=True,
        semantic_information_test=False, full_architecture_efficacy_proven=False)


def preflight(source):
    m = read(source / 'manifest.json'); aggregate = read(source / 'result.json'); protocol = read(CONFIG)
    require(m['status'] == aggregate['status'] == 'complete' and m['mode'] == 'N', 'Completed historical N batch required')
    expected = [(p, a) for p in ['D19-P00', 'D19-P01'] for a in ['A_complex_B_simple', 'A_simple_B_complex']]
    require([(c['parent'], c['assignment']) for c in m['cases']] == expected and [c['index'] for c in m['cases']] == list(range(4)), 'Four exact paired cases required')
    require(aggregate['total_physical_actions'] == 688 and aggregate['independent_process_replays'] == 4, 'Expected completed 688-action/four-replay baseline')
    require(protocol['replan_interval_actions'] == 5 and protocol['coverage_candidate_slots'] == 8 and protocol['task_asset_count'] == 6, 'V21 runtime protocol changed')
    require(protocol['planning_coverage_proxy']['kind'] == 'known_cells_over_fixed_public_roi' and protocol['planning_coverage_proxy']['target_fraction'] == .8, 'Unexpected V21 ledger protocol')
    verify_inventory(source)
    for c in m['cases']:
        require(c['mode'] == 'N' and c['budget'] == {'D19-P00': 160, 'D19-P01': 184}[c['parent']], 'Baseline mode/budget differs')
        v = read(source / f'case_{c["index"]:02d}/verification.json')
        require(v['status'] == 'passed' and v['independent_process'] and v['fresh_world_sensor_action_metric_replay']
                and v['physical_process_id'] != v['replay_process_id'], 'Original fresh-process replay receipt invalid')
    for name, wanted in m['source_sha256'].items():
        require(sha(ROOT / name) == wanted, 'Original V20 frozen source changed: ' + name)
    with zipfile.ZipFile(source / 'sources.zip') as archive:
        require(set(archive.namelist()) == set(m['source_sha256']), 'Old source archive inventory differs')
        for name, wanted in m['source_sha256'].items():
            require(hashlib.sha256(archive.read(name)).hexdigest() == wanted, 'Old frozen archive bytes differ')
    historical = Path(m['reference_source_root'])
    require(sha(historical / 'manifest.json') == m['reference_source_manifest_sha256'] and
            sha(historical / 'artifact_hashes.json') == m['reference_source_inventory_sha256'], 'Reference provenance changed')
    return m, protocol


def current_sources():
    paths = [p for d in ('env', 'nso', 'utils') for p in sorted((ROOT / d).glob('*.py'))]
    paths += [Path(__file__).resolve(), CONFIG, ROOT / 'scripts/probe_facility_v21.py']
    for name in ('nso/facility_runtime_v21.py', 'nso/coverage_budget_v21.py'):
        require((ROOT / name).is_file(), 'V21 runtime/ledger must be ready before verification')
    return {str(p.relative_to(ROOT)): sha(p) for p in paths}


def check_sources(sources):
    for name, wanted in sources.items():
        require(sha(ROOT / name) == wanted, 'Verification source changed during run: ' + name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE, help='Completed frozen V20 N4 history')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT, help='New verification directory; never resume or overwrite')
    args = parser.parse_args(); source = args.source.resolve(); output = args.output.resolve()
    require(not output.is_relative_to(source) and not source.is_relative_to(output), 'Output and source trees must be separate')
    require(not output.exists(), 'Verification output already exists; failure evidence must be preserved')
    m, protocol = preflight(source); frozen = current_sources()
    output.mkdir(parents=True, exist_ok=False)
    run = dict(status='running', kind='V21_N_saved_history_runtime_equivalence', source_root=str(source),
        source_manifest_sha256=sha(source / 'manifest.json'), source_inventory_sha256=sha(source / 'artifact_hashes.json'),
        source_sha256=frozen, configuration=protocol, normalization_contract=NORMALIZATION,
        fresh_sensor_simulation=False, original_physical_trajectories=4, new_physical_trajectories=0,
        semantic_information_test=False, full_architecture_efficacy_proven=False, process_id=os.getpid())
    dump(output / 'manifest.json', run)
    with zipfile.ZipFile(output / 'sources.zip', 'x', zipfile.ZIP_DEFLATED) as archive:
        for name in frozen:
            archive.write(ROOT / name, name)
    dump(output / 'source_sha256.json', frozen)
    completed = []; progress = {}; started = perf_counter()
    try:
        for case in m['cases']:
            check_sources(frozen)
            progress = dict(case_index=case['index'], matched_paid_actions=0)
            receipt = verify_case(source, m, case, protocol, progress)
            dump(output / f'case_{case["index"]:02d}.json', receipt); completed.append(receipt)
            print(f'case={case["index"]} passed: actions={receipt["paid_actions"]}, '
                  f'decisions={receipt["decisions_compared"]}, refreshes={receipt["route_refreshes_compared"]}', flush=True)
        counts = dict(saved_paid_actions_compared=sum(c['paid_actions'] for c in completed),
            decisions_compared=sum(c['decisions_compared'] for c in completed),
            route_refreshes_compared=sum(c['route_refreshes_compared'] for c in completed),
            paid_authorizations_compared=sum(c['paid_authorizations_compared'] for c in completed))
        require(counts == dict(saved_paid_actions_compared=688, decisions_compared=34,
            route_refreshes_compared=105, paid_authorizations_compared=688), 'Declared N4 comparison counts differ')
        check_sources(frozen); verify_inventory(source)
        require(sha(source / 'manifest.json') == run['source_manifest_sha256'] and
                sha(source / 'artifact_hashes.json') == run['source_inventory_sha256'], 'Historical provenance changed during verification')
        summary = dict(status='passed', **counts, original_physical_trajectories=4, new_physical_trajectories=0,
            saved_history_reexecutions=4, all_four_final_meshes_exact=True, all_four_terminations_exact=True,
            baseline_reuse_supported_for_this_fixed_N4_protocol=True,
            normalization_contract=NORMALIZATION, elapsed_seconds=perf_counter() - started,
            independent_process=True, fresh_sensor_simulation=False, evaluator_executed=False,
            future_packet_supplied_to_planner=False, semantic_information_test=False,
            full_architecture_efficacy_proven=False,
            scope='Exact V21 N decisions and mapping on four saved V20 N histories after only explicit budget diagnostics/version normalization. No new sensing, no G comparison or semantic gain claim.',
            cases=[{k: c[k] for k in ('case', 'paid_actions', 'decisions_compared', 'candidate_rows_compared',
                'route_refreshes_compared', 'guard_assessments_compared', 'paid_authorizations_compared',
                'trajectory_sha256', 'final_mesh_sha256', 'termination')} for c in completed])
        dump(output / 'result.json', summary); run['status'] = 'complete'
    except Exception as error:
        run.update(status='failed', error=repr(error))
        dump(output / 'failure.json', dict(status='failed', error=repr(error), progress=progress,
            completed_case_indices=[c['case']['index'] for c in completed],
            elapsed_seconds=perf_counter() - started, failed_evidence_preserved=True))
        raise
    finally:
        dump(output / 'manifest.json', run); seal(output)
    print(json.dumps(dict(status='passed', **counts, output=str(output)), indent=2), flush=True)


if __name__ == '__main__':
    main()
