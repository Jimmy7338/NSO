#!/usr/bin/env python3
"""Independent measured-map and charged-action evaluation for the 192-step study."""

import argparse
import csv
import hashlib
import importlib.util
from itertools import combinations
import json
from math import isfinite
import os
from pathlib import Path

for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '1'

ROOT = Path(__file__).resolve().parents[1]
SCENES = ROOT / 'scenes'
SOURCE_EVAL = ROOT.parent / 'same_goal_route_choice_20261121/measurement/evaluate.py'
PAID_CAP = 192
CLOCKS = (0, 48, 96, 144, 192)
SENSORS = ('depth_values_sha256', 'rgb_values_sha256', 'scan_ranges_sha256')
TRAJECTORY = ('step', 'action', 'heading', 'position', 'path_length_m', 'collision', 'packet_sha256',
              'local_route_id', 'active_option_id', *SENSORS)


def load_native():
    spec = importlib.util.spec_from_file_location('weighted_frozen_map_evaluator', SOURCE_EVAL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


native = load_native()
engine, files = native.engine, native.files


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_freeze(split):
    actor_path = SCENES / f'actor_freeze_{split}.json'
    eval_path = SCENES / f'evaluator_freeze_{split}.json'
    actor, evaluator = read(actor_path), read(eval_path)
    require(actor['layouts'] == evaluator['layouts'] and actor['actors_run'] == evaluator['actors_run'] == 0
            and actor['budget'] == evaluator['budget'] == PAID_CAP and actor['split'] == evaluator['split'] == split,
            'Cold starts and independent reference layouts differ')
    require(not any('evaluator_references' in path for path in actor['files_sha256']),
            'Reference data appeared in actor-facing manifest')
    for name, digest in (actor['files_sha256'] | evaluator['files_sha256']).items():
        files.verify(name, digest)
    return actor, evaluator, {'actor_freeze_sha256': sha(actor_path), 'evaluator_freeze_sha256': sha(eval_path),
                              'native_evaluator_sha256': sha(SOURCE_EVAL), 'measurement_sha256': sha(Path(__file__))}


def load_reference(layout, initial_receipt):
    base = SCENES / 'evaluator_references' / layout
    receipt = read(base / 'RECEIPT.json')
    initial_path = SCENES / 'prepared' / layout / 'initial_state.json'
    initial = read(initial_path)
    require(initial['initial_packet_sha256'] == initial_receipt['initial_packet_sha256']
            and initial['anchor'] == initial_receipt['anchor'], f'Unsealed actor start for {layout}')
    with files.np.load(base / 'evaluator_truth.npz', allow_pickle=False) as truth:
        reachable = truth['reachable']
    require(list(reachable.shape) == initial['shape'] and int(reachable.sum()) == receipt['reachable_cells'],
            f'Correct-free denominator drift for {layout}')
    samples, weights = {}, {}
    with files.np.load(base / 'fixed_reference.npz', allow_pickle=False) as fixed:
        for group, key in {'scene': 'all', **files.GROUPS}.items():
            samples[group] = fixed[key]
            counts = receipt['metadata'][key]
            weights[group] = counts['area_per_sample_m2']
            require(samples[group].shape == (counts['visible_samples'], 3), f'Reference sampling drift for {layout}')
    return {'initial': initial, 'reachable': reachable, 'source_mesh': files.read_mesh(base / 'source_mesh.ply'),
            'samples': samples, 'weights': weights, 'source_reference_receipt_sha256': sha(base / 'RECEIPT.json')}


def verify_pool(directory, manifest, initial, trace):
    global_events = read(directory / 'global_choices.json')
    raw_local = read(directory / 'local_choices.json')
    by_call_id = {event['call_id']: event for event in global_events}
    require(len(by_call_id) == len(global_events) and all(e['outputs']['mode'] == 'G' for e in global_events),
            f'Non-neutral or duplicate global choice in {directory}')
    if 'local_route_choices' in manifest:
        require(len(raw_local) == manifest['local_route_choices'], f'Lost local event in {directory}')
    counts = {'local_decisions': len(raw_local), 'global_decisions': len(global_events),
              'full_pool_assessed': 0, 'at_least_two_legal_route_decisions': 0}
    for event in raw_local:
        decision = event['choice_audit']
        candidates = decision['candidates']
        proof = event['full_pool']
        admitted = [entry for entry in candidates if entry['admitted']]
        by_route = {entry['route_id']: entry for entry in candidates}
        original = by_route[decision['original_route_id']]
        selected = by_route[decision['selected_route_id']]
        goal = decision['native_goal_pose']
        paid = event['paid_before']
        observed_packet = trace[paid - 1]['packet_sha256'] if paid else initial['initial_packet_sha256']
        global_plan = by_call_id[event['selection_call_id']]['outputs']['selected']
        require(len(by_route) == len(candidates) == decision['assessed_candidate_count']
                and len({tuple(entry['outbound_actions']) for entry in candidates}) == len(candidates)
                and len(admitted) == decision['admitted_candidate_count']
                and decision['proposal_diagnostics']['distinct_alternate_traces'] == len(candidates) - 1
                and proof['enabled'] and proof['unassessed_after_cap'] == 0
                and proof['exhaustive_cap_upper_bound'] == 4 * initial['shape'][0] * initial['shape'][1] + 1
                and decision['candidate_pool_sha256'] == proof['full_candidate_pool_sha256'] ==
                    native.canonical_checksum([{k: v for k, v in e.items() if k != 'scores'} for e in candidates])
                and decision['score_table_sha256'] == proof['full_score_sha256'] ==
                    native.canonical_checksum([e['scores'] for e in candidates])
                and selected['admitted'] and event['packet_sha256'] == observed_packet
                and global_plan['pose'] == goal == event['same_global_target']
                and global_plan['return_actions'] == event['native_return_actions'] == event['rebound_return_actions'],
                f'Compared route pool changed native goal, proposal or return guard: {directory}, paid {paid}')
        for entry in admitted:
            native.verify_candidate_nominal(entry, original, goal, 8, .95)
            require(paid + entry['outbound_cost'] + entry['reserved_return_cost'] <= PAID_CAP,
                    f'Admitted candidate exceeds the action budget in {directory}, paid {paid}')
        verify_ranking(decision, manifest, admitted, original, selected, initial, directory)
        counts['full_pool_assessed'] += len(candidates)
        counts['at_least_two_legal_route_decisions'] += len(admitted) >= 2
    return counts


def verify_ranking(decision, manifest, admitted, original, selected, initial, directory):
    logger = decision['weighted_ranking']
    prior, unknown_weight, action_cost = (manifest[k] for k in ('study_prior', 'unknown_weight', 'action_cost'))
    require((logger['prior'], logger['unknown_weight'], logger['action_cost']) == (prior, unknown_weight, action_cost),
            f'Recorded score uses another prior/weight: {directory}')
    require(logger['choice_prior_uses_category'] == (prior != 'G') and logger['score_equality_tolerance'] == 1e-9,
            f'Unexpected category access or numerical tie margin: {directory}')
    if prior == 'G':
        require(all(set(row['scores']) == {'G'} for row in admitted)
                and logger['local_area_provider_calls']['S'] == logger['local_area_provider_calls']['X'] == 0,
                f'Neutral local eligibility/ranking consumed a category scores table: {directory}')
    if prior == 'X':
        require(all(entry['X'] == entry['S'][::-1] for entry in logger['observed_instance_shape_prior_exchange']),
                f'Structural prior vectors were not reversed for X: {directory}')
    ranked = []
    for candidate in admitted:
        neutral_area = float(candidate['scores'][prior])
        union_diff = (int(candidate['unknown_union_cells']) - int(original['unknown_union_cells'])) * initial['grid_resolution_m'] ** 2
        extra = int(candidate['outbound_cost']) - int(original['outbound_cost'])
        ranked.append({'route_id': candidate['route_id'], 'area_m2': neutral_area, 'delta_unknown_union_m2': union_diff,
                       'extra_outbound_actions': extra, 'score_m2': neutral_area + unknown_weight * union_diff - action_cost * extra})
    ranked_by_id = {entry['route_id']: entry['score_m2'] for entry in ranked}
    maximum = max(ranked_by_id.values())
    ties = [row for row in admitted if maximum - ranked_by_id[row['route_id']] <= logger['score_equality_tolerance']]
    winner = min(ties, key=lambda row: (row['outbound_cost'], row['route_id'] != original['route_id'], row['outbound_sha256']))
    require(ranked == logger['scores'] and native.canonical_checksum(ranked) == logger['ranking_scores_sha256']
            and winner == selected and decision['chosen_outbound_sha256'] == winner['outbound_sha256'],
            f'Execution selected a route inconsistent with the frozen weighted rule: {directory}')


def source_directory(root, split, layout, arm):
    requested = root / split / layout / arm
    path = requested / 'alias.json'
    if not path.is_file():
        return requested, arm, None, None
    alias = read(path)
    require((alias.get('schema') or alias.get('status')) == 'identical_frozen_configuration_alias_v1'
            and (alias['split'], alias['layout'], alias['arm']) == (split, layout, arm)
            and all(key in alias for key in ('prior', 'unknown_weight', 'action_cost', 'run_freeze_sha256')),
            f'Unregistered alias: {path}')
    source_arm = alias['source_arm']
    require(Path(source_arm).name == source_arm and source_arm not in (arm, '.', '..'), f'Invalid alias source: {path}')
    source = root / split / layout / source_arm
    require(not (source / 'alias.json').exists() and not (requested / 'manifest.json').exists(),
            f'Ambiguous alias or alias chain: {path}')
    for name, key in (('manifest.json', 'source_manifest_sha256'), ('failure.json', 'source_failure_sha256')):
        target = source / name
        require(alias.get(key) == (sha(target) if target.is_file() else None), f'Alias source bytes changed: {path}')
    return source, source_arm, alias, sha(path)


def evaluate_run(split, layout, arm, ref, metric, only_terminal, root, job, freeze_sha, protocol_sha):
    directory, source_arm, alias, alias_sha = source_directory(root, split, layout, arm)
    native_manifest = directory / 'manifest.json'
    row = {'split': split, 'layout': layout, 'arm': arm, 'native_budget': PAID_CAP, 'run_directory': str(directory)}
    if not job['literal'] and alias is None and not directory.exists():
        return row | {'status': 'pending', 'stop_reason': None}, [], None, []
    row |= {'physical_source_arm': source_arm, 'exact_configuration_alias': alias is not None, 'alias_sha256': alias_sha}
    require(source_arm == job['source_arm'] and job['literal'] == (alias is None),
            f'Materialized arm does not match frozen closure: {directory}')
    if not native_manifest.is_file():
        fail = directory / 'failure.json'
        status = 'failed' if fail.is_file() else 'pending'
        heartbeat = directory / 'heartbeat.json'
        return row | {'status': status, 'failure_sha256': sha(fail) if fail.is_file() else None,
                      'last_checkpoint_without_terminal': read(heartbeat).get('actual_paid') if heartbeat.is_file() else None,
                      'stop_reason': read(fail).get('error') if fail.is_file() else None}, [], None, []

    manifest, manifest_sha = engine.verify_manifest(directory, 'manifest.json')
    initial = ref['initial']
    require(manifest['status'] == 'completed_execution' and manifest['stage'] == 'holdout'
            and manifest.get('study_split', split) == split and manifest['layout'] == layout and manifest['arm'] == source_arm
            and manifest['native_budget'] == PAID_CAP and manifest['global_mode'] == 'G'
            and manifest['actual_closed_loop'] and not manifest['forced_route_or_goal']
            and not manifest['evaluator_inputs_read_by_actor'], f'Run receipt changed study or runtime contract: {directory}')
    attempt = read(directory / 'attempt.json')
    require((manifest['study_prior'], manifest['unknown_weight'], manifest['action_cost']) ==
                (job['prior'], job['unknown_weight'], job['action_cost'])
            and manifest['runner_freeze_sha256'] == attempt['runner_freeze_sha256'] == freeze_sha
            and manifest['source_protocol_sha256'] == attempt['source_protocol_sha256'] == protocol_sha
            and manifest['actual_scene_seed'] == initial['seed']
            and manifest['frozen_cold_input_sha256'] == sha(SCENES / 'prepared' / layout / 'initial_state.json'),
            f'Cold episode used a different sealed policy, source, or initialization: {directory}')
    if alias is not None:
        require(attempt['runner_freeze_sha256'] == alias['run_freeze_sha256']
                and (manifest['study_prior'], manifest['unknown_weight'], manifest['action_cost']) ==
                    (alias['prior'], alias['unknown_weight'], alias['action_cost']),
                f'Identity alias has a different runtime or policy weight: {directory}')
    require(attempt['initial_owner_sha256'] == initial['initial_owner_sha256']
            and attempt['initial_pose'] == initial['anchor'], f'Actual reset was not the frozen first frame: {directory}')
    initial_raw = read(directory / 'initial_raw_rgbd_scan_sha256.json')
    require(initial_raw['step'] == 0 and initial_raw['packet_sha256'] == initial['initial_packet_sha256']
            and all(initial_raw.get(field) for field in SENSORS), f'Raw start fingerprint missing for {directory}')
    trace = read(directory / 'steps.json')
    chain = native.verify_trace(trace, initial, PAID_CAP, manifest)
    native.verify_action_guard(trace, PAID_CAP)
    require(all(all(step.get(k) for k in SENSORS) for step in trace), f'Sensor fingerprints absent from {directory}')
    counts = verify_pool(directory, manifest, initial, trace)
    expected = ['start', *(f'paid_{clock}' for clock in CLOCKS if clock), 'terminal']
    require(set(manifest['checkpoints']) == set(expected), f'Registered checkpoint set changed in {directory}')
    clocks = ('start', 'terminal') if only_terminal else expected
    checked = []
    for clock in clocks:
        receipt, masks = native.evaluate_snapshot(directory, clock, manifest, initial, trace, ref, metric)
        if clock == 'start':
            baseline, baseline_masks = receipt, masks
        if clock == 'terminal':
            require(receipt['packet_chain_sha256'] == chain, f'Terminal mesh is not on the actual trace in {directory}')
            terminal = native.close_record(receipt, read(directory / 'runtime_summary.json'), trace, initial)
            require(terminal['returned'] == manifest['returned'] and terminal['terminal_pose'] == manifest['final_pose']
                    and terminal['termination'] == manifest['observed_native_termination'],
                    f'Terminal reason/return disagrees with manifest in {directory}')
            final_receipt = receipt
        elif clock.startswith('paid_'):
            paid = int(clock.removeprefix('paid_'))
            require(receipt['status'] == ('captured' if paid <= len(trace) else 'unavailable'),
                    f'Late measurement omitted or an early stop was padded in {directory}')
        if receipt['status'] == 'captured':
            receipt = engine.augment_measurement(receipt, masks, baseline, baseline_masks, ref)
        checked.append({'split': split, 'layout': layout, 'arm': arm, 'checkpoint': clock, **receipt})
    times = [step['decision_wall_s'] for step in trace]
    elapsed = manifest['elapsed_s']
    require(all(isfinite(t) and t >= 0 for t in [*times, elapsed]), f'Invalid monotonic timing in {directory}')
    row |= {**counts, **terminal, **{key: final_receipt[key] for key in ('C_free', 'F1_scene_05cm',
            'p_scene_05cm', 'r_scene_05cm', 'J', 'correct_free_cells', 'reachable_cells')},
            'status': 'completed_execution', 'elapsed_s': elapsed, 'decision_wall_s': sum(times),
            'study_prior': manifest['study_prior'], 'unknown_weight': manifest['unknown_weight'],
            'action_cost': manifest['action_cost'],
            'bootstrap_and_task_wall_s': manifest.get('bootstrap_and_task_wall_s'),
            'supplementary_local_neutral_audit_wall_s': manifest.get('supplementary_local_neutral_audit_wall_s'),
            'source_manifest_sha256': manifest_sha,
            'source_reference_receipt_sha256': ref['source_reference_receipt_sha256'],
            'terminal_mesh_sha256': manifest['files_sha256']['terminal/native_mesh.ply'],
            'terminal_belief_sha256': manifest['files_sha256']['terminal/belief.npz'],
            'terminal_checkpoint_sha256': manifest['files_sha256']['terminal/checkpoint.json'],
            'charged_steps_sha256': manifest['files_sha256']['steps.json'],
            'initial_packet_sha256': initial['initial_packet_sha256'], 'initial_owner_sha256': initial['initial_owner_sha256']}
    path = [{'split': split, 'layout': layout, 'arm': arm, **{key: step.get(key) for key in TRAJECTORY}} for step in trace]
    return row, checked, {'start': baseline, 'raw': initial_raw, 'trace': trace, 'receipt': final_receipt}, path


def pair_trace(layout, a_name, a, b_name, b):
    row = {'layout': layout, 'arm_a': a_name, 'arm_b': b_name}
    if a is None or b is None:
        return row | {'status': 'incomplete_pair'}
    require(native.exact_common_observation(a['start'], b['start']) and a['raw'] == b['raw'],
            f'Paired actors used different first observations for {layout}, {a_name}/{b_name}')
    a_steps, b_steps = a['trace'], b['trace']
    action, pose, observation = None, None, None
    for l_step, r_step in zip(a_steps, b_steps, strict=False):
        if action is None and l_step['action'] != r_step['action']:
            action = l_step['step'], l_step['action'], r_step['action']
        if pose is None and any(l_step[k] != r_step[k] for k in ('position', 'heading')):
            pose = l_step['step']
        if observation is None and any(l_step[k] != r_step[k] for k in SENSORS):
            observation = l_step['step']
    stops_differ = len(a_steps) != len(b_steps)
    return row | {'status': 'measured', 'first_action_difference_paid': action[0] if action else None,
                  'first_different_a_action': action[1] if action else None,
                  'first_different_b_action': action[2] if action else None,
                  'first_pose_difference_paid': pose, 'first_raw_observation_difference_paid': observation,
                  'equal_observed_action_prefix_paid': (action[0] - 1) if action else min(len(a_steps), len(b_steps)),
                  'first_different_stop_paid': min(len(a_steps), len(b_steps)) + 1 if stops_differ else None,
                  'exact_terminal_mesh': a['receipt']['files_sha256']['native_mesh.ply'] ==
                                         b['receipt']['files_sha256']['native_mesh.ply']}


def output_tables(output, name, rows, metadata):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (output / f'{name}.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)
    files.write_json(output / f'{name}.json', metadata | {'rows': rows})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', choices=('development', 'holdout'), required=True)
    parser.add_argument('--arms', nargs='+', required=True)
    parser.add_argument('--only-terminal', action='store_true')
    parser.add_argument('--require-finished', action='store_true')
    parser.add_argument('--run-root', type=Path, default=ROOT / 'runner/runs')
    parser.add_argument('--run-freeze', type=Path)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    require(len(set(args.arms)) == len(args.arms), 'Duplicate arm requested')
    actor, _, metadata = verify_freeze(args.split)
    run_freeze = args.run_freeze or ROOT / ('dev_freeze.json' if args.split == 'development' else 'holdout_freeze.json')
    frozen_schedule = read(run_freeze)
    freeze_sha, protocol_sha = sha(run_freeze), sha(ROOT / 'PROTOCOL.json')
    require(frozen_schedule['split'] == args.split and frozen_schedule['native_budget'] == PAID_CAP
            and frozen_schedule['protocol_sha256'] == protocol_sha
            and frozen_schedule['actor_scene_freeze_sha256'] == metadata['actor_freeze_sha256']
            and frozen_schedule['evaluator_scene_freeze_sha256'] == metadata['evaluator_freeze_sha256']
            and frozen_schedule['supervisor_only_evidence_sha256'][str(Path(__file__).resolve())] == sha(Path(__file__))
            and frozen_schedule['capture_paid'] == list(CLOCKS), 'Run freeze differs from native deadline/scope')
    jobs = {(job['layout'], job['arm']): job for job in frozen_schedule['schedule']}
    require(len(jobs) == len(frozen_schedule['schedule']), 'Non-unique registered policy layout')
    if args.require_finished:
        require(set(args.arms) == {job['arm'] for job in jobs.values()}, 'Official pass omits registered policy arms')
    metric = files.load_metric(read(files.SCENES / 'freeze.json'), read(files.OLD_EVAL / 'freeze.json'))
    episodes, checkpoints, divergences, paths = [], [], [], []
    for frozen in actor['layouts']:
        layout = frozen['layout']
        reference = load_reference(layout, frozen)
        pair_bindings = {}
        for arm in args.arms:
            episode, samples, binding, path = evaluate_run(args.split, layout, arm, reference, metric,
                                                         args.only_terminal, args.run_root, jobs[layout, arm],
                                                         freeze_sha, protocol_sha)
            episodes.append(episode)
            checkpoints.extend(samples)
            paths.extend(path)
            pair_bindings[arm] = binding
        divergences.extend(pair_trace(layout, a, pair_bindings[a], b, pair_bindings[b])
                           for a, b in combinations(args.arms, 2))
    output = args.out or ROOT / 'measurement/results' / args.split
    output.mkdir(exist_ok=True, parents=True)
    metadata |= {'split': args.split, 'arms': args.arms, 'budget': PAID_CAP, 'diagnostic_clocks': CLOCKS,
                 'run_freeze_sha256': freeze_sha, 'protocol_sha256': protocol_sha,
                 'independent_unit': 'building', 'only_terminal': args.only_terminal,
                 'measurement_is_actor_independent': True, 'policy_runs_started_by_evaluator': 0,
                 'task_clock': 'Native monotonic elapsed time from after startup to final native receipt.',
                 'planning_clock': 'Sum of measured planning calls associated with charged actions.'}
    for name, rows in [('episodes', episodes), ('checkpoints', checkpoints), ('first_divergence', divergences),
                       ('paths', paths)]:
        output_tables(output, name, rows, metadata)
    if args.require_finished and any(ep['status'] == 'pending' for ep in episodes):
        raise SystemExit('At least one registered execution has no terminal or failure receipt')
    print(f'{args.split}: {len(episodes)} assigned episodes, '
          f'{sum(ep["status"] == "completed_execution" for ep in episodes)} measured native terminals.')


if __name__ == '__main__':
    main()
