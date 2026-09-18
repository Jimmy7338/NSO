#!/usr/bin/env python3
"""Independent saved-evidence V35 audit and two preregistered counterfactuals.

Run only after the physical batch is terminal and sealed. No world, renderer,
mapper, TSDF or surface metric is constructed. At most two extra DP calls are
allowed: the semantic-only prior at the FIRST actual swapped correction for
each hypothesis, retaining the same future-information setting and state.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BATCH = ROOT / 'audit_results/v35_online_development_20260918'
OUTPUT = ROOT / 'audit_results/v35_semantic_chain_review_20260918'
PROTOCOL = ROOT / 'docs/research/V35_ONLINE_DEVELOPMENT_PROTOCOL_20260918.md'
MODES = ('G', 'S', 'swapped', 'swapped_no_feedback')
THRESHOLDS = ('02cm', '05cm', '10cm')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, reason):
    if not condition:
        raise AssertionError(reason)


def close(a, b, reason):
    require(math.isclose(float(a), float(b), rel_tol=1e-10, abs_tol=1e-12), reason)


def verify_seal(folder, filename, excluded=None):
    records = read(folder / filename)
    actual = {str(path.relative_to(folder)) for path in folder.rglob('*')
              if path.is_file() and path.name != filename and
              (excluded is None or path.relative_to(folder).parts[0] != excluded)}
    require(actual == set(records), 'sealed file set changed: ' + str(folder))
    for relative, expected in records.items():
        require(sha(folder / relative) == expected, 'sealed bytes changed: ' + relative)
    return records


def array_hash(value):
    import numpy as np
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    for part in (value.dtype.str.encode(), str(value.shape).encode(), value.tobytes()):
        digest.update(len(part).to_bytes(8, 'big')); digest.update(part)
    return digest.hexdigest()


def semantic_update(rgb, pose, mode, positions):
    """Independently decode the frozen controlled palette, never a GT label."""
    import numpy as np
    if mode == 'G':
        return 0.
    counts = {2: int(np.all(rgb == (40, 100, 220), axis=-1).sum()),
              3: int(np.all(rgb == (220, 60, 40), axis=-1).sum())}
    supported = [code for code, count in counts.items() if count >= 16]
    if len(supported) == 2:
        for value in positions.values():
            value.add(tuple(pose[:2]))
    elif len(supported) == 1 and sum(count > 0 for count in counts.values()) == 1:
        code = supported[0]
        if mode in ('swapped', 'swapped_no_feedback'):
            code = 5 - code
        positions[code].add(tuple(pose[:2]))
    present = [code for code, value in positions.items() if value]
    if len(present) == 1 and len(positions[present[0]]) >= 2:
        return math.log(9.) * (1 if present[0] == 2 else -1)
    return 0.


def metric_arithmetic(folder, result):
    """Check saved sample-score arithmetic and actual raster, no new Q call."""
    import numpy as np
    with np.load(folder / 'evaluation_floor.npz', allow_pickle=False) as floor:
        reachable = floor['reachable'].copy()
    checks = []
    for stage, record in result['stages'].items():
        measurement = record['measurement']
        with np.load(folder / (stage + '_maps.npz'), allow_pickle=False) as maps:
            coverage = float(np.mean(maps['belief'][reachable] != -1))
        close(measurement['C_map'], coverage, stage + ' actual raster coverage')
        for threshold in THRESHOLDS:
            row = measurement[threshold]
            precision, recall = float(row['precision']), float(row['recall'])
            require(0 <= precision <= 1 and 0 <= recall <= 1, 'invalid P/R')
            expected_f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.
            close(row['f1'], expected_f1, stage + '/' + threshold + ' F1 arithmetic')
            close(row['joint'], coverage * expected_f1, stage + '/' + threshold + ' J arithmetic')
        close(measurement['main_joint'], measurement['05cm']['joint'], 'primary metric identity')
        checks.append({'stage': stage, 'C_map': coverage, 'score_arithmetic_passed': True})
    return checks


def inspect_case(folder, case):
    import numpy as np
    verify_seal(folder, 'main_seal.json', 'replay')
    if not (folder / 'result.json').exists():
        return {'case': case, 'complete': False, 'failure': read(folder / 'failure.json')}, None
    result = read(folder / 'result.json')
    require(result['physical_case'] == case, 'case identity differs from frozen config')
    controller = read(folder / 'controller.json')
    actual = read(folder / 'trace.json')
    history = controller['history']
    require(len(history) == len(actual) == result['paid_actions'] + 1, 'paid observation/decision counts differ')
    require(history and len(history) <= 43, 'invalid observation budget')
    frozen = read(folder / 'prediction_freeze.json')
    require(frozen['before_first_quality_or_GT_reference_access'] is True, 'missing prediction-before-GT declaration')
    require('controller.json' in frozen['artifact_sha256'], 'controller not frozen before GT')
    for relative, expected in frozen['artifact_sha256'].items():
        require(sha(folder / relative) == expected, 'pre-evaluation prediction changed: ' + relative)
    plans = {row['step']: row for row in controller['plans']}
    require(len(plans) == len(controller['plans']), 'multiple selected plans for one paid state')
    calls = controller['calls']
    require({row['module'] for row in calls} == {'OV-SDF', 'STGHP', 'RPN-UQ', 'IGCR'}, 'four-module call set differs')
    semantic_positions = {2: set(), 3: set()}
    seen = set(); geometry = 0.; previous_true_probability = .5
    first_semantic = first_geometry = correction = None
    mode, hypothesis = case['mode'], case['hypothesis']
    checked_packets = 0
    for paid, (entry, observed) in enumerate(zip(history, actual)):
        state, posterior = entry['state'], entry['posterior']
        pose = tuple(state['pose'])
        require(state['step'] == posterior['step'] == observed['paid'] == paid, 'paid step mismatch')
        require(list(pose) == posterior['pose'] == observed['pose_v33'], 'observed/controller pose mismatch')
        require(posterior['mode'] == mode, 'posterior intervention mismatch')
        require(state['remaining_budget'] == 42 - paid, 'remaining budget mismatch')
        for module, operation in (('OV-SDF', 'observed_semantic_and_map_state'),
                                  ('IGCR', 'paid_geometry_residual_feedback'),
                                  ('STGHP', 'update_public_topology_state')):
            require(sum(row['module'] == module and row['operation'] == operation and row['step'] == paid for row in calls) == 1,
                    'paid observation lacks exactly one module update: ' + module)
        if paid:
            require(observed['action'] == history[paid-1]['next_action'], 'issued action not the next executed action')
        else:
            require(observed['action'] is None, 'initial observation must be unpaid')
        with np.load(folder / 'packets' / f'{paid:03d}.npz', allow_pickle=False) as packet:
            depth, rgb, ranges = packet['frame__depth_m'], packet['frame__color_rgb'], packet['scan__ranges_m']
            require(not np.any(packet['frame__semantic']), 'GT/semantic labels entered saved mapper frame')
            for name, value in (('depth', depth), ('rgb', rgb), ('ranges', ranges)):
                require(array_hash(value) == posterior['observation_sha256'][name], 'posterior did not use this paid packet: ' + name)
            semantic = semantic_update(rgb, pose, mode, semantic_positions)
        checked_packets += 1
        close(posterior['semantic_log_odds'], semantic, 'RGB-derived semantic prior differs')
        fresh = pose not in seen
        require(posterior['new_geometry_pose'] == fresh, 'repeat observation credited as a new pose')
        seen.add(pose)
        applied = float(posterior['geometry_applied_log_odds'])
        expected_applied = float(posterior['residuals']['log_likelihood_ratio']) if fresh and mode != 'swapped_no_feedback' else 0.
        close(applied, expected_applied, 'geometry update/repetition intervention differs')
        require(abs(applied) <= 6. + 1e-12, 'per-pose likelihood cap violated')
        geometry = min(24., max(-24., geometry + applied))
        close(posterior['geometry_log_odds'], geometry, 'cumulative geometry update differs')
        probability0 = 1 / (1 + math.exp(-(geometry + semantic)))
        close(posterior['probabilities'][0], probability0, 'posterior arithmetic differs')
        close(posterior['probabilities'][1], 1 - probability0, 'posterior complement differs')
        require(state['probabilities'] == posterior['probabilities'], 'planner state omitted observed posterior')
        true_probability = posterior['probabilities'][hypothesis]
        brief = {'paid_observation_step': paid, 'pose': list(pose),
                 'true_class_probability': true_probability, 'probabilities': posterior['probabilities'],
                 'semantic_log_odds': semantic, 'geometry_applied_log_odds': applied,
                 'next_action': entry['next_action'],
                 'next_action_execution_step': paid + 1 if entry['next_action'] is not None else None}
        if semantic and first_semantic is None:
            first_semantic = dict(brief)
        if applied and first_geometry is None:
            first_geometry = dict(brief)
        if (mode == 'swapped' and correction is None and applied and
                previous_true_probability < .5 < true_probability):
            correction = dict(brief, before_true_class_probability=previous_true_probability,
                              state=state, posterior=posterior,
                              visited_nodes=sorted({row['state']['node'] for row in history[:paid+1]}))
        previous_true_probability = true_probability
        action = entry['next_action']
        if action is not None:
            require(paid in plans, 'issued action has no global-plan evidence')
            plan = plans[paid]
            require(plan['action'] == action and plan['from_node'] == state['node'], 'global selection mismatch')
            require(plan['probabilities'] == posterior['probabilities'], 'selection used a different posterior')
            local = [row for row in calls if row['module'] == 'RPN-UQ' and row['step'] == paid]
            require(len(local) == 1 and local[0]['allowed'] and local[0]['action'] == action,
                    'issued action lacks a unique passing local validation')
            if paid >= 18:
                require(plan['planning']['phase'] == 'online_belief_global_planning', 'post-prefix action was not online planned')
                close(plan['planning']['probability0'], probability0, 'planner argument probability differs')
                require(plan['planning']['geometry_feedback'] == (mode != 'swapped_no_feedback'), 'future-feedback ablation differs')
    require(controller['actions'] == [row['action'] for row in actual[1:]], 'controller/actual action list differs')
    final = result['stages']['final']['measurement']
    eligible = bool(final['C_map'] >= .8 and result['returned'] and result['collisions'] == 0 and result['paid_actions'] <= 42)
    require(final['eligible'] == eligible, 'terminal qualification differs')
    replay = folder / 'replay'
    verify_seal(replay, 'seal.json')
    replay_result = read(replay / 'result.json') if (replay / 'result.json').exists() else None
    if replay_result is not None:
        require(replay_result['main_result_sha256'] == sha(folder / 'result.json'), 'replay refers to another main result')
        main_identity = read(folder / 'started.json')['process_identity']
        replay_identity = read(replay / 'started.json')['process_identity']
        core = ('pid', 'process_start_ticks', 'pid_namespace', 'boot_id')
        require(any(main_identity[key] != replay_identity[key] for key in core), 'replay was not a different process identity')
        require(replay_result['main_process_identity'] == main_identity and replay_result['replay_process_identity'] == replay_identity,
                'replay identity receipt differs from process-start records')
        require(replay_result['fresh_process_identity_verified'] is True and
                replay_result['all_controller_decisions_and_receipts_equal'] is True,
                'full online independent replay not established')
    report = {'case': case, 'complete': True, 'checked_paid_packets': checked_packets,
              'saved_evidence_checks_passed': True, 'eligible': eligible,
              'independent_replay_passed': bool(replay_result and replay_result['passed']),
              'controlled_prefix_cue_passed': bool(result['prefix_cue']['passed']),
              'four_module_paid_call_chain_passed': True,
              'first_semantic_prior': first_semantic, 'first_actual_geometry_feedback': first_geometry,
              'first_wrong_prior_correction': None if correction is None else {k: v for k, v in correction.items() if k not in ('state', 'posterior', 'visited_nodes')},
              'final_measurement': final, 'metric_arithmetic': metric_arithmetic(folder, result),
              'posterior_likelihood_scope': 'saved residual values checked for accounting; raw geometry likelihood correctness inherited from frozen unit/gate evidence',
              'action_attribution_scope': 'posterior update followed by an action is not alone evidence that feedback changed that action'}
    return report, {'controller': controller, 'trace': actual, 'correction': correction, 'result': result}


def first_divergence(a, b, label):
    left, right = a['controller']['history'], b['controller']['history']
    index = next((i for i, (x, y) in enumerate(zip(left, right)) if x['next_action'] != y['next_action']), None)
    if index is None:
        require(len(left) == len(right), 'different trajectory length without an action divergence')
        return {'comparison': label, 'first_decision_divergence': None, 'same_action_trajectory': True}
    x, y = left[index], right[index]
    xs, ys = x['state'], y['state']
    xp, yp = x['posterior'], y['posterior']
    same_geometry_history = all(a['trace'][j]['nonsemantic'] == b['trace'][j]['nonsemantic'] for j in range(index+1))
    geometry_seen_a = any(abs(row['posterior']['geometry_applied_log_odds']) > 0 for row in left[:index+1])
    geometry_seen_b = any(abs(row['posterior']['geometry_applied_log_odds']) > 0 for row in right[:index+1])
    same_posterior = xs['probabilities'] == ys['probabilities']
    if label == 'swapped_vs_swapped_no_feedback' and not geometry_seen_a and same_posterior:
        interpretation = 'divergence before actual geometry correction; future-information anticipation differs'
    elif label == 'S_vs_G' and same_geometry_history and xp['semantic_log_odds'] != yp['semantic_log_odds']:
        interpretation = 'same observed geometry with different RGB-derived semantic prior'
    else:
        interpretation = 'paired trajectory divergence; causal isolation requires the preregistered same-state audit'
    return {'comparison': label, 'same_action_trajectory': False,
            'first_decision_divergence': index, 'first_different_executed_action_step': index+1,
            'left_action': x['next_action'], 'right_action': y['next_action'],
            'same_pose': xs['pose'] == ys['pose'], 'same_masks': xs['masks_hex'] == ys['masks_hex'],
            'same_nonsemantic_observation_history': same_geometry_history,
            'same_posterior': same_posterior, 'left_probabilities': xs['probabilities'],
            'right_probabilities': ys['probabilities'],
            'left_actual_geometry_feedback_already_seen': geometry_seen_a,
            'right_actual_geometry_feedback_already_seen': geometry_seen_b,
            'interpretation': interpretation}


def paired_metrics(reports):
    if len(reports) != 8 or not all(row['complete'] for row in reports.values()):
        return None
    output = {}
    for left, right in (('S', 'G'), ('S', 'swapped'), ('swapped', 'swapped_no_feedback')):
        thresholds = {}
        for threshold in THRESHOLDS:
            a = [reports[h, left]['final_measurement'][threshold]['joint'] for h in (0, 1)]
            b = [reports[h, right]['final_measurement'][threshold]['joint'] for h in (0, 1)]
            differences = [x-y for x, y in zip(a, b)]
            average, baseline = sum(differences)/2, sum(b)/2
            thresholds[threshold] = {'left_mean_joint': sum(a)/2, 'right_mean_joint': baseline,
                                     'per_hypothesis_joint_differences': differences,
                                     'mean_joint_difference': average,
                                     'relative_mean_joint_difference': average/baseline if baseline else None}
        output[left + '_minus_' + right] = thresholds
    return output


def counterfactual(hypothesis, saved, counts):
    correction = saved['correction']
    if correction is None:
        return {'hypothesis': hypothesis, 'available': False, 'passed': False,
                'reason': 'no actual first wrong-prior correction; no later state search permitted'}
    state, posterior = correction['state'], correction['posterior']
    step = state['step']
    real_plan = next((row for row in saved['controller']['plans'] if row['step'] == step), None)
    if real_plan is None or correction['next_action'] is None:
        return {'hypothesis': hypothesis, 'available': False, 'passed': False,
                'reason': 'first correction has no subsequent executed action; no later state search permitted'}
    require(counts['counterfactual_DP_attempts'] < 2, 'counterfactual attempt cap exceeded')
    from nso.online_planner_v35 import ForecastBeliefPlannerV35, load_public_models_v35
    from nso.observation_belief_v35 import PublicTemplatesV35
    models, _ = load_public_models_v35(ROOT, 'P00')
    templates = PublicTemplatesV35.from_saved(ROOT / 'audit_results/v34_pixel_information_20260918', 'P00', models[0].poses)
    informative = [node for node in range(len(models[0].poses)) if templates.template_information(node)['informative']]
    planner = ForecastBeliefPlannerV35(models, informative, reliability=.99,
                                     maximum_states=1500000, maximum_seconds=20.)
    semantic_probability0 = 1 / (1 + math.exp(-posterior['semantic_log_odds']))
    counts['counterfactual_DP_attempts'] += 1
    branch = planner.select(state['node'], state['remaining_budget'],
                            tuple(int(value, 16) for value in state['masks_hex']),
                            semantic_probability0, geometry_feedback=True,
                            excluded_information_nodes=correction['visited_nodes'])
    counts['counterfactual_DP_completed'] += 1
    counts['counterfactual_memo_states'] += branch['memo_states']
    actual_action = correction['next_action']
    require(real_plan['planning']['geometry_feedback'] is True, 'actual corrected branch must include same future feedback')
    return {'hypothesis': hypothesis, 'available': True, 'first_correction_paid_step': step,
            'fixed_node': state['node'], 'fixed_pose': state['pose'], 'fixed_masks_hex': state['masks_hex'],
            'fixed_remaining_budget': state['remaining_budget'], 'fixed_visited_nodes': correction['visited_nodes'],
            'geometry_feedback_in_both_branches': True,
            'actual_probability0': posterior['probabilities'][0],
            'counterfactual_semantic_only_probability0': semantic_probability0,
            'actual_action_from_saved_online_plan': actual_action,
            'counterfactual_action': branch['action'], 'action_changed': actual_action != branch['action'],
            'passed': actual_action != branch['action'], 'counterfactual_plan': branch,
            'actual_corrected_branch_DP_recomputed': False, 'later_state_search_performed': False,
            'scope': 'same saved state and future information; substitute only posterior, no new physical outcome'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', type=Path, default=BATCH)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        require(os.environ.get(name) == '1', name + '=1 required')
    require(sys.dont_write_bytecode, '-B required')
    batch, output = args.batch.resolve(), args.output.resolve()
    require((batch / 'final_seal.json').is_file() and (batch / 'result.json').is_file(),
            'physical batch must finish or retain a terminal stop before this audit')
    from nso.research_evidence_v31 import freeze, write, seal, verify_sources, verify_inventory
    verify_seal(batch, 'final_seal.json')
    verify_sources(batch)
    config, summary = read(batch / 'config.json'), read(batch / 'result.json')
    require(summary['status'] in ('complete', 'stopped'), 'nonterminal physical batch')
    require([(c['hypothesis'], c['mode']) for c in config['physical_cases']] == [(h, mode) for h in (0, 1) for mode in MODES],
            'preregistered eight-cell matrix differs')
    output.mkdir(exist_ok=False)
    counts = {'new_worlds': 0, 'new_sensor_queries': 0, 'new_main_tasks': 0,
              'new_mapper_or_TSDF_calls': 0, 'new_surface_metric_calls': 0,
              'counterfactual_DP_attempts': 0, 'counterfactual_DP_completed': 0,
              'counterfactual_memo_states': 0}
    reports, saved = {}, {}
    try:
        freeze(output, [Path(__file__), PROTOCOL],
               input_sha256={str(batch / name): sha(batch / name) for name in ('config.json', 'result.json', 'final_seal.json', 'manifest.json')},
               registered_counterfactual_count_cap=2, later_state_search_allowed=False)
        for case in config['physical_cases']:
            folder = batch / f"case{case['index']:02d}"
            if not folder.exists():
                continue
            report, details = inspect_case(folder, case)
            key = case['hypothesis'], case['mode']
            reports[key] = report
            if details is not None:
                saved[key] = details
        comparisons = []
        interventions = []
        for hypothesis in (0, 1):
            for left, right, label in (('S', 'G', 'S_vs_G'), ('swapped', 'swapped_no_feedback', 'swapped_vs_swapped_no_feedback')):
                if (hypothesis, left) in saved and (hypothesis, right) in saved:
                    comparisons.append(dict(hypothesis=hypothesis,
                        **first_divergence(saved[hypothesis, left], saved[hypothesis, right], label)))
            if (hypothesis, 'swapped') in saved:
                interventions.append(counterfactual(hypothesis, saved[hypothesis, 'swapped'], counts))
                write(output, output / f'h{hypothesis}_first_correction_intervention.json', interventions[-1])
        metrics = paired_metrics(reports)
        for name, comparison in (metrics or {}).items():
            source_name = {'S_minus_G': 'S_vs_G', 'S_minus_swapped': 'S_vs_swapped',
                           'swapped_minus_swapped_no_feedback': 'swapped_vs_swapped_no_feedback'}[name]
            for threshold, values in comparison.items():
                declared = summary['prescribed_paired_comparison'][threshold][source_name]
                close(values['mean_joint_difference'], declared['paired_mean_joint_difference'], 'collector paired mean differs')
                for actual, expected in zip(values['per_hypothesis_joint_differences'], declared['paired_joint_differences']):
                    close(actual, expected, 'collector per-hypothesis difference differs')
        all_qualified = len(reports) == 8 and all(row.get('eligible', False) for row in reports.values())
        all_replays = len(reports) == 8 and all(row.get('independent_replay_passed', False) for row in reports.values())
        prefix_equal = len(saved) == 8 and all(len(item['trace']) >= 19 for item in saved.values())
        if prefix_equal:
            reference = next(iter(saved.values()))['trace'][:19]
            prefix_equal = all(all(a['nonsemantic'] == b['nonsemantic'] for a, b in zip(reference, item['trace'][:19]))
                               for item in saved.values())
        cue_passed = len(reports) == 8 and all(row.get('controlled_prefix_cue_passed', False) for row in reports.values())
        semantic_gain = bool(metrics and metrics['S_minus_G']['05cm']['mean_joint_difference'] > 0 and
                             all(metrics['S_minus_G'][threshold]['mean_joint_difference'] >= 0 for threshold in ('02cm', '10cm')))
        semantic_intervention_gain = bool(metrics and metrics['S_minus_swapped']['05cm']['mean_joint_difference'] > 0)
        feedback_gain = bool(metrics and metrics['swapped_minus_swapped_no_feedback']['05cm']['mean_joint_difference'] > 0)
        correction_action_change = any(row['passed'] for row in interventions)
        collector_prerequisites = bool(summary['qualification_and_replay_gate_passed'])
        result = {'status': 'passed', 'saved_evidence_verified': True,
                  'source_batch_result_sha256': sha(batch / 'result.json'),
                  'case_reports': list(reports.values()), 'first_action_divergences': comparisons,
                  'paired_metrics': metrics, 'first_correction_same_state_interventions': interventions,
                  'checks': {'all_eight_final_qualified': all_qualified, 'all_eight_independent_replays_passed': all_replays,
                             'collector_qualification_and_replay_contract_passed': collector_prerequisites,
                             'all_eight_nonsemantic_paid_prefixes_equal': prefix_equal,
                             'all_eight_controlled_prefix_cue_prerequisites_passed': cue_passed,
                             'S_minus_G_main_positive_secondary_not_reversed': semantic_gain,
                             'correct_minus_wrong_class_main_positive': semantic_intervention_gain,
                             'feedback_main_gain_positive': feedback_gain,
                             'at_least_one_first_actual_correction_changes_same_state_action': correction_action_change},
                  'complete_development_gate_passed': all((collector_prerequisites, all_qualified, all_replays, prefix_equal, cue_passed, semantic_gain, semantic_intervention_gain, feedback_gain, correction_action_change)),
                  'counts': counts, 'surface_distance_metrics_recomputed': False,
                  'quality_values_source': 'sealed actual TSDF evaluations; only arithmetic and saved raster checked independently',
                  'scope': 'one known two-template parent, artificial RGB, exact pose, CPU online development; not natural semantics or full learned ANS validation'}
        write(output, output / 'result.json', result)
        seal(output); verify_sources(output); verify_inventory(output)
        print(json.dumps({'status': 'passed', 'complete_development_gate_passed': result['complete_development_gate_passed'],
                          'counterfactual_DP_completed': counts['counterfactual_DP_completed'],
                          'result_sha256': sha(output / 'result.json')}))
    except BaseException:
        write(output, output / 'failure.json', {'status': 'failed', 'error': traceback.format_exc(),
                                              'counts': counts, 'case_reports_completed': list(reports.values()),
                                              'implicit_retry_allowed': False})
        if not (output / 'artifact_hashes.json').exists():
            seal(output)
        raise


if __name__ == '__main__':
    main()
