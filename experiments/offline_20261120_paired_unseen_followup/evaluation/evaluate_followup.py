"""Read only evaluation of the four-layout baseline and class-prior follow-up."""

import argparse
import csv
import json
import sys
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FOLLOWUP = HERE.parent
B11 = FOLLOWUP.parent / 'paired_unseen_gain_20261120'
sys.path.insert(0, str(B11 / 'evaluation/full_policy'))
from evaluate_full_run import (Evaluator, audit_frames, audit_trace, check_models,
                               checked, coverage_from_belief, digest,
                               first_correct_free_actions, read, resource, sha, write)

LAYOUTS = ('chain_a', 'cycle_a', 'chain_b', 'cycle_b')
COLUMNS = ('layout', 'arm', 'status', 'paid_actions', 'budget', 'coverage_pct',
           'A95', 'A100', 'shelf_recall_5cm_pct', 'cabinet_recall_5cm_pct',
           'object_weighted_recall_5cm_pct', 'scene_F1_5cm_pct', 'path_length_m',
           'returned', 'collisions', 'elapsed_s', 'decision_wall_s', 'termination')


def verify_g(attempt, result, layout, evaluator):
    binding = read(attempt / 'B11_execution_binding.json')
    assert binding['predictor'] == 'visibility'
    assert binding['selected_score_mode'] == 'G'
    assert binding['full_posterior_same_quadrature'] and binding['geometry_from_acquired_frames_only']
    assert not binding['short_cost_band_used']
    assert checked(binding['execution_freeze'], evaluator.inputs) == B11 / 'execution/freeze.json'
    extension = read(checked(binding['extension_freeze'], evaluator.inputs))
    assert checked(binding['new_protocol'], evaluator.inputs) == FOLLOWUP / 'g_visibility/protocol.json'
    checked(binding['parent_S_visibility'], evaluator.inputs)
    old_paths = read(B11 / 'execution/freeze.json')['files_sha256']
    for name, checksum in {**old_paths, **extension['files_sha256']}.items():
        p = Path(name)
        if str(p.resolve()) not in evaluator.inputs:
            checked(dict(path=name, sha256=checksum), evaluator.inputs)
        else:
            assert evaluator.inputs[str(p.resolve())] == checksum
    execution = read(attempt / 'attempt.json')
    assert execution['protocol_sha256'] == sha(FOLLOWUP / 'g_visibility/protocol.json')
    assert execution['prepared_freeze_sha256'] == sha(B11 / 'execution/layouts' / layout / 'prepared/freeze.json')
    assert result['effective_args']['cpu_score_mode'] == 'G'
    old_result = read(B11 / 'execution/layouts' / layout / 'runs/S_visibility/attempt_001/result.json')
    ignored = {'cpu_score_mode', 'dual_template_arm', 'run_id'}
    def matched_args(record):
        return {k: v for k, v in record['effective_args'].items() if k not in ignored}
    assert matched_args(result) == matched_args(old_result)

    calls = read(attempt / 'template_audit.json')
    for call in calls:
        assert call['arm'] == 'G_visibility' and call['B11_predictor'] == 'visibility'
        assert call['acquisition_diagnostics'] is None
        check_models(call)
    choices = [r['outputs'] for r in read(attempt / 'choices.json')]
    for choice in choices:
        candidates, scores = choice['candidates'], choice['scores']['G']
        assert choice['mode'] == 'G' and len(candidates) == len(scores)
        winner = min(range(len(scores)), key=lambda i: (-scores[i], candidates[i]['cost'],
                                                       candidates[i]['candidate_id']), default=None)
        expected = candidates[winner]['candidate_id'] if winner is not None and scores[winner] > 0 else None
        selected = choice['selected']
        assert expected == (selected['candidate_id'] if selected is not None else None)
    frame_audit = audit_frames(read(attempt / 'template_frame_receipts.json'),
                               read(attempt / 'final_template_receipt.json'),
                               result['paid_actions'], len(calls))
    assert len(calls) == result['global_planning_calls']
    return dict(model_calls_audited=len(calls), selection_calls_audited=len(choices), frames=frame_audit)


def verify_frontier(attempt, result, evaluator):
    binding = read(attempt / 'baseline_binding.json')
    assert binding['same_observation_update_and_guard'] and not binding['same_candidates']
    freeze = read(checked(binding['frontier_freeze'], evaluator.inputs))
    checked(binding['B11_execution_freeze'], evaluator.inputs)
    checked(binding['B11_initial_state'], evaluator.inputs)
    protocol = checked(binding['protocol'], evaluator.inputs)
    assert read(attempt / 'attempt.json')['protocol_sha256'] == sha(protocol)
    for path, checksum in freeze['files_sha256'].items():
        if str(Path(path).resolve()) not in evaluator.inputs:
            checked(dict(path=path, sha256=checksum), evaluator.inputs)
        else:
            assert evaluator.inputs[str(Path(path).resolve())] == checksum
    assert result['no_truth_based_stopping'] and result['no_reference_used_by_policy']
    assert result['template_scoring_calls'] == 0
    choices = read(attempt / 'choices.json')
    for choice in choices:
        candidates = choice['reachable_candidate_scored']
        winner = min(candidates, key=lambda c: (c['distance_cells'], -c['gain_cells'], c['cell']), default=None)
        assert not choice['truth_or_reference_input']
        assert bool(winner is not None) == choice['chosen']
        assert choice['selected_cell'] == (winner['cell'] if winner is not None else None)
        assert choice['selected_heading'] == (winner['heading'] if winner is not None else None)
    assert len(choices) == result['global_planning_calls']
    frames = audit_frames(read(attempt / 'frame_receipts.json'), read(attempt / 'final_frame_receipt.json'),
                          result['paid_actions'], 0)
    interventions = read(attempt / 'interventions.json')
    return dict(selection_calls_audited=len(choices), frames=frames,
                intervention_counts={key: sum(e['event'] == key for e in interventions)
                                     for key in sorted({e['event'] for e in interventions})})


def measure(attempt, layout, arm, evaluator):
    manifest = read(attempt / 'artifact_manifest.json')
    for name, checksum in manifest['files_sha256'].items():
        checked(dict(path=str(attempt / name), sha256=checksum), evaluator.inputs)
    required = {'result.json', 'steps.json', 'initial_evaluation.json', 'final_belief.npz',
                'final_tsdf_mesh.ply'}
    assert required <= set(manifest['files_sha256'])
    result = read(attempt / 'result.json')
    prep = B11 / 'execution/layouts' / layout / 'prepared'
    prep_freeze = read(checked(resource(prep / 'freeze.json'), evaluator.inputs))
    for name in ('initial_state.json', 'initial_frame.npz', 'initial_scan.npz'):
        checked(dict(path=str(prep / name), sha256=prep_freeze['prepared_files_sha256'][name]), evaluator.inputs)
    initial_state = read(prep / 'initial_state.json')
    assert result['layout'] == layout and result['arm'] == arm
    assert result['initial_packet_sha256'] == initial_state['initial_packet_sha256']
    assert result['paid_action_budget'] == 256 and 0 <= result['paid_actions'] <= 256
    trace = read(attempt / 'steps.json')
    assert len(trace) == result['paid_actions']
    trajectory_sha = digest([(s['action'], s['position'], s['heading']) for s in trace])
    assert trajectory_sha == result.get('trajectory_sha256', trajectory_sha)
    initial = read(attempt / 'initial_evaluation.json')
    assert initial == initial_state['evaluation']
    hits = first_correct_free_actions(initial['free_correct_fraction'], trace)
    for key, hit in hits.items():
        assert result.get('first_free_correct_action', hits)[key] == hit

    reference = evaluator.reference(B11 / 'scene/evaluator_references' / layout)
    truth = B11 / 'scene/prepared' / layout / 'evaluator_truth.npz'
    with np.load(truth, allow_pickle=False) as z:
        reachable = z['reachable']
    for event in trace:
        assert event['reference_free_cells'] == int(reachable.sum())
        assert event['free_correct_fraction'] == event['correctly_free_cells'] / event['reference_free_cells']
    with np.load(attempt / 'final_belief.npz', allow_pickle=False) as z:
        coverage = coverage_from_belief(z['belief'], reachable)
        final_pose = [*map(int, z['position']), int(z['heading'])]
    assert result['final_pose'] == final_pose
    for k, val in coverage.items():
        assert np.isclose(val, result['final_evaluation'][k], rtol=0, atol=1e-14), k
        assert not trace or np.isclose(val, trace[-1][k], rtol=0, atol=1e-14), k
    returned = final_pose == initial_state['anchor']
    assert returned == result['final_pose_equals_initial_anchor']
    kinematics = audit_trace(trace, initial_state['anchor'], result, cell_m=initial_state['actual_config']['resolution_m'])

    mesh = resource(attempt / 'final_tsdf_mesh.ply')
    metrics = evaluator.native(mesh, reference)['metrics']
    names = ('open31', 'closed118')
    recalls = {name: metrics['target_surface_metrics'][name]['recall_05cm'] for name in names}
    areas = {name: float(reference['weights'][name].sum()) for name in names}
    weighted = sum(recalls[name] * areas[name] for name in names) / sum(areas.values())
    prior_audit = verify_g(attempt, result, layout, evaluator) if arm == 'G_visibility' else None
    frontier_audit = verify_frontier(attempt, result, evaluator) if arm in ('F_nearest', 'F_strict') else None
    return dict(layout=layout, arm=arm, status=result['status'], attempt=str(attempt),
                initial_packet_sha256=result['initial_packet_sha256'], paid_actions=len(trace), budget=256,
                coverage=coverage, A95=hits['0.95'], A100=hits['1.0'], censoring_actions=len(trace),
                object_recall_5cm=recalls, object_weighted_recall_5cm=weighted,
                surface_metrics=metrics, path_length_m=result['path_length_m'], returned=returned,
                collisions=result['collisions'], elapsed_s=result['elapsed_s'],
                decision_wall_s=result['decision_wall_s_total'], termination=result['termination'],
                trajectory_sha256=trajectory_sha, native_mesh_sha256=mesh['sha256'],
                manifest_sha256=sha(attempt / 'artifact_manifest.json'),
                kinematics=kinematics, class_prior_audit=prior_audit, frontier_audit=frontier_audit)


def attempts(layout, arm):
    if arm == 'S_visibility':
        return [B11 / 'execution/layouts' / layout / 'runs/S_visibility/attempt_001']
    subdir = {'G_visibility': 'g_visibility', 'F_nearest': 'frontier', 'F_strict': 'frontier_strict'}[arm]
    return sorted((FOLLOWUP / subdir / 'layouts' / layout / 'runs' / arm).glob('attempt_*'))


def pair(base, candidate):
    assert base['initial_packet_sha256'] == candidate['initial_packet_sha256']
    assert base['budget'] == candidate['budget']
    a, b = (read(Path(row['attempt']) / 'steps.json') for row in (base, candidate))
    different = next((i for i, (x, y) in enumerate(zip(a, b), 1)
                      if any(x[k] != y[k] for k in ('action', 'position', 'heading'))), None)
    if different is None and len(a) != len(b):
        different = min(len(a), len(b)) + 1
    trace_prefix_end = min(len(a), len(b)) if different is None else different - 1
    assert all(x['packet_sha256'] == y['packet_sha256'] for x, y in zip(a[:trace_prefix_end], b[:trace_prefix_end]))
    first_target_change = None
    matched_plans = 0
    if candidate['arm'] == 'G_visibility':
        first_choices, second_choices = (read(Path(row['attempt']) / 'choices.json') for row in (base, candidate))
        for s, g in zip(first_choices, second_choices):
            so, go = s['outputs'], g['outputs']
            paid = 256 - s['inputs']['remaining_budget']
            if paid > trace_prefix_end or g['inputs'] != s['inputs']:
                break
            assert so['candidates'] == go['candidates'] and so['scores'] == go['scores']
            matched_plans += 1
            s_id, g_id = (o['selected']['candidate_id'] if o['selected'] else None for o in (so, go))
            if s_id != g_id:
                first_target_change = dict(after_actions=paid, S_candidate=s_id, G_candidate=g_id,
                                           common_candidate_count=len(so['candidates']))
                break
    return dict(first_different_action=different, matched_packet_prefix_actions=trace_prefix_end,
                matched_prior_plans_audited=matched_plans, first_matched_state_target_change=first_target_change,
                same_native_mesh=base['native_mesh_sha256'] == candidate['native_mesh_sha256'],
                delta_correct_free_pct=100 * (candidate['coverage']['free_correct_fraction'] - base['coverage']['free_correct_fraction']),
                delta_object_weighted_recall_5cm_pp=100 * (candidate['object_weighted_recall_5cm'] - base['object_weighted_recall_5cm']),
                delta_scene_F1_5cm_pp=100 * (candidate['surface_metrics']['f1_05cm'] - base['surface_metrics']['f1_05cm']),
                A95=[base['A95'], candidate['A95']], A100=[base['A100'], candidate['A100']],
                returned=[base['returned'], candidate['returned']])


def flat(row):
    return dict(layout=row['layout'], arm=row['arm'], status=row['status'], paid_actions=row['paid_actions'],
                budget=row['budget'], coverage_pct=100 * row['coverage']['free_correct_fraction'],
                A95=row['A95'], A100=row['A100'], shelf_recall_5cm_pct=100 * row['object_recall_5cm']['open31'],
                cabinet_recall_5cm_pct=100 * row['object_recall_5cm']['closed118'],
                object_weighted_recall_5cm_pct=100 * row['object_weighted_recall_5cm'],
                scene_F1_5cm_pct=100 * row['surface_metrics']['f1_05cm'], path_length_m=row['path_length_m'],
                returned=row['returned'], collisions=row['collisions'], elapsed_s=row['elapsed_s'],
                decision_wall_s=row['decision_wall_s'], termination=json.dumps(row['termination'], sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=HERE / 'results')
    parser.add_argument('--prior-only', action='store_true')
    parser.add_argument('--strict-frontier', action='store_true')
    args = parser.parse_args()
    arms = ('S_visibility', 'G_visibility') if args.prior_only else ('S_visibility', 'G_visibility', 'F_nearest')
    if args.strict_frontier and not args.prior_only:
        arms += ('F_strict',)
    args.output.mkdir(parents=True, exist_ok=True)
    evaluator = Evaluator()
    rows, incomplete, errors = {}, [], []
    for layout in LAYOUTS:
        rows[layout] = {}
        for arm in arms:
            directories = attempts(layout, arm)
            if not directories:
                incomplete.append(dict(layout=layout, arm=arm, reason='missing attempt'))
                continue
            for attempt in directories:
                manifest = attempt / 'artifact_manifest.json'
                finalized = manifest.is_file() and (arm != 'G_visibility' or
                            'B11_execution_binding.json' in read(manifest)['files_sha256'])
                if not finalized:
                    incomplete.append(dict(layout=layout, arm=arm, attempt=str(attempt), reason='no final manifest',
                                           files=sorted(p.name for p in attempt.iterdir())))
                    continue
                try:
                    measured = measure(attempt, layout, arm, evaluator)
                    rows[layout].setdefault(arm, []).append(measured)
                except Exception:
                    errors.append(dict(layout=layout, arm=arm, attempt=str(attempt), traceback=traceback.format_exc()))
    evaluator.verify()
    paired = {}
    for layout, conditions in rows.items():
        if 'S_visibility' not in conditions:
            continue
        base = conditions['S_visibility'][0]
        paired[layout] = {arm: [pair(base, row) for row in runs]
                          for arm, runs in conditions.items() if arm != 'S_visibility'}
    with (args.output / 'layouts.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        for conditions in rows.values():
            for runs in conditions.values():
                writer.writerows(flat(row) for row in runs)
    write(args.output / 'summary.json', dict(layout_count=4, independent_unit='constructed layout', rows=rows,
          comparisons_relative_to_S_visibility=paired, missing_or_partial=incomplete, integrity_errors=errors,
          evaluator={'source': resource(Path(__file__)), 'reference_and_utility_inputs': evaluator.inputs}))
    print(json.dumps(dict(evaluated=sum(len(x) for y in rows.values() for x in y.values()),
                          missing_or_partial=incomplete, integrity_errors=errors, output=str(args.output))))
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
