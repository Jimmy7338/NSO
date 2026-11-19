#!/usr/bin/env python3
"""Read completed episode artifacts; never import or execute actor/world code."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
ARMS = ('C', 'J', 'G', 'S', 'S_no_fit', 'S_no_history')
LAYOUTS = ('shelf_left', 'shelf_right')
MODE = {'C': 'C', 'J': 'N', 'G': 'G', 'S': 'S', 'S_no_fit': 'S', 'S_no_history': 'S'}


def load(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def span(values):
    values = list(values)
    return [min(values), max(values)] if values else None


def winner(output, mode):
    candidates = output['candidates']
    if not candidates:
        return None
    scores = output['scores'][mode]
    i = min(range(len(candidates)), key=lambda i: (-scores[i], candidates[i]['cost'], candidates[i]['candidate_id']))
    return candidates[i]['candidate_id'] if scores[i] > 0 else None


def summarize_model(calls, name):
    models = [x['model_audits'][name] for x in calls]
    objects = [(call['mapper_frames'] - 1, obj)
               for call in calls for obj in call['model_audits'][name]['objects']]
    marked = [(p, o) for p, o in objects if o['marked_points'] > 0]
    shifts = [(p, o, abs(o['shelf_probability'] - o['shelf_prior_probability'])) for p, o in objects]
    most = max(shifts, key=lambda x: x[2]) if shifts else None
    return dict(
        model_rebuild_snapshots=len(models),
        object_snapshot_count=len(objects), marked_object_snapshot_count=len(marked),
        marked_object_snapshots_with_fit_shift=sum(abs(o['shelf_probability'] - o['shelf_prior_probability']) > 1e-12 for _, o in marked),
        objects_per_snapshot=span(len(m['objects']) for m in models),
        marked_objects_per_snapshot=span(sum(o['marked_points'] > 0 for o in m['objects']) for m in models),
        first_paid_action_with_marked_object=min((p for p, _ in marked), default=None),
        prior_range=span(o['shelf_prior_probability'] for _, o in objects),
        posterior_range=span(o['shelf_probability'] for _, o in objects),
        marked_prior_range=span(o['shelf_prior_probability'] for _, o in marked),
        marked_posterior_range=span(o['shelf_probability'] for _, o in marked),
        raw_fit_llr_range=span(o['raw_geometry_log_likelihood_ratio'] for _, o in objects),
        consumed_fit_llr_range=span(o['consumed_geometry_log_likelihood_ratio'] for _, o in objects),
        max_absolute_prior_to_fit_change=None if most is None else dict(
            paid=most[0], support_track=most[1]['support_track'], marked_points=most[1]['marked_points'],
            prior=most[1]['shelf_prior_probability'], posterior=most[1]['shelf_probability'],
            raw_llr=most[1]['raw_geometry_log_likelihood_ratio'], consumed_llr=most[1]['consumed_geometry_log_likelihood_ratio'],
            absolute_change=most[2]),
        history_keyframes_range=span(m['history_frames'] for m in models),
        raw_template_points_range=span(m['raw_template_points'] for m in models),
        points_after_history_range=span(m['points_after_history'] for m in models),
        output_points_range=span(m['output_points'] for m in models),
        original_global_point_cap_activations=sum(m['points_after_history'] > m['original_global_point_cap'] for m in models),
        removed_by_free_space_per_snapshot=span(m['removed_by_free_space'] for m in models),
        removed_as_measured_or_depth_tolerance_per_snapshot=span(m['removed_as_measured_or_depth_tolerance'] for m in models),
        snapshots_with_actual_history_removal=sum(m['removed_by_free_space'] + m['removed_as_measured_or_depth_tolerance'] > 0 for m in models),
        history_counts_balance=all(m['raw_template_points'] - m['points_after_history'] == m['removed_by_free_space'] + m['removed_as_measured_or_depth_tolerance'] for m in models),
        fit_enabled_values=sorted(set(m['fit_update'] for m in models)),
        history_enabled_values=sorted(set(m['history_rejection'] for m in models)),
        hidden_truth_input_always_false=all(m['hidden_truth_input'] is False for m in models),
        prior_rebuilt_each_snapshot=all(m['prior_rebuilt_from_current_measurements'] for m in models),
        actual_source_gain_function_calls=sum(c['source_gain_new_calls'][name] for c in calls),
        source_gain_calls_are_function_evaluations_not_observations=True)


def summarize_run(path, arm):
    result = load(path / 'result.json')
    manifest = load(path / 'artifact_manifest.json')
    bad_hashes = [k for k, v in manifest['files_sha256'].items() if not (path / k).is_file() or sha(path / k) != v]
    choices = load(path / 'choices.json')
    calls = load(path / 'template_audit.json')
    frames = load(path / 'template_frame_receipts.json')
    final = load(path / 'final_template_receipt.json')
    steps = load(path / 'steps.json')
    assertions = []
    plan_rows = []
    for choice in choices:
        out = choice['outputs']
        paid = 256 - choice['inputs']['remaining_budget']
        rows = out['score_audit']
        selected = out['selected']
        selected_id = None if selected is None else selected['candidate_id']
        wg, ws = winner(out, 'G'), winner(out, 'S')
        assertions.append(selected_id == winner(out, MODE[arm]))
        delta = [r['prior_potentials']['S'] - r['prior_potentials']['G'] for r in rows]
        valid_scores = all(
            abs(out['scores']['C'][i] - r['coverage_only_proxy']) < 1e-9
            and abs(out['scores']['N'][i] - r['common_total_proxy']) < 1e-9
            and all(abs(out['scores'][m][i] - r['common_total_proxy'] - r['prior_potentials'][m]) < 1e-9 for m in ('G', 'S'))
            for i, r in enumerate(rows))
        assertions.append(valid_scores)
        assertions.append(all(r['analytic_prior_used'] is False and r['template_gain_coefficient'] == 1 for r in rows))
        assertions.append(all(r['template_prior_active_in_selected_score'] == (arm not in ('C', 'J')) for r in rows))
        plan_rows.append(dict(paid=paid, candidates=len(rows), selected=selected_id,
                              G_winner=wg, S_winner=ws, G_S_winner_diff=wg != ws,
                              candidate_semantic_delta_range=span(delta),
                              nonzero_semantic_deltas=sum(abs(x) > 1e-12 for x in delta)))
    nonempty = [c for c in choices if c['outputs']['candidates']]
    # Exactly one score call is retained for every nonempty actual plan.
    alignment = len(nonempty) == len(calls) and all(
        ch['inputs']['semantic_version'] == ca['mapper_frames']
        and [r['candidate_id'] for r in ch['outputs']['candidates']] == [r['candidate_id'] for r in ca['routes']]
        for ch, ca in zip(nonempty, calls))
    assertions.append(alignment)
    assertions.append(final['registered_frames'] == final['mapper_frames'] == len(frames) == len(steps) + 1)
    assertions.append(final['input_sequence_sha256'] == hashlib.sha256(''.join(f['sha256'] for f in frames).encode()).hexdigest())
    assertions.append(all(c['prior_replaced_not_added'] and c['reused_original_gain_function']
                          and c['original_V3_candidate_method_used'] is False for c in calls))
    same_geometry = all(
        len(c['model_audits']['G']['objects']) == len(c['model_audits']['S']['objects'])
        and all(all(a[k] == b[k] for k in ('center', 'dims', 'observed_points', 'measured_group',
                                         'support_track', 'fit_errors_m2', 'raw_geometry_log_likelihood_ratio'))
                for a, b in zip(c['model_audits']['G']['objects'], c['model_audits']['S']['objects']))
        for c in calls)
    assertions.append(same_geometry)
    assertions.append(all(o['shelf_prior_probability'] == .5 for c in calls for o in c['model_audits']['G']['objects']))
    model_stats = {name: summarize_model(calls, name) for name in ('G', 'S')}
    ablation = dict(selected_mode=MODE[arm], template_term_in_actual_selection=arm not in ('C', 'J'))
    if arm == 'S_no_fit':
        ablation.update(fit_actually_off=all(m['model_audits']['S']['fit_update'] is False for m in calls)
                        and all(o['consumed_geometry_log_likelihood_ratio'] == 0 for c in calls for o in c['model_audits']['S']['objects']),
                        raw_fit_evidence_exists=any(abs(o['raw_geometry_log_likelihood_ratio']) > 1e-12 for c in calls for o in c['model_audits']['S']['objects']),
                        posterior_equals_category_prior=all(abs(o['shelf_probability'] - o['shelf_prior_probability']) < 1e-12 for c in calls for o in c['model_audits']['S']['objects']))
        assertions += [ablation['fit_actually_off'], ablation['posterior_equals_category_prior']]
    if arm == 'S_no_history':
        ablation.update(history_and_consumption_actually_off=all(
            c['model_audits']['S']['history_rejection'] is False
            and c['model_audits']['S']['removed_by_free_space'] == 0
            and c['model_audits']['S']['removed_as_measured_or_depth_tolerance'] == 0
            and c['model_audits']['S']['points_after_history'] == c['model_audits']['S']['raw_template_points'] for c in calls),
            historical_frames_still_available=any(c['model_audits']['S']['history_frames'] > 1 for c in calls),
            fit_remains_active=all(c['model_audits']['S']['fit_update'] for c in calls))
        assertions += [ablation['history_and_consumption_actually_off'], ablation['fit_remains_active']]
    return dict(
        status=result['status'], artifact_hash_check_passed=not bad_hashes, bad_artifact_hashes=bad_hashes,
        internal_checks_passed=all(assertions), actual_plan_score_call_alignment=alignment,
        G_S_same_measured_template_geometry_and_raw_fit=same_geometry,
        paid_actions=result['paid_actions'], trajectory_sha256=result['trajectory_sha256'],
        final_coverage_fraction=result['final_coverage_fraction'], first_coverage_action=result['first_coverage_action'],
        path_length_m=result['path_length_m'], collisions=result['collisions'], termination=result['termination'],
        final_pose_equals_initial_anchor=result['final_pose_equals_initial_anchor'],
        planning_decisions=len(choices), nonempty_actual_planning_states=len(nonempty),
        score_column_comparison=(
            'G=neutral prior with fit/history; S=category prior with fit disabled: this comparison changes two factors, not semantics alone'
            if arm == 'S_no_fit' else
            'G=neutral prior with history; S=category prior without history/consumption: this comparison changes two factors, not semantics alone'
            if arm == 'S_no_history' else
            'G versus S changes only category prior at this actual accumulated-evidence state'),
        same_state_G_S_winner_disagreements=sum(p['G_S_winner_diff'] for p in plan_rows),
        same_state_G_S_nonzero_score_states=sum(p['nonzero_semantic_deltas'] > 0 for p in plan_rows),
        plans=plan_rows, models=model_stats, ablation=ablation,
        input_artifacts_sha256={n: sha(path / n) for n in ('result.json', 'choices.json', 'template_audit.json', 'template_frame_receipts.json', 'artifact_manifest.json')}), choices, steps


def pair_summary(g, s):
    gs, ss = g[2], s[2]
    signature = lambda r: (r['action'], r['position'], r['heading'])
    first = next((i + 1 for i, (a, b) in enumerate(zip(gs, ss)) if signature(a) != signature(b)), None)
    shared = min(len(gs), len(ss)) if first is None else first - 1
    same_complete = first is None and len(gs) == len(ss)
    # Compare actual choices only before trajectory divergence; beyond that the
    # agents do not have the same accumulated evidence or candidate collection.
    gplans = {256 - c['inputs']['remaining_budget']: c for c in g[1]}
    splans = {256 - c['inputs']['remaining_budget']: c for c in s[1]}
    common = sorted(p for p in gplans.keys() & splans.keys() if p <= shared)
    plan_divergences = []
    for paid in common:
        a, b = gplans[paid]['outputs'], splans[paid]['outputs']
        selected = lambda o: None if o['selected'] is None else o['selected']['pose']
        if selected(a) != selected(b):
            plan_divergences.append(dict(paid=paid, G_pose=selected(a), S_pose=selected(b),
                same_candidates=digest(a['candidates']) == digest(b['candidates']),
                same_C_scores=a['scores']['C'] == b['scores']['C'], same_N_scores=a['scores']['N'] == b['scores']['N']))
    return dict(complete_action_pose_trajectories_identical=same_complete,
                shared_action_pose_prefix_steps=shared, first_actual_action_pose_divergence=first,
                first_length_divergence=(min(len(gs), len(ss)) + 1 if first is None and len(gs) != len(ss) else None),
                same_packets_and_owner_masks_on_shared_prefix=all(
                    a['packet_sha256'] == b['packet_sha256'] and a['owner_sha256'] == b['owner_sha256']
                    for a, b in zip(gs[:shared], ss[:shared])),
                same_candidates_at_shared_history_plans=all(
                    digest(gplans[p]['outputs']['candidates']) == digest(splans[p]['outputs']['candidates']) for p in common),
                same_G_S_counterfactual_scores_at_shared_history_plans=all(
                    all(gplans[p]['outputs']['scores'][m] == splans[p]['outputs']['scores'][m] for m in ('C', 'N', 'G', 'S')) for p in common),
                actual_G_S_decision_divergences_on_shared_history=plan_divergences,
                matched_shared_history_planning_states=len(common),
                final_coverage_delta_S_minus_G=s[0]['final_coverage_fraction'] - g[0]['final_coverage_fraction'],
                path_length_delta_S_minus_G=s[0]['path_length_m'] - g[0]['path_length_m'])


def trajectory_intervention(baseline, intervention):
    a, b = baseline[2], intervention[2]
    signature = lambda r: (r['action'], r['position'], r['heading'])
    first = next((i + 1 for i, (x, y) in enumerate(zip(a, b)) if signature(x) != signature(y)), None)
    return dict(complete_trajectory_identical=first is None and len(a) == len(b),
                first_action_pose_difference=first,
                first_length_difference=min(len(a), len(b)) + 1 if first is None and len(a) != len(b) else None,
                baseline_first_full_coverage=baseline[0]['first_coverage_action']['1.0'],
                intervention_first_full_coverage=intervention[0]['first_coverage_action']['1.0'],
                baseline_path_length_m=baseline[0]['path_length_m'], intervention_path_length_m=intervention[0]['path_length_m'])


def main():
    output = dict(schema='dual_template_actual_episode_mechanism_summary_v1',
        protocol_sha256=sha(ROOT / 'protocol.json'), analysis_script_sha256=sha(Path(__file__)),
        evidence_unit='12 deterministic strategy episodes in two position permutations of one floorplan family; not 12 independent environments',
        interpretation='Same-state G/S winners are counterfactual diagnostics at actual states, not extra trajectories or independent samples. Model ranges concern cumulative-evidence rebuild snapshots, not unique removed surfaces.',
        mechanism_boundaries=[
            'Historical depth rejects candidate reward samples in measured free space or near measured surfaces; it does not enter the geometric-fit likelihood or update the box/shelf posterior directly.',
            'Original complete box/shelf hypothesis meshes remain in self-occlusion ray tests after sample rejection; history does not remove a contradicted backboard from that ray mesh or correct its topology.',
            'S_no_history jointly disables free-space rejection and already-measured surface consumption; it is not a pure memory-only ablation.',
            'S_no_history can also activate the unchanged 3500-point global weighted thinning cap because more hypothesis points remain; cap activations are reported, so trajectory differences cannot isolate history alone from this induced approximation.',
            'The shape probability is the source mean-distance Gibbs pseudo-posterior, not calibrated uncertainty.',
            'C and J still compute template counterfactuals for diagnosis but do not use their values for target selection; raw wall time is not a clean algorithm-cost comparison.',
            'Current visible ideal semantic masks and measured instance support are shared. No natural YOLOE recognition or real-robot efficacy is established.',
            'The source full template estimator and gain are integrated with v10 candidates; original V3 candidate proposals are not used.'
        ],
        completed_episodes=0, pending=[], failures=[], runs={}, G_S_pairs={}, ablation_trajectories={})
    data = {}
    for layout in LAYOUTS:
        freeze = load(ROOT / 'layouts' / layout / 'prepared/freeze.json')
        output.setdefault('frozen_code_checks', {})[layout] = all(sha(Path(p)) == v for p, v in freeze['code_sha256'].items())
        for arm in ARMS:
            paths = sorted((ROOT / 'layouts' / layout / 'runs' / arm).glob('attempt_*'))
            for path in paths:
                if (path / 'failure.json').exists():
                    output['failures'].append(dict(path=str(path), **load(path / 'failure.json')))
            completed = [p for p in paths if (p / 'result.json').exists() and (p / 'artifact_manifest.json').exists()]
            if not completed:
                output['pending'].append(f'{layout}/{arm}')
                continue
            if len(completed) != 1:
                raise RuntimeError(f'multiple completed attempts must be explicitly reviewed: {layout}/{arm}')
            path = completed[0]
            triple = summarize_run(path, arm)
            data[layout, arm] = triple
            output['runs'][f'{layout}/{arm}'] = dict(path=str(path), **triple[0])
            output['completed_episodes'] += 1
        if (layout, 'G') in data and (layout, 'S') in data:
            output['G_S_pairs'][layout] = pair_summary(data[layout, 'G'], data[layout, 'S'])
        for base, ablation in [('G', 'J'), ('S', 'S_no_fit'), ('S', 'S_no_history')]:
            if (layout, base) in data and (layout, ablation) in data:
                output['ablation_trajectories'][f'{layout}/{base}_vs_{ablation}'] = trajectory_intervention(data[layout, base], data[layout, ablation])
    output['all_planned_episodes_complete'] = output['completed_episodes'] == 12 and not output['pending']
    output['all_completed_internal_checks_passed'] = all(r['internal_checks_passed'] and r['artifact_hash_check_passed'] for r in output['runs'].values())
    target = Path(__file__).with_name('mechanism_summary.json')
    target.write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(output=str(target), sha256=sha(target), completed=output['completed_episodes'],
                         pending=output['pending'], passed=output['all_completed_internal_checks_passed'],
                         pairs=output['G_S_pairs']), ensure_ascii=False))


if __name__ == '__main__':
    main()
