#!/usr/bin/env python3
"""Parse an explicit list of sealed scene episodes; never execute a controller.

Counts describe mechanism activation, not measured gains. No World, mapper,
sensor, numerical evaluator, or policy implementation is imported.
"""
import argparse
import ast
from collections import Counter, defaultdict
import json
from pathlib import Path

from summarize_semantic_integration import COMPLETE, _records, digest, parse_steps, read, sha

ROOT = Path(__file__).resolve().parents[1]


def verify_episode(phase, run_id, ledger):
    if Path(run_id).name != run_id or run_id in ('.', '..'):
        raise ValueError('plain explicit run ID required')
    episode = phase / 'episodes' / run_id
    manifest = read(episode / 'artifact_manifest.json')
    if manifest.get('schema') != 'semantic.scene_experiment.artifacts.v1':
        raise ValueError('sealed scene experiment required')
    actual = {str(p.relative_to(episode)) for p in episode.rglob('*') if p.is_file()}
    if actual != set(manifest['files']) | {'artifact_manifest.json'}:
        raise ValueError('sealed inventory differs: ' + run_id)
    for name, record in manifest['files'].items():
        path = episode / name
        if (Path(name).is_absolute() or '..' in Path(name).parts or path.is_symlink()
                or not path.resolve().is_relative_to(episode.resolve())
                or path.stat().st_size != record['bytes'] or sha(path) != record['sha256']):
            raise ValueError('sealed artifact differs: ' + name)
    protocol, started, result = (read(episode / name) for name in
                                  ('protocol.json', 'started.json', 'result.json'))
    entries = [row for row in ledger['entries'] if row['run_id'] == run_id]
    if (len(entries) != 1 or protocol['status'] != 'frozen'
            or protocol['phase_id'] != phase.name or ledger['phase_id'] != phase.name
            or sha(episode / 'protocol.json') != manifest['protocol_sha256']
            or ledger['protocol_sha256'] != manifest['protocol_sha256']
            or ledger['slots'] != protocol['slots'] or started['run_id'] != run_id
            or started['slot'] != protocol['slots'][run_id]
            or entries[0]['metadata']['slot'] != started['slot']
            or entries[0]['metadata']['source_sha256'] != manifest['source_sha256']
            or started['source_sha256'] != manifest['source_sha256']
            or entries[0]['result_sha256'] != sha(episode / 'result.json')
            or entries[0]['status'] != result['status'] or not entries[0]['world_created']
            or result['status'] not in COMPLETE or result['error'] is not None
            or result['finalization_errors']):
        raise ValueError('episode/finite ledger binding or completion differs: ' + run_id)
    for name, pin in manifest['source_sha256'].items():
        if sha(episode / 'source' / name) != pin:
            raise ValueError('archived execution source differs: ' + name)
    count = result['acquired_and_saved_packets']
    if (count != result['executed_paid_actions'] + 1 or count != result['mapper_frames']
            or count != result['received_sensor_packets'] or len(result['actions']) != count - 1):
        raise ValueError('paid frame/action counters differ')
    return episode, manifest, protocol, started, result


def frontend_defaults(episode):
    """Read constructor defaults as syntax, without importing executable code."""
    tree = ast.parse((episode / 'source/nso/observed_instances_v41.py').read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef)
               and node.name == 'ObservedInstancesV41')
    init = next(node for node in cls.body if isinstance(node, ast.FunctionDef)
                and node.name == '__init__')
    values = {arg.arg: ast.literal_eval(default) for arg, default in
              zip(init.args.kwonlyargs, init.args.kw_defaults) if default is not None}
    keys = ('maximum_support_points', 'maximum_evidence_frames',
            'minimum_novel_points', 'minimum_novel_fraction', 'minimum_semantic_views')
    # Existing generic parser's class qualification is explicitly two supports.
    if values['minimum_semantic_views'] != 2:
        raise ValueError('frontend qualification changed; parser needs explicit adaptation')
    return {key: values[key] for key in keys}


def feedback_observation(row, association, residual, defaults):
    """Disaggregate refusal observations without relabelling calls as updates."""
    if type(row.get('applied')) is not bool or type(association.get('geometry_feedback_eligible')) is not bool:
        raise ValueError('explicit feedback applied/eligible booleans required')
    if row['applied'] and not association['geometry_feedback_eligible']:
        raise ValueError('applied feedback contradicts association eligibility')
    if not residual or not residual.get('accepted'):
        raise ValueError('feedback call lacks accepted current paid residual')
    state = row['instance']; size = len(state['support_points_world_m'])
    point_count = len(association['points_world_m'])
    lower = association['novel_support_voxels'] / point_count if point_count else 0.
    full = size >= defaults['maximum_support_points']
    no_insert = association['no_new_support']
    facts = dict(full_support_buffer=full, no_support_inserted=no_insert,
        duplicate_measurement=association['duplicate_measurement'],
        evidence_counter_at_or_above_cap=state['novel_support_frames'] >= defaults['maximum_evidence_frames'],
        minimum_novel_voxels_satisfied=association['novel_support_voxels'] >= defaults['minimum_novel_points'],
        minimum_novel_fraction_proven_by_lower_bound=lower >= defaults['minimum_novel_fraction'])
    facts['support_capacity_refusal_with_other_gates_satisfied'] = bool(
        not row['applied'] and full and no_insert and not facts['duplicate_measurement']
        and not facts['evidence_counter_at_or_above_cap']
        and facts['minimum_novel_voxels_satisfied']
        and facts['minimum_novel_fraction_proven_by_lower_bound'])
    return facts


def summarize_episode(phase, run_id, ledger):
    episode, manifest, protocol, started, result = verify_episode(phase, run_id, ledger)
    defaults = frontend_defaults(episode)
    first_plane, first_full, first_observed, observed_classes = {}, {}, {}, defaultdict(set)
    feedback = Counter(); refused_reasons = Counter(); all_reasons = Counter(); refusal_facts = Counter()
    feedback_by_instance = defaultdict(Counter); first_feedback = {}; first_applied = {}
    diagnostic_decisions = []; initializations = []; actual_initialization_actions = 0
    seen_feedback = set(); final_initialization_spent = 0

    def records():
        nonlocal actual_initialization_actions, final_initialization_spent
        for index, step, packet in _records(episode, result, manifest, protocol):
            evidence = step['controller_evidence']; decision = step['decision']
            accepted = {r['instance_id']: r for r in evidence['association']['accepted']}
            residuals = {r['instance_id']: r for r in evidence['observed_residual']['results']}
            for row in evidence['association']['instances']:
                key = row['instance_id']; first_observed.setdefault(key, index)
                if row['observed_class'] is not None: observed_classes[key].add(row['observed_class'])
                if len(row['support_points_world_m']) >= defaults['maximum_support_points']:
                    first_full.setdefault(key, index)
            for row in evidence['view_evidence']['results']:
                fit = row.get('current_plane_fit')
                if fit and fit['accepted']: first_plane.setdefault(row['instance_id'], index)
            for row in evidence['geometry_feedback']:
                key = row['instance_id']; pair = (index, key)
                if pair in seen_feedback: raise ValueError('duplicate feedback call for one paid frame/instance')
                seen_feedback.add(pair)
                facts = feedback_observation(row, accepted[key], residuals.get(key), defaults)
                applied = row['applied']; informative = bool(residuals[key].get('informative'))
                first_feedback.setdefault(key, index)
                if applied: first_applied.setdefault(key, index)
                for counter in (feedback, feedback_by_instance[key]):
                    counter['attempted'] += 1; counter['applied'] += applied
                    counter['refused'] += not applied; counter['informative_attempted'] += informative
                    counter['informative_applied'] += informative and applied
                    counter['association_eligible_at_call'] += accepted[key]['geometry_feedback_eligible']
                all_reasons[row['reason']] += 1
                if not applied:
                    refused_reasons[row['reason']] += 1
                    refusal_facts.update({key: int(value) for key, value in facts.items()})
            final_initialization_spent = evidence.get('initialization_actions_spent', 0)
            if decision.get('initialization_primitive_pending') and index < result['executed_paid_actions']:
                actual_initialization_actions += 1
            selection = decision.get('global_selection') or {}; choice = selection.get('selected') or {}
            if choice.get('kind') == 'measurement_initialization':
                initializations.append(dict(paid_step=index, instance_id=choice['instance_id'],
                    attempt_number=choice['attempt_number'], target=choice['target'],
                    planned_outbound_plus_observe=choice['outbound_cost'] + 1,
                    planned_total_with_return=choice['total_cost']))
            options = selection.get('diagnostic_options', [])
            if options:
                diagnostic_decisions.append(dict(paid_step=index, candidates=len(options),
                    positive_predicted_evi=sum(row.get('evi', 0.) > 1e-12 for row in options),
                    maximum_predicted_evi=max(row.get('evi', 0.) for row in options),
                    best_diagnostic_score=max(row['score'] for row in options),
                    best_direct_score=max((row['score'] for row in selection['direct_options']), default=None),
                    selected_kind=choice.get('kind'), selected_target=choice.get('target')))
            yield index, step, packet

    parsed = parse_steps(records())
    if (feedback['attempted'] != feedback['applied'] + feedback['refused']
            or feedback['refused'] != sum(refused_reasons.values())
            or feedback['applied'] != parsed['counts'].get('frontend_geometry_updates_applied', 0)
            or final_initialization_spent != actual_initialization_actions):
        raise ValueError('feedback or paid initialization accounting inconsistent')
    instances = []
    for row in parsed['observed_instances']:
        key = row['observed_instance_id']; belief = row['last_belief'] or {}
        instances.append(dict(instance_id=key, observed_classes_for_attribution=sorted(observed_classes[key]),
            first_observed_paid_step=first_observed.get(key), first_accepted_plane_paid_step=first_plane.get(key),
            first_full_support_buffer_paid_step=first_full.get(key),
            full_buffer_precedes_first_plane=bool(key in first_full and key in first_plane and first_full[key] < first_plane[key]),
            first_feedback_attempt_paid_step=first_feedback.get(key), first_applied_feedback_paid_step=first_applied.get(key),
            feedback=dict(feedback_by_instance[key]), rho_min=row['rho_min'], rho_max=row['rho_max'],
            final_peer_instance_ids=belief.get('peer_instance_ids', []),
            final_structure_probabilities=belief.get('structure_probabilities'),
            final_geometry_log_evidence=belief.get('geometry_log_evidence')))
    action_sequence = [row['controller_action'] for row in result['actions']]
    source_differences = [name for name, pin in manifest['source_sha256'].items()
                          if not (ROOT / name).is_file() or sha(ROOT / name) != pin]
    return dict(run_id=run_id, slot=started['slot'], episode_manifest_sha256=sha(episode / 'artifact_manifest.json'),
        status=result['status'], actions=result['executed_paid_actions'], saved_packets=result['acquired_and_saved_packets'],
        returned_xy_and_yaw=result['sensor_status']['returned_xy_and_yaw'], collisions=result['collisions'],
        elapsed_s=result['elapsed_s'], action_sequence_sha256=digest(action_sequence),
        full_action_records_sha256=digest(result['actions']), action_counts=dict(Counter(action_sequence)),
        mesh_npz_sha256=manifest['files']['prediction/mesh.npz']['sha256'],
        sealed_inventory_and_ledger_verified=True, current_sources_match=not source_differences,
        current_source_differences=source_differences, frontend_frozen_defaults=defaults,
        initialization=dict(spent=final_initialization_spent, selections=initializations),
        instances=instances, counts=parsed['counts'], first_events=parsed['first_events'],
        plane_fit_reasons=parsed['plane_fit_reasons'], residual_reasons=parsed['residual_reasons'],
        feedback=dict(counts=dict(feedback), all_call_reason_counts=dict(all_reasons),
            refusal_reason_counts=dict(refused_reasons), refusal_observations=dict(refusal_facts),
            refusal_observations_overlap_and_are_not_additive=True,
            existing_contract='V41_OBSERVED_INSTANCE_CONTRACT_20260920.md:26 explicitly requires actual support insertion before feedback eligibility.'),
        same_class_opportunity=parsed['same_class_opportunity'], diagnostic_decisions=diagnostic_decisions,
        diagnostic_evi_max=parsed['diagnostic_evi_max'], selected_diagnostic_evi_max=parsed['selected_diagnostic_evi_max'],
        mechanism_occurrence_is_not_measured_benefit=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase-root', type=Path, required=True)
    parser.add_argument('--run-id', nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); phase = args.phase_root.resolve(); output = args.output.resolve()
    if (len(set(args.run_id)) != len(args.run_id) or output.exists()
            or not output.is_relative_to(phase / 'diagnostics')):
        parser.error('distinct explicit run IDs and new output under phase/diagnostics required')
    ledger = read(phase / 'start_ledger.json')
    rows = [summarize_episode(phase, run_id, ledger) for run_id in args.run_id]
    value = dict(schema='semantic.activation_summary.v1', phase_id=ledger['phase_id'],
        protocol_sha256=ledger['protocol_sha256'], explicit_run_ids=args.run_id,
        parser_source_sha256={Path(__file__).name: sha(Path(__file__)),
            'summarize_semantic_integration.py': sha(Path(__file__).with_name('summarize_semantic_integration.py'))},
        episodes=rows, runtime=dict(new_worlds=0, new_sensor_packets=0, new_tsdf_integrations=0,
            policy_replay_performed=False, numerical_evaluation_performed=False),
        limitations=['Explicit sealed episodes only; no implicit selection of future or partial episodes.',
            'Counters describe activation, not surface accuracy or semantic causal benefit.',
            'Posterior/rho changes and predicted EVI are uncalibrated model quantities.',
            'Refusal fact counts overlap; the broad logged refusal reason is preserved separately.',
            'Action-sequence hash is canonical JSON of primitive action strings; full action-record hash also binds observation IDs.',
            'No terminal quality is calculated or inferred from a mesh hash.'])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps(dict(output=str(output), sha256=sha(output), episodes=len(rows)), sort_keys=True))


if __name__ == '__main__':
    main()
