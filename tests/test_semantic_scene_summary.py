"""Pure parser/aggregation fixtures; no World, trajectories or evaluation calls."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import summarize_semantic_scene_results as m
from scripts.summarize_semantic_integration import self_test as integration_parser_test


def fixture_rows():
    rows = []
    for index in range(6):
        for condition in ('nominal_relationship', 'structure_relationship_shift'):
            for method in ('G', 'bayes_semantic', 'shared_semantic'):
                identity = f'{index}_{condition}_{method}'
                rows.append(dict(run_id=identity, parent_id=f'SEM_P{index:02d}', condition=condition,
                    method=method, budget=120, noise_seed=230923, tie_rule='lexicographic', tie_seed=0,
                    metrics=dict(C_nav=1., Q=.2, J_nav=.2), returned_xy_and_yaw=True))
    protocol = dict(development_gates=dict(development_parent_count=6,
        mean_coverage_difference_minimum=-.01, positive_parent_count_minimum=4,
        maximum_one_parent_share_of_positive_J_differences=.5),
        analysis_blocks={'core': {'run_ids': [r['run_id'] for r in rows]}})
    return protocol, rows


def compare(protocol, rows, *, consistency=False):
    return m.core_comparison(protocol, rows, name='synthetic', condition='nominal_relationship',
        baseline='G', relative_minimum=.05, require_parent_consistency=consistency)


def review_fixture():
    sources = {'nso/complete_surface_evaluation.py': 'e'*64}
    pins = {'mapper.json': 'a'*64, 'occupancy.npz': 'b'*64, 'mesh.npz': 'c'*64}
    episode = dict(manifest=dict(source_sha256=sources, input_sha256={'reference': 'd'*64},
        files={'prediction/'+name: {'sha256': pin} for name, pin in pins.items()}),
        protocol=dict(phase_id='synthetic', asset_manifest_sha256='f'*64),
        result=dict(acquired_and_saved_packets=3), started=dict(run_id='synthetic'),
        manifest_sha256='1'*64, current_sources_match=True,
        reference=dict(manifest_sha256='2'*64), slot=dict(asset_id='synthetic_asset'))
    review = dict(schema='semantic.scene_experiment.review.v1', status='experiment_reviewed',
        run_id='synthetic', phase_id='synthetic', episode_manifest_sha256='1'*64,
        source_sha256=sources, input_sha256={'reference': 'd'*64}, current_sources_match=True,
        replay=dict(status='verified', prediction_verified=True, frames_verified=3),
        runtime=dict(before={'world': 0}, after={'world': 0}, no_new_world_or_sensor_action=True),
        evaluation=dict(schema='semantic_scene.complete_saved_prediction_evaluation.v1',
            asset_id='synthetic_asset', asset_manifest_sha256='f'*64, episode_manifest_sha256='1'*64,
            reference_manifest_sha256='2'*64, input_prediction_sha256=pins,
            evaluator_source_sha256='e'*64, all_task_instances_in_macro_denominator=True,
            prediction_roi_cropped=False, prediction_reintegrated=False,
            new_worlds=0, new_sensor_packets=0, new_tsdf_integrations=0,
            metrics=dict(C_nav=.8, Q=.5, J_nav=.4)))
    return review, episode


def method_correction_fixture():
    """In-memory declarations only: no real episode, ledger or protocol writes."""
    previous = {key: {'synthetic': key} for key in m.SCIENTIFIC_FIELDS}
    previous.update(phase_id='synthetic_old', controller={'synthetic': True},
                    slots={f'slot_{i:02d}': {'method': f'method_{i}'} for i in range(96)})
    protocol = deepcopy(previous)
    protocol.update(phase_id='synthetic_new', status='frozen',
        controller=dict(previous['controller'], measurement_acquisition=True),
        execution_source_sha256={'synthetic_execution.py': 'f'*64})
    old_ids = list(previous['slots'])[:5]
    ledger = dict(slots=deepcopy(previous['slots']), protocol_sha256='a'*64,
                  entries=[dict(run_id=k) for k in old_ids])
    summary = dict(phase_id=previous['phase_id'], protocol_sha256='a'*64,
        ledger_snapshot_sha256='b'*64, reviewed_metric_endpoints=5,
        states=dict(complete_episode=5, failed_attempt=0, integrity_error=0,
                    not_attempted=91, reserved_or_running=0),
        episodes=[dict(run_id=k, complete_episode=True, metrics={'J_nav': .4}) for k in old_ids])
    declaration = dict(original_phase_id=previous['phase_id'], status='frozen',
        execution_authorized=True, method_candidate_revision_number=1, maximum_method_candidate_revisions=1,
        total_primary_world_ceiling=101, total_attempt_ceiling_including_constructor_failure=102,
        corrected_matrix_slots=96, retained_original_worlds=5, retired_original_unstarted_slots=91,
        separate_constructor_failure_attempts=1)
    protocol['method_correction'] = declaration
    record = dict(schema='semantic.method_correction_draft.v1', status='draft', frozen=True,
        execution_authorized=True, original_phase_id=previous['phase_id'], proposed_phase_id=protocol['phase_id'],
        original_protocol_sha256='a'*64, original_ledger_sha256='b'*64,
        retained_original_episodes=[dict(run_id=k) for k in old_ids],
        paired_before_after_run_ids=old_ids, retired_original_unstarted_run_ids=list(previous['slots'])[5:],
        scientific_matrix_fields=list(m.SCIENTIFIC_FIELDS),
        scientific_matrix_sha256=m.digest({k: previous[k] for k in m.SCIENTIFIC_FIELDS}),
        method_candidate_revision_number=1, maximum_method_candidate_revisions=1,
        world_budget_accounting=dict(original_complete_worlds=5, corrected_matrix_maximum_new_worlds=96,
            total_primary_world_ceiling=101, net_additional_world_allowance_over_original_matrix=5),
        total_attempt_ceiling_including_constructor_failure=102,
        constructor_failure_separate=dict(count=1, worlds_created=0),
        second_method_revision_or_second_tuning_round_allowed=False, automatic_retry_allowed=False,
        execution_source_sha256=deepcopy(protocol['execution_source_sha256']))
    return protocol, record, previous, ledger, summary


def acquisition_records():
    selected = dict(kind='measurement_initialization', instance_id='observed_only',
        target=dict(node='n', heading=30), score=None, expected_gain=None, evi=None,
        semantic_information_used=False, counted_as_surface_gain=False)
    receipt = dict(execution=dict(pose_xyyaw_rad=[0., 0., 0.]))
    first = dict(controller_evidence=dict(initialization_actions_spent=0, first_actual_reliable_planes=[]),
        decision=dict(action='left', initialization_actions_spent=0, initialization_action_cap=24,
            initialization_cancelled=None, global_selection=dict(candidate_pool=dict(selected=1),
                direct_options=[], diagnostic_options=[], forecasts=[], selected=selected)))
    second = dict(controller_evidence=dict(initialization_actions_spent=1,
                                          first_actual_reliable_planes=['observed_only']),
        decision=dict(action='forward', initialization_actions_spent=1, initialization_action_cap=24,
            initialization_cancelled=dict(instance_id='observed_only', target=selected['target'],
                reason='plane_acquired', actions_refunded=False, attempts_refunded=False)))
    return [(0, first, deepcopy(receipt)), (1, second, deepcopy(receipt))]


class SceneSummaryCorrectionTests(unittest.TestCase):
    def test_one_revision_retains_old_five_and_new_96_separately(self):
        values = method_correction_fixture()
        result = m.validate_method_layout(*values)
        self.assertEqual(result['total_primary_world_ceiling'], 101)
        self.assertEqual(result['total_attempt_ceiling'], 102)
        self.assertEqual(result['retained_primary_worlds'], 5)
        self.assertEqual(result['retired_unstarted_slots'], 91)
        self.assertFalse(result['old_results_included_in_current_core_comparisons'])
        protocol, record, *_ = values
        # A historical draft status is retained; explicit freeze flags govern.
        self.assertEqual(record['status'], 'draft')
        self.assertEqual(protocol['method_correction']['status'], 'frozen')

    def test_rejects_changed_science_retirement_counts_and_second_revision(self):
        changes = [
            lambda p,r,o,l,s: p['slots'].pop('slot_95'),
            lambda p,r,o,l,s: p['evaluation'].update(synthetic='changed metric'),
            lambda p,r,o,l,s: r['retired_original_unstarted_run_ids'].__setitem__(0, 'slot_00'),
            lambda p,r,o,l,s: r.update(method_candidate_revision_number=2),
            lambda p,r,o,l,s: r.update(automatic_retry_allowed=True),
            lambda p,r,o,l,s: r.update(total_attempt_ceiling_including_constructor_failure=103),
            lambda p,r,o,l,s: p['method_correction'].update(retained_original_worlds=0),
            lambda p,r,o,l,s: s['episodes'].pop(),
            lambda p,r,o,l,s: r['world_budget_accounting'].update(corrected_matrix_maximum_new_worlds=91),
        ]
        for change in changes:
            values = method_correction_fixture(); change(*values)
            with self.subTest(change=change), self.assertRaises(ValueError):
                m.validate_method_layout(*values)

    def test_frozen_authorization_and_source_declaration_cannot_disagree(self):
        for owner, key, value in [('record', 'execution_authorized', False),
                                 ('record', 'frozen', False),
                                 ('record', 'execution_source_sha256', {'wrong.py': '0'*64}),
                                 ('declaration', 'status', 'draft')]:
            values = method_correction_fixture()
            target = values[1] if owner == 'record' else values[0]['method_correction']
            target[key] = value
            with self.subTest(owner=owner, key=key), self.assertRaises(ValueError):
                m.validate_method_layout(*values)

    def test_draft_is_validated_without_claiming_execution_authorization(self):
        values = method_correction_fixture(); protocol, record, *_ = values
        protocol['status'] = 'draft'
        protocol['method_correction'].update(status='draft', execution_authorized=False)
        record.update(status='draft', frozen=False, execution_authorized=False)
        result = m.validate_method_layout(*values)
        self.assertEqual(result['current_matrix_world_ceiling'], 96)

    def test_initialization_none_scores_empty_forecasts_and_real_cancellation(self):
        result = m.parse_scene_steps(acquisition_records())
        detail = result['measurement_acquisition']
        self.assertEqual(detail['actual_initialization_actions_spent'], 1)
        self.assertEqual(detail['initialization_action_cap'], 24)
        self.assertEqual(detail['first_actual_reliable_plane_events'],
                         [dict(paid_step=1, observed_instance_id='observed_only')])
        self.assertEqual(detail['initialization_cancellation_reasons'], {'plane_acquired': 1})
        self.assertEqual(result['counts']['selected_measurement_initialization'], 1)
        self.assertEqual(result['counts']['diagnostic_options_scored'], 0)
        self.assertEqual(result['counts']['positive_direct_options'], 0)
        self.assertIsNone(result['diagnostic_evi_max'])
        self.assertIsNone(result['selected_diagnostic_evi_max'])
        self.assertIsNone(result['selected_global_options'][0]['score'])
        self.assertIsNone(detail['selected_initialization_events'][0]['expected_gain'])
        self.assertFalse(detail['initialization_counted_as_surface_gain'])
        self.assertFalse(detail['initialization_counted_as_diagnostic_evi'])

    def test_initialization_rejects_fake_gain_and_inconsistent_actual_spending(self):
        records = acquisition_records()
        records[0][1]['decision']['global_selection']['selected']['score'] = 0.
        with self.assertRaisesRegex(ValueError, 'surface utility or EVI'):
            m.parse_scene_steps(records)
        records = acquisition_records()
        records[1][1]['decision']['initialization_actions_spent'] = 0
        with self.assertRaisesRegex(ValueError, 'spending is inconsistent'):
            m.parse_scene_steps(records)
        records = acquisition_records()
        records[0][1]['decision']['initialization_action_cap'] = 0
        records[1][1]['decision']['initialization_action_cap'] = 0
        with self.assertRaisesRegex(ValueError, 'exceeds its allowance'):
            m.parse_scene_steps(records)

    def test_absent_initialization_receipts_are_unknown_not_zero(self):
        result = m.parse_scene_steps([(0, dict(controller_evidence={}, decision={}),
                                      dict(execution=dict(pose_xyyaw_rad=[0.,0.,0.])) )])
        self.assertFalse(result['measurement_acquisition']['receipts_present'])
        self.assertIsNone(result['measurement_acquisition']['actual_initialization_actions_spent'])

    def test_informative_residual_and_predicted_evi_are_not_applied_feedback(self):
        records = acquisition_records()
        evidence = records[1][1]['controller_evidence']
        evidence['geometry_feedback'] = [
            dict(instance_id='observed_only', applied=False, reason='repeated_support_or_feedback_or_history_cap'),
            dict(instance_id='another_observed', applied=False, reason='repeated_support_or_feedback_or_history_cap')]
        evidence['observed_residual'] = dict(results=[dict(accepted=True, informative=True, reason='synthetic')])
        choice = dict(kind='diagnose_then_observe', evi=.25, target=dict(node='n', heading=30))
        records[1][1]['decision']['global_selection'] = dict(diagnostic_options=[choice], forecasts=[])
        result = m.parse_scene_steps(records)
        feedback = result['feedback_application']
        self.assertEqual(feedback['feedback_attempts'], 2)
        self.assertEqual(feedback['feedback_applied'], 0)
        self.assertEqual(feedback['feedback_refused'], 2)
        self.assertEqual(feedback['informative_residuals'], 1)
        self.assertEqual(feedback['refusal_reasons'], {'repeated_support_or_feedback_or_history_cap': 2})
        self.assertEqual(result['diagnostic_evi_max'], .25)
        self.assertEqual(result['counts']['frontend_geometry_updates_applied'], 0)
        evidence['geometry_feedback'][0].update(applied=True, reason='synthetic_applied')
        feedback = m.parse_scene_steps(records)['feedback_application']
        self.assertEqual(feedback['feedback_attempts'], 2)
        self.assertEqual(feedback['feedback_applied'], 1)
        self.assertEqual(feedback['feedback_refused'], 1)

    def test_retired_reference_checks_pinned_bytes_and_archived_generator_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record = dict(asset_id='synthetic', asset_manifest_sha256='a'*64,
                evaluation_parameters={'synthetic': True},
                source_sha256={'synthetic_old_generator.py': 'b'*64})
            (root/'reference.json').write_text(json.dumps(record))
            for name in ('surface.npz', 'coverage_domain.npz', 'candidate_views.json'):
                (root/name).write_bytes(b'synthetic pinned bytes, never interpreted')
            files = {p.name: dict(sha256=m.sha(p), bytes=p.stat().st_size) for p in root.iterdir()}
            (root/'manifest.json').write_text(json.dumps(dict(files=files)))
            episode = dict(reference=dict(root='synthetic/relative', manifest_sha256=m.sha(root/'manifest.json')),
                slot=dict(asset_id='synthetic'), protocol=dict(asset_manifest_sha256='a'*64,
                    evaluation={'synthetic': True}),
                manifest=dict(source_sha256={'synthetic_old_generator.py': 'b'*64}))
            with patch.object(m, 'contained', return_value=root):
                self.assertEqual(m.historical_reference_record(episode), record)
                altered = deepcopy(episode)
                altered['manifest']['source_sha256']['synthetic_old_generator.py'] = 'c'*64
                with self.assertRaisesRegex(ValueError, 'archived executed sources'):
                    m.historical_reference_record(altered)
                (root/'surface.npz').write_bytes(b'tampered')
                with self.assertRaisesRegex(ValueError, 'reference bytes changed'):
                    m.historical_reference_record(episode)


class SceneSummaryTests(unittest.TestCase):
    def test_reuses_existing_paid_receipt_parser(self):
        result = integration_parser_test()
        self.assertEqual(result['status'], 'parser_self_test_passed')
        self.assertEqual(result['new_worlds'], 0)

    def test_unattempted_running_and_failed_are_distinct_without_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            slot = dict(asset_id='SEM_P00__nominal_relationship', method='G', budget=120,
                        noise_seed=230923, tie_rule='lexicographic', tie_seed=0)
            protocol = dict(slots={'synthetic': slot}, output_relative_path='episodes',
                            ledger_relative_path='start_ledger.json')
            entry = dict(run_id='synthetic', status='reserved_before_factory',
                reservation_time_unix_s=1., metadata=dict(slot=slot, run_id='synthetic',
                    protocol_sha256='f'*64, output=str(root/'episodes/synthetic')))
            with patch.object(m, 'contained', side_effect=lambda p: root/p):
                missing = m.summarize_slot(protocol, 'f'*64, 'synthetic', None, root/'reviews')
                running = m.summarize_slot(protocol, 'f'*64, 'synthetic', entry, root/'reviews')
                terminal = root/'episodes/synthetic/attempt_failure.json'
                terminal.parent.mkdir(parents=True)
                terminal.write_text(json.dumps(dict(status='experiment_attempt_failed',
                    world_created=False, message='synthetic constructor failure')))
                entry.update(status='experiment_attempt_failed', world_created=False,
                             result_sha256=m.sha(terminal))
                failed = m.summarize_slot(protocol, 'f'*64, 'synthetic', entry, root/'reviews')
            self.assertEqual([r['state'] for r in (missing, running, failed)],
                             ['not_attempted', 'reserved_or_running', 'failed_attempt'])
            self.assertTrue(running['world_creation_unknown_until_terminal'])
            self.assertTrue(failed['terminal_attempt'])
            self.assertFalse(failed['world_created'])
            for row in (missing, running, failed):
                self.assertTrue(all(row[k] is None for k in ('metrics', 'C_nav', 'Q', 'J_nav')))

    def test_missing_parent_cannot_pass_and_is_not_zero(self):
        protocol, rows = fixture_rows()
        for row in rows:
            if row['method'] == 'shared_semantic':
                row['metrics'].update(Q=.6, J_nav=.6)
        missing = next(r for r in rows if r['parent_id'] == 'SEM_P00' and r['method'] == 'G'
                       and r['condition'] == 'nominal_relationship')
        missing['metrics'] = None
        result = compare(protocol, rows)
        self.assertEqual(result['complete_reviewed_parent_pairs'], 5)
        self.assertEqual(result['status'], 'insufficient_complete_pairs')
        self.assertIsNone(result['pairs'][0]['delta_J'])
        self.assertIsNone(missing['metrics'])
        bounds = result['full_parent_hypothetical_sensitivity_bounds']['J_nav_mean_candidate_minus_baseline']
        self.assertAlmostEqual(bounds[0], .6-(1.+1.)/6)
        self.assertIsNone(result['full_development_gate_pass'])

    def test_ratio_of_means_not_mean_of_ratios(self):
        protocol, rows = fixture_rows()
        baseline = [.1, .2, .3, .4, .5, .6]
        for row in rows:
            value = baseline[int(row['parent_id'][-2:])]+(.03 if row['method'] == 'shared_semantic' else 0.)
            row['metrics'].update(Q=value, J_nav=value)
        result = compare(protocol, rows)
        self.assertAlmostEqual(result['relative_difference_of_means'], .03/.35)
        self.assertEqual(result['status'], 'numeric_thresholds_passed')

    def test_zero_baseline_is_undefined_not_infinite_pass(self):
        protocol, rows = fixture_rows()
        for row in rows:
            if row['method'] == 'G':
                row['metrics'].update(Q=0., J_nav=0.)
        result = compare(protocol, rows)
        self.assertEqual(result['status'], 'undefined_zero_baseline')
        self.assertIsNone(result['relative_difference_of_means'])
        self.assertGreater(result['complete_pair_absolute_mean_J_difference'], 0.)

    def test_one_parent_cannot_carry_algorithm_gate(self):
        protocol, rows = fixture_rows()
        for row in rows:
            if row['method'] == 'shared_semantic':
                value = .7 if row['parent_id'] == 'SEM_P00' else .21
                row['metrics'].update(Q=value, J_nav=value)
        result = compare(protocol, rows, consistency=True)
        self.assertEqual(result['positive_parent_count'], 6)
        self.assertFalse(result['numeric_checks']['positive_gain_concentration'])
        self.assertEqual(result['status'], 'numeric_thresholds_failed')

    def test_coverage_and_failed_return_remain_guards(self):
        protocol, rows = fixture_rows()
        for row in rows:
            if row['method'] == 'shared_semantic':
                row['metrics'].update(C_nav=.95, Q=.4, J_nav=.38)
                if row['parent_id'] == 'SEM_P00':
                    row['returned_xy_and_yaw'] = False
        result = compare(protocol, rows)
        self.assertTrue(result['numeric_checks']['relative_J'])
        self.assertFalse(result['numeric_checks']['coverage'])
        self.assertFalse(result['numeric_checks']['return_rate'])

    def test_bound_existing_review_is_only_metric_source(self):
        review, episode = review_fixture()
        self.assertEqual(m.review_metrics(review, episode), review['evaluation']['metrics'])
        review['status'] = 'experiment_replay_verified'
        self.assertIsNone(m.review_metrics(review, episode))

    def test_archived_review_allows_only_current_source_drift_when_explicit(self):
        review, episode = review_fixture()
        episode['current_sources_match'] = False
        with self.assertRaisesRegex(ValueError, 'bound sources'):
            m.review_metrics(review, episode)
        self.assertEqual(m.review_metrics(review, episode, require_current_sources=False),
                         review['evaluation']['metrics'])
        review['source_sha256'] = {'different.py': 'a'*64}
        with self.assertRaisesRegex(ValueError, 'source/input/episode pin'):
            m.review_metrics(review, episode, require_current_sources=False)

    def test_rejects_unbound_review_partial_replay_and_wrong_metric(self):
        review, episode = review_fixture()
        variants = []
        for key in ('episode_manifest_sha256', 'reference_manifest_sha256', 'evaluator_source_sha256'):
            altered = deepcopy(review)
            altered['evaluation'][key] = '0'*64
            variants.append(altered)
        altered = deepcopy(review); altered['replay']['frames_verified'] = 2; variants.append(altered)
        altered = deepcopy(review); altered['evaluation']['metrics']['J_nav'] = .5; variants.append(altered)
        altered = deepcopy(review); altered['runtime']['after']['world'] = 1; variants.append(altered)
        for item in variants:
            with self.subTest(item=item):
                with self.assertRaises(ValueError):
                    m.review_metrics(item, episode)


class SceneSummaryReuseTests(unittest.TestCase):
    """Reuse the producer's temporary analytic boundary fixture, never a run."""
    @classmethod
    def setUpClass(cls):
        from tests.test_reuse_semantic_scene_endpoint_evaluation import SemanticSceneEndpointReuseTests
        cls.fixture_type = SemanticSceneEndpointReuseTests
        cls.fixture_type.setUpClass()

    def setUp(self):
        from scripts import reuse_semantic_scene_endpoint_evaluation as producer
        self.producer = producer
        self.fixture = self.fixture_type(methodName='test_exact_reuse_rebinds_target_identity_inputs_and_success')
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        # This makes only an explicitly synthetic temporary fixture receipt;
        # its World/replay/evaluator are mocked or forbidden by that fixture.
        self.receipt = self.fixture.reuse()

    def check(self, receipt=None, **kwargs):
        with patch.object(self.producer, 'reuse', side_effect=AssertionError('summary must not create reuse')):
            result = m.reused_evaluation(self.receipt if receipt is None else receipt,
                                        self.fixture.episodes['target'], **kwargs)
        self.fixture.forbid_score.assert_not_called()
        self.fixture.forbid_replay.assert_not_called()
        return result

    def test_saved_chain_accepts_only_rebound_target_score_and_marks_reused(self):
        verified = self.check()
        self.assertEqual(verified['evaluation_execution'], 'reused')
        self.assertFalse(verified['numerical_evaluation_recomputed'])
        self.assertTrue(verified['independent_target_replay_verified'])
        self.assertEqual(verified['metrics'], self.fixture.reviews['source']['evaluation']['metrics'])
        self.assertFalse(verified['evaluation']['task_success'])
        self.assertEqual(verified['evaluation']['original_episode_status'], 'budget_exhausted')
        self.assertEqual(verified['evaluation']['episode_manifest_sha256'],
                         self.fixture.episodes['target']['manifest_sha256'])

    def test_rejects_forged_numeric_value_target_pin_and_equivalence_flag(self):
        variants = []
        altered = deepcopy(self.receipt)
        altered['evaluation']['metrics'].update(Q=.9, J_nav=.45)
        variants.append(altered)
        altered = deepcopy(self.receipt); altered['evaluation']['task_success'] = True; variants.append(altered)
        altered = deepcopy(self.receipt)
        altered['evaluation']['input_prediction_sha256']['mesh.npz'] = '0'*64
        variants.append(altered)
        altered = deepcopy(self.receipt)
        altered['equivalence_proof']['complete_source_and_target_saved_policy_TSDF_replays_verified'] = False
        variants.append(altered)
        altered = deepcopy(self.receipt); altered['target_replay']['frames_verified'] = 1; variants.append(altered)
        altered = deepcopy(self.receipt); altered['numerical_evaluation_recomputed'] = True; variants.append(altered)
        for receipt in variants:
            with self.subTest(change=receipt):
                with self.assertRaises(ValueError):
                    self.check(receipt)

    def test_target_independent_replay_cannot_be_replaced_by_source_replay(self):
        self.fixture.change_review('target', lambda r: r['replay'].update(frames_verified=1))
        changed = deepcopy(self.receipt)
        changed['target_review_sha256'] = m.sha(Path(changed['target_review_path']))
        with self.assertRaisesRegex(ValueError, 'independent complete'):
            self.check(changed)

    def test_original_complete_source_review_cannot_be_a_reuse_chain(self):
        self.fixture.change_review('source', lambda r: r.update(numerical_evaluation_recomputed=False))
        changed = deepcopy(self.receipt)
        changed['source_review_sha256'] = m.sha(Path(changed['source_review_path']))
        with self.assertRaisesRegex(ValueError, 'not a reuse chain'):
            self.check(changed)

    def test_rechecks_original_arrays_not_just_saved_equality_flags(self):
        import numpy as np
        vertices = self.fixture.vertices.copy()
        vertices[1, 0] = np.nextafter(1., 2.)
        self.fixture.save_mesh(self.fixture.episodes['target']['root'], vertices=vertices)
        self.fixture.refresh('target')
        changed = deepcopy(self.receipt)
        changed['episode_manifest_sha256'] = self.fixture.episodes['target']['manifest_sha256']
        changed['target_review_sha256'] = m.sha(Path(changed['target_review_path']))
        with self.assertRaisesRegex(ValueError, 'array differs exactly'):
            self.check(changed)

    def test_changed_original_review_and_producer_copy_pins_are_rejected(self):
        changed = deepcopy(self.receipt)
        changed['source_review_sha256'] = '0'*64
        with self.assertRaisesRegex(ValueError, 'external pin'):
            self.check(changed)
        copy = Path(self.receipt['supplemental_source_copy'])
        copy.write_bytes(copy.read_bytes()+b'\n# changed synthetic copy\n')
        with self.assertRaisesRegex(ValueError, 'producer source'):
            self.check()

    def test_supplemental_discovery_uses_run_identity_and_rejects_duplicates(self):
        directory = self.fixture.root/'supplemental'
        directory.mkdir()
        path = directory/'arbitrary_filename.json'
        path.write_text(json.dumps(self.receipt))
        slots = {self.receipt['run_id']: self.fixture.episodes['target']['slot']}
        result = m.reuse_index(directory, slots)
        self.assertEqual(list(result), [self.receipt['run_id']])
        (directory/'second_receipt.json').write_text(json.dumps(self.receipt))
        with self.assertRaisesRegex(ValueError, 'duplicate target'):
            m.reuse_index(directory, slots)

    def test_retired_reuse_checks_archives_without_current_source_identity(self):
        for episode in self.fixture.episodes.values():
            episode.update(current_sources_match=False,
                           current_source_differences=['synthetic later implementation'])
        copy = Path(self.receipt['supplemental_source_copy'])
        copy.write_bytes(copy.read_bytes()+b'\n# synthetic historical producer\n')
        self.receipt['supplemental_source_sha256'] = m.sha(copy)
        reference = dict(surface=dict(fingerprint='f'*64), coverage=self.fixture.coverage)
        with patch('nso.semantic_scene_experiment.inspect_experiment', side_effect=self.fixture.inspect), \
                patch.object(m, 'historical_reference_record', return_value=reference):
            result = self.check(historical_sources=True)
            self.assertEqual(result['source_validation'], 'archived_executed_sources')
            copy.write_bytes(copy.read_bytes()+b'# changed after pin\n')
            with self.assertRaisesRegex(ValueError, 'producer source'):
                self.check(historical_sources=True)


if __name__ == '__main__':
    unittest.main()
