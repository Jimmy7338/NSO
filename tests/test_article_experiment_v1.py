"""Article orchestration and measured-evidence regressions; no study World.

All run-ledger fixtures live in temporary directories. Controller checks use
finite analytic packets only, and the real physics counters must not change.
These are implementation tests, never development/pilot episode evidence.
"""
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from env import article_scene_sensor_v1 as sensor_module
from env.development_sensor_v41 import DevelopmentSensorV41, runtime_counts_v41
from nso import article_experiment_v1 as experiment
from nso.controller_article_v1 import (ArticleControllerV1, ArticleObservedResidualV1,
    ArticleViewPredictorV1, _without_article_audit)
from nso.controller_semantic_mechanism import SemanticMechanismController
from nso.instance_belief_v40 import PaidRGBDObservationV40
from nso.observed_instances_article_v1 import ObservedInstancesArticleV1
from nso.observed_instances_local import ObservedInstancesLocal
from nso.observed_mapper_v42 import ObservedMapperV42
from nso.observed_residual_v41 import observed_points_v41
from nso.observed_instances_v41 import support_digest
from nso.primitive_navigation_v41 import PrimitiveStateV41
from nso.prototype_observation_channel import _PLANE_KEYS, build_channel
from nso.surface_evaluation_v40 import CandidateViewV40
from nso.view_quality_v42 import ViewQualityPredictorV42
from tests.test_controller_v43 import graph, mapper, packet
from tests.test_observed_marker_planes_article_v1 import fixture as sparse_marker
from scripts import review_article_episode_20260928 as reviewer


def protocol_fixture(phase='development', count=1):
    split = 'T0' if phase == 'main' else 'DEV'
    return dict(schema='article.experiment_protocol.v1', status='frozen', phase=phase,
        slots={f'fixture_{i:03d}':dict(method='G', scene_id=f'ART1_AISLE_{split}',
                                     budget=48, noise_seed=100+i) for i in range(count)},
        output_relative_path='audit_results/article_stage_20260928/test_only',
        evaluation=deepcopy(experiment.EVALUATION), reserve_bytes=1024**3,
        maximum_episode_bytes=64*1024**2, maximum_phase_bytes=128*1024**2,
        maximum_elapsed_s=60, controller={},
        mapper=dict(resolution_m=.1), asset_root='not_a_real_asset',
        asset_manifest_sha256='a'*64,
        references={f'ART1_AISLE_{split}':{'fixture':True}},
        source_sha256={'fixture_source':'b'*64})


def article_controller(method='G', **options):
    values = dict(home=PrimitiveStateV41('home',0), budget=48,
        palette={'cabinet':[40,100,220]},
        structure_names=['planar','recessed','louvered','open_frame'],
        class_structure_prior={'cabinet':[.1,.7,.1,.1]},
        maximum_candidates=4, maximum_views_per_instance=4)
    values.update(options)
    return ArticleControllerV1(graph(),method=method,**values)


class NoWorldTests(unittest.TestCase):
    def setUp(self):
        self.physics_before = runtime_counts_v41()

    def tearDown(self):
        self.assertEqual(self.physics_before,runtime_counts_v41(),
                         'unit checks must not create a World or query physical sensors')


class ArticleProtocolTests(NoWorldTests):
    def test_phase_caps_and_split_separation(self):
        self.assertEqual(experiment.PHASE_CAPS,dict(development=12,main=96,ablation=24))
        for phase,cap in experiment.PHASE_CAPS.items():
            with self.subTest(phase=phase):
                experiment.validate_protocol(protocol_fixture(phase,cap))
                with self.assertRaisesRegex(ValueError,'ceiling'):
                    experiment.validate_protocol(protocol_fixture(phase,cap+1))
                wrong=protocol_fixture(phase)
                wrong['slots']['fixture_000']['scene_id'] = (
                    'ART1_AISLE_DEV' if phase=='main' else 'ART1_AISLE_T0')
                with self.assertRaisesRegex(ValueError,'test layouts'):
                    experiment.validate_protocol(wrong)

    def test_undeclared_modes_drafts_and_old_roots_are_rejected(self):
        changes=[('status','draft'),('output_relative_path','audit_results/v43_old'),
                 ('output_relative_path','audit_results/article_stage_20260928/../old'),
                 ('output_relative_path','audit_results/article_stage_20260928/phase/reset'),
                 ('maximum_elapsed_s',901),('maximum_episode_bytes',65*1024**2),
                 ('maximum_episode_bytes',2*1024**2-1),
                 ('reserve_bytes',1024**3-1)]
        for name,value in changes:
            p=protocol_fixture();p[name]=value
            with self.subTest(field=name,value=value),self.assertRaises(ValueError):
                experiment.validate_protocol(p)
        for name,value in [('method','old_S'),('scene_id','DEV_A_00'),
                           ('budget',True),('budget',161),('noise_seed',-1)]:
            p=protocol_fixture();p['slots']['fixture_000'][name]=value
            with self.subTest(field=name,value=value),self.assertRaises(ValueError):
                experiment.validate_protocol(p)
        p=protocol_fixture();p['controller']['method']='S'
        with self.assertRaisesRegex(ValueError,'shadow'):
            experiment.validate_protocol(p)

    def test_source_closure_contains_adapter_physics_mapper_and_measured_evidence(self):
        names=experiment.source_hashes()
        for name in ('nso/article_experiment_v1.py','scripts/run_article_experiment_20260928.py',
            'env/article_scene_sensor_v1.py','env/development_sensor_v41.py',
            'nso/article_scene_assets_v1.py','nso/controller_article_v1.py',
            'nso/observed_instances_article_v1.py','nso/observed_marker_planes_article_v1.py',
            'nso/observed_mapper_v42.py','nso/episode_driver_v43.py',
            'nso/surface_evaluation_v40.py'):
            self.assertIn(name,names)
            self.assertEqual(len(names[name]),64)

    def test_storage_gate_keeps_one_gib_reserve_and_episode_allowance(self):
        allowance=64*1024**2; reserve=1024**3
        with tempfile.TemporaryDirectory() as directory:
            for free,passed in ((reserve+allowance-1,False),(reserve+allowance,True)):
                with patch.object(sensor_module.shutil,'disk_usage',
                                  return_value=SimpleNamespace(free=free)):
                    result=sensor_module.article_storage_report(Path(directory)/'not-created')
                    self.assertEqual(result['passed'],passed)
                    self.assertFalse(result['old_v41_storage_gate_changed'])
            self.assertEqual(list(Path(directory).iterdir()),[])
        for args in ({'expected_episode_bytes':False},{'expected_episode_bytes':0},
                     {'reserve_bytes':reserve-1},{'reserve_bytes':float(reserve)}):
            with self.assertRaises(ValueError):
                sensor_module.article_storage_report('/tmp',**args)

    def test_capture_motion_return_and_tsdf_are_shared_original_implementations(self):
        for name in ('_capture','initial_observation','step','close','public_status'):
            self.assertIs(getattr(sensor_module.ArticleSceneSensorV1,name),
                          getattr(DevelopmentSensorV41,name))
        self.assertIs(experiment.ObservedMapperV42,ObservedMapperV42)


class ArticleLedgerTests(NoWorldTests):
    def test_failed_and_unfinished_reservations_cannot_be_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            p=protocol_fixture(count=2); output=Path(directory)/'phase'
            ledger=experiment.ArticleLedger(output,p,'a'*64)
            ledger.reserve('fixture_000')
            with self.assertRaisesRegex(ValueError,'no automatic retry'):
                ledger.reserve('fixture_000')
            ledger.finish('fixture_000',status='attempt_failed',world_created=False,
                          message='injected test failure',automatic_retry=False)
            ledger.reserve('fixture_001')
            restored=experiment.ArticleLedger(output,p,'a'*64)
            for run in p['slots']:
                with self.assertRaisesRegex(ValueError,'no automatic retry'):
                    restored.reserve(run)
            with self.assertRaisesRegex(ValueError,'active reservation'):
                restored.finish('fixture_000',status='controller_stop')
            entries=experiment.read(restored.path)['entries']
            self.assertEqual([x['status'] for x in entries],['attempt_failed','reserved'])
            self.assertEqual(entries[0]['message'],'injected test failure')

    def test_exhausted_or_rebound_ledger_is_not_a_fresh_queue(self):
        with tempfile.TemporaryDirectory() as directory:
            p=protocol_fixture(count=12); output=Path(directory)/'phase'
            ledger=experiment.ArticleLedger(output,p,'a'*64)
            for run in p['slots']:
                ledger.reserve(run)
            with self.assertRaises(ValueError):ledger.reserve('fixture_012')
            with self.assertRaisesRegex(ValueError,'different configuration'):
                experiment.ArticleLedger(output,p,'b'*64).reserve('fixture_000')
            self.assertEqual(len(experiment.read(ledger.path)['entries']),12)

    def test_two_output_revisions_share_cumulative_phase_cap(self):
        with tempfile.TemporaryDirectory() as directory:
            p=protocol_fixture(count=6)
            first=experiment.ArticleLedger(Path(directory)/'phase_v1',p,'a'*64)
            second=experiment.ArticleLedger(Path(directory)/'phase_v2',p,'b'*64)
            for ledger in (first,second):
                for run in p['slots']:
                    ledger.reserve(run)
                    ledger.finish(run,status='attempt_failed',world_created=False)
            third=experiment.ArticleLedger(Path(directory)/'phase_v3',protocol_fixture(),'c'*64)
            with self.assertRaisesRegex(ValueError,'cumulative phase start ceiling'):
                third.reserve('fixture_000')
            registry=experiment.read(Path(directory)/'execution_registry.json')
            self.assertEqual(len(registry['entries']),12)
            self.assertEqual({r['phase'] for r in registry['entries']},{'development'})

    def test_interrupted_local_ledger_write_still_consumes_global_slot(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger=experiment.ArticleLedger(Path(directory)/'phase',protocol_fixture(),'a'*64)
            original=experiment.write_new
            def interrupted(path,value):
                if Path(path)==ledger.path.with_suffix('.new'):
                    raise OSError('injected local ledger write interruption')
                return original(path,value)
            with patch.object(experiment,'write_new',side_effect=interrupted):
                with self.assertRaises(OSError):ledger.reserve('fixture_000')
            self.assertFalse(ledger.path.exists())
            self.assertEqual(len(experiment.read(ledger.registry_path)['entries']),1)
            with self.assertRaisesRegex(ValueError,'globally reserved attempt'):
                ledger.reserve('fixture_000')


class ArticleControllerTests(NoWorldTests):
    def test_g_b_s_share_geometry_frontend_and_action_zero_access(self):
        choices=[]; snapshots=[]
        for name in ('G','B','S'):
            c=article_controller(name,multi_view_planes=True,measurement_acquisition=True)
            self.assertIsInstance(c._ledger,ObservedInstancesArticleV1)
            m=mapper();obs=packet();m.update(obs);c.accept(obs,m)
            choices.append(c.choose());snapshots.append(c.snapshot()['geometry'])
            self.assertTrue(c._actual_feedback)
            self.assertTrue(c._future_information)
            self.assertFalse(choices[-1]['ground_truth_scene_input'])
        self.assertEqual(snapshots[0],snapshots[1]);self.assertEqual(snapshots[0],snapshots[2])
        for other in choices[1:]:
            self.assertEqual(choices[0]['global_selection'],other['global_selection'])
            self.assertEqual(choices[0]['action'],other['action'])

    def test_nbv_retains_paid_direct_route_and_has_no_diagnostic_lookahead(self):
        c=article_controller('NBV');m=mapper();obs=packet();m.update(obs);c.accept(obs,m)
        with patch.object(SemanticMechanismController,'_select_global',
                          side_effect=AssertionError('NBV must not invoke diagnostic lookahead')):
            result=c.choose()
        selection=result['global_selection']
        self.assertTrue(selection['myopic_nbv'])
        self.assertEqual(selection['diagnostic_options'],[])
        self.assertFalse(selection['future_sensor_rendered'])
        current,home=c._router.state,c._router.home
        for row in selection['direct_options']:
            state=PrimitiveStateV41(**row['target'])
            expected=c._graph.route(current,state).cost+1+c._graph.route(state,home).cost
            self.assertEqual(row['total_cost'],expected)
            self.assertLessEqual(expected,c._router.budget)
            self.assertAlmostEqual(row['score'],row['expected_gain']/expected)
        self.assertFalse(c.snapshot()['configuration']['independent_external_implementation_reproduction'])

    def test_one_paid_marker_packet_crosses_new_audit_whitelist_and_old_predictor(self):
        choices=[]
        for name in ('G','B','S','NBV'):
            c=article_controller(name);m=mapper();obs=packet(measured_plane=True)
            m.update(obs);receipt=c.accept(obs,m)
            self.assertTrue(receipt['association']['accepted'])
            self.assertIn('article_feedback_eligibility',receipt['association']['accepted'][0])
            choices.append(c.choose())
            self.assertFalse(c.snapshot()['poisoned'])
            self.assertTrue(choices[-1]['global_selection']['forecasts'])

    def test_audit_adapter_removes_only_declared_field_and_keeps_gt_rejection(self):
        obs=packet(measured_plane=True)
        ledger=ObservedInstancesArticleV1(palette={'cabinet':[40,100,220]},
            structure_names=['planar','recessed','louvered','open_frame'],
            class_structure_prior={'cabinet':[.1,.7,.1,.1]})
        rows=ledger.observe(obs)['accepted']; before=deepcopy(rows)
        clean=_without_article_audit(rows)
        self.assertEqual(rows,before)
        self.assertNotIn('article_feedback_eligibility',clean[0])
        with self.assertRaisesRegex(ValueError,'whitelist'):
            ViewQualityPredictorV42().observe(obs,rows)
        residual=ArticleObservedResidualV1()
        adapter=ArticleViewPredictorV1(residual=residual)
        adapter.observe(obs,rows)
        polluted=deepcopy(rows);polluted[0]['true_structure']='not permitted'
        with self.assertRaisesRegex(ValueError,'whitelist'):
            ArticleViewPredictorV1(residual=ArticleObservedResidualV1()).observe(obs,polluted)
        self.assertIn('true_structure',_without_article_audit(polluted)[0])

    def test_cache_repair_switch_and_feedback_ablations_are_explicit(self):
        self.assertIs(type(article_controller(cache_feedback_repair=False)._ledger),ObservedInstancesLocal)
        off=article_controller('S_no_feedback');future=article_controller('S_no_future')
        self.assertFalse(off._actual_feedback);self.assertTrue(off._future_information)
        self.assertTrue(future._actual_feedback);self.assertFalse(future._future_information)
        for options in ({'cache_feedback_repair':1},{'multi_view_planes':0}):
            with self.assertRaises(ValueError):article_controller(**options)

    def test_all_four_methods_refuse_an_action_that_cannot_return(self):
        for name in ('NBV','G','B','S'):
            c=article_controller(name,budget=1);m=mapper();obs=packet()
            m.update(obs);c.accept(obs,m);decision=c.choose()
            self.assertEqual(decision['action'],'stop')
            self.assertEqual(c.snapshot()['paid_step'],0)

    def test_collector_conflict_invalidates_cached_residual_and_forecast_channel(self):
        residual=ArticleObservedResidualV1(multi_view_planes=True)
        predictor=ArticleViewPredictorV1(residual=residual)
        first,row=sparse_marker(0,sparse=False)
        evidence=residual.observe(first,[row]);predictor.observe(first,[row])
        self.assertTrue(evidence['results'][0]['accepted'])
        self.assertIn('instance_test',residual._planes)
        second,row=sparse_marker(1,sparse=False,noise=.001)
        # Isolate the collector's later rejection, independently of the legacy
        # per-frame fit. The receipt API models measured conflict, never GT.
        with patch.object(residual.article_planes,'receipt',return_value={'conflict':True}):
            evidence=residual.observe(second,[row]);predictor.observe(second,[row])
        self.assertFalse(evidence['results'][0]['accepted'])
        self.assertEqual(evidence['results'][0]['reason'],'conflicting_paid_marker_planes')
        self.assertNotIn('instance_test',residual._planes)
        planes=predictor.observed_plane_receipts()
        self.assertTrue(planes['instance_test']['conflict'])
        candidate=CandidateViewV40(view_id='conflict-check',intrinsic=second.intrinsic,
            world_from_camera=second.world_from_camera,width=96,height=72,near_m=.1,far_m=4.)
        channel=build_channel(planes['instance_test'],candidate)
        self.assertEqual(channel['fallback_reason'],'conflicting_observed_label_planes')
        np.testing.assert_array_equal(channel['channel'],np.full((4,4),.25))

    def test_three_paid_marker_views_use_compatible_plane_and_channel_receipts(self):
        residual=ArticleObservedResidualV1(multi_view_planes=True)
        predictor=ArticleViewPredictorV1(residual=residual)
        for step,x in enumerate((-.2,0.,.2)):
            sparse,row=sparse_marker(step,origin_x=x,noise=.006)
            # Add finite measured neutral support while retaining 12 marker pixels.
            depth=sparse.depth_m.copy();depth[25:45,30:66]=1.08
            marker=np.asarray(row['marker_pixel_indices'])
            depth.ravel()[marker]=sparse.depth_m.ravel()[marker]
            obs=PaidRGBDObservationV40(sparse.frame_id,step,sparse.rgb,depth,
                                       sparse.intrinsic,sparse.world_from_camera)
            pixels=np.flatnonzero(depth);points=observed_points_v41(obs,pixels)
            row.update(observation_sha256=obs.sha256(),pixel_indices=pixels.tolist(),
                points_world_m=points.tolist(),support_sha256=support_digest(points),
                article_feedback_eligibility={'test_only':True})
            evidence=residual.observe(obs,[row]);predicted=predictor.observe(obs,[row])
        self.assertTrue(evidence['results'][0]['accepted'])
        self.assertEqual(evidence['results'][0]['reason'],
                         'current_depth_residual_with_paid_multiview_plane')
        self.assertEqual(predicted['article_multiview_supplemented_instances'],['instance_test'])
        plane_receipt=predictor.observed_plane_receipts()['instance_test']
        self.assertFalse(set(plane_receipt['plane_fit'])-_PLANE_KEYS)
        self.assertFalse(plane_receipt['plane_fit']['estimated_from_current_pixels'])
        candidate=CandidateViewV40(view_id='analytic-channel',intrinsic=obs.intrinsic,
            world_from_camera=obs.world_from_camera,width=96,height=72,near_m=.1,far_m=4.)
        result=build_channel(plane_receipt,candidate)
        self.assertIsInstance(result,dict)


class ArticleRunFailureTests(NoWorldTests):
    def fixture(self,stack):
        root=Path(stack.enter_context(tempfile.TemporaryDirectory()))
        stack.enter_context(patch.object(experiment,'ROOT',root))
        p=protocol_fixture();path=root/'protocol.json';experiment.write_new(path,p)
        stack.enter_context(patch.object(experiment,'source_hashes',return_value=p['source_sha256']))
        stack.enter_context(patch.object(experiment,'article_storage_report',return_value={'passed':True}))
        bundle=dict(public_spec={'task':dict(max_actions=160,budget_tier='standard')},
                    workspace={'fixture':True},graph_spec={'fixture':True})
        stack.enter_context(patch.object(experiment,'load_public_article_scene',return_value=bundle))
        reference=(None,None,{'asset_manifest_sha256':p['asset_manifest_sha256']})
        stack.enter_context(patch.object(experiment,'load_reference',return_value=reference))
        return root,p,path

    def test_source_mismatch_and_resource_block_never_reserve(self):
        with ExitStack() as stack:
            root,p,path=self.fixture(stack)
            with patch.object(experiment,'source_hashes',return_value={}), \
                 patch.object(experiment,'ArticleSceneSensorV1') as world:
                with self.assertRaisesRegex(ValueError,'sources differ'):
                    experiment.run_episode(path,'fixture_000')
                world.assert_not_called()
            with patch.object(experiment,'article_storage_report',return_value={'passed':False}), \
                 patch.object(experiment,'ArticleLedger') as ledger, \
                 patch.object(experiment,'ArticleSceneSensorV1') as world:
                result=experiment.run_episode(path,'fixture_000')
                self.assertEqual(result['status'],'resources_blocked_before_reservation')
                ledger.assert_not_called();world.assert_not_called()
            self.assertFalse((root/'audit_results').exists())

    def test_component_failure_is_saved_before_world_and_never_retried(self):
        with ExitStack() as stack:
            root,p,path=self.fixture(stack)
            with patch.object(experiment,'make_components',side_effect=RuntimeError('injected analytic failure')), \
                 patch.object(experiment,'ArticleSceneSensorV1') as world:
                result=experiment.run_episode(path,'fixture_000')
                world.assert_not_called()
                self.assertEqual(result['status'],'attempt_failed')
                self.assertFalse(result['world_created']);self.assertFalse(result['automatic_retry'])
                output=root/p['output_relative_path']
                failure=experiment.read(output/'episodes/fixture_000/attempt_failure.json')
                self.assertEqual(failure['message'],'injected analytic failure')
                archived=experiment.read(output/'episodes/fixture_000/public_spec.json')
                self.assertEqual(archived['task']['max_actions'],p['slots']['fixture_000']['budget'])
                entry=experiment.read(output/'start_ledger.json')['entries'][0]
                self.assertEqual(entry['status'],'attempt_failed')
                self.assertEqual(entry['result_sha256'],experiment.file_sha256(
                    output/'episodes/fixture_000/attempt_failure.json'))
                with self.assertRaises(FileExistsError):experiment.run_episode(path,'fixture_000')

    def test_writer_initialization_failure_is_retained_and_reserved_once(self):
        with ExitStack() as stack:
            root,p,path=self.fixture(stack)
            with patch.object(experiment,'CompressedStepWriterV44',side_effect=OSError('injected writer failure')), \
                 patch.object(experiment,'ArticleSceneSensorV1') as world:
                result=experiment.run_episode(path,'fixture_000')
            world.assert_not_called()
            self.assertEqual(result['status'],'attempt_failed')
            output=root/p['output_relative_path']
            self.assertEqual(experiment.read(output/'start_ledger.json')['entries'][0]['status'],
                             'attempt_failed')
            self.assertEqual(len(experiment.read(output.parent/'execution_registry.json')['entries']),1)
            failure=experiment.read(output/'episodes/fixture_000/attempt_failure.json')
            self.assertEqual(failure['message'],'injected writer failure')


class ArticleReadOnlyReviewTests(NoWorldTests):
    def test_running_reservation_is_pending_and_creates_no_review_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);episode=root/'phase/episodes/fixture'
            episode.mkdir(parents=True)
            experiment.write_new(root/'phase/start_ledger.json',dict(entries=[dict(
                run_id='fixture',status='reserved')]))
            target=root/'reviews/fixture'
            result=reviewer.review(episode,target)
            self.assertEqual(result['status'],'pending')
            self.assertFalse(target.exists())
            self.assertEqual(list(episode.iterdir()),[])

    def test_retained_pre_world_failure_is_reviewed_without_calling_sensor(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);episode=root/'phase/episodes/fixture'
            episode.mkdir(parents=True)
            failure=episode/'attempt_failure.json'
            experiment.write_new(failure,dict(status='attempt_failed',world_created=False,
                automatic_retry=False,message='finite review fixture'))
            experiment.write_new(root/'phase/start_ledger.json',dict(entries=[dict(
                run_id='fixture',status='attempt_failed',result_sha256=reviewer.sha(failure))]))
            before=failure.read_bytes();result=reviewer.review(episode,root/'reviews/fixture')
            self.assertEqual(result['status'],'failed_attempt_preserved')
            self.assertTrue(result['all_checks_passed'])
            self.assertEqual(failure.read_bytes(),before)
            self.assertFalse(result['qualified'])
            self.assertTrue((root/'reviews/fixture/manifest.json').exists())


if __name__=='__main__':
    unittest.main()
