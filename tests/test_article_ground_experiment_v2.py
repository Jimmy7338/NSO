"""Ground runner/controller regressions using analytic packets and mocked Worlds."""
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from env.development_sensor_v41 import runtime_counts_v41
from nso import article_experiment_ground_v2 as experiment
from nso import article_experiment_v1 as original
from nso.controller_article_ground_v2 import ArticleControllerGroundV2
from nso.controller_article_v1 import ArticleControllerV1
from nso.observed_instances_article_ground_v2 import ObservedInstancesArticleGroundV2
from nso.observed_instances_article_v1 import ObservedInstancesArticleV1
from nso.primitive_navigation_v41 import PrimitiveStateV41
from tests.test_article_experiment_v1 import protocol_fixture
from tests.test_controller_v43 import graph, mapper, packet


def ground_protocol(count=1, phase='ablation'):
    p = protocol_fixture(phase, count)
    p.update(schema=experiment.SCHEMA, metric_preprocessing=deepcopy(experiment.METRIC_PREPROCESSING))
    p['source_sha256'].update({name:'b'*64 for name in
        ('nso/surface_evaluation_v40.py','nso/article_prediction_mesh_adapter_v1.py')})
    p['controller']['ground_association'] = True
    if phase == 'ablation':
        p['paired_baseline_protocol'] = dict(path='baseline.json', sha256='a'*64)
        for name, slot in p['slots'].items():
            slot['paired_baseline_run_id'] = name
    return p


def controller(cls=ArticleControllerGroundV2, method='S', **extra):
    return cls(graph(), method=method, home=PrimitiveStateV41('home',0), budget=48,
        palette={'cabinet':[40,100,220]},
        structure_names=['planar','recessed','louvered','open_frame'],
        class_structure_prior={'cabinet':[.1,.7,.1,.1]},
        maximum_candidates=4, maximum_views_per_instance=4,
        multi_view_planes=True, measurement_acquisition=True, **extra)


class NoWorld(unittest.TestCase):
    def setUp(self):
        self.before = runtime_counts_v41()

    def tearDown(self):
        self.assertEqual(self.before, runtime_counts_v41())


class GroundControllerTests(NoWorld):
    def test_only_ledger_and_explicit_configuration_differ(self):
        for method in ('G','B','S','NBV'):
            old = controller(ArticleControllerV1, method)
            new = controller(method=method)
            self.assertIs(type(new._ledger), ObservedInstancesArticleGroundV2)
            self.assertIs(type(old._ledger), ObservedInstancesArticleV1)
            self.assertIs(type(new._residual), type(old._residual))
            self.assertIs(type(new._predictor), type(old._predictor))
            self.assertEqual(set(new._ledger.class_priors), set(old._ledger.class_priors))
            for key in old._ledger.class_priors:
                np.testing.assert_array_equal(new._ledger.class_priors[key],old._ledger.class_priors[key])
            self.assertEqual(new._ledger.mode, old._ledger.mode)
            a, b = deepcopy(old._configuration), deepcopy(new._configuration)
            b['article_version'] = a['article_version']
            for field in ('common_ground_association','ground_association_policy','ground_removal_scope'):
                b.pop(field)
            self.assertEqual(a,b)
            self.assertIs(ArticleControllerGroundV2._select_global, ArticleControllerV1._select_global)
            self.assertIs(ArticleControllerGroundV2.accept, ArticleControllerV1.accept)

    def test_ground_off_is_exact_old_frontend_and_switch_is_not_truthy(self):
        c = controller(ground_association=False)
        self.assertIs(type(c._ledger), ObservedInstancesArticleV1)
        for value in (0,1,None):
            with self.assertRaisesRegex(ValueError,'explicit bool'):
                controller(ground_association=value)
        with self.assertRaisesRegex(ValueError,'feedback repair'):
            controller(cache_feedback_repair=False)

    def test_original_packet_and_mapper_are_unchanged_and_receipt_is_exposed(self):
        obs = packet(measured_plane=True)
        before = obs.sha256()
        new = controller()
        association = new._ledger.observe(obs)
        new._residual.observe(obs, association['accepted'])
        new._predictor.observe(obs, association['accepted'])
        self.assertEqual(obs.sha256(), before)
        self.assertIs(experiment.ObservedMapperV42, original.ObservedMapperV42)
        ground = association['article_ground_association']
        self.assertEqual(ground['observation_sha256'], before)
        self.assertTrue(ground['full_frame_fusion_preserved'])
        self.assertFalse(ground['private_geometry_read'])
        self.assertFalse(ground['original_depth_modified'])


class GroundProtocolTests(NoWorld):
    def test_existing_caps_split_and_unique_pairings(self):
        experiment.validate_protocol(ground_protocol(12))
        experiment.validate_protocol(ground_protocol(96,'main'))
        for p in (ground_protocol(13),ground_protocol(1,'development')):
            with self.assertRaises(ValueError): experiment.validate_protocol(p)
        p=ground_protocol(2)
        p['slots']['fixture_001']['paired_baseline_run_id']='fixture_000'
        with self.assertRaisesRegex(ValueError,'unique paired'):
            experiment.validate_protocol(p)
        self.assertIs(experiment.ArticleLedger,original.ArticleLedger)
        self.assertEqual(experiment.PHASE_CAPS,original.PHASE_CAPS)

    def test_common_ground_and_canonical_metric_are_required(self):
        for field,value in [('ground_association',False),('ground_association',1),('cache_feedback_repair',False)]:
            p=ground_protocol();p['controller'][field]=value
            with self.assertRaises(ValueError):experiment.validate_protocol(p)
        p=ground_protocol();p['metric_preprocessing']['applied_equally_to_paired_baseline']=False
        with self.assertRaisesRegex(ValueError,'common canonical'):
            experiment.validate_protocol(p)
        p=ground_protocol();p['slots']['fixture_000']['controller_overrides']={'inspection_weight':2}
        with self.assertRaisesRegex(ValueError,'no slot overrides'):
            experiment.validate_protocol(p)
        p=ground_protocol();p['slots']['fixture_000']['method']='S_no_feedback'
        with self.assertRaisesRegex(ValueError,'four common'):
            experiment.validate_protocol(p)

    def test_source_closure_adds_new_sources_without_modifying_old_pins(self):
        pins=experiment.source_hashes()
        for name in ('nso/controller_article_ground_v2.py','nso/observed_instances_article_ground_v2.py',
                     'nso/article_prediction_mesh_adapter_v1.py','nso/article_experiment_ground_v2.py',
                     'scripts/run_article_ground_experiment_20260928.py'):
            self.assertIn(name,pins)
        frozen=original.read(original.ROOT/'configs/virtual3d/article_development_v1_20260928.json')['source_sha256']
        self.assertTrue(all(original.file_sha256(original.ROOT/name)==pin for name,pin in frozen.items()))
        self.assertTrue(all(pins[name]==pin for name,pin in frozen.items() if name in pins))


class GroundRunTests(NoWorld):
    def fixture(self,stack):
        root=Path(stack.enter_context(tempfile.TemporaryDirectory()))
        stack.enter_context(patch.object(experiment,'ROOT',root))
        baseline=protocol_fixture()
        baseline['references']['ART1_AISLE_DEV']['manifest_sha256']='c'*64
        original.write_new(root/'baseline.json',baseline)
        p=ground_protocol()
        p['references']=deepcopy(baseline['references'])
        p['paired_baseline_protocol']['sha256']=original.file_sha256(root/'baseline.json')
        path=root/'protocol.json';original.write_new(path,p)
        stack.enter_context(patch.object(experiment,'source_hashes',return_value=p['source_sha256']))
        stack.enter_context(patch.object(experiment,'runtime_metadata',return_value={'mock_runtime':True}))
        stack.enter_context(patch.object(experiment,'article_storage_report',return_value={'passed':True}))
        bundle=dict(public_spec={'task':dict(max_actions=160,budget_tier='standard')},
                    workspace={'fixture':True},graph_spec={'fixture':True})
        stack.enter_context(patch.object(experiment,'load_public_article_scene',return_value=bundle))
        stack.enter_context(patch.object(experiment,'load_reference',return_value=(
            None,None,dict(asset_manifest_sha256=p['asset_manifest_sha256'],coverage={}))))
        return root,p,path

    def test_pairing_rejects_every_compound_change_before_reservation(self):
        with ExitStack() as stack:
            root,p,path=self.fixture(stack)
            self.assertTrue(experiment.validate_pairing(p)['original_sources_preserved'])
            variants=[]
            for key in ('method','noise_seed','budget'):
                q=deepcopy(p);q['slots']['fixture_000'][key]={'method':'B','noise_seed':200,'budget':40}[key]
                variants.append(q)
            q=deepcopy(p);q['controller']['inspection_weight']=3;variants.append(q)
            q=deepcopy(p);q['mapper']['resolution_m']=.2;variants.append(q)
            q=deepcopy(p);q['paired_baseline_protocol']['sha256']='0'*64;variants.append(q)
            for q in variants:
                with self.assertRaises(ValueError):experiment.validate_pairing(q)
            self.assertFalse((root/'audit_results').exists())

    def test_resource_or_runtime_block_creates_no_attempt(self):
        with ExitStack() as stack:
            root,p,path=self.fixture(stack)
            with patch.object(experiment,'ArticleSceneSensorV1') as world, \
                 patch.object(experiment,'ArticleLedger') as ledger, \
                 patch.object(experiment,'article_storage_report',return_value={'passed':False}):
                result=experiment.run_episode(path,'fixture_000')
                self.assertEqual(result['status'],'resources_blocked_before_reservation')
                world.assert_not_called();ledger.assert_not_called()
            q=deepcopy(p);q['numerical_runtime']={'wrong_build':True}
            changed=root/'different_runtime.json';original.write_new(changed,q)
            with self.assertRaisesRegex(ValueError,'runtime differs'):
                experiment.run_episode(changed,'fixture_000')
            self.assertFalse((root/'audit_results').exists())

    def test_pre_world_failure_is_durable_and_cannot_retry(self):
        with ExitStack() as stack:
            root,p,path=self.fixture(stack)
            with patch.object(experiment,'make_components',side_effect=ValueError('fixture failure')), \
                 patch.object(experiment,'ArticleSceneSensorV1') as world:
                result=experiment.run_episode(path,'fixture_000')
            world.assert_not_called()
            self.assertEqual(result['status'],'attempt_failed')
            self.assertFalse(result['automatic_retry']);self.assertFalse(result['world_created'])
            output=root/p['output_relative_path']
            self.assertEqual(original.read(output/'start_ledger.json')['entries'][0]['status'],'attempt_failed')
            entries=original.read(output.parent/'execution_registry.json')['entries']
            self.assertEqual(len(entries),1);self.assertEqual(entries[0]['phase'],'ablation')
            with self.assertRaises(FileExistsError):experiment.run_episode(path,'fixture_000')

    def mock_execution(self,sensor,controller,mapper,writer,**kwargs):
        vertices=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,0.]])
        triangles=np.array([[0,1,2],[0,0,3]],dtype=np.int64)
        writer.arrays('prediction/mesh.npz',vertices=vertices,triangles=triangles)
        writer.arrays('prediction/occupancy.npz',belief=np.ones((2,2)))
        writer.json('prediction/mapper.json',{'mock':True})
        result=dict(status='controller_stop',collisions=[],sensor_status={'returned_xy_and_yaw':True},
            mapper_frames=1,acquired_and_saved_packets=1,mapper_tsdf_integrations=1,
            elapsed_s=.001,executed_paid_actions=0)
        writer.json('result.json',result,terminal=True)
        return result

    def test_success_seals_raw_mesh_and_records_preprocessing_before_evaluation(self):
        with ExitStack() as stack:
            root,p,path=self.fixture(stack)
            model=Mock();model.snapshot.return_value={'mock':True}
            stack.enter_context(patch.object(experiment,'make_components',return_value=(model,object())))
            world=stack.enter_context(patch.object(experiment,'ArticleSceneSensorV1'))
            stack.enter_context(patch.object(experiment,'execute_episode_v43',side_effect=self.mock_execution))
            stack.enter_context(patch.object(experiment,'measure_navigation_coverage_v44',return_value={'C_nav':.4}))
            episode=root/p['output_relative_path']/'episodes/fixture_000'
            def evaluate(reference,vertices,triangles,**kwargs):
                self.assertEqual(len(triangles),1)
                self.assertTrue((episode/'prediction_preprocessing.json').exists())
                self.assertTrue((episode/'prediction_seal.json').exists())
                return dict(C_map=.4,J=.32,Q=.8)
            stack.enter_context(patch.object(experiment,'evaluate_surface_v40',side_effect=evaluate))
            result=experiment.run_episode(path,'fixture_000')
            self.assertEqual(result['status'],'controller_stop',result);self.assertTrue(result['qualified'])
            world.assert_called_once()
            raw=original.read(episode/'prediction_seal.json')
            for name,pin in raw.items():self.assertEqual(original.file_sha256(episode/'prediction'/name),pin)
            with np.load(episode/'prediction/mesh.npz') as data:self.assertEqual(len(data['triangles']),2)
            receipt=original.read(episode/'prediction_preprocessing.json')
            self.assertEqual(receipt['removed_faces'],1)
            evaluation=original.read(episode/'evaluation.json')
            self.assertAlmostEqual(evaluation['metrics']['J_nav'],.32)
            self.assertEqual(evaluation['metric_preprocessing']['receipt_sha256'],
                             original.file_sha256(episode/'prediction_preprocessing.json'))
            manifest=original.read(episode/'artifact_manifest.json')
            self.assertIn('prediction_preprocessing.json',manifest['files'])

    def test_evaluation_failure_keeps_original_seal_and_consumes_attempt(self):
        with ExitStack() as stack:
            root,p,path=self.fixture(stack)
            model=Mock();model.snapshot.return_value={'mock':True}
            stack.enter_context(patch.object(experiment,'make_components',return_value=(model,object())))
            stack.enter_context(patch.object(experiment,'ArticleSceneSensorV1'))
            stack.enter_context(patch.object(experiment,'execute_episode_v43',side_effect=self.mock_execution))
            stack.enter_context(patch.object(experiment,'measure_navigation_coverage_v44',return_value={'C_nav':.4}))
            stack.enter_context(patch.object(experiment,'evaluate_surface_v40',side_effect=ValueError('metric fixture failure')))
            result=experiment.run_episode(path,'fixture_000')
            self.assertEqual(result['status'],'attempt_failed')
            episode=root/p['output_relative_path']/'episodes/fixture_000'
            self.assertTrue((episode/'prediction_preprocessing.json').exists())
            pins=original.read(episode/'prediction_seal.json')
            self.assertEqual(pins['mesh.npz'],original.file_sha256(episode/'prediction/mesh.npz'))
            with self.assertRaises(FileExistsError):experiment.run_episode(path,'fixture_000')


if __name__=='__main__':
    unittest.main()
