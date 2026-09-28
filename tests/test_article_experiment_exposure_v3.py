"""Exposure controller and mechanically inherited lifecycle; zero real Worlds."""
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
import numpy as np
from env.development_sensor_v41 import runtime_counts_v41
from nso import article_experiment_exposure_v3 as experiment
from nso import article_experiment_v1 as original
from nso.controller_article_exposure_v3 import ArticleControllerExposureV3
from nso.controller_article_ground_v2 import ArticleControllerGroundV2
from nso.view_quality_article_exposure_v3 import ArticleExposureViewPredictorV3
from nso.controller_article_v1 import ArticleViewPredictorV1
from tests.test_article_ground_experiment_v2 import ground_protocol,controller,NoWorld


def exposure_protocol(count=1,phase='ablation'):
    p=ground_protocol(count,phase)
    p['schema']=experiment.SCHEMA
    p['controller']['same_center_exposure_dedup']=True
    p['numerical_runtime']={'mock_runtime':True}
    return p


class ExposureControllerTests(NoWorld):
    def test_all_four_methods_change_only_predictor_and_three_receipt_fields(self):
        for method in ('G','B','S','NBV'):
            old=controller(ArticleControllerGroundV2,method)
            new=controller(ArticleControllerExposureV3,method)
            self.assertIs(type(new._predictor),ArticleExposureViewPredictorV3)
            self.assertIs(type(new._ledger),type(old._ledger))
            self.assertIs(type(new._residual),type(old._residual))
            self.assertIs(new._predictor.article_residual,new._residual)
            a,b=deepcopy(old._configuration),deepcopy(new._configuration)
            for name in ('same_center_exposure_dedup','article_exposure_version','exposure_predictor_configuration_sha256'):b.pop(name)
            self.assertEqual(a,b)
            self.assertEqual(new._configuration['exposure_predictor_configuration_sha256'],new._predictor.configuration_sha256)
            self.assertIs(ArticleControllerExposureV3.accept,ArticleControllerGroundV2.accept)
            self.assertIs(ArticleControllerExposureV3._select_global,ArticleControllerGroundV2._select_global)
            self.assertNotEqual(old._configuration_sha256,new._configuration_sha256)

    def test_off_is_old_predictor_but_nonboolean_switch_rejected(self):
        new=controller(ArticleControllerExposureV3,same_center_exposure_dedup=False)
        self.assertIs(type(new._predictor),ArticleViewPredictorV1)
        for invalid in (None,0,1):
            with self.assertRaisesRegex(ValueError,'explicit bool'):
                controller(ArticleControllerExposureV3,same_center_exposure_dedup=invalid)


class ExposureProtocolTests(NoWorld):
    def test_common_switch_caps_method_and_main_split(self):
        experiment.validate_protocol(exposure_protocol(12))
        experiment.validate_protocol(exposure_protocol(96,'main'))
        for p in (exposure_protocol(13),exposure_protocol(1,'development')):
            with self.assertRaises(ValueError):experiment.validate_protocol(p)
        for key,value in (('same_center_exposure_dedup',False),('same_center_exposure_dedup',1),('ground_association',False)):
            p=exposure_protocol();p['controller'][key]=value
            with self.assertRaises(ValueError):experiment.validate_protocol(p)
        p=exposure_protocol();p['slots']['fixture_000']['controller_overrides']={'inspection_weight':3}
        with self.assertRaises(ValueError):experiment.validate_protocol(p)
        self.assertIs(experiment.ArticleLedger,original.ArticleLedger)
        self.assertEqual(experiment.PHASE_CAPS,{'development':12,'main':96,'ablation':24})

    def test_dynamic_sources_preserve_all_old_ground_files(self):
        pins=experiment.source_hashes()
        for name in ('nso/controller_article_exposure_v3.py','nso/view_quality_article_exposure_v3.py',
                     'nso/article_experiment_exposure_v3.py','scripts/run_article_exposure_experiment_20260928.py'):
            self.assertIn(name,pins)
        frozen=original.read(original.ROOT/'configs/virtual3d/article_ground_ablation_v2_20260928.json')['source_sha256']
        self.assertTrue(all(original.file_sha256(original.ROOT/name)==pin for name,pin in frozen.items()))
        self.assertTrue(all(pins[name]==pin for name,pin in frozen.items() if name in pins))


class ExposureRunTests(NoWorld):
    def fixture(self,stack):
        root=Path(stack.enter_context(tempfile.TemporaryDirectory()))
        stack.enter_context(patch.object(experiment,'ROOT',root))
        baseline=ground_protocol()
        baseline['numerical_runtime']={'mock_runtime':True}
        baseline['references']['ART1_AISLE_DEV']['manifest_sha256']='c'*64
        original.write_new(root/'baseline.json',baseline)
        p=exposure_protocol()
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
            with self.assertRaisesRegex(ValueError,'runtime'):
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
            self.assertEqual(evaluation['schema'],'article.exposure_v3.canonical_evaluation.v1')
            self.assertEqual(evaluation['metric_version'],'article.common_numeric_face_evaluation.v1')
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
