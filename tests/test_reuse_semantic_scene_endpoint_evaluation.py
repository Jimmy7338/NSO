"""Exact-input reuse rejection fixtures; no real episode/replay/score/World."""
from copy import deepcopy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from scripts import reuse_semantic_scene_endpoint_evaluation as module


class SemanticSceneEndpointReuseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sources={name:module.file_sha256(module.ROOT/name) for name in module.source_names()}

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.asset='SEM_P00__nominal_relationship'
        self.reference=dict(root='explicit_analytic_reference_double',manifest_sha256='d'*64)
        self.input_pins=dict(zip(module.INPUT_NAMES,('a'*64,'b'*64,'c'*64,'d'*64)))
        self.protocol=dict(schema=module.SCHEMA,status='frozen',phase_id='semantic_fixture',primary_experiment_started=True,
            asset_root='explicit_analytic_assets',asset_manifest_sha256='a'*64,navigation_manifest_sha256='b'*64,
            reference_index_sha256='c'*64,evaluation=deepcopy(module.EVALUATION),
            controller=dict(measurement_acquisition=True),execution_source_sha256=deepcopy(self.sources))
        self.coverage=dict(shape=[2,2],resolution_m=.1,origin_xy_m=[0.,0.],grid_convention='analytic grid')
        self.belief=np.array([[0,0],[-1,-1]],np.int8)
        self.vertices=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]])
        self.triangles=np.array([[0,1,2]],np.int32)
        self.episodes={};self.reviews={}
        for role,method,status in (('source','G','controller_stop'),('target','bayes_semantic','budget_exhausted')):
            root=self.root/role;(root/'prediction').mkdir(parents=True)
            self.save_mesh(root)
            np.savez(root/'prediction/occupancy.npz',belief=self.belief,observed=self.belief>=0)
            mapper=dict(self.coverage,frames=2,backend_poisoned=False,tsdf_integration_count=2,
                occupancy_sha256=module._array_sha(self.belief),voxel_m=.04,sdf_trunc_m=.12,
                near_m=.1,far_m=4.,backend='analytic_fixture')
            self.write(root/'prediction/mapper.json',mapper)
            for name in ('public_workspace.json','public_spec.json','public_graph.json'):
                self.write(root/name,dict(analytic=True))
            manifest=dict(schema=module.ARTIFACT_SCHEMA,source_sha256=deepcopy(self.sources),
                input_sha256=deepcopy(self.input_pins),run_id='synthetic_'+role,files={})
            episode=dict(root=root,protocol=deepcopy(self.protocol),reference=deepcopy(self.reference),
                slot=dict(asset_id=self.asset,method=method,budget=2,noise_seed=0,tie_seed=0,tie_rule='lexicographic'),
                started=dict(run_id='synthetic_'+role),manifest=manifest,current_sources_match=True,current_source_differences=[],
                result=dict(status=status,acquired_and_saved_packets=2,
                    sensor_status=dict(returned_xy_and_yaw=status=='controller_stop')))
            replay=dict(status='verified',verification_kind='same_driver_saved_packet_policy_and_TSDF_replay',
                prediction_verified=True,occupancy_exact=True,mesh_order_invariant_tolerance_m=1e-9,
                frames_verified=2,terminal_status_verified=status,replayed_tsdf_integrations=2,
                counterfactual_trajectory=False,new_worlds=0,new_sensor_packets=0,physical_actions=0,
                runtime=dict(before={},after={}))
            review=dict(schema='semantic.scene_experiment.review.v1',
                status='experiment_reviewed' if role=='source' else 'experiment_replay_verified',
                run_id='synthetic_'+role,phase_id='semantic_fixture',source_sha256=deepcopy(self.sources),
                input_sha256=deepcopy(self.input_pins),current_sources_match=True,current_source_differences=[],
                replay=replay,runtime=dict(before={},after={},no_new_world_or_sensor_action=True),elapsed_s=.01)
            if role=='source':
                review['evaluation']=dict(schema='semantic_scene.complete_saved_prediction_evaluation.v1',
                    asset_id=self.asset,asset_manifest_sha256='a'*64,reference_manifest_sha256='d'*64,
                    evaluator_source_sha256=self.sources['nso/complete_surface_evaluation.py'],
                    task_success=True,original_episode_status=status,all_task_instances_in_macro_denominator=True,
                    prediction_roi_cropped=False,prediction_reintegrated=False,independent_replay_performed_here=False,
                    new_worlds=0,new_sensor_packets=0,new_tsdf_integrations=0,
                    coverage=dict(C_nav=.5,denominator=self.coverage),metrics=dict(reference_fingerprint='f'*64,
                        prediction_mesh_validation='strict_positive_area_without_absolute_area_floor',
                        prediction_sample_spacing_m=.3,prediction_seed=4002,threshold_m=.05,C_nav=.5,Q=.8,J_nav=.4))
            self.episodes[role]=episode;self.reviews[role]=review;self.refresh(role)
        self.addCleanup(patch.stopall)
        patch.object(module,'inspect_experiment',side_effect=self.inspect).start()
        patch.object(module,'source_names',return_value=sorted(self.sources)).start()
        patch.object(module,'load_semantic_scene_reference',return_value=(SimpleNamespace(fingerprint='f'*64),
            np.ones((2,2),bool),dict(coverage=self.coverage))).start()
        patch.object(module,'runtime_counts_v41',return_value={}).start()
        self.forbid_score=patch('nso.semantic_scene_experiment.evaluate_endpoint',
            side_effect=AssertionError('actual evaluation forbidden')).start()
        self.forbid_replay=patch('nso.semantic_scene_experiment.replay_saved_episode',
            side_effect=AssertionError('actual replay forbidden')).start()

    def write(self,path,value):
        path.write_bytes(module.canonical_bytes(value))

    def save_mesh(self,root,*,vertices=None,compressed=False,extra=None):
        function=np.savez_compressed if compressed else np.savez
        function(root/'prediction/mesh.npz',vertices=self.vertices if vertices is None else vertices,
            triangles=self.triangles,vertex_colors=np.zeros_like(self.vertices),**(extra or {}))

    def refresh(self,role):
        episode=self.episodes[role];root=episode['root'];manifest=episode['manifest']
        manifest['files']={str(path.relative_to(root)):dict(sha256=module.file_sha256(path),bytes=path.stat().st_size)
            for path in root.rglob('*') if path.is_file() and path.name!='artifact_manifest.json'}
        self.write(root/'artifact_manifest.json',manifest)
        pin=module.file_sha256(root/'artifact_manifest.json');episode['manifest_sha256']=pin
        self.reviews[role]['episode_manifest_sha256']=pin
        if role=='source':
            self.reviews[role]['evaluation'].update(episode_manifest_sha256=pin,
                input_prediction_sha256={name:manifest['files']['prediction/'+name]['sha256'] for name in module.PREDICTION_FILES})
        self.write(self.root/(role+'_review.json'),self.reviews[role])

    def inspect(self,root,pin):
        episode=next(value for value in self.episodes.values() if value['root']==Path(root))
        if module.file_sha256(Path(root)/'artifact_manifest.json')!=pin:
            raise ValueError('analytic episode manifest pin mismatch')
        for name,row in episode['manifest']['files'].items():
            if module.file_sha256(Path(root)/name)!=row['sha256']:
                raise ValueError('analytic inspected artifact changed')
        return deepcopy(episode)

    def reuse(self,**changes):
        args=dict(output=self.root/'reused.json')
        for role in ('source','target'):
            review=self.root/(role+'_review.json')
            args.update({role+'_episode':self.episodes[role]['root'],role+'_manifest_sha256':self.episodes[role]['manifest_sha256'],
                role+'_review':review,role+'_review_sha256':module.file_sha256(review)})
        args.update(changes)
        return module.reuse(**args)

    def change_review(self,role,change):
        change(self.reviews[role]);self.write(self.root/(role+'_review.json'),self.reviews[role])

    def test_exact_reuse_rebinds_target_identity_inputs_and_success(self):
        result=self.reuse()
        self.assertEqual(result['schema'],'semantic.scene_endpoint_evaluation_reused.v1')
        self.assertEqual(result['status'],'experiment_endpoint_evaluation_reused')
        self.assertFalse(result['numerical_evaluation_recomputed'])
        self.assertFalse(result['evaluation']['task_success'])
        self.assertEqual(result['evaluation']['episode_manifest_sha256'],self.episodes['target']['manifest_sha256'])
        self.assertNotEqual(result['evaluation']['episode_manifest_sha256'],self.episodes['source']['manifest_sha256'])
        self.assertEqual(result['evaluation']['metrics'],self.reviews['source']['evaluation']['metrics'])
        self.assertEqual(result['equivalence_proof']['executed_source_count'],len(self.sources))
        self.assertTrue((self.root/'reused.source.py').is_file())
        self.forbid_score.assert_not_called();self.forbid_replay.assert_not_called()

    def test_equal_arrays_in_different_zip_container_rebind_target_file_hash(self):
        self.save_mesh(self.episodes['target']['root'],compressed=True);self.refresh('target')
        result=self.reuse();proof=result['equivalence_proof']['exact_original_arrays']['prediction/mesh.npz']
        self.assertFalse(proof['zip_bytes_identical'])
        self.assertEqual(result['evaluation']['input_prediction_sha256']['mesh.npz'],proof['target_file_sha256'])
        self.assertNotEqual(result['evaluation']['input_prediction_sha256']['mesh.npz'],proof['source_file_sha256'])

    def test_one_ulp_geometry_difference_is_rejected(self):
        changed=self.vertices.copy();changed[1,0]=np.nextafter(1.,2.)
        self.save_mesh(self.episodes['target']['root'],vertices=changed);self.refresh('target')
        with self.assertRaisesRegex(ValueError,'array differs exactly'):self.reuse()

    def test_unknown_occupancy_or_mapper_coordinate_change_is_rejected(self):
        path=self.episodes['target']['root']/'prediction/occupancy.npz'
        changed=self.belief.copy();changed[0,0]=-1
        np.savez(path,belief=changed,observed=changed>=0);self.refresh('target')
        with self.assertRaisesRegex(ValueError,'array differs exactly'):self.reuse()
        np.savez(path,belief=self.belief,observed=self.belief>=0)
        mapper=module.read_json(path.parent/'mapper.json');mapper['origin_xy_m']=[.1,0.]
        self.write(path.parent/'mapper.json',mapper);self.refresh('target')
        with self.assertRaisesRegex(ValueError,'mapper coordinate'):self.reuse()

    def test_each_independent_replay_is_required(self):
        self.change_review('target',lambda row:row['replay'].update(frames_verified=1))
        with self.assertRaisesRegex(ValueError,'independent complete'):self.reuse()

    def test_source_must_be_complete_original_review(self):
        self.change_review('source',lambda row:row.update(status='experiment_replay_verified'))
        with self.assertRaisesRegex(ValueError,'original successful new-scene review'):self.reuse()

    def test_review_asset_navigation_reference_pins_cannot_drift(self):
        for key in module.INPUT_NAMES:
            with self.subTest(key=key):
                changed=deepcopy(self.input_pins);changed[key]='0'*64
                self.change_review('target',lambda row:row.update(input_sha256=changed))
                with self.assertRaisesRegex(ValueError,'all input pins'):self.reuse()
        self.change_review('target',lambda row:row.update(input_sha256=deepcopy(self.input_pins)))
        self.episodes['target']['reference']['manifest_sha256']='e'*64
        with self.assertRaisesRegex(ValueError,'input pins disagree'):self.reuse()

    def test_old_or_incomplete_execution_closure_is_rejected(self):
        self.episodes['target']['current_sources_match']=False
        with self.assertRaisesRegex(ValueError,'executed source closure'):self.reuse()
        self.episodes['target']['current_sources_match']=True
        self.episodes['target']['manifest']['source_sha256'].pop(next(iter(self.sources)))
        with self.assertRaisesRegex(ValueError,'executed source closure'):self.reuse()

    def test_declared_acquisition_closure_records_actual_59_sources(self):
        self.assertEqual(len(self.sources),59)
        self.assertIn('nso/observed_view_acquisition.py',self.sources)
        result=self.reuse()
        self.assertEqual(result['equivalence_proof']['executed_source_count'],59)
        self.assertEqual(result['source_sha256'],self.protocol['execution_source_sha256'])
        self.forbid_score.assert_not_called();self.forbid_replay.assert_not_called()

    def test_wrong_incomplete_or_empty_declared_source_maps_are_rejected(self):
        wrong=deepcopy(self.sources);wrong['nso/observed_view_acquisition.py']='0'*64
        incomplete=deepcopy(self.sources);incomplete.pop('nso/observed_view_acquisition.py')
        extra=dict(self.sources,undeclared_source='0'*64)
        for role in ('source','target'):
            for declared in (wrong,incomplete,extra,{},None):
                with self.subTest(role=role,declared_type=type(declared).__name__,count=len(declared or {})):
                    self.episodes[role]['protocol']['execution_source_sha256']=deepcopy(declared)
                    with self.assertRaisesRegex(ValueError,'protocol execution_source_sha256'):self.reuse()
                    self.episodes[role]['protocol']['execution_source_sha256']=deepcopy(self.sources)
        self.assertFalse((self.root/'reused.json').exists())

    def test_acquisition_or_correction_cannot_omit_declared_sources_even_with_58(self):
        legacy={name:pin for name,pin in self.sources.items() if name!='nso/observed_view_acquisition.py'}
        episode=self.episodes['target'];episode['manifest']['source_sha256']=legacy
        episode['protocol'].pop('execution_source_sha256')
        with patch.object(module,'source_names',return_value=sorted(legacy)):
            for controller,correction in ((dict(measurement_acquisition=True),False),({},True)):
                with self.subTest(controller=controller,correction=correction):
                    episode['protocol']['controller']=controller
                    if correction:episode['protocol']['method_correction']={}
                    with self.assertRaisesRegex(ValueError,'protocol execution_source_sha256'):
                        module._checked_episode(episode['root'],episode['manifest_sha256'])

    def test_legacy_58_branch_preserves_exact_inventory_and_reuse(self):
        # Analytic inspection models the former matching checkout. It does not
        # claim old real episodes are replayable under today's changed sources.
        legacy={name:pin for name,pin in self.sources.items() if name!='nso/observed_view_acquisition.py'}
        self.assertEqual(len(legacy),58)
        for role in ('source','target'):
            episode=self.episodes[role]
            episode['protocol'].pop('execution_source_sha256')
            episode['protocol']['controller']={}
            episode['manifest']['source_sha256']=deepcopy(legacy)
            self.reviews[role]['source_sha256']=deepcopy(legacy)
            self.refresh(role)
        with patch.object(module,'source_names',return_value=sorted(legacy)):
            result=self.reuse()
            self.assertEqual(result['equivalence_proof']['executed_source_count'],58)
            # Neither a shorter inventory nor a falsely enlarged current
            # inventory may silently enter this undeclared legacy branch.
            self.episodes['target']['manifest']['source_sha256'].pop(next(iter(legacy)))
            with self.assertRaisesRegex(ValueError,'executed source closure'):
                module._checked_episode(self.episodes['target']['root'],self.episodes['target']['manifest_sha256'])

    def test_undeclared_59_source_stage_is_not_accepted_as_legacy(self):
        episode=self.episodes['target'];episode['protocol'].pop('execution_source_sha256')
        episode['protocol']['controller']={}
        with self.assertRaisesRegex(ValueError,'legacy undeclared phase'):
            module._checked_episode(episode['root'],episode['manifest_sha256'])

    def test_numeric_source_or_source_prediction_identity_is_rejected(self):
        evaluation=self.reviews['source']['evaluation']
        original=evaluation['evaluator_source_sha256'];evaluation['evaluator_source_sha256']='0'*64
        self.change_review('source',lambda row:None)
        with self.assertRaisesRegex(ValueError,'original complete numerical evaluation'):self.reuse()
        evaluation['evaluator_source_sha256']=original;evaluation['episode_manifest_sha256']='0'*64
        self.change_review('source',lambda row:None)
        with self.assertRaisesRegex(ValueError,'original complete numerical evaluation'):self.reuse()

    def test_undeclared_pair_configuration_and_roi_arrays_are_rejected(self):
        self.episodes['target']['slot']['noise_seed']=1
        with self.assertRaisesRegex(ValueError,'paired slot'):self.reuse()
        self.episodes['target']['slot']['noise_seed']=0
        self.save_mesh(self.episodes['target']['root'],extra=dict(triangle_instance_id=np.array([0])))
        self.refresh('target')
        with self.assertRaisesRegex(ValueError,'complete original prediction array inventory'):self.reuse()

    def test_review_external_pin_is_required(self):
        with self.assertRaisesRegex(ValueError,'external pin'):self.reuse(target_review_sha256='0'*64)

    def test_final_external_pin_check_rejects_mutation_during_proof(self):
        original=module._equal_arrays
        def mutate(*args):
            proof=original(*args);path=self.root/'target_review.json'
            path.write_bytes(path.read_bytes()+b' ');return proof
        with patch.object(module,'_equal_arrays',side_effect=mutate):
            with self.assertRaisesRegex(ValueError,'external pin'):self.reuse()
        self.assertFalse((self.root/'reused.json').exists())


if __name__=='__main__':
    unittest.main()
