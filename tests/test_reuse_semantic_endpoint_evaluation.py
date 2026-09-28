"""Analytic pinned-artifact doubles; no actual episode, replay or scoring."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from scripts import reuse_semantic_endpoint_evaluation as module


class ReuseSemanticEndpointEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.sources={name:module.file_sha256(module.ROOT/name) for name in module.EVALUATOR_SOURCES}
        self.protocol=dict(phase_id=module.PHASE,primary_experiment_started=False,
            slots={run:dict(method=method,asset_id='DEV_A_00') for run,method in module.RUNS.items()},
            evaluation=dict(module.EVALUATION),reference_root='explicit_analytic_reference_double',
            reference_manifest_sha256='c'*64)
        self.coverage=dict(shape=[2,2],resolution_m=.1,origin_xy_m=[0.,0.],grid_convention='analytic grid')
        self.belief=np.array([[0,0],[-1,-1]],np.int8)
        self.vertices=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]])
        self.triangles=np.array([[0,1,2]],np.int32)
        self.episodes={};self.reviews={}
        for name,run,status in (('source','integration_A_G','controller_stop'),('target','integration_A_B','budget_exhausted')):
            root=self.root/name;(root/'prediction').mkdir(parents=True)
            self.save_mesh(root)
            np.savez(root/'prediction/occupancy.npz',belief=self.belief,observed=self.belief>=0)
            mapper=dict(self.coverage,frames=2,backend_poisoned=False,tsdf_integration_count=2,
                        occupancy_sha256=module._array_sha(self.belief))
            self.write(root/'prediction/mapper.json',mapper)
            self.write(root/'public_workspace.json',dict(analytic=True))
            self.write(root/'public_spec.json',dict(analytic=True))
            manifest=dict(source_sha256=self.sources)
            self.write(root/'artifact_manifest.json',manifest)
            episode=dict(root=root,started=dict(run_id=run),protocol=deepcopy(self.protocol),
                slot=deepcopy(self.protocol['slots'][run]),manifest=manifest,
                manifest_sha256=module.file_sha256(root/'artifact_manifest.json'),
                result=dict(status=status,acquired_and_saved_packets=2,
                    sensor_status=dict(returned_xy_and_yaw=status=='controller_stop')),
                current_sources_match=True,current_source_differences=[])
            self.episodes[name]=episode
            replay=dict(status='verified',verification_kind='same_driver_saved_packet_policy_and_TSDF_replay',
                prediction_verified=True,occupancy_exact=True,mesh_order_invariant_tolerance_m=1e-9,
                frames_verified=2,terminal_status_verified=status,replayed_tsdf_integrations=2,
                counterfactual_trajectory=False,new_worlds=0,new_sensor_packets=0,physical_actions=0,
                runtime=dict(before={},after={}))
            review=dict(schema='semantic.experiment.review.v1',status='experiment_reviewed' if name=='source' else 'experiment_replay_verified',
                run_id=run,phase_id=module.PHASE,episode_manifest_sha256=episode['manifest_sha256'],
                source_sha256=self.sources,current_sources_match=True,current_source_differences=[],
                runtime=dict(before={},after={},no_new_world_or_sensor_action=True),replay=replay,elapsed_s=.01)
            if name=='source':
                review['evaluation']=dict(reference_manifest_sha256='c'*64,task_success=True,
                    original_episode_status=status,all_task_instances_in_macro_denominator=True,
                    prediction_roi_cropped=False,new_worlds=0,new_sensor_packets=0,new_tsdf_integrations=0,
                    coverage=dict(C_nav=.5,denominator=self.coverage),
                    metrics=dict(reference_fingerprint='f'*64,prediction_sample_spacing_m=.3,
                        prediction_seed=4002,threshold_m=.05,C_nav=.5,Q=.8,J_nav=.4))
            self.reviews[name]=review
            self.write(self.root/(name+'_review.json'),review)
        self.addCleanup(patch.stopall)
        patch.object(module,'inspect_experiment',side_effect=self.inspect).start()
        patch.object(module,'load_reference_v44',return_value=(SimpleNamespace(fingerprint='f'*64),
            np.ones((2,2),bool),dict(coverage=self.coverage))).start()
        patch.object(module,'runtime_counts_v41',return_value={}).start()
        self.forbid_score=patch('nso.surface_evaluation_v40.evaluate_surface_v40',
            side_effect=AssertionError('numerical evaluation must never be invoked')).start()

    def write(self,path,value):
        path.write_bytes(module.canonical_bytes(value))

    def save_mesh(self,root,*,vertices=None,compressed=False):
        function=np.savez_compressed if compressed else np.savez
        function(root/'prediction/mesh.npz',vertices=self.vertices if vertices is None else vertices,
            triangles=self.triangles,vertex_colors=np.zeros_like(self.vertices))

    def inspect(self,root,pin):
        selected=next(episode for episode in self.episodes.values() if episode['root']==Path(root))
        if pin!=module.file_sha256(Path(root)/'artifact_manifest.json'):
            raise ValueError('analytic manifest pin mismatch')
        return deepcopy(selected)

    def reuse(self,**changes):
        args=dict(output=self.root/'reused.json')
        for name in ('source','target'):
            episode=self.episodes[name];review=self.root/(name+'_review.json')
            args.update({name+'_episode':episode['root'],name+'_manifest_sha256':episode['manifest_sha256'],
                name+'_review':review,name+'_review_sha256':module.file_sha256(review)})
        args.update(changes)
        return module.reuse(**args)

    def change_review(self,role,callback):
        callback(self.reviews[role])
        self.write(self.root/(role+'_review.json'),self.reviews[role])

    def numerical_supplement(self):
        evaluation=deepcopy(self.reviews['source']['evaluation'])
        self.change_review('source',lambda row:row.update(status='experiment_review_failed',
            error=deepcopy(module.POSITIVE_FACE_FAILURE)))
        self.reviews['source'].pop('evaluation')
        self.write(self.root/'source_review.json',self.reviews['source'])
        tiny=self.vertices*1e-7
        for episode in self.episodes.values():
            self.save_mesh(episode['root'],vertices=tiny)
        xyz=tiny[self.triangles]
        areas=np.linalg.norm(np.cross(xyz[:,1]-xyz[:,0],xyz[:,2]-xyz[:,0]),axis=1)/2
        evaluation['metrics'].update(prediction_mesh_validation='strict_positive_area_without_absolute_area_floor',
            positive_subthreshold_faces_preserved=1,submitted_prediction_triangles=1)
        sources=dict(self.sources)
        sources.update({name:module.file_sha256(module.ROOT/name) for name in module.SUPPLEMENTAL_SOURCES})
        archive=self.root/'numerical_source'
        for name in sources:
            path=archive/name;path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes((module.ROOT/name).read_bytes())
        episode=self.episodes['source']
        value=dict(schema='semantic.endpoint.numeric_supplement.v1',status='semantic_endpoint_evaluation_completed',
            numerical_evaluation_recomputed=True,original_replay_rerun=False,
            episode_manifest_sha256=episode['manifest_sha256'],run_id=episode['started']['run_id'],
            episode_root=str(episode['root']),original_review_path=str(self.root/'source_review.json'),
            original_review_sha256=module.file_sha256(self.root/'source_review.json'),
            numerical_source_sha256=sources,numerical_source_root=str(archive),evaluation=evaluation,
            diagnosis=dict(prediction_triangles=1,finite_vertices=True,zero_area_faces=0,
                positive_faces_below_old_floor=1,minimum_area_m2=float(areas.min()),all_faces_preserved=True,
                mesh_sha256=module.file_sha256(episode['root']/'prediction/mesh.npz'),old_floor_twice_area_m2=1e-12),
            runtime=dict(before={},after={},no_new_world_or_sensor_action=True),elapsed_s=.01)
        path=self.root/'numeric_receipt.json';self.write(path,value)
        return value,path

    def reuse_with_supplement(self,path):
        return self.reuse(source_numerical_evaluation=path,source_numerical_evaluation_sha256=module.file_sha256(path))

    def test_exact_reuse_preserves_target_success_and_is_not_a_review(self):
        result=self.reuse()
        self.assertEqual(result['status'],'experiment_endpoint_evaluation_reused')
        self.assertFalse(result['numerical_evaluation_recomputed'])
        self.assertFalse(result['evaluation']['task_success'])
        self.assertEqual(result['evaluation']['original_episode_status'],'budget_exhausted')
        self.assertEqual(result['evaluation']['metrics'],self.reviews['source']['evaluation']['metrics'])
        self.assertEqual(result['new_worlds'],0)
        self.assertTrue((self.root/'reused.source.py').is_file())
        self.forbid_score.assert_not_called()

    def test_zip_container_bytes_can_differ_only_when_arrays_are_bit_identical(self):
        self.save_mesh(self.episodes['target']['root'],compressed=True)
        result=self.reuse()
        proof=result['equivalence_proof']['exact_original_arrays']['prediction/mesh.npz']
        self.assertFalse(proof['zip_bytes_identical'])
        self.assertTrue(all(row['values_dtype_shape_and_bits_identical'] for row in proof['arrays'].values()))

    def test_one_ulp_mesh_difference_is_rejected(self):
        changed=self.vertices.copy();changed[1,0]=np.nextafter(1.,2.)
        self.save_mesh(self.episodes['target']['root'],vertices=changed)
        with self.assertRaisesRegex(ValueError,'array differs exactly'):
            self.reuse()

    def test_changed_occupancy_and_mapper_coordinates_are_rejected(self):
        path=self.episodes['target']['root']/'prediction/mapper.json'
        mapper=module.read_json(path);mapper['resolution_m']=.2;self.write(path,mapper)
        with self.assertRaisesRegex(ValueError,'mapper coordinate'):
            self.reuse()
        mapper['resolution_m']=.1;self.write(path,mapper)
        belief=self.belief.copy();belief[0,0]=1
        np.savez(path.parent/'occupancy.npz',belief=belief,observed=belief>=0)
        with self.assertRaisesRegex(ValueError,'array differs exactly'):
            self.reuse()

    def test_each_independent_replay_is_required(self):
        self.change_review('target',lambda row:row['replay'].update(prediction_verified=False))
        with self.assertRaisesRegex(ValueError,'independent complete'):
            self.reuse()

    def test_source_must_be_actual_complete_review_not_reuse(self):
        self.change_review('source',lambda row:row.update(status='experiment_replay_verified'))
        with self.assertRaisesRegex(ValueError,'original successful review'):
            self.reuse()

    def test_stale_source_or_wrong_phase_is_rejected(self):
        self.episodes['target']['current_sources_match']=False
        with self.assertRaisesRegex(ValueError,'executed sources'):
            self.reuse()
        self.episodes['target']['current_sources_match']=True
        self.episodes['target']['protocol']['phase_id']='not_integration'
        with self.assertRaisesRegex(ValueError,'three frozen integration slots'):
            self.reuse()

    def test_changed_evaluation_configuration_or_metric_seed_is_rejected(self):
        self.episodes['target']['protocol']['evaluation']['max_samples']=50000
        with self.assertRaisesRegex(ValueError,'integration slots'):
            self.reuse()
        self.episodes['target']['protocol']['evaluation']=deepcopy(module.EVALUATION)
        self.change_review('source',lambda row:row['evaluation']['metrics'].update(prediction_seed=99))
        with self.assertRaisesRegex(ValueError,'original numerical evaluation'):
            self.reuse()

    def test_review_pin_cannot_be_replaced(self):
        with self.assertRaisesRegex(ValueError,'external SHA256 pin'):
            self.reuse(target_review_sha256='0'*64)

    def test_final_pin_recheck_rejects_evidence_changed_during_proof(self):
        original=module._equal_arrays
        def mutate(*args):
            proof=original(*args)
            path=self.root/'target_review.json';path.write_bytes(path.read_bytes()+b' ')
            return proof
        with patch.object(module,'_equal_arrays',side_effect=mutate):
            with self.assertRaisesRegex(ValueError,'changed during reuse proof'):
                self.reuse()
        self.assertFalse((self.root/'reused.json').exists())

    def test_positive_face_supplement_preserves_original_failure_and_target_status(self):
        value,path=self.numerical_supplement()
        result=self.reuse_with_supplement(path)
        self.assertEqual(result['source_original_review_status'],'experiment_review_failed')
        self.assertEqual(result['source_original_review_error'],module.POSITIVE_FACE_FAILURE)
        self.assertTrue(result['source_numeric_supplement']['numerical_evaluation_recomputed'])
        self.assertFalse(result['numerical_evaluation_recomputed'])
        self.assertFalse(result['evaluation']['task_success'])
        self.forbid_score.assert_not_called()

    def test_numeric_archive_tampering_is_rejected(self):
        value,path=self.numerical_supplement()
        code=Path(value['numerical_source_root'])/module.SUPPLEMENTAL_SOURCES[0]
        code.write_bytes(code.read_bytes()+b'\n')
        with self.assertRaisesRegex(ValueError,'archived or current supplemental'):
            self.reuse_with_supplement(path)

    def test_supplement_reuse_chain_or_changed_diagnosis_is_rejected(self):
        value,path=self.numerical_supplement()
        value['numerical_evaluation_recomputed']=False;self.write(path,value)
        with self.assertRaisesRegex(ValueError,'completed original positive-face supplement'):
            self.reuse_with_supplement(path)
        value['numerical_evaluation_recomputed']=True
        value['diagnosis']['all_faces_preserved']=False;self.write(path,value)
        with self.assertRaisesRegex(ValueError,'positive-face diagnosis'):
            self.reuse_with_supplement(path)

    def test_unrelated_original_failure_cannot_use_supplement(self):
        value,path=self.numerical_supplement()
        self.change_review('source',lambda row:row.update(error=dict(type='ValueError',message='other failure')))
        with self.assertRaisesRegex(ValueError,'original successful review'):
            self.reuse_with_supplement(path)


if __name__=='__main__':
    unittest.main()
