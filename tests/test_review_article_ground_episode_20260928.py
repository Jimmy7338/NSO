"""Independent GroundV2 review boundaries; no World or TSDF is constructed."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from nso.article_prediction_mesh_adapter_v1 import prepare_prediction_mesh_v1
from scripts import review_article_ground_episode_20260928 as review


def protocols():
    baseline=json.loads((review.ROOT/'configs/virtual3d/article_development_v1_20260928.json').read_text())
    ground=deepcopy(baseline)
    ground.update(schema=review.SCHEMA,phase='ablation',metric_preprocessing={
        'schema':review.PREPROCESSING_SCHEMA,'applied_equally_to_paired_baseline':True},
        numerical_runtime={**{key:'declared-build' for key in ('python_executable','python_version','numpy_version','numpy_path',
            'numpy_build','scipy_version','scipy_path','scipy_build')},
            'thread_environment':dict(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')})
    ground['controller']['ground_association']=True
    ground['slots']={'ground_'+key:dict(value,paired_baseline_run_id=key) for key,value in baseline['slots'].items()}
    return ground,baseline


def mesh():
    vertices=np.array([[0.,0,0],[1.,0,0],[0,2e-13,0],[0,1.,0]])
    triangles=np.array([[0,1,2],[0,1,3]],dtype=np.int32)
    _,_,receipt=prepare_prediction_mesh_v1(vertices,triangles)
    return vertices,triangles,receipt


class GroundEpisodeReviewTests(unittest.TestCase):
    def test_complete_ground_matrix_pairs_exact_original_slots(self):
        ground,baseline=protocols()
        self.assertTrue(review.verify_pairing(ground,baseline))

    def test_replaced_cli_may_leave_dependency_closure_but_shared_science_cannot(self):
        baseline={'scripts/run_article_experiment_20260928.py':'a'*64,'nso/science.py':'b'*64}
        current={'scripts/run_article_ground_experiment_20260928.py':'c'*64,'nso/science.py':'b'*64}
        review.verify_source_relationship(current,baseline)
        wrong=dict(current);wrong['nso/science.py']='d'*64
        with self.assertRaises(ValueError):review.verify_source_relationship(wrong,baseline)
        del wrong['nso/science.py']
        with self.assertRaises(ValueError):review.verify_source_relationship(wrong,baseline)

    def test_unknown_baseline_or_undeclared_method_override_rejected(self):
        ground,baseline=protocols();key=next(iter(ground['slots']))
        ground['slots'][key]['paired_baseline_run_id']='unknown'
        with self.assertRaisesRegex(ValueError,'unknown paired'):review.verify_pairing(ground,baseline)
        ground,baseline=protocols();ground['slots'][key]['controller_overrides']={'inspection_weight':20}
        with self.assertRaisesRegex(ValueError,'slot fields'):review.verify_pairing(ground,baseline)

    def test_wrong_schema_partial_favourable_matrix_or_shared_knob_change_rejected(self):
        for kind in ('schema','subset','weights','runtime'):
            ground,baseline=protocols()
            if kind=='schema':ground['schema']='article.experiment_protocol.v1'
            elif kind=='subset':ground['slots'].pop(next(iter(ground['slots'])))
            elif kind=='weights':ground['controller']['inspection_weight']=2.
            else:ground['numerical_runtime']['thread_environment']['OPENBLAS_NUM_THREADS']='2'
            with self.subTest(kind=kind),self.assertRaises(ValueError):review.verify_pairing(ground,baseline)

    def test_independent_threshold_face_indices_and_raw_arrays_are_verified(self):
        vertices,triangles,receipt=mesh()
        result=review.verify_preprocessing_arrays(vertices,triangles,receipt)
        self.assertEqual(result['removed_original_face_indices'],[0])
        self.assertEqual(result['retained_faces'],1)
        self.assertEqual(receipt['original_triangles_array_sha256'],review.array_sha(triangles))

    def test_count_only_lie_and_wrong_raw_hash_rejected(self):
        vertices,triangles,receipt=mesh()
        for field,value in [('removed_original_face_indices',[1]),('retained_faces',2),
            ('original_vertices_array_sha256','0'*64),('adapted_triangles_array_sha256','0'*64),
            ('removed_total_area_m2',0.)]:
            broken=deepcopy(receipt);broken[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):review.verify_preprocessing_arrays(vertices,triangles,broken)

    def test_no_roi_semantic_selection_or_wider_threshold_can_be_hidden(self):
        vertices,triangles,receipt=mesh()
        for field,value in [('roi_applied',True),('semantic_labels_read',True),('gt_reference_read',True),
            ('fixed_maximum_removed_face_area_m2',5e-12),('crop_box',[0,1,0,1])]:
            broken=deepcopy(receipt);broken[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):review.verify_preprocessing_arrays(vertices,triangles,broken)

    def test_zero_face_removal_requires_exact_preserved_arrays(self):
        vertices,triangles,_=mesh();triangles=triangles[1:]
        _,_,receipt=prepare_prediction_mesh_v1(vertices,triangles)
        review.verify_preprocessing_arrays(vertices,triangles,receipt)
        self.assertEqual(receipt['original_vertices_array_sha256'],receipt['adapted_vertices_array_sha256'])
        self.assertEqual(receipt['original_triangles_array_sha256'],receipt['adapted_triangles_array_sha256'])
        receipt['prediction_arrays_exactly_unchanged']=False
        with self.assertRaises(ValueError):review.verify_preprocessing_arrays(vertices,triangles,receipt)

    def test_evaluation_version_settings_and_source_pins_are_all_required(self):
        protocol=dict(evaluation=review.EVALUATION,source_sha256={
            'nso/surface_evaluation_v40.py':'a'*64,'nso/article_prediction_mesh_adapter_v1.py':'b'*64})
        evaluation=dict(schema='article.ground_v2.canonical_evaluation.v1',metric_version=review.METRIC_VERSION,
            evaluation_settings=review.EVALUATION,evaluation_source_sha256=protocol['source_sha256'],
            metric_preprocessing=dict(schema=review.PREPROCESSING_SCHEMA,receipt='prediction_preprocessing.json',receipt_sha256='c'*64),
            original_prediction_files_unchanged=True,no_roi_crop=True,semantic_weights_used=False,fixed_target_instances=4)
        review.verify_evaluation_metadata(evaluation,protocol,'c'*64)
        for field in ('metric_version','evaluation_settings','evaluation_source_sha256','no_roi_crop','fixed_target_instances'):
            broken=deepcopy(evaluation);broken.pop(field)
            with self.subTest(field=field),self.assertRaises(ValueError):review.verify_evaluation_metadata(broken,protocol,'c'*64)

    def test_reserved_running_episode_is_pending_without_reading_half_written_artifacts(self):
        with tempfile.TemporaryDirectory() as folder:
            phase=Path(folder)/'ground';episode=phase/'episodes'/'one';episode.mkdir(parents=True)
            (phase/'start_ledger.json').write_text(json.dumps(dict(entries=[dict(run_id='one',status='reserved')])))
            (episode/'protocol.json').write_text('{broken in-flight JSON')
            output=Path(folder)/'review'
            result=review.review(episode,output)
            self.assertEqual(result['status'],'pending');self.assertFalse(output.exists())
            self.assertEqual(result['new_worlds'],0)

    def test_unstarted_slot_does_not_get_failure_or_fabricated_output(self):
        with tempfile.TemporaryDirectory() as folder:
            output=Path(folder)/'review';episode=Path(folder)/'ground/episodes/one'
            result=review.review(episode,output)
            self.assertEqual(result['status'],'unstarted');self.assertFalse(output.exists())


if __name__=='__main__':unittest.main()
