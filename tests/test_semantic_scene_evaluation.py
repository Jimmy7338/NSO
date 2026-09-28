"""Analytic reference/prediction fixtures; no World, episode or real score."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from nso.offline_evaluation_v44 import _array_sha,sha256
from nso.semantic_scene_evaluation import (
    EVALUATION_PARAMETERS,PREDICTION_FILES,evaluate_semantic_scene_endpoint,
    evaluate_semantic_scene_prediction,load_semantic_scene_reference,prepare_semantic_scene_reference,
)
from nso.surface_evaluation_v40 import CandidateViewV40,freeze_reference_v40


class SemanticSceneEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.asset_id='SEM_P00__nominal_relationship'
        self.asset_pin='a'*64
        self.vertices=np.array([[0,0,2],[0,1,2],[1,1,2],[1,0,2],
            [1.5,0,2],[1.5,1,2],[2.5,1,2],[2.5,0,2]],float)
        self.triangles=np.array([[0,1,2],[0,2,3],[4,5,6],[4,6,7]])
        self.owners=np.array([0,0,1,1])
        self.workspace=dict(bounds_xy_m=[[0.,0.],[4.,4.]],room_inner_bounds_xy_m=[[0.,0.],[4.,4.]],
            start_position_world_m=[.5,2.,.9])
        self.asset=dict(asset_id=self.asset_id,manifest_sha256=self.asset_pin,
            arrays=dict(vertices=self.vertices,triangles=self.triangles,triangle_instance_id=self.owners),
            public_workspace=self.workspace,public_spec={},metadata=dict(parent_id='SEM_P00',
                condition='nominal_relationship',public_workspace=self.workspace,background_boxes=[],
                private_instances=[dict(instance_id=i,world_aabb_m=[[x,0.,1.9],[x+1.,1.,2.1]])
                                   for i,x in enumerate((0.,1.5))]))
        view_row=dict(intrinsic=[[10.,0.,30.],[0.,10.,30.],[0.,0.,1.]],
            world_from_camera=np.eye(4).tolist(),width=100,height=100)
        self.views=[CandidateViewV40(**view_row)]
        self.addCleanup(patch.stopall)
        patch('nso.semantic_scene_evaluation._asset',return_value=self.asset).start()
        patch('nso.semantic_scene_evaluation.reachable_candidates',
              return_value=(self.views,dict(candidate_views=[view_row]))).start()
        self.prepared=prepare_semantic_scene_reference(self.asset_id,self.root/'reference',
            asset_root=self.root/'explicit_analytic_asset_double',asset_manifest_sha256=self.asset_pin)
        self.reference,self.domain,self.record=self.load()
        self.prediction=self.root/'prediction';self.prediction.mkdir()
        self.belief=np.full(self.domain.shape,-1,np.int8)
        self.belief[self.domain]=0
        descriptor=self.record['coverage']
        self.mapper=dict(backend_poisoned=False,frames=1,occupancy_sha256=_array_sha(self.belief),
            **{key:descriptor[key] for key in ('shape','resolution_m','origin_xy_m','grid_convention')})
        self.save_prediction(self.triangles)

    def load(self,**changes):
        args=dict(manifest_sha256=self.prepared['manifest_sha256'],asset_id=self.asset_id,
            asset_root=self.root/'explicit_analytic_asset_double',asset_manifest_sha256=self.asset_pin)
        args.update(changes)
        return load_semantic_scene_reference(self.root/'reference',**args)

    def save_prediction(self,triangles,*,vertices=None,extra=None):
        vertices=self.vertices if vertices is None else vertices
        np.savez(self.prediction/'mesh.npz',vertices=vertices,triangles=triangles,
            vertex_colors=np.zeros_like(vertices),**(extra or {}))
        np.savez(self.prediction/'occupancy.npz',belief=self.belief,observed=self.belief>=0)
        (self.prediction/'mapper.json').write_text(json.dumps(self.mapper))

    def evaluate(self,**changes):
        args=dict(expected_prediction_sha256={name:sha256(self.prediction/name) for name in PREDICTION_FILES},
            expected_frames=1,public_workspace=self.workspace)
        args.update(changes)
        return evaluate_semantic_scene_prediction(self.prediction,self.reference,self.domain,self.record,**args)

    def test_prepared_reference_equals_unchanged_mathematical_builder(self):
        direct=freeze_reference_v40(self.vertices,self.triangles,self.owners,self.views,
            sample_spacing_m=.3,seed=4001,max_samples=50000)
        self.assertEqual(self.reference.fingerprint,direct.fingerprint)
        self.assertEqual(self.record['target_instance_inventory'],[0,1])
        self.assertEqual(self.record['evaluation_parameters'],EVALUATION_PARAMETERS)

    def test_exact_whole_mesh_and_measured_coverage(self):
        result=self.evaluate()
        self.assertEqual(result['metrics']['Q'],1.)
        self.assertEqual(result['metrics']['C_nav'],1.)
        self.assertEqual(result['metrics']['J_nav'],1.)
        self.assertNotIn('J',result['metrics'])
        self.assertFalse(result['formal_performance_evidence'])
        self.assertEqual(result['new_worlds'],0)

    def test_missing_facility_remains_in_macro_denominator(self):
        self.save_prediction(self.triangles[:2])
        result=self.evaluate()['metrics']
        self.assertEqual(result['Q'],.5)
        self.assertEqual(len(result['per_instance']),2)
        self.assertEqual(result['per_instance'][1]['f1'],0.)

    def test_false_geometry_outside_targets_is_not_cropped(self):
        vertices=np.vstack([self.vertices,[[3.,0.,2.],[3.,1.,2.],[4.,1.,2.],[4.,0.,2.]]])
        triangles=np.vstack([self.triangles,[[8,9,10],[8,10,11]]])
        self.save_prediction(triangles,vertices=vertices)
        result=self.evaluate()['metrics']
        self.assertGreater(result['extra_false_positive_area_m2'],.99)
        self.assertLess(result['Q'],1.)
        self.assertEqual(result['submitted_prediction_triangles'],6)

    def test_empty_mesh_scores_zero_for_all_facilities(self):
        self.save_prediction(np.empty((0,3),int),vertices=np.empty((0,3)))
        result=self.evaluate()['metrics']
        self.assertEqual(result['Q'],0.)
        self.assertEqual(len(result['per_instance']),2)

    def test_false_occupied_cells_do_not_count_as_coverage(self):
        self.belief[self.domain]=1
        self.mapper['occupancy_sha256']=_array_sha(self.belief)
        self.save_prediction(self.triangles)
        result=self.evaluate()
        self.assertEqual(result['coverage']['touched_domain_fraction'],1.)
        self.assertEqual(result['metrics']['C_nav'],0.)
        self.assertEqual(result['metrics']['Q'],1.)

    def test_prediction_pin_and_mapper_digest_are_required(self):
        pins={name:sha256(self.prediction/name) for name in PREDICTION_FILES}
        pins['mesh.npz']='0'*64
        with self.assertRaisesRegex(ValueError,'prediction differs'):
            self.evaluate(expected_prediction_sha256=pins)
        self.belief[self.domain]=1
        self.save_prediction(self.triangles)
        with self.assertRaisesRegex(ValueError,'occupancy disagrees'):
            self.evaluate()

    def test_coordinate_history_and_roi_arrays_are_rejected(self):
        with self.assertRaisesRegex(ValueError,'coordinates/history'):
            self.evaluate(expected_frames=2)
        self.save_prediction(self.triangles,extra=dict(triangle_instance_id=self.owners))
        with self.assertRaisesRegex(ValueError,'complete original mesh'):
            self.evaluate()

    def test_reference_asset_pin_and_artifact_tampering_are_rejected(self):
        with self.assertRaisesRegex(ValueError,'asset binding'):
            self.load(asset_manifest_sha256='b'*64)
        path=self.root/'reference/candidate_views.json'
        path.write_text(path.read_text()+' ')
        with self.assertRaisesRegex(ValueError,'artifact changed'):
            self.load()

    def test_endpoint_cannot_ignore_executed_source_mismatch(self):
        with self.assertRaisesRegex(ValueError,'executed sources'):
            evaluate_semantic_scene_endpoint(dict(current_sources_match=False),'unused',
                reference_manifest_sha256='0'*64,asset_root='unused',asset_manifest_sha256='0'*64)


if __name__=='__main__':
    unittest.main()
