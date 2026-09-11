"""Actual V3 geometric interfaces and independent route-contract validation."""
import copy
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import open3d as o3d
from env.grid_exploration import GridConfig
from env.virtual3d import VirtualConfig
from nso.counterfactual_candidates_v6 import augment_candidate_routes
from nso.counterfactual_view_scoring import CounterfactualScoreConfig, _validate_routes
from scripts.eval_counterfactual_views import candidate_routes, states_to_actions, advance_states


class MeasuredLedger:
    def __init__(self, category=2):
        self.config=VirtualConfig(depth_sigma_m=0.,dropout=0.,voxel_m=.03)
        self.shape=(30,40);self.belief=np.zeros(self.shape,np.int8);self.belief[:,24:]=-1
        self.keyframes=[];self.quality={}
        for y in np.linspace(2.5,3.15,6):
            for z in np.linspace(.3,1.2,6):
                point=np.array([4.,y,z]);key=tuple(np.floor(point/.15).astype(int))
                self.quality[key]={'point':point,'normal':np.array([1.,0.,0.]),'n':1,'bits':1,
                                   'label':category,'information':.1,'best_range':2.,'residual':.001,'normal_dispersion':0.}
    def quality_evidence(self,max_points=1600):
        rows=list(self.quality.values())[:max_points]
        return None if not rows else {key:np.asarray([r[key] for r in rows]) for key in rows[0]}
    def evidence(self):
        q=self.quality_evidence()
        return q['point'],q['bits'],q['label']
    def mesh(self):return o3d.geometry.TriangleMesh()


class CandidateAugmentationTests(unittest.TestCase):
    def setUp(self):
        self.mapper=MeasuredLedger()
        self.obs=SimpleNamespace(position=(15,10),heading=1,step=20,belief=self.mapper.belief,collision=False)
        self.original,_=candidate_routes(self.mapper,self.obs)
        self.assertEqual(len(self.original),12)

    def test_original_twelve_unchanged_new_routes_paid_safe_and_distinct(self):
        before=json.dumps(self.original,sort_keys=True);belief=self.mapper.belief.copy()
        extra,audit=augment_candidate_routes(self.mapper,self.obs,self.original)
        self.assertEqual(before,json.dumps(self.original,sort_keys=True))
        np.testing.assert_array_equal(self.mapper.belief,belief)
        self.assertEqual(len(extra),4)
        self.assertFalse({tuple(r['pose']) for r in extra}&{tuple(r['pose']) for r in self.original})
        self.assertEqual(len({r['candidate_id'] for r in self.original+extra}),16)
        config=CounterfactualScoreConfig(GridConfig(resolution_m=.2,robot_radius_m=.2,sensor_range_m=4.,sensor_fov_deg=360.),self.mapper.config)
        _validate_routes(self.original+extra,self.mapper,config)
        for route in extra:
            states=[tuple(s) for s in route['states']]
            self.assertEqual(states_to_actions(states),route['actions'])
            self.assertEqual(advance_states(states[0],route['actions']),states)
            self.assertEqual(states[route['arrival_action']],tuple(route['pose']))
            self.assertLessEqual(route['cost'],48)
        self.assertEqual(len(audit['heading_candidates']),4*audit['ring_cells'])

    def test_swapped_absent_labels_and_gain_method_cannot_change_selection(self):
        outcomes=[]
        with patch('nso.semantic_completion_v3.ObjectCompletionModel.gain',side_effect=AssertionError('candidate selection used gain')):
            for category in (2,3,0):
                mapper=MeasuredLedger(category)
                extra,audit=augment_candidate_routes(mapper,self.obs,self.original)
                outcomes.append((extra,audit))
        self.assertEqual(outcomes[0],outcomes[1]);self.assertEqual(outcomes[0],outcomes[2])

    def test_directional_representation_requires_observed_support_in_fov(self):
        states=[[15,10,1],[15,10,2],[15,10,3],[15,10,0],[15,10,1]]
        away=[{'candidate_id':0,'states':states,'pose':[15,10,3],'arrival_action':2,'cost':4}]
        toward=[{'candidate_id':0,'states':states,'pose':[15,10,1],'arrival_action':4,'cost':4}]
        a,wrong=augment_candidate_routes(self.mapper,self.obs,away,max_extra=0)
        b,right=augment_candidate_routes(self.mapper,self.obs,toward,max_extra=0)
        self.assertEqual(a,[]);self.assertEqual(b,[])
        self.assertEqual(sum(map(sum,wrong['initial_representation_counts'])),0)
        self.assertEqual(sum(map(sum,right['initial_representation_counts'])),1)

    def test_new_route_count_deterministic_and_bounded(self):
        first,a=augment_candidate_routes(self.mapper,self.obs,self.original,max_extra=2)
        second,b=augment_candidate_routes(self.mapper,self.obs,copy.deepcopy(self.original),max_extra=2)
        self.assertEqual(first,second);self.assertEqual(a,b);self.assertEqual(len(first),2)
        with self.assertRaises(ValueError):augment_candidate_routes(self.mapper,self.obs,self.original,max_extra=5)

    def test_invalid_original_action_and_unknown_footprint_rejected(self):
        broken=copy.deepcopy(self.original);broken[0]['cost']+=1
        with self.assertRaises(ValueError):augment_candidate_routes(self.mapper,self.obs,broken)
        broken=[{'candidate_id':0,'states':[[15,10,1],[15,25,1],[15,10,1]],'pose':[15,25,1],'arrival_action':1,'cost':2}]
        with self.assertRaises(ValueError):augment_candidate_routes(self.mapper,self.obs,broken)


if __name__=='__main__':unittest.main()
