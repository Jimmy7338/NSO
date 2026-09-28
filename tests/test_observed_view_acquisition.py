"""Analytic paid packets and public graph only; no renderer/World/evaluation."""
from copy import deepcopy
import json
import unittest

import numpy as np

from env.development_sensor_v41 import runtime_counts_v41
from nso.instance_belief_v40 import PaidRGBDObservationV40
from nso.observed_view_acquisition import ObservedViewAcquisition,PROXY_MAXIMUM_AXIAL_DEPTH_M
from nso.primitive_navigation_v41 import PrimitiveStateV41,PublicPrimitiveGraphV41
from nso.public_navigation_v43 import select_candidate_states_v43


def fixture():
    nodes={'home':[-1.,0.],'current':[0.,0.],'near':[.25,0.],'back':[2.25,0.]}
    edges=[['home','current'],['current','near'],['near','back']]
    for i in range(1,8):
        nodes[f'north{i}']=[0.,float(i)];edges.append(['current' if i==1 else f'north{i-1}',f'north{i}'])
    graph=PublicPrimitiveGraphV41(dict(schema_version='v41.public_navigation.v1',
        source_kind='provided_navigation_prior',nodes=nodes,edges=edges))
    transform=np.array([[0.,0.,1.,-1.],[-1.,0.,0.,0.],[0.,-1.,0.,.9],[0.,0.,0.,1.]])
    rgb=np.zeros((72,96,3),np.uint8);depth=np.zeros((72,96),np.float32)
    depth[33:39,45:51]=2.4;rgb[33:39,45:51]=[12,34,56]
    k=np.array([[48.,0.,47.5],[0.,48.,35.5],[0.,0.,1.]])
    packet=PaidRGBDObservationV40('analytic_paid_0',0,rgb,depth,k,transform)
    pix=np.flatnonzero(depth.ravel()>0).tolist()
    association=dict(instance_id='observed_0',pixel_indices=pix,marker_pixel_indices=pix,
        frame_id=packet.frame_id,paid_step=0,observation_sha256=packet.sha256())
    camera=dict(intrinsic=k.tolist(),width=96,height=72)
    return graph,packet,association,camera


class ObservedViewAcquisitionTests(unittest.TestCase):
    def setUp(self):
        self.before=runtime_counts_v41();self.graph,self.packet,self.association,self.camera=fixture()
        self.current=PrimitiveStateV41('current',3);self.home=PrimitiveStateV41('home',0)
        self.instances=[dict(instance_id='observed_0',association_uncertain=False)]
        self.helper=ObservedViewAcquisition();self.helper.observe(self.packet,[self.association])

    def tearDown(self):self.assertEqual(self.before,runtime_counts_v41())

    def propose(self,**changes):
        args=dict(graph=self.graph,current=self.current,home=self.home,remaining=120,
            paid_states=[self.home,self.current],camera=self.camera,instances=self.instances,planes={})
        args.update(changes);return self.helper.candidates(**args)

    def test_observed_anchor_heading_survives_nearest_four_heading_quota(self):
        wanted=PrimitiveStateV41('current',0)
        old,_=select_candidate_states_v43(self.graph,self.current,[self.home,self.current],32)
        self.assertNotIn(wanted,old)
        states,pool,protected,rows=self.propose()
        self.assertIn(wanted,states);self.assertIn(wanted,protected['observed_0'])
        self.assertLessEqual(len(states),32);self.assertEqual(len(states),len(set(states)))
        self.assertLessEqual(len(protected['observed_0']),2)
        json.dumps(pool,allow_nan=False)
        self.assertTrue(rows)

    def test_all_rows_include_paid_observe_and_full_heading_return_cost(self):
        _,_,_,rows=self.propose()
        for row in rows:
            self.assertEqual(row['outbound_cost'],self.graph.route(self.current,row['target']).cost)
            self.assertEqual(row['return_cost'],self.graph.route(row['target'],self.home).cost)
            self.assertEqual(row['total_cost'],row['outbound_cost']+1+row['return_cost'])
            _,_,_,smaller=self.propose(remaining=row['total_cost']-1)
            self.assertNotIn(row['target'],[item['target'] for item in smaller])

    def test_class_and_posterior_fields_cannot_change_candidates_or_state(self):
        expected=self.propose();before=deepcopy(self.helper._records)
        poisoned=deepcopy(self.instances);poisoned[0].update(observed_class=object(),
            structure_probabilities=object(),active_structure_prior=object(),semantic_conditioning_used=object())
        actual=self.propose(instances=poisoned)
        self.assertEqual(expected,actual)
        self.assertEqual(self.helper._last_step,0)
        for key in before['observed_0']:
            np.testing.assert_equal(before['observed_0'][key],self.helper._records['observed_0'][key])
        other=ObservedViewAcquisition();row=dict(self.association,observed_class=object(),structure_probabilities=object())
        other.observe(self.packet,[row])
        self.assertEqual(other.candidates(self.graph,self.current,self.home,120,[self.home,self.current],
            self.camera,poisoned,{}),expected)

    def test_uncertain_conflicting_resolved_and_backside_handling(self):
        for instances,planes in (([dict(instance_id='observed_0',association_uncertain=True)],{}),
            (self.instances,{'observed_0':{'plane_fit':None,'conflict':True}}),
            (self.instances,{'observed_0':{'plane_fit':{'accepted':True},'conflict':False}})):
            with self.subTest(instances=instances,planes=planes):self.assertEqual(self.propose(instances=instances,planes=planes)[3],[])
        _,_,_,rows=self.propose()
        self.assertTrue(all(row['quality_proxy']['same_side_cosine']>=.5-1e-12 for row in rows))
        self.assertNotIn('back',[row['target'].node for row in rows])
        self.assertAlmostEqual(PROXY_MAXIMUM_AXIAL_DEPTH_M,1.299038105676658)

    def test_small_pool_still_reserves_observed_facing_candidate(self):
        states,pool,protected,rows=self.propose(limit=1)
        self.assertEqual(len(states),1);self.assertEqual(protected['observed_0'],states)
        self.assertEqual(rows[0]['target'],states[0])

    def test_duplicate_unpaid_and_ambiguous_marker_inputs_are_rejected(self):
        with self.assertRaisesRegex(ValueError,'consecutive'):self.helper.observe(self.packet,[self.association])
        helper=ObservedViewAcquisition();bad=dict(self.association,observation_sha256='0'*64)
        with self.assertRaisesRegex(ValueError,'binding'):helper.observe(self.packet,[bad])
        fields={key:deepcopy(getattr(self.packet,key)) for key in self.packet.__dataclass_fields__}
        fields['depth_m'][50:53,45:48]=2.4
        split=PaidRGBDObservationV40.from_mapping(fields);pix=np.flatnonzero(split.depth_m.ravel()>0).tolist()
        assoc=dict(self.association,pixel_indices=pix,marker_pixel_indices=pix,observation_sha256=split.sha256())
        helper=ObservedViewAcquisition();receipt=helper.observe(split,[assoc])
        self.assertEqual(receipt['updated'],[]);self.assertEqual(receipt['skipped'][0]['reason'],'ambiguous_marker_components')

    def test_no_measurement_creates_no_instance_and_quality_proxy_is_finite(self):
        helper=ObservedViewAcquisition();receipt=helper.observe(self.packet,[])
        self.assertFalse(receipt['updated'])
        result=helper.candidates(self.graph,self.current,self.home,120,[self.home,self.current],self.camera,[],{})
        self.assertEqual(result[2],{});self.assertEqual(result[3],[])
        _,_,_,rows=self.propose()
        for row in rows:
            json.dumps(row['quality_proxy'],allow_nan=False)
            self.assertTrue(row['quality_proxy']['not_reconstruction_area'])
            self.assertTrue(row['quality_proxy']['not_a_success_probability'])
            self.assertNotIn('score',row)


if __name__=='__main__':unittest.main()
