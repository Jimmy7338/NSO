"""Information boundary and exact known-structure reference checks."""
import unittest
import numpy as np

from nso.known_structure_reference import (KnownStructureBelief,
    PaidInstanceStructureMatcher, KnownStructureReferenceController)
from nso.semantic_reliability import discrete_observation_evi
from nso.primitive_navigation_v41 import PrimitiveStateV41
from test_controller_v43 import graph, mapper, packet


class KnownStructureReferenceTests(unittest.TestCase):
    def test_only_discovered_instances_receive_true_structure_and_no_diagnosis_value(self):
        belief = KnownStructureBelief(geometry_prior=[.5,.5], class_structure_priors={})
        with self.assertRaises(ValueError):
            belief.disclose('unseen',0)
        belief.register('seen'); belief.disclose('seen',1)
        belief.replace_log_evidence('seen',[0.,-20.])
        self.assertEqual(belief.posterior('seen')['structure_probabilities'],[0.,1.])
        value=discrete_observation_evi(belief,'seen',np.eye(2),{'seen':[[10.,0.],[0.,3.]]})
        self.assertEqual(value['evi'],0.)
        self.assertEqual(value['best_utility_before_observation'],3.)
        belief.disclose('seen',None)
        self.assertGreater(belief.posterior('seen')['structure_probabilities'][0],.99)

    def test_matcher_discloses_no_pose_or_class_and_refuses_ambiguity(self):
        matcher=PaidInstanceStructureMatcher([
            {'marker_center_world_m':[1.,0.,.8],'structure':'a'},
            {'marker_center_world_m':[1.4,0.,.8],'structure':'b'}],structure_names=['a','b'])
        def item(key,x,uncertain=False):
            return dict(instance_id=key,marker_anchor_world_m=[x,0.,.8],association_uncertain=uncertain)
        self.assertEqual(matcher.resolve([item('ambiguous',1.2)]),{'ambiguous':None})
        self.assertEqual(matcher.resolve([item('first',.9)]),{'first':0})
        self.assertEqual(matcher.resolve([item('duplicate',.9)]),{'duplicate':None})
        self.assertEqual(matcher.resolve([item('first',.9,True)]),{'first':None})
        self.assertEqual(matcher.resolve([item('unmatched',3.)]),{'unmatched':None})

    def test_controller_has_explicit_truth_receipt_and_keeps_observed_only_area_inputs(self):
        names=['planar','recessed','louvered','open_frame']
        matcher=PaidInstanceStructureMatcher([],structure_names=names)
        c=KnownStructureReferenceController(graph(),structure_matcher=matcher,
            home=PrimitiveStateV41('home',0),budget=48,palette={'cabinet':[40,100,220]},
            structure_names=names,class_structure_prior={'cabinet':[.1,.7,.1,.1]},
            maximum_candidates=4,maximum_views_per_instance=4)
        mapping,obs=mapper(),packet(measured_plane=True)
        mapping.update(obs); receipt=c.accept(obs,mapping)
        self.assertTrue(receipt['reference_structure_disclosures'])
        self.assertFalse(receipt['reference_all_observed_instances_resolved'])
        self.assertEqual(c._planning_instances(),c._ledger.snapshot()['instances'])
        decision=c.choose()
        self.assertTrue(decision['ground_truth_structure_input'])
        self.assertFalse(decision['ground_truth_pose_used_in_planning'])
        self.assertFalse(decision['reconstruction_optimal_upper_bound'])

    def test_reference_action_values_use_exact_structure_not_geometry_weighted_summaries(self):
        names=['planar','recessed','louvered','open_frame']
        matcher=PaidInstanceStructureMatcher([dict(marker_center_world_m=[1.83,.75,.9],
                                                   structure='louvered')],structure_names=names)
        c=KnownStructureReferenceController(graph(),structure_matcher=matcher,
            home=PrimitiveStateV41('home',0),budget=48,palette={'cabinet':[40,100,220]},
            structure_names=names,class_structure_prior={'cabinet':[.1,.7,.1,.1]},
            maximum_candidates=4,maximum_views_per_instance=4,discovery_weight=0.)
        mapping,obs=mapper(),packet(measured_plane=True)
        mapping.update(obs); receipt=c.accept(obs,mapping)
        self.assertTrue(receipt['reference_all_observed_instances_resolved'])
        self.assertEqual(list(receipt['reference_structure_disclosures'].values()),['louvered'])
        selection=c.choose()['global_selection']
        forecasts=selection['forecasts']
        self.assertTrue(forecasts)
        areas={row['view_id']:row['structure_new_surface_area_m2']
               for forecast in forecasts for row in forecast['candidates']}
        for row in selection['direct_options']:
            key=f"{row['target']['node']}:{row['target']['heading']}"
            self.assertAlmostEqual(row['expected_gain'],areas[key][2])


if __name__=='__main__':
    unittest.main()
