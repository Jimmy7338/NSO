import unittest
import json
import numpy as np
from nso.semantic_taxonomy_v15 import canonical_vote,canonical_observed_records,canonical_feature_record


class TaxonomyTests(unittest.TestCase):
    def feature(self,tail):
        return dict(geometry=[0.]*18+[.2]*8,semantic=[0.]*18+list(tail),
            residual_confidence=.7,candidate_id=8,observed_instance_count=2)

    def test_same_meaning_from_opposite_raw_encodings_matches(self):
        self.assertEqual(canonical_vote(-1,'competition_v9'),canonical_vote(1,'inspection_v4'))
        self.assertEqual(canonical_vote(1,'competition_v9'),canonical_vote(-1,'inspection_v4'))
        self.assertEqual(canonical_vote(-1,'competition_v9'),-1.)

    def test_mixed_votes_keep_linear_probability_meaning(self):
        self.assertAlmostEqual(canonical_vote(.6,'inspection_v4'),-.6)
        self.assertEqual(canonical_vote(0,'inspection_v4'),0.)

    def test_all_signed_moments_align_without_changing_geometry_or_capacity(self):
        a=self.feature(np.arange(8)/10);b=self.feature(-np.arange(8)/10)
        ca=canonical_feature_record(a,'competition_v9');cb=canonical_feature_record(b,'inspection_v4')
        self.assertEqual(ca['semantic'],cb['semantic'])
        self.assertEqual(cb['geometry'],b['geometry']);self.assertEqual(len(cb['semantic']),26)
        self.assertEqual(cb['residual_confidence'],.7)
        self.assertEqual(b['semantic'][19],-.1)

    def test_zero_confidence_remains_zero_and_input_is_not_mutated(self):
        f=self.feature([0.]*8);f['residual_confidence']=0.
        result=canonical_feature_record(f,'inspection_v4')
        self.assertEqual(result['semantic'],f['semantic']);self.assertNotIn('semantic_encoding',f)

    def test_unknown_source_and_double_encoding_fail_closed(self):
        with self.assertRaises(ValueError):canonical_vote(1,'robot_unknown')
        f=canonical_feature_record(self.feature([.1]*8),'inspection_v4')
        with self.assertRaises(ValueError):canonical_feature_record(f,'inspection_v4')

    def test_missing_evidence_has_same_serialization_across_source_encodings(self):
        a=canonical_feature_record(self.feature([0.]*8),'competition_v9')
        b=canonical_feature_record(self.feature([-0.]*8),'inspection_v4')
        self.assertEqual(json.dumps(a['semantic']),json.dumps(b['semantic']))
        self.assertEqual(json.dumps(canonical_vote(0.,'inspection_v4')),'0.0')

    def test_observed_records_preserve_instance_order_confidence_and_input(self):
        raw=[dict(asset_index=5,class_vote=1.,semantic_confidence=.4),dict(asset_index=2,class_vote=-1.,semantic_confidence=.8)]
        result=canonical_observed_records(raw,'inspection_v4')
        self.assertEqual([r['asset_index'] for r in result],[5,2])
        self.assertEqual([r['class_vote'] for r in result],[-1,1])
        self.assertEqual(raw[0]['class_vote'],1.)
        with self.assertRaises(ValueError):canonical_observed_records(result,'inspection_v4')

    def test_inconsistent_or_nonfinite_features_cannot_be_mislabeled_as_valid(self):
        for bad in [float('nan'),float('inf'),2.,True]:
            with self.assertRaises(ValueError):canonical_vote(bad,'inspection_v4')
        f=self.feature([.1]*8);f['geometry'][0]=1.
        with self.assertRaises(ValueError):canonical_feature_record(f,'inspection_v4')


if __name__=='__main__':unittest.main()
