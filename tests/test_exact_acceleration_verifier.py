"""Receipt comparison must never mask a numeric forecast difference."""
from copy import deepcopy
import unittest

from scripts.verify_local_exact_acceleration import (
    canonical_sha,first_differences,historical_forecast_reference,
)


class ExactAccelerationVerifierTests(unittest.TestCase):
    def forecast(self):
        geometry=dict(source_frames=[dict(paid_step=2,frame_id='a')],support_sha256='support')
        return dict(instance_id='i',paid_step=2,observed_geometry=geometry,
            geometry_evidence_sha256=canonical_sha(geometry),candidates=[dict(area=.4)])

    def test_only_later_aliased_metadata_is_restored(self):
        original=self.forecast()
        polluted=deepcopy(original)
        polluted['observed_geometry']['source_frames'].append(dict(paid_step=3,frame_id='b'))
        repaired,receipt=historical_forecast_reference(polluted)
        self.assertEqual(repaired,original)
        self.assertEqual(receipt['removed_source_frames_count'],1)
        self.assertEqual(len(polluted['observed_geometry']['source_frames']),2)

    def test_changed_geometry_cannot_be_normalized_away(self):
        changed=self.forecast()
        changed['observed_geometry']['support_sha256']='changed'
        with self.assertRaises(AssertionError):
            historical_forecast_reference(changed)

    def test_numeric_candidate_difference_remains_exactly_visible(self):
        expected=self.forecast()
        changed=deepcopy(expected)
        changed['candidates'][0]['area']+=1e-15
        reference,_=historical_forecast_reference(expected)
        differences=first_differences(changed,reference)
        self.assertEqual(differences[0]['path'],'$.candidates[0].area')
        self.assertNotEqual(changed,reference)

    def test_future_metadata_reports_its_path(self):
        a=self.forecast()
        b=deepcopy(a)
        b['observed_geometry']['source_frames'].append(dict(paid_step=3))
        self.assertEqual(first_differences(a,b),[dict(
            path='$.observed_geometry.source_frames',kind='list_length',left=1,right=2)])


if __name__=='__main__':
    unittest.main()
