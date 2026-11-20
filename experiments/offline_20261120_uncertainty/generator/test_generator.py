"""Small renderer/generator contract checks, no template fitting or scoring."""
import copy
import tempfile
import unittest
from pathlib import Path
import numpy as np
from spec import objects,CONDITIONS,SENSOR_CONFIG
from build_cases import private_truth
from render_cases import render_case,dependencies


class GeneratorTests(unittest.TestCase):
    def test_split_and_frame_counts(self):
        records=objects();self.assertEqual(len(records),24)
        self.assertEqual(len({r['object_id']for r in records}),24)
        self.assertEqual({s:sum(r['split']==s for r in records)for s in ('train','validation','heldout')},dict(train=12,validation=6,heldout=6))
        self.assertEqual(len(records)*sum(map(len,CONDITIONS.values())),288)

    def test_truth_extent_matches_independent_geometry(self):
        for record in objects():
            truth=private_truth(record);points=np.asarray(truth['object_vertices_world']);rotation=np.asarray(truth['world_rotation'])
            local=points@rotation
            np.testing.assert_allclose(np.ptp(local,axis=0),truth['true_width_depth_height_m'],rtol=0,atol=1e-12)

    def test_current_hit_mask_and_sanitized_contract(self):
        import json
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp)/'case';render_case(objects()[0],'occluded',root)
            manifest=json.loads((root/'manifest.json').read_text())
            for row in manifest['frames']:
                with np.load(row['frame']['path'])as frame,np.load(row['owner']['path'])as owner:
                    self.assertFalse(owner['visible_owner'][frame['depth_m']==0].any())
                    self.assertTrue(np.array_equal(owner['visible_owner']>0,frame['semantic']>0))
                    self.assertNotIn('reference_geometry_id',frame.files)
            self.assertNotIn('truth',manifest);self.assertNotIn('split',manifest)

    def test_counterfactual_identical_front(self):
        # Same physical front and view, different unseen enclosure depths.
        # This is a unit condition, not an additional independent calibration object.
        record=copy.deepcopy(objects()[0]);record.update(width_m=1.4,height_m=2.1,yaw_deg=0.,depth_m=.3)
        with tempfile.TemporaryDirectory()as tmp:
            a=Path(tmp)/'a';b=Path(tmp)/'b';render_case(record,'front',a,counterfactual=True)
            record['depth_m']=1.1;render_case(record,'front',b,counterfactual=True)
            with np.load(a/'frame_00.npz')as x,np.load(b/'frame_00.npz')as y:
                for name in x.files:np.testing.assert_array_equal(x[name],y[name],err_msg=name)
            with np.load(a/'owner_00.npz')as x,np.load(b/'owner_00.npz')as y:
                np.testing.assert_array_equal(x['visible_owner'],y['visible_owner'])


if __name__=='__main__':unittest.main()
