"""Evaluator-only compatibility checks; fixtures never count as method runs."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('objectfit_eval',HERE/'evaluate_objectfit.py')
ev=importlib.util.module_from_spec(spec);spec.loader.exec_module(ev)

class EvaluatorTests(unittest.TestCase):
    def assert_nested_equivalent(self,a,b):
        if isinstance(a,dict):
            self.assertEqual(set(a),set(b))
            for key in a:self.assert_nested_equivalent(a[key],b[key])
        elif isinstance(a,float):self.assertAlmostEqual(a,b,places=12)
        else:self.assertEqual(a,b)

    def test_track_identity_follows_hypothesis_pairs(self):
        arrays={'G_hypothesis_ids':np.array([0,1,2,3,4,5])}
        audit={'objects':[{'support_track':0},{'support_track':4},{'support_track':8}]}
        np.testing.assert_array_equal(ev.tracks_for_points(arrays,'G',audit),[0,0,4,4,8,8])

    def test_existing_pooling_track_subset_reproduces_object_attribution(self):
        layout,paid='shelf_left',24
        e=ev.Evaluator();arrays,receipt=e.baseline(layout,paid)
        observed=ev.ROOT/'replay/observed_prefixes'/f'{layout}_{paid}'/'acquired_points.npz'
        with np.load(observed,allow_pickle=False)as source:
            ix=source['model_indices']
            e.support_arrays={'quality_points':source['all_point'][ix],'quality_normals':source['all_normal'][ix],
                'quality_labels':source['all_label'][ix],'quality_support_track':source['all_support_track'][ix]}
        e.support_metadata={'cropped':dict(point_key='quality_points',normal_key='quality_normals',label_key='quality_labels',track_key='quality_support_track')}
        manifest=json.loads((ev.V1/'replay'/f'{layout}_{paid}_attempt_001'/'manifest.json').read_text())
        row=copy.deepcopy(next(r for r in manifest['stages']if r['variant']=='B2'))
        cid=row['candidate_id'];new_arrays={};new_receipt=copy.deepcopy(receipt)
        new_receipt['fixture_only']='evaluator compatibility, not a new model prediction experiment'
        for name in ('G','S'):
            audit=copy.deepcopy(receipt['model_audits'][name]);tracks=ev.tracks_for_points(arrays,name,audit)
            keep=tracks>0;objects=[copy.deepcopy(o)for o in audit['objects']if o['support_track']>0]
            order={o['support_track']:i for i,o in enumerate(objects)}
            for o in objects:
                ids=e.support_arrays['quality_support_track']==o['support_track']
                for field,key,dtype in [('fit_input_points_sha256','quality_points',np.float64),('fit_input_normals_sha256','quality_normals',np.float64),('fit_input_labels_sha256','quality_labels',np.int64)]:
                    o[field]=hashlib.sha256(np.ascontiguousarray(e.support_arrays[key][ids],dtype=dtype).tobytes()).hexdigest()
            audit.update(objects=objects,object_support=True,full_observed_support=False,
                observed_input_point_count=len(e.support_arrays['quality_points']))
            new_receipt['model_audits'][name]=audit
            new_arrays[name+'_points']=arrays[name+'_points'][keep]
            new_arrays[name+'_weights']=arrays[name+'_weights'][keep]
            new_arrays[name+'_hypothesis_ids']=np.array([2*order[int(t)]+int(i)%2 for t,i in zip(tracks[keep],arrays[name+'_hypothesis_ids'][keep])])
            new_arrays[name+f'_gain_{cid}']=arrays[name+f'_gain_{cid}'][keep]
        with tempfile.TemporaryDirectory(prefix='objectfit_eval_fixture_')as d:
            d=Path(d);np.savez_compressed(d/'fixture.npz',**new_arrays)
            (d/'fixture.json').write_text(json.dumps(new_receipt))
            row.update(variant='object_support',fit_support_basis='cropped',
                predictions=dict(path=str(d/'fixture.npz'),sha256=ev.sha(d/'fixture.npz')),
                prediction_receipt=dict(path=str(d/'fixture.json'),sha256=ev.sha(d/'fixture.json')))
            result=e.stage(row,layout)
            for name in ('G','S'):
                x=result['observed_object_attribution'][name]
                self.assertEqual(x['admitted_tracks'],[1,2])
                self.assertFalse(x['lost_baseline_tracks'])
                # Keeping old zero-weight background slots changes floating
                # summation order only, not supported mass or classifications.
                self.assert_nested_equivalent(x['current'],x['legacy_pooling_same_track_baseline'])
                self.assertTrue(all(all(v.values())for v in x['same_track_forecast_parity'].values()))
            # Input provenance mismatches must be rejected before interpreting
            # support rates as changed fitting quality.
            broken=copy.deepcopy(new_receipt['model_audits']['S'])
            broken['objects'][0]['fit_input_points_sha256']='0'*64
            e.support_cache.clear()
            with self.assertRaises(AssertionError):e.validate_support(row,'S',broken)

if __name__=='__main__':unittest.main()
