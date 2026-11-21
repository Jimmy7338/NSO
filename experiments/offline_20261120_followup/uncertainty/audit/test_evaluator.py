import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import evaluate_intervals as ev

class EvaluationTests(unittest.TestCase):
    def test_interval_score_known_values(self):
        i={'lower_m':1.,'upper_m':2.,'finite':True}
        self.assertEqual(ev.interval_metrics(i,1.5)['interval_score_m'],1.)
        self.assertEqual(ev.interval_metrics(i,0.)['interval_score_m'],41.)
        self.assertEqual(ev.interval_metrics(i,3.)['interval_score_m'],41.)
        self.assertFalse(ev.interval_metrics(i,0.)['covered'])
    def test_unbounded_width_is_not_finite(self):
        row=ev.interval_metrics({'lower_m':0.,'upper_m':None,'finite':False},900.)
        self.assertTrue(row['covered']); self.assertIsNone(row['width_m']); self.assertIsNone(row['interval_score_m'])
    def test_joint_units_are_objects(self):
        rows=[]
        for obj in ('a','b'):
            for i,c in enumerate(ev.CONDITIONS):
                row={'object_id':obj,'condition':c,'base_interval_metrics':dict(covered=not(obj=='a' and i==3),width_m=1.,interval_score_m=1.,unbounded=False)}
                rows.append(row)
        result=ev.aggregate(rows,'base')
        self.assertEqual(result['objects'],2); self.assertEqual(result['cases'],8)
        self.assertEqual(result['covered_cases'],7); self.assertEqual(result['simultaneous_covered_objects'],1)
        self.assertEqual(result['simultaneous_object_coverage_fraction'],.5)
    def test_split_boundary_resolves_symlink(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); cal=root/'cal.json'; test=root/'test.json'; link=root/'alias.json'
            cal.write_text('{}'); test.write_text('{}'); link.symlink_to(test)
            script="""import sys,json
from pathlib import Path
sys.path.insert(0,sys.argv[1]); import evaluate_intervals as e
cal,test,link=map(Path,sys.argv[2:])
d={'cases':[{'truth':{'path':str(cal)},'split':'calibration'},{'truth':{'path':str(test)},'split':'test'}]}
opened,blocked=e.install_truth_boundary(d,'calibration')
assert cal.read_text()=='{}'
for p in (test,link):
 try: p.read_text(); raise AssertionError('test truth readable')
 except PermissionError: pass
assert len(blocked)==2 and opened=={str(cal.resolve())}
"""
            p=subprocess.run([sys.executable,'-c',script,str(Path(ev.__file__).parent),str(cal),str(test),str(link)],capture_output=True,text=True)
            self.assertEqual(p.returncode,0,p.stderr)
    def test_original_pmf_unchanged_after_application(self):
        cal=dict(schema='B9_group_conformal_interval_v1',prediction_family=ev.conformal.FAMILY,alpha=.05,
                 calibration_objects=24,finite=True,s0_m=.04,q=2.)
        m=ev.dm.metrics([.2,.6,1.4],[.1,.8,.1],.65)
        row=dict(object_id='a',condition='front',metrics=m,target={'projected_true_extent_m':.65})
        out=ev.attach_intervals([row],cal)[0]
        self.assertEqual(out['metrics']['crps_m'],m['crps_m'])
        self.assertEqual(out['intervals']['raw_distribution']['probability_mass'],m['depth_probability_mass'])
        self.assertGreater(out['calibrated_interval_metrics']['width_m'],out['base_interval_metrics']['width_m'])

if __name__=='__main__': unittest.main()
