"""Full evaluator interface test with synthetic tables, no rendered study inputs."""
import json,subprocess,sys,tempfile,unittest
from pathlib import Path
from test_calibration_isolation import write_resource,prediction
HERE=Path(__file__).resolve().parent

class PipelineTest(unittest.TestCase):
    def test_scale_freezes_before_untouched_split_evaluation(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);good=write_resource(p/'prediction.json',prediction())
            bad_data=prediction();bad_data['tracks']*=2;bad=write_resource(p/'failed.json',bad_data)
            objects=[];cases=[];index=[]
            for i in range(24):
                split='train' if i<12 else 'validation' if i<18 else 'heldout';oid=f'obj{i}'
                truth=write_resource(p/(oid+'.json'),dict(object_id=oid,object_vertices_world=[[0,0,0],[1,.5,1]],world_rotation=[[1,0,0],[0,1,0],[0,0,1]]))
                ids=[]
                for condition in ('front','front_side','around','occluded'):
                    cid=oid+'_'+condition;ids.append(cid)
                    cases.append(dict(case_id=cid,object_id=oid,split=split,condition=condition,truth=truth))
                    index.append(dict(case_id=cid,status='completed',prediction=bad if i==0 and condition=='front' else good))
                objects.append(dict(object_id=oid,split=split,cases=ids))
            ds=p/'dataset.json';ds.write_text(json.dumps(dict(objects=objects,cases=cases)))
            ix=p/'index.json';ix.write_text(json.dumps(dict(status='completed_isolated_banks',pending_case_ids=[],cases=index)))
            cmd=[sys.executable,str(HERE/'evaluate_calibration.py')]
            a=subprocess.run(cmd+['calibrate','--dataset',str(ds),'--predictions',str(ix),'--output',str(p/'train')],capture_output=True,text=True)
            self.assertEqual(a.returncode,0,a.stderr)
            selected=json.loads((p/'train/selected_scale.json').read_text())
            self.assertEqual(selected['sigma_m'],.22);self.assertEqual(selected['fallback_case_count'],1)
            training=json.loads((p/'train/training_scores.json').read_text())
            expected={str(p/(f'obj{i}.json')) for i in range(12)}
            self.assertEqual(set(training['opened_truth_paths']),expected)
            self.assertEqual(len(training['scales']),8)
            b=subprocess.run(cmd+['evaluate','--dataset',str(ds),'--predictions',str(ix),'--selected-scale',str(p/'train/selected_scale.json'),'--output',str(p/'test')],capture_output=True,text=True)
            self.assertEqual(b.returncode,0,b.stderr)
            assessment=json.loads((p/'test/assessment.json').read_text());self.assertFalse(assessment['distribution_metric_criterion_passes'])
            for split in ('validation','heldout'):
                result=json.loads((p/'test'/f'{split}.json').read_text());self.assertEqual(result['evaluable_cases'],24)
                self.assertEqual(result['summary']['posterior']['all_conditions']['objects'],6)
                self.assertEqual(result['summary']['posterior']['informative']['cases'],12)

if __name__=='__main__':unittest.main(verbosity=2)
