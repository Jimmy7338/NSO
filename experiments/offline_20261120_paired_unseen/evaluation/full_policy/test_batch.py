import copy,hashlib,json,tempfile,unittest
from pathlib import Path
from evaluate_full_run import audit_frames,audit_trace
from evaluate_batch import inventory,summarize,pair,LAYOUTS,ARMS

def row(layout='chain_a',arm='S_visibility'):
 return dict(layout=layout,arm=arm,budget=256,initial_packet_sha256='shared',reference_sha256='r',paid_actions=250,A95=30,A100=None,censoring_actions=250,final_coverage=dict(coverage_fraction=.99,free_correct_fraction=.98,false_occupied_fraction=.01),path_length_m=20.,collisions=0,returned_to_anchor=True,elapsed_s=500.,decision_wall_s_total=400.,surface_metrics=dict(f1_05cm=.7,f1_10cm=.8,precision_05cm=.8,recall_05cm=.65),object_recall_5cm={'open31':.5,'closed118':.6},object_recall_10cm={'open31':.7,'closed118':.8},object_macro_recall_5cm=.55,object_area_weighted_recall_5cm=.56,trajectory_sha256='t',native_mesh_sha256='m')

class TestBatch(unittest.TestCase):
 def test_frame_hash_order_and_complete_paid_count(self):
  frames=[dict(timestamp_s=float(i),sha256=hashlib.sha256(str(i).encode()).hexdigest())for i in range(3)]
  final=dict(mapper_frames=3,registered_frames=3,score_call_count=1,input_sequence_sha256=hashlib.sha256(''.join(f['sha256']for f in frames).encode()).hexdigest())
  self.assertEqual(audit_frames(frames,final,2,1)['actual_frame_count'],3)
  swapped=copy.deepcopy(frames);swapped[1]['sha256'],swapped[2]['sha256']=swapped[2]['sha256'],swapped[1]['sha256']
  with self.assertRaises(AssertionError):audit_frames(swapped,final,2,1)
  with self.assertRaises(AssertionError):audit_frames(frames,final,3,1)
 def test_path_collision_and_wall_time_independently_reduced(self):
  trace=[dict(position=[0,1],collision=False,decision_wall_s=2.),dict(position=[0,1],collision=True,decision_wall_s=.5)]
  r=dict(path_length_m=.2,collisions=1,decision_wall_s_total=2.5)
  self.assertEqual(audit_trace(trace,[0,0,0],r)['collisions_recomputed'],1)
  r['path_length_m']=.4
  with self.assertRaises(AssertionError):audit_trace(trace,[0,0,0],r)
 def test_partial_manifest_without_binding_not_completed(self):
  with tempfile.TemporaryDirectory()as d:
   p=Path(d)/'attempt_001';p.mkdir();(p/'result.json').write_text(json.dumps({'status':'completed_execution','paid_actions':256}));(p/'artifact_manifest.json').write_text(json.dumps({'files_sha256':{}}))
   self.assertEqual(inventory(Path(d))[0]['status'],'partial_or_running')
   (p/'failure.json').write_text('{}');self.assertEqual(inventory(Path(d))[0]['status'],'failed_or_interrupted')
 def test_other_attempts_retained_not_selected_implicitly(self):
  with tempfile.TemporaryDirectory()as d:
   for name in ('attempt_001','attempt_002'):
    p=Path(d)/name;p.mkdir();(p/'failure.json').write_text('{}')
   self.assertEqual(len(inventory(Path(d))),2)
 def test_pair_preserves_negative_outcomes_and_censoring(self):
  old=row();new=row(arm='S_acquisition');new['object_recall_5cm']['open31']=.3;new['A95']=40;new['A100']=240
  x=pair(old,new);self.assertAlmostEqual(x['deltas']['open31_recall_5cm'],-.2);self.assertEqual(x['deltas']['A95'],10);self.assertIsNone(x['deltas']['A100']);self.assertEqual(x['A100_pair'],[None,240])
 def test_pair_rejects_different_prefix_reference_or_budget(self):
  for key,value in [('initial_packet_sha256','other'),('reference_sha256','other'),('budget',257)]:
   old=row();new=row();new[key]=value
   with self.assertRaises(AssertionError):pair(old,new)
 def test_eight_arms_aggregate_as_four_layout_pairs(self):
  rows={l:{a:row(l,a) for a in ARMS}for l in LAYOUTS};x=summarize(rows,{},[],{})
  self.assertEqual(x['design']['completed_runs'],8);self.assertEqual(len(x['paired_differences']),4);self.assertEqual(x['paired_equal_layout_means']['A95']['complete_pairs'],4)
  self.assertEqual(x['status'],'completed_independent_eight_run_evaluation')
 def test_failure_retained_full_denominator_and_no_pair_imputation(self):
  rows={'chain_a':{'S_visibility':row()}};failure={'chain_a/S_acquisition':dict(error='failed')};x=summarize(rows,failure,[],{})
  self.assertEqual(len(x['table']),8);self.assertEqual(len(x['paired_differences']),0);self.assertEqual(x['status'],'completed_with_failures');self.assertEqual(len(x['failures']),1)
if __name__=='__main__':unittest.main()
