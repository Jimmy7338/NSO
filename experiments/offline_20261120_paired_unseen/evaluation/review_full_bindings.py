"""Second independent numeric binding review of the declared complete matrix."""
from pathlib import Path
import argparse,hashlib,json,time
import numpy as np
from evaluate_state import Evaluator,read,write,resource,sha
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
LAYOUTS=('chain_a','cycle_a','chain_b','cycle_b');ARMS=('S_visibility','S_acquisition')
def main():
 p=argparse.ArgumentParser();p.add_argument('--summary',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--wait',action='store_true');a=p.parse_args()
 while True:
  d=read(a.summary)
  if not d['pending']:break
  if not a.wait:raise RuntimeError('Full matrix is still pending')
  time.sleep(15)
 a.output.mkdir(parents=True,exist_ok=False);e=Evaluator();review=[];source_hash=sha(a.summary)
 for layout in LAYOUTS:
  for arm in ARMS:
   row=d['rows'].get(layout,{}).get(arm)
   if row is None:
    assert f'{layout}/{arm}'in d['failures'];review.append(dict(layout=layout,arm=arm,status='declared_failure_preserved',failure=d['failures'][f'{layout}/{arm}']));continue
   path=Path(row['attempt']);assert path.name=='attempt_001' and path.parent.name==arm and path.parents[2].name==layout
   for f,h in row['input_sha256'].items():assert sha(f)==h,f
   raw=read(path/'result.json');trace=read(path/'steps.json');initial=read(path/'initial_evaluation.json');prepared=read(ROOT/'execution/layouts'/layout/'prepared/initial_state.json');assert row['initial_packet_sha256']==prepared['initial_packet_sha256']==raw['initial_packet_sha256']
   assert row['budget']==raw['paid_action_budget']==256 and len(trace)==row['paid_actions']==raw['paid_actions']
   assert row['trajectory_sha256']==hashlib.sha256(json.dumps([(x['action'],x['position'],x['heading'])for x in trace],sort_keys=True).encode()).hexdigest()
   assert [x['step']for x in trace]==list(range(1,len(trace)+1))
   for t,key in [(.95,'A95'),(1.,'A100')]:
    hit=0 if initial['free_correct_fraction']>=t else next((x['step']for x in trace if x['free_correct_fraction']>=t),None);assert hit==row[key]==raw['first_free_correct_action'][str(t)]
   with np.load(path/'final_belief.npz',allow_pickle=False)as z:belief=z['belief'];pose=[*map(int,z['position']),int(z['heading'])]
   with np.load(ROOT/'scene/prepared'/layout/'evaluator_truth.npz',allow_pickle=False)as z:reachable=z['reachable']
   free=int(np.count_nonzero((belief==0)&reachable));n=int(reachable.sum());assert free/n==row['final_coverage']['free_correct_fraction']
   assert pose==raw['final_pose'] and (pose==prepared['anchor'])==row['returned_to_anchor']
   reference=e.reference(ROOT/'scene/evaluator_references'/layout);native=e.native(resource(path/'final_tsdf_mesh.ply'),reference);assert sha(path/'final_tsdf_mesh.ply')==row['native_mesh_sha256']
   expected=row['surface_metrics'];actual=native['metrics']
   for key in ['precision_05cm','recall_05cm','f1_05cm','precision_10cm','recall_10cm','f1_10cm']:assert actual[key]==expected[key],(layout,arm,key)
   for obj in ['open31','closed118']:
    recall=float(np.mean(native['distances'][obj]<=np.float32(.05)))
    assert np.isclose(recall,row['object_recall_5cm'][obj],atol=1e-15,rtol=0),(layout,arm,obj)
   review.append(dict(layout=layout,arm=arm,status='independently_rebound',A95=row['A95'],A100=row['A100'],correct_free_fraction=free/n,scene_F1_5cm=actual['f1_05cm'],object_recall_5cm=row['object_recall_5cm'],returned=row['returned_to_anchor'],collisions=row['collisions'],path_m=row['path_length_m'],paid_actions=row['paid_actions'],result_sha256=sha(path/'result.json'),native_sha256=row['native_mesh_sha256']))
 for layout in LAYOUTS:
  pair=[x for x in review if x['layout']==layout and x['status']=='independently_rebound']
  if len(pair)==2:
   old,new=pair;stored=d['paired_differences'][layout]['deltas'];assert np.isclose(new['scene_F1_5cm']-old['scene_F1_5cm'],stored['scene_F1_5cm'],atol=1e-15,rtol=0)
 assert sha(a.summary)==source_hash;e.verify();out=dict(status='independent_full_matrix_numeric_binding_passed',declared_layouts=4,declared_episodes=8,complete_episodes=sum(x['status']=='independently_rebound'for x in review),failed_episodes=sum(x['status']=='declared_failure_preserved'for x in review),rows=review,summary_sha256=source_hash,source_sha256=sha(__file__),frozen_full_evaluator_sha256=sha(HERE/'full_policy/freeze.json'),fresh_native_metrics_repeated=True,new_actor_actions=0,scope='Complete S_visibility/S_acquisition paired matrix; not full G/S exploration ablation',limitations=['Correct-free threshold chronology uses recorded intermediate scalar maps; final belief is independently checked.','Fresh source-reference native metrics verify the published scalar bindings, not robustness to a new reference sample or a real sensor.']);write(a.output/'REVIEW.json',out);print(json.dumps(out))
if __name__=='__main__':main()
