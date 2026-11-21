"""Integration self-check on frozen old B9 outcomes;not new B11 evidence."""
from pathlib import Path
import json,hashlib
from evaluate_state import Evaluator,resource,write
HERE=Path(__file__).resolve().parent;BASE=HERE.parents[1];OLD=BASE/'dual_template_followup_20261120';B3=BASE/'dual_template_geometry_20261120';B=BASE/'dual_template_exploration_20261119'
def read(p):return json.loads(Path(p).read_text())
def main():
 source=OLD/'structure/replay_audit/summary.json';state=next(x for x in read(source)['states']if x['layout']=='shelf_left'and x['state_action']==24)
 out=HERE/'selfcheck_existing_evidence';out.mkdir(exist_ok=False)
 conditions={}
 for arm in ['G','S']:
  rr=[dict(candidate_id=x['candidate_id'],predicted_gain_m2=x['new_template_gain_m2'],common_score=x['common_score'],score=x['new_total_score'],cost=x['cost'])for x in state['arms'][arm]['all_candidates'][:3]]
  from paired_metrics import winner
  for estimator in ['visibility','acquisition']:
   conditions[estimator+'_'+arm]=dict(rows=rr,selected_candidate=winner(rr),identity_estimator_fixture_only=True)
 write(out/'predictions.json',dict(status='identity_mock_using_old_frozen_predicted_scores_not_a_new_estimator_trial',selected_candidate_ids=[0,1,2],conditions=conditions))
 folder=B3/'replay/shelf_left_24_attempt_001';branches=[]
 for cid in [0,1,2]:
  p=folder/f'candidate_{cid:03d}';branches.append(dict(candidate_id=cid,execution=resource(p/'execution.json'),after_mesh=resource(p/'native_tsdf_mesh.ply')))
 manifest=dict(layout='shelf_left',topology='old_B9_fixture_only',prefix_id='old_p24',predictions=resource(out/'predictions.json'),before_mesh=resource(folder/'prefix_native_tsdf_mesh.ply'),reference_directory=str(B/'audit/references/shelf_left'),branches=branches)
 write(out/'manifest.json',manifest);e=Evaluator();result=e.state(out/'manifest.json');e.verify()
 expected={x['candidate_id']:x['actual']for x in state['arms']['G']['all_candidates']}
 errors={}
 for cid,a in result['actual_by_candidate'].items():
  errors[str(cid)]={}
  for key,oldkey in [('object_gross_new_m2','actual_object_gross_new_m2'),('object_net_new_m2','actual_object_net_new_m2'),('global_net_new_m2','actual_global_net_new_m2')]:
   err=abs(a[key]-expected[cid][oldkey]);assert err<1e-11;(errors[str(cid)]).update({key:err})
 write(out/'receipt.json',dict(status='integration_evaluator_matches_frozen_old_evidence',not_new_B11_evidence=True,old_branch_count=3,areas_absolute_errors=errors,source=resource(source),no_new_actor_actions=True,input_sha256=e.inputs))
 print(json.dumps(dict(status='integration_evaluator_matches_frozen_old_evidence',max_absolute_error=max(v for x in errors.values()for v in x.values()))))
if __name__=='__main__':main()
