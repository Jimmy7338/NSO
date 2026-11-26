"""Audit all T16 fixed-prefix predictions against sealed B8 arrays/outcomes.

No source geometry queries, sensor actions, visibility recomputation or tuning.
"""
from pathlib import Path
import hashlib,json,math,csv,importlib.util
import numpy as np
from scipy.stats import spearmanr
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
B8=ROOT.parents[1]/'dual_template_uncertainty_20261120'
STATES={'shelf_left':[24,54,109],'shelf_right':[24,53,110]}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def resource(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def load(ref):
 assert sha(ref['path'])==ref['sha256'],ref['path']
 return read(ref['path'])
def write(p,d):Path(p).write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
def rho(x,y):
 if len(x)<2 or len(set(x))<2 or len(set(y))<2:return None
 return float(spearmanr(x,y).statistic)
def independent_joint(obj,T):
 q=np.asarray(obj['prior_mass']);e=np.asarray(obj['energy_m2']);shape=np.asarray(obj['shape_prior'])
 with np.errstate(divide='ignore'):a=np.log(q[:,None])-e/(2*.01**2)
 maximum=a.max(axis=0);v=np.exp(a-maximum);s=v.sum(axis=0);conditional=v/s
 z=maximum+np.log(s);logit=(z[1]-z[0])/T+np.log(shape[1]/shape[0])
 p=np.array([np.exp(-np.logaddexp(0.,logit)),np.exp(-np.logaddexp(0.,-logit))])
 return conditional*p,conditional
def winner(rows):return min(rows,key=lambda r:(-r['score'],r['cost'],r['candidate_id']))['candidate_id']

def verify_arrays(new,old,group,objects,oldobjects,T):
 for key in ('points','history_keep','object_ids','particle_ids','hypothesis_ids'):
  np.testing.assert_array_equal(new[key],old[group+'_'+key])
 np.testing.assert_array_equal(new['candidate_ids'],old['candidate_ids'])
 np.testing.assert_array_equal(new['weights'],new['base_weights']*new['history_keep'])
 assert len(objects)==len(oldobjects)
 reconstructed=np.zeros(len(new['points']));oid=new['object_ids'];ki=new['particle_ids'];hi=new['hypothesis_ids']%2
 for i,(obj,previous)in enumerate(zip(objects,oldobjects)):
  for key in ('support_track','particles','prior_mass','energy_m2','shape_prior','conditional_geometry','fit_input_points_sha256','fit_input_normals_sha256','fit_input_labels_sha256'):
   assert obj[key]==previous[key],key
  joint,conditional=independent_joint(obj,T);oldjoint,oldconditional=independent_joint(previous,1.)
  np.testing.assert_allclose(conditional,oldconditional,rtol=0,atol=0)
  np.testing.assert_allclose(joint,obj['joint_probability_mass'],rtol=0,atol=1e-13)
  np.testing.assert_allclose(np.asarray(obj['joint_probability_mass']).sum(axis=0),obj['shape_probability_mass'],rtol=0,atol=1e-13)
  mask=oid==i;previous_mass=np.asarray(previous['joint_probability_mass'])[ki[mask],hi[mask]]
  # All actual frozen B8 particles have positive base mass. No history-zero
  # point is removed from this common external quadrature.
  assert np.all(previous_mass>0)
  area=old[group+'_base_weights'][mask]/previous_mass
  reconstructed[mask]=area*joint[ki[mask],hi[mask]]
 np.testing.assert_allclose(new['base_weights'],reconstructed,rtol=2e-12,atol=1e-14)
 matrix=new['route_union_gain_matrix'];oldmatrix=old[group+'_route_union_gain_matrix']
 assert matrix.shape==oldmatrix.shape and matrix.shape[1]==len(new['points'])
 assert np.isfinite(matrix).all()and np.all(matrix>=0)
 active=new['weights']>0;oldactive=old[group+'_weights']>0
 zero_to_nonzero=int((active&~oldactive).sum())
 assert zero_to_nonzero==0,'Unexpected newly active basis requires additional geometry visibility verification'
 # Recomputed T16 visibility must agree on identical geometry even though
 # its evaluation did not reuse the T1 gain cache.
 visibility=oldmatrix[:,oldactive]/old[group+'_weights'][oldactive]
 np.testing.assert_allclose(visibility,np.round(visibility),rtol=0,atol=1e-13)
 np.testing.assert_allclose(matrix[:,oldactive],visibility*new['weights'][oldactive],rtol=2e-12,atol=1e-14)
 assert np.all(matrix[:,~active]==0)
 return dict(points=len(new['points']),full_basis_identical=True,conditional_geometry_identical=True,
             posterior_recomputed_independently=True,history_exact=True,area_weights_verified=True,
             newly_active_from_old_zero_weight_points=zero_to_nonzero,
             independently_recomputed_visibility_matches_same_geometry=True)

def evaluate_state(layout,paid,source_state):
 folder=ROOT/'replay_attempt_001'/f'{layout}_{paid}';path=folder/'result.json';r=read(path)
 assert r['status']=='complete_two_arm_temperature_fixed_replay'and r['new_acquisition_actions']==0 and r['temperature']==16.
 assert r['acquired_mapper_unchanged']and r['temperature_one_B8_points_weights_exact']
 oldfolder=B8/'replay'/f'{layout}_{paid}_attempt_001';routes=read(oldfolder/'fixed_candidates.json');manifest=read(oldfolder/'manifest.json')
 assert sha(oldfolder/'fixed_candidates.json')==r['source_candidates_sha256']
 assert sha(oldfolder/'result.json')==r['source_B8_result_sha256']
 assert r['native_mesh_sha256']==read(oldfolder/'prefix_execution.json')['native_mesh_rebuilt_sha256']
 source_eval=load(source_state['evaluation']);del source_eval # verifies original complete evaluation receipt
 routes_by={int(x['candidate_id']):x for x in routes};records={};arrays={};execution_refs={}
 for name in ('G','S'):
  arm=r['arms'][name];ref=arm['arrays'];assert sha(ref['path'])==ref['sha256']
  oldreceipt=read(oldfolder/(name+'_uncertainty_predictions.json'))
  with np.load(ref['path'],allow_pickle=False)as z:arrays[name]={k:z[k]for k in z.files}
  with np.load(oldfolder/(name+'_uncertainty_predictions.npz'),allow_pickle=False)as z:old={k:z[k]for k in z.files}
  audit=verify_arrays(arrays[name],old,name,arm['model_audit']['objects'],oldreceipt['model_audits'][name]['objects'],16.)
  previous=source_state['arms'][name+'_uncertainty'];previous_rows={x['candidate_id']:x for x in previous['all_candidates']}
  assert len(arm['rows'])==len(routes)==len(previous_rows)
  output=[];reference_stages={x['candidate_id']:x for x in manifest['stages']if x['variant']==name+'_uncertainty'}
  for i,row in enumerate(arm['rows']):
   cid=row['candidate_id'];prior=previous_rows[cid];route=routes_by[cid]
   assert cid==int(arrays[name]['candidate_ids'][i]);assert row['cost']==route['cost']
   assert row['outbound_actions']==len(route['outbound_actions'])
   np.testing.assert_allclose(row['common_total_proxy'],prior['common_score'],rtol=0,atol=1e-12)
   gain=float(arrays[name]['route_union_gain_matrix'][i].sum())
   np.testing.assert_allclose(gain,row['template_gain_m2'],rtol=0,atol=1e-12)
   np.testing.assert_allclose(row['score'],row['common_total_proxy']+gain,rtol=0,atol=1e-12)
   ref=reference_stages[cid]['execution_receipt'];execution=load(ref);execution_refs[ref['path']]=ref
   actual=prior['actual'];assert actual==source_state['arms']['G_uncertainty']['all_candidates'][i]['actual']
   for key in ('completed_actions','planned_actions','path_length_m','endpoint_reached','simulated_collisions'):
    assert actual[key]==execution[key],key
   coverage=100*(execution['final_coverage']['coverage_fraction']-execution['initial_coverage']['coverage_fraction'])
   np.testing.assert_allclose(coverage,actual['coverage_delta_percentage_points'],rtol=0,atol=1e-12)
   output.append(dict(candidate_id=cid,old_template_gain_m2=prior['template_score'],new_template_gain_m2=gain,
                      old_total_score=prior['total_score'],new_total_score=row['score'],common_score=prior['common_score'],
                      cost=row['cost'],actual=actual,execution_receipt=ref))
  newwinner=winner(arm['rows']);assert newwinner==arm['selected_candidate']
  oldscored=[dict(candidate_id=x['candidate_id'],score=x['total_score'],cost=routes_by[x['candidate_id']]['cost'])for x in previous['all_candidates']]
  oldwinner=winner(oldscored);assert oldwinner==previous['selected']['candidate_id']
  chosen={x['candidate_id']:x for x in output};a=chosen[oldwinner]['actual'];b=chosen[newwinner]['actual']
  deltas={key:b[key]-a[key]for key in ('actual_object_gross_new_m2','actual_object_lost_m2','actual_object_net_new_m2','actual_global_net_new_m2','coverage_delta_percentage_points','completed_actions','path_length_m')}
  records[name]=dict(old_selected_candidate=oldwinner,new_selected_candidate=newwinner,target_changed=oldwinner!=newwinner,
                     old_selected_actual=a,new_selected_actual=b,selected_actual_delta= deltas,
                     old_template_rank_rho_vs_object_net=previous['template_rank_rho_vs_object_net'],
                     new_template_rank_rho_vs_object_net=rho([x['new_template_gain_m2']for x in output],[x['actual']['actual_object_net_new_m2']for x in output]),
                     new_total_rank_rho_vs_object_net=rho([x['new_total_score']for x in output],[x['actual']['actual_object_net_new_m2']for x in output]),
                     arrays_audit=audit,all_candidates=output)
 for key in ('points','history_keep','object_ids','particle_ids','hypothesis_ids','candidate_ids'):
  np.testing.assert_array_equal(arrays['G'][key],arrays['S'][key])
 return dict(layout=layout,state_action=paid,candidates=len(routes),arms=records,G_S_new_winner_differs=records['G']['new_selected_candidate']!=records['S']['new_selected_candidate'],
             G_S_basis_identical=True,source_result=resource(path),old_source_evaluation=source_state['evaluation'],actual_execution_receipts=list(execution_refs.values()))

def main():
 freeze=read(HERE/'freeze.json')
 for path,h in freeze['files_sha256'].items():assert sha(path)==h,path
 parent=read(ROOT/'replay_freeze.json')
 for path,h in parent['files_sha256'].items():assert sha(path)==h,path
 original=read(B8/'audit/particle_replay/summary.json');lookup={(r['layout'],r['state_action']):r for r in original['states']}
 results=[]
 for layout,states in STATES.items():
  for paid in states:
   results.append(evaluate_state(layout,paid,lookup[(layout,paid)]))
 assert sum(r['candidates']for r in results)==66
 changes=sum(arm['target_changed']for r in results for arm in r['arms'].values())
 readiness=dict(status='completed_independent_T16_fixed_replay_audit',eligible_for_predeclared_four_policy_runs=changes>0,
                complete_prefixes=6,reused_actual_branches=66,new_prediction_candidate_rows=132,
                actual_selected_target_changes_vs_T1=changes,new_acquisition_actions=0,all_implementation_checks_pass=True,
                positive_branch_gain_required=False,G_S_new_winner_differences=sum(r['G_S_new_winner_differs']for r in results),
                conditions='Historical six development prefixes; actual branch endpoints remain reused evidence, not new independent trials.',
                code_sha256=freeze['files_sha256'])
 summary=dict(**readiness,states=results,source_B8_summary=resource(B8/'audit/particle_replay/summary.json'),
              selected_temperature=resource(ROOT/'audit/calibration_attempt_001/selected_temperature.json'),
              interpretation='Complete source/array/history/common-score audit; actual signed outcomes preserved. No new source-surface matching run or new robot/simulator branch acquisition.')
 write(HERE/'summary.json',summary);write(HERE/'readiness.json',dict(**readiness,summary=resource(HERE/'summary.json')))
 rows=[]
 for state in results:
  for name,a in state['arms'].items():
   rows.append(dict(layout=state['layout'],state_action=state['state_action'],arm=name,
                    old_selected=a['old_selected_candidate'],new_selected=a['new_selected_candidate'],
                    target_changed=a['target_changed'],old_rank=a['old_template_rank_rho_vs_object_net'],new_rank=a['new_template_rank_rho_vs_object_net'],**a['selected_actual_delta']))
 with (HERE/'results.csv').open('w',newline='')as f:
  writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
 print(json.dumps(readiness))

if __name__=='__main__':main()
