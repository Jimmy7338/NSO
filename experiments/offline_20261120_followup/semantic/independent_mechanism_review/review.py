"""Read-only second audit; reuse frozen native distances, no resampling or actors."""
from pathlib import Path
import hashlib,json,importlib.util,csv
import numpy as np
HERE=Path(__file__).resolve().parent;SRC=HERE.parent/'mechanism_v2_meshfix';B9=HERE.parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def resource(p):return dict(path=str(p),sha256=sha(p))
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def verify_freeze(path):
 x=read(path)
 for p,h in x['files_sha256'].items():assert sha(p)==h,p
 return resource(path)
def loadresource(x):assert sha(x['path'])==x['sha256'];return read(x['path'])
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def assertclose(x,y):np.testing.assert_allclose(x,y,atol=1e-11,rtol=1e-12)
def main():
 summary=read(SRC/'audit_attempt_001/summary.json');assert summary['actual_branches']==8 and summary['actual_actions']==114 and len(summary['conditions'])==12
 freezes=[verify_freeze(SRC/n)for n in ('scene_freeze.json','candidate_extraction_freeze_attempt002.json','audit_attempt_001/evaluation_freeze.json')]
 branches={(x['layout'],x['candidate_id']):x for x in summary['branches']};actual=[];rows=[];layoutrows={}
 for layout in ('shelf_left','shelf_right'):
  folder=SRC/'candidate_attempt_002'/layout;original=read(folder/'original_plan.json');routes=read(folder/'fixed_candidates.json')
  assert routes==original['candidates'] and len(routes)==4
  freezes.append(verify_freeze(SRC/'branches_attempt_001'/layout/'freeze.json'))
  manifest=read(SRC/'branches_attempt_001'/layout/'manifest.json');assert manifest['all_candidates_complete']
  ref=read(SRC/'audit_attempt_001'/layout/'reference_receipt.json');assert not ref['observed_actor_trajectory_used'] and ref['source_geometry_only_in_evaluator'] and ref['threshold_m']==.05
  for stage in manifest['stages']:
   for key in ('before_mesh','after_mesh'):assert sha(stage[key]['path'])==stage[key]['sha256']
   e=loadresource(stage['execution']);r=branches[(layout,stage['candidate_id'])];route=next(x for x in routes if x['candidate_id']==r['candidate_id'])
   assert len(e['steps'])==e['completed_actions']==len(route['outbound_actions'])==r['completed_actions']
   assert [x['action']for x in e['steps']]==route['outbound_actions']
   assert [x['pose']for x in e['steps']]==route['outbound_states'][1:]
   assert e['endpoint_reached']==r['endpoint_reached'] and e['collisions']==r['collisions']==0
   assert e['prefix_mesh_byte_identical'] and not e['predicted_geometry_fused'] and not e['source_geometry_supplied_to_mapper']
   for name,m in r['surface'].items():
    assertclose(m['after_correct_area_m2']-m['before_correct_area_m2'],m['net_correct_area_m2'])
    assertclose(m['new_correct_area_m2']-m['lost_correct_area_m2'],m['net_correct_area_m2'])
    assertclose(m['after_correct_area_m2']/ref['metadata'][name]['visible_area_m2'],m['final_recall'])
   assertclose(r['object_net_area_m2'],sum(r['surface'][name]['net_correct_area_m2']for name in ('left','right')))
   assertclose(r['object_new_area_m2'],sum(r['surface'][name]['new_correct_area_m2']for name in ('left','right')))
   assertclose(r['global_net_area_m2'],r['surface']['all']['net_correct_area_m2'])
   for name,key in [('correct_free_coverage_fraction','correct_free_coverage_increment'),('coverage_fraction','known_coverage_increment')]:
    assertclose(r[key],e['final_coverage'][name]-e['initial_coverage'][name])
   assert e['path_length_m']==r['path_length_m']
   actual.append(dict(layout=layout,candidate_id=r['candidate_id'],execution=stage['execution'],native_mesh=stage['after_mesh'],metrics_verified=True))
  layoutconditions=[];basis=None;common=None;observed=None;bank=None
  for T in (1,16):
   for cue in ('correct','neutral','swapped'):
    path=folder/f'T{T}_{cue}_predictions.json';r=read(path)
    assert r['candidate_sha256']==digest(routes) and r['common_score_sha256']==digest(original['scores']['N'])
    assert r['simulator_actions']==0 and not r['source_truth_to_prediction']
    if observed is None:observed=r['observed_input_sha256']
    assert r['observed_input_sha256']==observed
    arrays=np.load(path.with_suffix('.npz'),allow_pickle=False)
    current={key:arrays['G_'+key]for key in ('points','object_ids','particle_ids','shape_ids')}
    for key in current:np.testing.assert_array_equal(current[key],arrays['S_'+key])
    if basis is None:basis=current
    else:
     for key in current:np.testing.assert_array_equal(current[key],basis[key])
    for name in ('G','S'):
     objs=r['model_audits'][name]['objects'];shared=[{k:o[k]for k in ('support_track','particles','prior_mass','energy_m2','conditional_geometry','fit_input_points_sha256','fit_input_normals_sha256')}for o in objs]
     if bank is None:bank=shared
     else:assert shared==bank
     mat=arrays[name+'_route_union_gain_matrix'];gains=mat.sum(axis=1)
     for i,c in enumerate(r['candidates']):
      assert c['candidate_id']==routes[i]['candidate_id'] and c['pose']==routes[i]['pose'] and c['cost']==routes[i]['cost']
      assert c['outbound_actions']==len(routes[i]['outbound_actions'])
      assertclose(c['common'],original['scores']['N'][i]);assertclose(c[name+'_gain'],gains[i]);assertclose(c[name+'_total'],c['common']+gains[i])
     chosen=min(r['candidates'],key=lambda c:(-c[name+'_total'],c['cost'],c['candidate_id']))['candidate_id'];assert chosen==r[name+'_winner']
    if cue=='neutral':np.testing.assert_array_equal(arrays['G_weights'],arrays['S_weights'])
    c=next(x for x in summary['conditions']if x['layout']==layout and x['temperature']==T and x['cue']==cue)
    assert c['prediction_receipt_sha256']==sha(path)
    g=branches[(layout,r['G_winner'])];s=branches[(layout,r['S_winner'])]
    for name,b in [('G',g),('S',s)]:
     assert c['selected'][name]['candidate_id']==r[name+'_winner']
     for key in ('object_net_area_m2','object_new_area_m2','global_net_area_m2','completed_actions','path_length_m','correct_free_coverage_increment','endpoint_reached','collisions'):assertclose(c['selected'][name][key],b[key])
    row=dict(layout=layout,temperature=T,cue=cue,G_candidate=r['G_winner'],S_candidate=r['S_winner'],target_changed=r['G_winner']!=r['S_winner'],
      object_net_delta_m2=s['object_net_area_m2']-g['object_net_area_m2'],object_new_delta_m2=s['object_new_area_m2']-g['object_new_area_m2'],
      global_net_delta_m2=s['global_net_area_m2']-g['global_net_area_m2'],global_new_delta_m2=s['surface']['all']['new_correct_area_m2']-g['surface']['all']['new_correct_area_m2'],
      correct_free_delta_pp=100*(s['correct_free_coverage_increment']-g['correct_free_coverage_increment']),
      known_coverage_delta_pp=100*(s['known_coverage_increment']-g['known_coverage_increment']),actions_delta=s['completed_actions']-g['completed_actions'],path_delta_m=s['path_length_m']-g['path_length_m'])
    rows.append(row);layoutconditions.append(dict(prediction=resource(path),arrays=resource(path.with_suffix('.npz'))))
  layoutrows[layout]=dict(candidate_sha256=digest(routes),common_score_sha256=digest(original['scores']['N']),observed_input_sha256=observed,within_layout_common_input_and_bank_exact=True,conditions=layoutconditions)
 assert len(actual)==8 and sum(x['completed_actions']for x in branches.values())==114
 add=read(SRC/'render_precision_addendum.json');assert add['maximum_local_float32_ULPs']==2 and add['observed_max_depth_difference_m']==4.76837158203125e-07
 assert layoutrows['shelf_left']['candidate_sha256']==layoutrows['shelf_right']['candidate_sha256']
 assert layoutrows['shelf_left']['common_score_sha256']!=layoutrows['shelf_right']['common_score_sha256']
 output=dict(status='passed_independent_receipt_and_formula_review',source_summary=resource(SRC/'audit_attempt_001/summary.json'),frozen_sources=freezes,
  all_actual_branch_receipts=actual,layouts=layoutrows,all_twelve_conditions=rows,actual_branches=8,actual_actions=114,all_endpoints_reached=True,simulated_collisions=0,
  expensive_source_surface_sampling_repeated=False,original_outcomes_and_source_hashes_verified=True,
  limits=['Only 8 actual fixed branches underlie all 12 condition rows; cue/temperature interventions are dependent counterfactual scoring, not 12 autonomous trials.',
   'All four candidates per layout are original cold-start NSO candidates; all original outbound routes were executed. No manual target insertion or winner-based branch selection.',
   'Within each layout observations, shape likelihood, point basis and candidate/common scores are exact across cues. Between layouts first-hit depth differs by up to2float32ULPs and common-score hashes differ; do not describe cross-layout input as byte-identical.',
   'Native TSDF surface metrics use independently fixed visible source samples. Object new/net area is the sum for two objects; global new/net also includes room/screen. Net subtracts regression; gross-new and net must not be conflated.',
   'T1 correct S loses object surface at left while gaining global surface. T16 correct S helps object surface only at right, while losing global surface but gaining correct-free coverage. Retain all twelve rows, including swapped-cue positive object outcomes.',
   'Initial known fraction~78.7%, one acquisition state, procedural category cues and exact/noiseless sensing: mechanism evidence only, not complete unknown exploration, natural recognition or robot results.'],code=resource(Path(__file__)))
 write(HERE/'review.json',output)
 with (HERE/'all_conditions.csv').open('w',newline='')as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 print(json.dumps(dict(status=output['status'],review=resource(HERE/'review.json'),rows=rows)))
if __name__=='__main__':main()
