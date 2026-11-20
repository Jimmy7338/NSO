"""Predeclared three-route paired acquisition comparisons on fixed prefixes."""
from pathlib import Path
import argparse,copy,hashlib,json,os,sys,time,traceback
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import numpy as np
import open3d as o3d
import runtime_binding as R
from full_posterior_forecast import FullPosteriorForecast,digest
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent

def cost_band(routes,width=3):
 """Choose on route costs alone, before any gain or actual outcome is read."""
 counts={int(r['candidate_id']):len(r['outbound_actions']) for r in routes}
 possible=[]
 for low in sorted(set(counts.values())):
  ids=sorted(cid for cid,n in counts.items()if low<=n<=low+width)
  possible.append(((-len(ids),low,tuple(ids)),ids))
 if not possible:return [],dict(status='no_native_routes',all_outbound_action_costs=counts)
 _,ids=min(possible);low=min(counts[c]for c in ids)
 audit=dict(status='eligible'if len(ids)>=3 else'insufficient_cost_matched_routes',width_actions=width,lower_actions=low,upper_actions=low+width,selected_pool_ids=ids,all_outbound_action_costs=counts,selection_used_predictions=False,selection_used_future_outcomes=False,tie_rule='largest membership, lower band, ordered IDs')
 return [r for r in routes if int(r['candidate_id'])in ids],audit

def winner(rows):return min(rows,key=lambda r:(-r['score'],r['cost'],r['candidate_id']))['candidate_id']
def final_three(pool_predictions):
 a=winner(pool_predictions['visibility_S']);b=winner(pool_predictions['acquisition_S'])
 ids=sorted({a,b});remaining=sorted(r['candidate_id']for r in pool_predictions['visibility_S']if r['candidate_id']not in ids)
 return sorted(ids+remaining[:3-len(ids)])

def verify():
 freeze=json.loads((HERE/'freeze.json').read_text())
 for path,h in freeze['files_sha256'].items():assert R.sha(path)==h,path
 assert R.protocol()['status']=='frozen_before_B11_predictions_and_policy_runs'
 return freeze

def mesh_resource(mapper,path):
 assert o3d.io.write_triangle_mesh(str(path),mapper.mesh(),write_ascii=False)
 return R.resource(path)

def branch(context,route,path,before):
 from nso.observed_runtime_mapper_v10 import ObservedRuntimeMapperV10
 from nso.cpu_sensor_contract_v10 import digest as jsondigest
 path.mkdir();original=context['world'];stamp=(tuple(original.position),original.heading,original.step_count,original.moves,original.collisions)
 world=copy.copy(original);mapper=ObservedRuntimeMapperV10(world.shape,world.config)
 for p in context['packets']:mapper.update(p.frame,p.scan)
 clone=mesh_resource(mapper,path/'prefix_clone.ply');assert clone['sha256']==before['sha256'];(path/'prefix_clone.ply').unlink()
 np.testing.assert_array_equal(mapper.belief,context['mapper'].belief)
 steps=[];started=time.monotonic();initial=R.old.base.evaluate(mapper.belief,world.reachable)
 assert list(route['outbound_states'][0])==[*world.position,world.heading]
 for action,expected in zip(route['outbound_actions'],route['outbound_states'][1:]):
  frame,collision,done=world.step(action);p=R.old.packet(world,context['transform'],frame,context['layout'],action,collision,done);mapper.update(frame,p.scan)
  pose=[*world.position,world.heading];steps.append(dict(action=action,pose=pose,packet_sha256=p.sha256(),collision=bool(collision),pose_matches_expected=pose==list(expected)))
  if collision or pose!=list(expected):break
 after=mesh_resource(mapper,path/'native_tsdf_mesh.ply');endpoint=[*world.position,world.heading]
 ok=len(steps)==len(route['outbound_actions'])and endpoint==list(route['outbound_states'][-1])and not any(s['collision']for s in steps)
 assert stamp==(tuple(original.position),original.heading,original.step_count,original.moves,original.collisions)
 receipt=dict(status='completed'if ok else'failed_fixed_outbound',diagnostic_only=True,closed_loop_policy_trial=False,candidate_id=int(route['candidate_id']),completed_actions=len(steps),planned_actions=len(route['outbound_actions']),initial_paid_action=original.step_count,branch_actions=steps,endpoint_pose=endpoint,expected_endpoint_pose=list(route['outbound_states'][-1]),endpoint_reached=ok,simulated_collisions=sum(s['collision']for s in steps),initial_coverage=initial,final_coverage=R.old.base.evaluate(mapper.belief,world.reachable),path_length_m=(world.moves-original.moves)*world.config.resolution_m,prefix_mesh_sha256=clone['sha256'],prefix_mesh_byte_identical=True,original_world_pose_counters_unchanged=True,branch_belief_sha256=jsondigest(mapper.belief),no_predicted_geometry_fused=True,elapsed_s=time.monotonic()-started)
 R.write(path/'execution.json',receipt)
 return dict(candidate_id=int(route['candidate_id']),execution=R.resource(path/'execution.json'),after_mesh=after)

def run(layout,prefix_id,out):
 verify();out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
 try:
  context=R.bootstrap_scripted_prefix(layout,prefix_id);mapper=context['mapper']
  routes,common,candidate_audit=R.native_candidates(context);pool,band=cost_band(routes,R.protocol()['short_cost_band_width_actions'])
  before=mesh_resource(mapper,out/'before_native.ply')
  R.write(out/'all_original_candidates.json',dict(routes=routes,common=common,candidate_audit=candidate_audit,cost_band=band))
  prefix=dict(layout=layout,prefix_id=prefix_id,externally_scripted_not_policy=True,prefix_preparation=context['prefix_preparation'],prefix_override_applied=context['prefix_override'] is not None,paid_actions=len(context['scripted_actions']),actions=context['scripted_actions'],packet_sha256=[p.sha256()for p in context['packets']],owner_receipts=context['owners'],before_mesh=before,final_pose=[*context['world'].position,context['world'].heading],frames=mapper.frames,source_geometry_given_to_predictor=False)
  R.write(out/'prefix.json',prefix)
  if len(pool)<3:
   R.write(out/'failure.json',dict(status='preparation_failure_cost_pool',cost_band=band,scene_replaced=False,prefix_replaced=False));return
  # The full original candidate set and cost-only selection are persisted first.
  cache=R.old.method._cache(mapper,'S_visibility');models=cache['models']
  engine=FullPosteriorForecast(mapper,models,mapper._b11_acquired_frames)
  pool_predictions={f'{mode}_{arm}':[]for mode in ('visibility','acquisition')for arm in ('G','S')};hyp_results=[]
  by_common={int(r['candidate_id']):c for r,c in zip(routes,common)}
  for route in pool:
   cid=int(route['candidate_id']);states=route['outbound_states'][1:];visibility=engine.visibility(states);acquisition=engine.acquisition(states)
   hyp_results.append(dict(candidate_id=cid,**acquisition));c=float(by_common[cid]['common_total_proxy'])
   for mode,values in [('visibility',visibility),('acquisition',acquisition['gross_m2'])]:
    for arm in ('G','S'):
     pool_predictions[f'{mode}_{arm}'].append(dict(candidate_id=cid,predicted_gain_m2=values[arm],common_score=c,score=c+values[arm],cost=route['cost'],outbound_actions=len(route['outbound_actions'])))
   R.write(out/'prediction_progress.json',dict(completed_pool_candidates=len(hyp_results),pool_candidates=len(pool),last_candidate=cid,elapsed_s=time.monotonic()-started))
  selected=final_three(pool_predictions);conditions={}
  for name,rows in pool_predictions.items():
   final=sorted([r for r in rows if r['candidate_id']in selected],key=lambda r:r['candidate_id']);conditions[name]=dict(rows=final,selected_candidate=winner(final))
  binding=engine.assert_unchanged();binding.update(points_sha256=digest(models['G'].points),G_base_weights_sha256=digest(models['G'].base_weights),S_base_weights_sha256=digest(models['S'].base_weights),history_keep_sha256=digest(models['G'].history_keep),shared_full_posterior=True,shared_original_022_quadrature=True,sigma_m=.01,structural_temperature=16.)
  R.write(out/'hypothesis_forecasts.json',dict(rows=hyp_results,positive_mass_pruning=False))
  R.write(out/'predictions.json',dict(schema='B11_three_shared_candidates_v1',layout=layout,prefix_id=prefix_id,conditions=conditions,pool_conditions={k:dict(rows=v,selected_candidate=winner(v))for k,v in pool_predictions.items()},selected_candidate_ids=selected,all_original_candidate_ids=[int(r['candidate_id'])for r in routes],cost_band=band,shared_basis=binding,paired_predictors_differ_only_in_acquisition_gain=True,source_truth_read_by_predictor=False))
  # No source or actual future result has been read to choose the fixed three.
  branches=[]
  for cid in selected:
   route=next(r for r in pool if int(r['candidate_id'])==cid);branches.append(branch(context,route,out/f'candidate_{cid}',before))
  verify()
  manifest=dict(status='completed_three_fixed_branches',layout=layout,prefix_id=prefix_id,topology=layout.split('_')[0],predictions=R.resource(out/'predictions.json'),before_mesh=before,prefix=R.resource(out/'prefix.json'),all_original_candidates=R.resource(out/'all_original_candidates.json'),hypothesis_forecasts=R.resource(out/'hypothesis_forecasts.json'),reference_directory=str(ROOT/'scene/evaluator_references'/layout),branches=branches,elapsed_s=time.monotonic()-started,protocol=R.resource(HERE/'protocol.json'),new_complete_policy_runs=0)
  R.write(out/'manifest.json',manifest);print(json.dumps(dict(layout=layout,prefix_id=prefix_id,status=manifest['status'],elapsed_s=manifest['elapsed_s'])),flush=True)
 except BaseException as e:
  R.write(out/'failure.json',dict(status='failed_or_interrupted',error=repr(e),traceback=traceback.format_exc(),elapsed_s=time.monotonic()-started));raise

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--layout',required=True);p.add_argument('--prefix',required=True,choices=('p1','p2'));p.add_argument('--output',required=True,type=Path);a=p.parse_args();run(a.layout,a.prefix,a.output)
