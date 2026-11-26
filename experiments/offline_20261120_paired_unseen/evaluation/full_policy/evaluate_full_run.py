"""Independent native-map evaluation for a finalized B11 complete episode.

No simulator/policy import or action execution; truth is evaluator-only.
"""
from pathlib import Path
import argparse,hashlib,json,time,traceback,sys
HERE=Path(__file__).resolve().parent
TOP=HERE.parent
sys.path.append(str(TOP))
import numpy as np
from evaluate_state import Evaluator,read,write,resource,sha,checked
ROOT=TOP.parent

def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()
def first_correct_free_actions(initial,trace):
 assert 0<=initial<=1
 assert [x['step']for x in trace]==list(range(1,len(trace)+1))
 assert all(0<=x['free_correct_fraction']<=1 for x in trace)
 return {str(t):0 if initial>=t else next((x['step']for x in trace if x['free_correct_fraction']>=t),None)for t in (.95,1.)}
def coverage_from_belief(belief,reachable):
 assert belief.shape==reachable.shape and np.isin(belief,[-1,0,1]).all()
 assert reachable.dtype==np.bool_
 denominator=int(reachable.sum());assert denominator>0
 known=int(((belief!=-1)&reachable).sum());free=int(((belief==0)&reachable).sum());false_occupied=int(((belief==1)&reachable).sum())
 return dict(reference_free_cells=denominator,known_cells=known,correctly_free_cells=free,false_occupied_cells=false_occupied,coverage_fraction=known/denominator,free_correct_fraction=free/denominator,false_occupied_fraction=false_occupied/denominator)
def check_models(call):
 assert call['B11_predictor']in ('visibility','acquisition')
 assert call['extent_mode']=='uncertainty'and call['sigma_m']==.01 and call['structural_temperature']==16.
 assert not call['conditional_mean_or_fine_basis']and call['shared_G_S_full_basis']and call['original_candidate_provider']and not call['full_policy_cost_band_applied']
 if call['B11_predictor']=='acquisition'and call['acquisition_diagnostics']is not None:
  a=call['acquisition_diagnostics'];assert a['original_basis_and_posterior_unchanged']and a['all_past_fused']and a['no_positive_probability_pruning']and not a['geometry_collapse']and not a['source_geometry_read']and not a['policy_candidates_changed']
 if call['B11_predictor']=='acquisition'and call['acquisition_diagnostics']is None:assert call['candidate_count']==0
 models=call['model_audits'];identities=[]
 for name in ['G','S']:
  m=models[name];assert m['extent_mode']=='uncertainty'and m['sigma_m']==.01 and m['structural_temperature']==16.
  assert m['applied_global_point_cap']is None and not m['source_geometry_consumed']and not m['old_surface_likelihood_reapplied']
  own=[]
  for obj in m['objects']:
   q=np.asarray(obj['prior_mass']);energy=np.asarray(obj['energy_m2']);sp=np.asarray(obj['shape_prior']);mass=np.asarray(obj['joint_probability_mass'])
   assert q.ndim==1 and len(q)>0 and energy.shape==(len(q),2) and mass.shape==energy.shape and sp.shape==(2,)
   assert np.isfinite(q).all() and np.isfinite(energy).all() and np.isfinite(mass).all() and np.isfinite(sp).all()
   assert (q>=0).all() and (mass>=0).all() and (sp>0).all()
   assert np.isclose(q.sum(),1,atol=1e-12,rtol=0) and np.isclose(mass.sum(),1,atol=1e-12,rtol=0) and np.isclose(sp.sum(),1,atol=1e-12,rtol=0)
   if name=='G':np.testing.assert_array_equal(sp,[.5,.5])
   with np.errstate(divide='ignore'):a=np.log(q[:,None])-energy/(2*.01**2)
   maximum=a.max(0);v=np.exp(a-maximum);conditional=v/v.sum(0);z=maximum+np.log(v.sum(0));logit=(z[1]-z[0])/16+np.log(sp[1]/sp[0]);shape=np.exp([-np.logaddexp(0,logit),-np.logaddexp(0,-logit)])
   np.testing.assert_allclose(mass,conditional*shape,atol=1e-13,rtol=0)
   own.append({k:obj[k]for k in ['support_track','particles','prior_mass','energy_m2','fit_input_points_sha256','fit_input_normals_sha256','fit_input_labels_sha256','conditional_geometry']})
  identities.append(own)
 assert identities[0]==identities[1]

def audit_frames(frames,final,paid_actions,call_count):
 assert final['mapper_frames']==final['registered_frames']==len(frames)==paid_actions+1
 assert final['score_call_count']==call_count
 assert [f['timestamp_s'] for f in frames]==list(map(float,range(paid_actions+1)))
 assert all(len(f['sha256'])==64 and all(c in '0123456789abcdef' for c in f['sha256']) for f in frames)
 seq=hashlib.sha256(''.join(f['sha256'] for f in frames).encode()).hexdigest()
 assert seq==final['input_sequence_sha256']
 return dict(actual_frame_count=len(frames),input_sequence_sha256=seq,one_frame_per_paid_action_plus_initial=True,raw_frame_values_independently_rehashed=False,receipt_hash_chain_verified=True)

def audit_trace(trace,anchor,result,cell_m=.2):
 positions=np.asarray([anchor[:2]]+[s['position'] for s in trace],dtype=float)
 path=float(np.linalg.norm(np.diff(positions,axis=0),axis=1).sum()*cell_m)
 collisions=sum(bool(s['collision']) for s in trace)
 decision=float(sum(s['decision_wall_s'] for s in trace))
 assert np.isclose(path,result['path_length_m'],atol=1e-10,rtol=0)
 assert collisions==result['collisions']
 assert np.isclose(decision,result['decision_wall_s_total'],atol=1e-8,rtol=0)
 return dict(path_recomputed_m=path,collisions_recomputed=collisions,decision_wall_s_recomputed=decision,source='saved executed grid poses and per-action receipts')

def evaluate(attempt,layout,arm):
 e=Evaluator();protocol=read(ROOT/'execution/protocol.json');r=read(attempt/'result.json');manifest=read(attempt/'artifact_manifest.json')
 e.inputs[str((attempt/'artifact_manifest.json').resolve())]=sha(attempt/'artifact_manifest.json')
 assert r['status']==manifest['status']=='completed_execution';assert r['layout']==layout and r['arm']==arm
 assert r['paid_action_budget']==protocol['policy']['paid_action_budget'];assert r['paid_actions']<=r['paid_action_budget']
 required={'result.json','steps.json','choices.json','final_tsdf_mesh.ply','final_belief.npz','initial_evaluation.json','template_audit.json','template_frame_receipts.json','final_template_receipt.json','B11_execution_binding.json','attempt.json'}
 assert required<=set(manifest['files_sha256']), 'Missing critical artifact hash'
 for name,h in manifest['files_sha256'].items():checked(dict(path=str(attempt/name),sha256=h),e.inputs)
 binding=read(attempt/'B11_execution_binding.json')
 execution_freeze=read(checked(binding['execution_freeze'],e.inputs))
 for path,h in execution_freeze['files_sha256'].items():checked(dict(path=path,sha256=h),e.inputs)
 assert binding['execution_freeze']['sha256']==sha(ROOT/'execution/freeze.json')
 ar=read(attempt/'attempt.json')
 assert ar['protocol_sha256']==sha(ROOT/'execution/protocol.json')
 assert ar['prepared_freeze_sha256']==sha(ROOT/'execution/layouts'/layout/'prepared/freeze.json')
 assert binding['predictor']==('visibility'if arm=='S_visibility'else'acquisition')and binding['full_posterior_same_quadrature']and not binding['short_cost_band_used']and binding['geometry_from_acquired_frames_only']
 reference=e.reference(ROOT/'scene/evaluator_references'/layout);native=e.native(resource(attempt/'final_tsdf_mesh.ply'),reference)
 before=read(ROOT/'execution/layouts'/layout/'prepared/initial_state.json');assert r['initial_packet_sha256']==before['initial_packet_sha256']
 trace=read(attempt/'steps.json');assert len(trace)==r['paid_actions'];assert digest([(x['action'],x['position'],x['heading'])for x in trace])==r['trajectory_sha256']
 with np.load(attempt/'final_belief.npz',allow_pickle=False)as z:belief=z['belief'];position=z['position'];heading=int(z['heading'])
 truth_path=ROOT/'scene/prepared'/layout/'evaluator_truth.npz';e.inputs[str(truth_path.resolve())]=sha(truth_path)
 with np.load(truth_path,allow_pickle=False)as z:reachable=z['reachable']
 independent=coverage_from_belief(belief,reachable)
 for k,x in independent.items():
  if k in r['final_evaluation']:assert np.isclose(x,r['final_evaluation'][k],atol=1e-14,rtol=0),k
 final_pose=[*map(int,position),heading];assert final_pose==r['final_pose'];returned=final_pose==before['anchor'];assert returned==r['final_pose_equals_initial_anchor']
 initial=read(attempt/'initial_evaluation.json');hits=first_correct_free_actions(initial['free_correct_fraction'],trace)
 for key,hit in hits.items():assert hit==r['first_free_correct_action'][key]
 calls=read(attempt/'template_audit.json')
 for call in calls:
  check_models(call)
  assert call['B11_predictor']==binding['predictor']
  diag=call.get('acquisition_diagnostics')
  if diag is not None:assert diag['acquired_frames']==call['mapper_frames']
 frame_audit=audit_frames(read(attempt/'template_frame_receipts.json'),read(attempt/'final_template_receipt.json'),r['paid_actions'],len(calls))
 kinematic_audit=audit_trace(trace,before['anchor'],r)
 assert len(calls)==r['global_planning_calls']
 metrics=native['metrics'];areas={name:float(reference['weights'][name].sum())for name in ['open31','closed118']};recalls={name:metrics['target_surface_metrics'][name]['recall_05cm']for name in areas};recalls10={name:metrics['target_surface_metrics'][name]['recall_10cm']for name in areas};total=sum(areas.values())
 weighted=sum(areas[name]*recalls[name]for name in areas)/total if total else None
 e.verify()
 return dict(status='complete_independent_full_run_evaluation',layout=layout,topology=layout.split('_')[0],arm=arm,attempt=str(attempt),paid_actions=r['paid_actions'],budget=r['paid_action_budget'],initial_packet_sha256=r['initial_packet_sha256'],initial_correct_free_fraction=initial['free_correct_fraction'],final_coverage=independent,
  A95=hits['0.95'],A100=hits['1.0'],unreached_thresholds_are_censored_at_observed_actions=True,censoring_actions=r['paid_actions'],path_length_m=r['path_length_m'],collisions=r['collisions'],returned_to_anchor=returned,termination=r['termination'],elapsed_s=r['elapsed_s'],decision_wall_s_total=r['decision_wall_s_total'],global_planning_calls=r['global_planning_calls'],trajectory_sha256=r['trajectory_sha256'],native_mesh_sha256=sha(attempt/'final_tsdf_mesh.ply'),
  object_recall_5cm=recalls,object_recall_10cm=recalls10,frame_audit=frame_audit,kinematic_audit=kinematic_audit,object_area_weighted_recall_5cm=weighted,object_macro_recall_5cm=float(np.mean(list(recalls.values()))),surface_metrics=metrics,full_particle_calls_audited=len(calls),artifact_manifest_sha256=sha(attempt/'artifact_manifest.json'),protocol_sha256=sha(ROOT/'execution/protocol.json'),reference_sha256=reference['receipt']['reference_npz_sha256'],input_sha256=e.inputs,new_actor_actions=0,evaluator_uses_truth_actor_declares_no_truth=True,separation_evidence='Frozen actor source/API review and runtime receipts; not an OS-level filesystem access trace',timing_not_isolated_realtime_benchmark=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('--attempt',type=Path,required=True);p.add_argument('--layout',required=True);p.add_argument('--arm',choices=['S_visibility','S_acquisition'],required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);t=time.monotonic()
 try:
  result=evaluate(a.attempt,a.layout,a.arm);result['evaluator_wall_s']=time.monotonic()-t;result['evaluator_source_sha256']={str(x):sha(x)for x in [Path(__file__),TOP/'evaluate_state.py',TOP/'paired_metrics.py']};write(a.output/'result.json',result);print(json.dumps(dict(status=result['status'],layout=a.layout,arm=a.arm,F1=result['surface_metrics']['f1_05cm'],A95=result['A95'],A100=result['A100'],returned=result['returned_to_anchor'])))
 except BaseException as ex:write(a.output/'failure.json',dict(error=repr(ex),traceback=traceback.format_exc()));raise
if __name__=='__main__':main()
