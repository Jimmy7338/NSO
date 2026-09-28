#!/usr/bin/env python3
"""One completed raw history only; no world instance or evaluator invoked."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import csv,gzip,hashlib,json,math,sys,tempfile,time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from nso.observed_runtime_mapper_v10 import ObservedRuntimeMapperV10
from nso.coverage_budget_v20 import ObservedCoverageLedgerV20,scan_hit_mask,assess_coverage_budget
from nso.decision_replay_v13 import load_packet,array_hash
from nso.cpu_sensor_contract_v10 import digest,json_value
from utils.grid_geometry import visible_mask,inflated_obstacles
OUT=Path(__file__).resolve().parent
BATCH=ROOT/'eval_results/facility_v20_coverage_probe_20260915';CASE=BATCH/'case_00'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def atomic(path,content):
 fd,name=tempfile.mkstemp(prefix=path.name+'.',suffix='.tmp',dir=path.parent)
 try:
  with os.fdopen(fd,'w') as f:f.write(content);f.flush();os.fsync(f.fileno())
  os.replace(name,path)
 finally:
  if os.path.exists(name):os.unlink(name)
def dump(name,value):atomic(OUT/name,json.dumps(json_value(value),indent=2,sort_keys=True,allow_nan=False)+'\n')
started=time.perf_counter();manifest=read(BATCH/'manifest.json');result=read(CASE/'result.json')
assert result['paid_actions']==160 and result['case']['index']==0
assert result['case']==manifest['cases'][0]
assert not result['termination']['failed'] and result['collisions']==0
sources=manifest['source_sha256'];assert all(sha(ROOT/n)==v for n,v in sources.items())
inventory=read(CASE/'artifact_hashes.json');assert all(sha(CASE/n)==v for n,v in inventory.items())
with gzip.open(CASE/'module_calls.json.gz','rt') as f:calls=json.load(f)
assert digest([{k:v for k,v in row.items() if k!='elapsed_s'} for row in calls])==result['module_calls_sha256']
name='references/'+result['case']['parent']+'_'+result['case']['assignment']+'_2026.npz'
ref=Path(manifest['reference_source_root'])/name;assert sha(ref)==manifest['reference_sha256'][name]
# Read only declared public configuration, not the reference arrays or world geometry.
with np.load(ref,allow_pickle=False) as data:config=SimpleNamespace(**json.loads(str(data['metadata'].item()))['signature_payload']['config'])
shape=(round(config.height_m/config.resolution_m),round(config.width_m/config.resolution_m))
selects={r['action_id']:r for r in calls if r['method']=='select_topo_target'}
guards={r['action_id']+1:r for r in calls if r['method']=='assess_local_action'}
events={r['action_id']:r for r in calls if r['method']=='observe_unique_coverage_v20'}
assert len(guards)==len(events)==160
bootstrap=next(r for r in calls if r['method']=='coverage_bootstrap_v20')['outputs']
true_curve={r['action_id']:r['coverage_2d'] for r in result['coverage_trace']}
mapper=ObservedRuntimeMapperV10(shape,config)
first=load_packet(CASE/'packets/0000.npz');assert first.sha256()==result['case']['initial_packet_sha256']
mapper.update(first.frame,first.scan)
ledger=ObservedCoverageLedgerV20(shape,first.position,config.robot_radius_m/config.resolution_m,prefix_actions=manifest['config']['replan_interval_actions'])
ledger.start(mapper.belief,radar_hits=scan_hit_mask(first.scan,shape,config.resolution_m))
assert ledger.snapshot()==bootstrap
rows=[];decisions=[];known_changes={}
def record(i):
 state=ledger.state;s=state['safe'];p=state['possible'];b=mapper.belief;known=b!=-1
 snap=ledger.snapshot();safe_all=~inflated_obstacles(b!=0,config.robot_radius_m/config.resolution_m)
 known_p=known&p;free_p=(b==0)&p;omitted=known_p&~s
 footprint_excluded=free_p&~safe_all;disconnected=free_p&safe_all&~s;occupied=(b==1)&p
 assert int(omitted.sum())==int(footprint_excluded.sum()+disconnected.sum()+occupied.sum())
 n=int(p.sum());c_known=float(known_p.sum()/n);c_free=float(free_p.sum()/n)
 row=dict(action_id=i,evaluator_coverage=true_curve[i],S_over_P=snap['C_plan'],known_intersect_P_over_P=c_known,
  known_free_intersect_P_over_P=c_free,S_cells=int(s.sum()),P_cells=n,known_P_cells=int(known_p.sum()),
  known_free_P_cells=int(free_p.sum()),known_occupied_P_cells=int(occupied.sum()),
  known_free_footprint_excluded_cells=int(footprint_excluded.sum()),known_free_safe_disconnected_cells=int(disconnected.sum()),
  observed_known_total=int(known.sum()),observed_free_total=int((b==0).sum()),
  conservative_safe_rate=snap['conservative_rate'],safe_deficit_cells=snap['deficit_cells'],
  known_P_deficit_cells=max(0,math.ceil(.8*n)-int(known_p.sum())),union_yield=snap['union_yield'],
  true_minus_safe=true_curve[i]-snap['C_plan'],known_P_minus_true=c_known-true_curve[i])
 rate_ids=[j for block in snap['coverage_prefixes'] for j in range(block['first_action_id'],block['last_action_id']+1)]
 known_rate=.5*max(0,sum(known_changes[j]['new_in_P']-known_changes[j]['lost_in_P'] for j in rate_ids))/len(rate_ids) if rate_ids else None
 row['conservative_known_P_rate']=known_rate
 rows.append(row)
 if i in selects:
  call=selects[i];outputs=call['outputs'];assert snap==outputs['coverage_state']
  candidates=outputs['score_audit'];budgets=[x['v20_coverage']['coverage_budget'] for x in candidates]
  allowed=[x['allowed'] for x in budgets]
  assert not any(allowed)
  visibility_cache={};alternatives=[]
  for c,route in zip(candidates,outputs['candidates']):
   mask=np.zeros(shape,bool)
   for pose in route['outbound_states'][1:]:
    cell=tuple(pose[:2])
    if cell not in visibility_cache:
     visibility_cache[cell]=visible_mask(b==1,cell,0,int(config.max_depth_m/config.resolution_m),360.)&(b==-1)
    mask|=visibility_cache[cell]
   assert digest(mask)==c['v20_coverage']['predicted_mask_sha256']
   assert int(mask.sum())==c['v20_coverage']['unique_predicted_unknown_cells']
   count=int((mask&p).sum());gain=count*snap['union_yield']*.5
   alternate=assess_coverage_budget(row['known_P_deficit_cells'],call['inputs']['remaining_budget'],
    route['outbound_cost'],route['return_cost'],gain,known_rate,5)
   alternatives.append(dict(candidate_id=c['candidate_id'],group=route['group'],
    predicted_unknown_intersect_P_cells=count,union_yield=snap['union_yield'],gain_discount=.5,
    budget=alternate,diagnostic_only=True,policy_or_sensor_executed=False))
  decisions.append(dict(action_id=i,candidate_count=len(candidates),budget_allowed_count=sum(allowed),
   reported_admitted_count=len(outputs['quality_budget_admitted_candidates']),
   alternative_known_P_allowed_count=sum(x['budget']['allowed'] for x in alternatives),
   alternative_known_P_quality_role_allowed_count=sum(x['budget']['allowed'] and not x['group'].startswith('coverage_') for x in alternatives),
   alternative_known_P_candidates=alternatives,
   reasons=dict(Counter(x['reason'] for x in budgets)),coverage=row,
   candidates=[dict(candidate_id=c['candidate_id'],group=r['group'],G_score=outputs['scores']['G'][j],
     budget=c['v20_coverage']['coverage_budget']) for j,(c,r) in enumerate(zip(candidates,outputs['candidates']))]))
record(0)
for i in range(1,161):
 p=load_packet(CASE/f'packets/{i:04d}.npz');a=result['actions'][i-1]
 assert p.action_id==i and p.sha256()==a['packet_sha256'] and p.action==a['action']
 assert [*p.position,p.heading]==a['pose'] and not p.collision
 guard=guards[i];assert guard['outputs']['allowed'] and guard['outputs']['next_pose']==a['pose']
 assert guard['inputs']['belief_sha256']==digest(mapper.belief)
 prediction=visible_mask(mapper.belief==1,tuple(guard['outputs']['next_pose'][:2]),0,int(config.max_depth_m/config.resolution_m),360.)&(mapper.belief==-1)
 previous_belief=mapper.belief.copy();previous_P=ledger.state['possible'].copy()
 mapper.update(p.frame,p.scan)
 event=ledger.observe(mapper.belief,i,events[i]['inputs']['coverage_intent'],predicted_mask=prediction,
  radar_hits=scan_hit_mask(p.scan,shape,config.resolution_m))
 assert event==events[i]['outputs'],('event diverged',i)
 known_changes[i]=dict(new_in_P=int(((previous_belief==-1)&(mapper.belief!=-1)&ledger.state['possible']).sum()),
  lost_in_P=int(((previous_belief!=-1)&(mapper.belief==-1)&previous_P).sum()))
 record(i)
 if i%20==0:print(json.dumps(rows[-1]),flush=True)
mesh=mapper.mesh();mesh_hashes={k:array_hash(np.asarray(getattr(mesh,k))) for k in ('vertices','triangles','vertex_colors')}
assert mesh_hashes==result['final_mesh_sha256']
assert all(sha(ROOT/n)==v for n,v in sources.items())
assert all(sha(CASE/n)==v for n,v in inventory.items())
from io import StringIO
stream=StringIO();writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
atomic(OUT/'coverage_curve.csv',stream.getvalue())
errors_safe=np.array([r['S_over_P']-r['evaluator_coverage'] for r in rows]);errors_known=np.array([r['known_P_minus_true'] for r in rows])
summary=dict(status='passed',scope='one completed N case, saved observations only; no new sensing/policy/world/evaluator run',
 case=result['case'],paid_actions=160,raw_packets=161,all_packet_hashes_match=True,all_160_ledger_events_exact=True,
 all_8_decision_snapshots_exact=True,all_160_pre_action_beliefs_exact=True,final_mesh_hashes_exact=True,
 frozen_sources_match=True,case_inventory_match=True,final_mesh_sha256=mesh_hashes,
 decisions=len(decisions),total_candidates=sum(d['candidate_count'] for d in decisions),
 budget_allowed_candidates=sum(d['budget_allowed_count'] for d in decisions),
 rejection_reasons=dict(sum((Counter(d['reasons']) for d in decisions),Counter())),
 candidate_G_scores_positive=sum(c['G_score']>0 for d in decisions for c in d['candidates']),
 coverage_curve_every20=[r for r in rows if r['action_id']%20==0],
 evaluator_gate_first_action=next((r['action_id'] for r in rows if r['evaluator_coverage']>=.8),None),
 safe_proxy_gate_first_action=next((r['action_id'] for r in rows if r['S_over_P']>=.8),None),
 known_P_proxy_gate_first_action=next((r['action_id'] for r in rows if r['known_intersect_P_over_P']>=.8),None),
 descriptive_error_all161_states=dict(S_over_P_mean_absolute=float(abs(errors_safe).mean()),
  known_P_mean_absolute=float(abs(errors_known).mean()),known_P_max_overestimate=float(errors_known.max()),
  known_P_max_underestimate=float(-errors_known.min()),known_P_overestimate_states=int((errors_known>0).sum())),
 alternative_proxy_is_only_diagnostic=True,ground_truth_sent_to_planner=False,alternative_is_certified_lower_bound=False,
 alternative_budget_admission_not_evaluated=False,semantic_efficacy_test=False,
 alternative_known_P_allowed_count=sum(d['alternative_known_P_allowed_count'] for d in decisions),
 alternative_known_P_quality_role_allowed_count=sum(d['alternative_known_P_quality_role_allowed_count'] for d in decisions),
 alternative_budget_definition=dict(deficit='ceil(.8*|P|)-|known intersect P|, clipped at zero',
 rate='.5 * net actual unknown-to-known inside contemporary P / all paid actions in the same coverage prefixes',
 prediction='original route union mask intersect current P * original union_yield * .5',
 reserve='unchanged original outbound+return+5+ceil(residual_deficit/rate)',
 denominator_changes_not_credited_as_sensor_gain=True,saved_candidate_mask_hashes_exact=True,
 timing='each calculation uses only the observed state and prefixes available at that original decision'),
 elapsed_seconds=time.perf_counter()-started)
dump('summary.json',summary);dump('decisions.json',decisions)
dump('provenance.json',dict(script_sha256=sha(Path(__file__)),source_sha256=sources,
 case_artifact_sha256=inventory,reference_metadata_file=str(ref),reference_file_sha256=sha(ref),
 config_only_read_from_reference=True,batch_manifest_status_at_start=manifest['status'],
 batch_manifest_is_live_not_required_complete=True))
print(json.dumps({k:summary[k] for k in ('status','total_candidates','budget_allowed_candidates','evaluator_gate_first_action','safe_proxy_gate_first_action','known_P_proxy_gate_first_action','descriptive_error_all161_states','elapsed_seconds')}),flush=True)
