"""Software reconstruction/rebase of genuine S3 history. No new robot samples."""
import sys,json,copy,types,hashlib,time,math
from pathlib import Path
import numpy as np
W=Path('/home/ouc/nso_robot/workspaces/nso_hybrid_20261029');O=Path(__file__).parent;sys.path.insert(0,str(W/'nso_bridge'));sys.path.insert(0,str(O/'candidate'))
from nso_bridge import joint_engine
from nso_bridge.primitive_feedback_unbounded_v1 import FrozenEpochPrimitivePilot
from primitive_geometry_rebase_v1 import rebuild_geometry_preserving_actual_feedback,stats_fingerprint
R=Path('/home/ouc/nso_robot/reports/nso-hybrid-20261115/NSO_primitive_feedback_fresh_S_20261115_01');lines=[json.loads(x)for x in (R/'joint/decisions.jsonl').read_text().splitlines()]
settings=copy.deepcopy(lines[0]['request']['params']);saved_config=copy.deepcopy(settings['semantic_configuration']);settings['semantic_configuration']=None;settings['semantic_output']=str(R/'joint/semantics');settings['execution_allowed']=False
settings['feedback_output']=str(O/'original_epoch_software_reproduction');pilot=FrozenEpochPrimitivePilot(**settings)
cache={}
for file in (R/'joint/semantics').glob('*.json'):
 receipt=json.loads(file.read_text())
 with np.load(receipt['mask_file'],allow_pickle=False)as z:labels=z['semantic'].copy();owner=z['instance_owner'].copy()if'instance_owner'in z else None
 cache[str(receipt['frame_id'])]=dict(labels=labels,owner=owner,receipt=receipt)
def original_cached(this,z,meta,sha):
 entry=cache[str(meta['frame_id'])];this._instance_owner_map=None if entry['owner']is None else entry['owner'].copy();this._attempt_semantic_receipt=copy.deepcopy(entry['receipt']);return entry['labels'].copy()
pilot.inner._semantic_labels=types.MethodType(original_cached,pilot.inner)
observations=[x for x in lines if x['request']['method']=='observe_primitive_result'and x['response']['ok']]
choice=next(x['response']['result']for x in lines if x['request']['method']=='next_action')
started=time.monotonic();pilot.observe_primitive_result(**observations[0]['request']['params'])
state=pilot.inner.runtime.states[0];scene=pilot.inner.backend.scenes[0]
state['option']=copy.deepcopy(choice['selected']);state['active_actions']=copy.deepcopy(choice['selected']['outbound_actions']);state['phase']='outbound';scene['plans']=1;scene['selected']=copy.deepcopy(choice['selected']);scene['last_selection']=copy.deepcopy(choice['selection'])
checks=[]
for row in observations[1:]:
 decision=pilot.next_action();value=pilot.observe_primitive_result(**row['request']['params']);expected=row['response']['result']['primitive_outcome']['gain_snapshot']
 if value['primitive_outcome']['gain_snapshot']!=expected:raise ValueError('software original-epoch reproduction does not match actual saved Beta')
 checks.append(dict(action_id=row['request']['params']['action_id'],Beta_exact_original=True,actual_primitive_arrival=row['response']['result']['primitive_outcome']['primitive_arrival_verified']))
# Freeze the literal actual saved outcomes, rather than presenting the software
# reproduction's file paths/events as newly observed physical outcomes.
pilot.inner.primitive_outcomes=copy.deepcopy(observations[-1]['response']['result']['summary']['primitive_outcomes'])
original=stats_fingerprint(pilot.inner)
with np.load(observations[-1]['request']['params']['path'],allow_pickle=False)as z:meta=json.loads(str(z['joint_metadata_json'].item()));grid=z['joint_belief_grid'].copy()
graph=meta['rtabmap_graph_snapshot']
new,receipt=rebuild_geometry_preserving_actual_feedback(pilot,graph,grid,output=O/'actual_graph_rebase',semantic_cache=cache)
if stats_fingerprint(new.inner)!=original:raise ValueError('actual latest graph changed historical feedback')
new_next=new.next_action()
if new_next['pending']['assessment']['observed_gain_prediction']['posterior_before']!=original['gain_snapshot']['posterior_mean']:raise ValueError('next actual prediction lost original learned posterior')
if new.inner.backend.scenes[0]['gain'].snapshot()['pending_action_ids']!=[4]:raise ValueError('next original action prediction not prepared exactly once')
# Separate stress test: translated optimized geometry, never an experimental
# loop closure or replacement sensor measurement. Only verifies conservation.
perturbed=copy.deepcopy(graph);shift=np.eye(4);shift[0,3]=.4;shift[1,3]=-.2
for node in perturbed['nodes']:
 if node.get('map_from_base')is not None:node['map_from_base']=(shift@np.asarray(node['map_from_base'])).tolist()
perturbed['map_from_odom']=(shift@np.asarray(perturbed['map_from_odom'])).tolist();perturbed['graph_revision_sha256']='SOFTWARE_TRANSLATION_STRESS_NOT_ACTUAL_LOOP'
other,stress=rebuild_geometry_preserving_actual_feedback(pilot,perturbed,grid,output=O/'software_translation_stress',semantic_cache=cache)
if stats_fingerprint(other.inner)!=original:raise ValueError('software geometry perturbation rewrote original actual feedback')
summary=dict(status='staged_software_checks_passed_not_deployed',actual_recorded_trial=str(R),actual_recorded_primitive_count=3,source_history_count=len(pilot.records),new_robot_motion=False,new_sensor_observations=False,new_model_inference=False,original_epoch_software_reproduction_checks=checks,actual_graph_geometry_rebase=receipt,software_only_translation_stress=stress,actual_current_graph_beta_and_events_exact=True,software_stress_beta_and_events_exact=True,original_recorded_arrivals_preserved=True,next_original_pending_action_id=new_next['pending']['action_id'],next_pending_uses_same_learned_posterior=True,wall_s=time.monotonic()-started,full_policy_deployment_ready=False,blocker='Literal prefix conservation is verified, but graph corrections can change its geometric endpoint relative to fixed selected macro target. No hidden target overwrite, rescore, stop or route repair is introduced; review this separately before full runs.')
(O/'actual_rebase_check.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n');print(json.dumps({k:v for k,v in summary.items()if k not in ['actual_graph_geometry_rebase','software_only_translation_stress']},indent=2))
