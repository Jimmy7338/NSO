"""Real S3 completed-boundary test; no hypothetical new feedback sample."""
import sys,copy,json,time,hashlib
from pathlib import Path
W=Path('/home/ouc/nso_robot/workspaces/nso_hybrid_20261029');O=Path(__file__).parent;sys.path.insert(0,str(W/'nso_bridge'));sys.path.insert(0,str(O))
import nso_bridge
from nso_bridge import joint_engine
nso_bridge.__path__=[str(O/'candidate')]+list(nso_bridge.__path__)
from nso_bridge.optimized_primitive_feedback_v1 import OptimizedRebasePrimitivePilot,PROFILE
from nso_bridge.primitive_geometry_rebase_v1 import stats_fingerprint
from actual_history_fixture import build_original_actual_pilot,REPORT
started=time.monotonic();old,observations,cache,settings=build_original_actual_pilot(O/'live_boundary_original_epoch_fixture')
calibration=copy.deepcopy(settings['calibration']);calibration['primitive_geometry_profile']=PROFILE
actor=OptimizedRebasePrimitivePilot(configuration=settings['configuration'],execution_allowed=False,calibration=calibration,semantic_configuration=None,semantic_output=str(REPORT/'joint/semantics'),feedback_output=O/'live_boundary_actual_check')
for key in ['configuration','inner','directory','snapshot','grid','records','pending_raw_goal','frozen_prediction_inputs']:setattr(actor,key,getattr(old,key))
actor.rebase_root=O/'live_boundary_actual_check';original=stats_fingerprint(actor.inner);physical=copy.deepcopy(observations[-1]['response']['result'])
result=actor.rebase_after_committed_actual_feedback(observations[-1]['request']['params']['path'],physical)
assert stats_fingerprint(actor.inner)['gain_snapshot']==original['gain_snapshot']
assert actor.inner.primitive_outcomes==original['primitive_outcomes']
assert actor.inner.runtime.states[0]['arrived_count']==original['arrived_count']
assert sum(x['primitive_arrival_verified']for x in actor.inner.primitive_outcomes)==3
assert result['primitive_outcome']==physical['primitive_outcome']
assert len(actor.records)==4 and actor.geometry_generation==1
assert result['geometry_rebase']['gain_observe_calls']==0 and result['geometry_rebase']['ledger_observe_calls']==0
assert result['geometry_rebase']['explicit_execution_route_repair']['target_before']==result['geometry_rebase']['explicit_execution_route_repair']['target_after']
assert stats_fingerprint(old.inner)==original
next_decision=actor.next_action();assert next_decision['pending']['action_id']==4
assert next_decision['pending']['assessment']['observed_gain_prediction']['posterior_before']==original['gain_snapshot']['posterior_mean']
receipt=dict(schema='optimized_primitive_live_boundary_software_check/1',success=True,actual_source_report=str(REPORT),source_actual_primitive_events=3,source_original_sensor_packets=4,new_sensor_observations=False,new_Beta_observation_events=0,new_semantic_inference=False,new_robot_actuation=False,counterfactual_Beta_events_used_as_actual=False,actual_original_event_and_verification_records_exact=True,next_original_action_id=4,next_pending_uses_original_learned_posterior=True,optimized_geometry_generation=actor.geometry_generation,geometry_rebuild_elapsed_s=result['geometry_rebase']['geometry_rebuild_elapsed_s'],fixed_map_target_unchanged=True,route_repair=result['geometry_rebase']['explicit_execution_route_repair'],all_history_replay_gain_calls=0,wall_s=time.monotonic()-started,scope='Staged full-wrapper completed-boundary integration using actual preserved S3 history. Future action4 is a prepared software prediction, not a new measured trial.')
(O/'live_boundary_receipt.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n');print(json.dumps({k:v for k,v in receipt.items()if k!='route_repair'},indent=2))
