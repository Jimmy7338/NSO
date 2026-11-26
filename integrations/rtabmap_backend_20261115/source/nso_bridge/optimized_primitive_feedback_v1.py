"""Opt-in latest optimized geometry after one genuine primitive feedback event.

No actuator I/O. Original G/S candidate/scorer and fixed map target are kept.
Only a literal prefix invalidated by the coordinate rebase is repaired with the
existing unknown-admitting orientation graph. Historical gains are never learnt
again. Whole-waypoint and frozen-epoch primitive modes remain separate options.
"""
from copy import deepcopy
import hashlib,json,time,os
from pathlib import Path
import numpy as np
from scipy.sparse.csgraph import dijkstra
from nso.route_coverage_v2 import orientation_graph,recover_actions
from nso_bridge.primitive_feedback_unbounded_v1 import FrozenEpochPrimitivePilot
from nso_bridge.primitive_geometry_rebase_v1 import (rebuild_geometry_preserving_actual_feedback,literal_prefix_endpoint,stats_fingerprint)

PROFILE='optimized_rebase_after_feedback_v1'

def persist_completed_records(directory,records):
    """Publish unchanged committed records before exposing the new directory."""
    path=Path(directory)/'records.json';temporary=path.with_name(path.name+'.tmp.'+str(os.getpid()))
    data=json.dumps(records,indent=2,allow_nan=False)+'\n'
    temporary.write_text(data,encoding='utf8');os.replace(str(temporary),str(path))
    return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        records=len(records),action_ids=[r['action_id']for r in records],record_values_unchanged=True)

def repair_same_fixed_map_target(inner):
    """Repair only execution actions; preserve the literal selected target/score."""
    state=inner.runtime.states[0];scene=inner.backend.scenes[0];packet=state['packet']
    if state['pending']is not None or scene['gain']._pending:raise ValueError('route repair only at a completed actual-feedback boundary')
    option=state.get('option');old_actions=deepcopy(state['active_actions']);target=None if option is None else list(option['pose'])
    receipt=dict(profile='same_fixed_map_target_route_repair_v1',fixed_planning_transform=deepcopy(inner.configuration['planning_from_odom']),target_before=target,target_after=deepcopy(target),original_option_id=None if option is None else option.get('option_id'),original_selected_nominal_route=None if option is None else deepcopy(option.get('outbound_states')),actions_before=old_actions,actions_after=deepcopy(old_actions),current_actual_discrete_pose=[*packet.position,packet.heading],repaired=False,global_chooser_called=False,candidate_or_score_changed=False,paid_action_or_observation_created=False,arrival_credit_created=False,automatic_return=False,new_stop_or_limit=False,reason=None)
    if option is None or not old_actions:
        receipt['reason']='no_remaining_commitment_original_rechoose_trigger';return receipt
    endpoint=literal_prefix_endpoint(packet.position,packet.heading,old_actions);receipt['old_literal_prefix_endpoint']=endpoint
    if endpoint==target:
        receipt['reason']='literal_prefix_still_reaches_original_fixed_map_target';return receipt
    belief=np.asarray(inner.mapper.belief);safe=(belief!=1).copy();row,col=packet.position
    safe[row,col]=True  # Existing virtual-current admission; no footprint-safe gate.
    reason='navigation_route_invalidated_by_rebase';actions=None;states=None
    tr,tc,th=target
    if 0<=tr<safe.shape[0]and 0<=tc<safe.shape[1]and safe[tr,tc]:
        graph,cells,ids=orientation_graph(safe);start=int(ids[row,col])*4+packet.heading;end=int(ids[tr,tc])*4+th
        costs,pred=dijkstra(graph,directed=True,indices=start,return_predecessors=True)
        if np.isfinite(costs[end]):
            actions=list(recover_actions(pred,start,end,cells));states=[];cursor=end
            while cursor!=start:
                states.append([*map(int,cells[cursor//4]),int(cursor%4)]);cursor=int(pred[cursor])
            states.append([row,col,packet.heading]);states.reverse()
    if actions is None:
        state['active_actions']=[];state['option']=None;state['phase']='planning'
        receipt.update(reason=reason,route_available=False,actions_after=[],original_target_record_preserved=True,next_behavior='existing empty-active original chooser trigger; no fake action, score or stop')
    else:
        if literal_prefix_endpoint(packet.position,packet.heading,actions)!=target:raise RuntimeError('same-target route repair endpoint mismatch')
        state['active_actions']=actions
        receipt.update(reason='repaired_execution_route_to_same_fixed_map_target',route_available=True,repaired=True,actions_after=deepcopy(actions),execution_states_after=states,repair_cost=len(actions),zero_action_route_without_arrival_credit=not actions)
    inner.runtime.audit.append(dict(event=receipt['reason'],route_repair=deepcopy(receipt)))
    if receipt['target_after']!=receipt['target_before']:raise RuntimeError('route repair must never translate/replace the selected map target')
    return receipt

class OptimizedRebasePrimitivePilot(FrozenEpochPrimitivePilot):
    def __init__(self,configuration,feedback_output=None,**kwargs):
        calibration=kwargs.get('calibration')or{}
        if calibration.get('primitive_geometry_profile')!=PROFILE:raise ValueError('explicit optimized primitive geometry profile required')
        if calibration.get('geometry_pose_backend')!='rtabmap_optimized_replay_v1':raise ValueError('actual optimized RTAB input geometry required')
        super().__init__(configuration,feedback_output=feedback_output,**kwargs)
        self.rebase_root=self.directory;self.geometry_generation=0;self.optimized_geometry_receipts=[];self.route_repair_receipts=[]
    def identity(self):
        result=super().identity();result['receipt']['geometry_pilot'].update(profile=PROFILE,external_grid='latest actual full RTAB grid at terminal-feedback boundaries; held within each pending prediction',full_optimized_history_replay_implemented=True,history_scope='all actual NSO RGBD/scan observation packets and cached actual semantic labels',optimization_relearning=False,calibration_event_order='real terminal frame feedback in its original pending epoch, then geometry-only rebuild with frozen events/statistics',same_fixed_map_target_route_repair=True,original_policy_equivalence=False)
        result['receipt']['sensor_ranges']['observed_navigation_map_range_m']=None
        result['receipt']['sensor_ranges']['observed_navigation_map_scope']='actual full RTAB navigation grid; no extra8m crop'
        return result
    def summary(self):
        result=super().summary();result.update(primitive_geometry_profile=PROFILE,geometry_rebase_generations=self.geometry_generation,geometry_rebase_receipts=deepcopy(self.optimized_geometry_receipts),execution_route_repair_receipts=deepcopy(self.route_repair_receipts),full_optimized_history_replay_implemented=True,historical_feedback_relearned=False)
        result['sensor_ranges']['observed_navigation_map_range_m']=None
        result['sensor_ranges']['observed_navigation_map_scope']='actual full RTAB grid rebased after terminal feedback; no extra8m crop'
        return result
    def observe_primitive_result(self,path,action_id,ros_status=None,reason=None):
        # Preserve original immutable prediction-epoch event and measured arrival.
        result=super().observe_primitive_result(path,action_id,ros_status=ros_status,reason=reason)
        return self.rebase_after_committed_actual_feedback(path,result)
    def rebase_after_committed_actual_feedback(self,path,result):
        # Exposed for an actual recorded terminal-boundary replay test. This
        # method never creates a new feedback sample or observation.
        action_id=len(self.records)-1
        if action_id and (self.inner.runtime.states[0]['pending']is not None or self.inner.backend.scenes[0]['gain']._pending):raise ValueError('actual feedback must be fully committed before optimized rebase')
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=self.records[-1]['original_sha256']:raise ValueError('terminal source differs from committed actual record')
        execution_epoch_state=deepcopy(result.get('state'))
        if action_id==0:
            result.update(primitive_geometry_profile=PROFILE,planning_state_after_rebase=execution_epoch_state,summary=self.summary());return result
        with np.load(path,allow_pickle=False)as z:
            metadata=json.loads(str(z['joint_metadata_json'].item()));snapshot=metadata['rtabmap_graph_snapshot'];current_grid=z['joint_belief_grid'].copy()
        before=stats_fingerprint(self.inner);generation=self.geometry_generation+1
        folder=self.rebase_root/'optimized_geometry'/f'generation_{generation:06d}'
        started=time.monotonic();candidate,receipt=rebuild_geometry_preserving_actual_feedback(self,snapshot,current_grid,output=folder)
        # No chooser, target translation, blacklisting or new budget is involved.
        route=repair_same_fixed_map_target(candidate.inner)
        receipt.update(geometry_rebuild_elapsed_s=time.monotonic()-started,explicit_execution_route_repair=route,history_packets=len(self.records),cost_scaling='full original observation history rebuild; no truncation or hidden cap')
        if candidate.inner.backend.scenes[0]['gain'].snapshot()!=before['gain_snapshot']:raise RuntimeError('route repair changed actual learned gain history')
        # summary() advertises directory/records.json immediately after swap.
        # The preceding actual commit was persisted in the OLD directory;
        # atomically publish its unchanged records in the new one first.
        receipt['completed_records_manifest']=persist_completed_records(candidate.directory,candidate.records)
        # Swap only the completed candidate. No prior active state has been
        # modified by geometry reconstruction or the execution-route repair.
        for name in ['configuration','inner','directory','snapshot','grid','records','pending_raw_goal','frozen_prediction_inputs']:
            setattr(self,name,getattr(candidate,name))
        self.geometry_generation=generation
        self.optimized_geometry_receipts.append(receipt);self.route_repair_receipts.append(route)
        (folder/'boundary_receipt.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
        result.update(primitive_geometry_profile=PROFILE,execution_epoch_state=execution_epoch_state,planning_state_after_rebase=self.inner._state(),geometry_rebase=receipt,historical_feedback_relearned=False,summary=self.summary())
        return result
