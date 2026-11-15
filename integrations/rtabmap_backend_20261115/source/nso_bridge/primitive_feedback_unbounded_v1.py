"""Opt-in staged bounded-pilot adapter for original per-primitive gain feedback.

Original vendor unchanged. Unknown admission/no-return/no-total-budget remain
explicit physical variants. A pilot freezes its actual RTAB bootstrap geometry
and external nav grid, then fuses every real subsequent RGBD/scan in that epoch.
The global RTAB/ROS collision map remains live; outputs command raw-odom goals.
This deliberately does not pretend to supply full optimized-history replay.
"""
from copy import deepcopy
import hashlib,json,math,tempfile
from pathlib import Path
import numpy as np
from nso_bridge.joint_engine import JointSession
from nso_bridge.ros_checked_exploration_v2 import ROSCheckedCPUFourModules
from nso_bridge.exploration_joint_policy import ExplorationJointRuntimeIntegration
from nso_bridge.ros_viewpoint_unbounded_v4 import UnboundedActualLedger
from nso_bridge.ros_viewpoint_exploration_v3 import HorizonLedgerView
from nso_bridge.rtabmap_geometry import corrected_geometry,correction_at
from nso.cpu_four_modules_v10 import CPUFourModules
from utils.grid_geometry import DIRECTIONS

PROFILE='primitive_feedback_unbounded_v1'
ADMISSION='ros_viewpoint_exploration_v4_unbounded'

class PrimitiveLedger(UnboundedActualLedger):
    def snapshot(self):
        value=super().snapshot();value['accounting_unit']='actual_primitive_attempt_unbounded';return value

class PrimitiveCPU(ROSCheckedCPUFourModules):
    policy_variant=PROFILE
    admission_profile=ADMISSION
    def start_scene(self,scene_idx,**kwargs):
        kwargs=dict(kwargs,total_budget=1)
        super().start_scene(scene_idx,**kwargs)
        scene=self.scenes[scene_idx];old=scene['ledger'];new=PrimitiveLedger.__new__(PrimitiveLedger)
        new.__dict__.update(old.__dict__);new.total_budget=None
        scene['ledger']=new;scene['feedback']['remaining_budget']=None
    def select_target(self,scene_idx):
        scene=self.scenes[scene_idx];actual=scene['ledger'];scene['ledger']=HorizonLedgerView(actual,None)
        try:return super().select_target(scene_idx)
        finally:scene['ledger']=actual
    def assess_action(self,scene_idx,action):
        scene=self.scenes[scene_idx];packet=scene['packet']
        check=scene['guard'].assess(scene['mapper'].belief,packet.position,packet.heading,action,math.inf)
        heading=(packet.heading+(1 if action=='right' else -1 if action=='left' else 0))%4
        predicted=None
        if check.allowed:
            predicted=scene['gain'].prepare_action(scene['mapper'],position=packet.position,heading=packet.heading,action=action,action_id=packet.action_id+1,planning_camera_poses=scene['ledger'].planning_camera_poses())
        result=dict(allowed=bool(check.allowed),reason=check.reason,action=action,next_pose=[*check.target,heading],remaining_budget=None,reserved_return_cost=None,return_certificate=False,admission_profile=ADMISSION,learned_probability=None,calibrated_uncertainty=None,observed_gain_prediction=predicted)
        self._record(scene_idx,'RPN-UQ','assess_local_action',dict(anchor=scene['return_anchor'],remaining_budget=None),result)
        return result
    # Restore original real per-primitive ledger + gain.observe. Do not inherit
    # ViewpointCPU.compute_reward, which disables primitive calibration for macros.
    compute_reward=CPUFourModules.compute_reward

class PrimitiveRuntime(ExplorationJointRuntimeIntegration):
    policy_variant=PROFILE
    def next_local_action(self,scene_idx):
        state=self.states[scene_idx]
        if state is None or state['closed']:return None
        if state['pending'] is not None:raise RuntimeError('consume real pending primitive outcome first')
        comp=self.components;packet=state['packet'];scene=comp._cpu_backend.scenes[scene_idx]
        if not state['active_actions']:
            self.choose_goal(scene_idx,list(packet.position),(0,self.full_shape[0],0,self.full_shape[1]))
        if not state['active_actions']:
            self.close_sensor_episode(scene_idx,reason='no_positive_current_option');return None
        action=state['active_actions'][0]
        # Original latest-map primitive admission; unknown remains admitted by
        # the explicitly retained ROSCheckedGuard and no return budget is added.
        check=scene['guard'].assess(state['mapper'].belief,packet.position,packet.heading,action,math.inf)
        if not check.allowed:
            rejected=state['option'];state['active_actions']=[];state['option']=None
            self.choose_goal(scene_idx,list(packet.position),(0,self.full_shape[0],0,self.full_shape[1]))
            self.audit.append(dict(event='original_trigger_replan_after_known_occupied_successor',reason=check.reason,old_option_id=None if rejected is None else rejected['option_id']))
            if not state['active_actions']:
                self.close_sensor_episode(scene_idx,reason='no_positive_current_option');return None
            action=state['active_actions'][0]
        comp.bind_execution_option(scene_idx,state['option'])
        assessment=comp.assess_local_action(scene_idx,action)
        if not assessment['allowed']:
            raise RuntimeError('freshly selected primitive not admitted; no invented fallback')
        state['pending']=assessment
        self.audit.append(dict(event='paid_action_authorized',frame_id=packet.frame_id,next_action_id=packet.action_id+1,phase='outbound_no_return',assessment=deepcopy(assessment),option_id=None if state['option']is None else state['option']['option_id']))
        return action
    def close_sensor_episode(self,scene_idx,*,reason):
        state=self.states[scene_idx]
        if state['pending']is not None:raise RuntimeError('consume terminal actual observation before close')
        if state['closed']:return self.sensor_episode_summary(scene_idx)
        packet=state['packet'];scene=self.components._cpu_backend.scenes[scene_idx]
        state['closed']=True
        state['termination']=dict(reason=reason,failed=bool(packet.collision),return_required=False,returned_to_anchor=tuple((*packet.position,packet.heading))==scene['return_anchor'],coverage_complete_verified=False,final_frame_id=packet.frame_id,final_action_id=packet.action_id)
        scene['feedback']=scene['ledger'].finish(reason=reason,failed=bool(packet.collision))
        return self.sensor_episode_summary(scene_idx)

class PrimitiveSession(JointSession):
    def __init__(self,configuration,**kwargs):
        original=deepcopy(configuration)
        if original.get('policy_variant')!=PROFILE:raise ValueError('explicit primitive feedback profile required')
        if original.get('total_budget')is not None:raise ValueError('this declared profile has no total budget')
        config=deepcopy(original)
        for key in ['execution_limits_profile','return_policy','max_viewpoint_attempts','nominal_planning_horizon']:config.pop(key,None)
        config.update(policy_variant='ros_checked_exploration_v2',total_budget=1)
        super().__init__(config,**kwargs)
        self.policy_variant=PROFILE;self.configuration=original
        self.backend=PrimitiveCPU(self.args,1,self.shape);self.components._cpu_backend=self.backend
        self.runtime=PrimitiveRuntime(self.components,1,self.shape)
        self.primitive_outcomes=[]
    def identity(self):
        result=super().identity();result['receipt']['sensor_ranges']['observed_navigation_map_range_m']=None;result['receipt']['sensor_ranges']['observed_navigation_map_scope']='actual bootstrap RTAB grid; no8m crop'
        result['receipt']['primitive_feedback']=dict(profile=PROFILE,original_method_equivalence=False,original_gain_prepare_and_observe=True,forward_nominal_grid_m=.2,turn_nominal_rad=math.pi/2,no_total_budget=True,no_auto_return=True,no_candidate_filter_added=True,no_current_footprint_known_safe_gate=True,scope='bounded actual per-primitive feedback mechanism pilot; declared ROS unknown admission remains')
        return result
    def summary(self):
        result=super().summary();result['sensor_ranges']['observed_navigation_map_range_m']=None;result['sensor_ranges']['observed_navigation_map_scope']='actual bootstrap RTAB grid; no8m crop'
        result.update(accounting_unit='actual_primitive_attempt',actual_primitive_attempts=len(self.primitive_outcomes),completed_measured_primitives=sum(x['primitive_arrival_verified']for x in self.primitive_outcomes),macro_arrivals_verified=0 if self.runtime.states[0]is None else self.runtime.states[0]['arrived_count'],primitive_outcomes=deepcopy(self.primitive_outcomes))
        return result
    def measured_state(self,path):
        with np.load(path,allow_pickle=False)as z:base=np.asarray(z['base_odom_pose'],float)
        point=self.planning_from_odom@[base[0],base[1],0.,1.]
        yaw=base[2]+math.atan2(self.planning_from_odom[1,0],self.planning_from_odom[0,0])
        cell=self.transform.world_to_cell(point[:2]);heading=int(math.floor((math.pi/2-yaw)/(math.pi/2)+.5))%4
        return [*cell,heading],point,yaw
    def observe_primitive_result(self,path,action_id,ros_status=None,reason=None):
        if action_id==0:
            if ros_status is not None or self.last_packet is not None:raise ValueError('bootstrap has no ROS outcome')
            return self.observe(path,0)
        state=self.runtime.states[0];pending=None if state is None else state['pending']
        if pending is None or action_id!=self.last_packet.action_id+1:raise ValueError('one exact pending primitive required')
        if type(ros_status)is not int or ros_status not in (2,3,4,5,8,9):raise ValueError('actual terminal ROS status required')
        actual,point,yaw=self.measured_state(path);goal=self._pending();xy=self.transform.cell_to_world(tuple(pending['next_pose'][:2]));wanted=math.pi/2-pending['next_pose'][2]*math.pi/2
        xy_error=float(np.linalg.norm(point[:2]-xy));yaw_error=abs(math.atan2(math.sin(yaw-wanted),math.cos(yaw-wanted)))
        verified=ros_status==3 and actual==pending['next_pose']  # Original grid-state contract; metric errors remain diagnostic.
        action=pending['action'];original_pending=deepcopy(pending)
        if not verified:
            # Existing declared failed-primitive observation semantics: retain the
            # actual partial pose and one paid attempt; never fabricate arrival.
            state['pending']=dict(pending,next_pose=actual,failed_execution_attempt=True)
            state['option']=None;state['active_actions']=[];state['phase']='planning'
        result=self.observe(path,action_id,action=action)
        outcome=dict(action_id=action_id,action=action,ros_status=ros_status,reason=reason,primitive_arrival_verified=bool(verified),actual_discrete_pose=actual,expected_discrete_pose=original_pending['next_pose'],xy_error_m=xy_error,yaw_error_rad=yaw_error,source_path=str(path),source_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),predicted_gain=original_pending['observed_gain_prediction'],actual_gain_event=deepcopy(self.backend.scenes[0]['gain_feedback']),gain_snapshot=deepcopy(self.backend.scenes[0]['gain'].snapshot()),actual_primitive_attempt_charged=True)
        self.primitive_outcomes.append(outcome)
        result.update(primitive_outcome=deepcopy(outcome),summary=self.summary());return result

class FrozenEpochPrimitivePilot:
    """Small runnable pilot; immutable original prediction/actual-feedback pairs.

    Later RTAB graph optimizations are retained in original packets but never
    retroactively recompute or relabel these already-observed Beta samples.
    Full geometry-history replay is deliberately not claimed by this pilot.
    """
    def __init__(self,configuration,feedback_output=None,**kwargs):
        self.configuration=deepcopy(configuration);self.inner=PrimitiveSession(configuration,**kwargs)
        self.directory=Path(feedback_output or tempfile.mkdtemp(prefix='nso-primitive-pilot-'));self.directory.mkdir(parents=True,exist_ok=True)
        self.snapshot=None;self.grid=None;self.records=[];self.pending_raw_goal=None;self.frozen_prediction_inputs={}
    def __getattr__(self,name):return getattr(self.inner,name)
    def identity(self):
        result=self.inner.identity();result['receipt']['geometry_pilot']=dict(profile='bootstrap_RTAB_epoch_frozen_actual_sensor_fusion_v1',raw_source_inputs_immutable=True,external_grid='actual bootstrap grid held; later new support only actual RGBD/scan',optimization_relearning=False,ROS_execution_goal_frame='odom',full_optimized_history_replay_implemented=False)
        return result
    def _derive(self,path):
        raw=Path(path).read_bytes()
        with np.load(path,allow_pickle=False)as z:arrays={k:z[k].copy()for k in z.files}
        meta=json.loads(str(arrays['joint_metadata_json'].item()))
        if self.snapshot is None:
            self.snapshot=deepcopy(meta['rtabmap_graph_snapshot']);self.grid=arrays['joint_belief_grid'].copy()
        updates,receipt=corrected_geometry(arrays,self.snapshot)
        for key in ['world_from_camera','world_from_laser','base_odom_pose']:arrays[key]=np.asarray(updates[key],float)
        arrays['joint_belief_grid']=self.grid.copy()
        meta['primitive_pilot_epoch']=dict(profile='bootstrap_RTAB_epoch_frozen_actual_sensor_fusion_v1',prediction_graph_revision=self.snapshot['graph_revision_sha256'],actual_acquisition_graph_revision=meta['rtabmap_graph_snapshot']['graph_revision_sha256'],raw_source_path=str(path),raw_source_sha256=hashlib.sha256(raw).hexdigest(),pixels_ranges_timestamps_unchanged=True,geometry=receipt,external_navigation_grid_held_at_actual_bootstrap=True,original_feedback_never_recomputed_after_graph_optimization=True)
        arrays['joint_metadata_json']=np.asarray(json.dumps(meta,sort_keys=True,allow_nan=False))
        output=self.directory/f'actual_{len(self.records):06d}.npz';np.savez_compressed(output,**arrays)
        return output,meta,receipt
    def observe_primitive_result(self,path,action_id,ros_status=None,reason=None):
        if action_id!=len(self.records):raise ValueError('one fresh sequential actual primitive observation required')
        derived,meta,receipt=self._derive(path)
        result=self.inner.observe_primitive_result(str(derived),action_id,ros_status=ros_status,reason=reason)
        actual_inputs=None
        if action_id:
            after=self.directory/f'actual_feedback_inputs_{action_id:06d}.npz'
            np.savez_compressed(after,belief_after=self.inner.mapper.belief,camera_seen_after=self.inner.mapper.camera_seen)
            actual_inputs=dict(path=str(after),sha256=hashlib.sha256(after.read_bytes()).hexdigest(),scope='actual sensor fusion in the same frozen prediction epoch; no optimized counterfactual replay')
            result['primitive_outcome']['frozen_prediction_inputs']=deepcopy(self.frozen_prediction_inputs.get(action_id));result['primitive_outcome']['frozen_actual_feedback_inputs']=actual_inputs
        record=dict(action_id=action_id,original_path=str(path),original_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),derived_path=str(derived),geometry=receipt,ros_status=ros_status,reason=reason,original_prediction_and_actual_feedback=deepcopy(result.get('primitive_outcome')),frozen_prediction_inputs=deepcopy(self.frozen_prediction_inputs.get(action_id)),frozen_actual_feedback_inputs=actual_inputs)
        self.records.append(record);(self.directory/'records.json').write_text(json.dumps(self.records,indent=2,allow_nan=False)+'\n')
        result['prediction_geometry_epoch']=self.snapshot['graph_revision_sha256'];result['historical_feedback_relearned']=False
        return result
    def next_action(self):
        result=self.inner.next_action();pending=result.get('pending')
        if pending is not None:
            # Original logical grid goal is in the frozen RTAB epoch. Convert it
            # back into raw odom before asking ROS; do not mislabel old map data
            # as the currently optimized live map. ROS transforms odom normally.
            stamp=self.records[-1]['geometry']['rgbd']['sensor_stamp_ns']
            correction,_=correction_at(self.snapshot,stamp)
            original=np.asarray(pending['target_odom_pose'],float)
            raw=np.linalg.inv(correction)@np.array([original[0],original[1],0.,1.]);yaw=original[2]-math.atan2(correction[1,0],correction[0,0])
            pending['target_prediction_epoch_pose']=original.tolist();pending['target_odom_pose']=[float(raw[0]),float(raw[1]),math.atan2(math.sin(yaw),math.cos(yaw))]
            pending['target_world_frame']='odom';pending['execution_goal_profile']='raw_odom_original_primitive_v1'
            pending['prediction_graph_revision']=self.snapshot['graph_revision_sha256']
            aid=pending['action_id'];gain=self.inner.backend.scenes[0]['gain'];row=gain._pending[aid]
            archive=self.directory/f'prediction_inputs_{aid:06d}.npz'
            np.savez_compressed(archive,radar_prediction=row['radar'],camera_prediction=row['camera'],known_before=row['known_before'],camera_before=row['camera_before'])
            self.frozen_prediction_inputs[aid]=dict(path=str(archive),sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),original_source_before_sha256=self.records[-1]['original_sha256'],prediction_graph_revision=self.snapshot['graph_revision_sha256'],posterior_before=deepcopy(gain.snapshot()['posterior_mean']),field_scope='original gain.prepare_action masks and actual before-support; immutable calibration evidence')
            pending['frozen_prediction_inputs']=deepcopy(self.frozen_prediction_inputs[aid])
        result['geometry_pilot_profile']='bootstrap_RTAB_epoch_frozen_actual_sensor_fusion_v1'
        return result
    def summary(self):
        result=self.inner.summary();result.update(geometry_pilot_profile='bootstrap_RTAB_epoch_frozen_actual_sensor_fusion_v1',original_actual_source_frames=len(self.records),historical_feedback_relearned=False,full_optimized_history_replay_implemented=False,original_records=str(self.directory/'records.json'));return result
