"""Staged atomic geometry rebase preserving actual feedback, never counterfactual Beta.

No active worker hook. Rebuilds only genuine archived observations/cached labels.
Historical admission, gains and arrivals remain their original recorded facts.
The next pending action may be prepared only after this atomic boundary. A
receipt exposes whether remaining literal actions still end at the selected
fixed-map target; no new chooser/target override/stop/route repair is hidden.
"""
from copy import deepcopy
import hashlib,json,tempfile,types
from pathlib import Path
import numpy as np
from nso_bridge.primitive_feedback_unbounded_v1 import PrimitiveSession,FrozenEpochPrimitivePilot
from nso_bridge.rtabmap_geometry import corrected_geometry
from nso_bridge.rtabmap_viewpoint_session import projected_anchor
from nso.observed_feedback_v10 import _map_evidence
from nso.cpu_sensor_contract_v10 import digest
from utils.grid_geometry import DIRECTIONS
PROFILE='primitive_geometry_rebase_preserve_actual_feedback_v1'

def stats_fingerprint(inner):
    scene=inner.backend.scenes[0];state=inner.runtime.states[0];ledger=scene['ledger'];gain=scene['gain']
    return dict(gain_snapshot=deepcopy(gain.snapshot()),ledger_events=deepcopy(ledger._events),paid_actions=ledger._paid_actions,feedback_version=ledger._version,planning_feedback_version=ledger._planning_version,action_ids=sorted(ledger._actions),source_frame_ids=[x[1]for x in ledger._frames],arrived_count=state['arrived_count'],primitive_outcomes=deepcopy(inner.primitive_outcomes),remaining_committed_actions=deepcopy(state['active_actions']),selected_option=deepcopy(state['option']),global_plans=scene['plans'])

def literal_prefix_endpoint(position,heading,actions):
    row,col=position
    for action in actions:
        if action=='forward':
            dr,dc=DIRECTIONS[heading];row+=int(dr);col+=int(dc)
        elif action=='left':heading=(heading-1)%4
        elif action=='right':heading=(heading+1)%4
        else:raise ValueError('unknown recorded original primitive')
    return [row,col,heading]

class GeometryOnlySession(PrimitiveSession):
    """Internal unexposed builder; historical fusion invokes no feedback method."""
    def _fuse_packet(self,packet,frame,scan,action_id,action,collision,done):
        if getattr(self,'_geometry_rebuild_active',True) is False:
            return super()._fuse_packet(packet,frame,scan,action_id,action,collision,done)
        if self.last_packet is None:
            result=super()._fuse_packet(packet,frame,scan,action_id,action,collision,done)
            self._bootstrap_geometry_evidence=_map_evidence(self.mapper)
            return result
        state=self.runtime.states[0];previous=self.last_packet
        if (packet.action_id!=previous.action_id+1 or packet.frame_id in state['packet_hashes']
            or packet.frame.timestamp_s<=previous.frame.timestamp_s):raise ValueError('original actual history order/hash differs')
        self.mapper.update(frame,scan)
        state['packet']=packet;state['packet_hashes'][packet.frame_id]=packet.sha256();state['action_frames'][packet.action_id]=packet.frame_id
        self.backend.bind_packet(0,packet,arrived=False)
        self.runtime._update_sensor_modules(0,initial=True)  # Geometry/semantics/topology only.
        self.last_packet=packet
        if self._attempt_semantic_receipt is not None:self.semantic_receipt=deepcopy(self._attempt_semantic_receipt)
        return dict(accepted=True,geometry_replay_only=True,original_execution_reassessed=False,gain_observe_calls=0)

def read_cached_labels(pilot,arrays,meta,cache):
    fid=str(meta['frame_id'])
    if fid in cache:return cache[fid]
    original=pilot.inner
    if original.semantic_frontend is None and original.semantic_configuration is None:
        labels=arrays['semantic'].copy()if'semantic'in arrays else np.zeros_like(arrays['depth'],dtype=np.int32)
        return dict(labels=labels,owner=None,receipt=None)
    folder=original.semantic_output
    if folder is None:raise ValueError('real frontend history requires actual archived mask receipts')
    matches=[]
    for path in Path(folder).glob('*.json'):
        receipt=json.loads(path.read_text())
        if str(receipt.get('frame_id'))==fid:matches.append(receipt)
    if len(matches)!=1:raise ValueError('one exact cached real semantic receipt required for '+fid)
    receipt=matches[0];archive=Path(receipt['mask_file'])
    if hashlib.sha256(archive.read_bytes()).hexdigest()!=receipt['mask_file_sha256']:raise ValueError('original cached masks changed')
    if hashlib.sha256(arrays['rgb_raw'].tobytes()).hexdigest()!=receipt['rgb_sha256']:raise ValueError('semantic RGB source differs')
    with np.load(archive,allow_pickle=False)as z:
        labels=z['semantic'].copy();owner=z['instance_owner'].copy()if'instance_owner'in z else None
    if hashlib.sha256(labels.tobytes()).hexdigest()!=receipt['canonical_semantic_sha256']:raise ValueError('original canonical labels changed')
    return dict(labels=labels,owner=owner,receipt=receipt)

def restore_frozen_feedback(old,new,corrected_camera_poses):
    """Move original evidence state intact, rebase future geometry baselines only."""
    old_scene=old.backend.scenes[0];new_scene=new.backend.scenes[0];old_state=old.runtime.states[0];new_state=new.runtime.states[0]
    if old_state['pending']is not None or old_scene['gain']._pending:raise ValueError('geometry rebase requires an actual terminal-feedback boundary')
    if len(corrected_camera_poses)!=len(old_state['packet_hashes'])or set(new_state['packet_hashes'])!=set(old_state['packet_hashes']):raise ValueError('rebase changed original actual frame identities')
    baseline=_map_evidence(new.mapper);boot=new._bootstrap_geometry_evidence
    ledger=deepcopy(old_scene['ledger']);gain=deepcopy(old_scene['gain'])
    # Source-time calibration events/counts remain exact originals. Geometry
    # baselines prevent graph correction becoming future "new observation".
    ledger._explored_ever=baseline['explored'].copy();ledger._camera_ever=baseline['camera'].copy();ledger._support_ever=set(baseline['supported'])
    ledger._old_keys=frozenset(boot['supported']);ledger._old_quality=ledger._old_mean(baseline['quality']);ledger._initial_old_quality=ledger._old_mean(boot['quality'])
    ledger._poses=[np.asarray(x,float).copy()for x in corrected_camera_poses]
    planning_count=len(ledger._planning_poses);ledger._planning_poses=[x.copy()for x in ledger._poses[:planning_count]]
    ledger._last_map_version=new.mapper.frames
    if gain._pending:raise ValueError('a historical prediction was left pending')
    new_scene['ledger']=ledger;new_scene['gain']=gain;new_scene['feedback']=deepcopy(old_scene['feedback']);new_scene['gain_feedback']=deepcopy(old_scene['gain_feedback'])
    new_scene['planning_direction_support']=deepcopy(new_scene['current_direction_support'])
    for key in ['plans','selected','last_selection','execution_option']:
        if key in old_scene:new_scene[key]=deepcopy(old_scene[key])
    for key in ['phase','active_actions','option','pending','closed','arrived_count','externally_scripted_prefix_actions','termination']:
        if key in old_state:new_state[key]=deepcopy(old_state[key])
    new.primitive_outcomes=deepcopy(old.primitive_outcomes)
    new.runtime.audit=deepcopy(old.runtime.audit)
    # Freeze all original observed statistics exactly; future geometric proxy
    # baselines are separately labeled, never substituted into old events.
    if stats_fingerprint(old)!=stats_fingerprint(new):raise ValueError('geometry rebase altered actual feedback or execution records')

def rebuild_geometry_preserving_actual_feedback(pilot,snapshot,current_navigation_grid,*,output=None,semantic_cache=None):
    """Return an unexposed new candidate; caller swaps only after receipt checks."""
    before=stats_fingerprint(pilot.inner)
    before_ledger_snapshot=deepcopy(pilot.inner.backend.scenes[0]['ledger'].snapshot())
    if not pilot.records:raise ValueError('actual bootstrap history required')
    if pilot.inner.runtime.states[0]['pending']is not None:raise ValueError('do not change an executing prediction epoch')
    current=np.asarray(current_navigation_grid)
    if current.shape!=tuple(pilot.configuration['shape'])or not np.isin(current,[-1,0,1]).all():raise ValueError('actual latest grid must match fixed planning lattice')
    root=Path(output or tempfile.mkdtemp(prefix='nso-primitive-geometry-rebase-'));root.mkdir(parents=True,exist_ok=True)
    cache={}if semantic_cache is None else semantic_cache;derived=[];poses=[];geometries=[];cached={}
    for index,record in enumerate(pilot.records):
        path=Path(record['original_path']);raw=path.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=record['original_sha256']:raise ValueError('original source changed before rebase')
        with np.load(path,allow_pickle=False)as z:arrays={k:z[k].copy()for k in z.files}
        meta=json.loads(str(arrays['joint_metadata_json'].item()))
        source=meta.get('rtabmap_graph_snapshot',{})
        for key in ['source_database_path','source_epoch']:
            if source.get(key)and snapshot.get(key)and source[key]!=snapshot[key]:raise ValueError('cannot rebase across an unrelated map/DB epoch')
        labels=read_cached_labels(pilot,arrays,meta,cache);cached[str(meta['frame_id'])]=labels
        updates,receipt=corrected_geometry(arrays,snapshot)
        for key in ['world_from_camera','world_from_laser','base_odom_pose']:arrays[key]=np.asarray(updates[key],float)
        arrays['joint_belief_grid']=current.copy()if index==len(pilot.records)-1 else np.full(current.shape,-1,np.int8)
        meta['primitive_geometry_rebase']=dict(profile=PROFILE,original_path=str(path),original_sha256=record['original_sha256'],graph_revision_sha256=snapshot['graph_revision_sha256'],actual_execution_and_gain_reobserved=False,grid_scope='latest_actual_current_grid'if index==len(pilot.records)-1 else'unknown_external_grid_original_RGBD_scan_only',geometry=receipt)
        arrays['joint_metadata_json']=np.asarray(json.dumps(meta,sort_keys=True,allow_nan=False))
        path_out=root/f'geometry_{index:06d}.npz';np.savez_compressed(path_out,**arrays);derived.append(path_out);poses.append(pilot.inner.planning_from_odom@updates['world_from_camera']);geometries.append(receipt)
        if index==0:bootstrap=updates['base_odom_pose']
    config=deepcopy(pilot.configuration);anchor=projected_anchor(config,bootstrap);config['return_anchor']=anchor
    old=pilot.inner
    candidate=GeometryOnlySession(config,execution_allowed=old.execution_allowed,calibration=deepcopy(old.calibration),semantic_configuration=None,semantic_output=old.semantic_output)
    candidate.semantic_frontend=old.semantic_frontend;candidate.semantic_configuration=deepcopy(old.semantic_configuration)
    candidate.backend.capabilities.update(deepcopy(old.backend.capabilities));candidate.components.capabilities.update(deepcopy(old.components.capabilities))
    original_live_semantics=candidate._semantic_labels
    def actual_cached(this,z,meta,sha):
        fid=str(meta['frame_id'])
        if fid not in cached:
            if getattr(this,'_geometry_rebuild_active',True):raise ValueError('historical semantic cache missing; no reinference allowed')
            return original_live_semantics(z,meta,sha)
        item=cached[fid];this._instance_owner_map=None if item['owner']is None else item['owner'].copy();this._attempt_semantic_receipt=deepcopy(item['receipt']);return item['labels'].copy()
    candidate._semantic_labels=types.MethodType(actual_cached,candidate)
    # Forbidden during reconstruction: no predictions, no gain observations and
    # no ledger.observe. Every history item is geometry-only actual fusion.
    def forbidden(*args,**kwargs):raise RuntimeError('counterfactual feedback called during geometry replay')
    calls=[]
    for index,(record,path)in enumerate(zip(pilot.records,derived)):
        if index:
            candidate.backend.scenes[0]['gain'].prepare_action=forbidden
            candidate.backend.scenes[0]['gain'].observe=forbidden
            candidate.backend.scenes[0]['ledger'].observe=forbidden
        action=None if index==0 else before['primitive_outcomes'][index-1]['action']
        calls.append(candidate.observe(str(path),index,action=action))
    restore_frozen_feedback(old,candidate,poses)
    after=stats_fingerprint(candidate)
    after_ledger_snapshot=candidate.backend.scenes[0]['ledger'].snapshot()
    geometry_snapshot_changes={k:dict(before=before_ledger_snapshot.get(k),after=after_ledger_snapshot.get(k))for k in set(before_ledger_snapshot)|set(after_ledger_snapshot)if before_ledger_snapshot.get(k)!=after_ledger_snapshot.get(k)}
    allowed_geometry_fields={'actual_camera_poses_sha256','planning_camera_poses_sha256','fixed_old_support_count','fixed_old_support_sha256','initial_old_quality_proxy','old_quality_proxy','old_quality_proxy_total_change','observed_2d_area_m2'}
    if set(geometry_snapshot_changes)-allowed_geometry_fields:raise ValueError('geometry replay changed a frozen ledger field: '+str(set(geometry_snapshot_changes)-allowed_geometry_fields))
    candidate._geometry_rebuild_active=False
    new=FrozenEpochPrimitivePilot.__new__(FrozenEpochPrimitivePilot)
    new.configuration=deepcopy(config);new.inner=candidate;new.directory=root;new.snapshot=deepcopy(snapshot);new.grid=current.copy();new.records=deepcopy(pilot.records);new.pending_raw_goal=None;new.frozen_prediction_inputs=deepcopy(pilot.frozen_prediction_inputs)
    packet=candidate.last_packet;state=candidate.runtime.states[0];endpoint=literal_prefix_endpoint(packet.position,packet.heading,state['active_actions']);option=state.get('option');target=None if option is None else option['pose']
    receipt=dict(profile=PROFILE,status='geometry_rebase_candidate_not_activated',records=len(pilot.records),source_files_unchanged=True,semantic_reinference=False,actual_sensor_fusions=len(calls),gain_prepare_calls=0,gain_observe_calls=0,ledger_observe_calls=0,frozen_feedback_exact=before==after,before_feedback_sha256=digest(before),after_feedback_sha256=digest(after),graph_revision=snapshot['graph_revision_sha256'],actual_geometry_receipts=geometries,bootstrap_anchor_reexpression=dict(old=old.configuration['return_anchor'],new=anchor,same_physical_first_observation=True,automatic_return=False),historical_arrival_credit_changed=False,history_camera_poses_rebased=True,remaining_actions_preserved=state['active_actions']==before['remaining_committed_actions'],original_selected_target_unchanged=True,literal_prefix_endpoint_from_rebased_pose=endpoint,selected_fixed_map_target=target,literal_prefix_still_reaches_selected_target=target is None or endpoint==target,no_extra_candidate_filter_or_score_or_stop_or_target_override=True,activation_limitation='If a loop correction changes the grid state, unchanged literal remaining actions may no longer end at the fixed-map selected target. This receipt surfaces it; caller must review a declared same-target route repair instead of hidden reselect/target mutation.',geometry_derived_ledger_snapshot_changes=geometry_snapshot_changes,frozen_ledger_event_statistics_exact=True,future_actual_feedback_enabled=True,future_geometry_baselines='Current rebuilt support/masks/initial physical support rebase; source-time feedback events remain frozen. Changed current reconstruction proxies are not new measured action yield or claimed method improvement.')
    (root/'receipt.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
    if stats_fingerprint(pilot.inner)!=before:raise ValueError('old active state mutated during rebuild')
    return new,receipt
