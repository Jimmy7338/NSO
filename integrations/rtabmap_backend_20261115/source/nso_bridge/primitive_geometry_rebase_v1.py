"""Staged atomic geometry rebase preserving actual feedback, never counterfactual Beta.

No active worker hook. Rebuilds only genuine archived observations/cached labels.
Historical admission, gains and arrivals remain their original recorded facts.
The next pending action may be prepared only after this atomic boundary. A
receipt exposes whether remaining literal actions still end at the selected
fixed-map target; no new chooser/target override/stop/route repair is hidden.
"""
from copy import deepcopy
import hashlib,json,tempfile,types
import os
from pathlib import Path
import numpy as np
from nso_bridge.primitive_feedback_unbounded_v1 import PrimitiveSession,FrozenEpochPrimitivePilot
from nso_bridge.rtabmap_geometry import corrected_geometry
from nso_bridge.rtabmap_viewpoint_session import projected_anchor
from nso.observed_feedback_v10 import _map_evidence
from nso.cpu_sensor_contract_v10 import digest
from utils.grid_geometry import DIRECTIONS
PROFILE='primitive_geometry_rebase_preserve_actual_feedback_v1'
STORAGE_PROFILE='transient_primitive_geometry_npz_v1'

def array_hash(value):
    value=np.asarray(value)
    return hashlib.sha256(str(value.dtype).encode()+str(value.shape).encode()+value.tobytes()).hexdigest()

def atomic_json(path,value):
    path=Path(path);temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');os.replace(temporary,path)

def original_arrays(record,keys=None):
    path=Path(record['original_path']);raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=record['original_sha256']:raise ValueError('original source changed before rebase')
    with np.load(path,allow_pickle=False)as z:
        return {k:z[k].copy()for k in (z.files if keys is None else keys)}

def regenerate_transient_archive(manifest_path,index,destination):
    """Reproduce discarded geometry inputs from retained originals and matrices."""
    manifest=json.loads(Path(manifest_path).read_text())
    if manifest.get('profile')!=STORAGE_PROFILE:raise ValueError('unsupported transient geometry manifest')
    item=manifest['archives'][index];arrays=original_arrays(item)
    for key,value in item['geometry_updates'].items():arrays[key]=np.asarray(value,float)
    if item['grid_role']=='unknown_historical':arrays['joint_belief_grid']=np.full(manifest['grid_shape'],-1,np.int8)
    else:
        source=manifest['current_grid_source'];path=Path(source['path'])
        if hashlib.sha256(path.read_bytes()).hexdigest()!=source['file_sha256']:raise ValueError('retained current-grid source changed')
        with np.load(path,allow_pickle=False)as z:grid=z[source['key']].copy()
        if array_hash(grid)!=manifest['current_grid_array_sha256']:raise ValueError('current-grid payload differs')
        arrays['joint_belief_grid']=grid
    meta=json.loads(str(arrays['joint_metadata_json'].item()));meta['primitive_geometry_rebase']=item['metadata_patch']
    arrays['joint_metadata_json']=np.asarray(json.dumps(meta,sort_keys=True,allow_nan=False))
    if list(arrays)!=item['array_order']:raise ValueError('source array order differs')
    np.savez(destination,**arrays)
    return dict(path=str(destination),encoded_sha256=hashlib.sha256(Path(destination).read_bytes()).hexdigest(),expected_encoded_sha256=item['transient_sha256'],encoded_byte_identical=hashlib.sha256(Path(destination).read_bytes()).hexdigest()==item['transient_sha256'])

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
    cache={}if semantic_cache is None else semantic_cache;plans=[];poses=[];geometries=[];cached={}
    # First pass reads only receipts and computes transforms. Keep no N-frame
    # raw image/depth array list and write no N-frame temporary archive prefix.
    for index,record in enumerate(pilot.records):
        path=Path(record['original_path']);arrays=original_arrays(record,['receipt_json','joint_metadata_json'])
        meta=json.loads(str(arrays['joint_metadata_json'].item()))
        source=meta.get('rtabmap_graph_snapshot',{})
        for key in ['source_database_path','source_epoch']:
            if source.get(key)and snapshot.get(key)and source[key]!=snapshot[key]:raise ValueError('cannot rebase across an unrelated map/DB epoch')
        updates,receipt=corrected_geometry(arrays,snapshot)
        patch=dict(profile=PROFILE,original_path=str(path),original_sha256=record['original_sha256'],graph_revision_sha256=snapshot['graph_revision_sha256'],actual_execution_and_gain_reobserved=False,grid_scope='latest_actual_current_grid'if index==len(pilot.records)-1 else'unknown_external_grid_original_RGBD_scan_only',geometry=receipt)
        plans.append(dict(updates=updates,metadata_patch=patch));poses.append(pilot.inner.planning_from_odom@updates['world_from_camera']);geometries.append(receipt)
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
    latest=pilot.records[-1];last_grid=original_arrays(latest,['joint_belief_grid'])['joint_belief_grid']
    if last_grid.dtype==current.dtype and np.array_equal(last_grid,current):
        grid_source=dict(path=latest['original_path'],file_sha256=latest['original_sha256'],key='joint_belief_grid',role='retained original last actual observation')
    else:
        grid_path=root/'retained_rebase_current_grid.npz';np.savez_compressed(grid_path,grid=current)
        grid_source=dict(path=str(grid_path),file_sha256=hashlib.sha256(grid_path.read_bytes()).hexdigest(),key='grid',role='retained supplied measured navigation-grid boundary input; not a new observation')
    manifest_path=root/'derived_geometry_manifest.json'
    manifest=dict(profile=STORAGE_PROFILE,numpy_version=np.__version__,encoding='np.savez ZIP_STORED for transient derived packets only',raw_sources_reencoded=False,grid_shape=list(current.shape),current_grid_source=grid_source,current_grid_array_sha256=array_hash(current),archives=[],only_transient_geometry_archives_discarded=True,prediction_actual_mask_and_source_archives_deleted=False)
    calls=[]
    for index,(record,plan)in enumerate(zip(pilot.records,plans)):
        arrays=original_arrays(record);meta=json.loads(str(arrays['joint_metadata_json'].item()))
        labels=read_cached_labels(pilot,arrays,meta,cache);cached[str(meta['frame_id'])]=labels
        for key,value in plan['updates'].items():arrays[key]=np.asarray(value,float)
        arrays['joint_belief_grid']=current.copy()if index==len(pilot.records)-1 else np.full(current.shape,-1,np.int8)
        meta['primitive_geometry_rebase']=plan['metadata_patch'];arrays['joint_metadata_json']=np.asarray(json.dumps(meta,sort_keys=True,allow_nan=False))
        path=root/f'geometry_{index:06d}.npz';np.savez(path,**arrays)
        archive=dict(original_path=record['original_path'],original_sha256=record['original_sha256'],transient_path=str(path),transient_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),transient_bytes=path.stat().st_size,array_order=list(arrays),geometry_updates={k:v.tolist()for k,v in plan['updates'].items()},metadata_patch=plan['metadata_patch'],grid_role='latest_current'if index==len(pilot.records)-1 else'unknown_historical',discarded_after_successful_fusion=False)
        manifest['archives'].append(archive);atomic_json(manifest_path,manifest)
        if index:
            candidate.backend.scenes[0]['gain'].prepare_action=forbidden
            candidate.backend.scenes[0]['gain'].observe=forbidden
            candidate.backend.scenes[0]['ledger'].observe=forbidden
        action=None if index==0 else before['primitive_outcomes'][index-1]['action']
        calls.append(candidate.observe(str(path),index,action=action))
        # observe() has closed its NPZ and materialized the frame/scan payloads.
        # Keep source archives, actual feedback/prediction masks and all receipts.
        path.unlink();archive['discarded_after_successful_fusion']=True;atomic_json(manifest_path,manifest)
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
    receipt['derived_geometry_storage']=dict(profile=STORAGE_PROFILE,manifest=str(manifest_path),encoding='uncompressed transient NPZ; original sensor archives unchanged',transient_archives_discarded=len(manifest['archives']),retained_transient_geometry_archives=0,largest_transient_archive_bytes=max(a['transient_bytes']for a in manifest['archives']),retained_original_sources=len(pilot.records),derived_zip_hashes_changed_by_encoding=True,canonical_arrays_and_array_order_unchanged=True,reproducible_from_original_sources_and_retained_matrices_grid=True,no_TSDF_or_mapper_clone=True)
    (root/'receipt.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
    if stats_fingerprint(pilot.inner)!=before:raise ValueError('old active state mutated during rebuild')
    return new,receipt
