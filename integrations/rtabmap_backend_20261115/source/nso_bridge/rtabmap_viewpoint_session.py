"""Optional graph-replayed v4 representation. No new chooser, score or actuator.

Raw observations and original execution results remain immutable. The legacy
``odom`` field names below refer to the RTAB map alias only within this wrapper.
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import tempfile
import types
import numpy as np

PROFILE='rtabmap_optimized_replay_v1'
WORLD='nso_pair_map'

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False,separators=(',',':')).encode()).hexdigest()

def projected_anchor(configuration,base):
    """Re-express the same measured physical bootstrap; never move its pose."""
    T=np.asarray(configuration['planning_from_odom'],float)
    p=T@np.array([base[0],base[1],0.,1.]);h,w=configuration['shape']
    # Exact original GridTransform .2m, top-to-bottom rows, cardinal rounding.
    cell=[h-1-int(math.floor(p[1]/.2)),int(math.floor(p[0]/.2))]
    yaw=float(base[2])+math.atan2(T[1,0],T[0,0])
    heading=int(math.floor((math.pi/2-yaw)/(math.pi/2)+.5))%4
    return cell+[heading]

def outcome_counts(inner):
    rows=inner.waypoint_outcomes
    return dict(attempts=len(rows),ROS_successes=sum(x['ros_status']==3 for x in rows),
        statuses=[x['ros_status'] for x in rows],reasons=[x.get('reason') for x in rows])

def assert_conserved(before,after):
    if before!=after:raise ValueError('Graph replay changed recorded attempt/outcome counts')

class RtabmapViewpointSession:
    def __init__(self,configuration,*,replay_directory=None,session_factory=None,
                 geometry_corrector=None,graph_provider=None,**kwargs):
        if configuration.get('policy_variant')!='ros_viewpoint_exploration_v4_unbounded':
            raise ValueError('Graph replay wrapper supports declared v4 macro viewpoints only')
        if kwargs.get('shared_observed_prefix'):
            raise ValueError('Legacy same-continuous-odom free prefix is not graph-replay history')
        if session_factory is None:
            from nso_bridge.ros_viewpoint_unbounded_v4 import UnboundedViewpointSession
            session_factory=UnboundedViewpointSession
        self.configuration=deepcopy(configuration);self.kwargs=dict(kwargs)
        self._factory=session_factory;self._corrector=geometry_corrector;self._graph_provider=graph_provider
        self.directory=Path(replay_directory or tempfile.mkdtemp(prefix='nso-rtab-replay-'))
        self.directory.mkdir(parents=True,exist_ok=True)
        self.records=[];self.semantic_cache={};self.snapshot=None;self.revision=None
        self.current_grid=None;self._grid_revision=None;self.generation=0;self.replay_receipts=[];self.selection_history=[]
        self.inner=self._new_inner(self.configuration)
        self.frontend=getattr(self.inner,'semantic_frontend',None)
        self.semantic_configuration=deepcopy(getattr(self.inner,'semantic_configuration',self.kwargs.get('semantic_configuration')))
        self.semantic_capabilities={}
        backend=getattr(self.inner,'backend',None)
        if backend is not None:
            self.semantic_capabilities={k:deepcopy(v)for k,v in backend.capabilities.items()if k=='semantic'}
        if self.frontend is not None:
            self.semantic_capabilities['semantic']='real prompt instance segmentation on measured RGB-D; declared category prior; class accuracy, hidden structure and decision benefit unvalidated'
        self._restore_semantic_metadata(self.inner)
        self._attach_semantics(self.inner)
    def __getattr__(self,name):
        inner=self.__dict__.get('inner')
        if inner is None:raise AttributeError(name)
        return getattr(inner,name)
    def _new_inner(self,c):
        kwargs=dict(self.kwargs)
        # Reuse the already constructed actual frontend; graph replay never loads
        # another model and never re-infers any cached observation.
        if hasattr(self,'frontend'):kwargs['semantic_configuration']=None
        inner=self._factory(deepcopy(c),**kwargs)
        if 'frontend' in self.__dict__:
            inner.semantic_frontend=self.frontend
            self._restore_semantic_metadata(inner)
        return inner
    def _restore_semantic_metadata(self,inner):
        inner.semantic_configuration=deepcopy(self.semantic_configuration)
        backend=getattr(inner,'backend',None)
        if backend is not None:backend.capabilities.update(deepcopy(self.semantic_capabilities))
        components=getattr(inner,'components',None)
        if components is not None and hasattr(components,'capabilities'):
            components.capabilities.update(deepcopy(self.semantic_capabilities))
    def _attach_semantics(self,inner):
        original=inner._semantic_labels
        def labels(this,z,meta,source_sha256):
            fid=str(meta['frame_id'])
            if fid in self.semantic_cache:
                entry=self.semantic_cache[fid]
                this._instance_owner_map=None if entry['owner']is None else entry['owner'].copy()
                this._attempt_semantic_receipt=deepcopy(entry['receipt'])
                return entry['labels'].copy()
            value=original(z,meta,source_sha256)
            owner=getattr(this,'_instance_owner_map',None)
            self.semantic_cache[fid]=dict(labels=np.array(value,copy=True),
                owner=None if owner is None else np.array(owner,copy=True),
                receipt=deepcopy(getattr(this,'_attempt_semantic_receipt',None)))
            return value
        inner._semantic_labels=types.MethodType(labels,inner)
    def set_graph_snapshot(self,snapshot,current_navigation_grid=None):
        value=deepcopy(snapshot)
        if not value.get('graph_revision_sha256') or not value.get('nodes'):
            raise ValueError('Explicit nonempty optimized graph snapshot required')
        self.snapshot=value
        if current_navigation_grid is not None:
            grid=np.asarray(current_navigation_grid)
            if tuple(grid.shape)!=tuple(self.configuration['shape'])or not np.isin(grid,[-1,0,1]).all():
                raise ValueError('Current RTAB navigation grid must use the fixed planning lattice')
            self.current_grid=np.array(grid,copy=True);self._grid_revision=value['graph_revision_sha256']
    def _geometry(self,arrays,snapshot):
        if self._corrector is not None:return self._corrector(arrays,snapshot)
        from nso_bridge.rtabmap_geometry import corrected_geometry
        return corrected_geometry(arrays,snapshot)
    def _read(self,record):
        raw=Path(record['path']).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=record['source_sha256']:
            raise ValueError('Original observation changed before replay')
        with np.load(record['path'],allow_pickle=False)as z:return {k:z[k].copy()for k in z.files}
    def _derive(self,record,snapshot,folder,index,is_last):
        arrays=self._read(record);meta=json.loads(str(arrays['joint_metadata_json'].item()))
        updates,receipt=self._geometry(arrays,snapshot)
        for key in ('world_from_camera','world_from_laser','base_odom_pose'):
            arrays[key]=np.asarray(updates[key],float)
        arrays['joint_belief_grid']=self.current_grid.copy()if is_last else np.full(self.configuration['shape'],-1,np.int8)
        meta.update(geometry_world_frame=WORLD,planning_from_odom=self.configuration['planning_from_odom'],
            rtabmap_graph_snapshot=snapshot,rtabmap_replay=dict(profile=PROFILE,
                original_path=record['path'],original_sha256=record['source_sha256'],
                graph_revision_sha256=snapshot['graph_revision_sha256'],geometry=receipt,
                grid_role='latest_current_RTAB_navigation_grid'if is_last else'unknown_external_grid_actual_depth_scan_reintegration',
                source_headers_and_pixels_unchanged=True,semantic_reinference=False))
        arrays['joint_metadata_json']=np.asarray(json.dumps(meta,sort_keys=True,allow_nan=False))
        path=folder/f'{index:06d}.npz'
        np.savez_compressed(path,**arrays)
        return path,updates,receipt
    def _grid_from_observation(self,arrays,meta,snapshot):
        source_revision=meta.get('rtabmap_graph_snapshot',{}).get('graph_revision_sha256')
        if source_revision==snapshot['graph_revision_sha256']:
            self.set_graph_snapshot(snapshot,arrays['joint_belief_grid'])
        elif self._grid_revision!=snapshot['graph_revision_sha256']:
            raise ValueError('Updated graph requires its current corrected navigation grid, not the old observation grid')
    def _rebuild(self,records,snapshot):
        if self._grid_revision!=snapshot['graph_revision_sha256']:
            raise ValueError('Current navigation grid revision differs from graph revision')
        before=outcome_counts(self.inner)
        self.generation+=1;folder=self.directory/f'generation_{self.generation:06d}';folder.mkdir(exist_ok=False)
        derived=[];geometry_receipts=[];anchor_old=deepcopy(self.inner.configuration['return_anchor'])
        for i,row in enumerate(records):
            path,updates,receipt=self._derive(row,snapshot,folder,i,i==len(records)-1)
            derived.append(path);geometry_receipts.append(receipt)
            if i==0:bootstrap=updates['base_odom_pose']
        c=deepcopy(self.configuration);c['return_anchor']=projected_anchor(c,bootstrap)
        candidate=self._new_inner(c);self._attach_semantics(candidate);result=None
        for row,path in zip(records,derived):
            # Recorded real selected target is restored, no next_viewpoint called.
            if row['attempt_id']:
                candidate.viewpoint_pending=deepcopy(row['pending'])
            result=candidate.observe_viewpoint_result(str(path),row['attempt_id'],
                ros_status=row['ros_status'],reason=row['reason'])
            if row.get('original_outcome') and row['attempt_id']:
                candidate.waypoint_outcomes[-1]['recorded_execution_outcome']=deepcopy(row['original_outcome'])
                candidate.waypoint_outcomes[-1]['execution_verification_graph_revision']=row['acquisition_revision']
                if 'actual_goal_pose_verified' in row['original_outcome']:
                    candidate.waypoint_outcomes[-1]['replayed_discrete_cell_matches_old_target']=candidate.waypoint_outcomes[-1].get('actual_goal_pose_verified')
                    candidate.waypoint_outcomes[-1]['actual_goal_pose_verified']=row['original_outcome']['actual_goal_pose_verified']
        expected=deepcopy(before)
        new_count=sum(r['attempt_id']>0 for r in records)
        expected=dict(attempts=new_count,ROS_successes=sum(r['ros_status']==3 for r in records if r['attempt_id']),
            statuses=[r['ros_status']for r in records if r['attempt_id']],reasons=[r['reason']for r in records if r['attempt_id']])
        assert_conserved(expected,outcome_counts(candidate))
        # Keep real selection numbering/targets without replaying chooser calls.
        if hasattr(candidate,'backend') and getattr(candidate.backend,'scenes',None):
            candidate.backend.scenes[0]['plans']=len(self.selection_history)
        receipt=dict(profile=PROFILE,generation=self.generation,graph_revision_sha256=snapshot['graph_revision_sha256'],
            original_observations=len(records),free_prefix_imported=0,synthetic_observations=0,
            actual_selected_target_history=deepcopy(self.selection_history),selection_replay_calls=0,
            counts_before=before,counts_after=outcome_counts(candidate),expected_recorded_counts=expected,
            planning_frame_fixed=True,world_frame=WORLD,legacy_odom_field_means=WORLD,
            same_physical_bootstrap_anchor_reexpressed=dict(old=anchor_old,new=c['return_anchor']),
            historical_external_grid='unknown; actual original depth/scan only, fully reintegrated',
            latest_external_grid='current corrected RTAB navigation grid only',
            semantic_cache_frames=sorted(self.semantic_cache),geometry_receipts=geometry_receipts,
            original_method_equivalence=False,representation_profile_explicit=True)
        # Publish the new derived state only after every actual historical result replayed successfully.
        self.inner=candidate;self.records=records;self.revision=snapshot['graph_revision_sha256']
        self.replay_receipts.append(receipt)
        (folder/'receipt.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
        return result
    def observe_viewpoint_result(self,path,attempt_id,ros_status=None,reason=None):
        if type(attempt_id)is not int or attempt_id!=len(self.records):
            raise ValueError('Actual observations must be sequential; no retries/free-history import')
        if attempt_id and self.inner.viewpoint_pending is None:raise ValueError('Actual selected pending target required')
        raw=Path(path).read_bytes()
        with np.load(path,allow_pickle=False)as z:
            arrays={k:z[k].copy()for k in z.files}
        meta=json.loads(str(arrays['joint_metadata_json'].item()))
        if meta.get('geometry_world_frame')!=WORLD:raise ValueError('Collector must explicitly supply RTAB alias-frame geometry')
        snapshot=meta['rtabmap_graph_snapshot'];self.set_graph_snapshot(snapshot)
        self._grid_from_observation(arrays,meta,snapshot)
        record=dict(path=str(Path(path).resolve()),source_sha256=hashlib.sha256(raw).hexdigest(),
            attempt_id=attempt_id,ros_status=ros_status,reason=reason,
            pending=deepcopy(self.inner.viewpoint_pending),frame_id=str(meta['frame_id']),
            acquisition_revision=snapshot['graph_revision_sha256'])
        result=self._rebuild(self.records+[record],snapshot)
        self.records[-1]['original_outcome']=deepcopy(result.get('outcome'))
        result['rtabmap_replay']=deepcopy(self.replay_receipts[-1])
        return result
    def next_viewpoint(self,remaining_time_s=None,return_requested=False,graph_snapshot=None,current_navigation_grid=None):
        if graph_snapshot is None and self._graph_provider is not None:
            graph_snapshot,current_navigation_grid=self._graph_provider()
        if graph_snapshot is not None:self.set_graph_snapshot(graph_snapshot,current_navigation_grid)
        if self.records and self.snapshot['graph_revision_sha256']!=self.revision:
            if self.inner.viewpoint_pending is not None:raise ValueError('Graph rebuild only at an existing decision boundary, never replace an executing target')
            self._rebuild(self.records,self.snapshot)
        result=self.inner.next_viewpoint(remaining_time_s=remaining_time_s,return_requested=return_requested)
        if result.get('target') is not None:
            result['target']['target_world_frame']=WORLD
            self.inner.viewpoint_pending['target_world_frame']=WORLD
        if result.get('target') is not None:self.selection_history.append(deepcopy(result['target']))
        result['geometry_pose_backend']=PROFILE
        return result
    def identity(self):
        result=self.inner.identity()
        result['receipt']['rtabmap_optimized_replay']=dict(profile=PROFILE,world_frame=WORLD,
            legacy_planning_from_odom_means='fixed planning_from_RTABmap',legacy_target_odom_pose_means='RTABmap alias-frame target; coordinator must not send in odom',
            original_option_preserved=True,primitive_supported=False,old_free_prefix_allowed=False,
            original_method_equivalence=False,source_observations_immutable=True)
        return result
    def summary(self):
        result=self.inner.summary();result['rtabmap_replay']=dict(profile=PROFILE,revision=self.revision,
            recorded_observations=len(self.records),replayed_real_attempts=len(self.inner.waypoint_outcomes),
            frame=WORLD,generation=self.generation)
        return result
    def next_action(self):raise ValueError('RTAB replay profile is v4 viewpoint-only; no primitive execution')
