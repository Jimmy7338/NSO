"""Common observed-only facility representation; never a planning interface.

Actual RGB marker geometry may associate measured surfaces here, but neither
these seeds nor this evaluator may enter G/S runtime state. No world is read
until evaluate(), which operates after terminal snapshot and never corrects
predictions using truth. Missing/extra detections are reported, not repaired.
"""
from copy import deepcopy
import numpy as np
import open3d as o3d
from nso.cpu_sensor_contract_v10 import SensorPacket, json_value, digest
from nso.decision_replay_v13 import array_hash
from scripts.probe_facility_choice_v25r1 import MeasuredOnlyBackend
from scripts.probe_facility_choice_v24_prefix import MarkerTracks, geometry_evidence_sha
from scripts.replay_facility_choice_shape_v24 import component_seed_hits, fixed_seed_association, mesh_hash
from scripts.probe_facility_shape_v23 import mesh_hashes
from env.virtual3d_inspection_v4 import MARKER_COLORS


VERSION = 'facility-measured-only-v26-1'
TRACK_CONFIG = dict(marker_minimum_valid_pixels_per_component=16,
    marker_track_association_distance_m=.75, minimum_supported_distinct_poses_per_track=2)
OUTLINE_CONFIG = dict(boundary_spacing_m=.01, thresholds=[.02,.05,.10],
    completion_tolerance_m=.05, completion_minimum_iou=.9)
TASK_ASSET_COUNT = 2


class FacilityMeasurementV26:
    """Two fixed measurement slots receiving every actually obtained frame.

    ``observe`` returns a JSON-friendly receipt; an exact latest retry is
    idempotent. The first packet must be action 0, then contiguous paid actions.
    ``snapshot`` is terminal: no further observe calls are allowed. It extracts
    the mapper mesh once and invokes the frozen V24 observed cleanup unchanged.
    """
    def __init__(self):
        self.backends = [MeasuredOnlyBackend() for _ in range(TASK_ASSET_COUNT)]
        self.tracks = MarkerTracks(dict(TRACK_CONFIG))
        self.seeds = [None]*TASK_ASSET_COUNT
        self.extra_observed_track_receipts = []
        self.frame_count = 0
        self._identity = None; self._last_action = None; self._last_time = None
        self._last_frame = None; self._last_packet_sha = None; self._last_receipt = None
        self._frame_ids = set(); self._terminal = False; self._poisoned = False

    def observe(self, packet):
        if self._terminal or self._poisoned:
            raise ValueError('measurement history already finalized or failed')
        if not isinstance(packet, SensorPacket):
            raise ValueError('actual SensorPacket required')
        if (type(packet.action_id) is not int or packet.action_id < 0
                or any(not isinstance(x,str) or not x for x in
                    (packet.scene_id,packet.episode_id,packet.frame_id,packet.sensor_source,packet.pose_source))):
            raise ValueError('valid sensor identity and nonnegative paid action required')
        packet.frame.validate()
        if packet.frame.color_rgb.dtype != np.uint8 or not np.issubdtype(packet.frame.depth_m.dtype,np.floating):
            raise ValueError('aligned uint8 RGB and floating depth required')
        timestamp=float(packet.frame.timestamp_s); identity=(packet.scene_id,packet.episode_id)
        packet_sha=packet.sha256()
        if self._identity is None:
            if packet.action_id != 0:
                raise ValueError('complete measurement history must start at action 0')
        else:
            if identity != self._identity:
                raise ValueError('measurement history cannot cross scene/episode identity')
            if packet.action_id == self._last_action and packet_sha == self._last_packet_sha:
                return deepcopy(self._last_receipt)
            if (packet.action_id != self._last_action+1 or timestamp <= self._last_time
                    or packet.frame_id in self._frame_ids):
                raise ValueError('unique contiguous chronological sensor history required')
        # All validation above is before any tracking/backend state change.
        try:
            components=self.tracks.consume(packet)
            hits=component_seed_hits(packet.frame,TRACK_CONFIG['marker_minimum_valid_pixels_per_component'])
            if len(components) != len(hits):
                raise ValueError('binary marker component order inconsistent')
            new_seeds={}; first_receipts=[]; extra_receipts=[]
            for component,hit in zip(components,hits):
                slot=component['observed_track_index']
                if slot >= TASK_ASSET_COUNT:
                    extra=dict(action_id=packet.action_id,frame_id=packet.frame_id,
                        packet_sha256=packet_sha,observed_track_index=slot,**hit)
                    self.extra_observed_track_receipts.append(extra);extra_receipts.append(extra)
                    continue
                if self.seeds[slot] is None:
                    rr,cc=hit['pixel']; color=packet.frame.color_rgb[rr,cc]
                    codes=[int(key) for key,value in MARKER_COLORS.items() if np.array_equal(color,value)]
                    if len(codes) != 1:
                        raise ValueError('seed is not an actual known-color marker pixel')
                    seed=dict(action_id=packet.action_id,observation_id=packet.action_id,
                        frame_id=packet.frame_id,observed_slot=slot,packet_sha256=packet_sha,
                        actual_marker_code=codes[0],**hit)
                    self.seeds[slot]=seed;new_seeds[slot]=hit['observed_seed_xyz'];first_receipts.append(seed)
            frame=packet.frame
            for slot,backend in enumerate(self.backends):
                if not backend.observe(frame.depth_m,frame.intrinsic,frame.world_from_camera,
                        observation_id=packet.action_id,observed_seed_xyz=new_seeds.get(slot)):
                    raise ValueError('duplicate frame in measured backend')
        except Exception:
            # Partial mutation must not be silently reused as a valid history.
            self._poisoned=True
            raise
        self.frame_count+=1;self._identity=identity;self._last_action=packet.action_id
        self._last_time=timestamp;self._last_frame=packet.frame_id;self._last_packet_sha=packet_sha
        self._frame_ids.add(packet.frame_id)
        receipt=dict(action_id=packet.action_id,frame_id=packet.frame_id,packet_sha256=packet_sha,
            observed_frames=self.frame_count,marker_components=components,
            first_seed_receipts=first_receipts,extra_track_receipts=extra_receipts,
            discovered_seed_slots=[i for i,seed in enumerate(self.seeds) if seed is not None],
            world_or_reference_input=False)
        self._last_receipt=json_value(receipt)
        return deepcopy(self._last_receipt)

    def snapshot(self, mapper):
        if self._terminal or self._poisoned:
            raise ValueError('terminal snapshot is single-use and requires an intact history')
        if mapper.frames != self.frame_count:
            raise ValueError('mapper and common measurement frame counts disagree')
        raw=mapper.mesh(); rows=[];meshes=[]
        for slot,backend in enumerate(self.backends):
            snap=backend.snapshot(raw_mesh=raw)
            observed=snap['observed_mesh']; observed_hash=mesh_hash(observed)
            if (len(snap['inferred_mesh'].triangles) or len(snap['inferred_mesh'].vertices)
                    or observed_hash != mesh_hash(snap['completed_mesh'])):
                raise ValueError('frozen measured-only backend unexpectedly produced inference')
            arrays={key:dict(count=len(snap[key]),sha256=array_hash(snap[key])) for key in
                ('measured_points_xyz','ground_points_xyz','cleaned_points_xyz','unassigned_points_xyz')}
            if self.seeds[slot] is None and len(observed.triangles):
                raise ValueError('unseeded backend unexpectedly assigned measured geometry')
            rows.append(dict(observed_slot=slot,seed=deepcopy(self.seeds[slot]),
                observed_frames=snap['observed_frames'],ground=snap['ground_plane'],
                geometry_arrays=arrays,observed_mesh_sha256=observed_hash,
                observed_mesh_size=dict(vertices=len(observed.vertices),triangles=len(observed.triangles)),
                missing_seed=self.seeds[slot] is None,missing_observed_mesh=not len(observed.triangles),
                inferred_mesh_disabled=True,unseen_surfaces_measured=False,
                completion=deepcopy(snap['completion'])))
            meshes.append(observed)
        duplicate=bool(rows[0]['geometry_arrays']['cleaned_points_xyz']['count']>0 and
            rows[0]['geometry_arrays']['cleaned_points_xyz']==rows[1]['geometry_arrays']['cleaned_points_xyz'])
        nonempty=all(row['geometry_arrays']['cleaned_points_xyz']['count']>0 and
            row['observed_mesh_size']['triangles']>0 for row in rows)
        tracks=self.tracks.summary()
        support=bool(len(tracks)==TASK_ASSET_COUNT and all(
            t['max_valid_pixels']>=TRACK_CONFIG['marker_minimum_valid_pixels_per_component'] and
            t['distinct_paid_poses']>=TRACK_CONFIG['minimum_supported_distinct_poses_per_track'] for t in tracks))
        metadata=json_value(dict(version=VERSION,task_asset_count=TASK_ASSET_COUNT,
            observed_frames=self.frame_count,last_action_id=self._last_action,
            last_frame_id=self._last_frame,last_packet_sha256=self._last_packet_sha,
            instances=rows,raw_mesh_sha256=mesh_hashes(raw),
            class_stripped_mapper_geometry_sha256=geometry_evidence_sha(mapper),
            observed_track_summary=tracks,extra_observed_track_receipts=self.extra_observed_track_receipts,
            missing_seed_slots=[i for i,seed in enumerate(self.seeds) if seed is None],
            missing_observed_mesh_slots=[row['observed_slot'] for row in rows if row['missing_observed_mesh']],
            duplicate_seeded_components=duplicate,all_observed_instances_nonempty=nonempty,
            minimum_instance_separation_gate_passed=nonempty and not duplicate,
            minimum_separation_scope='nonempty/different observed components only; not validated semantic segmentation',
            observed_marker_support_gate_passed=support,marker_support_rule=dict(TRACK_CONFIG),
            marker_support_is_not_mission_qualification=True,
            snapshot_complete_before_reference=True,inference_disabled=True,
            backend_observe_calls=TASK_ASSET_COUNT*self.frame_count,backend_snapshots=TASK_ASSET_COUNT,
            planner_must_not_receive_marker_association=True,world_or_reference_input=False))
        self._terminal=True
        return dict(metadata=metadata,metadata_sha256=digest(metadata),raw_mesh=raw,observed_meshes=tuple(meshes))

    @staticmethod
    def evaluate(snapshot, reference_world, coverage, returned, collisions, failed, paid_actions, budget=400):
        """Evaluation-only; missing window geometry scores zero, absence alone never raises.

        Qualification stays the frozen budget/C80/return/collision/failure rule.
        Seed association, minimum separation, foreign-window geometry and marker
        support are explicit separate diagnostics, never truth-based repairs.
        An unseeded slot is empty, but another slot's foreign-window geometry
        can still contribute to the global window score; that is NOT successful
        instance attribution and remains explicitly flagged.
        """
        from utils.facility_outline_v23 import OutlineEvaluatorV23
        metadata=snapshot['metadata'];meshes=snapshot['observed_meshes'];raw=snapshot['raw_mesh']
        if digest(metadata)!=snapshot.get('metadata_sha256'):
            raise ValueError('pre-reference measurement metadata changed after snapshot')
        if type(returned) is not bool or type(failed) is not bool:
            raise ValueError('returned and failed must be actual booleans')
        if (metadata.get('version')!=VERSION or not metadata.get('snapshot_complete_before_reference')
                or len(meshes)!=TASK_ASSET_COUNT or len(metadata['instances'])!=TASK_ASSET_COUNT):
            raise ValueError('complete two-slot terminal measurement snapshot required')
        if mesh_hashes(raw)!=metadata['raw_mesh_sha256'] or any(
                mesh_hash(mesh)!=row['observed_mesh_sha256'] for mesh,row in zip(meshes,metadata['instances'])):
            raise ValueError('measurement geometry changed after terminal snapshot')
        if len(reference_world.objects)!=TASK_ASSET_COUNT:
            raise ValueError('fixed task reference must retain both facilities')
        evaluator=OutlineEvaluatorV23.from_world(reference_world,**OUTLINE_CONFIG)
        args=dict(coverage=coverage,returned=returned,collisions=collisions,failed=failed,
            paid_actions=paid_actions,budget=budget)
        association=fixed_seed_association([row['seed'] for row in metadata['instances']],
            reference_world.objects,len(metadata['observed_track_summary']))
        joined=o3d.geometry.TriangleMesh()
        for mesh in meshes:joined+=mesh
        main=evaluator.evaluate(joined,**args);secondary=evaluator.evaluate(raw,**args)
        rows=deepcopy(metadata['instances']);foreign=[]
        for slot,row in enumerate(rows):
            scores=evaluator.evaluate(meshes[slot],**args)['instances'];match=association['rows'][slot]['reference_id']
            row.update(per_reference_window=scores,fixed_seed_association=association['rows'][slot],
                foreign_window_nonempty_reference_ids=[x['id'] for x in scores if not x['missing'] and x['id']!=match])
            foreign+=row['foreign_window_nonempty_reference_ids']
        return json_value(dict(main_observed=main,raw_secondary=secondary,
            reference_signature=evaluator.reference_signature,instances=rows,
            union_observed_mesh_sha256=mesh_hash(joined),measurement_metadata_sha256=digest(metadata),
            fixed_seed_association=association,foreign_window_nonempty_reference_ids=foreign,
            evaluation_association_and_minimum_separation_passed=bool(association['seed_association_gate_passed']
                and metadata['minimum_instance_separation_gate_passed'] and not foreign),
            instance_association_gate_scope='fixed seed mapping/minimum separation/no other task-window geometry; no truth-based correction',
            observed_marker_support_gate_passed=metadata['observed_marker_support_gate_passed'],
            missing_seed_slots=metadata['missing_seed_slots'],missing_observed_mesh_slots=metadata['missing_observed_mesh_slots'],
            missing_is_zero_score_not_runtime_exception=True,inference_disabled=True,
            reference_used_after_snapshot_only=True,geometry_returned_to_planner=False))
