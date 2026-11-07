"""Standalone JSONL inference adapter for the preserved unknown/surface V10 policy.

The original policy remains in byte-identical vendored sources. An explicitly
selected exploration_joint_v1 policy is implemented separately and reports its
method changes. Physical input differs explicitly from the simulator packet:
actual K, sensor extrinsics and independent source timestamps remain measured;
base pose alone is discretized to the existing 0.2 m/NESW planning state. No
world, scene mesh, truth map, evaluator or hidden semantic label is accepted.
"""
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

VENDOR = Path(__file__).resolve().parents[1] / 'joint_vendor'
# Original modules use absolute nso/env/utils imports. Keep this engine in its
# own worker process rather than mixing a V35 module cache into the namespace.
if 'nso' in sys.modules and not str(getattr(sys.modules['nso'], '__file__', '')).startswith(str(VENDOR)):
    raise RuntimeError('joint engine requires its standalone original-source worker namespace')
sys.path.insert(0, str(VENDOR))
import numpy as np
from env.virtual3d_v2 import VirtualConfigV2
from nso.cpu_four_modules_v10 import CPUFourModules
from nso.cpu_sensor_contract_v10 import GridTransform, SensorPacket, json_value
from nso.observed_runtime_mapper_v10 import ObservedRuntimeMapperV10
from nso.runtime_integration import NSORuntimeIntegration
from utils.rgbd_contract import RGBDFrame, PlanarScan
from nso_bridge.sensor_range_adapter import validate_profile, adapt_scan, range_identity
from nso_bridge.measured_extent_adapter import (validate_profile as validate_extent_profile,
    install_dispatcher as install_extent_dispatcher, profile_receipt as extent_receipt)


def _rigid(value, name):
    a = np.asarray(value, float)
    if (a.shape != (4, 4) or not np.isfinite(a).all()
            or not np.allclose(a[3], [0, 0, 0, 1])
            or not np.allclose(a[:3, :3].T @ a[:3, :3], np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(a[:3, :3]), 1., atol=1e-5)):
        raise ValueError(name+' must be a measured/fixed rigid transform')
    return a


@dataclass(frozen=True)
class PhysicalPacket(SensorPacket):
    base_pose_planning: tuple = ()
    observed_grid: object = None
    source_file_sha256: str = ''

    def validate(self, transform, config):
        """Real sensor contract; deliberately does not assert ideal simulator K/TF."""
        transform.validate_cpu_mapper(config)
        self.frame.validate()
        if (self.frame.depth_m.shape != (config.height_px, config.width_px)
                or np.any(self.frame.depth_m > config.max_depth_m+1e-6)):
            raise ValueError('canonical measured depth shape/range differs from declared sensor configuration')
        if (self.frame.color_rgb.dtype != np.uint8
                or not np.issubdtype(self.frame.semantic.dtype, np.integer)
                or np.any(self.frame.semantic < 0)):
            raise ValueError('measured aligned RGB and nonnegative semantic labels required')
        if (np.asarray(self.observed_grid).shape != transform.shape
                or not np.isin(self.observed_grid, [-1, 0, 1]).all()):
            raise ValueError('three-state measured map must match fixed planning grid')
        if (len(self.base_pose_planning) != 3 or not np.isfinite(self.base_pose_planning).all()
                or transform.world_to_cell(self.base_pose_planning[:2]) != self.position
                or any(not 0 <= p < size for p, size in zip(self.position, transform.shape))):
            raise ValueError('actual measured base pose lies outside or differs from planning cell')
        _rigid(self.scan.world_from_laser, 'world_from_laser')
        if (self.scan.ranges_m.ndim != 1 or not len(self.scan.ranges_m)
                or not np.isfinite([self.scan.timestamp_s, self.scan.angle_min_rad,
                    self.scan.angle_increment_rad, self.scan.range_max_m]).all()
                or self.scan.angle_increment_rad <= 0 or self.scan.range_max_m <= 0):
            raise ValueError('actual scan calibration/timestamp required')
        if (type(self.action_id) is not int or self.action_id < 0
                or self.action not in (None, 'forward', 'left', 'right')
                or type(self.heading) is not int or self.heading not in range(4)
                or type(self.done) is not bool or type(self.collision) is not bool
                or any(not isinstance(x, str) or not x for x in
                       (self.scene_id, self.episode_id, self.frame_id, self.sensor_source, self.pose_source))):
            raise ValueError('physical observation identity/action contract invalid')
        return self


class _MeasuredMapMapper(ObservedRuntimeMapperV10):
    """Original measured surface backend plus supplied observed navigation grid.

    Occupied wins when actual RGBD/scan and actual mapping disagree. Free space
    is supplied only by at least one real measurement; unknown stays unknown.
    This is a declared physical representation adapter, not a policy change.
    """
    def update(self, frame, scan=None):
        super().update(frame, scan)
        self.belief[:] = np.maximum(self.belief, self.input_observed_grid)
        self.visible[:] |= self.input_observed_grid != -1
        pending = getattr(self, '_instance_pending', None)
        entry = None if pending is None else pending.pop(id(frame), None)
        if entry is not None:
            owner, frame_id = entry
            self.instance_support_index.observe(frame, owner, frame_id)


class _Components:
    """Forward original NSO_Components CPU methods without neural dependencies."""
    def __init__(self, backend, args):
        self._cpu_backend, self.args = backend, args
        self.use_ov_sem = self.use_topo = self.use_rpn_uq = self.use_igcr = True
        self.capabilities = backend.capabilities
    def update_semantic(self, scene, *args, **kwargs): return self._cpu_backend.update_semantic(scene)
    def compute_reward(self, scene, *args, **kwargs): return self._cpu_backend.compute_reward(scene)
    def update_topo(self, step, scene, *args, **kwargs): return self._cpu_backend.update_topo(scene)
    def get_sem_density(self, scene): return self._cpu_backend.scenes[scene]['density']
    def select_topo_target(self, scene, *args): return self._cpu_backend.select_target(scene)
    def get_selected_option(self, scene): return deepcopy(self._cpu_backend.scenes[scene]['selected'])
    def bind_execution_option(self, scene, option):
        self._cpu_backend.scenes[scene]['execution_option'] = deepcopy(option)
    def assess_local_action(self, scene, action): return self._cpu_backend.assess_action(scene, action)
    def plan_observed_return(self, scene): return self._cpu_backend.plan_return(scene)
    def reset_scene(self, scene): return self._cpu_backend.reset_scene(scene)


class JointSession:
    def __init__(self, configuration, execution_allowed=False, calibration=None, semantic_configuration=None, semantic_output=None, shared_observed_prefix=None):
        allowed = {'shape','planning_from_odom','return_anchor','total_budget','scene_id','episode_id',
                   'robot_radius_m','sensor_config','planner_revision','score_mode','policy_variant','sensor_range_profile','measured_extent_profile','asset_grouping_profile'}
        if not isinstance(configuration, dict) or set(configuration)-allowed:
            raise ValueError('joint configuration accepts declared measured-grid/sensor/task fields only')
        self.configuration = deepcopy(configuration)
        c = self.configuration
        self.revision = c.get('planner_revision')
        self.policy_variant = c.get('policy_variant', 'original')
        if self.policy_variant not in ('original', 'exploration_joint_v1'):
            raise ValueError('explicit original or exploration_joint_v1 policy variant required')
        if self.policy_variant != 'original' and self.revision != 'v10_3_1':
            raise ValueError('exploration_joint_v1 requires its declared v10_3_1 base')
        if self.revision not in ('v10', 'v10_1', 'v10_3_1'):
            raise ValueError('explicit historically aligned revision v10, v10_1 or v10_3_1 required; no learned-model fallback')
        self.shape = tuple(c['shape'])
        self.transform = GridTransform(self.shape, .2)
        self.planning_from_odom = _rigid(c['planning_from_odom'], 'planning_from_odom')
        if not np.allclose(self.planning_from_odom[:3, 2], [0, 0, 1]):
            raise ValueError('planning frame must have fixed planar rotation; floor-height translation is retained')
        sensor = dict(c['sensor_config'])
        if sensor.get('resolution_m', .2) != .2:
            raise ValueError('original V10 uses 0.2 m primitive grid')
        sensor.update(resolution_m=.2, robot_radius_m=c['robot_radius_m'])
        self.config = VirtualConfigV2(**sensor)
        self.transform.validate_cpu_mapper(self.config)
        self.sensor_range_profile = c.get('sensor_range_profile')
        validate_profile(self.sensor_range_profile, policy_variant=self.policy_variant,
                         camera_range_m=self.config.max_depth_m)
        self.asset_grouping_profile = c.get('asset_grouping_profile')
        if self.asset_grouping_profile is not None and (
                self.asset_grouping_profile != 'measured_instance_support_v1'
                or self.policy_variant != 'exploration_joint_v1'):
            raise ValueError('measured_instance_support_v1 requires exploration_joint_v1')
        self._instance_owner_map = None
        self.measured_extent_profile = c.get('measured_extent_profile')
        validate_extent_profile(self.measured_extent_profile, policy_variant=self.policy_variant)
        install_extent_dispatcher()
        self.last_scan_adapter_receipt = None
        if type(c['total_budget']) is not int or c['total_budget'] < 1:
            raise ValueError('explicit original action budget required')
        self.anchor = tuple(c['return_anchor'])
        self.args = SimpleNamespace(cpu_planner_revision=self.revision, cpu_score_mode=c.get('score_mode', 'S'),
            cpu_max_candidates=12, cpu_coverage_slots=4, cpu_measured_novelty_floor=.25, eval=True, train_global=False, run_id=c['episode_id'])
        backend_type, runtime_type = CPUFourModules, NSORuntimeIntegration
        if self.policy_variant == 'exploration_joint_v1':
            from nso_bridge.exploration_joint_policy import (ExplorationJointCPUFourModules,
                ExplorationJointRuntimeIntegration)
            backend_type, runtime_type = ExplorationJointCPUFourModules, ExplorationJointRuntimeIntegration
        self.backend = backend_type(self.args, 1, self.shape)
        self.components = _Components(self.backend, self.args)
        self.runtime = runtime_type(self.components, 1, self.shape)
        self.mapper = _MeasuredMapMapper(self.shape, self.config)
        if self.asset_grouping_profile is not None:
            from nso_bridge.instance_support import InstanceSupportIndex
            self.mapper.asset_grouping_profile = self.asset_grouping_profile
            self.mapper.instance_support_index = InstanceSupportIndex()
            self.mapper._instance_pending = {}
        if self.sensor_range_profile is not None:
            self.mapper.sensor_range_profile = self.sensor_range_profile
        if self.measured_extent_profile is not None:
            self.mapper.measured_extent_profile = self.measured_extent_profile
        self.execution_allowed = execution_allowed is True
        self.calibration = deepcopy(calibration or {})
        self.semantic_configuration = deepcopy(semantic_configuration)
        self.semantic_frontend = None
        self.semantic_receipt = None
        self._attempt_semantic_receipt = None
        self.semantic_output = Path(semantic_output) if semantic_output is not None else None
        if semantic_configuration is not None:
            if (not isinstance(semantic_configuration, dict) or
                    set(semantic_configuration) - {'model_dir', 'confidence', 'imgsz', 'prior_profile'}):
                raise ValueError('explicit semantic model_dir/confidence/imgsz/prior_profile configuration required')
            if self.semantic_output is None:
                raise ValueError('semantic receipt output directory required')
            from nso_bridge.real_semantic_frontend import RealSemanticFrontend
            self.semantic_frontend = RealSemanticFrontend(**semantic_configuration)
            self.backend.capabilities['semantic'] = (
                'real prompt instance segmentation on measured RGB-D; declared category prior; '
                'class accuracy, hidden structure and decision benefit unvalidated')
            self.semantic_output.mkdir(parents=True, exist_ok=True)
        from nso_bridge.shared_observed_prefix import descriptors
        self.shared_observed_prefix = descriptors(deepcopy(shared_observed_prefix))
        self._prefix_failed = False
        self.last_packet = None
        self.source_manifest_sha256 = hashlib.sha256((VENDOR/'SOURCE_MANIFEST.json').read_bytes()).hexdigest()

    def instance_adapter_receipt(self):
        if self.asset_grouping_profile is None:
            return None
        index = self.mapper.instance_support_index.summary()
        return dict(profile=self.asset_grouping_profile, candidate_set_may_change=True,
            original_method_equivalence=False, shared_instance_frontend_G_S=True,
            G_is_pure_no_semantics=False, prefix_without_masks='unknown_geometry',
            native_labels_without_masks='unknown_geometry',
            index={k: index[k] for k in ('frames', 'tracks', 'owned_voxels')},
            last_owner_map_sha256=getattr(self, '_last_fused_owner_hash', None))

    def identity(self):
        state = self.runtime.states[0]
        return json_value(dict(schema='nso_joint_runtime_v10/1', execution_allowed=self.execution_allowed,
            **({'shared_observed_prefix': dict(descriptors=self.shared_observed_prefix,
                source_scope='audited native-category measured history; automatic versus provided provenance in receipts/calibration', acquisition_cost_excluded=True, paid_prefix_actions=0,
                old_navigation_grid_discarded=True, live0_is_current_state=True)} if self.shared_observed_prefix else {}),
            calibration=self.calibration, configuration=self.configuration,
            grid=dict(shape=self.shape, resolution_m=.2, origin_xy_m=[0., 0.],
                      planning_from_odom=self.planning_from_odom),
            receipt=dict(source_manifest_path=str(VENDOR/'SOURCE_MANIFEST.json'),
                source_manifest_sha256=self.source_manifest_sha256, policy_revision=self.revision,
                policy_variant=self.policy_variant,
                sensor_ranges=range_identity(self.sensor_range_profile, self.config.max_depth_m,
                                             self.last_scan_adapter_receipt),
                research_policy_changes=[] if self.policy_variant == 'original' else [
                    'camera coverage and camera yield target actual unknown cells only',
                    'coverage candidate shortlist uses historical spatial bins',
                    'invalidated outbound macro replans on latest measured map before return'],
                **({'geometry_adapter': extent_receipt(self.measured_extent_profile)}
                   if self.measured_extent_profile is not None else {}),
                **({'instance_asset_adapter': self.instance_adapter_receipt()}
                   if self.asset_grouping_profile is not None else {}),
                original_vendor_sources_unchanged=True,
                semantic_prior_profile=(self.semantic_configuration or {}).get('prior_profile', 'strict_v10'),
                semantic_prior_extended=(self.semantic_configuration or {}).get('prior_profile', 'strict_v10') != 'strict_v10',
                original_policy_parameters=dict(max_candidates=12, coverage_slots=4,
                    measured_novelty_floor=.25 if self.revision == 'v10_3_1' else None),
                physical_adapter='actual extrinsics/K/timestamps; discrete base-state projection; observed-map union',
                semantic_scope=(('strict V10 labels 0/2/3; measured instance masks only'
                    if (self.semantic_configuration or {}).get('prior_profile', 'strict_v10') == 'strict_v10'
                    else 'explicit '+self.semantic_configuration['prior_profile']+' structural prior extension; measured instance masks only')
                    if self.semantic_frontend is not None else 'native labels only; no semantic frontend'),
                semantic_frontend=None if self.semantic_frontend is None else dict(
                    implementation='RealSemanticFrontend', configuration=self.semantic_configuration,
                    model_manifest=self.semantic_frontend.manifest),
                nominal_forecast_camera='original centered/cardinal planner camera; physical forecast calibration unverified'),
            current_state=None if self.last_packet is None else self._state(),
            pending=None if state is None else self._pending(),
            selected=None if state is None else state['option'], summary=self.summary()))

    def _state(self):
        p = self.last_packet
        return dict(action_id=p.action_id, step=p.action_id, pose=[*p.position, p.heading],
                    actual_base_pose_planning=p.base_pose_planning, frame_id=p.frame_id)

    def _pending(self):
        s = self.runtime.states[0]
        if s is None or s['pending'] is None: return None
        pose = s['pending']['next_pose']
        x, y = self.transform.cell_to_world(tuple(int(v) for v in pose[:2]))
        yaw = math.pi/2-pose[2]*math.pi/2
        odom = np.linalg.inv(self.planning_from_odom)
        point = odom @ [x, y, 0., 1.]
        target_yaw = yaw+math.atan2(odom[1, 0], odom[0, 0])
        return dict(action_id=self.last_packet.action_id+1, action=s['pending']['action'],
            pose=pose, target_odom_pose=[float(point[0]), float(point[1]),
                math.atan2(math.sin(target_yaw), math.cos(target_yaw))], assessment=s['pending'])

    def _semantic_labels(self, z, meta, source_sha256):
        depth = z['depth']
        self._instance_owner_map = None
        if 'semantic' in z:
            labels = z['semantic'].copy()
            if self.semantic_frontend is not None:
                raise ValueError('native semantic array present; refusing frontend overwrite')
        elif self.semantic_frontend is None:
            return np.zeros(depth.shape, np.int32)
        else:
            if 'rgb_raw' not in z or 'depth_m' not in z:
                raise ValueError('real semantics requires aligned original rgb_raw/depth_m')
            raw_depth = z['depth_m']
            rgb = z['rgb_raw']
            stride = self.calibration.get('pixel_stride', meta.get('pixel_stride'))
            if type(stride) is not int or stride not in (1, 2, 4, 8):
                raise ValueError('declared calibration pixel_stride required')
            valid = np.isfinite(raw_depth) & (raw_depth > 0) & (raw_depth <= self.config.max_depth_m)
            if meta.get('approved_self_mask_applied'):
                mask_receipt = meta.get('approved_self_mask_receipt') or {}
                asset = mask_receipt.get('asset_path')
                expected_hash = mask_receipt.get('file_sha256')
                # Archived captures bind their own mask, rather than whatever
                # version happens to be active during a later replay.
                if asset is None and self.calibration.get('self_mask_asset_directory'):
                    asset = str(Path(self.calibration['self_mask_asset_directory'])/'fixed-imu-mask.npy')
                    expected_hash = self.calibration.get('self_mask_file_sha256')
                if asset is not None:
                    mask_bytes = Path(asset).read_bytes()
                    if expected_hash is None or hashlib.sha256(mask_bytes).hexdigest() != expected_hash:
                        raise ValueError('observation self-mask identity mismatch')
                    import io
                    mask = np.load(io.BytesIO(mask_bytes), allow_pickle=False)
                    if mask.dtype != np.bool_:
                        raise ValueError('observation self-mask must be boolean')
                else:
                    runtime_path = Path(__file__).resolve().parents[2] / 'runtime'
                    sys.path.insert(0, str(runtime_path))
                    from visual_sensor_bridge import load_approved_mask
                    mask, _ = load_approved_mask()
                if mask.shape != valid.shape:
                    raise ValueError('approved self-mask shape mismatch')
                valid &= ~mask
            if valid[::stride, ::stride].shape != depth.shape:
                raise ValueError('raw/canonical stride alignment mismatch')
            if 'intrinsic_raw' in z:
                expected_intrinsic = np.asarray(z['intrinsic_raw'], float).copy()
                expected_intrinsic[:2] /= stride
                if not np.allclose(expected_intrinsic, z['intrinsic'], rtol=1e-9, atol=1e-9):
                    raise ValueError('canonical intrinsic must sample original intrinsic exactly once')
            expected = np.where(valid[::stride, ::stride], raw_depth[::stride, ::stride], 0.)
            if not np.allclose(expected, depth, rtol=1e-6, atol=1e-7):
                raise ValueError('canonical depth differs from actual sampled geometry')
            result = self.semantic_frontend.predict(rgb, valid, frame_id=str(meta['frame_id']))
            raw_labels = np.asarray(result['semantic'])
            if (raw_labels.shape != valid.shape or not np.issubdtype(raw_labels.dtype, np.integer)
                    or not np.isin(raw_labels, [0, 2, 3]).all() or np.any(raw_labels[~valid])):
                raise ValueError('frontend must return valid-depth V10 pixel labels 0/2/3')
            labels = np.ascontiguousarray(raw_labels[::stride, ::stride])
            receipt = deepcopy(result['receipt'])
            if receipt.get('model_execution_performed') is not True:
                raise ValueError('real semantic model execution receipt required')
            receipt.update(source_file_sha256=source_sha256, pixel_stride=stride,
                canonical_labelled_pixels=int(np.count_nonzero(labels)),
                canonical_semantic_sha256=hashlib.sha256(labels.tobytes()).hexdigest(),
                decision_effect_validated=False)
            arrays = {'semantic': labels}
            if self.asset_grouping_profile is not None:
                from nso_bridge.instance_support import canonical_owner_map
                self._instance_owner_map = canonical_owner_map(result['instances'], valid, stride)
                arrays['instance_owner'] = self._instance_owner_map
                receipt['instance_owner_sha256'] = hashlib.sha256(self._instance_owner_map.tobytes()).hexdigest()
            for i, item in enumerate(result['instances']):
                mask = np.asarray(item['mask'])
                if mask.dtype != bool or mask.shape != valid.shape:
                    raise ValueError('actual aligned boolean instance masks required')
                arrays['mask_%d' % i] = mask
                arrays['source_id_%d' % i] = np.asarray(item['source_class_id'], np.int32)
                arrays['mask_sha256_%d' % i] = np.asarray(hashlib.sha256(mask.tobytes()).hexdigest())
            # Content-addressed names avoid treating arbitrary frame IDs as paths.
            token = hashlib.sha256((str(meta['frame_id'])+source_sha256).encode()).hexdigest()[:24]
            mask_path = self.semantic_output / (token+'.npz')
            np.savez_compressed(mask_path, **arrays)
            receipt['mask_file'] = str(mask_path)
            receipt['mask_file_sha256'] = hashlib.sha256(mask_path.read_bytes()).hexdigest()
            (self.semantic_output / (token+'.json')).write_text(json.dumps(json_value(receipt), allow_nan=False))
            self._attempt_semantic_receipt = receipt
        if (labels.shape != depth.shape or not np.issubdtype(labels.dtype, np.integer)
                or not np.isin(labels, [0, 2, 3]).all() or np.any(labels[depth == 0])):
            raise ValueError('V10 semantic labels must be aligned valid-depth integers 0/2/3')
        return labels

    def semantic_diagnostics(self):
        q = self.mapper.quality_evidence(max_points=10000)
        state = self.backend.scenes[0] or {}
        scores = (state.get('last_selection') or {}).get('scores', {})
        return dict(**({'instance_asset_adapter': self.instance_adapter_receipt()}
                       if self.asset_grouping_profile is not None else {}),
            frontend_enabled=self.semantic_frontend is not None,
            receipt=self.semantic_receipt,
            quality_labels={str(k): 0 if q is None else int(np.count_nonzero(q['label'] == k)) for k in (2, 3)},
            measured_assets=[dict(group=a['group'], marked_points=a['marked_points'], class_vote=a['class_vote'])
                             for a in state.get('assets', [])],
            same_candidate_S_minus_G=[float(a-b) for a,b in zip(scores.get('S', []), scores.get('G', []))],
            decision_effect_validated=False)

    def observe(self, path, action_id, action=None, collision=False, done=False):
        self._attempt_semantic_receipt = None
        if self._prefix_failed: raise ValueError('prefix bootstrap failed; create a fresh session')
        raw = Path(path).read_bytes()
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z['joint_metadata_json'].item()))
            if (meta.get('scope') != 'real_observed_joint_input'
                    or not np.allclose(meta['planning_from_odom'], self.planning_from_odom, rtol=0, atol=1e-12)):
                raise ValueError('observation does not bind actual sources to this fixed planning frame')
            base = np.asarray(z['base_odom_pose'], float)
            point = self.planning_from_odom @ [base[0], base[1], 0., 1.]
            yaw = base[2]+math.atan2(self.planning_from_odom[1, 0], self.planning_from_odom[0, 0])
            cell = self.transform.world_to_cell(point[:2])
            heading = int(math.floor((math.pi/2-yaw)/(math.pi/2)+.5)) % 4
            frame = RGBDFrame(float(z['rgbd_timestamp_s']), z['depth'].copy(), z['color_rgb'].copy(),
                z['intrinsic'].copy(), self.planning_from_odom @ z['world_from_camera'],
                self._semantic_labels(z, meta, hashlib.sha256(raw).hexdigest()))
            scan = PlanarScan(float(z['scan_timestamp_s']), z['ranges'].copy(), float(z['angle_min']),
                float(z['angle_increment']), float(z['range_max']),
                self.planning_from_odom @ z['world_from_laser'])
            scan, self.last_scan_adapter_receipt = adapt_scan(scan, self.sensor_range_profile)
            packet = PhysicalPacket(scene_id=self.configuration['scene_id'], episode_id=self.configuration['episode_id'],
                frame_id=str(meta['frame_id']), action_id=action_id, frame=frame, scan=scan,
                position=cell, heading=heading, sensor_source='actual_RGBD_scan_and_observed_navigation_map',
                pose_source='measured_base_odom_fixed_frame_projection', action=action, collision=collision, done=done,
                base_pose_planning=(float(point[0]), float(point[1]), float(yaw)),
                observed_grid=z['joint_belief_grid'].copy(), source_file_sha256=hashlib.sha256(raw).hexdigest())
        packet.validate(self.transform, self.config)
        self.mapper.input_observed_grid = packet.observed_grid
        pending_owner = self._instance_owner_map
        if self.asset_grouping_profile is not None and pending_owner is not None:
            self.mapper._instance_pending[id(frame)] = (pending_owner, str(meta['frame_id']))
        try:
            return self._fuse_packet(packet, frame, scan, action_id, action, collision, done)
        finally:
            if self.asset_grouping_profile is not None:
                self.mapper._instance_pending.pop(id(frame), None)

    def _fuse_packet(self, packet, frame, scan, action_id, action, collision, done):
        if self.last_packet is None:
            if action_id != 0 or action is not None or collision or done:
                raise ValueError('fresh joint episode starts at actual observation0 without a forced prefix')
            # Original runtime's bootstrap is reproduced with this physical mapper;
            # its subsequent execution/feedback path remains the original code.
            if (*packet.position, packet.heading) != self.anchor:
                raise ValueError('actual initial base state differs from fixed declared return anchor')
            prefix_packets = []
            if self.shared_observed_prefix:
                self._prefix_failed = True
                try:
                    from nso_bridge.shared_observed_prefix import decode
                    prefix_packets = decode(self, self.shared_observed_prefix, packet)
                    for prefix in prefix_packets:
                        self.mapper.input_observed_grid = prefix.observed_grid
                        self.mapper.update(prefix.frame, prefix.scan)
                    self.mapper.input_observed_grid = packet.observed_grid
                    self.mapper.update(frame, scan)
                except Exception:
                    self._prefix_failed = True
                    raise
            else:
                self.mapper.update(frame, scan)
            s = dict(cpu_packet_backend=True, mapper=self.mapper, transform=self.transform,
                packet=packet, packet_hashes={p.frame_id:p.sha256() for p in prefix_packets+[packet]},
                action_frames={0:packet.frame_id}, phase='planning', active_actions=[], option=None,
                pending=None, closed=False, arrived_count=0, externally_scripted_prefix_actions=0)
            self.runtime.states[0] = s
            self.backend.start_scene(0, mapper=self.mapper, packet=packet, prefix_packets=prefix_packets+[packet],
                total_budget=self.configuration['total_budget'], paid_prefix_actions=0, return_anchor=self.anchor)
            self.runtime._update_sensor_modules(0, initial=True)
            self._prefix_failed = False
            outcome = dict(accepted=True, duplicate=False)
        else:
            outcome = self.runtime._observe_sensor_packet(0, packet)
        if outcome.get('accepted'):
            if self._attempt_semantic_receipt is not None:
                self.semantic_receipt = self._attempt_semantic_receipt
            self.last_packet = packet
            if self.asset_grouping_profile is not None and self._instance_owner_map is not None:
                self._last_fused_owner_hash = hashlib.sha256(self._instance_owner_map.tobytes()).hexdigest()
        return json_value(dict(**outcome, state=self._state(), summary=self.summary()))

    def next_action(self):
        if self.last_packet is None: raise ValueError('consume real observation0 before joint planning')
        audit_start = len(self.runtime.audit)
        action = self.runtime.next_local_action(0)
        s = self.runtime.states[0]
        return json_value(dict(
            **({'geometry_adapter': extent_receipt(self.measured_extent_profile)}
               if self.measured_extent_profile is not None else {}),
            **({'instance_asset_adapter': self.instance_adapter_receipt()}
               if self.asset_grouping_profile is not None else {}),
            action=action, pending=self._pending(), selected=s['option'],
            selection=self.backend.scenes[0].get('last_selection'), summary=self.summary(),
            runtime_events=self.runtime.audit[audit_start:]))

    def summary(self):
        s = self.runtime.states[0]
        return json_value(dict(policy_revision=self.revision, actual_paid_actions=0 if self.last_packet is None else self.last_packet.action_id,
            policy_variant=self.policy_variant,
            **({'instance_asset_adapter': self.instance_adapter_receipt()}
               if self.asset_grouping_profile is not None else {}),
            sensor_ranges=range_identity(self.sensor_range_profile, self.config.max_depth_m,
                                         self.last_scan_adapter_receipt),
            system_variant=(self.revision if self.policy_variant == 'original' else self.policy_variant)+'+'+((self.semantic_configuration or {}).get('prior_profile', 'strict_v10')
                if self.semantic_frontend is not None else 'no_real_semantics')
                + ('+'+self.sensor_range_profile if self.sensor_range_profile is not None else ''),
            total_budget=self.configuration['total_budget'], closed=False if s is None else s['closed'],
            semantics_available=False if self.last_packet is None else bool(np.any(self.last_packet.frame.semantic > 0)),
            semantic_diagnostics=self.semantic_diagnostics(),
            runtime=None if s is None else self.runtime.sensor_episode_summary(0)))


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    session = None
    with (out/'decisions.jsonl').open('a', buffering=1) as journal:
        for line in sys.stdin:
            request = None
            try:
                request = json.loads(line); method = request['method']; params = request.get('params', {})
                if method == 'init':
                    if session is not None: raise ValueError('joint worker already initialized')
                    if params.get('semantic_configuration') is not None:
                        params = dict(params, semantic_output=str(out/'semantics'))
                    session = JointSession(**params); result = session.identity()
                elif method == 'shutdown': result = dict(shutdown=True)
                elif session is None: raise ValueError('initialize the joint worker first')
                elif method == 'observe': result = session.observe(**params)
                elif method == 'next_action': result = session.next_action()
                elif method == 'summary': result = session.summary()
                else: raise ValueError('unknown joint RPC method')
                response = dict(id=request['id'], ok=True, result=result)
            except Exception as error:
                response = dict(id=None if request is None else request.get('id'), ok=False,
                                error=dict(type=type(error).__name__, message=str(error)))
            journal.write(json.dumps(dict(request=request, response=response), allow_nan=False)+'\n')
            print(json.dumps(response, allow_nan=False), flush=True)
            if request and request.get('method') == 'shutdown': break


if __name__ == '__main__': main()
