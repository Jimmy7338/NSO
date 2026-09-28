"""Versioned full-prefix coverage history, using one authoritative mapper pass.

The paid-prefix validation/state-installation block follows
NSORuntimeIntegration.start_sensor_episode (runtime_integration.py, frozen V24.1
source). It is copied here because that function constructs its mapper internally
and offers no per-frame hook. No old source or global class is replaced.

External common-coverage intent must be explicitly True. All its paid turns,
return travel and zero/negative map gains count. The unchanged fixed-window V21
rate can still be zero and reject quality routes; this is not an efficacy fix.
"""
from copy import deepcopy
import numpy as np

from nso.cpu_four_modules_v16 import CPUFourModulesV16, axis_components_v16
from nso.cpu_sensor_contract_v10 import digest
from nso.coverage_budget_v21 import ObservedCoverageLedgerV21, scan_hit_mask
from nso.facility_runtime_v21 import FacilityBackendV21, FacilityRuntimeV21
from nso.observed_runtime_mapper_v10 import ObservedRuntimeMapperV10
from utils.grid_geometry import DIRECTIONS

VERSION_V24 = 'full-paid-prefix-coverage-runtime-v24-1'
PREFIX_INTENT_SOURCE = 'explicit_public_external_common_coverage_prefix_all_paid_actions'


def _map_prefix_with_coverage_v24(mapper, packets, anchor, prefix_actions):
    """Internal single-pass seam; callers must first validate every packet.

    Only post-update authoritative belief and the corresponding real scan enter
    the ledger. The scripted prefix had no model predictions, so union-yield
    calibration receives no invented predicted mask or pseudo-observation.
    """
    if mapper.frames != 0:
        raise ValueError('prefix mapping requires a fresh authoritative mapper')
    coverage = ObservedCoverageLedgerV21(mapper.shape, anchor[:2],
        mapper.config.robot_radius_m/mapper.config.resolution_m, prefix_actions=prefix_actions)
    initial = None
    observations = []
    for expected, packet in enumerate(packets):
        if packet.action_id != expected:
            raise ValueError('complete consecutive prefix from action zero required')
        mapper.update(packet.frame, packet.scan)
        if mapper.frames != expected+1:
            raise RuntimeError('authoritative mapper must update once per prefix packet')
        hits = scan_hit_mask(packet.scan, mapper.shape, mapper.config.resolution_m)
        if expected == 0:
            initial = coverage.start(mapper.belief, action_id=0, radar_hits=hits)
        else:
            event = coverage.observe(mapper.belief, action_id=expected,
                coverage_intent=True, predicted_mask=None, radar_hits=hits)
            observations.append(dict(action_id=expected, frame_id=packet.frame_id,
                action=packet.action, mapper_frames=mapper.frames, belief_sha256=digest(mapper.belief),
                coverage_event=event))
    if initial is None:
        raise ValueError('actual initial observation required')
    final_short_prefix = coverage.flush_prefix()
    receipt = dict(version=VERSION_V24, intent_source=PREFIX_INTENT_SOURCE,
        external_prefix_coverage_intent=True, paid_actions=len(packets)-1,
        initial_action_id=0, final_action_id=packets[-1].action_id,
        mapper_updates=len(packets), authoritative_mapper_single_pass=True,
        initial_coverage=initial, paid_observations=observations,
        final_short_prefix=final_short_prefix, final_coverage=coverage.snapshot(),
        prediction_feedback_invented=False, class_used=False, evaluation_truth_used=False,
        history_window_unchanged=True, positive_rate_or_budget_admission_guaranteed=False)
    return coverage, receipt


class FacilityBackendV24(FacilityBackendV21):
    def __init__(self, args, num_scenes, shape):
        super().__init__(args, num_scenes, shape)
        self.capabilities['v24'] = ('full actual paid-prefix coverage history initialization only; '
                                    'positive rate, budget admission and semantic efficacy unproven')

    def start_scene(self, scene_idx, *, prefix_coverage=None, prefix_coverage_receipt=None, **kwargs):
        if prefix_coverage is None:
            if kwargs['paid_prefix_actions']:
                raise ValueError('paid V24 prefix requires its single-pass observed coverage ledger')
            return super().start_scene(scene_idx, **kwargs)
        mapper, packet = kwargs['mapper'], kwargs['packet']
        if (not isinstance(prefix_coverage, ObservedCoverageLedgerV21)
                or kwargs['paid_prefix_actions'] < 1
                or prefix_coverage.action_id != packet.action_id
                or len(prefix_coverage.events) != kwargs['paid_prefix_actions']
                or not np.array_equal(prefix_coverage.belief, mapper.belief)
                or tuple(prefix_coverage.shape) != tuple(mapper.shape)
                or tuple(prefix_coverage.anchor) != tuple(kwargs['return_anchor'][:2])
                or prefix_coverage.pending_prefix is not None
                or prefix_coverage_receipt is None
                or prefix_coverage_receipt['intent_source'] != PREFIX_INTENT_SOURCE):
            raise ValueError('prefix coverage must match the same complete authoritative map history')
        # Same base bootstrap used by V21: actual feedback ledger and full axis
        # frames. Deliberately avoid V21.start_scene's endpoint-only new ledger.
        CPUFourModulesV16.start_scene(self, scene_idx, **kwargs)
        state = self.scenes[scene_idx]
        state['coverage_v21'] = prefix_coverage
        state['v21_pending_prediction'] = None
        state['v24_prefix_coverage_receipt'] = deepcopy(prefix_coverage_receipt)
        self._record(scene_idx, 'IGCR', 'coverage_bootstrap_v24',
            dict(paid_prefix_actions=kwargs['paid_prefix_actions'],
                 intent_source=PREFIX_INTENT_SOURCE, authority='same mapper after each actual packet'),
            prefix_coverage_receipt)


class FacilityRuntimeV24(FacilityRuntimeV21):
    def start_sensor_episode(self, scene_idx, *, config, transform, packets,
                             total_budget, return_anchor, paid_prefix_actions=0,
                             external_prefix_coverage_intent=None):
        if paid_prefix_actions == 0:
            if external_prefix_coverage_intent is not None:
                raise ValueError('an ordinary action-zero start has no external paid-prefix intent')
            return super().start_sensor_episode(scene_idx, config=config, transform=transform,
                packets=packets,total_budget=total_budget,return_anchor=return_anchor,
                paid_prefix_actions=paid_prefix_actions)
        if external_prefix_coverage_intent is not True:
            raise ValueError('paid V24 prefix requires explicit external_prefix_coverage_intent=True')
        if not isinstance(self.components._cpu_backend, FacilityBackendV24):
            raise RuntimeError('paid V24 prefix requires facility_components_v24')
        if self.states[scene_idx] is not None:
            raise RuntimeError('reset the preceding sensor episode first')
        transform.validate_cpu_mapper(config)
        if tuple(transform.shape) != self.full_shape:
            raise ValueError('runtime and sensor map shape disagree')
        packets = list(packets)
        if (not packets or type(paid_prefix_actions) is not int
                or paid_prefix_actions != len(packets)-1):
            raise ValueError('all actual prefix frames and paid prefix actions are required')
        if type(total_budget) is not int or total_budget < 1 or not 0 <= paid_prefix_actions <= total_budget:
            raise ValueError('invalid total task budget')
        first = packets[0]; identity = first.scene_id, first.episode_id
        if identity in self._sensor_episode_ids:
            raise ValueError('a fresh sensor episode ID is required after reset')
        ids=set(); last_time=-float('inf')
        for i, packet in enumerate(packets):
            packet.validate(transform,config)
            if (packet.scene_id != first.scene_id or packet.episode_id != first.episode_id
                    or packet.frame_id in ids or packet.action_id != i or packet.done or packet.collision
                    or packet.frame.timestamp_s <= last_time or (i==0)!=(packet.action is None)):
                raise ValueError('invalid, terminal or incomplete prefix history')
            if i:
                before=packets[i-1]
                dr,dc=DIRECTIONS[before.heading] if packet.action=='forward' else (0,0)
                expected=(before.position[0]+int(dr),before.position[1]+int(dc),
                    (before.heading+(1 if packet.action=='right' else -1 if packet.action=='left' else 0))%4)
                if (*packet.position,packet.heading) != expected:
                    raise ValueError('prefix odometry does not follow its recorded action')
            ids.add(packet.frame_id); last_time=packet.frame.timestamp_s
        anchor=tuple(return_anchor)
        if (len(anchor)!=3 or any(type(x) is not int for x in anchor)
                or anchor[2] not in range(4)
                or any(not 0<=v<n for v,n in zip(anchor[:2],self.full_shape))):
            raise ValueError('return anchor requires in-map integer cell and heading')
        interval=int(getattr(self.args,'cpu_v20_replan_interval',5))
        mapper=ObservedRuntimeMapperV10(self.full_shape,config)
        coverage,receipt=_map_prefix_with_coverage_v24(mapper,packets,anchor,interval)
        packet=packets[-1]
        self.states[scene_idx]=dict(cpu_packet_backend=True,mapper=mapper,transform=transform,
            packet=packet,packet_hashes={p.frame_id:p.sha256() for p in packets},
            action_frames={p.action_id:p.frame_id for p in packets},phase='planning',
            active_actions=[],option=None,pending=None,closed=False,arrived_count=0,
            externally_scripted_prefix_actions=paid_prefix_actions)
        try:
            self.components._cpu_backend.start_scene(scene_idx,mapper=mapper,packet=packet,
                prefix_packets=packets,total_budget=total_budget,paid_prefix_actions=paid_prefix_actions,
                return_anchor=anchor,prefix_coverage=coverage,prefix_coverage_receipt=receipt)
            self._update_sensor_modules(scene_idx,initial=True)
        except Exception:
            self.states[scene_idx]=None
            self.components._cpu_backend.reset_scene(scene_idx)
            raise
        self._sensor_episode_ids.add(identity)
        return self.sensor_episode_summary(scene_idx)


def facility_components_v24(args, shape):
    components=axis_components_v16(args,shape)
    components._cpu_backend=FacilityBackendV24(args,1,shape)
    components.capabilities=dict(components._cpu_backend.capabilities)
    return components
