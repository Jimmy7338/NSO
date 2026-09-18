"""Cold-start, observation-only ANS interface for the V26 boundary prototype.

The sensor driver/evaluator owns the world; this runtime only sees packets.
No prepaid route, known facility count, or service pose argument is accepted.
"""
from copy import deepcopy

from nso.cpu_sensor_contract_v10 import GridTransform
from nso.observed_runtime_mapper_v10 import ObservedRuntimeMapperV10
from nso.observed_state_v26 import geometry_state_v26, VisibleSemanticMemoryV26
from nso.observed_planner_v26 import ObservedPlannerV26
from utils.grid_geometry import DIRECTIONS


class ObservedANSRuntimeV26:
    def __init__(self, shape, sensor_config, total_budget, mode='G', replan_interval=5):
        if type(total_budget) is not int or total_budget < 1:
            raise ValueError('positive paid action budget required')
        if type(replan_interval) is not int or replan_interval < 1:
            raise ValueError('positive global replan interval required')
        self.config, self.budget = sensor_config, total_budget
        self.transform = GridTransform(tuple(shape), sensor_config.resolution_m)
        self.transform.validate_cpu_mapper(sensor_config)
        self.mapper = ObservedRuntimeMapperV10(shape, sensor_config)
        self.planner = ObservedPlannerV26(mode)
        self.semantic = None if mode == 'G' else VisibleSemanticMemoryV26()
        self.replan_interval = replan_interval
        self.packet = self.state = self.anchor = None
        self.selected = self.pending = None
        self.cues = ()
        self.last_plan_action = None
        self.terminal_reason = None
        self.plans = []
        self.seen_frame_ids = set()

    def accept(self, packet):
        packet.validate(self.transform, self.config)
        if self.terminal_reason is not None:
            raise ValueError('episode already stopped')
        if packet.frame_id in self.seen_frame_ids:
            raise ValueError('frame identity already consumed in this episode')
        if self.packet is None:
            if packet.action_id != 0 or packet.action is not None:
                raise ValueError('V26 requires real action-zero cold start; no imposed prefix')
            self.anchor = (*packet.position, packet.heading)
        else:
            old = self.packet
            if (self.pending is None or packet.action != self.pending
                    or packet.action_id != old.action_id + 1
                    or packet.scene_id != old.scene_id or packet.episode_id != old.episode_id
                    or packet.frame_id == old.frame_id
                    or packet.frame.timestamp_s <= old.frame.timestamp_s):
                raise ValueError('new observation does not match the issued paid action')
            expected_position, expected_heading = old.position, old.heading
            if self.pending == 'forward':
                delta = DIRECTIONS[old.heading]
                expected_position = tuple(x + d for x, d in zip(old.position, delta))
            else:
                expected_heading = (old.heading + (1 if self.pending == 'right' else -1)) % 4
            if not packet.collision and (packet.position != expected_position or packet.heading != expected_heading):
                raise ValueError('odometry disagrees with issued discrete motion')
        if packet.action_id > self.budget:
            raise ValueError('paid action budget exceeded')
        before, old_cues = self.state, self.cues
        self.mapper.update(packet.frame, packet.scan)
        self.state = geometry_state_v26(self.mapper, packet, self.anchor, self.budget - packet.action_id)
        if self.semantic is not None:
            self.cues = self.semantic.update(packet)
        if before is not None:
            if packet.collision:
                self.planner._record('IGCR', 'collision_transition_no_gain_calibration',
                    dict(action_id=packet.action_id, collision=True, quality_credit=False))
            else:
                self.planner.observe_transition(before, self.state, self.selected, old_cues)
        self.packet, self.pending = packet, None
        self.seen_frame_ids.add(packet.frame_id)
        if packet.collision:
            self.terminal_reason = 'collision_observed'
        elif packet.done or packet.action_id == self.budget:
            self.terminal_reason = ('sensor_or_budget_end_returned' if (*packet.position, packet.heading) == self.anchor
                                    else 'budget_or_sensor_end_without_return')
        return self.state

    def next_action(self):
        if self.state is None:
            raise ValueError('consume action-zero observation first')
        if self.pending is not None:
            raise ValueError('consume the previous paid observation before requesting another action')
        if self.terminal_reason is not None:
            return None
        pose = (*self.state.position, self.state.heading)
        reached = self.selected is not None and tuple(self.selected['pose']) == pose
        if (self.selected is None or reached or self.last_plan_action is None
                or self.state.action_id - self.last_plan_action >= self.replan_interval):
            result = self.planner.plan(self.state, self.cues)
            self.selected = deepcopy(result['selected'])
            self.last_plan_action = self.state.action_id
            self.plans.append(result)
        action = self.planner.local_action(self.state, self.selected)
        if action is None:
            self.terminal_reason = ('returned_or_no_positive_observed_action' if pose == self.anchor
                                    else 'observed_motion_or_return_unavailable')
        self.pending = action
        return action

    def summary(self):
        return dict(mode=self.planner.mode, budget=self.budget,
            actual_paid_actions=0 if self.packet is None else self.packet.action_id,
            terminal_reason=self.terminal_reason, anchor=self.anchor,
            current_pose=None if self.packet is None else (*self.packet.position, self.packet.heading),
            global_plans=len(self.plans), module_calls=len(self.planner.calls),
            modules_called=sorted({row['module'] for row in self.planner.calls}),
            semantic_source='none' if self.semantic is None else 'observed_artificial_RGB_marker',
            trained_quality_head=False, trained_RPN_UQ=False, full_method_efficacy_proven=False,
            no_service_catalogue_input=True, no_fixed_prefix=True)
