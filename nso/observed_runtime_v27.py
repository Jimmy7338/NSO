"""V27-A runtime: separate feedback snapshots, unchanged planning/local policy.

accept() validation is copied from frozen V26 (e82d2690...c9c0), not monkeypatched.
Every accepted packet is mapped once; no World or evaluator is accessed.
Option lifecycle is provenance only and does not change target selection.
"""
from nso.observed_runtime_v26 import ObservedANSRuntimeV26
from nso.observed_state_v26 import geometry_state_v26
from nso.observed_planner_v27 import ObservedPlannerV27
from nso.observed_feedback_v27 import capture_feedback_support_v27
from utils.grid_geometry import DIRECTIONS


class ObservedANSRuntimeV27(ObservedANSRuntimeV26):
    def __init__(self, shape, sensor_config, total_budget, mode='G', replan_interval=5):
        super().__init__(shape, sensor_config, total_budget, mode, replan_interval)
        self.planner = ObservedPlannerV27(mode)
        self.feedback_support = None
        self.feedback_lifecycle = []
        self._option_serial = 0
        self._open_option = None

    def _close_option(self, reason, action_id):
        if self._open_option is not None:
            self.feedback_lifecycle.append(dict(event='closed', option_id=self._open_option,
                action_id=action_id, reason=reason, cancellation_reward_added=False))
            self._open_option = None

    def accept(self, packet):
        packet.validate(self.transform, self.config)
        if self.terminal_reason is not None:
            raise ValueError('episode already stopped')
        if packet.frame_id in self.seen_frame_ids:
            raise ValueError('frame identity already consumed in this episode')
        if self.packet is None:
            if packet.action_id != 0 or packet.action is not None:
                raise ValueError('V27 requires real action-zero cold start; no imposed prefix')
            self.anchor = (*packet.position, packet.heading)
        else:
            old = self.packet
            if (self.pending is None or packet.action != self.pending
                    or packet.action_id != old.action_id+1
                    or packet.scene_id != old.scene_id or packet.episode_id != old.episode_id
                    or packet.frame_id == old.frame_id
                    or packet.frame.timestamp_s <= old.frame.timestamp_s):
                raise ValueError('new observation does not match the issued paid action')
            expected_position, expected_heading = old.position, old.heading
            if self.pending == 'forward':
                delta = DIRECTIONS[old.heading]
                expected_position = tuple(x+d for x, d in zip(old.position, delta))
            else:
                expected_heading = (old.heading+(1 if self.pending == 'right' else -1)) % 4
            if not packet.collision and (packet.position != expected_position or packet.heading != expected_heading):
                raise ValueError('odometry disagrees with issued discrete motion')
        if packet.action_id > self.budget:
            raise ValueError('paid action budget exceeded')
        before, old_cues, old_support = self.state, self.cues, self.feedback_support
        self.mapper.update(packet.frame, packet.scan)
        self.state = geometry_state_v26(self.mapper, packet, self.anchor, self.budget-packet.action_id)
        self.feedback_support = capture_feedback_support_v27(self.mapper, packet, self.anchor,
            self.budget-packet.action_id, self.state)
        if self.semantic is not None:
            self.cues = self.semantic.update(packet)
        if before is not None:
            if packet.collision:
                self.planner._record('IGCR', 'collision_transition_no_gain_calibration',
                    dict(action_id=packet.action_id, collision=True, quality_credit=False,
                         option_id=self._open_option))
            else:
                event = self.planner.observe_transition(before, self.state, self.selected, old_cues,
                    support_before=old_support, support_after=self.feedback_support)
                if event['selected_endpoint_reached']:
                    self._close_option('reached_after_feedback', packet.action_id)
        self.packet, self.pending = packet, None
        self.seen_frame_ids.add(packet.frame_id)
        if packet.collision:
            self.terminal_reason = 'collision_observed'
        elif packet.done or packet.action_id == self.budget:
            self.terminal_reason = ('sensor_or_budget_end_returned' if (*packet.position, packet.heading) == self.anchor
                                    else 'budget_or_sensor_end_without_return')
        if self.terminal_reason is not None:
            self._close_option(self.terminal_reason, packet.action_id)
        return self.state

    def next_action(self):
        count = len(self.plans)
        action = super().next_action()
        if len(self.plans) != count:
            self._close_option('replanned_after_previous_paid_feedback', self.state.action_id)
            if self.selected is not None:
                self._option_serial += 1
                self._open_option = self._option_serial
                # selected is V26's private deepcopy, not the candidate stored in plans.
                self.selected['feedback_option_id'] = self._open_option
                self.feedback_lifecycle.append(dict(event='opened', option_id=self._open_option,
                    action_id=self.state.action_id, pose=list(self.selected['pose'])))
        if action is None and self.state is not None:
            self._close_option(self.terminal_reason or 'no_action_issued', self.state.action_id)
        return action

    def summary(self):
        result = super().summary()
        result.update(feedback_version='V27-A_support_only_last_step',
            whole_option_reward=False, planning_sample_limit=256,
            feedback_support_count=0 if self.feedback_support is None else len(self.feedback_support.patches))
        return result
