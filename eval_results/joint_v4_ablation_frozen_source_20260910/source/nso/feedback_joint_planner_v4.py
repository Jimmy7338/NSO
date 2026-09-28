"""Development-only coverage and measured-return guards for the V3 planner.

This CPU mechanism is NOT the complete ANS/four-module system. Predicted route
coverage is optimistic; its rate floor is not a terminal-coverage guarantee.
The feedback ledger measures sensor evidence, not true surface area or F1.
"""
from collections import deque
import hashlib
import math

import numpy as np

from nso.quality_coverage3d import QualityCoveragePolicy
from nso.frontier_policy import Decision
from nso.semantic_completion_v3 import ObjectJointPlannerV3
from nso.camera_joint_planner_v2 import CameraJointPlannerV2
from utils.grid_geometry import DIRECTIONS


def evidence_snapshot(mapper):
    """Fixed world voxel keys; repeated observations saturate rather than pay.

    Require two supporting frames. The score is a bounded, monotone empirical
    observation-quality surrogate; it is not calibrated stereo information.
    Labels never enter this measurement and no reconstruction reference is used.
    """
    scores = {}
    for key, row in mapper.quality.items():
        # Membership uses the immutable voxel center. A noisy moving centroid
        # must not drop and reintroduce the same identity at a height threshold.
        if row['n'] < 2 or not .12 < (key[2] + .5) * .15 < 1.8:
            continue
        directions = min(int(row['bits']).bit_count() / 2., 1.)
        information = max(0., float(row['information']))
        scores[key] = .15 ** 2 * (.6 * directions + .4 * information / (information + .25))
    return scores


def evidence_increment(before, after):
    return float(sum(max(0., value - before.get(key, 0.)) for key, value in after.items()))


class FeedbackJointPlannerV4(ObjectJointPlannerV3):
    def __init__(self, *args, coverage_retention=.8, inspection_fraction=.12,
                 reserve_fraction=.2, minimum_evidence_rate=.01,
                 coverage_guard=True, feedback_guard=True, use_object_hypotheses=True, **kwargs):
        if not 0 <= coverage_retention <= 1 or not 0 <= inspection_fraction <= 1:
            raise ValueError('rate retention and inspection fraction must be in [0, 1]')
        if not 0 <= reserve_fraction <= 1 or minimum_evidence_rate < 0:
            raise ValueError('invalid reserve or evidence-rate threshold')
        self.coverage_retention = coverage_retention
        self.inspection_fraction = inspection_fraction
        self.reserve_fraction = reserve_fraction
        self.minimum_evidence_rate = minimum_evidence_rate
        self.coverage_guard = coverage_guard
        self.feedback_guard = feedback_guard
        self.use_object_hypotheses = use_object_hypotheses
        super().__init__(*args, **kwargs)
        self.anchor = QualityCoveragePolicy(self.config, self.virtual_config, 'coverage')

    def reset(self):
        super().reset()
        self.inspection_actions = 0
        self.automatic_scan_actions = 0
        self.feedback_rejections = self.coverage_rejections = 0
        self.low_return_streak = 0
        self._view_history = {}
        self._evidence_before = None
        self._evidence_ledger = {}

    @property
    def inspection_limit(self):
        return math.floor(self.config.max_steps * self.inspection_fraction)

    def _inspection_cells(self, safe, distances):
        if self.use_object_hypotheses:
            return super()._inspection_cells(safe, distances)
        self.object_model = None
        return CameraJointPlannerV2._inspection_cells(self, safe, distances)

    def _fingerprint(self, obs, cell):
        radius = int(math.ceil(self.config.sensor_range_m / self.config.resolution_m))
        r, c = cell
        patch = obs.belief[max(0, r-radius):r+radius+1, max(0, c-radius):c+radius+1]
        return hashlib.sha256(patch.tobytes()).hexdigest()

    def _route_gain(self, obs, actions):
        """Same-start comparison counts predicted sensing along all paid moves."""
        seen = np.zeros_like(obs.belief, dtype=bool)
        position, heading = obs.position, obs.heading
        for action in actions:
            if action == 'forward':
                dr, dc = DIRECTIONS[heading]
                position = (position[0] + dr, position[1] + dc)
            elif action in ('left', 'right'):
                heading = (heading + (1 if action == 'right' else -1)) % 4
            if action != 'stop':
                seen |= self._visible(obs, position, heading)
        return int(np.count_nonzero(seen & self._gain_mask(obs)))

    def _select(self, obs, safe):
        audit_start = len(self.selection_audit)
        has_joint = super()._select(obs, safe)
        self.anchor.reset()
        self.anchor.set_surface_evidence(self.mapper.evidence())
        has_anchor = self.anchor._select(obs, safe)
        if not has_joint and not has_anchor:
            return False
        anchor_gain = self._route_gain(obs, self.anchor._actions) if has_anchor else 0
        anchor_rate = anchor_gain / max(1, len(self.anchor._actions))
        joint_gain = self._route_gain(obs, self._actions) if has_joint else 0
        joint_rate = joint_gain / max(1, len(self._actions))
        inspection = has_joint and (joint_gain == 0 or joint_rate < self.coverage_retention * anchor_rate)
        reasons = []
        if has_joint and self.feedback_guard:
            key = (*self.goal, self.goal_heading)
            # A reached identical view in unchanged local occupancy has already
            # been observed. Raw voxel noise must not resurrect its novelty.
            if self._view_history.get(key) == self._fingerprint(obs, self.goal):
                reasons.append('repeated_view_unchanged_local_map')
            if inspection and self.low_return_streak >= 2:
                reasons.append('two_low_return_inspections')
        if inspection and self.coverage_guard:
            remaining = self.config.max_steps - obs.step
            if remaining - len(self._actions) < math.ceil(self.config.max_steps * self.reserve_fraction):
                reasons.append('coverage_reserve')
            if self.inspection_actions + len(self._actions) > self.inspection_limit:
                reasons.append('inspection_action_budget')
        use_anchor = not has_joint or bool(reasons)
        if reasons:
            self.feedback_rejections += int(any(r.startswith(('repeated_', 'two_')) for r in reasons))
            self.coverage_rejections += int(any(r in ('coverage_reserve', 'inspection_action_budget') for r in reasons))
        if use_anchor:
            for row in self.selection_audit[audit_start:]:
                row['proposal_only'] = True
                if row.get('selected'):
                    row['proposal_selected'] = True
                    row['selected'] = False
                    row['guard_reasons'] = reasons
            if has_joint:
                self.goal_count -= 1
            if not has_anchor:
                self._actions.clear()
                self._active = None
                self.goal = None
                return False
            for name in ('goal', 'goal_heading', '_gain', '_score', '_distance'):
                setattr(self, name, getattr(self.anchor, name))
            self._actions = deque(self.anchor._actions)
            self._active = dict(self.anchor._active)
            self.goal_count += 1
            inspection = False
            self.selection_audit.append(dict(step=obs.step, goal=list(self.goal),
                heading=self.goal_heading, selected=True, controller='coverage_anchor',
                estimated_actions=len(self._actions), guard_reasons=reasons))
        self._active.update(heading=self.goal_heading, inspection=bool(inspection),
            controller='coverage_anchor' if use_anchor else 'joint',
            predicted_route_gain_cells=anchor_gain if use_anchor else joint_gain,
            anchor_predicted_gain_per_action=anchor_rate,
            joint_predicted_gain_per_action=joint_rate,
            start_known_cells=int(np.count_nonzero(obs.belief != -1)),
            guard_reasons=reasons)
        for key, value in evidence_snapshot(self.mapper).items():
            self._evidence_ledger[key] = max(value, self._evidence_ledger.get(key, 0.))
        self._evidence_before = dict(self._evidence_ledger)
        return True

    def _finish(self, obs, status):
        if self._active is not None and self._evidence_before is not None:
            elapsed = obs.step - self._active['start_step']
            for key, value in evidence_snapshot(self.mapper).items():
                self._evidence_ledger[key] = max(value, self._evidence_ledger.get(key, 0.))
            returned = evidence_increment(self._evidence_before, self._evidence_ledger)
            rate = returned / max(1, elapsed)
            self._active.update(measured_evidence_gain=returned, measured_evidence_per_action=rate,
                measured_known_cell_increment=int(np.count_nonzero(obs.belief != -1))-self._active['start_known_cells'])
            if self._active['inspection'] and elapsed:
                self.low_return_streak = self.low_return_streak + 1 if rate < self.minimum_evidence_rate else 0
            elif elapsed and rate >= self.minimum_evidence_rate:
                # New productive coverage can expose different inspection needs.
                self.low_return_streak = 0
            if status == 'reached':
                key = (*self._active['goal'], self._active['heading'])
                self._view_history[key] = self._fingerprint(obs, self._active['goal'])
        self._evidence_before = None
        super()._finish(obs, status)

    def act(self, obs):
        if (self.coverage_guard and self._active is not None and self._active.get('inspection')
                and (self.inspection_actions >= self.inspection_limit
                     or obs.step >= self.config.max_steps-math.ceil(self.config.max_steps*self.reserve_fraction))):
            self._finish(obs, 'inspection_budget_or_reserve_exhausted')
            self._actions.clear()
            self.goal = None
        decision = super().act(obs)
        if self._active is not None and self._active.get('inspection') and decision.action != 'stop':
            self.inspection_actions += 1
        elif self._active is None and decision.action in ('left', 'right'):
            # The base controller's last-resort scans also consume the explicit
            # inspection allowance; they must not bypass a rejected view budget.
            if self.coverage_guard and (self.inspection_actions >= self.inspection_limit
                    or obs.step >= self.config.max_steps-math.ceil(self.config.max_steps*self.reserve_fraction)):
                return Decision('stop', None, None, 0, 0, 0., False)
            self.inspection_actions += 1
            self.automatic_scan_actions += 1
        return decision

    def diagnostics(self):
        return super().diagnostics() | dict(inspection_actions=self.inspection_actions,
            inspection_limit=self.inspection_limit, feedback_rejections=self.feedback_rejections,
            coverage_rejections=self.coverage_rejections, low_return_streak=self.low_return_streak,
            automatic_scan_actions=self.automatic_scan_actions)
