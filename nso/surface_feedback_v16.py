"""Observed 3D support transitions for a future IGCR implementation.

This observer does not fuse data, score candidates, infer object identity, or
read labels. Counts describe 15 cm measurement-support keys, not surface area.
The quality expression is the existing reliability proxy, not calibrated F1.
"""
from numbers import Integral
import math

import numpy as np


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError('invalid ' + name)
    return int(value)


def capture_support(mapper):
    """Copy eligible measured keys; never use changing point-cloud sampling."""
    frame = _integer(mapper.frames, 'map version', 1)
    belief = np.asarray(mapper.belief)
    resolution = float(mapper.config.resolution_m)
    if belief.ndim != 2 or not np.isin(belief, [-1, 0, 1]).all():
        raise ValueError('invalid observed occupancy')
    if not math.isfinite(resolution) or resolution <= 0:
        raise ValueError('invalid resolution')
    support = {}
    for raw_key, row in mapper.quality.items():
        if len(raw_key) != 3 or any(isinstance(v, (bool, np.bool_)) or
                not isinstance(v, Integral) for v in raw_key):
            raise ValueError('support keys must be three integers')
        key = tuple(int(v) for v in raw_key)
        point = np.asarray(row['point'], dtype=float)
        information, residual = float(row['information']), float(row['residual'])
        bits = _integer(row['bits'], 'direction bit mask')
        _integer(row['n'], 'measurement count', 1)
        if point.shape != (3,) or not np.isfinite(point).all() or bits > 255:
            raise ValueError('invalid measured support')
        if not np.isfinite([information, residual]).all() or min(information, residual) < 0:
            raise ValueError('invalid quality evidence')
        r, c = mapper.grid_cell(point)
        if .12 < point[2] < 1.8 and 0 <= r < belief.shape[0] and 0 <= c < belief.shape[1] and belief[r, c] != -1:
            support[key] = (information / (information + .25 + 100. * residual), bits)
    return frame, (tuple(belief.shape), resolution), support


class SurfaceFeedbackV16:
    """Causal auxiliary observations, independent of 2D camera novelty.

    Caller must fuse each authorized observation once before calling observe.
    A changed coordinate epoch requires a new observer after remapping: world
    coordinates and quality keys are not invariant to a pose-graph correction.
    Epoch consistency is caller supplied, not an external calibration proof.
    """

    def __init__(self, mapper, *, action_id, coordinate_epoch):
        self.action_id = _integer(action_id, 'bootstrap action')
        if not isinstance(coordinate_epoch, str) or not coordinate_epoch:
            raise ValueError('explicit coordinate epoch required')
        self.coordinate_epoch = coordinate_epoch
        self.frame, self.grid, self.previous = capture_support(mapper)
        self.ever_bits = {key: row[1] for key, row in self.previous.items()}
        self.initial_support_count = len(self.previous)
        self.new_support_total = 0
        self.events = 0

    def observe(self, mapper, *, action_id, coordinate_epoch):
        action_id = _integer(action_id, 'action')
        if action_id != self.action_id + 1:
            raise ValueError('one consecutive paid action required; duplicate or gap rejected')
        if coordinate_epoch != self.coordinate_epoch:
            raise ValueError('coordinate epoch changed; remapping requires explicit reset')
        frame, grid, current = capture_support(mapper)
        if frame != self.frame + 1 or grid != self.grid:
            raise ValueError('exactly one map update in the same grid required')
        old_keys, current_keys = set(self.previous), set(current)
        novel = current_keys - self.ever_bits.keys()
        reacquired = (current_keys - old_keys) - novel
        removed = old_keys - current_keys
        # Freeze the comparison set at the start of THIS action. Missing old
        # support remains in the denominator at quality zero. New supports do
        # not improve this statistic merely by joining its comparison set.
        deltas = [current.get(k, (0., 0))[0] - self.previous[k][0] for k in sorted(old_keys)]
        direction_additions = sum(bin(current[k][1] & ~self.ever_bits.get(k, 0)).count('1')
                                  for k in current_keys - novel)
        event = dict(schema='observed_surface_feedback_v16/1', action_id=action_id,
            map_version_before=self.frame, map_version_after=frame,
            coordinate_epoch=self.coordinate_epoch,
            prior_support_count=len(old_keys), current_support_count=len(current_keys),
            first_observed_support_count=len(novel), reacquired_support_count=len(reacquired),
            missing_prior_support_count=len(removed),
            new_directions_on_previously_seen_support=direction_additions,
            initial_directions_on_new_support=sum(bin(current[k][1]).count('1') for k in novel),
            signed_prior_quality_sum_change=math.fsum(deltas),
            signed_prior_quality_mean_change=math.fsum(deltas) / len(old_keys) if old_keys else 0.,
            improved_prior_support_count=sum(x > 0 for x in deltas),
            worsened_prior_support_count=sum(x < 0 for x in deltas),
            calibrated=False, ground_truth_used=False, instance_attribution_available=False)
        # Commit only after all validation and statistics succeed.
        for key, (_, bits) in current.items():
            self.ever_bits[key] = self.ever_bits.get(key, 0) | bits
        self.previous = current
        self.frame, self.action_id = frame, action_id
        self.new_support_total += len(novel)
        self.events += 1
        return event

    def snapshot(self):
        return dict(schema='observed_surface_feedback_state_v16/1',
            coordinate_epoch=self.coordinate_epoch, action_id=self.action_id,
            map_version=self.frame, observed_actions=self.events,
            initial_support_count=self.initial_support_count,
            ever_support_count=len(self.ever_bits), new_support_total=self.new_support_total)
