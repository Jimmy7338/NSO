"""Online sensor-yield calibration for the V10.1 IGCR planning path.

Predicted grid visibility is only a proposal.  After each paid action, the
actual newly observed cells update a Beta-Bernoulli yield model.  The planning
posterior and attempted-view mask freeze in the no-feedback ablation, while
diagnostic actual outcomes are still recorded.  No simulator truth is read.
"""
from copy import deepcopy
import hashlib
import numpy as np

from utils.grid_geometry import DIRECTIONS, inflated_obstacles, visible_mask


class ObservedGainCalibration:
    def __init__(self, shape, *, feedback_enabled=True, consume_predicted_attempts=True):
        self.shape = tuple(shape)
        self.feedback_enabled = bool(feedback_enabled)
        self.consume_predicted_attempts = bool(consume_predicted_attempts)
        self.alpha = {"radar": 1, "camera": 1}
        self.beta = {"radar": 1, "camera": 1}
        self._pending = {}
        self.events = []

    @staticmethod
    def _pose(mapper, world_from_camera):
        pose = np.asarray(world_from_camera, dtype=float)
        cell = mapper.grid_cell(pose[:3, 3])
        forward = pose[:2, 2]
        axes = np.asarray(((0., 1.), (1., 0.), (0., -1.), (-1., 0.)))
        heading = int(np.argmax(axes @ forward))
        if not np.allclose(forward, axes[heading], atol=1e-5):
            raise ValueError("planning history requires a discrete horizontal camera heading")
        return tuple(cell), heading

    def attempted_camera_mask(self, mapper, planning_camera_poses):
        """Union of expected footprints at actually attempted planning poses."""
        if not self.consume_predicted_attempts:
            # The mapper's camera_seen already consumes cells that were truly
            # observed.  An occluded cell inside a predicted footprint remains
            # eligible from a later viewpoint.
            return np.zeros(self.shape, bool)
        occupied = np.asarray(mapper.belief) == 1
        radius = int(mapper.config.max_depth_m / mapper.config.resolution_m)
        result = np.zeros(self.shape, bool)
        for pose in planning_camera_poses:
            cell, heading = self._pose(mapper, pose)
            if all(0 <= x < n for x, n in zip(cell, self.shape)):
                result |= visible_mask(occupied, cell, heading, radius, mapper.config.fov_deg)
        return result

    def posterior_mean(self, sensor):
        if sensor not in self.alpha:
            raise ValueError("unknown gain channel")
        return self.alpha[sensor] / (self.alpha[sensor] + self.beta[sensor])

    def route_masks(self, mapper, states, attempted_camera_mask):
        belief = np.asarray(mapper.belief)
        attempted = np.asarray(attempted_camera_mask, dtype=bool)
        if attempted.shape != self.shape:
            raise ValueError("attempted camera mask shape mismatch")
        occupied = belief == 1
        unknown = (belief == -1) & ~inflated_obstacles(
            occupied, mapper.config.robot_radius_m / mapper.config.resolution_m)
        camera_unknown = ~np.asarray(mapper.camera_seen, bool) & (belief != 1) & ~attempted
        radius = int(mapper.config.max_depth_m / mapper.config.resolution_m)
        radar = np.zeros(self.shape, bool); camera = np.zeros(self.shape, bool)
        for raw in states:
            cell, heading = tuple(raw[:2]), int(raw[2])
            radar |= visible_mask(occupied, cell, heading, radius, 360.) & unknown
            camera |= visible_mask(occupied, cell, heading, radius,
                                   mapper.config.fov_deg) & camera_unknown
        return radar, camera

    def prepare_action(self, mapper, *, position, heading, action, action_id,
                       planning_camera_poses):
        if action_id in self._pending:
            raise ValueError("gain prediction already exists for paid action")
        if action not in ("forward", "left", "right"):
            raise ValueError("gain calibration accepts paid primitive actions")
        position = tuple(position); next_position = position
        next_heading = int(heading)
        if action == "forward":
            dr, dc = DIRECTIONS[next_heading]
            next_position = position[0] + int(dr), position[1] + int(dc)
        else:
            next_heading = (next_heading + (1 if action == "right" else -1)) % 4
        attempted = self.attempted_camera_mask(mapper, planning_camera_poses)
        radar, camera = self.route_masks(mapper, [(*next_position, next_heading)], attempted)
        self._pending[action_id] = dict(
            action=action, next_pose=(*next_position, next_heading),
            radar=radar, camera=camera, known_before=(mapper.belief != -1).copy(),
            camera_before=np.asarray(mapper.camera_seen, bool).copy(),
            attempted_camera_sha256=hashlib.sha256(attempted.tobytes()).hexdigest())
        return dict(action_id=action_id, next_pose=list((*next_position, next_heading)),
                    predicted_radar_cells=int(radar.sum()),
                    predicted_camera_cells=int(camera.sum()),
                    attempted_camera_sha256=self._pending[action_id]["attempted_camera_sha256"],
                    posterior_before=self.snapshot()["posterior_mean"])

    def observe(self, mapper, *, action_id):
        row = self._pending.pop(action_id, None)
        if row is None:
            raise ValueError("actual gain has no preceding action prediction")
        actual_known = (mapper.belief != -1) & ~row["known_before"]
        actual_camera = np.asarray(mapper.camera_seen, bool) & ~row["camera_before"]
        result = {"action_id": int(action_id), "feedback_consumed_for_planning": self.feedback_enabled}
        for sensor, actual in (("radar", actual_known), ("camera", actual_camera)):
            predicted = row[sensor]
            success = int(np.count_nonzero(predicted & actual))
            failure = int(np.count_nonzero(predicted & ~actual))
            outside = int(np.count_nonzero(actual & ~predicted))
            if self.feedback_enabled:
                self.alpha[sensor] += success
                self.beta[sensor] += failure
            result[sensor] = dict(predicted_cells=int(predicted.sum()),
                                  realized_predicted_cells=success,
                                  unrealized_predicted_cells=failure,
                                  unpredicted_realized_cells=outside)
        result["posterior_after"] = self.snapshot()["posterior_mean"]
        self.events.append(deepcopy(result))
        return result

    def snapshot(self):
        return dict(schema_version="observed_gain_calibration_v10_1/1",
                    feedback_enabled=self.feedback_enabled,
                    consume_predicted_attempts=self.consume_predicted_attempts,
                    posterior_mean={name: self.posterior_mean(name) for name in self.alpha},
                    alpha=dict(self.alpha), beta=dict(self.beta),
                    observed_actions=len(self.events), pending_action_ids=sorted(self._pending),
                    events=deepcopy(self.events))
