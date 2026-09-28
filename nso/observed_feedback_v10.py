"""Sensor-only IGCR feedback inside the preserved four-module architecture.

The mapper is read, never fused or changed here.  Its stable 15 cm quality keys
are measurement support, not reconstructed surface area.  For keys present at
bootstrap, q = I / (I + .25 + 100 residual) is an *observed reliability proxy*.
The old-support mean has a fixed denominator and its signed changes are kept;
neither this quantity nor its change is ground-truth F1 or information gain.

Disabling feedback freezes only the camera history supplied for planning
deduplication.  Actual observations, map versions, costs and failures continue
to be recorded, including the terminal observation.  Planned poses never enter
this ledger.  Runtime remains responsible for preventing duplicate map fusion.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import numpy as np


FEEDBACK_VERSION = "observed_feedback_v10.2"
SUPPORT_CELL_M = .15


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _identity(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (str, int, np.integer)):
        raise ValueError("frame IDs must be strings or integers")
    if isinstance(value, (int, np.integer)):
        return ("int", int(value))
    if not value:
        raise ValueError("frame IDs cannot be empty")
    return ("str", value)


def _frame_digest(frame):
    frame.validate()
    digest = hashlib.sha256()
    for name in ("timestamp_s", "depth_m", "color_rgb", "intrinsic",
                 "world_from_camera", "semantic"):
        value = np.ascontiguousarray(np.asarray(getattr(frame, name)))
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _map_evidence(mapper):
    """Return copies of observed support; do not sample a changing point set."""
    belief = np.asarray(mapper.belief)
    if belief.ndim != 2 or not np.isin(belief, (-1, 0, 1)).all():
        raise ValueError("feedback requires a three-state observed occupancy map")
    resolution = float(mapper.config.resolution_m)
    if not np.isfinite(resolution) or resolution <= 0:
        raise ValueError("invalid map resolution")
    explored = belief != -1
    camera = np.asarray(mapper.camera_seen, dtype=bool)
    if camera.shape != belief.shape:
        raise ValueError("camera and occupancy maps must have the same shape")
    quality, supported = {}, set()
    excluded = 0
    for raw_key, row in mapper.quality.items():
        key = tuple(int(x) for x in raw_key)
        if len(key) != 3 or any(a != b for a, b in zip(key, raw_key)):
            raise ValueError("quality keys must be stable three-dimensional integer cells")
        point = np.asarray(row["point"], dtype=float)
        information, residual = float(row["information"]), float(row["residual"])
        if point.shape != (3,) or not np.isfinite(point).all():
            raise ValueError("invalid measured quality point")
        if not np.isfinite([information, residual]).all() or information < 0 or residual < 0:
            raise ValueError("invalid observed quality statistics")
        if int(row["n"]) < 1:
            raise ValueError("quality support requires an actual measurement")
        quality[key] = information / (information + .25 + 100. * residual)
        if .12 < point[2] < 1.8:
            r, c = mapper.grid_cell(point)
            if 0 <= r < belief.shape[0] and 0 <= c < belief.shape[1] and explored[r, c]:
                supported.add(key)
            else:
                excluded += 1
    return dict(explored=explored.copy(), camera=(camera & explored).copy(),
                quality=quality, supported=supported, resolution=resolution,
                unobserved_support_excluded=excluded)


class FeedbackLedger:
    """One scene/episode ledger; create a new instance after episode reset.

    ``total_budget`` and ``paid_actions`` use the same task interval.  For a
    48-action continuation after a separately accounted 150-action prefix, use
    total_budget=48, paid_actions=0 and last_action_id=150.  For a complete
    198-action task, use total_budget=198 and paid_actions=150 instead.
    """

    def __init__(self, scene_id, episode_id, total_budget, *, feedback_enabled=True):
        if scene_id is None or episode_id is None:
            raise ValueError("scene and episode IDs are required")
        self.scene_id, self.episode_id = str(scene_id), str(episode_id)
        self.total_budget = _integer(total_budget, "total_budget", 1)
        self.feedback_enabled = bool(feedback_enabled)
        self._initialized = False
        self._events = []
        self._frames = {}
        self._actions = {}
        self._poses = []
        self._planning_poses = []
        self._version = 0
        self._planning_version = 0
        self._paid_actions = 0
        self._done = False

    def bootstrap(self, mapper, *, frames, frame_ids, map_version,
                  last_action_id=0, paid_actions=0):
        """Initialize from all actual prefix frames and its already fused map.

        Prefix observations produce no fresh reward.  Every actual camera pose
        is retained, including rotations and frames omitted by keyframe caches.
        ``frames`` and ``frame_ids`` must cover the same chronological history.
        """
        if self._initialized:
            raise RuntimeError("a ledger can be bootstrapped only once")
        frames, frame_ids = list(frames), list(frame_ids)
        if not frames or len(frames) != len(frame_ids):
            raise ValueError("bootstrap requires matching nonempty frames and IDs")
        identities = [_identity(value) for value in frame_ids]
        if len(set(identities)) != len(identities):
            raise ValueError("duplicate bootstrap frame ID")
        map_version = _integer(map_version, "map_version")
        last_action_id = _integer(last_action_id, "last_action_id")
        paid_actions = _integer(paid_actions, "paid_actions")
        if paid_actions > self.total_budget:
            raise ValueError("bootstrap exceeds task budget")
        digests = [_frame_digest(frame) for frame in frames]
        times = np.asarray([float(frame.timestamp_s) for frame in frames])
        if np.any(np.diff(times) <= 0):
            raise ValueError("actual prefix timestamps must increase strictly")
        evidence = _map_evidence(mapper)
        poses = [np.asarray(frame.world_from_camera).copy() for frame in frames]
        self._resolution = evidence["resolution"]
        self._explored_ever = evidence["explored"]
        self._camera_ever = evidence["camera"]
        self._support_ever = set(evidence["supported"])
        self._old_keys = frozenset(evidence["supported"])
        self._old_quality = self._old_mean(evidence["quality"])
        self._initial_old_quality = self._old_quality
        self._last_map_version = map_version
        self._last_action_id = last_action_id
        self._last_timestamp = float(times[-1])
        self._paid_actions = paid_actions
        self._poses = poses
        self._planning_poses = [pose.copy() for pose in poses]
        self._version = self._planning_version = 1
        self._prefix_frame_ids = [identity[1] for identity in identities]
        self._frames = {identity: dict(digest=digest, action_id=None, event_index=0)
                        for identity, digest in zip(identities, digests)}
        self._frames[identities[-1]]["action_id"] = last_action_id
        self._actions = {last_action_id: identities[-1]}
        self._initialized = True
        event = dict(module="IGCR", algorithm_version=FEEDBACK_VERSION,
                     scene_id=self.scene_id, episode_id=self.episode_id,
                     event_type="bootstrap", accepted=True, duplicate=False,
                     feedback_version=1, planning_feedback_version=1,
                     frame_id=identities[-1][1], action_id=last_action_id,
                     before_map_version=None, map_version=map_version,
                     timestamp_s=self._last_timestamp, action_cost=0,
                     paid_actions=paid_actions, remaining_budget=self.remaining_budget,
                     actual_camera_pose_count=len(poses), planning_camera_pose_count=len(poses),
                     fixed_old_support_count=len(self._old_keys),
                     old_quality_proxy=self._old_quality, reward=0., parts={},
                     arrived=False, failed=False, collision=False, done=False, reason=None)
        self._events.append(event)
        return deepcopy(event)

    @property
    def remaining_budget(self):
        return self.total_budget - self._paid_actions

    def _old_mean(self, quality):
        if not self._old_keys:
            return 0.
        # A missing old measurement remains in the denominator, at quality zero.
        return float(sum(quality.get(key, 0.) for key in sorted(self._old_keys)) / len(self._old_keys))

    def observe(self, mapper, *, frame, frame_id, action_id, map_version,
                action_cost=1, collision=False, arrived=False, done=False,
                failed=False, reason=None, scene_id=None, episode_id=None):
        """Commit one real paid action after runtime fused its terminal frame.

        Exact duplicate delivery is idempotent.  A reused frame/action ID with
        changed payload is an error.  New actions after terminal, skipped action
        IDs, backwards timestamps/map versions and budget overruns are rejected
        before ledger mutation.  Rejected execution without a real action must
        be logged by runtime and must not call this method with a free frame.
        """
        if not self._initialized:
            raise RuntimeError("bootstrap is required before feedback")
        if scene_id is not None and str(scene_id) != self.scene_id:
            raise ValueError("scene mismatch")
        if episode_id is not None and str(episode_id) != self.episode_id:
            raise ValueError("episode mismatch")
        identity = _identity(frame_id)
        action_id = _integer(action_id, "action_id")
        map_version = _integer(map_version, "map_version")
        action_cost = _integer(action_cost, "action_cost", 1)
        if action_cost != 1:
            raise ValueError("V10 uses one paid primitive action per observation")
        frame_digest = _frame_digest(frame)
        prior = self._frames.get(identity)
        if prior is not None:
            if prior["digest"] != frame_digest or prior["action_id"] != action_id:
                raise ValueError("a frame ID was reused with a different payload/action")
            old_event = self._events[prior["event_index"]]
            if (old_event["map_version"] != map_version or
                 any(old_event[k] != v for k, v in (
                     ("collision", bool(collision)), ("arrived", bool(arrived)),
                     ("done", bool(done)), ("failed", bool(failed or collision)),
                     ("reason", reason if reason is not None else ("environment_done" if done else None))))):
                raise ValueError("duplicate frame has conflicting map/event metadata")
            result = deepcopy(old_event)
            result.update(accepted=False, duplicate=True)
            return result
        if action_id in self._actions:
            raise ValueError("a paid action cannot consume a different second frame")
        if self._done:
            raise RuntimeError("terminal feedback is already committed; reset the episode")
        if action_id != self._last_action_id + 1:
            raise ValueError("paid action IDs must be consecutive")
        if map_version <= self._last_map_version:
            raise ValueError("a new sensor observation requires a newer map version")
        if float(frame.timestamp_s) <= self._last_timestamp:
            raise ValueError("actual observation timestamps must increase")
        if self.remaining_budget < action_cost:
            raise ValueError("paid action exceeds remaining budget")
        evidence = _map_evidence(mapper)
        if evidence["explored"].shape != self._explored_ever.shape or evidence["resolution"] != self._resolution:
            raise ValueError("map geometry changed within an episode")
        fresh = evidence["explored"] & ~self._explored_ever
        camera_fresh = evidence["camera"] & ~self._camera_ever
        new_support = evidence["supported"] - self._support_ever
        old_quality = self._old_mean(evidence["quality"])
        parts = dict(new_observed_2d_area_m2=float(fresh.sum()) * self._resolution ** 2,
                     new_camera_observed_2d_area_m2=float(camera_fresh.sum()) * self._resolution ** 2,
                     measured_support_proxy_m2=len(new_support) * SUPPORT_CELL_M ** 2,
                     fixed_old_quality_proxy_change=old_quality - self._old_quality,
                     action_cost=action_cost)
        # Compatibility scalar only: heterogeneous terms, not calibrated utility.
        reward = (parts["new_observed_2d_area_m2"] + parts["measured_support_proxy_m2"] +
                  parts["fixed_old_quality_proxy_change"])
        no_progress = all(parts[name] <= 1e-12 for name in (
            "new_observed_2d_area_m2", "new_camera_observed_2d_area_m2",
            "measured_support_proxy_m2", "fixed_old_quality_proxy_change"))
        pose = np.asarray(frame.world_from_camera).copy()
        event = dict(module="IGCR", algorithm_version=FEEDBACK_VERSION,
                     scene_id=self.scene_id, episode_id=self.episode_id,
                     event_type="paid_observation", accepted=True, duplicate=False,
                     feedback_version=self._version + 1,
                     planning_feedback_version=self._version + 1 if self.feedback_enabled else self._planning_version,
                     frame_id=identity[1], frame_digest=frame_digest, action_id=action_id,
                     before_map_version=self._last_map_version, map_version=map_version,
                     timestamp_s=float(frame.timestamp_s), action_cost=action_cost,
                     paid_actions=self._paid_actions + action_cost,
                     remaining_budget=self.remaining_budget - action_cost,
                     actual_camera_pose_count=len(self._poses) + 1,
                     planning_camera_pose_count=len(self._planning_poses) + int(self.feedback_enabled),
                     fixed_old_support_count=len(self._old_keys),
                     old_quality_proxy_before=self._old_quality, old_quality_proxy=old_quality,
                     new_measured_support_count=len(new_support),
                     unknown_support_excluded=evidence["unobserved_support_excluded"],
                     new_observed_cell_count=int(fresh.sum()), reward=float(reward), parts=parts,
                     no_progress=no_progress, arrived=bool(arrived), failed=bool(failed or collision),
                     collision=bool(collision), done=bool(done),
                     reason=reason if reason is not None else ("environment_done" if done else None))
        self._explored_ever |= evidence["explored"]
        self._camera_ever |= evidence["camera"]
        self._support_ever |= evidence["supported"]
        self._old_quality = old_quality
        self._poses.append(pose)
        if self.feedback_enabled:
            self._planning_poses.append(pose.copy())
            self._planning_version = self._version + 1
        self._version += 1
        self._paid_actions += action_cost
        self._last_action_id = action_id
        self._last_map_version = map_version
        self._last_timestamp = float(frame.timestamp_s)
        self._done = bool(done)
        self._frames[identity] = dict(digest=frame_digest, action_id=action_id, event_index=len(self._events))
        self._actions[action_id] = identity
        self._events.append(event)
        return deepcopy(event)

    def planning_camera_poses(self):
        """Return copies of actual history selected by the feedback ablation."""
        if not self._initialized:
            raise RuntimeError("bootstrap is required before planning")
        return [pose.copy() for pose in self._planning_poses]

    def finish(self, reason, failed=False):
        """Close at the last actual observation without minting a sensor frame.

        A planner/guard may stop between paid actions.  This is a logical
        termination event with zero cost and reward, referring to the existing
        last frame.  It never fuses a map, appends a pose, or retroactively changes
        that frame's ``done`` flag.  Repeated identical closure is idempotent;
        conflicting logical closures are rejected.  If the sensor's real
        terminal observation was already committed, that event is returned
        unchanged (apart from duplicate/accepted delivery flags).
        """
        if not self._initialized:
            raise RuntimeError("cannot finish before the initial actual observation")
        if not isinstance(reason, str) or not reason:
            raise ValueError("logical termination requires an explicit reason")
        if type(failed) is not bool:
            raise ValueError("logical failure status must be boolean")
        previous = self._events[-1]
        if self._done:
            if (previous["event_type"] == "logical_termination" and
                    (previous["reason"] != reason or previous["failed"] != failed)):
                raise ValueError("logical terminal state cannot be replaced")
            result = deepcopy(previous)
            result.update(accepted=False, duplicate=True)
            return result
        event = dict(module="IGCR", algorithm_version=FEEDBACK_VERSION,
                     scene_id=self.scene_id, episode_id=self.episode_id,
                     event_type="logical_termination", accepted=True, duplicate=False,
                     feedback_version=self._version + 1,
                     planning_feedback_version=self._planning_version,
                     reference_feedback_version=previous["feedback_version"],
                     frame_id=previous["frame_id"], action_id=self._last_action_id,
                     before_map_version=self._last_map_version, map_version=self._last_map_version,
                     timestamp_s=self._last_timestamp, action_cost=0,
                     paid_actions=self._paid_actions, remaining_budget=self.remaining_budget,
                     actual_camera_pose_count=len(self._poses),
                     planning_camera_pose_count=len(self._planning_poses),
                     fixed_old_support_count=len(self._old_keys),
                     old_quality_proxy_before=self._old_quality, old_quality_proxy=self._old_quality,
                     new_measured_support_count=0, new_observed_cell_count=0,
                     reward=0., parts=dict(new_observed_2d_area_m2=0.,
                                           new_camera_observed_2d_area_m2=0.,
                                           measured_support_proxy_m2=0.,
                                           fixed_old_quality_proxy_change=0., action_cost=0),
                     no_progress=False, arrived=False, failed=failed, collision=False,
                     done=True, reason=reason)
        self._version += 1
        self._done = True
        self._events.append(event)
        return deepcopy(event)

    def snapshot(self):
        """JSON-serializable audit state without mutable mapper/frame objects."""
        if not self._initialized:
            return dict(algorithm_version=FEEDBACK_VERSION, scene_id=self.scene_id,
                        episode_id=self.episode_id, initialized=False,
                        feedback_version=0, remaining_budget=self.total_budget)
        pose_digest = hashlib.sha256(np.asarray(self._poses, dtype=np.float64).tobytes()).hexdigest()
        planning_digest = hashlib.sha256(np.asarray(self._planning_poses, dtype=np.float64).tobytes()).hexdigest()
        old_keys_digest = hashlib.sha256(json.dumps(sorted(self._old_keys)).encode()).hexdigest()
        return dict(algorithm_version=FEEDBACK_VERSION, scene_id=self.scene_id,
                    episode_id=self.episode_id, initialized=True,
                    feedback_enabled=self.feedback_enabled, feedback_version=self._version,
                    planning_feedback_version=self._planning_version,
                    total_budget=self.total_budget, paid_actions=self._paid_actions,
                    remaining_budget=self.remaining_budget, map_version=self._last_map_version,
                    last_action_id=self._last_action_id, done=self._done,
                    consumed_frame_ids=[identity[1] for identity in self._frames],
                    consumed_action_ids=[event["action_id"] for event in self._events
                                         if event["event_type"] == "paid_observation"],
                    prefix_frame_ids=list(self._prefix_frame_ids),
                    actual_camera_pose_count=len(self._poses), planning_camera_pose_count=len(self._planning_poses),
                    actual_camera_poses_sha256=pose_digest, planning_camera_poses_sha256=planning_digest,
                    fixed_old_support_count=len(self._old_keys), fixed_old_support_sha256=old_keys_digest,
                    initial_old_quality_proxy=self._initial_old_quality, old_quality_proxy=self._old_quality,
                    old_quality_proxy_total_change=self._old_quality - self._initial_old_quality,
                    paid_measurement_support_count=sum(event.get("new_measured_support_count", 0) for event in self._events),
                    observed_2d_area_m2=float(self._explored_ever.sum()) * self._resolution ** 2,
                    no_progress_events=sum(bool(event.get("no_progress", False)) for event in self._events),
                    failure_events=sum(bool(event["failed"]) for event in self._events),
                    last_event=deepcopy(self._events[-1]),
                    capability_status=dict(interface="IGCR", backend="observed_sensor_ledger",
                                           ground_truth_input=False, learned_reward=False,
                                           calibrated_information_gain=False, ppo_training_active=False,
                                           quality_is_ground_truth_f1=False))
