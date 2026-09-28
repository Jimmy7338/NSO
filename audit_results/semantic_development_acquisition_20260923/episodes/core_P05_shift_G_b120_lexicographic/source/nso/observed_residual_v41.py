"""Current paid-pixel IGCR evidence for a controlled, visible-label fixture.

No World, scene instance metadata, GT geometry, ROI, evaluator or future-pose
depth bank is read. This is a profiled robust residual, NOT a calibrated sensor
likelihood or a general object-pose estimator. Upright objects with the common
development label mounting convention are a deliberately narrow assumption.
"""
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np

from nso.development_geometry_v40 import STRUCTURES, structure_boxes
from nso.instance_belief_v40 import PaidRGBDObservationV40


DEPTH_CONVENTION = "optical_axial_z_metres; near=0.1; far=4.0; zero=missing"
NOMINAL_DIMENSIONS_M = (1.2, .8, 1.6)
SCALE_CANDIDATES = (.85, 1., 1.15)
YAW_MULTIPLIERS = (-1., 0., 1.)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def _array_hash(array):
    a = np.ascontiguousarray(array)
    h = hashlib.sha256(f"{a.dtype.str}:{a.shape}:".encode())
    h.update(a.tobytes())
    return h.hexdigest()


def _indices(value, shape, name):
    a = np.asarray(value)
    if a.size == 0:
        return np.zeros(0, dtype=np.int64)
    if a.ndim != 1 or a.dtype.kind not in "iu" or a.dtype.kind == "b":
        raise ValueError(name + " must be sorted unique flat integer pixels")
    a = a.astype(np.int64)
    if np.any(a < 0) or np.any(a >= np.prod(shape)) or np.any(np.diff(a) <= 0):
        raise ValueError(name + " must be sorted unique in-bounds pixels")
    return a


def pixel_mask_sha256_v41(indices, shape):
    indices = _indices(indices, shape, "pixels")
    mask = np.zeros(shape, dtype=np.uint8)
    mask.flat[indices] = 1
    return _array_hash(mask)


def observed_points_v41(observation, indices):
    """Measured points only; optical depth multiplies a ray with camera z=1."""
    y, x = np.divmod(indices, observation.depth_m.shape[1])
    rays = np.stack((x, y, np.ones(len(x))), axis=1) @ np.linalg.inv(observation.intrinsic).T
    rays /= rays[:, 2, None]
    camera_points = rays * observation.depth_m.ravel()[indices, None]
    return camera_points @ observation.world_from_camera[:3, :3].T + observation.world_from_camera[:3, 3]


@dataclass(frozen=True)
class PrototypeBankV41:
    """Fixed shared nominal library; no constructor input from a scene asset."""
    candidates: tuple
    receipt_json: str
    sha256: str

    def receipt(self):
        return json.loads(self.receipt_json)


def build_prototype_bank_v41():
    candidates = []
    hashes = []
    for structure in STRUCTURES:
        for scale in SCALE_CANDIDATES:
            dims = np.asarray(NOMINAL_DIMENSIONS_M) * scale
            boxes, marker = structure_boxes(structure, dims)
            boxes = np.asarray(boxes, dtype=np.float64)
            anchor = np.asarray(marker["center_local_m"], dtype=np.float64)
            boxes.flags.writeable = False
            anchor.flags.writeable = False
            candidates.append((structure, float(scale), boxes, anchor))
            hashes.append(dict(structure=structure, scale=scale, boxes_sha256=_array_hash(boxes),
                               label_anchor_sha256=_array_hash(anchor)))
    source = Path(inspect.getfile(structure_boxes))
    receipt = dict(schema="shared_nominal_structure_library_v41", structure_names=list(STRUCTURES),
        nominal_dimensions_m=list(NOMINAL_DIMENSIONS_M), scale_candidates=list(SCALE_CANDIDATES),
        yaw_multipliers=list(YAW_MULTIPLIERS), yaw_radius_rule="max(5deg, observed fit uncertainty); at most 12deg",
        source_path="nso/development_geometry_v40.py",
        source_file_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        source_function_sha256=hashlib.sha256(inspect.getsource(structure_boxes).encode()).hexdigest(),
        candidates=hashes, actual_scene_parameters_used=False,
        label_mounting_assumption="common upright development label plate; known relative mount, not object centre",
        trained_or_calibrated=False)
    encoded = json.dumps(receipt, sort_keys=True, separators=(",", ":"))
    return PrototypeBankV41(tuple(candidates), encoded, _digest(receipt))


def fit_marker_plane_v41(observation, marker_pixel_indices):
    """Fit from measured marker pixels; return an auditable refusal if ambiguous.

    A near-complete rectangular label is required. The anchor is its estimated
    visible-plane centre, not an oracle object centre. Completeness is only a
    conservative image-resolution check, not proof against every occlusion.
    """
    if type(observation) is not PaidRGBDObservationV40:
        raise TypeError("strict PaidRGBDObservationV40 required")
    pixels = _indices(marker_pixel_indices, observation.depth_m.shape, "marker pixels")
    result = dict(accepted=False, frame_id=observation.frame_id, paid_step=observation.paid_step,
                  observation_sha256=observation.sha256(),
                  marker_mask_sha256=pixel_mask_sha256_v41(pixels, observation.depth_m.shape),
                  marker_pixels=int(len(pixels)), depth_convention=DEPTH_CONVENTION)
    def reject(reason):
        result["reason"] = reason
        return result
    depth = observation.depth_m.ravel()[pixels]
    valid = (depth >= .1) & (depth <= 4.)
    result["valid_depth_pixels"] = int(valid.sum())
    if len(pixels) < 16 or valid.sum() < 16:
        return reject("insufficient_marker_depth")
    if valid.mean() < .9:
        return reject("marker_depth_incomplete")
    points = observed_points_v41(observation, pixels[valid])
    center = np.mean(points, axis=0)
    _, singular, vt = np.linalg.svd(points-center, full_matrices=False)
    if singular[1] < .08:
        return reject("marker_support_collinear_or_too_small")
    normal = vt[-1]
    toward_camera = observation.world_from_camera[:3, 3]-center
    if np.dot(normal, toward_camera) < 0:
        normal = -normal
    plane_errors = (points-center) @ normal
    rms = float(np.sqrt(np.mean(plane_errors**2)))
    result["plane_rms_m"] = rms
    if rms > .015 or singular[2]/singular[1] > .15:
        return reject("marker_not_reliably_planar")
    if abs(normal[2]) > .15:
        return reject("marker_not_upright")
    incidence = float(np.dot(normal, toward_camera)/np.linalg.norm(toward_camera))
    if incidence < .35:
        return reject("marker_view_too_oblique")
    # The nominal family is gravity aligned. Tilt is recorded, not silently
    # promoted to a known six-DoF object pose.
    normal = normal.copy(); normal[2] = 0.; normal /= np.linalg.norm(normal)
    up = np.array([0., 0., 1.])
    horizontal = np.cross(up, normal)
    uv = np.column_stack(((points-center)@horizontal, (points-center)@up))
    extent = np.ptp(uv, axis=0)
    axial = float(np.median(depth[valid]))
    pixel_width = axial / min(observation.intrinsic[0, 0], observation.intrinsic[1, 1]) / incidence
    tolerance = min(.065, 2.*pixel_width + .008)
    result.update(observed_marker_extent_m=extent.tolist(), pixel_footprint_bound_m=float(pixel_width),
                  marker_completeness_tolerance_m=float(tolerance))
    if np.any(np.abs(extent-np.array([.32, .30])) > tolerance):
        return reject("marker_extent_incomplete_or_wrong_mount")
    anchor = center + horizontal * (uv[:, 0].max()+uv[:, 0].min())*.5 + up*(uv[:, 1].max()+uv[:, 1].min())*.5
    anchor_uncertainty = max(.005, pixel_width + 2.*rms)
    yaw_uncertainty = max(1., float(np.degrees(np.arctan2(2.*rms+pixel_width, max(extent[0], .001)))))
    if anchor_uncertainty > .06 or yaw_uncertainty > 12.:
        return reject("marker_pose_uncertainty_too_large")
    result.update(accepted=True, reason="observed_upright_label_plane", anchor_world_m=anchor.tolist(),
                  outward_normal_world=normal.tolist(), horizontal_world=horizontal.tolist(),
                  up_world=up.tolist(), anchor_uncertainty_m=float(anchor_uncertainty),
                  yaw_uncertainty_deg=yaw_uncertainty, estimated_from_current_pixels=True,
                  object_centre_known=False, completeness_guaranteed=False)
    return result


def _ray_box_depths(origins, directions, boxes):
    """First hit on union of local solid boxes; ray parameter is camera z."""
    intervals = []
    for flat_box in boxes:
        box = flat_box.reshape(3, 2)
        lower = np.full(len(directions), -np.inf)
        upper = np.full(len(directions), np.inf)
        possible = np.ones(len(directions), dtype=bool)
        for axis in range(3):
            moving = np.abs(directions[:, axis]) > 1e-12
            possible &= moving | ((origins[axis] >= box[axis, 0]) & (origins[axis] <= box[axis, 1]))
            enter = np.full(len(directions), -np.inf)
            leave = np.full(len(directions), np.inf)
            a = (box[axis, 0]-origins[axis])/directions[moving, axis]
            b = (box[axis, 1]-origins[axis])/directions[moving, axis]
            enter[moving] = np.minimum(a, b); leave[moving] = np.maximum(a, b)
            lower = np.maximum(lower, enter); upper = np.minimum(upper, leave)
        valid = possible & (upper >= lower)
        intervals.append(np.column_stack((np.where(valid, lower, np.inf),
                                           np.where(valid, upper, -np.inf))))
    intervals = np.stack(intervals, axis=1)
    order = np.argsort(intervals[:, :, 0], axis=1, kind="stable")
    ordered = np.take_along_axis(intervals, order[:, :, None], axis=1)
    closest = np.full(len(directions), np.inf)
    component_enter = np.full(len(directions), np.inf)
    component_exit = np.full(len(directions), -np.inf)

    def finish(selected):
        # Near clipping inside a connected solid union exposes its exit. The
        # entry of an overlapping second box is an internal face, never a hit.
        hit = np.where(component_enter >= .1, component_enter, component_exit)
        valid = selected & (component_exit >= component_enter) & (hit >= .1) & (hit <= 4.)
        return np.where(valid, hit, np.inf)

    for box_index in range(len(boxes)):
        enter, leave = ordered[:, box_index, 0], ordered[:, box_index, 1]
        connected = enter <= component_exit + 1e-12
        closest = np.minimum(closest, finish(~connected))
        component_enter = np.where(connected, component_enter, enter)
        component_exit = np.where(connected, np.maximum(component_exit, leave), leave)
    closest = np.minimum(closest, finish(np.ones(len(directions), dtype=bool)))
    return closest


class ObservedResidualV41:
    """Stateful current-frame caller. Geometry scores are identical for G/S.

    Call immediately after the geometry-only association frontend. It retains
    only observed label-plane fits, never actual scene positions or meshes.
    """
    def __init__(self, *, maximum_sample_pixels=2048, truncation_m=.25, evidence_scale_m=.05):
        if type(maximum_sample_pixels) is not int or not 16 <= maximum_sample_pixels <= 6912:
            raise ValueError("maximum_sample_pixels must be an integer in [16,6912]")
        if not np.isfinite(truncation_m) or not 0 < truncation_m <= 1.:
            raise ValueError("positive bounded truncation required")
        if not np.isfinite(evidence_scale_m) or evidence_scale_m <= 0:
            raise ValueError("positive evidence scale required")
        self.bank = build_prototype_bank_v41()
        self.maximum_sample_pixels = maximum_sample_pixels
        self.truncation_m = float(truncation_m)
        self.evidence_scale_m = float(evidence_scale_m)
        self.source_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        self.configuration_sha256 = _digest(dict(maximum_sample_pixels=maximum_sample_pixels,
            truncation_m=self.truncation_m, evidence_scale_m=self.evidence_scale_m,
            prototype_bank_sha256=self.bank.sha256, source_sha256=self.source_sha256))
        self._planes = {}
        self._last_step = -1
        self._frames = set()

    def _score(self, observation, pixels, plane):
        y, x = np.divmod(pixels, observation.depth_m.shape[1])
        rays = np.stack((x, y, np.ones(len(x))), axis=1) @ np.linalg.inv(observation.intrinsic).T
        rays /= rays[:, 2, None]
        directions = rays @ observation.world_from_camera[:3, :3].T
        measured = observation.depth_m.ravel()[pixels]
        outward = np.asarray(plane["outward_normal_world"])
        up = np.array([0., 0., 1.])
        horizontal = np.cross(up, outward)
        base_rotation = np.column_stack((horizontal, -outward, up))
        anchor = np.asarray(plane["anchor_world_m"])
        uncertainty = plane["anchor_uncertainty_m"]
        anchor_offsets = ((0., 0.), (-uncertainty, 0.), (uncertainty, 0.), (0., -uncertainty), (0., uncertainty))
        best = {name: None for name in STRUCTURES}
        for name, scale, boxes, local_marker_anchor in self.bank.candidates:
            for multiplier in YAW_MULTIPLIERS:
                yaw = multiplier * max(5., plane["yaw_uncertainty_deg"])
                theta = np.radians(yaw)
                yaw_rotation = np.array([[np.cos(theta), -np.sin(theta), 0.],
                                         [np.sin(theta), np.cos(theta), 0.], [0., 0., 1.]])
                rotation = yaw_rotation @ base_rotation
                local_directions = directions @ rotation
                for du, dv in anchor_offsets:
                    observed_anchor = anchor+du*horizontal+dv*up
                    translation = observed_anchor - rotation @ local_marker_anchor
                    local_origin = (observation.world_from_camera[:3, 3]-translation) @ rotation
                    predicted = _ray_box_depths(local_origin, local_directions, boxes)
                    hits = np.isfinite(predicted)
                    errors = np.full(len(pixels), self.truncation_m)
                    errors[hits] = np.minimum(np.abs(predicted[hits]-measured[hits]), self.truncation_m)
                    loss = float(np.mean(errors))
                    if best[name] is None or loss < best[name]["loss_m"]:
                        best[name] = dict(loss_m=loss, valid_sample_count=int(len(pixels)),
                            predicted_hit_count=int(hits.sum()), predicted_miss_count=int((~hits).sum()),
                            best_scale=float(scale), best_yaw_offset_deg=float(yaw),
                            best_anchor_offset_uv_m=[float(du), float(dv)], candidate_count=45)
        losses = np.array([best[name]["loss_m"] for name in STRUCTURES])
        scores = -(losses-losses.min())/self.evidence_scale_m
        return best, losses.tolist(), scores.tolist()

    def observe(self, observation, accepted_associations):
        if type(observation) is not PaidRGBDObservationV40:
            raise TypeError("strict PaidRGBDObservationV40 required")
        if observation.paid_step <= self._last_step or observation.frame_id in self._frames:
            raise ValueError("residual caller requires new monotonically ordered paid packets")
        if not isinstance(accepted_associations, (list, tuple)):
            raise TypeError("current accepted association list required")
        packet_hash = observation.sha256()
        validated = []
        seen = set()
        for association in accepted_associations:
            if not isinstance(association, dict):
                raise TypeError("current association dictionary required")
            forbidden = {"actual_scene_pose", "actual_dimensions", "world_aabb_m", "ground_truth_owner",
                         "triangle_instance_id", "local_solid_boxes", "true_structure", "template_depths"}
            if forbidden.intersection(association):
                raise ValueError("private scene fields are forbidden at residual boundary")
            instance = association["instance_id"]
            if not isinstance(instance, str) or not instance or instance in seen:
                raise ValueError("one unique opaque instance ID per packet required")
            if association["observation_sha256"] != packet_hash:
                raise ValueError("association paid packet SHA mismatch")
            if association.get("frame_id") != observation.frame_id or association.get("paid_step") != observation.paid_step:
                raise ValueError("association current frame ID/paid step mismatch")
            pixels = _indices(association["pixel_indices"], observation.depth_m.shape, "support pixels")
            marker_pixels = _indices(association.get("marker_pixel_indices", []), observation.depth_m.shape, "marker pixels")
            if not np.isin(marker_pixels, pixels).all():
                raise ValueError("marker mask must belong to currently associated support")
            points = np.asarray(association["points_world_m"], dtype=np.float64)
            measured = observed_points_v41(observation, pixels)
            if points.shape != measured.shape or not np.allclose(points, measured, atol=1e-7, rtol=0):
                raise ValueError("association points differ from current measured depth")
            if association["support_sha256"] != _array_hash(points):
                raise ValueError("association support SHA mismatch")
            seen.add(instance)
            validated.append((association, instance, pixels, marker_pixels))
        self._last_step = observation.paid_step
        self._frames.add(observation.frame_id)
        results = []
        for association, instance, pixels, marker_pixels in validated:
            result = dict(accepted=False, instance_id=instance, frame_id=observation.frame_id,
                paid_step=observation.paid_step, observation_sha256=packet_hash,
                support_sha256=association.get("support_sha256"),
                mask_sha256=pixel_mask_sha256_v41(pixels, observation.depth_m.shape),
                prototype_bank_sha256=self.bank.sha256,
                prototype_source_sha256=self.bank.receipt()["source_file_sha256"],
                residual_source_sha256=self.source_sha256, configuration_sha256=self.configuration_sha256,
                structure_names=list(STRUCTURES), depth_convention=DEPTH_CONVENTION,
                ground_truth_used=False, future_observation_used=False, calibrated_likelihood=False)
            results.append(result)
            if len(marker_pixels):
                fitted = fit_marker_plane_v41(observation, marker_pixels)
                result["current_plane_fit"] = fitted
                if fitted["accepted"] and instance not in self._planes:
                    self._planes[instance] = deepcopy(fitted)
                elif fitted["accepted"]:
                    previous = self._planes[instance]
                    displacement = np.linalg.norm(np.asarray(fitted["anchor_world_m"])-previous["anchor_world_m"])
                    agreement = np.dot(fitted["outward_normal_world"], previous["outward_normal_world"])
                    if displacement > .15 or agreement < np.cos(np.radians(20.)):
                        result["reason"] = "inconsistent_observed_label_plane"
                        continue
            if instance not in self._planes:
                result["reason"] = "no_reliable_observed_label_plane"
                continue
            plane = self._planes[instance]
            result["plane_fit"] = deepcopy(plane)
            valid = (observation.depth_m.ravel()[pixels] >= .1) & (observation.depth_m.ravel()[pixels] <= 4.)
            selected = pixels[valid]
            result["associated_pixel_count"] = int(len(pixels))
            result["missing_or_out_of_range_ignored_count"] = int((~valid).sum())
            if len(selected) < 16:
                result["reason"] = "insufficient_current_valid_depth"
                continue
            if len(selected) > self.maximum_sample_pixels:
                selected = selected[np.linspace(0, len(selected)-1, self.maximum_sample_pixels, dtype=int)]
            per_structure, losses, scores = self._score(observation, selected, plane)
            result.update(accepted=True, reason="current_observed_depth_residual", per_structure=per_structure,
                losses_m=losses, valid_sample_counts=[len(selected)]*len(STRUCTURES),
                log_likelihoods=scores, informative=bool(np.ptp(scores) > 1e-6),
                evaluated_pixel_mask_sha256=pixel_mask_sha256_v41(selected, observation.depth_m.shape),
                missing_depth_is_empty_space_evidence=False, loss="mean_min_absolute_axial_residual",
                truncation_m=self.truncation_m, evidence_scale_m=self.evidence_scale_m,
                association_geometry_feedback_eligible=bool(association.get("geometry_feedback_eligible", False)),
                nuisance_parameters_profiled=True, candidate_count_per_structure=45,
                sample_rule="all valid associated pixels or deterministic row-major uniform subsample")
        return dict(schema="current_paid_observed_residual_v41", frame_id=observation.frame_id,
                    observation_sha256=packet_hash, results=results, prototype_bank=self.bank.receipt(),
                    ans_module="IGCR", world_or_reconstruction_executed=False)
