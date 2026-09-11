"""Sensor-only scoring of fixed, complete inspection round trips.

All six scorers share route states, observed geometry, visibility, quality and
coverage. Only ObjectCompletionModel probabilities/area weights differ. Scores
are dimensionless planning surrogates per action, never evaluator F1. No world,
reference surface, evaluator, or future observation is accepted by this module.
"""
from dataclasses import asdict, dataclass, is_dataclass
import hashlib
import json

import numpy as np
from scipy.ndimage import binary_closing, label

from env.grid_exploration import GridConfig
from nso.camera_joint_planner_v2 import CameraJointPlannerV2
from nso.semantic_completion_v3 import ObjectCompletionModel
from utils.grid_geometry import DIRECTIONS, inflated_obstacles, visible_mask
from utils.reconstruction_metrics import ray_scene


SCORERS = ("G", "O", "S", "X", "M", "N")


@dataclass(frozen=True)
class CounterfactualScoreConfig:
    grid_config: GridConfig
    virtual_config: object
    hidden_weight: float = 1.0
    quality_weight: float = 1.0
    max_route_actions: int = 48
    score_objective: str = "rate"

    def __post_init__(self):
        if not np.isfinite([self.hidden_weight, self.quality_weight]).all():
            raise ValueError("score weights must be finite")
        if min(self.hidden_weight, self.quality_weight) < 0:
            raise ValueError("score weights must be nonnegative")
        if self.max_route_actions < 1:
            raise ValueError("max_route_actions must be positive")
        if self.score_objective not in ("rate", "horizon_auc"):
            raise ValueError("score_objective must be rate or horizon_auc")
        g, v = self.grid_config, self.virtual_config
        # The frozen object model explicitly projects at 0.2 m. Reject a
        # mismatched experiment instead of silently projecting with its default.
        if not np.isclose(g.resolution_m, .2) or not np.isclose(v.resolution_m, .2):
            raise ValueError("ObjectCompletionModel requires a 0.2 m grid")
        if not np.isclose(g.robot_radius_m, v.robot_radius_m):
            raise ValueError("grid and sensor robot radii differ")
        if not np.isclose(g.sensor_fov_deg, 360.):
            raise ValueError("the protocol requires a 360 degree planar scanner")
        if not np.isclose(g.sensor_range_m, v.max_depth_m):
            raise ValueError("grid range must match the declared sensor depth range")


def _array_hash(named_arrays):
    digest = hashlib.sha256()
    for name, value in named_arrays:
        array = np.ascontiguousarray(value)
        digest.update(json.dumps([name, array.dtype.str, array.shape]).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _config_dict(config):
    return asdict(config) if is_dataclass(config) else vars(config)


def _physical_history_hash(mapper):
    # SemanticHistoryMapper retains subsampled keyframes, not all raw scans.
    # The caller must separately freeze the full RGB-D/scan prefix provenance.
    arrays = []
    for i, frame in enumerate(mapper.keyframes):
        for key in ("timestamp_s", "depth_m", "color_rgb", "intrinsic", "world_from_camera"):
            arrays.append((f"{i}/{key}", getattr(frame, key)))
    return _array_hash(arrays)


def _mapper_snapshot(mapper):
    mesh = mapper.mesh()
    quality = mapper.quality_evidence(max_points=10**9)
    qa = [] if quality is None else [(key, quality[key]) for key in sorted(quality) if key != "label"]
    # Keys make any different sampling/order of the evidence dictionary visible.
    qa.append(("evidence_keys", np.asarray(list(mapper.quality), dtype=np.int64).reshape(-1, 3)))
    points, bits, labels = mapper.evidence()
    arrays = [("belief", mapper.belief), ("visible", mapper.visible),
              ("camera_seen", mapper.camera_seen), ("surface_points", points),
              ("surface_bits", bits), ("frames", mapper.frames)]
    mesh_arrays = [(key, np.asarray(getattr(mesh, key)))
                   for key in ("vertices", "triangles", "vertex_colors", "vertex_normals")]
    return {
        "mesh": mesh, "quality": quality,
        "hashes": {
            "map_and_surface_sha256": _array_hash(arrays),
            "quality_nonlabel_sha256": _array_hash(qa),
            "tsdf_mesh_sha256": _array_hash(mesh_arrays),
            "retained_physical_keyframes_sha256": _physical_history_hash(mapper),
            "sensor_config_sha256": _json_hash(_config_dict(mapper.config)),
            "surface_labels_sha256": _array_hash([("labels", labels)]),
        },
    }


def _require_equal_arrays(a, b, message):
    if not np.array_equal(a, b):
        raise ValueError(message)


def _validate_mappers(mappers, config):
    if set(mappers) != {"aligned", "shuffled", "absent"}:
        raise ValueError("mappers must contain exactly aligned, shuffled, absent")
    snapshots = {key: _mapper_snapshot(mapper) for key, mapper in mappers.items()}
    base = snapshots["aligned"]
    for mode, snapshot in snapshots.items():
        for key, value in base["hashes"].items():
            if key != "surface_labels_sha256" and snapshot["hashes"][key] != value:
                raise ValueError(f"shared-map invariant failed: {mode}/{key}")
        if _config_dict(mappers[mode].config) != _config_dict(config.virtual_config):
            raise ValueError(f"mapper and scoring sensor configurations differ: {mode}")
    for i, a in enumerate(mappers["aligned"].keyframes):
        x = mappers["shuffled"].keyframes[i]
        m = mappers["absent"].keyframes[i]
        expected = np.where(a.semantic == 2, 3, np.where(a.semantic == 3, 2, a.semantic))
        _require_equal_arrays(x.semantic, expected, "shuffled keyframe is not a 2/3 swap")
        if np.any(m.semantic):
            raise ValueError("absent keyframe contains semantic labels")
    if base["quality"] is not None:
        a = base["quality"]["label"]
        expected = np.where(a == 2, 3, np.where(a == 3, 2, a))
        _require_equal_arrays(snapshots["shuffled"]["quality"]["label"], expected,
                              "shuffled quality labels are not a 2/3 swap")
        if np.any(snapshots["absent"]["quality"]["label"]):
            raise ValueError("absent quality ledger contains semantic labels")
    return snapshots


def _validate_routes(routes, mapper, config):
    prepared = []
    identifiers = set()
    common_start = None
    safe = ~inflated_obstacles(mapper.belief != 0,
                               config.grid_config.robot_radius_m / config.grid_config.resolution_m)
    for route in routes:
        cid = route["candidate_id"]
        if not isinstance(cid, (str, int)) or isinstance(cid, bool) or cid in identifiers:
            raise ValueError("candidate_id must be a unique string or integer")
        identifiers.add(cid)
        raw = np.asarray(route["states"])
        if raw.ndim != 2 or raw.shape[1] != 3 or len(raw) < 2:
            raise ValueError("states must contain initial state and every action successor")
        if not np.issubdtype(raw.dtype, np.number) or not np.isfinite(raw).all() or not np.equal(raw, np.floor(raw)).all():
            raise ValueError("route states must contain integer coordinates/headings")
        states = [tuple(map(int, state)) for state in raw]
        if common_start is None:
            common_start = states[0]
        if states[0] != common_start or states[-1] != common_start:
            raise ValueError("all routes must return to their common initial pose and heading")
        cost = len(states) - 1
        if cost > config.max_route_actions or route.get("cost", cost) != cost:
            raise ValueError("route cost differs from action count or exceeds budget")
        for r, c, heading in states:
            if not (0 <= r < mapper.shape[0] and 0 <= c < mapper.shape[1] and heading in range(4)):
                raise ValueError("route state lies outside map/headings")
            # Same explicit current-cell allowance as the frozen controller.
            if (r, c) != common_start[:2] and not safe[r, c]:
                raise ValueError("route traverses a cell not certified safe by the shared map")
        for before, after in zip(states, states[1:]):
            r, c, h = before
            dr, dc = DIRECTIONS[h]
            allowed = {(r + dr, c + dc, h), (r, c, (h + 1) % 4), (r, c, (h - 1) % 4)}
            if after not in allowed:
                raise ValueError("route contains a teleport, free observation, or invalid turn")
        prepared.append({"candidate_id": cid, "states": states, "cost": cost})
    return prepared


def _object_geometry_hash(model):
    arrays = [("points", model.points), ("hypothesis_ids", model.hypothesis_ids)]
    for i, obj in enumerate(model.objects):
        for key in ("center", "dims", "observed_points", "fit_errors_m2"):
            arrays.append((f"{i}/{key}", obj[key]))
    return _array_hash(arrays)


def _cluster_statistics(mapper):
    """Describe measured support; an unmarked cluster is not GT background."""
    q = mapper.quality_evidence(max_points=10000)
    if q is None:
        return {"clusters": [], "accepted_clusters": 0, "marker_supported_clusters": 0,
                "unmarked_potential_background_clusters": 0, "known_background_clusters": 0}
    pts = q["point"]
    r = mapper.shape[0] - 1 - np.floor(pts[:, 1] / .2).astype(int)
    c = np.floor(pts[:, 0] / .2).astype(int)
    inside = (r >= 0) & (r < mapper.shape[0]) & (c >= 0) & (c < mapper.shape[1])
    occupancy = np.zeros(mapper.shape, bool)
    occupancy[r[inside], c[inside]] = True
    groups, _ = label(binary_closing(occupancy, structure=np.ones((3, 3))))
    ids = np.zeros(len(pts), int)
    ids[inside] = groups[r[inside], c[inside]]
    clusters = []
    object_index = 0
    for group in sorted(set(ids) - {0}):
        selected = ids == group
        low, high = np.quantile(pts[selected], [.05, .95], axis=0)
        span = high - low
        accepted = int(selected.sum()) >= 8 and max(span[:2]) <= 2.2 and span[2] >= .25
        labels = q["label"][selected]
        counts = {str(int(value)): int(np.count_nonzero(labels == value)) for value in np.unique(labels)}
        clusters.append({"group_id": int(group), "accepted": bool(accepted),
                         "object_index": object_index if accepted else None,
                         "measured_points": int(selected.sum()), "label_counts": counts,
                         "marker_supported": bool(np.any((labels == 2) | (labels == 3))),
                         "known_background_supported": bool(np.any(labels == 1))})
        object_index += int(accepted)
    accepted = [row for row in clusters if row["accepted"]]
    return {"clusters": clusters, "accepted_clusters": len(accepted),
            "marker_supported_clusters": sum(row["marker_supported"] for row in accepted),
            "unmarked_potential_background_clusters": sum(not row["marker_supported"] for row in accepted),
            "known_background_clusters": sum(row["known_background_supported"] for row in accepted)}


def _object_records(model, cluster_stats):
    records = []
    supported = [row for row in cluster_stats["clusters"] if row["accepted"]]
    if len(supported) != len(model.objects):
        raise ValueError("cluster diagnostic and frozen object model disagree")
    for i, (obj, cluster) in enumerate(zip(model.objects, supported)):
        record = {key: (value.tolist() if isinstance(value, np.ndarray) else value)
                  for key, value in obj.items()}
        record.update(object_index=i, marker_supported=cluster["marker_supported"],
                      label_counts=cluster["label_counts"],
                      box_weighted_area_m2=float(model.weights[model.hypothesis_ids == 2 * i].sum()),
                      shelf_weighted_area_m2=float(model.weights[model.hypothesis_ids == 2 * i + 1].sum()))
        record["unexplained_hypothesis_area_m2"] = record["box_weighted_area_m2"] + record["shelf_weighted_area_m2"]
        records.append(record)
    return records


def fixed_horizon_gain_auc(cumulative_gains, budget):
    """Trapezoidal mean gain over one common action budget, then hold.

    Entry zero is the already observed prefix (zero incremental gain).
    The remaining budget is intentionally idle: this is a one-route surrogate,
    not a prediction of additional receding-horizon decisions or noisy fusion.
    """
    values = np.asarray(cumulative_gains, dtype=float)
    if (values.ndim != 1 or len(values) < 2 or len(values) > budget + 1
            or not np.isfinite(values).all() or values[0] != 0 or budget < 1):
        raise ValueError("invalid cumulative gain trace or common budget")
    cost = len(values) - 1
    return float((np.sum((values[:-1] + values[1:]) / 2)
                  + (budget - cost) * values[-1]) / budget)


def score_routes(mappers, routes, config, model_factory=None):
    """Return JSON-serializable scores, decompositions, rankings and invariants.

    ``routes`` contain ``candidate_id`` and ``states`` including the shared
    initial state followed by EVERY action successor, including the return to
    the initial orientation. Optional ``cost`` must equal len(states)-1.
    Only states[1:] yield predicted observations; the initial frame is in the
    prefix. Routes are never generated, changed or selected using semantics.

    G: geometric posterior; O: binary objectness; S: aligned fine categories;
    X: swapped categories; M: actual absent-label semantic pathway; N: identical
    candidates and observed terms with zero hidden utility. S+ is not included.
    """
    if isinstance(config, dict):
        config = CounterfactualScoreConfig(**config)
    if not isinstance(config, CounterfactualScoreConfig):
        raise TypeError("config must be CounterfactualScoreConfig or its keyword dict")
    model_factory = ObjectCompletionModel if model_factory is None else model_factory
    snapshots = _validate_mappers(mappers, config)
    mapper = mappers["aligned"]
    prepared = _validate_routes(routes, mapper, config)
    models = {
        "G": model_factory(mapper, semantic=False),
        "O": model_factory(mapper, semantic=True, fine_categories=False),
        "S": model_factory(mapper, semantic=True, fine_categories=True),
        "X": model_factory(mappers["shuffled"], semantic=True, fine_categories=True),
        "M": model_factory(mappers["absent"], semantic=True, fine_categories=True),
    }
    geometry_hashes = {name: _object_geometry_hash(model) for name, model in models.items()}
    if len(set(geometry_hashes.values())) != 1:
        raise ValueError("shape points, hypothesis IDs or geometric fit differ among scorers")
    _require_equal_arrays(models["M"].weights, models["G"].weights,
                          "actual absent-label pathway does not restore geometric weights")
    for a, b in zip(models["M"].objects, models["G"].objects):
        for key in ("shelf_prior_probability", "shelf_probability", "object_probability"):
            if a[key] != b[key]:
                raise ValueError("actual absent-label posterior differs from geometry")
    if len(models["G"].weights) and not np.all(models["G"].weights > 0):
        raise ValueError("geometric weights must be positive to share visibility masks")

    planner = CameraJointPlannerV2(config.grid_config, config.virtual_config,
                                   semantic=False, route=False,
                                   quality_weight=config.quality_weight, semantic_weight=0.)
    planner.set_mapping(mapper)
    planner.quality = mapper.quality_evidence()
    mesh = snapshots["aligned"]["mesh"]
    planner.ray = ray_scene(mesh) if len(mesh.triangles) else None
    npoints = max(1, 0 if planner.quality is None else len(planner.quality["point"]))
    normalizer = max(1, int(np.count_nonzero(mapper.belief != 1)))
    c0 = np.count_nonzero(mapper.belief == 0) / normalizer
    camera_fraction = np.count_nonzero(mapper.camera_seen & (mapper.belief != 1)) / normalizer
    q0 = .5
    if planner.quality is not None:
        q = planner.quality
        counts = np.array([int(b).bit_count() for b in q["bits"]])
        reliability = q["information"] / (q["information"] + .25 + 100 * q["residual"])
        q0 = float(.55 * np.mean(np.minimum(counts / 2, 1)) + .45 * np.mean(reliability))
    area_scale = 2. * np.prod(mapper.shape) * config.grid_config.resolution_m**2
    unknown = (mapper.belief == -1) & ~inflated_obstacles(
        mapper.belief == 1, config.grid_config.robot_radius_m / config.grid_config.resolution_m)
    camera_unknown = ~mapper.camera_seen & (mapper.belief != 1)
    coverage_gate = .05 + .95 * c0**3
    r0 = camera_fraction * q0

    # Evaluate each sensor-derived pose once, then union/max over entire paths.
    cache = {}
    for state in sorted({state for route in prepared for state in route["states"][1:]}):
        row, col, heading = state
        cell = (row, col)
        quality, _, _ = planner._quality_gain(cell, heading)
        hidden_visible = models["G"].gain(cell, heading, planner.ray) > 0
        radar = visible_mask(mapper.belief == 1, cell, heading,
                             int(config.grid_config.sensor_range_m / config.grid_config.resolution_m),
                             config.grid_config.sensor_fov_deg) & unknown
        camera = visible_mask(mapper.belief == 1, cell, heading,
                              int(config.virtual_config.max_depth_m / config.grid_config.resolution_m),
                              config.virtual_config.fov_deg) & camera_unknown
        cache[state] = (quality, hidden_visible, radar, camera)

    candidates = []
    def proxy_gain(observed, hidden, radar, camera, scorer):
        dq = planner._recall_increment(observed, npoints, camera_fraction, coverage_gate, q0)
        area = 0. if scorer == "N" else float(models[scorer].weights[hidden].sum())
        recall = min(1., r0 + q0 * int(camera.sum()) / normalizer + dq
                     + config.hidden_weight * area / area_scale)
        return float(min(1., c0 + int(radar.sum()) / normalizer) * 2 * recall / (1 + recall)
                     - c0 * 2 * r0 / (1 + r0))

    for route in prepared:
        observed = np.zeros(0 if planner.quality is None else len(planner.quality["point"]))
        hidden = np.zeros(len(models["G"].points), bool)
        radar = np.zeros(mapper.shape, bool)
        camera = np.zeros(mapper.shape, bool)
        traces = {name: [0.] for name in SCORERS}
        for state in route["states"][1:]:
            q, h, r, c = cache[state]
            observed = np.maximum(observed, q)
            hidden |= h
            radar |= r
            camera |= c
            if config.score_objective == "horizon_auc":
                for scorer in SCORERS:
                    traces[scorer].append(proxy_gain(observed, hidden, radar, camera, scorer))
        dq = planner._recall_increment(observed, npoints, camera_fraction, coverage_gate, q0)
        radar_count, camera_count = int(radar.sum()), int(camera.sum())
        dc, dcamera = radar_count / normalizer, camera_count / normalizer
        scores = {}
        for scorer in SCORERS:
            area = 0. if scorer == "N" else float(models[scorer].weights[hidden].sum())
            hidden_increment = config.hidden_weight * area / area_scale
            r1 = min(1., r0 + q0 * dcamera + dq + hidden_increment)
            gain = float(min(1., c0 + dc) * 2 * r1 / (1 + r1) - c0 * 2 * r0 / (1 + r0))
            scores[scorer] = {
                "hidden_area_m2": area,
                "observed_quality_gain": float(observed.sum()),
                "observed_quality_gain_per_point": float(observed.sum()) / npoints,
                "observed_quality_recall_increment": float(dq),
                "hidden_recall_increment": float(hidden_increment),
                "camera_coverage_cells": camera_count,
                "camera_coverage_m2": camera_count * config.grid_config.resolution_m**2,
                "radar_coverage_cells": radar_count,
                "radar_coverage_m2": radar_count * config.grid_config.resolution_m**2,
                "cost": route["cost"], "joint_proxy_gain": gain,
                "score": gain / route["cost"],
                "final_score": gain / route["cost"],
            }
            if config.score_objective == "horizon_auc":
                auc = fixed_horizon_gain_auc(traces[scorer], config.max_route_actions)
                scores[scorer].update(score=auc, final_score=auc,
                                      predicted_joint_gain_auc=auc,
                                      rate_score=gain / route["cost"])
        if scores["G"] != scores["M"]:
            raise ValueError("actual missing-label scores differ from geometry")
        candidates.append({"candidate_id": route["candidate_id"], "cost": route["cost"],
                           "states_sha256": _json_hash(route["states"]), "scores": scores,
                           "union_hashes": {"radar_sha256": _array_hash([("radar", radar)]),
                                            "camera_sha256": _array_hash([("camera", camera)]),
                                            "observed_gain_sha256": _array_hash([("gain", observed)]),
                                            "hidden_visibility_sha256": _array_hash([("hidden", hidden)])}})
    rankings = {scorer: [row["candidate_id"] for _, row in sorted(
        enumerate(candidates), key=lambda pair: (-pair[1]["scores"][scorer]["final_score"],
                                                  pair[1]["cost"], pair[0]))]
        for scorer in SCORERS}
    stats = {mode: _cluster_statistics(m) for mode, m in mappers.items()}
    mode_for = {"G": "aligned", "O": "aligned", "S": "aligned", "X": "shuffled", "M": "absent"}
    objects = {name: _object_records(model, stats[mode_for[name]]) for name, model in models.items()}
    return {
        "schema_version": 1, "scorers": list(SCORERS), "candidates": candidates,
        "rankings": rankings, "selected": {name: (ids[0] if ids else None) for name, ids in rankings.items()},
        "normalization": {
            "score": ("joint_proxy_gain / executed_route_actions" if config.score_objective == "rate"
                      else "mean cumulative joint proxy gain over common action budget; hold after route"),
            "observed_quality": "pointwise maximum gain, sum / common observed point count",
            "hidden": "pointwise visibility union weighted by shape areas / declared area scale",
            "area_scale_m2": float(area_scale), "area_scale_definition": "2*grid_height*grid_width*resolution_m^2",
            "area_scale_is_ground_truth": False, "quality_point_count": npoints,
            "nonoccupied_grid_cells": normalizer, "initial_coverage_proxy": float(c0),
            "initial_camera_fraction": float(camera_fraction), "initial_conditional_quality": q0,
            "coverage_gate": float(coverage_gate), "hidden_weight": config.hidden_weight,
            "quality_weight": config.quality_weight, "old_category_approach_bonus": 0.,
            "saturation": "recall proxy clipped to 1; no claim of calibrated F1",
        },
        "objects": objects, "cluster_statistics": stats,
        "diagnostics": {
            "objectness_degenerate_with_geometry": bool(np.array_equal(models["O"].weights, models["G"].weights)),
            "missing_restores_geometry": True,
            "marked_pixels_in_retained_keyframes": sum(int(np.count_nonzero(
                (frame.semantic == 2) | (frame.semantic == 3))) for frame in mapper.keyframes),
            "geometry_hypothesis_points": len(models["G"].points),
            "unmarked_is_not_verified_background": True,
            "unmarked_potential_background_area_m2": {
                name: sum(obj["unexplained_hypothesis_area_m2"] for obj in records if not obj["marker_supported"])
                for name, records in objects.items()},
            "N_geometry_source": "G; hidden reward set to zero on the same candidate pool",
        },
        "invariants": {
            "validated": True, "mapper_hashes": {name: snap["hashes"] for name, snap in snapshots.items()},
            "shape_geometry_sha256": geometry_hashes,
            "shape_weights_sha256": {name: _array_hash([("weights", model.weights)]) for name, model in models.items()},
            "candidate_routes_sha256": _json_hash(prepared),
            "config_sha256": _json_hash(asdict(config)),
            "history_scope": "retained physical RGB-D keyframes; caller must hash full raw RGB-D/laser prefix",
            "deduplication": "states[1:] only; per-point max and mask union over entire round trip",
            "quality_branch": "one shared CameraJointPlannerV2 semantic=False semantic_weight=0 instance",
        },
    }
