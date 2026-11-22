"""Evidence-gated positive category increment on the original particle bank."""

import numpy as np
from scipy.spatial import cKDTree


class SupportedHidden:
    def __init__(self, mapper, models, ray, gains, classify_history, settings):
        from env.virtual3d import camera_pose

        self.mapper = mapper
        self.camera_pose = camera_pose
        self.classify_history = classify_history
        self.model = models["S"]
        self.ray = ray
        self.gains = gains
        self.fractions = settings["clearance_ray_fractions"]
        self.cached = {}
        assert self.model.bank is models["G"].bank
        np.testing.assert_array_equal(self.model.points, models["G"].points)
        np.testing.assert_array_equal(self.model.history_keep, models["G"].history_keep)

        rows = list(mapper.quality.items())
        keys = np.asarray([key for key, _ in rows], dtype=np.int64).reshape(-1, 3)
        tracks = mapper.instance_support_index.memberships_for_keys(keys) if rows else ()
        points = np.asarray([row["point"] for _, row in rows], dtype=float).reshape(-1, 3)
        labels = np.asarray([row["label"] for _, row in rows], dtype=np.int8)
        tracks = np.asarray(tracks, dtype=np.int64)
        allowed = np.zeros(len(self.model.points), dtype=bool)
        uplift = np.maximum(0.0, self.model.weights - models["G"].weights)
        self.track_audit = []
        for index, item in enumerate(self.model.objects):
            track = int(item["support_track"])
            track_mask = (tracks == track) & np.isin(labels, (2, 3))
            n = int(track_mask.sum())
            shelf = int((track_mask & (labels == 2)).sum())
            majority = max(shelf, n - shelf) / n if n else 0.0
            ids = np.flatnonzero((self.model.object_ids == index) & (uplift > 1e-15))
            if n >= settings["same_instance_measured_voxels_min"] and majority >= settings["majority_vote_threshold"]:
                distance, _ = cKDTree(points[track_mask]).query(self.model.points[ids], workers=1)
                allowed[ids] = (distance <= settings["same_instance_radius_m"]) & (
                    distance > settings["minimum_unmeasured_separation_m"])
            assert not (allowed[ids] & ~self.model.history_keep[ids]).any()
            self.track_audit.append({
                "track":track, "actual_depth_voxels":n, "majority_vote_fraction":majority,
                "observed_majority_class":(2 if shelf > n - shelf else 3) if n and shelf != n-shelf else None,
                "structural_shape_prior":item["shape_prior"], "positive_hidden_quadrature":int(allowed[ids].sum()),
                "expected_supported_uplift_m2":float(uplift[ids][allowed[ids]].sum()),
            })
        self.active = np.flatnonzero(allowed)
        self.uplift = uplift[self.active]

    def _pose(self, pose):
        key = tuple(map(int, pose))
        if key in self.cached:
            return self.cached[key]
        accepted = np.zeros(len(self.active), dtype=bool)
        if len(self.active):
            if key not in self.gains:
                self.gains[key] = self.model.gain(key[:2], key[2], self.ray)
            visible = np.flatnonzero(np.asarray(self.gains[key])[self.active] > 0)
            if len(visible):
                center = self.camera_pose(
                    key[:2], key[2], self.mapper.config, self.mapper.shape[0])[:3,3]
                delta = self.model.points[self.active[visible]] - center
                probes = np.concatenate([center + fraction*delta for fraction in self.fractions])
                free = self.classify_history(probes, self.mapper.keyframes)["strict_free"]
                accepted[visible] = np.asarray(free).reshape(len(self.fractions), len(visible)).all(axis=0)
        self.cached[key] = accepted
        return accepted

    def score(self, states):
        coverage = np.zeros(len(self.active), dtype=bool)
        for pose in states[1:]:
            coverage |= self._pose(pose)
        return float(self.uplift[coverage].sum()), int(coverage.sum())


def compare_routes(state, rows, settings, weights, tolerance, rank_direct, rank_admitted, classify_history):
    mapper = state["mapper"]
    cache = mapper._dual_template_state["cache"]
    assert cache["predictor"] == "visibility"
    d, direct = rank_direct(
        state, rows, weights, settings["direct_calibration_scale"], tolerance, rank_admitted)
    gate = SupportedHidden(mapper, cache["models"], cache["ray"], cache["gains"]["S"],
                           classify_history, settings["supported_hidden"])
    base = {row["route_id"]:row["area_m2"] for row in direct["scored_pool"]}
    scale = settings["supported_hidden"]["increment_weight"]
    eligible = []
    for row in rows:
        if row["admitted"]:
            added, support_points = gate.score(row["outbound_states"])
            scores = row["scores"] | {"D":base[row["route_id"]], "D+":base[row["route_id"]]+scale*added}
            eligible.append(row | {"scores":scores, "supported_hidden_m2":added,
                                   "supported_hidden_quadrature":support_points})
    native = next(row for row in eligible if row["route_id"] == "native")
    winners, rankings = {"D":d["route_id"]}, {"D":direct["scored_pool"]}
    for arm in ("D+", "S"):
        winner, ranking = rank_admitted(eligible, native, arm, weights, mapper.config.resolution_m**2, tolerance)
        winners[arm] = winner["route_id"]
        rankings[arm] = ranking
    picked = set(winners.values()) | {native["route_id"]}
    if len(picked) == 1:
        other = min((row for row in eligible if row["route_id"] not in picked),
                    key=lambda row: (row["outbound_cost"], row["outbound_sha256"]))
        picked.add(other["route_id"])
    branch_routes = [row for row in eligible if row["route_id"] in picked]
    assert len({row["outbound_sha256"] for row in branch_routes}) == len(branch_routes)
    return {
        "selected": winners, "rankings":rankings, "eligible_full_pool":eligible,
        "branch_routes":branch_routes, "direct":direct, "support_tracks":gate.track_audit,
    }
