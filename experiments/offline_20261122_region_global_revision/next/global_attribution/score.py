"""Preserve a candidate pool while attributing regional and native route gains."""

from math import isclose


def rerank(planner, audit, routes, native_rows, resolution, target, outbound):
    """Change only score terms. All eligible candidates and tie rules come from v1."""
    by_pose = {tuple(row["pose"]): row for row in routes}
    by_id = {row["candidate_id"]: row for row in native_rows}
    regions = {row["region_id"]: row for row in audit["regions"]}
    params = planner.config
    ranked = []
    for original in audit["ranking"]:
        pose = tuple(original["pose"])
        standing = original["region_id"]
        hint = original.get("target_region_hint")
        proposal = planner.latest_proposals.get(pose, {})
        doorway = proposal.get("proposal_kind") == "doorway_view" and original.get("entrance_id") is not None
        credited = hint if target and doorway and hint in regions else standing
        region = regions.get(credited, {})
        frontier = region.get("remaining_frontier_cells", 0)
        unvisited = bool(region and not region["visited"])
        topology_delta = params.unvisited_bonus_m2 * (unvisited - original["unvisited_region"])
        topology_delta += params.frontier_m2_per_m * (
            min(frontier * resolution, params.frontier_m_cap)
            - min(original["remaining_frontier_cells"] * resolution, params.frontier_m_cap)
        )
        endpoint = min(original["visible_unknown_cells"] * resolution**2, params.unknown_m2_cap)
        native = by_id[by_pose[pose]["candidate_id"]]
        native_coverage = native["coverage_only_proxy"]
        native_quality = native["quality_total_proxy"]
        native_gain = native_coverage + native_quality
        if not isclose(native_gain, native["common_total_proxy"], abs_tol=1e-10, rel_tol=1e-12):
            raise ValueError("Native full outbound components do not add to the measured geometric score.")
        new_score = original["score_m2"] + topology_delta + (native_gain - endpoint if outbound else 0.0)
        row = original | {
            "standing_region_id": standing, "credited_region_id": credited,
            "remaining_frontier_cells": frontier, "unvisited_region": unvisited,
            "endpoint_visibility_capped_m2": endpoint, "native_outbound_coverage_m2": native_coverage,
            "native_outbound_measured_quality_m2": native_quality, "attribution_delta_m2": topology_delta,
            "original_score_m2": original["score_m2"], "score_m2": new_score,
        }
        total_travel = original["outbound_actions"] + original["return_actions"]
        ranked.append((-new_score, total_travel, pose, row))
    ranked.sort(key=lambda row: row[:3])
    rows = [item[3] for item in ranked]
    return by_pose[tuple(rows[0]["pose"])], rows
