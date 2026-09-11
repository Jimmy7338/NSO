"""Post-run diagnosis only; archived raw mapper plus independent orientation BFS.

No world construction, future sensor rendering, evaluator surface, GT map,
new policy outcome or coefficient selection. Production candidate generation
is replayed only to recover its actual pool; independent BFS defines the
larger physically legal pool and supplies explicit return-path witnesses.
"""
import hashlib
import json
import os
from pathlib import Path
from collections import Counter, deque
import sys
import tempfile
from types import SimpleNamespace
import zipfile

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
import numpy as np

ROOT = Path("/root/NSO")
RUN = ROOT / "eval_results/cpu_four_module_v10_integration_20260911"
OUT = Path(__file__).resolve().parent
DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    return value


def safe_stencil(belief):
    # Frozen radius/resolution=1: obstacle cells within 1+sqrt(2)/2 cells
    # give exactly this all-free 3x3 stencil, including the padded boundary.
    free = np.pad(belief == 0, 1, constant_values=False)
    safe = np.ones(belief.shape, bool)
    for dr in range(3):
        for dc in range(3):
            safe &= free[dr:dr + belief.shape[0], dc:dc + belief.shape[1]]
    return safe


def bfs(safe, start, reverse=False):
    distance, parent = {start: 0}, {start: None}
    if not safe[start[:2]]:
        return {}, {}
    queue = deque([start])
    while queue:
        state = queue.popleft()
        r, c, h = state
        dr, dc = DIRS[h]
        moves = [(r, c, (h - 1) % 4), (r, c, (h + 1) % 4),
                 (r - dr, c - dc, h) if reverse else (r + dr, c + dc, h)]
        for nxt in moves:
            if (0 <= nxt[0] < safe.shape[0] and 0 <= nxt[1] < safe.shape[1]
                    and safe[nxt[:2]] and nxt not in distance):
                distance[nxt] = distance[state] + 1
                parent[nxt] = state
                queue.append(nxt)
    return distance, parent


def chain(parents, state):
    result = []
    while state is not None:
        result.append(state)
        state = parents[state]
    return result


def actions(states):
    result = []
    for a, b in zip(states, states[1:]):
        if a[:2] == b[:2]:
            result.append("right" if (b[2] - a[2]) % 4 == 1 else "left")
        else:
            assert b == (a[0] + DIRS[a[2]][0], a[1] + DIRS[a[2]][1], a[2])
            result.append("forward")
    return result


def main():
    manifest, raw = read(RUN / "manifest.json"), read(RUN / "raw_manifest.json")
    calls = read(RUN / "module_calls.json")
    choices = {c["action_id"]: c for c in calls if c["method"] == "select_topo_target"}
    expected_hashes = read(RUN / "artifact_hashes.json")
    assert all(sha(RUN / name) == expected for name, expected in expected_hashes.items())
    freeze = read(RUN / "freeze.json")
    assert sha(RUN / "sources.zip") == freeze["archive_sha256"]
    rows, frames, all_candidates = [], {}, []
    with tempfile.TemporaryDirectory(prefix="cpu-v10-stall-archive-") as temp:
        archive_root = Path(temp)
        with zipfile.ZipFile(RUN / "sources.zip") as archive:
            for name, expected in freeze["source_sha256"].items():
                payload = archive.read(name)
                assert hashlib.sha256(payload).hexdigest() == expected
                target = archive_root / name
                assert target.resolve().is_relative_to(archive_root)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(payload)
        sys.path.insert(0, str(archive_root))
        from nso.observed_runtime_mapper_v10 import ObservedRuntimeMapperV10
        from nso.hierarchical_options_v10 import generate_options
        from nso.response_features_v7 import response_features
        from utils.grid_geometry import visible_mask, inflated_obstacles
        from utils.rgbd_contract import RGBDFrame, PlanarScan
        for name in ("nso.observed_runtime_mapper_v10", "nso.hierarchical_options_v10",
                     "nso.response_features_v7", "utils.grid_geometry", "utils.rgbd_contract"):
            assert Path(sys.modules[name].__file__).is_relative_to(archive_root)
        config = SimpleNamespace(**manifest["config"])
        assert config.resolution_m == config.robot_radius_m == .2
        mapper = ObservedRuntimeMapperV10(tuple(manifest["shape"]), config)
        anchor = tuple(manifest["initial_anchor"])
        for packet in raw:
            frame = RGBDFrame.load(RUN / packet["frame_path"])
            scan = PlanarScan.load(RUN / packet["scan_path"])
            mapper.update(frame, scan)
            step = packet["action_id"]
            frames[step] = dict(belief=mapper.belief.copy(), camera_seen=mapper.camera_seen.copy())
            if step not in choices:
                continue
            logged = choices[step]
            current = (*packet["position"], packet["heading"])
            budget = manifest["total_budget"] - step
            safe = safe_stencil(mapper.belief)
            assert hashlib.sha256(safe.tobytes()).hexdigest() == logged["outputs"]["candidate_audit"]["safe_sha256"]
            outward, parent = bfs(safe, current)
            home, home_parent = bfs(safe, anchor, reverse=True)
            feasible = sorted(s for s, cost in outward.items()
                              if cost > 0 and s in home and cost + home[s] <= budget)
            assert len(feasible) == logged["outputs"]["candidate_audit"]["feasible_states"]
            translated = [s for s in feasible if s[:2] != current[:2]]
            witness = min(translated, key=lambda s: (outward[s] + home[s], outward[s], s)) if translated else None
            proof = None
            if witness is not None:
                out = chain(parent, witness)[::-1]
                back = chain(home_parent, witness)
                proof = dict(target=witness, outbound_states=out, outbound_actions=actions(out),
                             return_states=back, return_actions=actions(back),
                             outbound_cost=outward[witness], return_cost=home[witness],
                             total_cost=outward[witness] + home[witness], anchor=anchor)
                assert out[0] == current and back[-1] == anchor
                assert all(safe[s[:2]] for s in out + back)

            # Recover production's actual submitted pool; never use it as an
            # oracle for independent reachability or for selecting a new run.
            routes, audit = generate_options(mapper, current[:2], current[2], budget, anchor,
                                             max_candidates=manifest["protocol"]["max_candidates"])
            assert clean(audit) == logged["outputs"]["candidate_audit"]
            assert len(routes) == len(logged["outputs"]["score_audit"])
            selected = logged["outputs"]["selected"]
            assert all(clean(v) == selected[k] for k, v in routes[selected["candidate_id"]].items())
            for route in routes:
                target = tuple(route["pose"])
                assert target in feasible
                assert route["outbound_cost"] == outward[target]
                assert route["return_cost"] == home[target]
                all_candidates.append(dict(action_id=step, **route))

            unknown = (mapper.belief == -1) & ~inflated_obstacles(mapper.belief == 1, 1.)
            camera_unknown = ~mapper.camera_seen & (mapper.belief != 1)
            ranges = int(config.max_depth_m / config.resolution_m)
            def masks(state):
                return (visible_mask(mapper.belief == 1, state[:2], state[2], ranges, 360) & unknown,
                        visible_mask(mapper.belief == 1, state[:2], state[2], ranges, config.fov_deg) & camera_unknown)
            pool = []
            for state in feasible:
                radar, camera = masks(state)
                count = int(radar.sum() + camera.sum())
                cost = outward[state] + home[state]
                pool.append(dict(pose=state, radar_unknown_cells=int(radar.sum()),
                                 camera_unknown_cells=int(camera.sum()), unknown_cells=count,
                                 cost=cost, rate=count / cost))
            best_rate = min(pool, key=lambda r: (-r["rate"], r["cost"], r["pose"]))
            best_total = min(pool, key=lambda r: (-r["unknown_cells"], r["cost"], r["pose"]))
            moving_pool = [r for r in pool if r["pose"][:2] != current[:2]]
            best_move = min(moving_pool, key=lambda r: (-r["rate"], r["cost"], r["pose"])) if moving_pool else None
            best_move_total = min(moving_pool, key=lambda r: (-r["unknown_cells"], r["cost"], r["pose"])) if moving_pool else None
            coverage = next(r for r in routes if r["group"] == "coverage_anchor")
            assert list(best_rate["pose"]) == coverage["pose"]

            features, _ = response_features(mapper, [dict(r, states=r["outbound_states"], cost=r["outbound_cost"]) for r in routes])
            q = mapper.quality_evidence()
            npoints = 0 if q is None else len(q["point"])
            score_parts = []
            for index, route in enumerate(routes):
                c = route["outbound_cost"]
                a = features["N"][index]
                parts = dict(candidate_id=route["candidate_id"], group=route["group"],
                             pose=route["pose"], radar_area_proxy_m2=float(a[0] * 16 * c),
                             camera_area_proxy_m2=float(a[1] * 16 * c),
                             quality_proxy=float(a[2] * max(1, npoints) * .15**2 * c))
                assert abs(sum(parts[k] for k in ("radar_area_proxy_m2", "camera_area_proxy_m2", "quality_proxy"))
                           - logged["outputs"]["score_audit"][index]["common_total_proxy"]) < 1e-9
                score_parts.append(parts)
            predicted_radar, predicted_camera = np.zeros(mapper.shape, bool), np.zeros(mapper.shape, bool)
            for state in selected["outbound_states"][1:]:
                r, c = masks(tuple(state))
                predicted_radar |= r
                predicted_camera |= c
            row = dict(action_id=step, call_id=logged["call_id"], remaining_budget=budget,
                       safe_cell_count=int(safe.sum()), reachable_cell_count=len({s[:2] for s in outward}),
                       feasible_state_count=len(feasible), feasible_translation_state_count=len(translated),
                       feasible_translation_cell_count=len({s[:2] for s in translated}),
                       independent_translation_witness=proof,
                       generated_candidate_count=len(routes), generated_translation_count=sum(r["pose"][:2] != list(current[:2]) for r in routes),
                       observed_asset_count=audit["observed_asset_count"],
                       missing_asset_role_reasons=dict(Counter(r["reason"] for r in audit["target_audit"])),
                       selected_group=selected["group"], selected_pose=selected["pose"],
                       selected_outbound_cost=selected["outbound_cost"],
                       selected_total_score=logged["outputs"]["scores"]["S"][selected["candidate_id"]],
                       selected_score_parts=score_parts[selected["candidate_id"]],
                       candidate_score_parts=score_parts,
                       rate_prefilter_winner=best_rate, largest_endpoint_total_proxy=best_total,
                       best_translation_rate=best_move, largest_translation_endpoint_proxy=best_move_total,
                       _predicted_radar=predicted_radar, _predicted_camera=predicted_camera)
            rows.append(row)
        with np.load(RUN / "final_map.npz", allow_pickle=False) as final:
            np.testing.assert_array_equal(mapper.belief, final["belief"])
            np.testing.assert_array_equal(mapper.camera_seen, final["camera_seen"])

    for row in rows:
        start = row["action_id"]
        end = start + row["selected_outbound_cost"]
        before, after = frames[start], frames[end]
        fresh = (after["belief"] != -1) & (before["belief"] == -1)
        new_camera = after["camera_seen"] & ~before["camera_seen"]
        radar, camera = row.pop("_predicted_radar"), row.pop("_predicted_camera")
        row.update(actual_arrival_action_id=end, actual_new_observed_2d_cells=int(fresh.sum()),
                   actual_new_camera_projection_cells=int(new_camera.sum()),
                   predicted_radar_unknown_cells=int(radar.sum()), predicted_camera_unknown_cells=int(camera.sum()),
                   realized_predicted_camera_cells=int((camera & new_camera).sum()),
                   predicted_camera_cells_still_unseen_at_arrival=int((camera & ~after["camera_seen"]).sum()),
                   predicted_camera_cells_still_unseen_at_run_end=int((camera & ~frames[32]["camera_seen"]).sum()))

    paid = [c["outputs"] for c in calls if c["method"] == "compute_reward"]
    semantic = read(ROOT / "audit_results/cpu_semantic_packet_v10_development_20260911/verification.json")
    summary = dict(schema="cpu_v10_rotation_stall_review/1", scope="post-run development diagnosis; no new policy execution",
        status="completed", root_cause="candidate screening and non-realizable persistent observation proxy; not absence of known-safe translation space",
        input_run=str(RUN), raw_packets=len(raw), paid_actions=len(raw) - 1,
        actions=dict(Counter(p["action"] for p in raw[1:])),
        distinct_cells=len({tuple(p["position"]) for p in raw}),
        distinct_full_poses=len({(*p["position"], p["heading"]) for p in raw}),
        choices=len(rows), submitted_candidates=len(all_candidates),
        submitted_translation_candidates=sum(r["pose"][:2] != raw[0]["position"] for r in all_candidates),
        choices_with_budget_feasible_translation=sum(r["feasible_translation_state_count"] > 0 for r in rows),
        choices_without_budget_feasible_translation=[r["action_id"] for r in rows if not r["feasible_translation_state_count"]],
        rate_prefilter_at_current_cell_count=sum(r["rate_prefilter_winner"]["pose"][:2] == tuple(raw[0]["position"]) for r in rows),
        choices_where_translation_endpoint_total_proxy_exceeds_selected_coverage=sum(
            r["largest_translation_endpoint_proxy"] is not None and r["largest_translation_endpoint_proxy"]["unknown_cells"] > r["rate_prefilter_winner"]["unknown_cells"] for r in rows),
        all_47_semantic_potentials_zero=all(all(float(v) == 0 for v in a["prior_potentials"].values()) for c in choices.values() for a in c["outputs"]["score_audit"]),
        action_observations_with_zero_new_2d=sum(p["new_observed_cell_count"] == 0 for p in paid),
        action_observations_with_zero_new_camera_2d=sum(p["parts"]["new_camera_observed_2d_area_m2"] == 0 for p in paid),
        igcr_no_progress_events=sum(p["no_progress"] for p in paid),
        summed_predicted_selected_camera_cells=sum(r["predicted_camera_unknown_cells"] for r in rows),
        summed_realized_predicted_selected_camera_cells=sum(r["realized_predicted_camera_cells"] for r in rows),
        selected_plans_zero_realized_predicted_camera_cells=sum(r["realized_predicted_camera_cells"] == 0 for r in rows),
        independent_full_33_packet_mapper_replay=True, final_belief_camera_exact_match=True,
        independent_footprint_and_bfs=True, archived_candidate_pool_reproduced=True,
        archived_scores_common_breakdown_reproduced=True, source_and_all_run_artifact_hashes_verified=True,
        world_constructed=False, new_future_sensor_actions=0, gt_map_read=False, evaluator_reconstruction_metrics_computed=False,
        rule_or_source_modified=False, semantic_packet_intervention_scope={
            "verification_path":"audit_results/cpu_semantic_packet_v10_development_20260911/verification.json",
            "actual_prefix_packets_per_replay":semantic["actual_prefix_packets_per_replay"],
            "actual_prefix_replays":semantic["actual_prefix_replays"],
            "S_score_changed":semantic["S_score_changed"],
            "interpretation":"Separate 150-paid-action prefix condition; input causality does not show semantics activated or effective in this fresh episode."},
        source_archive_sha256=sha(RUN / "sources.zip"), run_manifest_sha256=sha(RUN / "manifest.json"),
        raw_manifest_sha256=sha(RUN / "raw_manifest.json"), module_calls_sha256=sha(RUN / "module_calls.json"),
        worker_sha256=sha(__file__), decisions=rows)
    (OUT / "review.json").write_text(json.dumps(clean(summary), indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    (OUT / "reproduced_candidate_pool.json").write_text(json.dumps(clean(all_candidates), indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(clean({k: v for k, v in summary.items() if k != "decisions"}), ensure_ascii=False, indent=2))
    print("witnesses", json.dumps(clean([rows[i] for i in (0, 3, 10, 21, 24)]), ensure_ascii=False))


if __name__ == "__main__":
    main()
