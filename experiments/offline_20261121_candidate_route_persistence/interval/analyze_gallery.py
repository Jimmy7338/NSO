"""Explain archived Gallery candidate 2/3 outcomes between paid actions 38 and 58."""

import argparse
from collections import Counter
import csv
import gzip
import importlib.util
import itertools
import json
import math
from pathlib import Path

ANALYSES = Path(__file__).resolve().parents[2]
SHORT = ANALYSES / "candidate_route_yield_20261121"
OLD = ANALYSES / "semantic_large_scene_20261121"
OUT = Path(__file__).resolve().parent
LAYOUTS = ("gallery_left", "gallery_right")
START = 38
END = 58
Y_BANDS = ("launch_court", "approach", "divided_lanes", "merged_transfer", "annex")
X_BANDS = ("left", "center", "right")
HEIGHT_BANDS = ("near_ground", "elevated")
TOLERANCE = 0.05
HEADINGS = ("north", "east", "south", "west")


def import_measure():
    path = SHORT / "measurement/measure.py"
    spec = importlib.util.spec_from_file_location("frozen_short_measurement", path)
    measure = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(measure)
    return measure


def require(condition, description):
    if not condition:
        raise ValueError(description)


def pose(step):
    return [*step["position"], step["heading"]]


def xy(grid_pose, scene):
    resolution = scene["width_m"] / scene["shape"][1]
    row, col = grid_pose[:2]
    return (round((col + 0.5) * resolution, 12), round(scene["height_m"] - (row + 0.5) * resolution, 12))


def equal(a, b):
    return math.isclose(a, b, abs_tol=1e-12, rel_tol=0)


def read_bound(m, path, manifest, relative):
    m.verify(path, manifest["files_sha256"][relative])
    return m.read_json(path)


def paths_and_choices(run, checkpoint, scene):
    steps, audit = run["steps"], run["audit"]
    run_id = (checkpoint["layout"], checkpoint["candidate_id"])
    frame_choices = {
        int(choice["frame_id"].removeprefix("frame-")): choice
        for choice in audit if choice["event"] == "global_choice"
    }
    require(set(frame_choices) & set(range(START, END)) == {START}, f"Unexpected neutral replans in {run_id}")
    signed = frame_choices[START]["option"]
    primary = frame_choices[0]["option"]
    first_target, final = pose(steps[START - 1]), pose(steps[END - 1])
    require(first_target == primary["pose"] and steps[START - 1]["matched_target_pose_after_step"],
            f"Original candidate did not reach its target at step {START}: {run_id}")
    require(first_target == signed["outbound_states"][0], f"Neutral choice has different initial pose in {run_id}")
    window = steps[START:END]
    require([step["step"] for step in window] == list(range(START + 1, END + 1)), f"Invalid action indices in {run_id}")
    require([step["action"] for step in window] == signed["outbound_actions"][:END - START]
            and [pose(step) for step in window] == signed["outbound_states"][1:END - START + 1]
            and all(step["active_option_id"] == signed["option_id"]
                    and step["phase_before_action"] == "outbound" and not step["collision"] for step in window),
            f"Observed execution diverged from archived selected neutral option in {run_id}")
    require(not any(step["matched_target_pose_after_step"] for step in window),
            f"Unexpected continued-target arrival through {END} in {run_id}")
    require(len(signed["outbound_actions"]) == signed["outbound_cost"]
            and len(signed["return_actions"]) == signed["return_cost"],
            f"Archived costs disagree with path in {run_id}")
    observed_actions = Counter(step["action"] for step in window)
    positions = [first_target] + [pose(step) for step in window]
    directions = Counter(
        ("north" if b[0] < a[0] else "south" if b[0] > a[0] else "west" if b[1] < a[1]
         else "east" if b[1] > a[1] else "rotation_or_hold")
        for a, b in zip(positions, positions[1:])
    )
    start_x, start_y = xy(first_target, scene)
    goal_x, goal_y = xy(signed["pose"], scene)
    actual_x, actual_y = xy(final, scene)
    trace_rows = []
    for i, step in enumerate(window, 1):
        planned = signed["outbound_states"][i]
        prior = positions[i - 1]
        actual = pose(step)
        x, y = xy(actual, scene)
        dx, dy = (x - xy(prior, scene)[0], y - xy(prior, scene)[1])
        trace_rows.append({
            **checkpoint, "paid": step["step"], "neutral_target_id": step["active_candidate_id"],
            "neutral_target_group": signed["group"], "goal_x_m": goal_x, "goal_y_m": goal_y,
            "command_action": signed["outbound_actions"][i-1], "actual_action": step["action"],
            "planned_grid_pose": json.dumps(planned), "actual_grid_pose": json.dumps(actual),
            "actual_x_m": x, "actual_y_m": y, "actual_heading": HEADINGS[step["heading"]],
            "dx_m": dx, "dy_m": dy, "collision": step["collision"], "target_reached": step["matched_target_pose_after_step"],
            "packet_sha256": step["packet_sha256"], "packet_chain_sha256": step["packet_chain_sha256"],
        })
    counters = {
        **{f"actions_{action}": observed_actions[action] for action in ("forward", "left", "right")},
        **{f"steps_{heading}": directions[heading] for heading in HEADINGS},
    }
    return ({
        **checkpoint, "second_choice_paid": START, "checkpoint_paid": END,
        "neutral_target_id": signed["candidate_id"], "neutral_target_group": signed["group"],
        "neutral_target_grid_pose": json.dumps(signed["pose"]),
        "neutral_target_x_m": goal_x, "neutral_target_y_m": goal_y,
        "neutral_target_heading": HEADINGS[signed["pose"][2]],
        "neutral_planned_outbound": signed["outbound_cost"], "neutral_reserved_return": signed["return_cost"],
        "start_grid_pose": json.dumps(first_target), "start_x_m": start_x, "start_y_m": start_y,
        "end_grid_pose": json.dumps(final), "end_x_m": actual_x, "end_y_m": actual_y,
        "end_heading": HEADINGS[final[2]], "movement_actions": sum(directions[direction] for direction in HEADINGS),
        "translation_m": steps[END - 1]["path_length_m"] - steps[START - 1]["path_length_m"],
        "neutral_target_reached_by_end": False, "matches_archived_neutral_outbound_prefix": True,
        **counters,
    }, trace_rows)


def partition(scene, np):
    annex = [box[1] for box in scene["wall_boxes_xyz_size"]
             if box[1] > scene["rejoin_y"] and box[4] == 0.2 and box[3] < scene["width_m"]]
    require(len(set(annex)) == 1, "Cannot infer original Gallery annex wall from frozen architecture")
    y_cuts = [scene["split_y"], scene["lane_test_y"], scene["rejoin_y"], annex[0]]
    route = scene["route_evaluation"]
    x_cuts = [round(route["left_limit_x_m"], 12), round(route["right_limit_x_m"], 12)]

    def bins(xs, ys):
        return (np.searchsorted(x_cuts, xs, side="right"), np.searchsorted(y_cuts, ys, side="right"))

    return bins, x_cuts, y_cuts


def compare(before, after, selector, np):
    before, after = before[selector], after[selector]
    gross = int(np.count_nonzero(~before & after))
    lost = int(np.count_nonzero(before & ~after))
    return {
        "reference_count": len(before), "represented_38": int(before.sum()), "represented_58": int(after.sum()),
        "retained_count": int(np.count_nonzero(before & after)),
        "gross_new_count": gross, "lost_count": lost, "net_count": gross - lost,
    }


def section_coverage(first, second, reference, scene, bins, meta, np):
    reachable = reference["reachable"]
    rows, cols = np.indices(reachable.shape)
    res = scene["width_m"] / scene["shape"][1]
    xs = (cols[reachable] + 0.5) * res
    ys = scene["height_m"] - (rows[reachable] + 0.5) * res
    x_idx, y_idx = bins(xs, ys)
    a, b = first[reachable] == 0, second[reachable] == 0
    records = []
    for xi, yi in itertools.product(range(len(X_BANDS)), range(len(Y_BANDS))):
        row = compare(a, b, (x_idx == xi) & (y_idx == yi), np)
        records.append({
            **meta, "x_zone": X_BANDS[xi], "y_zone": Y_BANDS[yi], **row,
            "net_m2": row["net_count"] * res**2,
            "net_fraction_of_total_true_reachable": row["net_count"] / int(reachable.sum()),
        })
    require(sum(r["reference_count"] for r in records) == int(reachable.sum()), "Coverage partition is incomplete")
    return records


def section_surfaces(before, after, reference, bins, meta, np):
    records = []
    for group, points in reference["samples"].items():
        x_idx, y_idx = bins(points[:, 0], points[:, 1])
        height_idx = points[:, 2] > 0.10
        for xi, yi, zi in itertools.product(range(len(X_BANDS)), range(len(Y_BANDS)), range(len(HEIGHT_BANDS))):
            row = compare(before[group], after[group], (x_idx == xi) & (y_idx == yi) & (height_idx == zi), np)
            weight = reference["weights"][group]
            records.append({
                **meta, "surface_reference_group": group, "x_zone": X_BANDS[xi], "y_zone": Y_BANDS[yi],
                "height_band": HEIGHT_BANDS[zi], "m2_per_sample": weight, **row,
                "gross_new_m2": row["gross_new_count"] * weight, "lost_m2": row["lost_count"] * weight,
                "net_m2": row["net_count"] * weight,
            })
        require(sum(row["reference_count"] for row in records if row["surface_reference_group"] == group) == len(points),
                f"Incomplete sample partition for {group}")
    return records


def decompose(a, b, meta):
    """Exact, symmetric decomposition of the product change; endpoints may be routes or times."""
    ca, cb = a["C_free"], b["C_free"]
    fa, fb = a["F1_scene_05cm"], b["F1_scene_05cm"]
    d_c, d_f = cb - ca, fb - fa
    coverage_term = d_c * (fa + fb) / 2
    f1_term = d_f * (ca + cb) / 2
    delta_j = b["J"] - a["J"]
    require(equal(coverage_term + f1_term, delta_j), "J product terms do not recompose")
    return {
        **meta, "a_C_free": ca, "b_C_free": cb, "delta_C_free": d_c,
        "a_F1_scene_05cm": fa, "b_F1_scene_05cm": fb, "delta_F1_scene_05cm": d_f,
        "a_J": a["J"], "b_J": b["J"], "delta_J": delta_j,
        "coverage_term_delta_J": coverage_term, "F1_term_delta_J": f1_term,
    }


def write_rows(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(key for row in rows for key in row)), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def analyze(output):
    m = import_measure()
    np = m.np
    protocol_path = OUT.parent / "PROTOCOL.json"
    protocol = m.read_json(protocol_path)
    require(protocol["frozen_before_new_terminal_outcomes"]
            and protocol["measurement"]["main_interval"] == [START, END]
            and Path(protocol["source_local"]) == SHORT,
            "Archived diagnostic does not match the persistence protocol")
    published_path = SHORT / "publication/receipt.json"
    published = m.read_json(published_path)
    require(published["status"] == "verified_published" and published["head"] == protocol["source_commit"],
            "Short trial archive differs from the protocol's verified publication")
    checkpoint_source = SHORT / "measurement/results/candidate_rows.json"
    published_rows_gz = SHORT / "publication/payload/measurement/results/candidate_rows.json.gz"
    require(checkpoint_source.read_bytes() == gzip.decompress(published_rows_gz.read_bytes()),
            "Local checkpoint evidence differs from the verified publication payload")
    checkpoints = m.read_json(checkpoint_source)
    m.verify(SHORT / "measurement/measure.py", checkpoints["evaluator_sha256"])
    m.verify(m.NATIVE_METRIC, checkpoints["native_metric_sha256"])
    m.verify(m.RECON_METRIC, checkpoints["native_reconstruction_metric_sha256"])
    short_runs = SHORT / "runner/runs"
    sfile = OLD / "mechanism/scenes/freeze.json"
    rfile = OLD / "mechanism/runner/freeze.json"
    efile = OLD / "evaluation/freeze.json"
    for path, key in ((sfile, "old_scene_freeze_sha256"), (rfile, "old_runner_freeze_sha256"), (efile, "old_evaluation_freeze_sha256")):
        m.verify(path, checkpoints[key])
    sfreeze, rfreeze, efreeze = map(m.read_json, (sfile, rfile, efile))
    metric = m.load_metric(sfreeze, efreeze)
    received_rows = {(row["layout"], row["candidate_id"]): row for row in checkpoints["rows"]}
    inputs = {
        str(path): m.sha256(path) for path in (
            Path(__file__), protocol_path, published_path, checkpoint_source, published_rows_gz, sfile, rfile,
            efile, SHORT / "measurement/measure.py", m.NATIVE_METRIC, m.RECON_METRIC,
        )
    }
    output.mkdir(parents=True, exist_ok=True)
    texts = {key: [] for key in ("neutral_move_summary", "neutral_steps", "correct_free_by_region",
                                "fixed_surface_by_region", "J_product_decomposition", "interval_outcomes")}
    mask_paths = []
    for layout in LAYOUTS:
        scene = sfreeze["layout_specs"][layout]
        bins, xcuts, ycuts = partition(scene, np)
        reference = m.load_reference(layout, sfreeze, rfreeze)
        inputs.update(reference["files_sha256"])
        measures = {}
        for cid in (2, 3):
            branch = short_runs / layout / f"candidate_{cid}"
            row = received_rows[layout, cid]
            manifest_path = branch / "manifest.json"
            m.verify(manifest_path, row["source_hashes"]["run_manifest_sha256"])
            inputs[str(manifest_path)] = m.sha256(manifest_path)
            manifest = m.read_json(manifest_path)
            require(manifest["neutral_g_continuation"] and not manifest["evaluator_inputs_read"],
                    f"Wrong execution or actor access claim for {layout}/{cid}")
            for relative in ("audit.json", "steps.json", "first_anchor.json"):
                path = branch / relative
                m.verify(path, manifest["files_sha256"][relative])
                inputs[str(path)] = manifest["files_sha256"][relative]
            run = {"steps": m.read_json(branch / "steps.json"), "audit": m.read_json(branch / "audit.json")}
            native_side = row["side"]
            meta = {"layout": layout, "candidate_id": cid, "initial_lane": native_side,
                    "initial_object": "open_shelf" if native_side == scene["shelf_side"] else "closed_cabinet"}
            move, trace = paths_and_choices(run, meta, scene)
            texts["neutral_move_summary"].append(move)
            texts["neutral_steps"].extend(trace)
            beliefs, masks, measures[cid] = {}, {}, {}
            for label, paid in (("arrival", START), ("paid_58", END)):
                check = row["checkpoints"][label]
                require(check["status"] == "captured" and check["actual_paid"] == paid,
                        f"Invalid archived paid checkpoint for {layout}/{cid}/{label}")
                checkpoint = read_bound(m, branch / label / "checkpoint.json", manifest, f"{label}/checkpoint.json")
                require(checkpoint["packet_sha256"] == run["steps"][paid-1]["packet_sha256"],
                        f"Snapshot does not match trace for {layout}/{cid}/{label}")
                inputs[str(branch / label / "checkpoint.json")] = manifest["files_sha256"][f"{label}/checkpoint.json"]
                for name in ("native_mesh.ply", "belief.npz"):
                    path = branch / label / name
                    m.verify(path, checkpoint["files_sha256"][name])
                    inputs[str(path)] = checkpoint["files_sha256"][name]
                belief, snapshot_pose = m.read_belief(branch / label / "belief.npz")
                require(snapshot_pose == pose(run["steps"][paid-1]), f"Snapshot pose mismatch for {layout}/{cid}/{label}")
                mesh = m.read_mesh(branch / label / "native_mesh.ply")
                ray = metric.ray_scene(mesh)
                state = {group: ray.compute_distance(m.o3d.core.Tensor(points), nthreads=1).numpy() <= TOLERANCE
                         for group, points in reference["samples"].items()}
                require(equal(float(state["scene"].mean()), check["r_scene_05cm"])
                        and int(np.count_nonzero((belief == 0) & reference["reachable"])) == check["correct_free_cells"],
                        f"Frozen area/coverage reproduction differs for {layout}/{cid}/{label}")
                masks[paid], beliefs[paid], measures[cid][paid] = state, belief, check
            coverage = section_coverage(beliefs[START], beliefs[END], reference, scene, bins, meta, np)
            surfaces = section_surfaces(masks[START], masks[END], reference, bins, meta, np)
            texts["correct_free_by_region"].extend(coverage)
            texts["fixed_surface_by_region"].extend(surfaces)
            texts["J_product_decomposition"].append(decompose(measures[cid][START], measures[cid][END], {
                **meta, "comparison": "within_route_58_minus_38",
            }))
            a, b = (measures[cid][paid] for paid in (START, END))
            overall = {
                **meta,
                "C_free_38": a["C_free"], "C_free_58": b["C_free"],
                "F1_scene_05cm_38": a["F1_scene_05cm"], "F1_scene_05cm_58": b["F1_scene_05cm"],
                "precision_05cm_38": a["p_scene_05cm"], "precision_05cm_58": b["p_scene_05cm"],
                "recall_05cm_38": a["r_scene_05cm"], "recall_05cm_58": b["r_scene_05cm"],
                "J_38": a["J"], "J_58": b["J"],
            }
            for field in ("gross_new_count", "lost_count", "net_count"):
                overall[f"correct_free_{field}"] = sum(r[field] for r in coverage)
            require(overall["correct_free_net_count"] == b["correct_free_cells"] - a["correct_free_cells"],
                    f"Correct-free count disagrees with measured difference for {layout}/{cid}")
            for group in reference["samples"]:
                for field in ("gross_new_m2", "lost_m2", "net_m2"):
                    overall[f"{group}_{field}"] = sum(r[field] for r in surfaces if r["surface_reference_group"] == group)
            texts["interval_outcomes"].append(overall)
            path = output / f"{layout}_candidate_{cid}_presence.npz"
            np.savez_compressed(path, **{
                "reachable": reference["reachable"], **{f"free_{paid}": belief == 0 for paid, belief in beliefs.items()},
                **{f"{group}_{paid}": mask for paid, groups in masks.items() for group, mask in groups.items()},
            })
            mask_paths.append(path)
        for step in (START, END):
            texts["J_product_decomposition"].append(decompose(measures[3][step], measures[2][step], {
                "layout": layout, "comparison": "candidate_2_minus_3", "paid": step,
                "a_candidate_id": 3, "b_candidate_id": 2,
            }))
        archive = received_rows[layout, 1]
        original = archive["checkpoints"]["paid_58"]
        require(archive["archive_selected"] and original["status"] == "captured" and original["actual_paid"] == END,
                f"Original candidate's equal-paid measure unavailable for {layout}")
        for cid in (2, 3):
            texts["J_product_decomposition"].append(decompose(original, measures[cid][END], {
                "layout": layout, "comparison": "candidate_minus_original_1", "paid": END,
                "a_candidate_id": 1, "b_candidate_id": cid,
            }))
        a = decompose(measures[3][START], measures[2][START], {})
        b = decompose(measures[3][END], measures[2][END], {})
        texts["J_product_decomposition"].append({
            "layout": layout, "comparison": "gap_change_58_minus_38",
            **{key: b[key] - a[key] for key in ("delta_J", "coverage_term_delta_J", "F1_term_delta_J")},
        })
    paths = [output / f"{name}.csv" for name in texts]
    for path, rows in zip(paths, texts.values()):
        write_rows(path, rows)
    m.write_json(output / "interval_receipt.json", {
        "status": "archived_short_natural_neutral_continuation_measured_offline",
        "source_commit": published["head"],
        "actor_started": False, "branch_count": 4, "parent_count": 1, "paid_interval": [START, END],
        "source_regions": "Every reachable grid center and fixed reference sample is assigned to one exhaustive band. Bands are fixed floorplan coordinates; surface groups include scene-all and separately sampled objects.",
        "zone_cuts_x_m": xcuts, "zone_cuts_y_m": ycuts, "height_cut_m": 0.10,
        "zone_boundary_membership": "At a cut, use the higher coordinate band. Near-ground surfaces include z <= .10 m.",
        "whole_scene_surface": "Fixed reference recall at 5 cm, weighted by original area per sample. Precision is separately sourced from the original predicted-mesh evaluation.",
        "J_symmetric_product": "delta(C*F) = delta(C)*mean(F) + delta(F)*mean(C)",
        "sha256_inputs": dict(sorted(inputs.items())),
        "sha256_outputs": {path.name: m.sha256(path) for path in paths + mask_paths},
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    analyze(parser.parse_args().output)
