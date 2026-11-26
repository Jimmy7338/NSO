#!/usr/bin/env python3
"""Draw the two registered A96 qualitative contrasts from native run files."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import numpy as np
import open3d as o3d


HERE = Path(__file__).resolve().parent
DEFAULT_STUDY = HERE.parents[1] / "region_independent_floors_20261123"
LAYOUTS = (("ind26_offset_ring", "Offset ring"), ("ind26_double_loop", "Double loop"))
ARMS = (("region_S", "Original regional planner", "#2e6487"),
        ("combined_S", "Full regional planner", "#187968"))
PAID = 96
GRID_COLORS = ("#d8dee4", "#fbfcfa", "#2a3948")
PAPER = "#ffffff"
INK = "#1d2b36"
MUTE = "#51616e"
AMBER = "#bb5c22"
TEAL = "#227465"
AMBER_DARK = "#91461b"
POSE_DIRECTIONS = np.array([[0, 1], [1, 0], [0, -1], [-1, 0]])


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def checked(path, digest):
    if sha(path) != digest:
        raise ValueError(f"Frozen artifact hash mismatch at {path}")
    return path


def metric_record(csv_path, layout, arm):
    with csv_path.open(newline="") as stream:
        row, = (row for row in csv.DictReader(stream)
                if row["layout"] == layout and row["arm"] == arm
                and row["checkpoint"] == f"paid_{PAID}")
    if row["status"] != "captured" or int(row["actual_paid"]) != PAID:
        raise ValueError(f"Required actual A{PAID} snapshot missing for {layout}/{arm}")
    return row


def world_xy(pose, shape, resolution):
    row, col = pose[:2]
    return np.array([(col + .5) * resolution, (shape[0] - row - .5) * resolution])


def load_case(study, layout, arm, reachable):
    directory = study / "execution/formal" / layout / arm / "attempt_001/run"
    manifest_path = directory / "manifest.json"
    manifest = read(manifest_path)
    if (manifest["status"], manifest["layout"], manifest["arm"], manifest["actual_closed_loop"]) != (
            "completed_execution", layout, arm, True):
        raise ValueError(f"Missing completed native closed-loop evidence: {directory}")
    watched = ("start/checkpoint.json", f"paid_{PAID}/checkpoint.json", f"paid_{PAID}/belief.npz",
               f"paid_{PAID}/native_mesh.ply", "steps.json", "global_choices.json",
               "regional_choices.json")
    paths = {name: checked(directory / name, manifest["files_sha256"][name]) for name in watched}
    checkpoint = read(paths[f"paid_{PAID}/checkpoint.json"])
    origin = read(paths["start/checkpoint.json"])
    step_rows = read(paths["steps.json"])[:PAID]
    if (checkpoint["actual_paid"] != PAID or checkpoint["status"] != "captured"
            or len(step_rows) != PAID or [row["step"] for row in step_rows] != list(range(1, PAID+1))):
        raise ValueError(f"Missing or inconsistent paid action history for {layout}/{arm}")
    last = step_rows[-1]
    if last["packet_sha256"] != checkpoint["packet_sha256"] or (
            last["position"] + [last["heading"]] != checkpoint["pose"]):
        raise ValueError(f"A{PAID} trajectory differs from its checkpoint for {layout}/{arm}")
    with np.load(paths[f"paid_{PAID}/belief.npz"], allow_pickle=False) as data:
        belief = data["belief"]
        if belief.shape != reachable.shape or not np.isin(belief, (-1, 0, 1)).all():
            raise ValueError(f"Bad observed occupancy: {layout}/{arm}")
    report = metric_record(study / "results/formal/checkpoints.csv", layout, arm)
    known_free = np.count_nonzero(reachable & (belief == 0))
    coverage = known_free / np.count_nonzero(reachable)
    f1 = float(report["F1_scene_05cm"])
    joint = float(report["J"])
    if (known_free != int(report["correct_free_cells"])
            or not math.isclose(coverage, float(report["C_free"]), rel_tol=0, abs_tol=1e-12)
            or not math.isclose(joint, coverage * f1, rel_tol=0, abs_tol=1e-12)):
        raise ValueError(f"Metric/occupancy discrepancy: {layout}/{arm}")
    regional = read(paths["regional_choices.json"])
    global_choices = read(paths["global_choices.json"])
    recorded = {str(path.relative_to(study)): sha(path) for path in paths.values()}
    recorded[str(manifest_path.relative_to(study))] = sha(manifest_path)
    return {"origin": origin, "checkpoint": checkpoint, "belief": belief, "steps": step_rows,
            "mesh": paths[f"paid_{PAID}/native_mesh.ply"], "metrics": (coverage, f1, joint),
            "regional": regional, "global": global_choices, "sources": recorded}


def first_difference(cases):
    """Use both observed trajectories and ranking receipts to label the fork."""
    a, b = cases
    if a["origin"]["packet_sha256"] != b["origin"]["packet_sha256"]:
        raise ValueError("Not a paired cold start")
    kinematic = next((k, left, right)
                     for k, (left, right) in enumerate(zip(a["steps"], b["steps"], strict=True), 1)
                     if (left["action"], left["position"], left["heading"]) != (
                         right["action"], right["position"], right["heading"]))
    first_dispatch = next((k, left, right)
                          for k, (left, right) in enumerate(zip(a["steps"], b["steps"], strict=True), 1)
                          if left["selected_target_pose"] != right["selected_target_pose"])
    paid, left, right = first_dispatch
    if paid > kinematic[0]:
        raise ValueError("Candidate difference must share a realized action prefix")
    chosen = []
    for case, trace in zip(cases, (left, right), strict=True):
        call_id = trace["selection_call_id"]
        regional, = (row for row in case["regional"] if row["selection_call_id"] == call_id)
        main, = (row for row in case["global"] if row["call_id"] == call_id)
        ranking = regional["ranking"]["ranking"]
        target = trace["selected_target_pose"]
        if (regional["paid_before"] != paid - 1 or main["action_id"] != paid - 1
                or regional["actual_target_pose"] != target or main["outputs"]["selected"]["pose"] != target
                or ranking[0]["pose"] != target):
            raise ValueError("Actual changed dispatch lacks a matching ranked selection")
        chosen.append({"call_id": call_id, "target": target, "before_paid": paid-1,
                       "input_sha256": main["input_sha256"],
                       "rank": ranking, "global_poses": [row["pose"] for row in main["outputs"]["candidates"]]})
    old, full = chosen
    pool = sorted(tuple(row["pose"]) for row in old["rank"])
    if (pool != sorted(tuple(row["pose"]) for row in full["rank"])
            or old["global_poses"] != full["global_poses"]
            or old["input_sha256"] != full["input_sha256"]):
        raise ValueError("Recorded first candidate difference lacks a common pool or state")
    readable = []
    for item in chosen:
        scores = {tuple(row["pose"]): row["score_m2"] for row in item["rank"]}
        readable.append({k: item[k] for k in ("call_id", "target", "before_paid", "input_sha256")}
                        | {"score_for_original_target_m2": scores[tuple(old["target"])],
                           "score_for_full_target_m2": scores[tuple(full["target"])]})
    return {"move": {"paid": kinematic[0],
                     "action_by_arm": [row["action"] for row in kinematic[1:]],
                     "after_pose_by_arm": [row["position"] + [row["heading"]] for row in kinematic[1:]]},
            "dispatch_before_paid": paid-1, "candidate_pool": len(pool),
            "comparison_targets": readable}


def screen_camera(max_x, max_y, resolution):
    """Orthographic ray origin with fixed world axes and a shared scale."""
    az, elevation = np.deg2rad((-48., 64.))
    front = np.array([np.cos(elevation)*np.cos(az), np.cos(elevation)*np.sin(az), -np.sin(elevation)])
    right = np.cross(front, [0, 0, 1])
    right /= np.linalg.norm(right)
    up = np.cross(right, front)
    center = [max_x/2, max_y/2, 1.2]
    box = np.array([(x, y, z) for x in (0, max_x) for y in (0, max_y) for z in (0., 2.5)]) - center
    width = np.max(abs(box @ right)) + 1.0 * resolution
    height = np.max(abs(box @ up)) + 1.0 * resolution
    pixels_x = 1152
    pixels_y = round(pixels_x * height / width)
    x = (np.arange(pixels_x) + .5) * 2.0*width/pixels_x - width
    y = height - (np.arange(pixels_y) + .5) * 2.0*height/pixels_y
    origins = center - 30.*front + x[None, :, None]*right + y[:, None, None]*up
    rays = np.concatenate((origins, np.broadcast_to(front, origins.shape)), axis=2).astype(np.float32)
    return {"rays": o3d.core.Tensor(rays), "front": front, "bounds": [-width, width, -height, height],
            "view": {"azimuth_deg": -48, "elevation_deg": 64, "center_xyz_m": center,
                     "extent_uv_m": [-width, width, -height, height], "pixels": [pixels_x, pixels_y]}}


def mesh_image(path, camera):
    mesh = o3d.io.read_triangle_mesh(str(path))
    if mesh.is_empty():
        raise ValueError(f"Missing fused surface: {path}")
    colors = np.asarray(mesh.vertex_colors)
    triangles = np.asarray(mesh.triangles)
    scene = o3d.t.geometry.RaycastingScene(nthreads=1)
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    cast = scene.cast_rays(camera["rays"], nthreads=1)
    hits = np.isfinite(cast["t_hit"].numpy())
    if not hits.any():
        raise ValueError(f"No native PLY surfaces render inside camera bounds: {path}")
    uv = cast["primitive_uvs"].numpy()[hits]
    ids = cast["primitive_ids"].numpy()[hits]
    weights = np.c_[1.-uv.sum(axis=1), uv]
    normal = cast["primitive_normals"].numpy()[hits]
    sun = np.array([-.35, -.55, .75])
    flip = np.sign(-normal @ camera["front"])
    lambert = np.maximum((normal * flip[:, None]) @ sun, 0)
    if len(colors):
        surface = np.einsum("ij,ijk->ik", weights, colors[triangles[ids]])
        surface = .82*surface + np.array([.06, .10, .14])
    else:
        surface = np.broadcast_to([.46, .56, .61], (len(ids), 3))
    albedo_scale = .66 + .36*lambert
    diffuse = surface * albedo_scale[:, None]
    pixels = np.full((*hits.shape, 3), [247, 249, 251], dtype=np.uint8)
    pixels[hits] = np.round(np.clip(diffuse*255., 0, 255)).astype(np.uint8)
    return pixels


def grid_panel(axis, record, fork, arm_index, resolution, world_max):
    color = ARMS[arm_index][2]
    shape = record["belief"].shape
    axis.set_facecolor(GRID_COLORS[0])
    axis.imshow(record["belief"], cmap=ListedColormap(GRID_COLORS),
                norm=BoundaryNorm([-1.5, -.5, .5, 1.5], 3),
                origin="upper", extent=(0, shape[1]*resolution, 0, shape[0]*resolution),
                interpolation="none", zorder=0)
    trail = [record["origin"]["pose"]]
    trail.extend(row["position"] + [row["heading"]] for row in record["steps"])
    track = np.array([world_xy(pose, shape, resolution) for pose in trail])
    axis.plot(*track.T, color="white", lw=2.4, zorder=3)
    axis.plot(*track.T, color=color, lw=1.2, zorder=4)
    ax, ay = track[0]
    axis.scatter(ax, ay, s=22, marker="o", edgecolors=INK, facecolors="white", lw=.9, zorder=7)
    ex, ey = track[-1]
    axis.scatter(ex, ey, s=21, marker="s", edgecolors="white", facecolors=color, lw=.65, zorder=8)
    sx, sy = world_xy(fork["comparison_targets"][arm_index]["target"], shape, resolution)
    axis.scatter(sx, sy, s=28, marker="D", edgecolors=AMBER_DARK, facecolors="white", lw=1.05, zorder=8)
    pose = fork["move"]["after_pose_by_arm"][arm_index]
    x, y = world_xy(pose, shape, resolution)
    dx, dy = POSE_DIRECTIONS[pose[2]] * .57
    axis.annotate("", xy=(x+dx, y+dy), xytext=(x, y), zorder=10,
                  arrowprops=dict(arrowstyle="-|>", mutation_scale=9, lw=1.2, color=AMBER,
                                  shrinkA=0, shrinkB=0, alpha=.95))
    axis.set_xlim(0, world_max[0])
    axis.set_ylim(0, world_max[1])
    axis.set_aspect("equal")
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_color("#aab7c0")
        spine.set_linewidth(.55)


def physical_scale(axis, label, vertical_m=.4):
    xlim, ylim = axis.get_xlim(), axis.get_ylim()
    x = xlim[0] + (xlim[1]-xlim[0])*.045
    y = ylim[0] + (ylim[1]-ylim[0])*.035
    axis.add_patch(Rectangle((x-.16, y-.12), 2.32, vertical_m+.33,
                             facecolor="white", edgecolor="none", alpha=.90, zorder=15))
    axis.plot([x, x+2], [y, y], color=INK, lw=1.10, zorder=16)
    axis.plot([x, x], [y-.06, y+.06], color=INK, lw=.9, zorder=16)
    axis.plot([x+2, x+2], [y-.06, y+.06], color=INK, lw=.9, zorder=16)
    axis.text(x+1, y+vertical_m*.83, label, color=INK, weight="medium", fontsize=6.2,
              ha="center", va="bottom", zorder=17)


def legend_handles():
    return [Patch(facecolor=color, edgecolor=edge, linewidth=.5, label=label)
            for color, edge, label in zip(GRID_COLORS, ("#aab7c0", "#b2bdc4", "#2a3948"),
                                         ("Unknown", "Observed free", "Observed occupied"), strict=True)] + [
        Line2D([0], [0], color="#365a70", lw=1.3, label="Executed trajectory"),
        Line2D([0], [0], marker="o", ls="none", color=INK, mfc="white", ms=4.2, label="Start"),
        Line2D([0], [0], marker="s", ls="none", color="#365a70", mfc="#365a70", ms=4.2, label="Action 96"),
        Line2D([0], [0], marker="D", ls="none", color=AMBER_DARK, mfc="white", ms=4.2,
               label="First logged target"),
        Line2D([0], [0], marker=">", ls="-", color=AMBER, mfc=AMBER, ms=4.2, lw=1.15,
               label="First different executed action"),
    ]


def plot(study, output):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "text.color": INK,
                         "pdf.fonttype": 42, "svg.fonttype": "none", "savefig.facecolor": PAPER})
    measurements = study / "results/formal/checkpoints.csv"
    measurement_audit = study / "results/formal/measurement_audit.json"
    checked(measurements, read(measurement_audit)["results_sha256"][measurements.name])
    rows = []
    for layout, name in LAYOUTS:
        reference = study / "scenes/evaluator_references" / layout
        ref_receipt = read(reference / "RECEIPT.json")
        mask_path = reference / "evaluator_truth.npz"
        suffix = mask_path.relative_to(study).as_posix()
        truth_sha, = (digest for path, digest in ref_receipt["source_inputs_sha256"].items()
                      if path.endswith("/" + suffix))
        checked(mask_path, truth_sha)
        checked(reference / "fixed_reference.npz", ref_receipt["reference_npz_sha256"])
        with np.load(mask_path, allow_pickle=False) as data:
            reachable = data["reachable"]
        cases = [load_case(study, layout, arm, reachable) for arm, *_ in ARMS]
        initial = read(study / "scenes/prepared" / layout / "initial_state.json")
        resolution = float(initial["actual_config"]["resolution_m"])
        rows.append((layout, name, cases, ref_receipt, resolution, first_difference(cases)))
    if len({row[-2] for row in rows}) != 1:
        raise ValueError("Layouts must share their plotted metric scale")
    max_height, max_width = np.max([row[2][0]["belief"].shape for row in rows], axis=0)
    world_max = max_width * resolution, max_height * resolution
    camera = screen_camera(*world_max, resolution)
    fig, axes = plt.subplots(2, 4, figsize=(11.6, 7.0))
    fig.subplots_adjust(left=.041, right=.981, bottom=.185, top=.826, wspace=.13, hspace=.64)

    for ri, (_, name, cases, _, resolution, fork) in enumerate(rows):
        for arm_index, case in enumerate(cases):
            grid, surface = axes[ri, arm_index*2:arm_index*2+2]
            grid_panel(grid, case, fork, arm_index, resolution, world_max)
            if arm_index == 0:
                physical_scale(grid, "2 m")
            surface.imshow(mesh_image(case["mesh"], camera), extent=camera["bounds"],
                           origin="upper", interpolation="none")
            surface.set_axis_off()
        fig.canvas.draw()
        y = max(axis.get_position().y1 for axis in axes[ri])
        x = axes[ri, 0].get_position().x0
        delta = 100. * (cases[1]["metrics"][2] - cases[0]["metrics"][2])
        fig.text(x, y+.062, f"{'AB'[ri]}  {name}", fontsize=13, weight="bold", va="bottom")
        fig.text(x+.208, y+.064, f"ΔJ {delta:+.2f} pp", color=TEAL if delta > 0 else AMBER_DARK,
                 fontsize=9.5, weight="bold", va="bottom")
        native = fork["move"]
        moves = " / ".join({"forward": "forward", "left": "turn left", "right": "turn right"}[key]
                           for key in native["action_by_arm"])
        first = fork["dispatch_before_paid"]
        fig.text(.98, y+.064, f"Different targets before action {first+1}  •  First different action {native['paid']}  ({moves})",
                 fontsize=8.1, color=MUTE, ha="right", va="bottom")
        for arm_index, case in enumerate(cases):
            grid, surface = axes[ri, arm_index*2:arm_index*2+2]
            a, b = grid.get_position(), surface.get_position()
            fig.text((a.x0+a.x1)/2, y+.017, "Observed grid + track", ha="center", color=MUTE, fontsize=8)
            fig.text((b.x0+b.x1)/2, y+.017, "Fused surface", ha="center", color=MUTE, fontsize=8)
            c, f1, j = 100.*np.array(case["metrics"])
            fig.text((a.x0+b.x1)/2, min(a.y0,b.y0)-.025, f"C {c:.2f}%    F1 {f1:.2f}%    J {j:.2f}%",
                     ha="center", fontsize=9.0, va="top", color=ARMS[arm_index][2], weight="semibold")

    for arm_index, (_, title, color) in enumerate(ARMS):
        sides = axes[0, arm_index*2:arm_index*2+2]
        left, right = sides[0].get_position().x0, sides[-1].get_position().x1
        fig.text((left+right)/2, .965, title, fontsize=13.3, color=color, weight="bold",
                 ha="center", va="center")
        fig.add_artist(Line2D([left, right], [.942, .942], transform=fig.transFigure, color=color, lw=1.0))
    fig.legend(handles=legend_handles(), loc="lower center", bbox_to_anchor=(.51, .044),
               ncol=4, fontsize=7.7, frameon=False, handlelength=1.44, columnspacing=2.1, labelspacing=.9)
    fig.text(.51, .024, "Simulation  •  Action 96  •  Shared metric scale and orthographic surface view  •  F1 tolerance 5 cm",
             color=MUTE, ha="center", va="center", fontsize=7.8)
    output.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png", "svg"):
        fig.savefig(output / f"checkpoints_a96.{ext}", dpi=420 if ext == "png" else 300)
    plt.close(fig)

    receipt = {"action": PAID, "measurements_root": str(study), "shared_world_extent_xy_m": world_max,
               "measurement_csv_sha256": sha(measurements), "measurement_audit_sha256": sha(measurement_audit),
               "shared_camera": camera["view"], "uses_native_fused_mesh": True,
               "rendering_does_not_filter_native_mesh": True, "layouts": []}
    for layout, name, cases, ref, resolution, fork in rows:
        receipt["layouts"].append({"layout": layout, "name": name, "grid_resolution_m": resolution,
                                   "fixed_reference_sha256": ref["reference_npz_sha256"],
                                   "fixed_visible_samples": ref["metadata"]["all"]["visible_samples"],
                                   "fixed_visible_area_m2": ref["metadata"]["all"]["estimated_visible_area_m2"],
                                   "reference_camera_pose_count": ref["camera_pose_count"],
                                   "fork": fork,
                                   "measurements": {arm[0]: dict(zip(("C", "F1_05cm", "J"), case["metrics"], strict=True))
                                                    for arm, case in zip(ARMS, cases, strict=True)},
                                   "source_sha256": {key: value for case in cases for key, value in case["sources"].items()}})
    receipt["figures_sha256"] = {f"checkpoints_a96.{ext}": sha(output / f"checkpoints_a96.{ext}")
                                 for ext in ("pdf", "png", "svg")}
    (output / "checkpoints_a96_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--study", type=Path, default=DEFAULT_STUDY)
    cli.add_argument("--output", type=Path, default=HERE)
    args = cli.parse_args()
    plot(args.study.resolve(), args.output.resolve())
