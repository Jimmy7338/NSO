#!/usr/bin/env python3
"""Explain T0's sealed paired prefix; never read candidate outcomes or simulate.

Top rows are archived step-20 RGB/depth. Bottom row is an explicitly post-hoc
GT exterior rendering in a fixed rear-side view, not a robot observation.
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FAMILIES = ("storage_shelves", "ventilation_baffles")
TITLES = ("Storage shelves | asset class 2", "Ventilation baffles | asset class 3")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_verified_npz(run, relative, manifest):
    path = run / relative
    if sha(path) != manifest[relative]:
        raise ValueError(f"input differs from its frozen manifest: {relative}")
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def local_object_triangles(reference, context):
    """Predeclared unit bounds, not an outcome-selected view or GT quality mask."""
    rotations = (((1, 0), (0, 1)), ((0, -1), (1, 0)),
                 ((-1, 0), (0, -1)), ((0, 1), (-1, 0)))
    rotation = np.asarray(rotations[context["rotation_quarter_turns"]])
    anchor = (np.array([4.1, 4.3]) + context["offset_xy_m"] - 4.2) @ rotation.T + 4.2
    vertices = reference["vertices"].copy()
    vertices[:, :2] = (vertices[:, :2] - anchor) @ rotation
    triangles = vertices[reference["triangles"]]
    center = triangles.mean(axis=1)
    side = 1.6 * context["visible_scale"]
    keep = (np.abs(center[:, 0]) <= side / 2 + 1e-8)
    keep &= (center[:, 1] >= -1e-8) & (center[:, 1] <= side + 1e-8)
    keep &= (center[:, 2] > 1e-8) & (center[:, 2] <= side + 1e-8)
    if not keep.any():
        raise ValueError("fixed object crop contains no GT exterior faces")
    return triangles[keep], side


def plot(run, prefix):
    seal = json.loads((run / "pre_outcome_seal.json").read_text())
    manifest = json.loads((run / "artifact_hashes.json").read_text())
    frames, references, fixtures, input_hashes = [], [], [], {}
    for family in FAMILIES:
        frame_name = f"{family}/prefix/frames/0020.npz"
        fixture_name = f"{family}/fixture.json"
        reference_name = f"{family}/reference.npz"
        # The already sealed observations are checked before reading GT.
        frame = read_verified_npz(run, frame_name, seal)
        if sha(run / fixture_name) != seal[fixture_name]:
            raise ValueError("fixture changed after input sealing")
        fixture = json.loads((run / fixture_name).read_text())
        if fixture["context"]["context_id"] != "T0":
            raise ValueError("this fixed illustration opens T0 only")
        reference = read_verified_npz(run, reference_name, manifest)
        frames.append(frame); fixtures.append(fixture); references.append(reference)
        for name in (frame_name, fixture_name, reference_name):
            input_hashes[name] = sha(run / name)
    a, b = frames
    for key in ("depth_m", "world_from_camera", "intrinsic", "timestamp_s"):
        np.testing.assert_array_equal(a[key], b[key])
    marker = a["semantic"] > 0
    np.testing.assert_array_equal(marker, b["semantic"] > 0)
    np.testing.assert_array_equal(a["color_rgb"][~marker], b["color_rgb"][~marker])
    difference = float(np.max(np.abs(a["depth_m"] - b["depth_m"])))
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.titlesize": 11, "pdf.fonttype": 42})
    figure = plt.figure(figsize=(10.0, 11.6), facecolor="white")
    grid = figure.add_gridspec(3, 2, left=.08, right=.92, bottom=.145, top=.90,
                              height_ratios=[1, 1, 1.20], hspace=.36, wspace=.12)
    figure.suptitle("Same observed geometry, different hidden asset structures", fontsize=16, y=.972)
    figure.text(.5, .941, "Predeclared training parent T0 • fixed paid prefix: 20 actions • no route outcomes shown",
                ha="center", color="#4c5560", fontsize=10)
    depth_axes = []
    triangle_counts = []
    for column, (family, title, frame, reference, fixture) in enumerate(zip(FAMILIES, TITLES, frames, references, fixtures)):
        rgb_ax = figure.add_subplot(grid[0, column])
        rgb_ax.imshow(frame["color_rgb"], interpolation="nearest")
        rgb_ax.set_title(title, pad=12, fontweight="bold")
        rgb_ax.set_xlabel("Observed RGB: visible artificial identifier")
        rgb_ax.set_xticks([]); rgb_ax.set_yticks([])
        depth_ax = figure.add_subplot(grid[1, column])
        depth = np.ma.masked_less_equal(frame["depth_m"], 0)
        color_map = plt.colormaps["viridis"].copy(); color_map.set_bad("#151a20")
        depth_image = depth_ax.imshow(depth, cmap=color_map, vmin=0, vmax=fixture["config"]["max_depth_m"],
                                     interpolation="nearest")
        depth_ax.set_title("Observed axial depth", pad=8)
        depth_ax.set_xlabel("Paired depth difference: exactly 0 m" if column == 0 else "Same pose, intrinsics and valid-pixel mask")
        depth_ax.set_xticks([]); depth_ax.set_yticks([])
        depth_axes.append(depth_ax)
        object_ax = figure.add_subplot(grid[2, column], projection="3d")
        triangles, side = local_object_triangles(reference, fixture["context"])
        triangle_counts.append(len(triangles))
        normal = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
        lighting = .5 + .5 * np.abs(normal @ (np.array([.4, -.5, .75]) / np.linalg.norm([.4, -.5, .75])))
        colors = np.column_stack([.35 + .37 * lighting, .40 + .35 * lighting, .46 + .32 * lighting,
                                  np.full(len(triangles), .88)])
        collection = Poly3DCollection(triangles, facecolors=colors, edgecolors=(.12, .17, .22, .20), linewidths=.30)
        object_ax.add_collection3d(collection)
        object_ax.set_xlim(-side / 2, side / 2); object_ax.set_ylim(0, side); object_ax.set_zlim(0, side)
        object_ax.set_box_aspect([1, 1, 1]); object_ax.view_init(elev=25, azim=35)
        object_ax.set_xlabel("Local x (m)", labelpad=2); object_ax.set_ylabel("Depth y (m)", labelpad=2)
        object_ax.set_zlabel("Height (m)", labelpad=2)
        object_ax.set_xticks([-.8, 0, .8]); object_ax.set_yticks([0, .8, 1.6]); object_ax.set_zticks([0, .8, 1.6])
        object_ax.tick_params(labelsize=8, pad=0)
        object_ax.set_title("GT only: horizontal shelves + rear board" if column == 0
                            else "GT only: vertical baffles + open rear", pad=6)
        for axis in (object_ax.xaxis, object_ax.yaxis, object_ax.zaxis):
            axis.pane.set_facecolor((.98, .98, .98, 1))
    color_axis = figure.add_axes([.939, .419, .012, .172])
    colorbar = figure.colorbar(depth_image, cax=color_axis)
    colorbar.set_label("Depth (m)", fontsize=9); colorbar.ax.tick_params(labelsize=8)
    figure.text(.5, .079, "GT is used only for this explanatory crop; it is never a planner input.",
                ha="center", fontsize=10, fontweight="bold")
    figure.text(.5, .039, "The GT panels share a fixed rear-side translucent rendering, not a robot observation.\n"
                "Synthetic asset-category / hidden-geometry association: a controlled positive test, not natural-semantic validation.",
                ha="center", fontsize=9, color="#4c5560", linespacing=1.5)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(prefix.with_suffix(".png"), dpi=220)
    figure.savefig(prefix.with_suffix(".pdf"))
    plt.close(figure)
    provenance = {"run": str(run.resolve()), "script_sha256": sha(Path(__file__)),
        "input_hashes": input_hashes, "step": 20, "context": "T0", "new_trajectories": 0,
        "candidate_outcome_files_read": [], "max_paired_depth_difference_m": difference,
        "marker_pixels_per_family": int(marker.sum()), "same_nonmarker_rgb": True,
        "gt_crop_rule": "all exterior triangles with centroid inside the predeclared T0 device bounds and z>0; no outcome-selected geometry",
        "gt_render": {"elevation_deg": 25, "azimuth_deg": 35, "face_alpha": .88,
                      "selected_triangle_counts": dict(zip(FAMILIES, triangle_counts)), "shared_view": True},
        "limits": "GT is post-hoc visualization only; synthetic artificial-asset association; no route utility or semantic advantage claim",
        "outputs": {prefix.with_suffix(ext).name: sha(prefix.with_suffix(ext)) for ext in (".png", ".pdf")}}
    prefix.with_suffix(".json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({"png": str(prefix.with_suffix('.png')), "pdf": str(prefix.with_suffix('.pdf'))}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "eval_results/response_v7_training_20260911/T0")
    parser.add_argument("--output-prefix", type=Path, default=ROOT / "docs/research/figures/response_v7_paired_prefix")
    args = parser.parse_args()
    plot(args.run, args.output_prefix)
