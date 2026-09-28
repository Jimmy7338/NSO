#!/usr/bin/env python3
"""Plot fixed P0 initial frames and its complete sealed paid prefix.

No world is constructed, no future frame or candidate outcome is opened, and no
view is selected by utility. The initial frame is a fixed chronological choice.
The common measured map comes from the archived 150-action prefix. Background
coverage statistics are post-hoc evaluation, never map or planner inputs.
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Rectangle, Patch
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ARRANGEMENTS = ("shelf_west", "shelf_east")
FRAME_STEP = 0


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plot(run, output):
    manifest_path = run / "artifact_hashes.json"
    manifest = json.loads(manifest_path.read_text())
    seal_path = run / "pre_evaluator_choices_seal.json"
    seal = json.loads(seal_path.read_text())
    inputs = {"artifact_hashes.json": sha(manifest_path),
              "pre_evaluator_choices_seal.json": sha(seal_path)}

    def read(relative, presealed=True):
        path = run / relative
        identity = sha(path)
        if manifest[relative] != identity or (presealed and seal[relative] != identity):
            raise ValueError(f"sealed input identity mismatch: {relative}")
        inputs[relative] = identity
        if path.suffix == ".npz":
            with np.load(path, allow_pickle=False) as archive:
                return {key: archive[key].copy() for key in archive.files}
        return json.loads(path.read_text())

    frames, maps, records, fixtures = [], [], [], []
    for arrangement in ARRANGEMENTS:
        base = f"P0/{arrangement}"
        frames.append(read(f"{base}/prefix/frames/{FRAME_STEP:04d}.npz"))
        maps.append(read(f"{base}/prefix_map.npz"))
        records.append(read(f"{base}/prefix/records.json"))
        fixtures.append(read(f"{base}/fixture.json"))
    summary = read("structure_summary.json", presealed=False)
    if summary["new_candidate_branches"] != 0:
        raise ValueError("this figure's source must remain the prefix-only V8 preparation")
    for a, b in zip(records[0], records[1]):
        if a != b:
            raise ValueError("paired paid action records differ")
    if len(records[0]) != 151 or records[0][-1]["step"] != 150:
        raise ValueError("the fixed full prefix must contain 150 paid actions")
    if any(row["collision"] for rows in records for row in rows):
        raise ValueError("prefix collision cannot be omitted from an explanatory figure")
    for key in ("depth_m", "world_from_camera", "intrinsic", "timestamp_s"):
        np.testing.assert_array_equal(frames[0][key], frames[1][key])
    for key in maps[0]:
        np.testing.assert_array_equal(maps[0][key], maps[1][key])
    mask = frames[0]["semantic"] > 0
    np.testing.assert_array_equal(mask, frames[1]["semantic"] > 0)
    np.testing.assert_array_equal(frames[0]["color_rgb"][~mask], frames[1]["color_rgb"][~mask])
    labels = frames[0]["semantic"]
    np.testing.assert_array_equal(frames[1]["semantic"], np.where(labels == 2, 3, np.where(labels == 3, 2, 0)))
    if any(not np.any(frame["semantic"] == category) for frame in frames for category in (2, 3)):
        raise ValueError("the fixed first frame must show both physical markers")
    p0_stats = [next(r for r in summary["histories"] if r["context"] == "P0" and r["arrangement"] == arrangement)
                for arrangement in ARRANGEMENTS]
    bg = [r["observable_background_fraction_actually_seen"] for r in p0_stats]
    coverage = [r["coverage_2d"] for r in p0_stats]
    depth_difference = float(np.max(np.abs(frames[0]["depth_m"] - frames[1]["depth_m"])))

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.titlesize": 10, "pdf.fonttype": 42})
    figure = plt.figure(figsize=(8.8, 10.0), facecolor="white")
    grid = figure.add_gridspec(3, 2, left=.055, right=.925, bottom=.17, top=.89,
                              height_ratios=[1., 1., 1.12], hspace=.32, wspace=.16)
    figure.suptitle("Same geometric prefix, swapped asset markers", fontsize=15, y=.974)
    figure.text(.5, .939, "Parent P0 | RGB-D shown: fixed first frame (step 0) | no future route executed",
                ha="center", fontsize=9, color="#4c5560")
    titles = ("A: shelf west / closed cabinet east", "B: closed cabinet west / shelf east")
    colors = {2: "#2864dc", 3: "#dc3c28"}
    depth_axes = []
    for column, (frame, title) in enumerate(zip(frames, titles)):
        axis = figure.add_subplot(grid[0, column])
        axis.imshow(frame["color_rgb"], interpolation="nearest")
        axis.set_title(title, fontweight="bold", pad=9)
        axis.set_xlabel("Actual RGB; boxes mark observed identifier pixels")
        axis.set_xticks([]); axis.set_yticks([])
        for category in (2, 3):
            yy, xx = np.where(frame["semantic"] == category)
            axis.add_patch(Rectangle((xx.min() - 1.5, yy.min() - 1.5),
                                    xx.max() - xx.min() + 3, yy.max() - yy.min() + 3,
                                    fill=False, edgecolor=colors[category], linewidth=1.0))
        depth_axis = figure.add_subplot(grid[1, column])
        cmap = plt.colormaps["viridis"].copy(); cmap.set_bad("#171d25")
        depth_image = depth_axis.imshow(np.ma.masked_less_equal(frame["depth_m"], 0),
                                       cmap=cmap, vmin=0, vmax=4, interpolation="nearest")
        depth_axis.set_title("Observed axial depth", pad=6)
        depth_axis.set_xlabel("Paired maximum depth difference = 0 m")
        depth_axis.set_xticks([]); depth_axis.set_yticks([])
        depth_axes.append(depth_axis)
    colorbar = figure.colorbar(depth_image, ax=depth_axes, fraction=.025, pad=.035)
    colorbar.set_label("Depth (m)", fontsize=8); colorbar.ax.tick_params(labelsize=8)

    map_axis = figure.add_subplot(grid[2, 0])
    cfg = fixtures[0]["config"]
    cmap = ListedColormap(["#b9c0c8", "#f7f8fa", "#34404f"])
    map_axis.imshow(maps[0]["belief"], cmap=cmap, norm=BoundaryNorm([-1.5, -.5, .5, 1.5], 3),
                    interpolation="nearest", extent=[0, cfg["width_m"], 0, cfg["height_m"]], origin="upper")
    xy = np.array([[(r["position"][1] + .5) * cfg["resolution_m"],
                    cfg["height_m"] - (r["position"][0] + .5) * cfg["resolution_m"]] for r in records[0]])
    map_axis.plot(xy[:, 0], xy[:, 1], color="#148580", linewidth=1.8, alpha=.9)
    map_axis.scatter(*xy[0], marker="o", s=34, color="#148580", edgecolor="white", linewidth=.7, zorder=4)
    map_axis.scatter(*xy[-1], marker="*", s=85, color="#8357bd", edgecolor="white", linewidth=.6, zorder=4)
    map_axis.annotate("step 0", xy[0], xytext=(5, 6), textcoords="offset points", fontsize=8)
    map_axis.annotate("step 150", xy[-1], xytext=(5, 6), textcoords="offset points", fontsize=8)
    map_axis.set_title("Measured map + actual 150-action path", pad=8)
    map_axis.set_xlabel("World x (m)"); map_axis.set_ylabel("World y (m)")
    map_axis.set_xticks([0, 2, 4, 6, 8]); map_axis.set_yticks([0, 2, 4, 6])
    figure.legend(handles=[Patch(facecolor="#34404f", label="Observed occupied"),
                           Patch(facecolor="#f7f8fa", edgecolor="#b9c0c8", label="Observed free"),
                           Patch(facecolor="#b9c0c8", label="Unknown")],
                  loc="lower center", bbox_to_anchor=(.5, .103), ncol=3, frameon=False, fontsize=8)
    notes = figure.add_subplot(grid[2, 1]); notes.axis("off")
    notes.text(0., .97, "Complete prefix: same measured map", fontsize=10, fontweight="bold", va="top")
    notes.text(0., .83,
        "150 paid actions = 120 moves + 30 turns\n"
        f"Background seen (A / B): {bg[0]*100:.2f}% / {bg[1]*100:.2f}%\n"
        f"2D coverage (A / B): {coverage[0]:.2f} / {coverage[1]:.2f}\n"
        "New candidate routes executed: 0",
        fontsize=9, va="top", linespacing=1.8)
    notes.text(0., .37, "Late-stage refinement positive control", fontsize=10, fontweight="bold", va="top")
    notes.text(0., .24,
        "Artificial asset-category / hidden-structure prior.\n"
        "No full-system coverage advantage is established.\n"
        "Original V8 candidate-role gate failed;\n"
        "the whole version is retained as a failed preparation.",
        fontsize=8.5, va="top", color="#4c5560", linespacing=1.55)
    figure.text(.5, .063, "Blue marker: storage shelf     |     Red marker: closed equipment cabinet",
                ha="center", fontsize=9)
    figure.text(.5, .027,
        "RGB-D and map are archived measurements. Background/coverage values are post-hoc GT evaluation.\n"
        "No GT geometry, hidden structure, candidate reward or future sensor frame is shown.",
        ha="center", fontsize=8.5, color="#4c5560", linespacing=1.5)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output.with_suffix(".png"), dpi=165)
    figure.savefig(output.with_suffix(".pdf"))
    plt.close(figure)
    outputs = {output.with_suffix(suffix).name: dict(sha256=sha(output.with_suffix(suffix)),
               bytes=output.with_suffix(suffix).stat().st_size) for suffix in (".png", ".pdf")}
    if sum(value["bytes"] for value in outputs.values()) >= 1_000_000:
        raise ValueError("combined image/PDF size exceeds the fixed one-MB budget")
    provenance = dict(run=str(run.resolve()), context="P0", frame_step=FRAME_STEP,
        selection_rule="fixed earliest archived prefix frame; no candidate outcome or utility-based view selection",
        map_step=150, prefix_paid_actions=150, new_worlds_constructed=0,
        new_candidate_routes_executed=0, candidate_outcome_files_read=[],
        gt_geometry_files_read=[], posthoc_gt_statistics_source="structure_summary.json.histories (P0 only)",
        paired_nonmarker_rgb_exact=True, paired_map_exact=True,
        paired_max_depth_difference_m=depth_difference,
        marker_pixels_per_class_per_frame=[{str(k): int((f['semantic'] == k).sum()) for k in (2, 3)} for f in frames],
        original_v8_structural_gate_passed=summary["passed"],
        input_hashes=inputs, script_sha256=sha(Path(__file__)), outputs=outputs,
        limits="late-stage refinement in an artificial asset-marker positive control; not full coverage or semantic efficacy evidence")
    output.with_suffix(".json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(outputs, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "eval_results/competition_v8_preparation_20260911")
    parser.add_argument("--output-prefix", type=Path, default=ROOT / "docs/research/figures/competition_v8_paired_prefix")
    args = parser.parse_args()
    plot(args.run, args.output_prefix)
