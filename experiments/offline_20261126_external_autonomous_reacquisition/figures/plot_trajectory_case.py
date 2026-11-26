"""Draw one documented experimental pair from measured action and coverage records."""

from argparse import ArgumentParser
import csv
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from evaluate import digest, read, require, write_json


COLOR = {"A": "#52799f", "B": "#bd4a25", "route": "#ae6a17", "view": "#bf7f12", "cross": "#00806d"}
ARMS = ("A", "B")


def as_table(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def choose_case(records, evaluations):
    order = []
    finished = {r["layout"] for r in records if r["arm"] == "A" and r["status"] == "completed_execution"}
    for log in evaluations["runs"]:
        layout = log["layout"]
        if log["arm"] != "B" or layout not in finished:
            continue
        rank = min((0 if row["crossing_paid"] is not None else 1 if row["arrival_paid"] is not None else 2
                    for row in log["attempts"]), default=3)
        order.append((rank, layout, log["attempts"]))
    require(order, "No A/B pair has two verified native trajectories")
    rank, layout, attempts = min(order)
    if attempts:
        attempt = min(attempts, key=lambda row: (
            row["crossing_paid"] is None, row["arrival_paid"] is None, row["dispatch_paid"],
        ))
    else:
        attempt = None
    return rank, layout, attempt


def world_xy(positions, shape, resolution):
    rows, cols = np.asarray(positions).T[:2]
    return ((cols+.5)*resolution, (shape[0]-rows-.5)*resolution)


def draw_gain_panel(ax, gain, points, shape, resolution, first, final):
    """Draw the measured set difference on one common fixed mask."""
    gain_count = int(np.count_nonzero(gain))
    extent = (0, shape[1]*resolution, 0, shape[0]*resolution)
    ax.imshow(np.flipud(np.ma.masked_where(~gain, gain)), origin="lower", interpolation="none", extent=extent,
              cmap=ListedColormap(("#5eb7aa",)), vmin=0, vmax=1, alpha=.65)
    xs, ys = world_xy(points, shape, resolution)
    ax.plot(xs, ys, color="#425772", lw=1.6, alpha=.9, zorder=3)
    ax.scatter(xs[0], ys[0], marker="D", color=COLOR["cross"], edgecolor="white", s=31, linewidth=.6, zorder=5)
    ax.scatter(xs[-1], ys[-1], marker="s", color=COLOR["view"], edgecolor="white", s=31, linewidth=.6, zorder=5)
    gain_xy = np.column_stack(world_xy(np.argwhere(gain), shape, resolution))
    gain_xy = np.vstack((gain_xy, np.column_stack((xs, ys))))
    gain_low, gain_high = gain_xy.min(axis=0), gain_xy.max(axis=0)
    ax.set(xlim=(gain_low[0]-.4, gain_high[0]+.4), ylim=(gain_low[1]-.38, gain_high[1]+.38),
           xlabel="x (m)", ylabel="y (m)", aspect="equal")
    ax.text(.98, .965 if gain_count else .77, f"{gain_count} correct free cells added",
            transform=ax.transAxes, ha="right", va="top", fontsize=8.2, color="#22695f",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": .75, "pad": 1})
    ax.set_title(f"(b) Newly covered area  p{first}–p{final}", loc="left",
                 fontsize=10.7, fontweight="bold")
    return gain_count


def render():
    result = ROOT / "results/formal"
    summary = read(result / "summary.json")
    audit_path = result / "AUTONOMOUS_EVIDENCE_AUDIT.json"
    episodes = as_table(result / "episodes.csv")
    for item in ("episodes.csv", "coverage_curve.csv", audit_path.name):
        require(digest(result / item) == summary["result_sha256"][item], f"Measured input differs {item}")
    evidence = read(audit_path)
    rank, layout, attempt = choose_case(episodes, evidence)
    freeze = read(ROOT / "FORMAL_FREEZE.json")
    source = Path(freeze["scene_root"]) / "prepared" / layout / "initial_state.json"
    cold = read(source)
    trace, prefixes, initial, raw_hashes = {}, {}, {}, {}
    possible_gain = attempt is not None and attempt["crossing_paid"] is not None
    gain_summary = None
    published = {row["arm"]: row for row in read(ROOT / "execution/formal/phase_receipt.json")["episodes"] if row["layout"] == layout}
    covers = as_table(result / "coverage_curve.csv")
    for arm in ARMS:
        run = Path(published[arm]["run_path"])
        manifest = read(run / "manifest.json")
        require(digest(run / "manifest.json") == published[arm]["result_sha256"], "Case manifest does not match native receipt")
        require(digest(run / "steps.json") == manifest["files_sha256"]["steps.json"], "Case trajectory checksum differs")
        require(digest(run / "observed_belief_trace.npz") == manifest["files_sha256"]["observed_belief_trace.npz"],
                "Case belief checksum differs")
        trace[arm] = read(run / "steps.json")
        prefixes[arm] = np.asarray([cold["anchor"], *[[*s["position"], s["heading"]] for s in trace[arm]]])
        with np.load(run / "observed_belief_trace.npz", allow_pickle=False) as archived:
            initial[arm] = archived["belief"][0]
            if arm == "B" and possible_gain:
                crossing = attempt["crossing_paid"]
                relevant = next(row["attempts"] for row in evidence["runs"] if row["layout"] == layout and row["arm"] == arm)
                later = min((row["arrival_paid"] for row in relevant
                             if row["arrival_paid"] is not None and row["arrival_paid"] > crossing),
                            default=len(trace[arm]))
                observed_belief = archived["belief"][[crossing, later]]
        raw_hashes[arm] = {"manifest_sha256": published[arm]["result_sha256"],
                           "steps_sha256": manifest["files_sha256"]["steps.json"]}
    require(np.array_equal(initial["A"], initial["B"]), "The paired initial observation maps differ")

    resolution, shape = cold["grid_resolution_m"], cold["shape"]
    if possible_gain:
        fig = plt.figure(figsize=(10.6, 5.4))
        grid = fig.add_gridspec(2, 2, width_ratios=(1.02, 1.15), height_ratios=(.84, 1.16))
        path_ax, gain_ax, cov_ax = fig.add_subplot(grid[:, 0]), fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, 1])
        mask_file = Path(freeze["scene_root"]) / "evaluator_references" / layout / "evaluator_truth.npz"
        expected = next(row["reach_mask_file_sha256"] for row in episodes if row["layout"] == layout and row["arm"] == "B")
        require(digest(mask_file) == expected, "Formal map overlay mask does not match measured coverage")
        with np.load(mask_file, allow_pickle=False) as measurement:
            mask = measurement["reachable"]
        gain = (observed_belief[1] == 0) & (observed_belief[0] != 0) & mask
        gained = draw_gain_panel(gain_ax, gain, prefixes["B"][crossing:later+1], shape, resolution, crossing, later)
        gain_summary = {"from_paid": crossing, "to_paid": later, "new_correct_free_cells": gained,
                        "mask_sha256": expected}
        axes = (path_ax, gain_ax, cov_ax)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.4), gridspec_kw={"width_ratios": (1.05, 1.0)})
        path_ax, cov_ax = axes
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=9)
    path_ax.imshow(np.flipud(initial["A"]), origin="lower", interpolation="none",
                   extent=(0, shape[1]*resolution, 0, shape[0]*resolution),
                   cmap=ListedColormap(("#eeeff1", "#ffffff", "#353f47")),
                   norm=BoundaryNorm((-1.5, -.5, .5, 1.5), 3), zorder=0)
    labels = {"A": "A · combined policy", "B": "B · combined + reacquisition"}
    for arm in ARMS:
        xy = world_xy(prefixes[arm], shape, resolution)
        path_ax.plot(*xy, color=COLOR[arm], lw=1.6, alpha=.85 if arm == "A" else .96)
        path_ax.scatter(xy[0][-1], xy[1][-1], marker="X" if trace[arm] and trace[arm][-1]["collision"] else "s",
                        s=36, facecolor=COLOR[arm], edgecolor="white", linewidth=.55, zorder=5)
        curve = [x for x in covers if (x["layout"], x["arm"]) == (layout, arm)]
        require([int(x["paid"]) for x in curve] == list(range(len(trace[arm])+1)), "Case curve has missing clocks")
        clocks = [int(x["paid"]) for x in curve]
        percentages = [100*float(x["C_free"]) for x in curve]
        cov_ax.step(clocks, percentages, where="post", color=COLOR[arm], lw=1.75, label=labels[arm])
        cov_ax.scatter(clocks[-1], percentages[-1], color=COLOR[arm], s=22, marker="X" if trace[arm] and trace[arm][-1]["collision"] else "s")
    home_x, home_y = world_xy(np.asarray([cold["anchor"]]), shape, resolution)
    path_ax.scatter(home_x, home_y, color="#27323c", marker="o", s=38, edgecolor="white", linewidth=.55, zorder=6)
    if attempt:
        start, arrival, crossed, stop = (attempt[k] for k in ("dispatch_paid", "arrival_paid", "crossing_paid", "last_paid"))
        native_b = prefixes["B"][start: stop+1]
        path_ax.plot(*world_xy(native_b, shape, resolution), color=COLOR["route"], lw=2.8, alpha=.72,
                     linestyle=(0, (2.4, 2)), zorder=4)
        cov_ax.axvspan(start, stop, color="#ae6a17", alpha=.09, zorder=0)
        annotations = ((arrival, "View acquired", "view", "^", (5, 7)), (crossed, "Edge crossed", "cross", "D", (5, -15)))
        for tick, label, key, marker, offset in annotations:
            if tick is None:
                continue
            x, y = world_xy(prefixes["B"][tick:tick+1], shape, resolution)
            path_ax.scatter(x, y, marker=marker, color=COLOR[key], edgecolor="white", linewidth=.65, s=58, zorder=7)
            path_ax.annotate(label, (x[0],y[0]), xytext=offset, textcoords="offset points",
                             fontsize=8.5, color=COLOR[key], zorder=8)
            cov_ax.axvline(tick, color=COLOR[key], linestyle=":", lw=1.15, alpha=.78)
            if key == "cross":
                ox, oy = world_xy(prefixes["B"][tick-1:tick], shape, resolution)
                path_ax.scatter(ox, oy, marker="o", facecolor="none", edgecolor="#383137", linewidth=1.2, s=48, zorder=8)
                path_ax.annotate("", (x[0], y[0]), xytext=(ox[0], oy[0]), zorder=9, arrowprops={
                    "arrowstyle": "-|>", "color": COLOR["cross"], "linewidth": 2.1, "shrinkA": 1.5, "shrinkB": 1.5,
                })
    all_xy = np.vstack([np.column_stack(world_xy(prefixes[arm], shape, resolution)) for arm in ARMS])
    lower, upper = all_xy.min(axis=0), all_xy.max(axis=0)
    margin = np.maximum(.5, (upper-lower)*.08)
    path_ax.set(xlim=(lower[0]-margin[0], upper[0]+margin[0]), ylim=(lower[1]-margin[1], upper[1]+margin[1]),
                xlabel="x (m)", ylabel="y (m)", aspect="equal")
    if possible_gain:
        gain_ax.set_anchor("NW")
        path_ax.set_anchor("NW")
    path_ax.set_title("(a) Recorded trajectories", loc="left", fontsize=11, fontweight="bold")
    cov_ax.set(xlabel="Paid action", ylabel="Correct free coverage (%)", xlim=(-3, 195))
    cov_ax.grid(axis="y", linewidth=.7, alpha=.22)
    cov_ax.set_title(f"({'c' if possible_gain else 'b'}) Coverage along execution", loc="left",
                     fontsize=11, fontweight="bold")
    cov_ax.legend(loc="lower right", frameon=False, fontsize=9)
    legend_note = "Grey = unobserved at start. Filled circle = anchor. Square = endpoint. × = collision."
    if possible_gain:
        legend_note += " Open circle = blocked origin."
    fig.text(.08, .035, legend_note, fontsize=8.4)
    fig.tight_layout(rect=(0, .065, 1, .985))
    for extension in ("pdf", "png"):
        fig.savefig(HERE / f"reacquisition_case.{extension}", dpi=240, bbox_inches="tight")
    plt.close(fig)
    selection = ("first layout with audited crossing", "first layout with acquired viewpoint",
                 "first layout with a dispatched inspection", "first complete pair without dispatch")[rank]
    write_json(HERE / "reacquisition_case_source.json", {
        "layout": layout, "selection_rule": selection, "inspection_id": attempt["inspection_id"] if attempt else None,
        "scope": "One illustrative simulation pair. The stationary background is the initial measured occupancy.",
        "newly_observed_area": gain_summary,
        "coverage_curve_sha256": summary["result_sha256"]["coverage_curve.csv"],
        "audit_sha256": digest(audit_path), "trajectory_digests": raw_hashes,
    })
    print(layout, selection)


def render_development():
    """Draw the native known-floor development trace by itself."""
    from audit_reacquisition import inspect_run

    manifest_path = ROOT / "execution/development/phase_receipt.json"
    phase = read(manifest_path)
    worker, = phase["episodes"]
    layout = worker["layout"]
    run = Path(worker["run_path"])
    manifest = read(run / "manifest.json")
    freeze = read(ROOT / "DEVELOPMENT_FREEZE.json")
    cold = read(Path(freeze["scene_root"]) / "prepared" / layout / "initial_state.json")
    require(digest(run / "manifest.json") == worker["result_sha256"], "Development manifest differs")
    for name in ("steps.json", "audit.json", "observed_belief_trace.npz", "initial_raw_rgbd_scan_sha256.json"):
        require(digest(run / name) == manifest["files_sha256"][name], f"Development source differs {name}")
    trace = read(run / "steps.json")
    with np.load(run / "observed_belief_trace.npz", allow_pickle=False) as archive:
        belief = archive["belief"]
        require(np.array_equal(archive["paid"], np.arange(len(trace)+1)), "Development clocks differ")
    reference = (ROOT.parent / "external_support_repair_20261126/scenes/evaluator_references"
                 / layout / "evaluator_truth.npz")
    require(digest(reference) == "b732f50b502b9222a3dabc735ca29651a8d69658d451898d4965f8ac08a79060",
            "Known development V1 mask hash changed")
    with np.load(reference, allow_pickle=False) as raw:
        mask = raw["reachable"]
    audit = inspect_run({"trace": trace, "actual_paid": len(trace), "beliefs": belief,
                         "events": read(run / "audit.json"),
                         "initial_raw": read(run / "initial_raw_rgbd_scan_sha256.json")}, cold, "B")
    data = np.asarray([cold["anchor"], *[[*row["position"], row["heading"]] for row in trace]])
    xs, ys = world_xy(data, cold["shape"], cold["grid_resolution_m"])
    counts = np.count_nonzero((belief == 0) & mask, axis=(1, 2))
    fig = plt.figure(figsize=(10.6, 5.4))
    grid = fig.add_gridspec(2, 2, width_ratios=(1.02, 1.15), height_ratios=(.84, 1.16))
    path_ax, gain_ax, cov_ax = fig.add_subplot(grid[:, 0]), fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, 1])
    for ax in (path_ax, gain_ax, cov_ax):
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=9)
    resolution, shape = cold["grid_resolution_m"], cold["shape"]
    path_ax.imshow(np.flipud(belief[0]), origin="lower", interpolation="none",
                   extent=(0, shape[1]*resolution, 0, shape[0]*resolution),
                   cmap=ListedColormap(("#f2f3f5", "#ffffff", "#4a565d")),
                   norm=BoundaryNorm((-1.5, -.5, .5, 1.5), 3), zorder=0)
    path_ax.plot(xs, ys, color="#506477", lw=1.45, alpha=.88, zorder=2, label="Other paid actions")
    path_ax.scatter(xs[0], ys[0], marker="o", s=37, color="#29353f", zorder=7)
    path_ax.scatter(xs[-1], ys[-1], marker="X" if trace[-1]["collision"] else "s", s=42,
                    color="#3f4851", zorder=7)
    cov_ax.step(range(len(counts)), 100*counts/int(mask.sum()), where="post", lw=1.7, color="#415771")
    labels_used = set()
    first_crossing = next(row["crossing_paid"] for row in audit["attempts"] if row["crossing_paid"] is not None)
    gain_clock = min(row["arrival_paid"] for row in audit["attempts"]
                     if row["arrival_paid"] is not None and row["arrival_paid"] > first_crossing)
    gain = (belief[gain_clock] == 0) & (belief[first_crossing] != 0) & mask
    new_cells = int(np.count_nonzero(gain))
    gain_points = np.argwhere(gain)
    require(new_cells > 0, "The shown post-crossing fixed-mask interval has no observed gain")
    gain_ax.imshow(np.flipud(np.ma.masked_where(~gain, gain)), origin="lower", interpolation="none",
                   extent=(0, shape[1]*resolution, 0, shape[0]*resolution),
                   cmap=ListedColormap(("#5eb7aa",)), vmin=0, vmax=1, alpha=.65)
    gain_ax.plot(xs[first_crossing:gain_clock+1], ys[first_crossing:gain_clock+1], color="#425772", lw=1.6,
                 alpha=.9, zorder=3)
    gain_ax.scatter(xs[first_crossing], ys[first_crossing], marker="D", color=COLOR["cross"], edgecolor="white",
                    s=31, linewidth=.6, zorder=5)
    gain_ax.scatter(xs[gain_clock], ys[gain_clock], marker="^", color=COLOR["view"], edgecolor="white",
                    s=31, linewidth=.6, zorder=5)
    gain_xy = np.column_stack(world_xy(gain_points, shape, resolution))
    gain_xy = np.vstack((gain_xy, np.column_stack((xs[first_crossing:gain_clock+1], ys[first_crossing:gain_clock+1]))))
    gain_low, gain_high = gain_xy.min(axis=0), gain_xy.max(axis=0)
    gain_ax.set(xlim=(gain_low[0]-.4, gain_high[0]+.4), ylim=(gain_low[1]-.38, gain_high[1]+.38),
                xlabel="x (m)", ylabel="y (m)", aspect="equal")
    gain_ax.text(.98, .965, f"{new_cells} newly observed correct free cells",
                 transform=gain_ax.transAxes, ha="right", va="top", fontsize=8.3, color="#22695f")
    gain_ax.set_title(f"(b) Newly covered area  p{first_crossing}–p{gain_clock}", loc="left",
                     fontsize=10.7, fontweight="bold")
    for attempt in audit["attempts"]:
        start, arrival, crossed, stop = (attempt[key] for key in (
            "dispatch_paid", "arrival_paid", "crossing_paid", "last_paid"))
        cov_ax.axvspan(start, stop, color=COLOR["view"], alpha=.105, zorder=0)
        if arrival is not None:
            label = "Acquisition route" if "view" not in labels_used else "_nolegend_"
            path_ax.plot(xs[start:arrival+1],ys[start:arrival+1], lw=2, color=COLOR["view"],
                         linestyle=(0,(2,1)), label=label, zorder=3)
            labels_used.add("view")
            path_ax.scatter(xs[arrival], ys[arrival], marker="^", color=COLOR["view"], edgecolor="white", s=44,
                            linewidth=.6,zorder=8)
            path_ax.annotate(f"p{arrival}", (xs[arrival], ys[arrival]), xytext=(5, 5), textcoords="offset points",
                             fontsize=8, color=COLOR["view"], zorder=9)
        if crossed is not None:
            label = "Certified crossing route" if "cross" not in labels_used else "_nolegend_"
            path_ax.plot(xs[arrival:crossed+1],ys[arrival:crossed+1], lw=2, color=COLOR["cross"],
                         label=label, alpha=.8, zorder=4)
            labels_used.add("cross")
            path_ax.scatter(xs[crossed], ys[crossed], marker="D", color=COLOR["cross"], edgecolor="white", s=44,
                            linewidth=.6,zorder=8)
            origin = crossed-1
            label = "Initially blocked edge origin" if "origin" not in labels_used else "_nolegend_"
            labels_used.add("origin")
            path_ax.scatter(xs[origin], ys[origin], marker="o", facecolors="none", edgecolors="#3b3133",
                            linewidth=1.2, s=48, label=label, zorder=8)
            path_ax.annotate("", (xs[crossed], ys[crossed]), xytext=(xs[origin], ys[origin]),
                             arrowprops={"arrowstyle": "-|>", "color": COLOR["cross"],
                                         "linewidth": 2.1, "shrinkA": 1.5, "shrinkB": 1.5}, zorder=9)
            # Labels in the coverage panel remain separated for nearby crossings.
            cov_ax.axvline(crossed, color=COLOR["cross"], linestyle=":", lw=1.2)
            cov_ax.annotate(f"Cross p{crossed}", (crossed, 100*counts[crossed]/int(mask.sum())),
                            xytext=(4, 11), textcoords="offset points", color=COLOR["cross"], fontsize=8)
    padding = .51
    path_ax.set(xlim=(xs.min()-padding, xs.max()+padding), ylim=(ys.min()-padding, ys.max()+padding),
                xlabel="x (m)", ylabel="y (m)", aspect="equal")
    path_ax.legend(loc="upper right", frameon=False, fontsize=8)
    path_ax.set_title("(a) Paid motions and viewpoints", loc="left", fontsize=11, fontweight="bold")
    cov_ax.set(xlabel="Paid action", ylabel="Correct free coverage (%)", xlim=(-2, len(trace)+3))
    cov_ax.grid(axis="y", linewidth=.7, alpha=.22)
    cov_ax.set_title("(c) Matched-reference coverage", loc="left", fontsize=11, fontweight="bold")
    fig.text(.07, .025, "HM3D 00770 development run. Grey = unobserved at start. × = collision. Mask = 1,640 cells.",
             fontsize=8.6)
    fig.tight_layout(rect=(0, .07, 1, 1))
    for extension in ("pdf", "png"):
        fig.savefig(HERE / f"development_reacquisition_case.{extension}", dpi=240, bbox_inches="tight")
    plt.close(fig)
    write_json(HERE / "development_reacquisition_case_source.json",
               {"layout": layout, "scope": "Single known development floor and start",
                "native_manifest_sha256": worker["result_sha256"], "audit_sha256": digest(run / "audit.json"),
                "reference_sha256": digest(reference), "terminal_paid": len(trace),
                "terminal_correct_free_cells": int(counts[-1]), "event_counts": audit["event_counts"],
                "new_area_from_paid": first_crossing, "new_area_to_paid": gain_clock,
                "new_correct_free_cells_in_panel_b": new_cells})


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--development", action="store_true", help="render the separate known-floor development run")
    args = parser.parse_args()
    render_development() if args.development else render()
