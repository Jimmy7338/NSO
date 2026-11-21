"""Summarize the two paid continuations and draw only measured J checkpoints."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent
METRICS = ("C_free", "F1_scene_05cm", "J")
ARMS = ("G", "S")
COLORS = {"G": "#53637A", "S": "#007B82"}
LABELS = {"G": "G · neutral prior", "S": "S · category prior"}
PANEL_LABELS = {"gallery_left": "a  Gallery · shelf left", "warehouse_right": "b  Warehouse · shelf right"}


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def flatten_rows(protocol, rows):
    by_layout = {row["layout"]: row for row in rows}
    if len(rows) != 2 or set(by_layout) != set(protocol["layouts_in_order"]):
        raise ValueError("Results must preserve both predeclared comparisons")
    gate = protocol["engineering_gate"]
    output, outcomes = [], []
    for layout in protocol["layouts_in_order"]:
        pair = by_layout[layout]
        config = protocol["pairs"][layout]
        timing = {
            "actual_split_verified": pair["verified_actual_split"],
            "first_command_difference_paid": (pair.get("first_command_difference") or {}).get("paid"),
            "first_pose_difference_paid": (pair.get("first_pose_difference") or {}).get("paid"),
            "first_sensor_content_difference_paid": (pair.get("first_sensor_content_difference") or {}).get("paid"),
        }
        by_checkpoint = {}
        for label in ("matched", "terminal"):
            points = pair.get(label) or {}
            row = {"layout": layout, "state_id": config["state_id"], "checkpoint": label,
                   "checkpoint_target_paid": config["equal_paid_checkpoint"] if label == "matched" else 256,
                   "pair_status": pair["status"], "prefix_paid": config["paid_before"], **timing}
            for arm in ARMS:
                point = points.get(arm) or {}
                for field in ("status", "actual_paid", *METRICS, "returned", "collisions", "stop_reason", "failed"):
                    row[f"{arm}_{field}"] = point.get(field)
                if point.get("J") is not None:
                    if not math.isclose(point["C_free"] * point["F1_scene_05cm"], point["J"], abs_tol=1e-12, rel_tol=0):
                        raise ValueError(f"Recorded J differs from C_free × F1 at {layout}/{arm}/{label}")
                    if label == "matched" and point["actual_paid"] != config["equal_paid_checkpoint"]:
                        raise ValueError(f"Nonmatching paid sample passed as equal budget in {layout}/{arm}")
            both_captured = all(row[f"{arm}_status"] == "captured" and row[f"{arm}_J"] is not None for arm in ARMS)
            row["both_checkpoints_measured"] = both_captured
            for metric in METRICS:
                row[f"S_minus_G_{metric}_pp"] = 100 * (row[f"S_{metric}"] - row[f"G_{metric}"]) if both_captured else None
            output.append(row)
            by_checkpoint[label] = row
        term = by_checkpoint["terminal"]
        safe = term["both_checkpoints_measured"] and all(term[f"{arm}_returned"] is True and
                term[f"{arm}_collisions"] == 0 and not term[f"{arm}_failed"] and
                term[f"{arm}_actual_paid"] <= protocol["total_paid_action_budget"] for arm in ARMS)
        threshold = gate["required_terminal_J_gain_strictly_greater_than_fraction"]
        amount = term["S_minus_G_J_pp"]
        outcomes.append({"layout": layout, "verified_actual_split": pair["verified_actual_split"],
                         "both_terminal_safety_passed": safe, "terminal_delta_J_pp": amount,
                         "strict_gain_passed": amount is not None and amount > 100 * threshold,
                         "matched_delta_J_pp": by_checkpoint["matched"]["S_minus_G_J_pp"]})
    return output, outcomes


def curve_samples(protocol, episodes):
    sampled = {}
    for row in episodes:
        if row["status"] != "captured" or row.get("J") is None:
            continue
        identity = row["layout"], row["mode"]
        clock = row["actual_paid"]
        if clock < protocol["pairs"][identity[0]]["paid_before"]:
            continue
        previous = sampled.setdefault(identity, {}).get(clock)
        if previous and not math.isclose(previous["J"], row["J"], abs_tol=1e-12, rel_tol=0):
            raise ValueError(f"Two different J observations at one paid clock {identity}/{clock}")
        if previous is None or row["checkpoint"] == "terminal":
            sampled[identity][clock] = row
    return sampled


def plot(protocol, pairs, checkpoints, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "text.color": "#1C2A39",
                         "axes.labelcolor": "#1C2A39", "axes.titlecolor": "#1C2A39", "xtick.color": "#465973",
                         "ytick.color": "#465973", "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#CCD7E1", "pdf.fonttype": 42, "svg.fonttype": "none",
                         "svg.hashsalt": "same-goal-live-witness-20261121"})
    sampled = curve_samples(protocol, checkpoints)
    fig, panels = plt.subplots(1, 2, figsize=(10.15, 4.35))
    fig.subplots_adjust(left=.11, right=.983, top=.68, bottom=.22, wspace=.15)
    fig.text(.11, .96, "Joint map quality after one same-goal choice", va="top", fontsize=20, weight="bold")
    fig.text(.11, .852, "Two archived development states · 256 paid actions including recorded prefix and return", color="#465973")
    handles = [Line2D([], [], color=COLORS[m], marker=marker, label=LABELS[m])
               for m, marker in zip(ARMS, ("o", "s"), strict=True)]
    handles.extend([Line2D([], [], color="#868F9F", linestyle=":", label="First issued command split"),
                    Line2D([], [], color="#B9C7D6", linestyle="--", label="Equal-paid checkpoint")])
    fig.legend(handles=handles, ncol=4, frameon=False, bbox_to_anchor=(.104, .826), loc="upper left",
               columnspacing=1, handlelength=1.6, handletextpad=.45, fontsize=9)
    pair_map = {row["layout"]: row for row in pairs}
    for axis, layout in zip(panels, protocol["layouts_in_order"], strict=True):
        rule, pair = protocol["pairs"][layout], pair_map[layout]
        values = [row["J"] * 100 for arm in ARMS for row in sampled.get((layout, arm), {}).values()]
        extent = max(values, default=100) - min(values, default=0)
        margin = max(.45, extent * .09)
        ymin = max(0, math.floor(min(values, default=0) - margin))
        ymax = min(102, math.ceil(max(values, default=100) + margin))
        split = pair.get("first_command_difference")
        if split:
            axis.axvline(split["paid"], color="#868F9F", linestyle=":", linewidth=1.1, zorder=1)
        axis.axvline(rule["equal_paid_checkpoint"], color="#B9C7D6", linestyle="--", linewidth=1.1, zorder=1)
        for arm, marker, style in zip(ARMS, ("o", "s"), ("-", "--"), strict=True):
            ordered = sorted(sampled.get((layout, arm), {}).items())
            axis.plot([clock for clock, _ in ordered], [100 * row["J"] for _, row in ordered],
                      color=COLORS[arm], linewidth=1.8, marker=marker, linestyle=style, markersize=4)
            endpoint = next(((clock, row) for clock, row in reversed(ordered) if row["checkpoint"] == "terminal"), None)
            if endpoint:
                clock, row = endpoint
                axis.scatter([clock], [100 * row["J"]], marker=marker, facecolor="white", edgecolor=COLORS[arm],
                             s=45, linewidth=1.4, zorder=5)
        axis.set_title(PANEL_LABELS[layout], loc="left", weight="semibold", fontsize=12, pad=8)
        axis.set_xlim(rule["paid_before"] - 4, 262)
        axis.set_ylim(ymin, ymax)
        main_ticks = [56, 104, 160, 208, 256] if layout == "gallery_left" else [142, 176, 206, 232, 256]
        axis.set_xticks(main_ticks)
        axis.set_xlabel("Paid actions")
        axis.grid(axis="y", color="#DFE7EF", linewidth=.72)
        terminal = pair.get("terminal") or {}
        difference = terminal.get("delta_S_minus_G", {}).get("J")
        if difference is not None:
            axis.text(.97, .065, f"Terminal S − G  {100*difference:+.3f} pp", ha="right", va="bottom",
                      transform=axis.transAxes, color="#465973", fontsize=9)
    panels[0].set_ylabel("Joint quality J (%)")
    fig.text(.11, .105, "Markers show native-map checkpoints. Lines connect sampled times. Hollow markers show actual stops.",
             color="#465973", fontsize=8.4)
    fig.text(.11, .061, "Panels use separate y-axis ranges. J = correctly mapped free-space fraction × whole-scene F1 at 5 cm.",
             color="#465973", fontsize=8.4)
    fig.savefig(output / "J_by_paid_action.pdf", metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(output / "J_by_paid_action.svg", metadata={"Date": None})
    fig.savefig(output / "J_by_paid_action.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    rules = ROOT / "PROTOCOL.json"
    paths = [ROOT / "measurement/results" / f"{name}.json" for name in ("pair_causal_audit", "checkpoints")]
    protocol = read(rules)
    pairs, checkpoints = [read(path)["rows"] for path in paths]
    rows, outcomes = flatten_rows(protocol, pairs)
    args.out.mkdir(parents=True, exist_ok=True)
    table = args.out / "paired_results.csv"
    with table.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot(protocol, pairs, checkpoints, args.out)
    decision = {"per_layout": outcomes, "continue_to_separate_zero_start_proposal": all(
        r["verified_actual_split"] and r["both_terminal_safety_passed"] and r["strict_gain_passed"] for r in outcomes),
        "input_sha256": {str(path.relative_to(ROOT)): sha(path) for path in [rules, *paths, Path(__file__)]},
        "result_sha256": {path.name: sha(path) for path in [table, *(args.out.glob("J_by_paid_action.*"))]}}
    (args.out / "decision.json").write_text(json.dumps(decision, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
