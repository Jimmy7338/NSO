"""Create the per-layout table and paired terminal-quality plot from sealed evaluation."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent
METRICS = ("C_free", "F1_scene_05cm", "J")
EPISODE_FIELDS = (*METRICS, "actual_paid", "returned", "collisions", "stop_reason", "status", "failed",
                  "legal_alternative_decisions", "legal_alternate_options", "different_hypothetical_winners",
                  "different_from_native_choices")
DISPLAY = {"warehouse_left": "Warehouse · left", "warehouse_right": "Warehouse · right",
           "gallery_left": "Gallery · left", "gallery_right": "Gallery · right",
           "ordinary_chain_13": "Chain 13 m", "ordinary_cycle_14": "Cycle 14 m",
           "ordinary_chain_15": "Chain 15 m", "ordinary_cycle_16": "Cycle 16 m"}
MODE_FORMAT = {
    "newG": ("#4F5D72", "o", "New G"), "newS": ("#007D87", "s", "New S"),
    "originalS": ("#C06A2E", "D", "Original S"), "F_strict": ("#856199", "^", "Nearest frontier"),
}


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def report_rows(episodes, audits, config):
    index = {(row["layout"], row["mode"]): row for row in episodes}
    expected = {(layout, mode) for layout in config["layouts"] for mode in config["modes"]}
    if len(index) != len(episodes) or index.keys() != expected:
        raise ValueError("Episode table does not exactly cover the declared matrix")
    causal = {row["layout"]: row for row in audits}
    if causal.keys() != set(config["layouts"]) or len(causal) != len(audits):
        raise ValueError("Missing same-input differential audit")
    rows = []
    for layout in config["layouts"]:
        pair = causal[layout]
        row = {"layout": layout, "native_budget": config["paid_action_budget"], "category_causal_status": pair["status"],
               "category_causal_paid_action": pair.get("causal_paid_action")}
        for mode in config["modes"]:
            episode = index[layout, mode]
            if episode.get("J") is not None and not math.isclose(episode["C_free"] * episode["F1_scene_05cm"], episode["J"], abs_tol=1e-12):
                raise ValueError(f"Inconsistent J for {layout}/{mode}")
            row.update({f"{mode}_{key}": episode.get(key) for key in EPISODE_FIELDS})
        for reference in (mode for mode in config["modes"] if mode != "newS"):
            for metric in METRICS:
                a, b = row[f"newS_{metric}"], row[f"{reference}_{metric}"]
                row[f"newS_minus_{reference}_{metric}_pp"] = 100 * (a - b) if a is not None and b is not None else None
        rows.append(row)
    return rows, index


def plot(index, config, stage, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": "#172735", "axes.labelcolor": "#172735",
                         "xtick.color": "#46596B", "ytick.color": "#46596B", "axes.titlesize": 11,
                         "axes.edgecolor": "#C9D4E1", "axes.spines.top": False, "axes.spines.right": False,
                         "axes.spines.left": False, "pdf.fonttype": 42, "svg.fonttype": "none",
                         "svg.hashsalt": "nso-same-goal-terminal-study-v1"})
    fig, panels = plt.subplots(1, 3, figsize=(10.5, 5.1), sharey=True)
    fig.subplots_adjust(left=.177, right=.979, bottom=.21, top=.68, wspace=.18)
    fig.text(.176, .945, "Terminal quality after same-goal route choice", fontsize=20, weight="bold", va="top")
    scope = "Development placements in two parent scenes" if stage == "development" else "Four independent held-out parent layouts"
    fig.text(.177, .848, f"{scope} · all actions charged to the common budget of {config['paid_action_budget']}",
             fontsize=10, color="#46596B")
    metric_titles = ("Correct free-space coverage", "Whole-scene F1 at 5 cm", "Joint quality J")
    handles = [Line2D([], [], color=MODE_FORMAT[mode][0], marker=MODE_FORMAT[mode][1],
                      linestyle="", label=MODE_FORMAT[mode][2], markersize=6) for mode in config["modes"]]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.167, .816), ncol=len(handles), frameon=False,
               columnspacing=1.1, handletextpad=.35)
    for axis, metric, title in zip(panels, METRICS, metric_titles, strict=True):
        known = [100 * row[metric] for row in index.values() if row.get(metric) is not None]
        low = max(0, (min(known) // 5) * 5 - 2) if known else 0
        low = min(low, 94)
        axis.set_title(title, weight="semibold", pad=9)
        if stage == "development" and metric == "J":
            axis.set_facecolor("#F3F7FA")
        axis.set_xlim(low, 101)
        axis.set_ylim(-.58, len(config["layouts"]) - .42)
        axis.set_yticks(list(range(len(config["layouts"]))), [DISPLAY[layout] for layout in config["layouts"]][::-1])
        axis.tick_params(axis="y", length=0, pad=7, labelsize=10)
        axis.tick_params(axis="x", labelsize=9)
        axis.grid(axis="x", color="#E0E7EF", linewidth=.8, zorder=0)
        axis.set_xlabel("Percent", fontsize=9)
        for rank, layout in enumerate(reversed(config["layouts"])):
            chosen = [index[layout, mode] for mode in config["modes"]]
            coords = [100 * row[metric] for row in chosen if row.get(metric) is not None]
            if len(coords) > 1:
                axis.hlines(rank, min(coords), max(coords), color="#AFC0CF", linewidth=1.6)
            for position, item in enumerate(chosen):
                if item.get(metric) is None:
                    continue
                color, marker, _ = MODE_FORMAT[item["mode"]]
                height = rank + (position - (len(chosen) - 1) / 2) * .103
                axis.scatter(100 * item[metric], height, marker=marker, color=color, s=36,
                             edgecolor="white", linewidth=.55, zorder=4)
    fig.text(.176, .096, "Each marker shows a measured terminal result. Paired markers are offset vertically for readability.",
             fontsize=8.5, color="#46596B")
    fig.text(.176, .063, "J = correctly mapped free-space fraction × whole-scene F1. Runs without a metric are kept in the CSV.",
             fontsize=8.5, color="#46596B")
    fig.savefig(out / "paired_quality.pdf", metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(out / "paired_quality.svg", metadata={"Date": None})
    fig.savefig(out / "paired_quality.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("development", "holdout"))
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    protocol = ROOT / "PROTOCOL.json"
    config = read(protocol)["development" if args.stage == "development" else "heldout"]
    measurements = ROOT / "measurement/results" / args.stage
    sources = [measurements / name for name in ("episodes.json", "choice_audit.json")]
    rows, index = report_rows(*(read(path)["rows"] for path in sources), config)
    out = args.out or ROOT / "results" / args.stage
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "paired_layouts.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot(index, config, args.stage, out)
    outputs = [csv_path, *(out / f"paired_quality.{suffix}" for suffix in ("pdf", "svg", "png"))]
    receipt = {"stage": args.stage, "layouts": len(rows),
               "input_sha256": {str(path.relative_to(ROOT)): sha(path) for path in (protocol, Path(__file__), *sources)},
               "output_sha256": {path.name: sha(path) for path in outputs}}
    (out / "REPORT_RECEIPT.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
