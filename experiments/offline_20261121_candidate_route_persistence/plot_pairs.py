"""Plot measured J at paired action checkpoints."""

import argparse
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

from summarize import ROOT, read, select_checkpoints


SHELF = "#086878"
CABINET = "#B1531B"
ORIGINAL = "#5A6271"
INK = "#182534"
LABELS = {
    "warehouse_left": "Warehouse · shelf left", "warehouse_right": "Warehouse · shelf right",
    "gallery_left": "Gallery · shelf left", "gallery_right": "Gallery · shelf right",
}


def plot(protocol, checkpoints, terminals, output):
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11, "axes.labelcolor": INK,
        "axes.titlecolor": INK, "text.color": INK, "axes.edgecolor": "#C9D4E1",
        "xtick.color": "#44566F", "ytick.color": "#44566F", "axes.spines.right": False,
        "axes.spines.top": False, "pdf.fonttype": 42, "svg.fonttype": "none",
        "svg.hashsalt": "nso-candidate-persistence-20261121",
    })
    by_route = select_checkpoints(checkpoints + terminals, protocol["measurement"]["numerical_tolerance"])
    terminal_by_route = {(row["layout"], row["candidate_id"]): row for row in terminals}
    figure, panels = plt.subplots(2, 2, figsize=(10.4, 6.5), sharex=True, sharey=True)
    figure.subplots_adjust(left=.105, right=.969, top=.794, bottom=.174, wspace=.14, hspace=.35)
    figure.text(.105, .965, "Joint map quality along paired routes", fontsize=21, weight="bold", va="top")
    figure.text(.105, .898, "Two parent scenes with shelves and cabinets swapped between sides.", color="#44566F")
    handles = [Line2D([], [], color=SHELF, marker="o", label="Shelf-side route"),
               Line2D([], [], color=CABINET, marker="s", label="Cabinet-side route"),
               Line2D([], [], color=ORIGINAL, marker="D", linestyle="--", label="Originally selected route (Gallery)")]
    figure.legend(handles=handles, loc="upper left", bbox_to_anchor=(.099, .874), ncol=3, frameon=False,
                  handlelength=1.7, columnspacing=1.45, handletextpad=.58, borderaxespad=0)
    for index, (axis, layout) in enumerate(zip(panels.flat, protocol["layouts_in_order"], strict=True)):
        gallery = layout.startswith("gallery")
        source = "gallery_vs_other_side" if gallery else "warehouse"
        shelf, cabinet = protocol["decision"][source][layout]
        specs = [(shelf, SHELF, "o", "-"), (cabinet, CABINET, "s", "-")]
        if gallery:
            specs.insert(0, (1, ORIGINAL, "D", "--"))
            axis.axvspan(38, 58, color="#E6EDF5", zorder=0, alpha=.9)
        for route, color, marker, style in specs:
            series = sorted(by_route[layout, route].items())
            x, y = [], []
            for clock, row in series:
                if x and clock - x[-1] > 80:
                    x.append(float("nan"))
                    y.append(float("nan"))
                x.append(clock)
                y.append(row["J"] * 100)
            axis.plot(x, y, color=color, marker=marker, markersize=3.9, linewidth=1.65,
                      linestyle=style, alpha=.92, zorder=4 if route == shelf else 3)
            terminal = terminal_by_route[layout, route]
            terminal_marker = marker if terminal["returned"] else "v"
            axis.scatter([terminal["actual_paid"]], [100 * terminal["J"]], s=52, zorder=7,
                         marker=terminal_marker, facecolor="white", edgecolor=color, linewidth=1.35)
        axis.set_title(f"{chr(97 + index)}  {LABELS[layout]}", fontsize=12, weight="semibold", loc="left", pad=9)
        axis.set_xlim(-5, 264)
        axis.set_ylim(0, 105)
        axis.set_xticks([0, 38, 58, 96, 128, 160, 192, 224, 256])
        axis.yaxis.set_major_locator(MultipleLocator(25))
        axis.grid(axis="y", color="#DFE6EF", linewidth=.75, zorder=0)
        axis.tick_params(axis="both", length=3.2, width=.7, labelsize=9)
        end_shelf = terminal_by_route[layout, shelf]["J"]
        comparison_text = f"Terminal shelf vs cabinet  {(end_shelf-terminal_by_route[layout, cabinet]['J'])*100:+.2f} pp"
        if gallery:
            comparison_text += f"\nTerminal shelf vs original  {(end_shelf-terminal_by_route[layout, 1]['J'])*100:+.2f} pp"
        axis.text(.974, .08, comparison_text, transform=axis.transAxes, ha="right", va="bottom",
                  fontsize=8.8, color="#44566F", linespacing=1.5)
    figure.text(.027, .5, "$J = C_{\\mathrm{free}} F_{1,5\\,\\mathrm{cm}}$ (%)", rotation=90, ha="center", va="center")
    figure.text(.535, .10, "Paid actions", ha="center", fontsize=11)
    figure.text(.105, .055, "Markers show native-map measurements. Large archive gaps are left open. Shading marks Gallery actions 38–58.",
                fontsize=8.5, color="#44566F")
    figure.text(.105, .029, "Hollow symbols mark actual terminal states. All plotted runs returned with zero simulated collisions.",
                fontsize=8.5, color="#44566F")
    output.mkdir(parents=True, exist_ok=True)
    figure.savefig(output / "joint_quality_by_action.pdf", metadata={"CreationDate": None, "ModDate": None})
    figure.savefig(output / "joint_quality_by_action.svg", metadata={"Date": None})
    figure.savefig(output / "joint_quality_by_action.png", dpi=190)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "PROTOCOL.json")
    parser.add_argument("--measurements", type=Path, default=ROOT / "measurement/results")
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    os.environ.setdefault("SOURCE_DATE_EPOCH", "1")
    plot(read(args.protocol), read(args.measurements / "checkpoints.json")["rows"],
         read(args.measurements / "terminal.json")["rows"], args.output)


if __name__ == "__main__":
    main()
