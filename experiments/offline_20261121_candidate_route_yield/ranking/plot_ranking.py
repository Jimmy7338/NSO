"""Draw pairwise predictions, observed surface and whole-scene yield at matched cost."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from archive_predictions import MODES, ROOT, digest, read_json, write_json

RANKING = Path(__file__).resolve().parent
TEXT = "#18283a"
MUTED = "#526479"
GRID = "#dce4ed"
PREDICTIONS = {
    "G": ("#626f80", "o", "G: neutral geometry"),
    "S": ("#0063a6", "^", "S: correct category"),
    "X": ("#b45309", "s", "X: swapped category"),
}
NET = "#683a83"
GROSS = "#a477ac"
QUALITY = "#13735c"


def draw_markers(axis, keys, rows, offsets, colors, markers, names, multiplier=1.0,
                 open_markers=(), comparable_only=True):
    for key, offset, color, marker, name in zip(keys, offsets, colors, markers, names):
        differences = [row[f"diff_{key}"] for row in rows]
        points = [
            (value * multiplier, len(rows) - 1 - index + offset)
            for index, (row, value) in enumerate(zip(rows, differences))
            if value is not None and (not comparable_only or row["comparable_direction"])
        ]
        x, y = zip(*points) if points else ((), ())
        axis.scatter(
            x, y, marker=marker, s=32, facecolor="white" if key in open_markers else color,
            edgecolor=color, linewidth=0.9, label=name, zorder=5,
        )


def align_scales(axis, rows, keys, threshold=None, multiplier=1.0, comparable_only=True):
    values = [
        abs(row[f"diff_{key}"] * multiplier)
        for key in keys for row in rows
        if (not comparable_only or row["comparable_direction"]) and row[f"diff_{key}"] is not None
    ]
    max_point = max([*values, threshold or 0.0, 1e-6])
    axis.set_xlim(-max_point * 1.16, max_point * 1.16)
    axis.xaxis.set_major_locator(MaxNLocator(nbins=5, symmetric=True, min_n_ticks=3))
    if threshold is not None:
        axis.axvspan(-threshold, threshold, color="#eef2f6", ec="none", zorder=0)
    axis.axvline(0, lw=1.0, color="#8193a7", zorder=2)
    axis.set_axisbelow(True)
    axis.xaxis.grid(True, lw=0.55, color=GRID, zorder=1)


def titles(axis, letter, heading, unit):
    axis.text(-0.005, 1.055, letter, transform=axis.transAxes, fontsize=13, weight="bold", color=TEXT)
    axis.text(0.10, 1.055, heading, transform=axis.transAxes, fontsize=11, weight="bold", color=TEXT)
    axis.text(0.10, 1.00, unit, transform=axis.transAxes, fontsize=8.5, color=MUTED)


def cost_label(row):
    a, b = [f"{row[f'{tag}_native_outbound']}/{row[f'{tag}_native_reserved_return']}" for tag in ("a", "b")]
    return a if a == b else f"{a} vs {b}"


def paid_label(row):
    a, b = ((row[f"{tag}_paid"] if row[f"{tag}_paid"] is not None else "NA") for tag in ("a", "b"))
    return str(a) if a == b else f"{a} vs {b}"


def render(summary, protocol, output):
    rows = [row for row in summary["pairwise"] if row["comparison"] in {"strict_arrival", "additional_common_paid"}]
    metrics = protocol["metrics"]
    with plt.rc_context({
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "text.color": TEXT,
        "axes.labelcolor": TEXT,
        "xtick.color": MUTED,
        "svg.fonttype": "none",
        "svg.hashsalt": "nso-candidate-ranks-20261121",
        "pdf.fonttype": 42,
    }):
        fig = plt.figure(figsize=(13.4, 5.8), facecolor="white")
        layout = fig.add_gridspec(
            1, 4, width_ratios=[1.8, 1.36, 1.16, 1.16],
            left=0.041, right=0.977, bottom=0.27, top=0.77, wspace=0.20,
        )
        labels, scores_axis, surface_axis, quality_axis = [fig.add_subplot(layout[0, i]) for i in range(4)]
        fig.text(0.042, 0.955, "Native route scores and observed exploration yield",
                 size=17, weight="bold", va="top")
        fig.text(0.042, 0.897, "Two parent layouts, each evaluated with left and right object-placement swaps",
                 size=10, color=MUTED, va="top")
        for axis in (labels, scores_axis, surface_axis, quality_axis):
            axis.set_ylim(-0.55, len(rows) - 0.26)
            axis.set_yticks([])
            axis.spines[["top", "left", "right"]].set_visible(False)
            axis.spines["bottom"].set_color(GRID)
            axis.tick_params(axis="x", length=2.5, width=0.65, pad=4)
            for i in range(len(rows) - 1):
                axis.axhline(i + 0.5, lw=0.60, color=GRID, zorder=3)
        labels.set_xlim(0, 1)
        labels.set_xticks([])
        labels.spines["bottom"].set_visible(False)
        for x, name in [(0.00, "Layout / object side"), (0.49, "Pair"), (0.64, "Native O/R"), (0.97, "Paid")]:
            labels.text(x, 1.052, name, transform=labels.transAxes, fontsize=9.2, weight="bold",
                        ha="right" if name == "Paid" else "left")
        for i, row in enumerate(rows):
            y = len(rows) - 1 - i
            layout_title = row["layout"].replace("_", " / ").capitalize()
            pair = f"{row['candidate_a']}–{row['candidate_b']}"
            if row["comparison"] == "additional_common_paid":
                pair += "†"
            for x, value in [(0.00, layout_title), (0.49, pair), (0.64, cost_label(row)), (0.97, paid_label(row))]:
                labels.text(x, y, value, va="center", ha="right" if x == 0.97 else "left")
            if not row["comparable_direction"]:
                for axis in (surface_axis, quality_axis):
                    axis.text(0.5, y, "Unavailable", color=MUTED, style="italic", ha="center",
                              va="center", transform=axis.get_yaxis_transform())

        terms = [f"object_{mode}_minus_N" for mode in MODES]
        align_scales(scores_axis, rows, terms, comparable_only=False)
        draw_markers(
            scores_axis, terms, rows, (-0.19, 0, 0.19),
            [PREDICTIONS[mode][0] for mode in MODES], [PREDICTIONS[mode][1] for mode in MODES], MODES,
            comparable_only=False,
        )
        titles(scores_axis, "A", "Predicted object score", "Δ object component of native score")
        scores_axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=3, frameon=False,
                           handletextpad=0.25, columnspacing=0.9, borderaxespad=0)

        surface_terms = ("object_net_m2", "object_gross_m2")
        align_scales(surface_axis, rows, surface_terms, threshold=metrics["material_object_area_m2"])
        draw_markers(surface_axis, surface_terms, rows, (-0.13, 0.13),
                     (NET, GROSS), ("o", "o"), ("Net", "Gross"), open_markers=(surface_terms[1],))
        titles(surface_axis, "B", "Measured object surface", "Δ area in m²")
        surface_axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=2, frameon=False,
                           handletextpad=0.3, columnspacing=0.7, borderaxespad=0)

        align_scales(quality_axis, rows, ("J",), threshold=100 * metrics["material_J_fraction"], multiplier=100)
        draw_markers(quality_axis, ("J",), rows, (0,), (QUALITY,), ("D",), ("J",), multiplier=100)
        titles(quality_axis, "C", "Whole-scene yield", "Δ J in percentage points")
        quality_axis.text(0.5, -0.125, r"$J=C_{\mathrm{free}}\; F_{1,5\,\mathrm{cm}}$", transform=quality_axis.transAxes,
                          va="top", ha="center", fontsize=8.5, color=MUTED)

        fig.text(0.042, 0.123, "Δ = lower-ID route minus higher-ID route. O/R = planned outbound / reserved return actions. "
                 "Paid = actual action checkpoint.", size=8.4, color=MUTED)
        fig.text(0.042, 0.084, "† Gallery pair 0–1 is compared at paid action 58 using neutral-policy continuation "
                 "for the shorter route. Other rows compare target arrival.", size=8.4, color=MUTED)
        fig.text(0.042, 0.045, "G: neutral geometry   ·   S: correct category   ·   X: swapped category. "
                 "Shaded bands show practical ties (±0.05 m² and ±0.10 percentage points).", size=8.4, color=MUTED)
        output.mkdir(parents=True, exist_ok=True)
        for extension, extra in (
            ("png", {"dpi": 210, "metadata": {"Software": "Matplotlib"}}),
            ("pdf", {"metadata": {"CreationDate": None, "ModDate": None}}),
            ("svg", {"metadata": {"Date": None}}),
        ):
            fig.savefig(output / f"candidate_rank_yield.{extension}", **extra)
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "PROTOCOL.json")
    parser.add_argument("--summary", type=Path, default=RANKING / "results/ranking_summary.json")
    parser.add_argument("--output", type=Path, default=RANKING / "results")
    args = parser.parse_args()
    render(read_json(args.summary), read_json(args.protocol), args.output)
    write_json(args.output / "figure_receipt.json", {
        "script_sha256": digest(Path(__file__)),
        "ranking_summary_sha256": digest(args.summary),
        "protocol_sha256": digest(args.protocol),
        "outputs_sha256": {
            f"candidate_rank_yield.{ext}": digest(args.output / f"candidate_rank_yield.{ext}")
            for ext in ("png", "pdf", "svg")
        },
    })
