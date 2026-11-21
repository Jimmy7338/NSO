#!/usr/bin/env python3
"""Render fixed-reference surface-matching audits from a measured panel archive."""

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, fontManager
from matplotlib.lines import Line2D
import numpy as np


TERMINALS = ("chain13_terminal", "cycle14_terminal", "chain15_terminal", "cycle16_terminal")
TIMELINE = ("cycle16_paid256", "cycle16_paid320", "cycle16_terminal")
STATES = (
    ("#BEC9D5", ".", 2.7, 0.52),  # matched in both
    ("#1B5FA6", "^", 5.5, 0.79),  # matched in G only
    ("#C84A11", "o", 5.2, 0.82),  # matched in S only
    ("#784875", "x", 4.7, 0.68),  # unmatched in both
)
STATE_CODES = (3, 1, 2, 0)
INK = "#172230"
MUTED = "#4C5B6D"
RULE = "#D9E0E7"
DEFAULT_CJK = Path("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf")
AZIMUTH_RAD = np.deg2rad(41)
ELEVATION_RAD = np.deg2rad(36)

COPY = {
    "en": {
        "title_terminal": "Reference surface matching at termination",
        "title_timeline": "Cycle 16 at equal action counts",
        "timeline_note": "Post hoc diagnostic",
        "terminal_panels": ("Chain 13", "Cycle 14", "Chain 15", "Cycle 16"),
        "time_panels": ("256 actions", "320 actions", "512 actions"),
        "orthographic": "Reference points at z ≠ 0 · orthographic view",
        "floor": "Reference points at z = 0 · top view",
        "states": ("Both match", "G only", "S only", "Neither matches"),
        "area": "Reference area (m²)",
        "footnote": ("A reference point matches if it is within 5 cm of the reconstructed surface. "
                     "Markers emphasize uncommon states. Areas use reference-sample weights."),
        "metrics": "Precision / recall / F1 (%)",
    },
    "zh": {
        "title_terminal": "终点参考表面匹配",
        "title_timeline": "环式 16 等动作数对照",
        "timeline_note": "事后诊断",
        "terminal_panels": ("链式 13", "环式 14", "链式 15", "环式 16"),
        "time_panels": ("256 步", "320 步", "512 步"),
        "orthographic": "z ≠ 0 的参考点 · 正交轴测",
        "floor": "z = 0 的参考点 · 俯视",
        "states": ("两者均匹配", "仅 G 匹配", "仅 S 匹配", "两者均未匹配"),
        "area": "参考表面面积（m²）",
        "footnote": "参考点到重建表面的距离不超过 5 cm 计为匹配。符号强调少数状态。面积按参考样本权重统计。",
        "metrics": "精确率 / 召回率 / F1（%）",
    },
}


@dataclass(frozen=True)
class Panel:
    xyz: np.ndarray
    status: np.ndarray
    weight_m2: np.ndarray

    @classmethod
    def read(cls, archive, key):
        xyz = archive[f"{key}__reference_xyz"]
        status = archive[f"{key}__reference_status"]
        weight_m2 = archive[f"{key}__reference_weight_m2"]
        if (xyz.ndim != 2 or xyz.shape[1] != 3 or not len(xyz)
                or status.shape != (len(xyz),) or weight_m2.shape != (len(xyz),)
                or not np.isfinite(xyz).all() or not np.isfinite(weight_m2).all()
                or np.any(weight_m2 <= 0)
                or not np.isin(status, STATE_CODES).all()):
            raise ValueError(f"Invalid measured reference arrays for {key}")
        return cls(xyz, status, weight_m2)

    def area(self, code):
        return float(np.sum(self.weight_m2[self.status == code], dtype=np.float64))


def project(xyz):
    x, y, z = xyz.T
    horizontal = np.cos(AZIMUTH_RAD) * x - np.sin(AZIMUTH_RAD) * y
    depth = np.sin(AZIMUTH_RAD) * x + np.cos(AZIMUTH_RAD) * y
    vertical = np.cos(ELEVATION_RAD) * z - np.sin(ELEVATION_RAD) * depth
    return np.column_stack((horizontal, vertical))


def common_frame(panels):
    xyz = np.concatenate([panel.xyz for panel in panels])
    projected = (project(xyz), xyz[:, :2])
    # The two views need different origins but share the same spans. Across
    # panels the positions are never rescaled or centered per building.
    frames = [(curve.min(axis=0), curve.max(axis=0)) for curve in projected]
    shared_span = np.max([end - start for start, end in frames], axis=0) + 1.2
    return [tuple(zip((start + end - shared_span) / 2, (start + end + shared_span) / 2))
            for start, end in frames]


def draw_points(ax, xy, status):
    for code, (color, marker, size, alpha) in zip(STATE_CODES, STATES):
        points = xy[status == code]
        ax.scatter(points[:, 0], points[:, 1], s=size, marker=marker, c=color,
                   linewidths=.65 if marker == "x" else 0, alpha=alpha)


def legend_handles():
    return [Line2D([], [], color=color, marker=marker, markersize=7, linestyle="none",
                   markeredgewidth=.9 if marker == "x" else 0)
            for color, marker, _, _ in STATES]


def draw_view(ax, coordinates, status, frame):
    ax.set_xlim(frame[0])
    ax.set_ylim(frame[1])
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()
    draw_points(ax, coordinates, status)


def fmt_area(area):
    # Keep nonzero minority states distinguishable from true zero, including
    # when a group contains significantly less than 0.01 m².
    if area == 0:
        return "0"
    return f"{area:.2f}" if area >= 0.01 else f"{area:.2g}"


def measure_text(ax, panel, width, face):
    for i, (code, (color, marker, _, _)) in enumerate(zip(STATE_CODES, STATES)):
        center = (i + 0.5) / len(STATES)
        ax.plot([center - .068], [0.43], linestyle="none", marker=marker,
                color=color, markersize=7, markeredgewidth=.8 if marker == "x" else 0,
                transform=ax.transAxes)
        ax.text(center - .036, 0.43, fmt_area(panel.area(code)), va="center",
                color=INK, fontsize=width, fontproperties=face, transform=ax.transAxes)


def fonts(language, chinese_file):
    families = ["DejaVu Sans"]
    if language == "zh":
        fontManager.addfont(chinese_file)
        families.append(FontProperties(fname=chinese_file).get_name())
    return FontProperties(family=families), FontProperties(family=families, weight="bold")


def read_registered_metrics(path):
    names = ("p_scene_05cm", "r_scene_05cm", "F1_scene_05cm")
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        all_rows = {row["panel"]: row for row in reader}
    stage_data = {}
    for index, panel in enumerate(TIMELINE):
        row = all_rows[panel]
        if int(row["actual_paid"]) != (256, 320, 512)[index]:
            raise ValueError(f"Invalid recorded action clock for {panel}")
        values = [[float(row[f"{name}_{arm}"]) for name in names] for arm in ("G", "S")]
        if not all(np.isfinite(value) and 0 <= value <= 1 for arm in values for value in arm):
            raise ValueError(f"Invalid registered metric values for {panel}")
        stage_data[panel] = values
    return stage_data


def draw_figure(panels, labels, frame, timeline_metrics, output, face, bold):
    count = len(panels)
    with_timeline = timeline_metrics is not None
    fig = plt.figure(figsize=(12.7, 6.65), facecolor="white")
    fig.text(.055, .958, labels["title_timeline" if with_timeline else "title_terminal"],
             fontsize=19, color=INK, va="top", fontproperties=bold)
    if with_timeline:
        fig.text(.945, .952, labels["timeline_note"], fontsize=11, color=MUTED,
                 ha="right", va="top", fontproperties=face)
    legend_face = face.copy()
    legend_face.set_size(11)
    fig.legend(legend_handles(), labels["states"], ncol=4, loc="upper left",
               bbox_to_anchor=(.05, .907), frameon=False, columnspacing=1.54,
               borderaxespad=0, handletextpad=.5,
               labelcolor=INK, prop=legend_face)
    fig.add_artist(Line2D([.055, .945], [.849, .849], transform=fig.transFigure,
                          color=RULE, linewidth=.7))
    fig.text(.055, .765, labels["orthographic"], fontsize=10.5, color=MUTED,
             fontproperties=face)
    fig.text(.055, .44, labels["floor"], fontsize=10.5, color=MUTED,
             fontproperties=face)
    fig.text(.055, .138, labels["area"], fontsize=10, color=MUTED, fontproperties=face)
    if with_timeline:
        fig.text(.055, .081, labels["metrics"], fontsize=9, color=MUTED, fontproperties=face)
    columns = labels["time_panels"] if with_timeline else labels["terminal_panels"]
    span = (.945 - .055) / count
    for index, (panel, label) in enumerate(zip(panels, columns)):
        left = .055 + index * span
        middle = left + span / 2
        fig.text(middle, .8, label, fontsize=13, color=INK, ha="center", fontproperties=bold)
        floor = panel.xyz[:, 2] == 0.0
        top_ax = fig.add_axes((left, .469, span, .28))
        floor_ax = fig.add_axes((left, .151, span, .28))
        draw_view(top_ax, project(panel.xyz[~floor]), panel.status[~floor], frame[0])
        draw_view(floor_ax, panel.xyz[floor, :2], panel.status[floor], frame[1])
        stats = fig.add_axes((left, .091, span, .043))
        stats.set_axis_off()
        measure_text(stats, panel, 9.5, face)
        if with_timeline:
            arms = timeline_metrics[TIMELINE[index]]
            numbers = [" / ".join(f"{100 * value:.2f}" for value in arm) for arm in arms]
            fig.text(middle, .067, f"G  {numbers[0]}     S  {numbers[1]}", fontsize=8.4,
                     va="center", ha="center", color=INK, fontproperties=face)
        if index == count - 1:
            draw_scale(floor_ax, frame[1], face)
    fig.add_artist(Line2D([.055, .945], [.050, .050], transform=fig.transFigure,
                          color=RULE, linewidth=.7))
    fig.text(.055, .027, labels["footnote"], fontsize=9.0, color=MUTED,
             va="center", fontproperties=face)
    for ext in ("pdf", "png"):
        fig.savefig(output.with_suffix(f".{ext}"), dpi=300, facecolor="white",
                    metadata={"Title": labels["title_timeline" if with_timeline else "title_terminal"]})
    plt.close(fig)


def draw_scale(ax, frame, face):
    (xmin, xmax), (ymin, ymax) = frame
    x = xmax - 2.45
    y = ymin + .055 * (ymax - ymin)
    ax.plot([x, x + 2.0], [y, y], color=INK, linewidth=1.05, solid_capstyle="butt")
    ax.plot([x, x], [y - .13, y + .13], color=INK, linewidth=.8)
    ax.plot([x + 2.0, x + 2.0], [y - .13, y + .13], color=INK, linewidth=.8)
    ax.text(x + 1.0, y + .4, "2 m", va="bottom", ha="center", color=INK,
            fontsize=8.5, fontproperties=face)
    north_x, north_tip = x + 1.0, ymax - 1.5
    ax.annotate('', xy=(north_x, north_tip), xytext=(north_x, north_tip - 1.5),
                arrowprops={'arrowstyle': '-|>', 'color': INK, 'linewidth': .8})
    ax.text(north_x, north_tip + .3, 'N', va='bottom', ha='center', color=INK,
            fontsize=8.5, fontproperties=face)


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=here / "spatial_pairs.npz")
    parser.add_argument("--metrics", type=Path, default=here / "spatial_pairs.csv")
    parser.add_argument("--output", type=Path, default=here / "figures")
    parser.add_argument("--chinese-font", type=Path, default=DEFAULT_CJK)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with np.load(args.data, allow_pickle=False) as archive:
        registered = set(archive["panels"].tolist())
        required = set(TERMINALS) | set(TIMELINE)
        if required != registered:
            raise ValueError(f"Panel universe differs from measurement archive: {registered ^ required}")
        terminal_panels = [Panel.read(archive, key) for key in TERMINALS]
        timeline_panels = [Panel.read(archive, key) for key in TIMELINE]
    if any(not np.array_equal(panel.xyz, timeline_panels[-1].xyz)
           or not np.array_equal(panel.weight_m2, timeline_panels[-1].weight_m2)
           for panel in timeline_panels):
        raise ValueError("Cycle 16 references differ across time checkpoints")
    frame = common_frame(terminal_panels)
    timeline_metrics = read_registered_metrics(args.metrics)
    plt.rcParams.update({"pdf.fonttype": 42, "axes.unicode_minus": False})
    for language, labels in COPY.items():
        face, bold = fonts(language, args.chinese_font)
        draw_figure(terminal_panels, labels, frame, None,
                    args.output / f"terminal_spatial_matching_{language}", face, bold)
        draw_figure(timeline_panels, labels, frame, timeline_metrics,
                    args.output / f"cycle16_spatial_matching_posthoc_{language}", face, bold)


if __name__ == "__main__":
    main()
