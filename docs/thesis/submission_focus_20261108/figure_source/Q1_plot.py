"""Plot frozen Q1_curves.csv; this script does not execute a policy or access robot data."""
from pathlib import Path
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
with (root / "Q1_curves.csv").open(newline="") as stream:
    rows = list(csv.DictReader(stream))
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), layout="constrained")
for mode, color, style in [("G", "#0072B2", "-"), ("S", "#D55E00", "-"), ("X", "#009E73", "--")]:
    selected = [r for r in rows if r["mode"] == mode]
    x = [int(r["paid_action"]) for r in selected]
    axes[0].plot(x, [float(r["coverage_percent"]) for r in selected], label=mode, color=color, ls=style, lw=1.6)
    axes[1].plot(x, [float(r["new_reference_visible_surface_m2"]) for r in selected], label=mode, color=color, ls=style, lw=1.6)
for axis in axes:
    axis.set_xlabel("Paid actions")
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", alpha=.18)
    axis.set_xlim(0, 96)
axes[0].set_ylabel("Observed area (%)")
axes[1].set_ylabel("New visible surface (m²)")
axes[1].legend(frameon=False, ncol=3, loc="lower right")
fig.savefig(root / "Q1_curves.pdf")
fig.savefig(root / "Q1_curves.png", dpi=220)
