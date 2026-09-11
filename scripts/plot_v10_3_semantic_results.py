#!/usr/bin/env python3
"""Create the thesis figure for the V10.3 paired development comparison."""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
run = ROOT / "eval_results/cpu_semantic_inspection_v10_3_development_20260911"
out = ROOT / "docs/research/figures"
rows = json.loads((run / "results.json").read_text())
contexts = [(c, a) for c in ("Q0", "Q1", "Q2", "Q3")
            for a in ("shelf_west", "shelf_east")]
lookup = {(r["context"], r["arrangement"], r["condition"]):
          r["new_area_times_final_f1_05cm"] for r in rows}
methods = ("S", "G", "N", "X")
colors = ("#0072B2", "#E69F00", "#009E73", "#D55E00")

fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.7), constrained_layout=True)
values = [[lookup[c, a, m] for c, a in contexts] for m in methods]
means = [np.mean(v) for v in values]
axes[0].bar(methods, means, color=colors, width=.68)
for i, vals in enumerate(values):
    axes[0].scatter(np.full(len(vals), i) + np.linspace(-.12, .12, len(vals)), vals,
                    color="black", s=13, alpha=.62, zorder=3)
axes[0].set_ylabel(r"New surface area $\times$ final F1@5cm")
axes[0].set_title("Eight paired industrial histories")
axes[0].grid(axis="y", alpha=.22)

for baseline, color, offset in (("G", colors[1], -.10), ("N", colors[2], .10)):
    diffs = [lookup[c, a, "S"] - lookup[c, a, baseline] for c, a in contexts]
    axes[1].scatter(np.arange(8) + offset, diffs, label=f"S − {baseline}",
                    color=color, s=31)
axes[1].axhline(0, color="black", linewidth=.8)
axes[1].set_xticks(range(8), [f"{c}\n{'W' if a.endswith('west') else 'E'}" for c, a in contexts])
axes[1].set_ylabel("Paired joint-metric difference")
axes[1].set_title("All contexts retained")
axes[1].legend(frameon=False)
axes[1].grid(axis="y", alpha=.22)

fig.suptitle("V10.3 semantic-conditioned coverage inspection", fontsize=12)
out.mkdir(parents=True, exist_ok=True)
fig.savefig(out / "v10_3_semantic_joint_results.png", dpi=220)
fig.savefig(out / "v10_3_semantic_joint_results.pdf")
