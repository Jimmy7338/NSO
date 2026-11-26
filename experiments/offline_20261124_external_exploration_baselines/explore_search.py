"""Read-only bridge to the compiled upstream Explore Lite frontier search."""

from pathlib import Path
import subprocess

import numpy as np


BINARY = Path(__file__).resolve().parent / "work/explore/frontier_search"


def rank_frontiers(belief, resolution, position):
    rows, cols = belief.shape
    row, col = position
    header = f"{cols} {rows} {resolution} {col} {row}\n"
    data = " ".join(str(v) for v in np.asarray(belief).ravel())
    process = subprocess.run([str(BINARY)], input=header + data, text=True,
                             capture_output=True, check=True, timeout=5)
    results = []
    for line in process.stdout.splitlines():
        cost, y, x, *indices = line.split()
        cells = [(int(r), int(c)) for r, c in zip(indices[::2], indices[1::2], strict=True)]
        results.append((float(cost), (float(y), float(x)), cells))
    return results
