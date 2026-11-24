"""Use upstream Explore Lite selection within the shared NSO simulation driver."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

from explore_search import BINARY, rank_frontiers


ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent / "region_independent_floors_20261123"
SOURCE = ROOT.parent / "region_structure_floor_20261122"
ARMS = ("F_strict", "region_S", "combined_S")
ARM = "ExploreLite_selection"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def explore_policy(policy_type, config, audit):
    from nso.frontier_policy import shortest_paths, trace_path
    from utils.grid_geometry import DIRECTIONS

    class ExploreLitePolicy(policy_type):
        def _select(self, obs, traversable):
            distances, parents = shortest_paths(traversable, obs.position)
            # Export only the observed costmap and the common footprint inflation.
            grid = np.where((obs.belief == 0) & ~traversable, 1, obs.belief)
            resolution = self.config.resolution_m
            clusters = rank_frontiers(grid, resolution, obs.position)
            row = dict(paid=obs.step, clusters=len(clusters), chosen=None)
            audit.append(row)
            for rank, (cost, center, members) in enumerate(clusters):
                # Centroids are typically in unknown space. Use the nearest
                # reachable cell adjacent to that same cluster in the shared grid.
                adjacent = {(r+dr, c+dc) for r, c in members for dr, dc in DIRECTIONS
                            if 0 <= r+dr < grid.shape[0] and 0 <= c+dc < grid.shape[1]
                            and grid[r+dr, c+dc] == 0 and distances[r+dr, c+dc] >= 0}
                def from_center(cell):
                    y, x = ((p+0.5)*resolution for p in cell)
                    return ((y-center[0])**2+(x-center[1])**2, int(distances[cell]), cell)
                for cell in sorted(adjacent, key=from_center):
                    gains = [self._gain_at(obs.belief, cell, h) for h in range(4)]
                    heading = min(range(4), key=lambda h: (-gains[h], min((h-obs.heading) % 4,
                                                                        (obs.heading-h) % 4), h))
                    if cell == obs.position and heading == obs.heading:
                        continue
                    self.goal, self.goal_heading = cell, heading
                    self._gain, self._distance, self._score = gains[heading], int(distances[cell]), -cost
                    self._path = trace_path(parents, obs.position, cell)
                    self.goal_count += 1
                    row["chosen"] = dict(native_rank=rank, native_cost=cost, native_frontier_size=len(members),
                                         native_centroid=center, executable_pose=[*cell, heading])
                    return True
            return False

    return ExploreLitePolicy(config)


def launch(layout, output):
    seal = json.loads((ROOT / "EXPLORE_PROTOCOL.json").read_text())
    assert layout in seal["layouts"]
    for reference, sha in seal["files_sha256"].items():
        assert digest((ROOT / reference).resolve()) == sha, reference
    assert not output.exists()
    sys.path.insert(0, str(BASE))
    sys.path.insert(0, str(SOURCE))
    import formal_runner
    assert Path(formal_runner.runner.__file__).resolve() == SOURCE / "runner.py"
    original = formal_runner.attach_frontier_receipts
    choices = []

    def attach(driver, runtime):
        driver.policy = explore_policy(type(driver.policy), driver.policy.config, choices)
        return original(driver, runtime)

    formal_runner.attach_frontier_receipts = attach
    spec = importlib.util.spec_from_file_location("frozen_floor_entry", BASE / "runner.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    try:
        # Keep the prior implementation's internal baseline dispatch label.
        # The separate output path and external receipt define the treatment.
        sys.argv = [__file__, layout, "F_strict", "--freeze", str(BASE / "FORMAL_FREEZE.json"),
                    "--out", str(output)]
        runner.main()
    finally:
        formal_runner.attach_frontier_receipts = original
    with (output / "explore_lite_receipt.json").open("x") as stream:
        json.dump({"arm": ARM, "layout": layout, "native_manifest_sha256": digest(output / "manifest.json"),
                   "external_protocol_sha256": digest(ROOT / "EXPLORE_PROTOCOL.json"),
                   "selection_calls": choices}, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("layout")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    launch(args.layout, args.out.resolve())
