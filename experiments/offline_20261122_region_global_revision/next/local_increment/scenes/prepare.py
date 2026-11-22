#!/usr/bin/env python3
"""Prepare native cold-state observations and independently sampled source references."""

import argparse
from scene import ANALYSES, LAYOUTS, SPECS, load_module


PREPARER = ANALYSES / "same_goal_route_choice_20261121/holdout/prepare.py"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("layout", choices=LAYOUTS)
    layout = parser.parse_args().layout
    inherited = load_module("nso_local_increment_native_preparer", PREPARER)
    inherited.SEED = SPECS[layout]["seed"]
    directory, reachable, config, metadata, unknown = inherited.initial(layout)
    inherited.reference(layout, directory, reachable, config, metadata)
    print(layout, "unknown reachable cells", unknown, "policy calls 0", flush=True)
