#!/usr/bin/env python3
"""Run the previous whole-floor actor with one optional regional score change."""

from dataclasses import replace
from math import inf
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT.parent / "region_structure_floor_20261122"
ARM_MODES = {"region_S": ("S", True), "revised_S": ("S", True), "F_strict": (None, False)}


def main():
    sys.path.insert(0, str(PREVIOUS))
    import formal_runner
    import regional_adapter
    import runner as native_runner

    original_install = regional_adapter.install

    def install(components, runtime, planner):
        if sys.argv[2] == "revised_S":
            assert planner.config.unknown_m2_cap == 2.0
            planner.config = replace(planner.config, unknown_m2_cap=inf)
        return original_install(components, runtime, planner)

    assert formal_runner.runner is native_runner
    native_runner.ROOT = ROOT
    native_runner.ARM_MODES = ARM_MODES
    regional_adapter.install = install
    formal_runner.start()


if __name__ == "__main__":
    main()
