#!/usr/bin/env python3
"""Prepare zero-action sensor observations and the inherited offline reference."""

import argparse

from scene import ATTRIBUTION, LAYOUTS, load_module


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("layout", choices=LAYOUTS)
    args = parser.parse_args()
    inherited = load_module("independent_floor_inherited_preparer", ATTRIBUTION / "prepare_scenes.py")
    inherited.prepare(args.layout)

