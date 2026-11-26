"""Registered floors using the original hidden renderer and geometry builder."""

import importlib.util
import json
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
ANALYSES = HERE.parents[1]
PRIOR_SCENES = ANALYSES / "region_structure_floor_20261122/scenes"
ATTRIBUTION = ANALYSES / "region_global_revision_20261122/next/global_attribution/scenes"
SOURCE = ANALYSES / "paired_unseen_gain_20261120/scene/unseen_world.py"
REGISTRATION = HERE / "LAYOUT_REGISTRATION.json"
REGISTERED = json.loads(REGISTRATION.read_text())
HOLDOUT = REGISTERED["layouts"]
LAYOUTS = tuple(HOLDOUT)
BUDGET = REGISTERED["budget_actions"]
RESOLUTION_M = REGISTERED["resolution_m"]
RADIUS_M = REGISTERED["robot_radius_m"]
SEED = next(iter(HOLDOUT.values()))["seed"]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


geometry = load_module("independent_floor_geometry_contract", PRIOR_SCENES / "scene.py")
source = load_module("independent_floor_hidden_renderer", SOURCE)
SPECS = {name: geometry.floor_spec(name, record) for name, record in HOLDOUT.items()}
source.SPECS = SPECS
source.LAYOUTS = LAYOUTS
source.HERE = HERE


def make_world(layout):
    return source.make_world(
        layout, seed=SPECS[layout]["seed"], config=source.ControlledConfig(max_steps=BUDGET))


def build_scene(layout):
    return source.build_scene(layout)

