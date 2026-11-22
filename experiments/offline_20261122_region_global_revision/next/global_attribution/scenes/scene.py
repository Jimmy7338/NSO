"""Cold corridor and cyclic-bypass layouts for the global attribution matrix."""

import importlib.util
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
ANALYSES = HERE.parents[3]
PRIOR_SCENES = ANALYSES / "region_structure_floor_20261122/scenes"
SOURCE = ANALYSES / "paired_unseen_gain_20261120/scene/unseen_world.py"
BUDGET = 192
RESOLUTION_M = 0.2
RADIUS_M = 0.2
SEED = 202611226671

HOLDOUT = {
    "attr26_corridor_crosspass": dict(
        topology="six_room_through_corridor_with_staggered_rooms_and_north_crosspass",
        seed=202611226673,
        width_m=11.6, height_m=8.0,
        rooms={
            "south_west": (0.0, 0.0, 3.0, 2.8),
            "south_middle": (3.0, 0.0, 7.0, 2.8),
            "south_east": (7.0, 0.0, 11.6, 2.8),
            "hall": (0.0, 2.8, 11.6, 5.0),
            "north_west": (0.0, 5.0, 5.4, 8.0),
            "north_east": (5.4, 5.0, 11.6, 8.0),
        },
        doors=(
            ("hall", "south_west", 1.5, 1.4),
            ("hall", "south_middle", 4.9, 1.6),
            ("hall", "south_east", 9.4, 1.4),
            ("hall", "north_west", 2.4, 1.6),
            ("hall", "north_east", 8.8, 1.6),
            ("north_west", "north_east", 6.6, 1.4),
        ),
        start_xy=(1.1, 3.9), heading=1,
        shelf=(7.9, 6.8, 16), cabinet=(4.8, 1.4, 24),
    ),
    "attr26_loop_bypass": dict(
        topology="seven_room_offset_cyclic_network_with_central_hub_and_parallel_south_bypasses",
        seed=202611226679,
        width_m=11.4, height_m=8.4,
        rooms={
            "south_west": (0.0, 0.0, 4.4, 2.8),
            "south_east": (4.4, 0.0, 11.4, 2.8),
            "west": (0.0, 2.8, 3.0, 5.2),
            "cross_hub": (3.0, 2.8, 8.0, 5.2),
            "east": (8.0, 2.8, 11.4, 5.2),
            "north_west": (0.0, 5.2, 6.2, 8.4),
            "north_east": (6.2, 5.2, 11.4, 8.4),
        },
        doors=(
            ("south_west", "south_east", 1.45, 1.4),
            ("south_west", "west", 1.4, 1.4),
            ("south_east", "cross_hub", 6.1, 1.4),
            ("west", "cross_hub", 3.95, 1.4),
            ("cross_hub", "east", 4.0, 1.4),
            ("east", "north_east", 9.85, 1.6),
            ("north_east", "north_west", 6.8, 1.6),
            ("north_west", "west", 1.5, 1.4),
            ("east", "south_east", 9.9, 1.4),
        ),
        start_xy=(1.3, 1.3), heading=1,
        shelf=(4.5, 7.0, -38), cabinet=(8.1, 1.4, -17),
    ),
}
LAYOUTS = tuple(HOLDOUT)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


geometry = load_module("global_attribution_geometry_contract", PRIOR_SCENES / "scene.py")
source = load_module("global_attribution_hidden_renderer", SOURCE)
SPECS = {name: geometry.floor_spec(name, record) for name, record in HOLDOUT.items()}
source.SPECS = SPECS
source.LAYOUTS = LAYOUTS
source.HERE = HERE


def make_world(layout):
    return source.make_world(
        layout, seed=SPECS[layout]["seed"], config=source.ControlledConfig(max_steps=BUDGET))


def build_scene(layout):
    return source.build_scene(layout)
