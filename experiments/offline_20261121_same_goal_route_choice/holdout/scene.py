"""Unseen ordinary four-room scenes without a forced route or diagnostic junction."""

import importlib.util
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
ANALYSES = HERE.parents[1]
SOURCE = ANALYSES / "paired_unseen_gain_20261120/scene/unseen_world.py"
sys.path.insert(0, str(ANALYSES.parent / "workspaces/nso_research_20261029/NSO"))
spec = importlib.util.spec_from_file_location("nso_ordinary_multiroom_source", SOURCE)
source = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = source
spec.loader.exec_module(source)

BUDGET = 512
SEED = 202611211607
ORDINARY = {
    "ordinary_chain_13": dict(width_m=13.2, height_m=12.2, split_x=6.2, split_y=5.4,
        topology="chain", door_bottom_y=2.5, door_top_y=9.1, door_right_x=9.1, door_left_x=None,
        shelf=[3.3, 4.0, 20], cabinet=[10.4, 8.1, 90], prefix_end_x=9.1),
    "ordinary_cycle_14": dict(width_m=14.2, height_m=12.8, split_x=7.4, split_y=6.0,
        topology="cycle", door_bottom_y=3.9, door_top_y=8.3, door_right_x=11.1, door_left_x=4.1,
        shelf=[4.0, 2.7, -25], cabinet=[10.0, 9.2, 0], prefix_end_x=9.1),
    "ordinary_chain_15": dict(width_m=15.2, height_m=13.4, split_x=6.6, split_y=6.8,
        topology="chain", door_bottom_y=4.3, door_top_y=10.5, door_right_x=11.9, door_left_x=None,
        shelf=[12.2, 2.8, 90], cabinet=[4.1, 5.0, -35], prefix_end_x=9.1),
    "ordinary_cycle_16": dict(width_m=16.0, height_m=14.4, split_x=8.2, split_y=7.8,
        topology="cycle", door_bottom_y=5.7, door_top_y=11.7, door_right_x=12.3, door_left_x=5.7,
        shelf=[4.5, 4.8, 30], cabinet=[11.2, 10.0, 60], prefix_end_x=9.1),
}
LAYOUTS = tuple(ORDINARY)
source.RAW = ORDINARY
source.LAYOUTS = LAYOUTS
SPECS = {name: source.layout_spec(name) for name in LAYOUTS}
for scene in SPECS.values():
    del scene["sealed_prefixes"]
    del scene["prefix_end_x"]
source.SPECS = SPECS
source.HERE = HERE
build_scene = source.build_scene


def make_world(layout):
    return source.make_world(layout, seed=SEED, config=source.ControlledConfig(max_steps=BUDGET))
