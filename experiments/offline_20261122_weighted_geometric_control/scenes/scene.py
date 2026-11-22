"""Known development geometries and four prospectively specified holdout buildings."""

import importlib.util
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
ANALYSES = HERE.parents[1]
SOURCE = ANALYSES / 'paired_unseen_gain_20261120/scene/unseen_world.py'
sys.path.insert(0, str(ANALYSES.parent / 'workspaces/nso_research_20261029/NSO'))
spec = importlib.util.spec_from_file_location('weighted_original_building_source', SOURCE)
source = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = source
spec.loader.exec_module(source)

BUDGET = 192
SEED = 202611227001
DEVELOPMENT = ('chain_a', 'cycle_a')
HOLDOUT = {
    'paired_chain_17': dict(width_m=9.8, height_m=8.6, split_x=4.4, split_y=4.0, topology='chain',
        door_bottom_y=2.3, door_top_y=6.1, door_right_x=7.5, door_left_x=None,
        shelf=[2.9, 1.9, 35], cabinet=[7.4, 6.4, -15]),
    'paired_cycle_18': dict(width_m=10.8, height_m=8.8, split_x=5.8, split_y=4.4, topology='cycle',
        door_bottom_y=1.9, door_top_y=7.1, door_right_x=8.5, door_left_x=2.5,
        shelf=[8.2, 6.8, -35], cabinet=[3.6, 2.9, 55]),
    'paired_chain_19': dict(width_m=11.2, height_m=9.0, split_x=5.2, split_y=4.2, topology='chain',
        door_bottom_y=2.9, door_top_y=6.7, door_right_x=8.0, door_left_x=None,
        shelf=[8.75, 2.05, 70], cabinet=[3.15, 2.8, -45]),
    'paired_cycle_20': dict(width_m=10.4, height_m=9.6, split_x=5.0, split_y=4.8, topology='cycle',
        door_bottom_y=3.1, door_top_y=7.9, door_right_x=7.7, door_left_x=2.8,
        shelf=[3.3, 6.55, 15], cabinet=[3.25, 2.75, 120]),
}
LAYOUTS = (*DEVELOPMENT, *HOLDOUT)
source.RAW = {**{name: source.RAW[name] for name in DEVELOPMENT},
              **{name: dict(layout, prefix_end_x=3.1) for name, layout in HOLDOUT.items()}}
source.LAYOUTS = LAYOUTS
SPECS = {name: source.layout_spec(name) for name in LAYOUTS}
for layout in SPECS.values():
    del layout['sealed_prefixes']
    del layout['prefix_end_x']
source.SPECS = SPECS
source.HERE = HERE
build_scene = source.build_scene


def make_world(layout):
    return source.make_world(layout, seed=SEED, config=source.ControlledConfig(max_steps=BUDGET))
