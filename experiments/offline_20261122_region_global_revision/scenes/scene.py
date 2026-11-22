"""Source geometry for two prospectively declared global-policy validation floors."""

import importlib.util
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
ANALYSES = HERE.parents[1]
PRIOR_SCENES = ANALYSES / 'region_structure_floor_20261122' / 'scenes'
SOURCE = ANALYSES / 'paired_unseen_gain_20261120' / 'scene' / 'unseen_world.py'
BUDGET = 192
RESOLUTION_M = .2
RADIUS_M = .2
SEED = 202611226201

# These layouts are declared before any new strategy runs. They use the original
# assets and renderer with new building boundaries, doors, placements and seeds.
HOLDOUT = {
    'rev26_corridor_receiving': dict(
        topology='vertical_through_corridor_with_five_staggered_side_rooms',
        seed=202611226209,
        width_m=8.2,
        height_m=12.4,
        rooms={
            'lower_west': (0, 0, 3.0, 4.0),
            'middle_west': (0, 4.0, 3.0, 8.4),
            'upper_west': (0, 8.4, 3.0, 12.4),
            'hall': (3.0, 0, 5.6, 12.4),
            'lower_east': (5.6, 0, 8.2, 5.8),
            'upper_east': (5.6, 5.8, 8.2, 12.4),
        },
        doors=(
            ('hall', 'lower_west', 2.3, 1.4),
            ('hall', 'middle_west', 6.1, 1.6),
            ('hall', 'upper_west', 10.1, 1.4),
            ('hall', 'lower_east', 4.25, 1.6),
            ('hall', 'upper_east', 9.25, 1.4),
        ),
        start_xy=(4.3, 1.1), heading=0,
        shelf=(1.4, 7.15, 47), cabinet=(6.95, 10.6, -28),
    ),
    'rev26_loop_courtyard': dict(
        topology='offset_outer_cycle_with_an_inner_leaf',
        seed=202611226221,
        width_m=11.2,
        height_m=8.6,
        rooms={
            'south_west': (0, 0, 4.8, 2.8),
            'south_east': (4.8, 0, 11.2, 2.8),
            'west': (0, 2.8, 2.8, 5.6),
            'inner_gallery': (2.8, 2.8, 8.0, 5.6),
            'east': (8.0, 2.8, 11.2, 5.6),
            'north_west': (0, 5.6, 5.8, 8.6),
            'north_east': (5.8, 5.6, 11.2, 8.6),
        },
        doors=(
            ('south_west', 'south_east', 1.4, 1.4),
            ('south_east', 'east', 9.55, 1.6),
            ('east', 'north_east', 9.9, 1.4),
            ('north_east', 'north_west', 7.25, 1.6),
            ('north_west', 'west', 1.65, 1.6),
            ('west', 'south_west', 1.25, 1.4),
            ('west', 'inner_gallery', 4.25, 1.4),
        ),
        start_xy=(2.1, 7.7), heading=1,
        shelf=(5.6, 4.4, -33), cabinet=(8.4, 1.4, 15),
    ),
}
LAYOUTS = tuple(HOLDOUT)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


geometry = load_module('regional_revision_geometry_contract', PRIOR_SCENES / 'scene.py')
source = load_module('regional_revision_hidden_renderer', SOURCE)
SPECS = {name: geometry.floor_spec(name, record) for name, record in HOLDOUT.items()}
source.SPECS = SPECS
source.LAYOUTS = LAYOUTS
source.HERE = HERE


def make_world(layout):
    return source.make_world(
        layout, seed=SPECS[layout]['seed'], config=source.ControlledConfig(max_steps=BUDGET))


def build_scene(layout):
    return source.build_scene(layout)
