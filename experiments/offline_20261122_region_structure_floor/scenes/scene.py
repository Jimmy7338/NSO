"""Fixed hidden-world definitions for the regional-structure floor experiment.

The simulator retains the established textured RGB-D, planar laser, current
visible instance labels, swept circular footprint, and charged motion. This
module is imported by the simulator fixture, never by the planning policy.
"""

import importlib.util
from itertools import combinations
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
ANALYSES = HERE.parents[1]
SOURCE = ANALYSES / 'paired_unseen_gain_20261120/scene/unseen_world.py'
ARCHIVED_SCENES = ANALYSES / 'weighted_geometric_control_20261122/scenes'
DEVELOPMENT = ('paired_cycle_18',)
BUDGET = 192
RESOLUTION_M = .2
RADIUS_M = .2
WALL_M = .2
HEIGHT_M = 2.4

# Room rectangles are given as (xmin, ymin, xmax, ymax). Entry locations are
# measured along their shared boundary. The geometry is specified before any
# policy outcome and incorporates no retry based on method performance.
HOLDOUT = {
    'floor26_ordinary': dict(
        topology='ordinary_offset_four_room_chain', seed=202611223501, width_m=9.6, height_m=7.8,
        rooms={'south_west': (0, 0, 4.0, 3.6), 'south_east': (4.0, 0, 9.6, 3.6),
               'north_west': (0, 3.6, 4.8, 7.8), 'north_east': (4.8, 3.6, 9.6, 7.8)},
        doors=(('south_west', 'south_east', 1.9, 1.4), ('south_west', 'north_west', 2.1, 1.2),
               ('south_east', 'north_east', 6.9, 1.4)),
        start_xy=(1.3, 1.3), heading=0, shelf=(2.7, 2.4, 20), cabinet=(7.3, 5.7, -15),
    ),
    'floor26_branch': dict(
        topology='five_spoke_hub', seed=202611223507, width_m=10.8, height_m=8.6,
        rooms={'west': (0, 0, 3.4, 8.6), 'east': (7.4, 0, 10.8, 8.6),
               'hub': (3.4, 2.6, 7.4, 5.8), 'south': (3.4, 0, 7.4, 2.6),
               'north_west': (3.4, 5.8, 5.4, 8.6), 'north_east': (5.4, 5.8, 7.4, 8.6)},
        doors=(('hub', 'west', 4.3, 1.4), ('hub', 'east', 4.5, 1.4), ('hub', 'south', 5.7, 1.6),
               ('hub', 'north_west', 4.3, 1.2), ('hub', 'north_east', 6.5, 1.2)),
        start_xy=(5.1, .9), heading=0, shelf=(1.5, 4.0, 60), cabinet=(9.0, 6.1, -35),
    ),
    'floor26_corridor': dict(
        topology='two_sided_long_corridor_with_side_connection', seed=202611223513, width_m=14.2, height_m=7.2,
        rooms={'south_west': (0, 0, 4.0, 2.6), 'south_middle': (4.0, 0, 9.0, 2.6),
               'south_east': (9.0, 0, 14.2, 2.6), 'hall': (0, 2.6, 14.2, 4.8),
               'north_west': (0, 4.8, 5.0, 7.2), 'north_middle': (5.0, 4.8, 10.4, 7.2),
               'north_east': (10.4, 4.8, 14.2, 7.2)},
        doors=(('hall', 'south_west', 2.3, 1.4), ('hall', 'south_middle', 6.3, 1.2),
               ('hall', 'south_east', 11.3, 1.6), ('hall', 'north_west', 2.7, 1.2),
               ('hall', 'north_middle', 7.7, 1.4), ('hall', 'north_east', 12.1, 1.4),
               ('south_middle', 'south_east', 1.3, 1.2)),
        start_xy=(1.1, 1.1), heading=0, shelf=(7.1, 6.0, 0), cabinet=(11.7, 1.3, 75),
    ),
    'floor26_loop': dict(
        topology='six_room_perimeter_loop', seed=202611223519, width_m=11.8, height_m=8.6,
        rooms={'south_west': (0, 0, 3.4, 4.2), 'south_middle': (3.4, 0, 7.4, 4.2),
               'south_east': (7.4, 0, 11.8, 4.2), 'north_west': (0, 4.2, 3.4, 8.6),
               'north_middle': (3.4, 4.2, 7.4, 8.6), 'north_east': (7.4, 4.2, 11.8, 8.6)},
        doors=(('south_west', 'south_middle', 2.3, 1.4), ('south_middle', 'south_east', 2.5, 1.2),
               ('south_east', 'north_east', 9.5, 1.6), ('north_east', 'north_middle', 6.7, 1.6),
               ('north_middle', 'north_west', 6.3, 1.4), ('north_west', 'south_west', 1.5, 1.2)),
        start_xy=(1.1, 1.3), heading=0, shelf=(1.8, 6.3, 95), cabinet=(5.1, 1.55, -80),
    ),
}
LAYOUTS = (*DEVELOPMENT, *HOLDOUT)


def load_module(name, path):
    loader = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(loader)
    sys.modules[name] = module
    loader.loader.exec_module(module)
    return module


def shared_boundary(a, b):
    x0, y0, x1, y1 = a
    u0, v0, u1, v1 = b
    if max(x0, u0) < min(x1, u1) and max(y0, v0) < min(y1, v1):
        raise ValueError('Room interiors overlap')
    if (x1 == u0 or u1 == x0) and max(y0, v0) < min(y1, v1):
        return 'vertical', max(x0, u0), max(y0, v0), min(y1, v1)
    if (y1 == v0 or v1 == y0) and max(x0, u0) < min(x1, u1):
        return 'horizontal', max(y0, v0), max(x0, u0), min(x1, u1)
    return None


def floor_spec(name, raw):
    width, height = raw['width_m'], raw['height_m']
    rooms = raw['rooms']
    if abs(sum((x1-x0)*(y1-y0) for x0,y0,x1,y1 in rooms.values())-width*height) > 1e-8:
        raise ValueError(f'Room rectangles must cover the floor: {name}')
    for x0, y0, x1, y1 in rooms.values():
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            raise ValueError(f'Invalid room rectangle: {name}')
    doors = {tuple(sorted((room, other))): (center, gap) for room, other, center, gap in raw['doors']}
    if len(doors) != len(raw['doors']):
        raise ValueError(f'Door pair repeated: {name}')
    walls = [[0,0,0,width,WALL_M,HEIGHT_M], [0,height-WALL_M,0,width,WALL_M,HEIGHT_M],
             [0,WALL_M,0,WALL_M,height-2*WALL_M,HEIGHT_M],
             [width-WALL_M,WALL_M,0,WALL_M,height-2*WALL_M,HEIGHT_M]]
    entries = []
    for left, right in combinations(sorted(rooms), 2):
        boundary = shared_boundary(rooms[left], rooms[right])
        gap = doors.get((left, right))
        if boundary is None:
            if gap is not None:
                raise ValueError(f'Door not on a shared boundary: {name}/{left}/{right}')
            continue
        axis, line, start, end = boundary
        outer_limit = height if axis == 'vertical' else width
        start = max(start, WALL_M)
        end = min(end, outer_limit-WALL_M)
        segments = [(start, end)]
        if gap is not None:
            center, clear_width = gap
            low, high = center-clear_width/2, center+clear_width/2
            if not start+.10 < low < high < end-.10 or clear_width <= 2*RADIUS_M:
                raise ValueError(f'Door too close to the wall end or narrower than the robot: {name}/{left}/{right}')
            segments = [(start, low), (high, end)]
            entries.append({'rooms': (left, right), 'axis': axis, 'wall_line_m': line,
                            'clear_interval_m': (low, high), 'clear_width_m': clear_width})
        for low, high in segments:
            if axis == 'vertical':
                walls.append([line-WALL_M/2,low,0,WALL_M,high-low,HEIGHT_M])
            else:
                walls.append([low,line-WALL_M/2,0,high-low,WALL_M,HEIGHT_M])
    start_x, start_y = raw['start_xy']
    return raw | {
        'layout_id': name, 'wall_boxes_xyz_size': walls, 'floor_box_xyz_size': [0,0,-.02,width,height,.02],
        'shape': [round(height/RESOLUTION_M), round(width/RESOLUTION_M)],
        'start': [int(round(height/RESOLUTION_M-.5-start_y/RESOLUTION_M)),
                  int(round(start_x/RESOLUTION_M-.5))],
        'resolution_m': RESOLUTION_M, 'robot_radius_m': RADIUS_M, 'architectural_doors': entries,
        'room_graph': {'rooms': list(rooms), 'edges': [list(entry['rooms']) for entry in entries]},
    }


sys.path.insert(0, str(ANALYSES.parent / 'workspaces/nso_research_20261029/NSO'))
archived = load_module('regional_development_weighted_world', ARCHIVED_SCENES/'scene.py')
source = load_module('regional_hidden_floor_fixture', SOURCE)
SPECS = {name: floor_spec(name, record) for name, record in HOLDOUT.items()}
source.SPECS = SPECS
source.LAYOUTS = tuple(HOLDOUT)
source.HERE = HERE
SEED = archived.SEED  # Prior-preparer compatibility. All fresh seeds are per layout.


def make_world(layout):
    if layout in DEVELOPMENT:
        return archived.make_world(layout)
    return source.make_world(layout, seed=SPECS[layout]['seed'], config=source.ControlledConfig(max_steps=BUDGET))


def build_scene(layout):
    return archived.build_scene(layout) if layout in DEVELOPMENT else source.build_scene(layout)
