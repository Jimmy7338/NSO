"""Two procedural warehouse/gallery parents, with an exact semantic-layout swap.

The simulator inherits B11 sensing and swept circular motion. Policy callers
take observations from this object. Source geometry and expected arm labels
are used only by the separate preflight and evaluation.
"""

from hashlib import sha256
import importlib.util
from pathlib import Path
import sys

import numpy as np
import trimesh

HERE = Path(__file__).resolve().parent
ANALYSES = HERE.parents[2]
PARENT = ANALYSES / 'paired_unseen_gain_20261120/scene/unseen_world.py'
WORKSPACE = ANALYSES.parent / 'workspaces/nso_research_20261029/NSO'
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))
spec = importlib.util.spec_from_file_location('nso_mechanism_b11_world', PARENT)
api = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = api
spec.loader.exec_module(api)

CELL = .2
SEED = 20261121
BUDGET = 256
ControlledConfig = api.ControlledConfig
LAYOUTS = ('warehouse_left', 'warehouse_right', 'gallery_left', 'gallery_right')

PARENTS = {
    'warehouse': dict(width_m=11., height_m=12.4, axis_x=5.5, start_y=1.5,
        pod_front_y=4.10, pod_center_offset=1.25, pod_width=1.30, pod_length=2.0,
        pod_height=1.58, divider_end_y=9.30, arm_offset=2.8, split_y=3.3,
        lane_test_y=5.1, rejoin_y=10.1, destination_y=11.5, style='transfer warehouse'),
    'gallery': dict(width_m=12.2, height_m=14.2, axis_x=6.1, start_y=1.7,
        pod_front_y=4.50, pod_center_offset=1.45, pod_width=1.54, pod_length=2.4,
        pod_height=1.66, divider_end_y=10.10, arm_offset=3.2, split_y=3.5,
        lane_test_y=5.5, rejoin_y=10.9, destination_y=13.3, style='exhibit and north annex'),
}


def cell(x, y, height):
    return api.xy_cell(x, y, height)


def architecture(q):
    width, height, axis = q['width_m'], q['height_m'], q['axis_x']
    thickness, wall_height = .2, 2.5
    walls = [
        [0, 0, -.02, width, height, .02],
        [0, 0, 0, width, thickness, wall_height],
        [0, height-thickness, 0, width, thickness, wall_height],
        [0, thickness, 0, thickness, height-2*thickness, wall_height],
        [width-thickness, thickness, 0, thickness, height-2*thickness, wall_height],
    ]
    gap_half = q['pod_center_offset']-q['pod_width']/2
    y = q['pod_front_y']
    walls.append([axis-gap_half, y, 0, 2*gap_half, q['divider_end_y']-y, wall_height])
    if q['parent'] == 'warehouse':
        # Two non-flush wall bays. They remain outside both clear corridors.
        walls += [[.2, 6.55, 0, .84, 1.55, 2.5], [width-1.04, 6.55, 0, .84, 1.55, 2.5]]
    else:
        # Narrow the transfer end of the central divider. The symmetric merger
        # precedes this additional common doorway to a second gallery.
        walls += [[axis-1.20, 7.12, 0, 2.40, 1.62, 2.5],
                  [.20, 11.62, 0, axis-1.40, .2, 2.5],
                  [axis+1.20, 11.62, 0, width-axis-1.40, .2, 2.5]]
    return walls


def layout_spec(layout):
    parent, shelf_side = layout.rsplit('_', 1)
    q = dict(PARENTS[parent])
    q.update(parent=parent, layout_id=layout, shelf_side=shelf_side, heading=0,
        start=list(cell(q['axis_x'], q['start_y'], q['height_m'])),
        shape=[round(q['height_m']/CELL), round(q['width_m']/CELL)],
        source_code_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
        room_graph={'rooms': ['sightline', 'left_lane', 'right_lane', 'transfer'] +
                    (['gallery_annex'] if parent == 'gallery' else []),
                    'edges': [['sightline', 'left_lane'], ['sightline', 'right_lane'],
                              ['left_lane', 'transfer'], ['right_lane', 'transfer']] +
                             ([['transfer','gallery_annex']] if parent == 'gallery' else [])})
    q['wall_boxes_xyz_size'] = architecture(q)
    q['route_evaluation'] = dict(
        north_gate_y_m=q['lane_test_y'], left_limit_x_m=q['axis_x']-.8,
        right_limit_x_m=q['axis_x']+.8, minimum_trigger_after_actions=0,
        expected_diagnostic_side=shelf_side,
        rule='First northbound trajectory crossing of gate classified by x; no crossing or inside the central band is unclassified. Use initial selected route states; actual first crossing reported separately. Expected diagnostic side is the outward aperture of the open shelf.')
    return q


SPECS = {name: layout_spec(name) for name in LAYOUTS}


def make_box(origin, extent, color):
    mesh = trimesh.creation.box(extents=extent, transform=trimesh.transformations.translation_matrix(
        np.asarray(origin) + np.asarray(extent)/2))
    mesh.visual.face_colors = np.array([*color, 255], np.uint8)
    return mesh


def unit_components(width, length, height, family):
    """Local coordinates. The near side is y=0 and the open face is x=0."""
    t = .06
    # The full height-matched plinth intercepts the planar scan and makes all
    # collision evidence independent of the shelves, partitions and door.
    yield 'plinth', [0, 0, 0, width, length, .32]
    yield 'south_side', [0, 0, .32, width, t, height-.32]
    yield 'north_side', [0, length-t, .32, width, t, height-.32]
    yield 'back', [width-t, t, .32, t, length-2*t, height-.32]
    yield 'top', [0, t, height-t, width-t, length-2*t, t]
    if family == 'closed':
        yield 'flat_door', [0, t, .32, t, length-2*t, height-.32-t]
        return
    for i, z in enumerate((.55, .94, 1.30)):
        yield f'interior_shelf_{i}', [0, t, z, width-t, length-2*t, .047]
    yield 'interior_partition', [.12, .51*length, .32, width-.12-t, .05, height-.32-t]


def build_scene(layout):
    q = SPECS[layout]
    scene, records, metadata = trimesh.Scene(), [], {}
    for i, box in enumerate(q['wall_boxes_xyz_size']):
        name = f'architecture_{i:02d}'
        scene.add_geometry(make_box(box[:3], box[3:], [164, 166, 168]), geom_name=name, node_name=name)
        metadata[name] = dict(instance_id=0, source_instance_id=0, semantic_class=0, owner=0, part=name)
    width, length, height = q['pod_width'], q['pod_length'], q['pod_height']
    for owner, side, sign in ((1, 'left', -1), (2, 'right', 1)):
        family = 'open' if side == q['shelf_side'] else 'closed'
        instance, category = (31, 2) if family == 'open' else (118, 3)
        center = q['axis_x'] + sign*q['pod_center_offset']
        min_x = center-width/2
        node_names = []
        for part, xyz_size in unit_components(width, length, height, family):
            x, y, z, sx, sy, sz = xyz_size
            if side == 'right':
                x = width-x-sx
            name = f'{side}/{part}'
            mesh = make_box([min_x+x, q['pod_front_y']+y, z], [sx, sy, sz], [157, 115, 79])
            scene.add_geometry(mesh, geom_name=name, node_name=name)
            metadata[name] = dict(instance_id=instance, source_instance_id=instance,
                                  semantic_class=category, owner=owner, part=part, slot=side)
            node_names.append(name)
        records.append(dict(slot=side, template=family, source_instance_id=instance,
            category=category, owner=owner, object_front_world=[sign, 0, 0],
            bounds_world_xyz_m=[[min_x,q['pod_front_y'],0],
                                [min_x+width,q['pod_front_y']+length,height]], source_mesh_nodes=node_names))
    return scene, records, metadata


api.HERE = HERE
api.SPECS = SPECS
api.build_scene = build_scene


def make_world(layout, seed=SEED, config=None):
    return api.make_world(layout, seed=seed, config=config or ControlledConfig(max_steps=BUDGET))
