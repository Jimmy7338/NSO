"""Private procedural room inputs for the calibration renderer and evaluator."""
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import trimesh

HERE = Path(__file__).resolve().parent
B11 = HERE.parents[1] / 'paired_unseen_gain_20261120'


def load_world_api():
    path = B11 / 'scene/unseen_world.py'
    spec = importlib.util.spec_from_file_location('gain_calibration_local_world_api', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


API = load_world_api()
PROTOCOL = json.loads((HERE / 'protocol.json').read_text())
LAYOUTS = {entry['id']: entry for entry in PROTOCOL['layouts']}
RESOLUTION_M = 0.2


def get_spec(layout):
    row = LAYOUTS[layout]
    width, length = row['room']
    center_x, center_y = row['center']
    shape = [round(length / RESOLUTION_M), round(width / RESOLUTION_M)]
    start_y = PROTOCOL['movement']['start_y_m']
    start = list(API.xy_cell(center_x, start_y, length))
    rib_x = 0.20 if row['variant'] % 2 else width - 0.60
    architecture = [
        [0.0, 0.0, -0.02, width, length, 0.02],
        [0.0, 0.0, 0.0, width, 0.2, 2.5],
        [0.0, length-0.2, 0.0, width, 0.2, 2.5],
        [0.0, 0.2, 0.0, 0.2, length-0.4, 2.5],
        [width-0.2, 0.2, 0.0, 0.2, length-0.4, 2.5],
        [rib_x, 3.6, 0.0, 0.40, 1.05, 2.35],
    ]
    sign = row['occluder_sign']
    if sign:
        architecture.append([center_x + sign * .58 - .15, 2.92, 0.0, .30, .11, 1.57])
    return dict(row, width_m=width, height_m=length, shape=shape, start=start, heading=0,
                architecture=architecture, room_graph={'rooms': ['main'], 'edges': []},
                source_code_sha=sha256(Path(__file__).read_bytes()).hexdigest())


def boxes(row):
    width, depth, height = (row[key] for key in ('width', 'depth', 'height'))
    variant = row['variant']
    panel = .030 + .005 * (variant % 3)
    x, y = -width / 2, -depth / 2
    if row['family'] == 'closed':
        yield [x, y, 0.0], [width, depth, panel]
        yield [x, y, height-panel], [width, depth, panel]
        yield [x, y, panel], [panel, depth, height-2*panel]
        yield [x+width-panel, y, panel], [panel, depth, height-2*panel]
        yield [x+panel, y+depth-panel, panel], [width-2*panel, panel, height-2*panel]
        inner_width = width - 2 * panel
        for index, fraction in enumerate((.035, .11, .06)):
            recess = fraction * depth * (1 + (variant % 3) * .3)
            yield [x+panel+inner_width*index/3, y+recess, panel], [inner_width/3, panel, height-2*panel]
        return
    post = .044 + .009 * (variant % 3)
    for offset_x in (0, width-post):
        for offset_y in (0, depth-post):
            yield [x+offset_x, y+offset_y, 0.0], [post, post, height]
    for index, level in enumerate(np.linspace(.05, .92, 3 + variant % 3)):
        local_depth = depth * (0.61 + .10 * ((variant + index) % 4))
        setback = (depth-local_depth) * (0.13 + .15 * (index % 3))
        yield [x, y+setback, height*level], [width, local_depth, panel]
    if variant in (2, 7):
        yield [x+width*.34, y+depth-panel, height*.22], [width*.38, panel, height*.48]


def make_mesh(origin, size, color):
    mesh = trimesh.creation.box(extents=size, transform=trimesh.transformations.translation_matrix(
        np.asarray(origin) + np.asarray(size)/2))
    mesh.visual.face_colors = np.array([*color, 255], np.uint8)
    return mesh


def build_scene(layout):
    row = API.SPECS[layout]
    scene = trimesh.Scene()
    labels = {}
    for index, box in enumerate(row['architecture']):
        name = f'architecture_{index:02}'
        part = make_mesh(box[:3], box[3:], [173, 178, 181])
        scene.add_geometry(part, geom_name=name, node_name=name)
        labels[name] = {'semantic_class': 0, 'owner': 0}
    cx, cy = row['center']
    transform = trimesh.transformations.translation_matrix([cx, cy, 0]) @ (
        trimesh.transformations.rotation_matrix(np.deg2rad(row['yaw']), [0, 0, 1]))
    for index, (origin, size) in enumerate(boxes(row)):
        name = f'object_{index:02}'
        part = make_mesh(origin, size, [154, 106, 78])
        scene.add_geometry(part, geom_name=name, node_name=name, transform=transform)
        labels[name] = {'semantic_class': 2 if row['family'] == 'open' else 3, 'owner': 1}
    return scene, [], labels


# A privately imported instance supplies the B11 renderer and movement functions
# to the new fixtures. Its disk path and files are never modified.
API.HERE = HERE
API.SPECS = {layout: get_spec(layout) for layout in LAYOUTS}
API.build_scene = build_scene


def make_world(layout):
    return API.make_world(layout, seed=PROTOCOL['seed'])
