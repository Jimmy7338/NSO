"""Frozen paired geometry for the native RGB-D and planar-lidar simulator.

The simulator alone reads this source geometry. Each condition inherits one
parent room and changes only a suspended, unlabeled front occluder.
"""

import importlib.util
from pathlib import Path
import sys

import numpy as np
import trimesh


HERE = Path(__file__).resolve().parent
ANALYSES = HERE.parents[1]
SOURCE = ANALYSES / "paired_unseen_gain_20261120/scene/unseen_world.py"
BUDGET = 192
SEED = 202611228801
SHELF_LABEL = 31
BOARD_NODE = "front_occluder"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


sys.path.insert(0, str(ANALYSES.parent / "workspaces/nso_research_20261029/NSO"))
source = load_module("nso_occlusion_native_sensor_world", SOURCE)


ASSETS = {
    "dev_split": dict(width=1.23, depth=.56, layers=(.17, .61, 1.04, 1.49, 1.95),
                      divides=((1, 0.), (3, -.17)), back_height=1.96),
    "A_divided": dict(width=1.24, depth=.57, layers=(.18, .65, 1.13, 1.54, 1.98),
                      divides=((1, -.20), (2, .18), (3, -.03)), back_height=1.99),
    "B_tall_bay": dict(width=1.33, depth=.66, layers=(.16, .52, 1.15, 1.61, 2.02),
                        divides=((1, .16), (3, -.21)), back_height=2.03),
    "C_staggered": dict(width=1.17, depth=.62, layers=(.19, .71, 1.23, 1.68, 1.92),
                         divides=((0, 0.), (2, -.22), (2, .20)), back_height=1.93),
}


def parent(seed, dimensions, doors, shelf, start, asset, flank, board_width, cabinet):
    width, height, split_x, split_y = dimensions
    bottom, top, right, left = doors
    return dict(seed=seed, width_m=width, height_m=height, split_x=split_x, split_y=split_y,
                door_bottom_y=bottom, door_top_y=top, door_right_x=right, door_left_x=left,
                topology="cycle" if left is not None else "chain",
                shelf=shelf, start_xy=start, heading=0, asset=asset, flank=flank,
                board_width_m=board_width, cabinet=cabinet)


PARENTS = {
    "shelf_dev": parent(202611228801, (8.8, 7.6, 4.4, 3.8), (2.1, 5.8, 6.6, None),
                        (2.40, 2.92), (2.3, 1.1), "dev_split", (1.50, -.32), .96, (6.9, 2.8)),
    "shelf_01": parent(202611228821, (9.4, 8.0, 4.7, 3.9), (2.0, 6.3, 7.0, 2.2),
                        (2.55, 3.05), (2.5, 1.3), "A_divided", (1.50, -.33), .94, (7.7, 2.9)),
    "shelf_02": parent(202611228827, (10.2, 8.8, 5.15, 4.35), (2.7, 6.8, 8.0, None),
                        (2.75, 3.36), (2.1, 1.1), "B_tall_bay", (-1.65, -.32), 1.09, (7.9, 3.3)),
    "shelf_03": parent(202611228829, (8.8, 8.4, 4.2, 4.25), (2.5, 6.6, 6.9, 2.1),
                        (2.31, 3.35), (1.9, 1.3), "C_staggered", (1.30, -.35), .92, (6.65, 2.75)),
    "shelf_04": parent(202611228847, (10.6, 8.0, 5.4, 3.95), (2.2, 6.2, 8.4, None),
                        (3.15, 2.97), (2.9, 1.5), "B_tall_bay", (-1.68, -.34), 1.01, (8.5, 2.85)),
    "shelf_05": parent(202611228857, (9.8, 9.0, 5.0, 4.60), (3.0, 7.2, 7.7, 2.6),
                        (2.75, 3.58), (2.5, .9), "A_divided", (1.63, -.38), 1.02, (7.8, 3.7)),
    "shelf_06": parent(202611228869, (10.8, 8.8, 5.6, 4.25), (2.5, 7.0, 8.8, 3.5),
                        (3.36, 3.28), (3.1, 1.3), "C_staggered", (-2.05, -.36), .98, (8.4, 3.15)),
}
DEV_PARENTS = ("shelf_dev",)
TEST_PARENTS = tuple(name for name in PARENTS if name not in DEV_PARENTS)
CONDITIONS = ("open", "screen")
LAYOUTS = tuple(f"{unit}_{condition}" for unit in PARENTS for condition in CONDITIONS)
source.RAW = {name: PARENTS[name.rsplit("_", 1)[0]] | {"prefix_end_x": 3.10} for name in LAYOUTS}
SPECS = {}
for name in LAYOUTS:
    unit, condition = name.rsplit("_", 1)
    spec = source.layout_spec(name)
    del spec["sealed_prefixes"], spec["prefix_end_x"]
    spec["start"] = list(source.xy_cell(*spec["start_xy"], spec["height_m"]))
    spec["parent_id"] = unit
    spec["condition"] = condition
    spec["role"] = "development" if unit in DEV_PARENTS else "heldout"
    spec["fixture_source_sha256"] = source.sha(Path(__file__))
    SPECS[name] = spec
source.SPECS = SPECS
source.LAYOUTS = LAYOUTS
source.HERE = HERE


def _box(scene, meta, name, center, extents, color, instance=0, label=0, owner=0, asset=None):
    mesh = trimesh.creation.box(extents=extents, transform=trimesh.transformations.translation_matrix(center))
    mesh.visual.face_colors = np.asarray([*color, 255], dtype=np.uint8)
    scene.add_geometry(mesh, geom_name=name, node_name=name)
    meta[name] = dict(instance_id=instance, source_instance_id=instance, semantic_class=label, owner=owner,
                      kind="fixture" if instance else "architecture", source_asset=asset,
                      center_xyz=list(map(float, center)), extents_xyz=list(map(float, extents)))


def panel(layout):
    q = SPECS[layout]
    x, y = q["shelf"]
    # Only upper sight lines are shadowed. The panel is above the 1.20 m
    # collision band, and the unchanged planar scan lies below it.
    return (x, y - .74, 1.56), (q["board_width_m"], .08, .60)


def build_scene(layout):
    q = SPECS[layout]
    scene, meta = trimesh.Scene(), {}
    for name, primitive in [("floor", q["floor_box_xyz_size"])] + [
        (f"wall_{idx:02}", data) for idx, data in enumerate(q["wall_boxes_xyz_size"])
    ]:
        x, y, z, w, d, h = primitive
        color = (142, 145, 147) if name == "floor" else (205, 207, 209)
        _box(scene, meta, name, (x + w / 2, y + d / 2, z + h / 2), (w, d, h), color)

    x, y = q["shelf"]
    asset = ASSETS[q["asset"]]
    w, d, levels = asset["width"], asset["depth"], asset["layers"]
    frame_color, shelf_color, back_color = (85, 100, 115), (179, 146, 103), (124, 133, 142)

    def add_shelf(name, pos, extent, color):
        _box(scene, meta, f"shelf/{name}", pos, extent, color, SHELF_LABEL, 2, 1, q["asset"])

    for i, height in enumerate(levels):
        add_shelf(f"deck_{i:02}", (x, y, height), (w, d, .047), shelf_color)
    for i, (dx, dy) in enumerate(((-1, -1), (1, -1), (-1, 1), (1, 1))):
        add_shelf(f"post_{i:02}", (x + dx * (w / 2 - .028), y + dy * (d / 2 - .028), levels[-1] / 2),
                  (.056, .056, levels[-1]), frame_color)
    add_shelf("rear_wall", (x, y + d / 2 - .024, asset["back_height"] / 2),
              (w - .105, .042, asset["back_height"]), back_color)
    for i, (bay, dx) in enumerate(asset["divides"]):
        lo, hi = levels[bay] + .024, levels[bay + 1] - .024
        add_shelf(f"divider_{i:02}", (x + dx, y + .045, (lo + hi) / 2), (.038, d - .17, hi - lo), shelf_color)
    cx, cy = q["cabinet"]
    _box(scene, meta, "cabinet", (cx, cy, .50), (.97, .59, 1.0), (115, 127, 136),
         118, 3, 2, "constant_closed_box")
    manifests = [
        dict(slot="shelf", asset=q["asset"], source_instance_id=SHELF_LABEL, category=2, owner=1,
             source_nodes=[name for name in meta if name.startswith("shelf/")],
             object_front_world=[0., -1., 0.], controlled_center_xy=[x, y],
             extent_xyz_m=[w, d, asset["back_height"]], structure=asset),
        dict(slot="cabinet", asset="constant_closed_box", source_instance_id=118, category=3, owner=2,
             source_nodes=["cabinet"], controlled_center_xy=[cx, cy]),
    ]
    if q["condition"] == "screen":
        center, extents = panel(layout)
        _box(scene, meta, BOARD_NODE, center, extents, (171, 179, 192))
    return scene, manifests, meta


source.build_scene = build_scene


def make_world(layout):
    return source.make_world(layout, seed=SPECS[layout]["seed"],
                             config=source.ControlledConfig(max_steps=BUDGET))
