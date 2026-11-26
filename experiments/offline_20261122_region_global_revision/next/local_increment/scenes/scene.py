"""Independent near-field cases, with geometry confined to simulation/evaluation."""

import importlib.util
from pathlib import Path
import sys

import numpy as np
import trimesh


HERE = Path(__file__).resolve().parent
ANALYSES = HERE.parents[3]
SOURCE = ANALYSES / "paired_unseen_gain_20261120/scene/unseen_world.py"
BUDGET = 192
SEED = 202611226511


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


sys.path.insert(0, str(ANALYSES.parent / "workspaces/nso_research_20261029/NSO"))
source = load_module("nso_local_increment_fixture_renderer", SOURCE)
RAW = {
    "increment_open": dict(width_m=8.8, height_m=7.6, split_x=4.4, split_y=3.8,
        topology="chain", door_bottom_y=2.1, door_top_y=5.95, door_right_x=6.8, door_left_x=None,
        shelf=[2.05, 2.8, 48.], cabinet=[6.8, 5.6, -23.], screen=None, seed=202611226519),
    "increment_screen": dict(width_m=9.2, height_m=8.2, split_x=4.6, split_y=4.0,
        topology="cycle", door_bottom_y=2.4, door_top_y=6.4, door_right_x=7.5, door_left_x=2.5,
        shelf=[2.28, 2.85, -36.], cabinet=[7.1, 5.85, 28.],
        screen=[3.04, 2.91, 90., .86, .10, 1.36], seed=202611226531),
}
LAYOUTS = tuple(RAW)
source.LAYOUTS = LAYOUTS
source.RAW = {name: settings | {"prefix_end_x":3.1} for name, settings in RAW.items()}
SPECS = {name: source.layout_spec(name) for name in LAYOUTS}
for settings in SPECS.values():
    del settings["sealed_prefixes"]
    del settings["prefix_end_x"]
    settings["fixture_source_sha256"] = source.sha(Path(__file__))
source.SPECS = SPECS
source.HERE = HERE
original_build = source.build_scene


def build_scene(layout):
    scene, records, metadata = original_build(layout)
    screen = RAW[layout]["screen"]
    if screen is not None:
        x, y, angle, width, depth, height = screen
        place = trimesh.transformations.translation_matrix([x, y, height / 2])
        orient = trimesh.transformations.rotation_matrix(np.deg2rad(angle), [0, 0, 1])
        panel = trimesh.creation.box(extents=[width, depth, height], transform=place @ orient)
        panel.visual.face_colors = np.array([184, 188, 194, 255], dtype=np.uint8)
        scene.add_geometry(panel, geom_name="opaque_architecture", node_name="opaque_architecture")
        metadata["opaque_architecture"] = dict(
            instance_id=0, source_instance_id=0, semantic_class=0, owner=0, kind="opaque_architecture")
    return scene, records, metadata


source.build_scene = build_scene


def make_world(layout):
    return source.make_world(layout, seed=SPECS[layout]["seed"],
                             config=source.ControlledConfig(max_steps=BUDGET))
