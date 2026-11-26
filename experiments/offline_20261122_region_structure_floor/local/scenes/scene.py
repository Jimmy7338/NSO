"""Prospective fixture-placement cases. Geometry belongs to the simulator only."""

import importlib.util
from pathlib import Path
import sys

import numpy as np
import trimesh


HERE = Path(__file__).resolve().parent
LOCAL = HERE.parent
ANALYSES = LOCAL.parents[1]
SOURCE = ANALYSES/'paired_unseen_gain_20261120/scene/unseen_world.py'
sys.path.insert(0, str(ANALYSES.parent/'workspaces/nso_research_20261029/NSO'))
spec = importlib.util.spec_from_file_location('local_inspection_fixture_source', SOURCE)
source = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = source
spec.loader.exec_module(source)

BUDGET = 192
SEED = 202611227116
RAW = {
    'east_visible': dict(width_m=9.6, height_m=8.4, split_x=4.8, split_y=4., topology='chain',
        door_bottom_y=2.3, door_top_y=6.3, door_right_x=7.1, door_left_x=None,
        shelf=[2.1,2.9,65.], cabinet=[7.5,5.5,-30.], screened=False, lateral_xy=[3.7,2.5], lateral_heading=3),
    'east_screened': dict(width_m=9.6, height_m=8.4, split_x=4.8, split_y=4., topology='chain',
        door_bottom_y=2.3, door_top_y=6.3, door_right_x=7.1, door_left_x=None,
        shelf=[2.1,2.9,65.], cabinet=[7.5,5.5,-30.], screened=True, lateral_xy=[3.7,2.5], lateral_heading=3),
    'west_visible': dict(width_m=10.2, height_m=9., split_x=5.2, split_y=4.4, topology='cycle',
        door_bottom_y=2.5, door_top_y=7.1, door_right_x=7.9, door_left_x=2.7,
        shelf=[2.7,3.3,-65.], cabinet=[8.3,2.9,40.], screened=False, lateral_xy=[.9,2.9], lateral_heading=1),
    'west_screened': dict(width_m=10.2, height_m=9., split_x=5.2, split_y=4.4, topology='cycle',
        door_bottom_y=2.5, door_top_y=7.1, door_right_x=7.9, door_left_x=2.7,
        shelf=[2.7,3.3,-65.], cabinet=[8.3,2.9,40.], screened=True, lateral_xy=[.9,2.9], lateral_heading=1),
}
LAYOUTS = tuple(RAW)
source.LAYOUTS = LAYOUTS
source.RAW = {key: dict(value, prefix_end_x=3.1) for key,value in RAW.items()}
SPECS = {key: source.layout_spec(key) for key in LAYOUTS}
for q in SPECS.values():
    del q['sealed_prefixes']
    del q['prefix_end_x']
    q['inspection_scene_source_sha256'] = source.sha(Path(__file__))
source.SPECS = SPECS
source.HERE = HERE
original_build = source.build_scene


def build_scene(layout):
    q = RAW[layout]
    scene, records, metadata = original_build(layout)
    if q['screened']:
        x,y,yaw = q['shelf']
        angle = np.deg2rad(yaw)
        normal = np.array([np.sin(angle),-np.cos(angle)])
        center = np.array([x,y])+.78*normal
        position = trimesh.transformations.translation_matrix([*center,.95])
        rotate = trimesh.transformations.rotation_matrix(angle, [0,0,1])
        screen = trimesh.creation.box(extents=[1.16,.09,1.86],transform=position@rotate)
        screen.visual.face_colors = np.array([189,194,201,255],np.uint8)
        scene.add_geometry(screen,geom_name='fixture_screen',node_name='fixture_screen')
        metadata['fixture_screen'] = dict(instance_id=0,source_instance_id=0,semantic_class=0,owner=0,kind='opaque_architecture')
    return scene, records, metadata


source.build_scene = build_scene


def make_world(layout):
    return source.make_world(layout,seed=SEED,config=source.ControlledConfig(max_steps=BUDGET))
