"""Frozen symmetric occlusion scene; source geometry is simulator/evaluator only.

No planner/model is imported to design geometry. Reuses original motion and
sensor contracts without changing their collision or measurement rules.
"""
import hashlib, importlib.util, json, sys, time
from pathlib import Path
import numpy as np
import trimesh

HERE=Path(__file__).resolve().parent
PARENT=Path('/home/ouc/nso_robot/analyses/dual_template_exploration_20261119/scene/controlled_world.py')
spec=importlib.util.spec_from_file_location('B9_mechanism_parent_world',PARENT)
parent=importlib.util.module_from_spec(spec);sys.modules[spec.name]=parent;spec.loader.exec_module(parent)

SPEC=dict(schema='B9_common_top_screen_scene_v1',width_m=6.2,height_m=6.4,shape=[32,31],
 resolution_m=.2,robot_radius_m=.2,start=[26,15],heading=0,
 slots=dict(left=[1.6,4.2],right=[4.6,4.2]),object_width_m=.9,object_depth_m=.65,object_height_m=.9,
 top_thickness_m=.08,post_width_m=.06,shelf_bottoms_m=[.18,.48],shelf_thickness_m=.06,
 screen_front_y_m=3.60,screen_width_m=1.70,screen_thickness_m=.08,screen_height_m=.91,
 floor=[0,0,-.02,6.2,6.4,.02],walls=[[0,0,0,6.2,.2,2.4],[0,6.2,0,6.2,.2,2.4],
  [0,.2,0,.2,6,2.4],[6,.2,0,.2,6,2.4],[3.,2.5,0,.2,3.7,2.4]],
 layouts=dict(shelf_left=dict(left='open',right='closed'),shelf_right=dict(left='closed',right='open')),
 cue_conditions=['correct','neutral','swapped'],
 cue_source='Controlled category-coloured visible surfaces with ideal first-hit class channel; no natural detector claim.',
 unknown_exploration='Initial sensor frame only; original NSO candidate generator, common scores, budget and return rule.',
 object_evaluation='New procedural geometry assembled independently of predictor primitives; not natural scenes.',
 design_selection='Dimensions fixed before candidate generation, scores or branch outcomes. No winner-based scene edits.',
 prefix_actions=[],initial_sensor_only=True)

def build_scene(layout):
    if layout not in SPEC['layouts']:raise ValueError(layout)
    scene=trimesh.Scene();metadata={};records=[]
    def box(name,bounds,owner=0,family=None):
        x,y,z,w,d,h=bounds
        mesh=trimesh.creation.box(extents=[w,d,h],transform=trimesh.transformations.translation_matrix([x+w/2,y+d/2,z+h/2]))
        rgb=[115,160,210]if family=='open'else[215,155,105]if family=='closed'else[185,190,195]
        mesh.visual.face_colors=np.array([*rgb,255],np.uint8)
        scene.add_geometry(mesh,node_name=name,geom_name=name)
        metadata[name]=dict(owner=owner,instance_id=owner,source_instance_id=owner,
            semantic_class=(2 if family=='open'else 3 if family=='closed'else 0),
            kind='controlled_procedural_object'if owner else'architecture',component=name)
    box('floor',SPEC['floor'])
    for i,wall in enumerate(SPEC['walls']):box(f'wall_{i}',wall)
    for owner,(slot,center)in enumerate(SPEC['slots'].items(),1):
        family=SPEC['layouts'][layout][slot];cx,cy=center;w=SPEC['object_width_m'];d=SPEC['object_depth_m'];h=SPEC['object_height_m']
        x,y=cx-w/2,cy-d/2;t=SPEC['top_thickness_m'];p=SPEC['post_width_m']
        box(slot+'_screen',[cx-SPEC['screen_width_m']/2,SPEC['screen_front_y_m'],0,SPEC['screen_width_m'],SPEC['screen_thickness_m'],SPEC['screen_height_m']])
        box(slot+'_shared_top',[x,y,h-t,w,d,t],owner,family)
        if family=='closed':box(slot+'_closed_body',[x,y,0,w,d,h-t],owner,family)
        else:
            for i,px in enumerate((x,x+w-p)):
                for j,py in enumerate((y,y+d-p)):box(f'{slot}_post{i}{j}',[px,py,0,p,p,h-t],owner,family)
            for i,z in enumerate(SPEC['shelf_bottoms_m']):box(f'{slot}_layer{i}',[x,y,z,w,d,SPEC['shelf_thickness_m']],owner,family)
        records.append(dict(slot=slot,owner=owner,family=family,bounds_world_xyz_m=[[x,y,0],[x+w,y+d,h]],
            category=2 if family=='open'else 3,procedural=True,source_mesh_nodes=[n for n,m in metadata.items()if m['owner']==owner]))
    return scene,records,metadata

class MechanismWorld(parent.ControlledFunctionalWorld):
    def __init__(self,layout='shelf_left',config=None):
        # The parent class binds source construction through its own module.
        # These in-memory bindings are confined to this process and do not edit
        # any frozen parent file or cache.
        parent.SPEC=SPEC;parent.build_scene=build_scene;parent.HERE=HERE
        super().__init__(config=config or parent.ControlledConfig(),seed=20261120,layout=layout)
    def source_scene_for_evaluator(self):return self._source_scene,self._records,self._node_meta

def freeze():
    path=HERE/'scene_freeze.json'
    assert not path.exists()
    paths=[Path(__file__),PARENT,HERE/'protocol.json']
    paths += [Path(parent.PREVIOUS/'scene_adapter/replicacad_world.py'),Path(parent.PREVIOUS/'assets/textured_raycast.py')]
    data=dict(status='frozen_before_render_candidate_or_score',spec=SPEC,
        files_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest()for p in paths},
        created_unix_s=time.time(),policy_actions=0,policy_scores_computed=0)
    path.write_text(json.dumps(data,indent=2)+'\n')
    return data

def verify():
    data=json.loads((HERE/'scene_freeze.json').read_text())
    assert data['spec']==SPEC
    for p,h in data['files_sha256'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
    return data

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');args=p.parse_args()
    if args.freeze:print(json.dumps(freeze(),indent=2))
