"""Second declared geometry-only mechanism construction.

Preserves failed v1. Physical tapered screen openings reveal only common front
corner-post faces, increasing observed vertical support without model changes.
"""
import copy,hashlib,importlib.util,json,sys,time
from pathlib import Path
import numpy as np
import trimesh
HERE=Path(__file__).resolve().parent
V1=HERE.parent/'mechanism'
spec=importlib.util.spec_from_file_location('B9_scene_v1_reused_constructor',V1/'scene.py')
base=importlib.util.module_from_spec(spec);sys.modules[spec.name]=base;spec.loader.exec_module(base)
PARENT=base.PARENT;parent=base.parent
SPEC=copy.deepcopy(base.SPEC)
SPEC.update(schema='B9_common_top_and_corner_slits_scene_v2',post_width_m=.10,
 screen_slit_inset_m=.005,
 revision='Physical view-aligned screen apertures expose common10cm front corner posts; v1 support failure retained.',
 failed_predecessor=str(V1/'candidate_attempt_001/result.json'),
 design_selection='Screen/common-post support revision after declared v1 input failure, before v2 scores; not selected by winner. Original scene family/start/dimensions unchanged.')

def screen_segment(xfront,xback,yfront,yback,height):
    vertices=np.array([[xfront[0],yfront,0],[xfront[1],yfront,0],[xback[1],yback,0],[xback[0],yback,0],
                       [xfront[0],yfront,height],[xfront[1],yfront,height],[xback[1],yback,height],[xback[0],yback,height]])
    faces=np.array([[0,2,1],[0,3,2],[4,5,6],[4,6,7],[0,1,5],[0,5,4],[1,2,6],[1,6,5],[2,3,7],[2,7,6],[3,0,4],[3,4,7]])
    return trimesh.Trimesh(vertices=vertices,faces=faces,process=False)

def build_scene(layout):
    base.SPEC=SPEC
    scene,records,metadata=base.build_scene(layout)
    camera_x=(SPEC['start'][1]+.5)*.2;camera_y=(SPEC['shape'][0]-SPEC['start'][0]-.5)*.2
    for slot,(cx,cy)in SPEC['slots'].items():
        screen=slot+'_screen';scene.delete_geometry(screen);del metadata[screen]
        object_front=cy-SPEC['object_depth_m']/2;xlo=cx-SPEC['object_width_m']/2;xhi=cx+SPEC['object_width_m']/2
        p=SPEC['post_width_m'];inset=SPEC['screen_slit_inset_m']
        holes=[(xlo+inset,xlo+p-inset),(xhi-p+inset,xhi-inset)]
        ys=[SPEC['screen_front_y_m'],SPEC['screen_front_y_m']+SPEC['screen_thickness_m']]
        boundaries=[]
        for y in ys:
            alpha=(y-camera_y)/(object_front-camera_y)
            projections=[tuple(camera_x+alpha*(x-camera_x)for x in hole)for hole in holes]
            boundaries.append([(cx-SPEC['screen_width_m']/2,projections[0][0]),
                (projections[0][1],projections[1][0]),(projections[1][1],cx+SPEC['screen_width_m']/2)])
        for i,(a,b)in enumerate(zip(*boundaries)):
            mesh=screen_segment(a,b,*ys,SPEC['screen_height_m']);mesh.visual.face_colors=np.array([185,190,195,255],np.uint8)
            name=f'{slot}_screen_solid_{i}';scene.add_geometry(mesh,node_name=name,geom_name=name)
            metadata[name]=dict(owner=0,instance_id=0,source_instance_id=0,semantic_class=0,kind='physical_screen_with_common_post_aperture',component=name)
    return scene,records,metadata

class MechanismWorld(parent.ControlledFunctionalWorld):
    def __init__(self,layout='shelf_left',config=None):
        parent.SPEC=SPEC;parent.build_scene=build_scene;parent.HERE=HERE
        super().__init__(config=config or parent.ControlledConfig(),seed=20261120,layout=layout)

def freeze():
    path=HERE/'scene_freeze.json';assert not path.exists()
    paths=[Path(__file__),V1/'scene.py',V1/'scene_freeze.json',V1/'candidate_attempt_001/result.json',PARENT,HERE/'protocol.json',
        parent.PREVIOUS/'scene_adapter/replicacad_world.py',parent.PREVIOUS/'assets/textured_raycast.py']
    value=dict(status='frozen_before_v2_render_candidate_or_score',spec=SPEC,files_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest()for p in paths},created_unix_s=time.time(),policy_actions=0,policy_scores_computed=0)
    path.write_text(json.dumps(value,indent=2)+'\n');return value
def verify():
    v=json.loads((HERE/'scene_freeze.json').read_text());assert v['spec']==SPEC
    for p,h in v['files_sha256'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
    return v
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');a=p.parse_args()
    if a.freeze:print(json.dumps(freeze(),indent=2))
