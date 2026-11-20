"""Geometrically identical v2 with matched visible-face tessellation.

The previous strict byte comparison differed only by one float32 ULP because
the same front plane used large vs small triangles. Keep that failed attempt;
this source-mesh revision makes the common faces literally identical triangles.
No sensor rounding, observation editing, model or policy change is made.
"""
import copy,hashlib,importlib.util,json,sys,time
from pathlib import Path
import numpy as np
import trimesh
HERE=Path(__file__).resolve().parent
V2=HERE.parent/'mechanism_v2'
spec=importlib.util.spec_from_file_location('B9_v2_physical_scene',V2/'scene.py')
base=importlib.util.module_from_spec(spec);sys.modules[spec.name]=base;spec.loader.exec_module(base)
parent=base.parent;PARENT=base.PARENT
SPEC=copy.deepcopy(base.SPEC)
SPEC.update(schema='B9_v2_matched_common_face_tessellation',
 mesh_revision='Split closed front plane at common corner-post boundaries using exact same triangles as open posts; physical surfaces and dimensions unchanged.',
 failed_predecessor=str(V2/'acceptance_attempt_001/result.json'))

def build_scene(layout):
    base.SPEC=SPEC
    world,records,metadata=base.build_scene(layout)
    for slot,(cx,cy)in SPEC['slots'].items():
        if SPEC['layouts'][layout][slot]!='closed':continue
        name=slot+'_closed_body';original=world.geometry[name]
        keep=original.face_normals[:,1]>-.9
        pieces=[trimesh.Trimesh(vertices=original.vertices.copy(),faces=original.faces[keep].copy(),process=False)]
        w,d,h=SPEC['object_width_m'],SPEC['object_depth_m'],SPEC['object_height_m'];p=SPEC['post_width_m'];t=SPEC['top_thickness_m']
        x,y=cx-w/2,cy-d/2
        for px,ww in ((x,p),(x+p,w-2*p),(x+w-p,p)):
            part=trimesh.creation.box(extents=[ww,p,h-t],transform=trimesh.transformations.translation_matrix([px+ww/2,y+p/2,(h-t)/2]))
            front=part.face_normals[:,1]<-.9
            pieces.append(trimesh.Trimesh(vertices=part.vertices.copy(),faces=part.faces[front].copy(),process=False))
        revised=trimesh.util.concatenate(pieces);revised.visual.face_colors=np.array([215,155,105,255],np.uint8)
        assert np.allclose(revised.bounds,original.bounds,rtol=0,atol=1e-14)
        assert np.isclose(revised.area,original.area,rtol=0,atol=1e-12)
        world.geometry[name]=revised
    return world,records,metadata

class MechanismWorld(parent.ControlledFunctionalWorld):
    def __init__(self,layout='shelf_left',config=None):
        parent.SPEC=SPEC;parent.build_scene=build_scene;parent.HERE=HERE
        super().__init__(config=config or parent.ControlledConfig(),seed=20261120,layout=layout)

def freeze():
    path=HERE/'scene_freeze.json';assert not path.exists()
    paths=[Path(__file__),V2/'scene.py',V2/'scene_freeze.json',V2/'acceptance_attempt_001/result.json',HERE/'protocol.json',
        V2.parent/'mechanism/scene.py',PARENT,parent.PREVIOUS/'scene_adapter/replicacad_world.py',parent.PREVIOUS/'assets/textured_raycast.py']
    v=dict(status='frozen_before_matched_tessellation_render_or_score',spec=SPEC,files_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest()for p in paths},created_unix_s=time.time(),policy_actions=0,policy_scores_computed=0)
    path.write_text(json.dumps(v,indent=2)+'\n');return v
def verify():
    v=json.loads((HERE/'scene_freeze.json').read_text());assert v['spec']==SPEC
    for p,h in v['files_sha256'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
    return v
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');a=p.parse_args()
    if a.freeze:print(json.dumps(freeze(),indent=2))
