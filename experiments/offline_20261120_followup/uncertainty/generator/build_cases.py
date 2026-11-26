"""Construct private truth meshes independently from any NSO template builder."""
import numpy as np
import trimesh
from spec import SCENE


def transform(record):
    angle=np.deg2rad(record['yaw_deg']);c,s=np.cos(angle),np.sin(angle)
    matrix=np.eye(4);matrix[:3,:3]=[[c,-s,0],[s,c,0],[0,0,1]]
    matrix[:3,3]=SCENE['front_anchor_world']
    return matrix


def build_object(record):
    width,depth,height=(record[k] for k in ('width_m','depth_m','height_m'))
    t=record['panel_thickness_m'];parts=[]
    def panel(name,origin,size):
        assert min(size)>0
        parts.append((name,trimesh.creation.box(extents=size,
            transform=trimesh.transformations.translation_matrix(np.asarray(origin)+np.asarray(size)/2))))
    x=-width/2
    if record['family']=='closed':
        # Hollow enclosure, independently specified panel thickness and recessed front.
        panel('bottom',[x,0,0],[width,depth,t]);panel('top',[x,0,height-t],[width,depth,t])
        panel('left',[x,0,t],[t,depth,height-2*t]);panel('right',[x+width-t,0,t],[t,depth,height-2*t])
        panel('back',[x+t,depth-t,t],[width-2*t,t,height-2*t])
        if record.get('shape_regime')=='part_mismatch':
            segment_width=(width-2*t)/3
            for i,fraction in enumerate(record['door_segment_depth_fractions']):
                panel(f'stepped_front{i}',[x+t+i*segment_width,float(fraction)*depth,t],
                    [segment_width,t,height-2*t])
        else:
            panel('front_door',[x+t,record['door_recess_m'],t],[width-2*t,t,height-2*t])
    else:
        post=record['post_width_m']
        for ix in (0,1):
            for iy in (0,1):panel(f'post{ix}{iy}',[x+ix*(width-post),iy*(depth-post),0],[post,post,height])
        for i,f in enumerate(record['shelf_level_bottom_fractions']):
            level_depth=depth;lower=0.
            if record.get('shape_regime')=='part_mismatch':
                level_depth=depth*record['shelf_depth_fractions'][i]
                lower=(depth-level_depth)*record['shelf_setback_allocations'][i]
            panel(f'level{i}',[x,lower,f*height],[width,level_depth,t])
        if record['rear_brace']:
            panel('rear_horizontal_brace',[x,depth-t,.78*height],[width,t,.055])
        if record.get('shape_regime')=='part_mismatch':
            back_width=width*record['partial_back_width_fraction']
            panel('partial_back_panel',[-back_width/2,depth-t,
                height*record['partial_back_lower_fraction']],
                [back_width,t,height*record['partial_back_height_fraction']])
    matrix=transform(record)
    result=[]
    for name,mesh in parts:
        mesh.apply_transform(matrix);mesh.visual.face_colors=np.array([156,172,185,255],np.uint8)
        result.append((name,mesh))
    return result


def build_scene(record,condition):
    scene=trimesh.Scene();target_nodes=[]
    for name,mesh in build_object(record):
        key='target/'+name;scene.add_geometry(mesh,node_name=key,geom_name=key);target_nodes.append(key)
    boxes=[('floor',SCENE['floor'],[130,135,140]),
           *[(f'wall{i}',box,[190,190,190])for i,box in enumerate(SCENE['background_walls'])]]
    if condition=='occluded':boxes.append(('foreground_panel',SCENE['occluder'],[110,120,125]))
    for name,box,color in boxes:
        xyz=np.asarray(box[:3]);size=np.asarray(box[3:]);mesh=trimesh.creation.box(extents=size,
            transform=trimesh.transformations.translation_matrix(xyz+size/2))
        mesh.visual.face_colors=np.r_[color,255].astype(np.uint8)
        scene.add_geometry(mesh,node_name=name,geom_name=name)
    return scene,target_nodes


def private_truth(record):
    vertices=np.concatenate([mesh.vertices for _,mesh in build_object(record)])
    return dict(schema='B9_private_procedural_object_truth_v1',object_id=record['object_id'],
        source='independent procedural panel assembly; evaluator-only',construction=record,
        object_vertices_world=vertices.tolist(),world_rotation=transform(record)[:3,:3].tolist(),
        yaw_radians=float(np.deg2rad(record['yaw_deg'])),
        true_width_depth_height_m=[record[k]for k in ('width_m','depth_m','height_m')],
        bounds_world=[vertices.min(0).tolist(),vertices.max(0).tolist()],
        evaluation_rule='Project all actual world vertices onto inferred unit depth axis; also report nominal axis/yaw error. Never pass these records to inference.')
