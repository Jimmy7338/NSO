#!/usr/bin/env python3
"""Geometric/sensor acceptance only, before any policy gains are computed."""
import hashlib,json,time,traceback
from pathlib import Path
import numpy as np
from PIL import Image
import scene
HERE=Path(__file__).resolve().parent

def digest(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def main():
    scene.verify();out=HERE/'acceptance_attempt_001';out.mkdir(exist_ok=False);started=time.monotonic();rows=[];packets={}
    try:
        from nso.observed_runtime_mapper_v10 import ObservedRuntimeMapperV10
        for layout in scene.SPEC['layouts']:
            world=scene.MechanismWorld(layout);frame=world.sense();scan=world.scan();owner=world.sense_visible_instances()
            folder=out/layout;folder.mkdir();frame.save(folder/'initial_frame.npz');scan.save(folder/'initial_scan.npz')
            np.savez_compressed(folder/'initial_owner.npz',visible_owner=owner)
            Image.fromarray(frame.color_rgb).save(folder/'initial_rgb.png')
            mapper=ObservedRuntimeMapperV10(world.shape,world.config);mapper.update(frame,scan)
            reachable=world.reachable;known=int(np.sum((mapper.belief!=-1)&reachable));total=int(reachable.sum())
            views=[]
            for slot,(cx,cy)in scene.SPEC['slots'].items():
                for side,(x,y)in dict(left=(cx-.85,cy),right=(cx+.85,cy),back=(cx,cy+.85)).items():
                    cell=(int(round(world.shape[0]-y/.2-.5)),int(round(x/.2-.5)))
                    views.append(dict(slot=slot,side=side,cell=list(cell),xy=list(world.xy(cell)),reachable=bool(reachable[cell])))
            meshes={}
            for record in world._records:
                parts=[]
                for node in record['source_mesh_nodes']:
                    T,name=world._source_scene.graph[node];mesh=world._source_scene.geometry[name].copy();mesh.apply_transform(T);parts.append(mesh)
                obj=scene.trimesh.util.concatenate(parts);meshes[record['slot']]=dict(family=record['family'],triangles=len(obj.faces),surface_area_m2=float(obj.area),vertices_sha256=digest(obj.vertices))
            item=dict(layout=layout,initial_depth_sha256=digest(frame.depth_m),owner_sha256=digest(owner),scan_sha256=digest(scan.ranges_m),
                depth_pixels=int((frame.depth_m>0).sum()),visible_owner_pixels={str(i):int((owner==i).sum())for i in(1,2)},
                current_footprint_conflict=bool(mapper.current_footprint_conflict),initial_known_reference_cells=known,
                reachable_reference_cells=total,initial_known_reference_fraction=known/total,unobserved_reachable_cells=total-known,
                side_back_views=views,objects=meshes,geometry_receipt=world.geometry_receipt,policy_actions=0,policy_scores=0)
            rows.append(item);packets[layout]=(frame,scan,owner)
        a,b=packets.values()
        parity=dict(depth_byte_identical=np.array_equal(a[0].depth_m,b[0].depth_m),
            owner_byte_identical=np.array_equal(a[2],b[2]),laser_byte_identical=np.array_equal(a[1].ranges_m,b[1].ranges_m),
            different_RGB_category_cues=not np.array_equal(a[0].color_rgb,b[0].color_rgb),
            different_category_labels=not np.array_equal(a[0].semantic,b[0].semantic))
        checks=dict(initial_geometry_and_masks_identical=all(parity[k]for k in('depth_byte_identical','owner_byte_identical','laser_byte_identical')),
            objects_observed=all(min(r['visible_owner_pixels'].values())>0 for r in rows),
            real_side_back_access=all(v['reachable']for r in rows for v in r['side_back_views']),
            incomplete_initial_map=all(r['unobserved_reachable_cells']>0 for r in rows),
            start_collision_free=all(not r['current_footprint_conflict']for r in rows),
            hidden_structures_differ=rows[0]['objects']['left']['vertices_sha256']!=rows[1]['objects']['left']['vertices_sha256'])
        scene.verify();write(out/'result.json',dict(status='accepted_geometry_only'if all(checks.values())else'failed_geometry_acceptance_retained',checks=checks,parity=parity,layouts=rows,
            no_policy_scores_computed=True,no_actor_actions=True,scene_freeze_sha256=hashlib.sha256((HERE/'scene_freeze.json').read_bytes()).hexdigest(),elapsed_s=time.monotonic()-started))
        print(json.dumps(dict(checks=checks,parity=parity,visible=[r['visible_owner_pixels']for r in rows],elapsed_s=time.monotonic()-started)),flush=True)
    except BaseException as e:
        write(out/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),layouts_completed=rows,elapsed_s=time.monotonic()-started));raise
if __name__=='__main__':main()
