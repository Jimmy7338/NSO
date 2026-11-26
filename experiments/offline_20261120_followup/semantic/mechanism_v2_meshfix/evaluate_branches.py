#!/usr/bin/env python3
"""Independent native-geometry surface evaluation for all mechanism branches."""
import hashlib,importlib.util,json,sys,time
from pathlib import Path
import numpy as np
import open3d as o3d
from scipy.stats import spearmanr
import scene
HERE=Path(__file__).resolve().parent
OLD=Path('/home/ouc/nso_robot/analyses/semantic_scene_redesign_20261119/exploration_evaluation/surface_reference.py')
spec=importlib.util.spec_from_file_location('B9_independent_surface_reference',OLD)
ref=importlib.util.module_from_spec(spec);sys.modules[spec.name]=ref;spec.loader.exec_module(ref)

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def load_resource(r):assert sha(r['path'])==r['sha256'];return read(r['path'])
def distance(mesh,points):
    return ref.ray_scene(mesh).compute_distance(o3d.core.Tensor(points.astype(np.float32)),nthreads=1).numpy()if len(mesh.triangles)else np.full(len(points),np.inf)

def reference(world,out):
    vertices=[];faces=[];labels=[];offset=0
    source,records,metadata=world.source_scene_for_evaluator()
    for node in source.graph.nodes_geometry:
        transform,name=source.graph[node];part=source.geometry[name]
        xyz=scene.trimesh.transform_points(part.vertices,transform);tri=np.array(part.faces,np.int32)
        vertices.append(xyz);faces.append(tri+offset);labels.extend([metadata[node]['owner']]*len(tri));offset+=len(xyz)
    mesh=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(np.concatenate(vertices)),o3d.utility.Vector3iVector(np.concatenate(faces)))
    sampling,label,duplicates=ref.unique_sampling_faces(mesh,np.array(labels));ray=ref.ray_scene(mesh)
    poses,cells=ref.fixed_camera_poses(world.reachable,world.config,world.start);refs={};meta={}
    for name,target,count,seed in [('all',None,100000,20261120),('left',1,10000,20261121),('right',2,10000,20261122)]:
        target_mesh=sampling if target is None else ref.subset_mesh(sampling,label==target)
        points,_=ref.surface_samples(target_mesh,count,seed)
        visible,first,rays=ref.visible_reference(points,ray,poses,world.config,tolerance_m=.001)
        refs[name]=points[visible].astype(np.float32)
        weight=target_mesh.get_surface_area()/len(points)
        meta[name]=dict(source_area_m2=float(target_mesh.get_surface_area()),drawn_samples=len(points),visible_samples=len(refs[name]),
            per_sample_area_m2=weight,visible_area_m2=weight*len(refs[name]),seed=seed,first_hit_rays=rays)
    np.savez_compressed(out/'reference.npz',**refs)
    o3d.io.write_triangle_mesh(str(out/'source_mesh.ply'),mesh)
    write(out/'reference_receipt.json',dict(metadata=meta,pose_count=len(poses),lattice_cells=cells,duplicate_handling=duplicates,
        observed_actor_trajectory_used=False,source_geometry_only_in_evaluator=True,source_reference_code_sha256=sha(OLD),
        scene_freeze_sha256=sha(HERE/'scene_freeze.json'),threshold_m=.05))
    return mesh,refs,meta

def main():
    scene.verify();out=HERE/'audit_attempt_001';out.mkdir(exist_ok=False);start=time.monotonic();results=[];selected=[]
    hashes={str(p):sha(p)for p in [Path(__file__),HERE/'branch_protocol.json',HERE/'scene_freeze.json',OLD]}
    write(out/'evaluation_freeze.json',dict(status='fixed_native_surface_evaluation',files_sha256=hashes))
    for layout in scene.SPEC['layouts']:
        folder=out/layout;folder.mkdir();world=scene.MechanismWorld(layout);source,refs,meta=reference(world,folder)
        manifest=read(HERE/'branches_attempt_001'/layout/'manifest.json');assert manifest['all_candidates_complete']
        before_file=manifest['stages'][0]['before_mesh'];assert sha(before_file['path'])==before_file['sha256']
        before=o3d.io.read_triangle_mesh(before_file['path']);before_d={n:distance(before,p)for n,p in refs.items()}
        rows=[]
        for stage in manifest['stages']:
            assert sha(stage['after_mesh']['path'])==stage['after_mesh']['sha256'];execution=load_resource(stage['execution'])
            after=o3d.io.read_triangle_mesh(stage['after_mesh']['path']);metrics={}
            for name,points in refs.items():
                a=before_d[name]<=.05;b=distance(after,points)<=.05;w=meta[name]['per_sample_area_m2']
                metrics[name]=dict(before_correct_area_m2=float(a.sum()*w),after_correct_area_m2=float(b.sum()*w),
                    net_correct_area_m2=float((b.sum()-a.sum())*w),new_correct_area_m2=float((b&~a).sum()*w),
                    lost_correct_area_m2=float((a&~b).sum()*w),final_recall=float(b.mean()))
            pred,_=ref.surface_samples(after,12000,812);accuracy=distance(source,pred)
            precision=float((accuracy<=.05).mean())if len(pred)else 0.;recall=metrics['all']['final_recall']
            row=dict(layout=layout,candidate_id=stage['candidate_id'],surface=metrics,
                object_net_area_m2=metrics['left']['net_correct_area_m2']+metrics['right']['net_correct_area_m2'],
                object_new_area_m2=metrics['left']['new_correct_area_m2']+metrics['right']['new_correct_area_m2'],
                global_net_area_m2=metrics['all']['net_correct_area_m2'],global_precision=precision,
                global_f1=2*precision*recall/(precision+recall)if precision+recall else 0.,
                completed_actions=execution['completed_actions'],path_length_m=execution['path_length_m'],collisions=execution['collisions'],endpoint_reached=execution['endpoint_reached'],
                correct_free_coverage_increment=execution['final_coverage']['correct_free_coverage_fraction']-execution['initial_coverage']['correct_free_coverage_fraction'],
                known_coverage_increment=execution['final_coverage']['coverage_fraction']-execution['initial_coverage']['coverage_fraction'],execution_sha256=stage['execution']['sha256'])
            rows.append(row);results.append(row)
        by={r['candidate_id']:r for r in rows};layouts=[]
        for T in (1,16):
            for cue in ('correct','neutral','swapped'):
                path=HERE/'candidate_attempt_002'/layout/f'T{T}_{cue}_predictions.json';receipt=read(path)
                actual=np.array([by[c['candidate_id']]['object_net_area_m2']for c in receipt['candidates']])
                chosen={}
                for group in ('G','S'):
                    cid=receipt[group+'_winner'];r=by[cid]
                    gain=np.array([c[group+'_gain']for c in receipt['candidates']])
                    rho=float(spearmanr(gain,actual).statistic)if np.ptp(gain)>0 and np.ptp(actual)>0 else None
                    chosen[group]=dict(candidate_id=cid,template_only_rank_rho=rho,**{k:r[k]for k in('object_net_area_m2','object_new_area_m2','global_net_area_m2','completed_actions','path_length_m','correct_free_coverage_increment','endpoint_reached','collisions')})
                row=dict(layout=layout,temperature=T,cue=cue,selected=chosen,
                    semantic_target_changed=receipt['G_winner']!=receipt['S_winner'],
                    S_minus_G_object_net_area_m2=chosen['S']['object_net_area_m2']-chosen['G']['object_net_area_m2'],
                    S_minus_G_actual_actions=chosen['S']['completed_actions']-chosen['G']['completed_actions'],
                    prediction_receipt_sha256=sha(path),branches_reused_not_independent=True)
                selected.append(row);layouts.append(row)
        write(folder/'result.json',dict(all_candidates=rows,all_conditions=layouts))
    assert hashes=={p:sha(p)for p in hashes};scene.verify()
    summary=dict(status='complete_all_native_branches_all_twelve_conditions',layouts=2,cue_conditions=3,temperatures=[1,16],
        actual_branches=len(results),actual_actions=sum(r['completed_actions']for r in results),branches=results,conditions=selected,
        no_full_exploration_claim=True,no_natural_detector_claim=True,all_negative_results_retained=True,
        cross_layout_float32_difference_disclosed=True,within_layout_shared_observation=True,elapsed_s=time.monotonic()-start)
    write(out/'summary.json',summary);print(json.dumps(dict(status=summary['status'],branches=len(results),conditions=selected)),flush=True)
if __name__=='__main__':main()
