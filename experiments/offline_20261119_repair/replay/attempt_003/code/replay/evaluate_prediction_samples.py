#!/usr/bin/env python3
"""Evaluator-only predicted sample matching; no actor imports or execution."""
import hashlib,json,os
from pathlib import Path
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
import open3d as o3d
HERE=Path(__file__).resolve().parent
OUT=HERE/'attempt_003'
B=HERE.parents[1]/'dual_template_exploration_20261119'
TOLERANCE=.05
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ray_scene(path):
    mesh=o3d.io.read_triangle_mesh(str(path));scene=o3d.t.geometry.RaycastingScene(nthreads=1)
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh));return scene
def distance(scene,points):return scene.compute_distance(o3d.core.Tensor(np.asarray(points,np.float32)),nthreads=1).numpy()
def summarize(weights,source_distance,before,after):
    active=weights>0;real=source_distance<=TOLERANCE;was=before<=TOLERANCE;now=after<=TOLERANCE
    masks={'false_geometry':active&~real,'true_already_before':active&real&was,
        'true_previously_unmatched_now_matched':active&real&~was&now,
        'true_previously_unmatched_still_unrealized':active&real&~was&~now}
    assert np.array_equal(sum(v.astype(int)for v in masks.values()),active.astype(int))
    total=float(weights.sum());true=float(weights[active&real].sum());available=float(weights[active&real&~was].sum())
    rows={key:{'predicted_points':int(mask.sum()),'weighted_template_area_m2':float(weights[mask].sum()),
        'fraction_of_active_weight':float(weights[mask].sum()/total)if total>0 else None}for key,mask in masks.items()}
    return {'active_predicted_points':int(active.sum()),'total_weighted_template_area_m2':total,
        'source_supported_weighted_template_area_m2':true,'source_supported_weight_fraction':true/total if total>0 else None,
        'source_supported_previously_unmatched_weight_m2':available,
        'realization_fraction_of_source_supported_unmatched_weight':rows['true_previously_unmatched_now_matched']['weighted_template_area_m2']/available if available>0 else None,
        'partition':rows,'previously_correct_now_lost_weighted_template_area_m2':float(weights[active&real&was&~now].sum())}
def main():
    execution=json.loads((OUT/'result.json').read_text());assert execution['status']=='completed_same_domain_point_capture_replay'
    freeze=json.loads((OUT/'freeze.json').read_text())
    for path,h in freeze['snapshot_sha256'].items():assert sha(path)==h
    reference=B/'audit/references/shelf_left';receipt=json.loads((reference/'RECEIPT.json').read_text())
    source_path=reference/'source_mesh.ply';assert sha(source_path)==receipt['source_mesh_sha256'];source=ray_scene(source_path)
    original_replay=json.loads((HERE/'attempt_001/result.json').read_text());assert original_replay['final_native_TSDF_exact_bytes_match']
    captures=json.loads((OUT/'gain_hook_checks.json').read_text());results=[];input_hashes={str(source_path):sha(source_path)}
    for capture in captures:
        start=capture['paid_action'];end={54:109,109:127}[start];npz=Path(capture['predictions_path']);assert sha(npz)==capture['predictions_sha256']
        arrays=dict(np.load(npz,allow_pickle=False));input_hashes[str(npz)]=sha(npz)
        before_path=HERE/f'attempt_001/checkpoints/action_{start:03d}/native_tsdf_mesh.ply'
        after_path=HERE/f'attempt_001/checkpoints/action_{end:03d}/native_tsdf_mesh.ply'
        for path,action in [(before_path,start),(after_path,end)]:
            metrics=json.loads((path.parent/'metrics.json').read_text());assert sha(path)==metrics['native_mesh_sha256'];input_hashes[str(path)]=sha(path)
        before_scene=ray_scene(before_path);after_scene=ray_scene(after_path)
        item={'start_action':start,'end_action':end,'original_selected_route_actions':capture['outbound_actions'],'models':{}};saved={}
        assert end-start==capture['outbound_actions']
        for name in ['G','S']:
            points=arrays[name+'_points'];source_distance=distance(source,points);before=distance(before_scene,points);after=distance(after_scene,points)
            closest=source.compute_closest_points(o3d.core.Tensor(points.astype(np.float32)),nthreads=1)
            projected=closest['points'].numpy();before_projected=distance(before_scene,projected);after_projected=distance(after_scene,projected)
            old=arrays[name+'_original_union_gain'];new=arrays[name+'_repaired_union_gain'];assert np.all(new>=old)
            for key,value in {'source_distance_m':source_distance,'before_distance_m':before,'after_distance_m':after,'closest_source_points':projected,'before_projected_source_distance_m':before_projected,'after_projected_source_distance_m':after_projected,'closest_source_primitive_id':closest['primitive_ids'].numpy()}.items():saved[name+'_'+key]=value
            item['models'][name]={}
            for variant,weights in [('original',old),('repaired',new),('repair_increment',new-old)]:
                item['models'][name][variant]={
                    'primary_predicted_point_match':summarize(weights,source_distance,before,after),
                    'secondary_closest_true_surface_point_match':summarize(weights,source_distance,before_projected,after_projected)}
            item['models'][name]['predicted_point_geometry_sha256']=hashlib.sha256(points.tobytes()).hexdigest()
        np.savez_compressed(OUT/f'state_{start:03d}_evaluator_distances.npz',**saved);results.append(item)
    output={'status':'completed_same_domain_weighted_prediction_matching','threshold_m':TOLERANCE,'evaluation_only':True,'actor_ever_received_source_truth':False,
        'sample_capture_execution':str(OUT/'result.json'),'before_after_TSDF_execution':str(HERE/'attempt_001/result.json'),'separate_verified_replays':True,
        'repair_source_sha256':execution['repair_source_sha256'],'results':results,'input_sha256':input_hashes,'evaluator_sha256':sha(__file__),
        'definitions':{'primary':'A rewarded predicted point is geometry-supported if source-triangle distance<=5cm; before/after matching also uses that SAME predicted point and native TSDF triangles at5cm.',
            'secondary':'For the same source-supported predicted points, evaluate the nearest actual source-surface point against before/after TSDF, avoiding conflating a template offset with a true-surface offset.',
            'weights':'The original and repaired route-union gains are weighted hypothetical template areas, not independently sampled real surface areas. They can include alternatives and duplicate nearby surface support.',
            'repair_increment':'Per-point repaired gain minus original gain on identical model points; nonnegative by explicit assertion.',
            'partitions':'Active weight partitions into false geometry, true already before, true newly matched after, true still unrealized; lost-after among already-before is also retained.',
            'inference_limit':'Two predeclared saved-state route intervals in one original S trajectory; no repaired route execution, calibration generalization, independent scene replication or causal reconstruction benefit is inferred.'}}
    (OUT/'prediction_matching.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    for row in results:
        print('INTERVAL',row['start_action'],row['end_action'])
        for name in ['G','S']:
            for variant in ['original','repaired','repair_increment']:
                v=row['models'][name][variant]['primary_predicted_point_match'];print(name,variant,'total',v['total_weighted_template_area_m2'],'source_purity',v['source_supported_weight_fraction'],'new_matched',v['partition']['true_previously_unmatched_now_matched']['weighted_template_area_m2'],'false',v['partition']['false_geometry']['weighted_template_area_m2'])
if __name__=='__main__':main()
