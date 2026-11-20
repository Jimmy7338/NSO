"""Read-only, post-hoc fit diagnosis from acquired points and frozen outputs."""
from pathlib import Path
import hashlib,importlib.util,json
import numpy as np

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    sources={};rows=[]
    for state in ('shelf_left_24','shelf_left_54','shelf_left_109','shelf_right_24','shelf_right_53','shelf_right_110'):
        inputs=ROOT/'replay/observed_prefixes'/state/'acquired_points.npz';data=np.load(inputs);sources[str(inputs)]=sha(inputs)
        predictions={}
        for stage in ('object_support','object_fit'):
            p=ROOT/'replay'/f'{state}_attempt_001'/f'{stage}_predictions.json';sources[str(p)]=sha(p)
            predictions[stage]={o['support_track']:o for o in json.loads(p.read_text())['model_audits']['S']['objects']}
        for track,o in predictions['object_support'].items():
            mask=data['all_support_track']==track
            if np.mean(data['all_label'][mask]==2)<.5:continue
            cloud=data['all_point'][mask];normals=data['all_normal'][mask]
            z=cloud[np.abs(normals[:,2])>.7,2]
            row=dict(state=state,track=int(track),observed_open_class_points=int(mask.sum()),
                horizontal_normal_candidate_points=len(z),observed_horizontal_z_m=z.tolist(),models={})
            for stage,objects in predictions.items():
                h=objects[track]['dims'][2];slabs=np.array([a*h+b for a in (.15,.5,.85) for b in (0.,.08)])
                distances=np.abs(z[:,None]-slabs).min(1) if len(z) else np.array([])
                row['models'][stage]=dict(height_m=h,slab_surface_z_m=slabs.tolist(),
                    horizontal_points_within_5cm=int(np.count_nonzero(distances<=.05)),
                    mean_nearest_slab_height_distance_m=float(distances.mean()) if len(z) else None)
            rows.append(row)
    summary=ROOT/'audit/summary.json';sources[str(summary)]=sha(summary)
    metrics=json.loads(summary.read_text())['equal_state_weight_aggregate']
    per_object={stage:metrics[stage]['models']['S']['mean_per_object'] for stage in ('object_support','object_fit')}
    p=ROOT/'method/geometry.py';spec=importlib.util.spec_from_file_location('frozen_geometry_diagnostic',p);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    data=np.load(ROOT/'replay/observed_prefixes/shelf_right_24/acquired_points.npz');mask=data['all_support_track']==1
    cloud=data['all_point'][mask];normals=data['all_normal'][mask]
    planes=module._vertical_planes(cloud,normals)
    diagonal=[dict(points=len(x['indices']),xy=cloud[x['indices'],:2].tolist(),
        unique_xy_at_1mm=len(np.unique(np.round(cloud[x['indices'],:2],3),axis=0)),
        normal=x['normal'].tolist(),rms_m=x['rms_m']) for x in planes]
    output=dict(schema='B5_posthoc_observed_fit_failure_diagnosis_v1',sources=sources,
        no_source_mesh_or_GT_dimensions_loaded=True,no_method_change=True,
        observed_horizontal_plane_proxy='abs(acquired normal z)>.7; includes candidate horizontal points, not independent validated planes',
        shelf_rows=rows,right24_candidate_planes=diagonal,independent_evaluator_S_mean_per_object=per_object,
        limitation='Height-only distances are an explanatory diagnostic, not 3D reconstruction accuracy or a causal ablation. Orientation, center, dimensions and conditional posterior also changed.')
    (HERE/'fit_failure_diagnosis.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({'right24_candidate_planes':diagonal,'S_mean_per_object':per_object},indent=2))

if __name__=='__main__':main()
