#!/usr/bin/env python3
"""Post-evaluation component attribution. Never changes or runs the policy."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'
import importlib.util
import json
import hashlib
from pathlib import Path
import sys
import numpy as np
import open3d as o3d

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
spec=importlib.util.spec_from_file_location('objectfit_auditor',HERE/'evaluate_objectfit.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
sys.path.insert(0,str(audit.SOURCE))
from nso.semantic_completion_v3 import ObjectCompletionModel

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def avg(rows):return float(np.mean(rows))

def parts(arr,obj,index):
    """Label the exact unchanged quadrature order, checking generated points."""
    points=arr['G_points'];ids=arr['G_hypothesis_ids']
    c=np.asarray(obj['center']);d=np.asarray(obj['dims']);R=np.asarray(obj['rotation'])
    origin=c-d/2;x,y,z=origin;sx,sy,h=d
    labels=np.full(len(points),'',dtype='U32')
    generated,_=ObjectCompletionModel._box(origin,d)
    generated=(np.asarray(generated)-c)@R.T+c
    np.testing.assert_allclose(generated,points[ids==2*index],rtol=0,atol=1e-12)
    labels[ids==2*index]='box'
    rows=np.flatnonzero(ids==2*index+1);offset=0
    def consume(label,where,size):
        nonlocal offset
        p,_=ObjectCompletionModel._box(where,size);p=(np.asarray(p)-c)@R.T+c
        selected=rows[offset:offset+len(p)]
        np.testing.assert_allclose(p,points[selected],rtol=0,atol=1e-12)
        labels[selected]=label;offset+=len(p)
    for level in (.15,.5,.85):consume('shelf_slabs',[x,y,z+level*h],[sx,sy,.08])
    for dx in (0,sx-.08):
        for dy in (0,sy-.08):consume('shelf_pillars',[x+dx,y+dy,z],[.08,.08,h])
    assert offset<len(rows)
    labels[rows[offset:]]='shelf_backboard'
    return labels

def main():
    evaluator=audit.Evaluator();result=[];source_rows={};inputs={}
    for directory in sorted((ROOT/'replay').glob('*_attempt_001')):
        manifest_path=directory/'manifest.json';manifest=json.loads(manifest_path.read_text())
        layout=manifest['layout'];inputs[str(manifest_path)]=sha(manifest_path)
        source=evaluator.core.reference(layout)
        reference=evaluator.reference_objects(layout)
        records=source['receipt']['source_object_records']
        observed=dict(np.load(manifest['observed_support']['path'],allow_pickle=False))
        observed_metadata=json.loads(Path(manifest['observed_support_metadata']['path']).read_text())
        for key in ('observed_support','observed_support_metadata'):
            inputs[manifest[key]['path']]=manifest[key]['sha256']
        if layout not in source_rows:
            helper=sys.modules['objectfit_frozen_source_mesh_builder']
            mesh,ids,_,_=helper.source_mesh(layout)
            p=np.asarray(mesh.vertices)[np.asarray(mesh.triangles)]
            cross=np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0]);area=np.linalg.norm(cross,axis=1)/2
            nz=np.abs(cross[:,2])/np.maximum(2*area,1e-12)
            records_out=[]
            for record in records:
                ident=record['source_instance_id'];selected=(ids==ident)&(nz>.99)
                levels={}
                for z,a in zip(p[selected,:,2].mean(1),area[selected]):
                    key=str(round(float(z),3));levels[key]=levels.get(key,0)+float(a)
                records_out.append(dict(owner=record['owner'],source_instance=ident,
                    bounds=record['bounds_world_xyz_m'],horizontal_triangle_area_by_height=levels))
            source_rows[layout]=records_out
        before=evaluator.core.mesh(manifest['stages'][0]['before_mesh'],layout)
        stages={}
        for row in manifest['stages']:stages.setdefault(row['variant'],[]).append(row)
        for variant,rows in stages.items():
            arr,receipt=evaluator.prediction(rows[0]['predictions'],rows[0]['prediction_receipt'])
            pts=arr['G_points'];d_before=audit.old.distance(before['ray'],pts)
            objects=receipt['model_audits']['G']['objects'];masks={};geometry=[]
            for i,obj in enumerate(objects):
                labels=parts(arr,obj,i)
                # Runtime support tracks are association IDs, not source owner
                # IDs: infer attribution from acquired measured points only.
                basis=observed_metadata[rows[0]['fit_support_basis']]
                acquired=observed[basis['point_key']][observed[basis['track_key']]==obj['support_track']]
                hits={target:int(np.count_nonzero(audit.old.distance(ray,acquired)<=.05))for target,ray in reference.items()}
                target=max(hits,key=hits.get)
                assert hits[target]>0 and hits[target]>max(v for k,v in hits.items()if k!=target)
                ident=31 if target=='open31' else 118
                record=next(r for r in records if r['source_instance_id']==ident)
                for part in ('box','shelf_slabs','shelf_pillars','shelf_backboard'):
                    masks[(target,part)]=labels==part
                lo,hi=np.asarray(record['bounds_world_xyz_m']);c=np.asarray(obj['center']);d=np.asarray(obj['dims'])
                geometry.append(dict(track=obj['support_track'],true_object=target,
                    acquired_point_source_hits=hits,
                    center_offset_from_true_bounds_m=(c-(lo+hi)/2).tolist(),
                    dimensions_minus_true_axis_extent_m=(d-(hi-lo)).tolist(),
                    predicted_slab_bottoms_m=[float(c[2]-d[2]/2+k*d[2])for k in (.15,.5,.85)],
                    geometry_fit_audit=obj.get('geometry_fit_audit')))
            distances={k:audit.old.distance(ray,pts)for k,ray in reference.items()}
            totals={name:{key:[]for key in masks}for name in ('G','S')}
            for row in rows:
                cid=row['candidate_id'];after=evaluator.core.mesh(row['after_mesh'],layout)
                d_after=audit.old.distance(after['ray'],pts)
                for name in ('G','S'):
                    gains=arr[f'{name}_gain_{cid}']
                    for key,mask in masks.items():
                        true=distances[key[0]]<=.05;real=true&(d_before>.05)&(d_after<=.05)
                        totals[name][key].append(dict(predicted=float(gains[mask].sum()),
                            supported=float(gains[mask&true].sum()),realized=float(gains[mask&real].sum())))
            result.append(dict(layout=layout,state_action=rows[0]['state_action'],variant=variant,
                geometry=geometry,models={name:{target:{part:{k:avg([r[k]for r in values])for k in ('predicted','supported','realized')}
                    for (obj,part),values in grouped.items()if obj==target}
                    for target in ('open31','closed118')}for name,grouped in totals.items()}))
    aggregate={}
    for variant in sorted(set(r['variant']for r in result)):
        states=[r for r in result if r['variant']==variant];assert len(states)==6
        aggregate[variant]={name:{target:{part:{field:avg([r['models'][name][target][part][field]for r in states])
                    for field in ('predicted','supported','realized')}
                for part in ('box','shelf_slabs','shelf_pillars','shelf_backboard')}
            for target in ('open31','closed118')}for name in ('G','S')}
    inputs.update(evaluator.core.inputs)
    for p,h in inputs.items():assert sha(p)==h,p
    out=dict(status='completed_readonly_geometry_component_diagnostic',states=result,
        equal_state_weight_aggregate=aggregate,source_geometry_evaluator_only=source_rows,input_sha256=inputs,
        diagnostic_script_sha256=sha(__file__),interpretation=[
            'Exact quadrature part order checked against frozen predictions. Truth is evaluator-only.',
            'Each component is an alternative-hypothesis weighted forecast, not unique physical area.',
            'Attribution is descriptive, not a causal counterfactual rerun; orientation, dimensions, weights and visibility co-vary.',
            'A point supported by another true object is excluded here: parts are matched to the acquired track source object.'])
    path=HERE/'geometry_component_diagnostic.json';path.write_text(json.dumps(out,indent=2)+'\n')
    print(str(path));print(json.dumps(aggregate))

if __name__=='__main__':main()
