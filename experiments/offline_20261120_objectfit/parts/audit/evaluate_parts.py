#!/usr/bin/env python3
"""B6 candidate evaluation against the frozen B5 full-support fitted model.

Only this independent evaluator accesses source object triangles. No actor is
imported or executed; source assembly is checked against the frozen reference.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import numpy as np
import open3d as o3d

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
V1=ROOT.parent/'dual_template_geometry_20261120'
B5=ROOT.parent/'dual_template_objectfit_20261120'
B=ROOT.parent/'dual_template_exploration_20261119'
SOURCE=ROOT.parent/'reproduction_20261104/simulation_publication_stage/experiments/unknown_exploration_20261104/source'
EVALUATOR=V1/'audit/evaluate_stage.py'
EVALUATOR_SHA='191f78ace67b6a04835a3d954acf0dd15f8d0c53438eb193c83eb7d05d2eed12'
B5_SUMMARY_SHA='3b03b9f497edfa5d056a92aacd3596b74a60b2216049b208b5cb88c2a935ef03'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod
assert sha(EVALUATOR)==EVALUATOR_SHA
old=load('objectfit_frozen_stage_evaluator',EVALUATOR)

def tracks_for_points(arrays,name,audit):
    ids=np.asarray(arrays[name+'_hypothesis_ids'],np.int64)
    objects=audit['objects']
    if len(ids):assert ids.min()>=0 and ids.max()<2*len(objects)
    lookup=np.asarray([int(obj['support_track'])for obj in objects],np.int64)
    return lookup[ids//2]if len(ids)else np.empty(0,np.int64)

class Evaluator:
    def __init__(self):
        self.core=old.Evaluator();self.refs={};self.arrays={};self.receipts={};self.distance_cache={}
        self.support_arrays={};self.support_metadata={};self.support_cache={};self.basis_identities={}
        p=B5/'audit/summary.json';assert sha(p)==B5_SUMMARY_SHA
        self.core.inputs[str(p)]=sha(p)
        self.baseline_states={(s['layout'],s['state_action']):s for s in json.loads(p.read_text())['states']}

    def reference_objects(self,layout):
        if layout not in self.refs:
            ref=self.core.reference(layout)
            for directory in (SOURCE,B/'audit'):
                if str(directory)not in sys.path:sys.path.insert(0,str(directory))
            helper_path=B/'audit/prepare_surface_reference.py'
            self.core.inputs[str(helper_path)]=sha(helper_path)
            helper=load('objectfit_frozen_source_mesh_builder',helper_path)
            mesh,labels,_,_=helper.source_mesh(layout)
            # Labels may be reused only if the assembled triangle ordering and
            # vertices agree with the pre-run source mesh, not a refitted truth.
            assert np.array_equal(np.asarray(mesh.triangles),np.asarray(ref['mesh'].triangles))
            np.testing.assert_allclose(np.asarray(mesh.vertices),np.asarray(ref['mesh'].vertices),rtol=0,atol=1e-12)
            scenes={}
            for name,identity in (('open31',31),('closed118',118)):
                selected=np.asarray(mesh.triangles)[labels==identity]
                part=o3d.geometry.TriangleMesh(mesh.vertices,o3d.utility.Vector3iVector(selected))
                assert len(part.triangles)>0
                scenes[name]=old.ray_scene(part)
            self.refs[layout]=scenes
        return self.refs[layout]

    def prediction(self,entry,receipt_entry):
        p=old.checked(entry,self.core.inputs);j=old.checked(receipt_entry,self.core.inputs)
        key=(entry['sha256'],receipt_entry['sha256'])
        if key not in self.arrays:
            self.arrays[key]=dict(np.load(p,allow_pickle=False));self.receipts[key]=json.loads(j.read_text())
        return self.arrays[key],self.receipts[key]

    def baseline(self,layout,paid):
        directory=B5/'replay'/f'{layout}_{paid}_attempt_001'
        evaluation=B5/'audit'/f'{layout}_{paid}'/'result.json'
        assert sha(evaluation)==self.baseline_states[(layout,paid)]['evaluation_sha256']
        self.core.inputs[str(evaluation)]=sha(evaluation)
        recorded=json.loads(evaluation.read_text())['input_sha256']
        receipt=directory/'object_fit_predictions.json'
        manifest_path=directory/'manifest.json'
        manifest=json.loads(manifest_path.read_text())
        assert sha(manifest_path)==json.loads(evaluation.read_text())['manifest_sha256']
        self.core.inputs[str(manifest_path)]=sha(manifest_path)
        entries=[s for s in manifest['stages']if s['variant']=='object_fit']
        assert entries and all(s['predictions']==entries[0]['predictions']for s in entries)
        assert sha(receipt)==recorded[str(receipt)]
        return self.prediction(entries[0]['predictions'],dict(path=str(receipt),sha256=recorded[str(receipt)]))

    def distances(self,layout,entry,points,before,after):
        key=(layout,entry['sha256'],sha(before),sha(after),hashlib.sha256(points.tobytes()).hexdigest())
        if key not in self.distance_cache:
            a=old.ray_scene(o3d.io.read_triangle_mesh(str(before)))
            z=old.ray_scene(o3d.io.read_triangle_mesh(str(after)))
            self.distance_cache[key]=dict(before=old.distance(a,points),after=old.distance(z,points),
                objects={name:old.distance(ray,points)for name,ray in self.reference_objects(layout).items()})
        return self.distance_cache[key]

    def attributed(self,points,weights,tracks,layout,entry,before,after):
        d=self.distances(layout,entry,points,before,after)
        scopes={'all_observed_object_tracks':tracks>0}
        scopes.update({f'track_{t}':tracks==t for t in sorted(set(tracks)-{0})})
        return {scope:{name:old.summarize_prediction(weights*mask,sd,d['before'],d['after'])
            for name,sd in d['objects'].items()}for scope,mask in scopes.items()}

    def validate_support(self,row,name,model):
        basis=row['fit_support_basis'];meta=self.support_metadata[basis]
        key=(row['prediction_receipt']['sha256'],name,basis)
        if key in self.support_cache:return self.support_cache[key]
        points=self.support_arrays[meta['point_key']]
        normals=self.support_arrays[meta['normal_key']]
        labels=self.support_arrays[meta['label_key']]
        tracks=self.support_arrays[meta['track_key']]
        assert model['observed_input_point_count']==len(points)
        assert model['object_support'] and not model['applicability'] and not model['free_space_likelihood']
        assert model['full_observed_support']==(basis=='full')
        assert model['applied_global_point_cap'] is None
        identities={}
        for obj in model['objects']:
            track=int(obj['support_track']);selected=tracks==track
            assert track>0 and obj['observed_points']==int(selected.sum())
            expected={
                'fit_input_points_sha256':hashlib.sha256(np.ascontiguousarray(points[selected],dtype=np.float64).tobytes()).hexdigest(),
                'fit_input_normals_sha256':hashlib.sha256(np.ascontiguousarray(normals[selected],dtype=np.float64).tobytes()).hexdigest(),
                'fit_input_labels_sha256':hashlib.sha256(np.ascontiguousarray(labels[selected],dtype=np.int64).tobytes()).hexdigest()}
            for field,value in expected.items():assert obj[field]==value,(track,field)
            identities[str(track)]=expected
        identity_key=(row['layout'],row['state_action'],basis)
        if identity_key in self.basis_identities:assert identities==self.basis_identities[identity_key]
        else:self.basis_identities[identity_key]=identities
        admitted=np.isin(tracks,[int(t)for t in identities])
        actual={}
        for target,ray in self.reference_objects(row['layout']).items():
            on_object=old.distance(ray,points)<=old.TOLERANCE
            marked=on_object&(tracks>0)
            kept=on_object&admitted
            actual[target]=dict(acquired_points_within_5cm=int(on_object.sum()),
                acquired_positive_track_points_within_5cm=int(marked.sum()),
                admitted_track_points_within_5cm=int(kept.sum()),
                positive_track_point_retention_fraction=float(kept.sum()/marked.sum())if marked.any()else None,
                all_measured_object_point_retention_fraction=float(kept.sum()/on_object.sum())if on_object.any()else None)
        input_tracks=sorted(set(map(int,tracks))-{0});admitted_tracks=sorted(map(int,identities))
        result=dict(basis=basis,input_points=len(points),input_positive_tracks=input_tracks,
            admitted_tracks=admitted_tracks,unadmitted_tracks=sorted(set(input_tracks)-set(admitted_tracks)),
            fit_input_hashes_verified=True,shared_support_identity=identities,
            source_object_measured_point_retention=actual,
            interpretation='Track admission and measured-object point retention are separate from future surface prediction recall and from detector accuracy.')
        self.support_cache[key]=result
        return result

    def stage(self,row,default_layout):
        result=self.core.stage(row,default_layout)
        layout,paid,cid=result['layout'],int(result['state_action']),int(result['candidate_id'])
        arrays,receipt=self.prediction(row['predictions'],row['prediction_receipt'])
        baseline,base_receipt=self.baseline(layout,paid)
        np.testing.assert_array_equal(arrays['G_points'],arrays['S_points'])
        np.testing.assert_array_equal(arrays['G_hypothesis_ids'],arrays['S_hypothesis_ids'])
        result['observed_object_attribution']={}
        for name in ('G','S'):
            points=arrays[name+'_points'];weights=arrays[name+f'_gain_{cid}']
            model=receipt['model_audits'][name];base_model=base_receipt['model_audits'][name]
            support_validation=self.validate_support(row,name,model)
            tracks=tracks_for_points(arrays,name,model);base_tracks=tracks_for_points(baseline,name,base_model)
            assert not np.any(tracks==0),'object-domain variant constructed an unassociated furniture hypothesis'
            own_tracks=sorted({int(o['support_track'])for o in model['objects']})
            prior_tracks=sorted({int(o['support_track'])for o in base_model['objects']if o['support_track']>0})
            assert own_tracks==prior_tracks,'B6 changed the baseline observed object set'
            baseline_objects={o['support_track']:o for o in base_model['objects']}
            for obj in model['objects']:
                for field in ('fit_input_points_sha256','fit_input_normals_sha256','fit_input_labels_sha256'):
                    assert obj[field]==baseline_objects[obj['support_track']][field],field
            assert row['baseline_predictions']['sha256']==sha(B5/'replay'/f'{layout}_{paid}_attempt_001/object_fit_predictions.npz')
            assert row['baseline_prediction_receipt']['sha256']==sha(B5/'replay'/f'{layout}_{paid}_attempt_001/object_fit_predictions.json')

            before=Path(row['before_mesh']['path']);after=Path(row['after_mesh']['path'])
            current=self.attributed(points,weights,tracks,layout,row['predictions'],before,after)
            base_path=B5/'replay'/f'{layout}_{paid}_attempt_001/object_fit_predictions.npz'
            baseline_metrics=self.attributed(baseline[name+'_points'],baseline[name+f'_gain_{cid}'],base_tracks,layout,
                dict(path=str(base_path),sha256=sha(base_path)),before,after)
            parity={}
            for track in sorted(set(own_tracks)|set(prior_tracks)):
                a,b=tracks==track,base_tracks==track
                shape_same=points[a].shape==baseline[name+'_points'][b].shape
                parity[str(track)]={
                    'points_equal':bool(shape_same and np.array_equal(points[a],baseline[name+'_points'][b])),
                    'history_weights_equal':bool(shape_same and np.allclose(arrays[name+'_weights'][a],baseline[name+'_weights'][b],rtol=0,atol=1e-12)),
                    'candidate_gain_equal':bool(shape_same and np.allclose(weights[a],baseline[name+f'_gain_{cid}'][b],rtol=0,atol=1e-12))}
            if row['variant']=='object_support':
                assert own_tracks==prior_tracks,'old-crop support stage changed eligible tracked object set'
                assert all(all(check.values())for check in parity.values()),'object-support stage changed retained-track forecast'
            result['observed_object_attribution'][name]=dict(admitted_tracks=own_tracks,baseline_admitted_tracks=prior_tracks,
                lost_baseline_tracks=sorted(set(prior_tracks)-set(own_tracks)),new_tracks=sorted(set(own_tracks)-set(prior_tracks)),
                same_track_forecast_parity=parity,current=current,B5_object_fit_same_track_baseline=baseline_metrics,
                support_validation=support_validation,
                object_audit=model['objects'],model_audit_metadata={k:v for k,v in model.items()if k!='objects'})
        return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    mh=sha(args.manifest);manifest=json.loads(args.manifest.read_text());e=Evaluator()
    support=old.checked(manifest['observed_support'],e.core.inputs)
    support_arrays=dict(np.load(support,allow_pickle=False))
    expected=('quality_points','quality_normals','quality_labels','quality_support_track')
    assert all(k in support_arrays for k in expected)
    n=len(support_arrays['quality_points']);assert all(len(support_arrays[k])==n for k in expected)
    metadata_path=old.checked(manifest['observed_support_metadata'],e.core.inputs)
    e.support_metadata=json.loads(metadata_path.read_text());e.support_arrays=support_arrays
    for basis in ('cropped','full'):
        meta=e.support_metadata[basis]
        assert all(len(support_arrays[meta[k]])==meta['points']for k in ('point_key','normal_key','label_key','track_key','key_key'))
    np.testing.assert_array_equal(support_arrays['quality_points'],support_arrays['full_quality_points'][support_arrays['quality_full_indices']])
    results=[e.stage(row,manifest['layout'])for row in manifest['stages']]
    for path,h in e.core.inputs.items():assert sha(path)==h,path
    assert sha(args.manifest)==mh
    result=dict(status='completed_parts_object_attributed_fixed_candidate_evaluation',manifest_sha256=mh,stages=results,
        observed_support_counts={str(t):int(np.count_nonzero(support_arrays['quality_support_track']==t))for t in np.unique(support_arrays['quality_support_track'])},
        input_sha256=e.core.inputs,evaluator_sha256={str(p):sha(p)for p in [Path(__file__),EVALUATOR,*e.core.helper_paths]},
        threshold_m=.05,no_registration=True,no_actor_executed=True,source_truth_loaded_only_by_evaluator=True,
        source_object_triangles_verified_against_frozen_source=True,elapsed_s=time.monotonic()-started,
        interpretation=['New component fixes are compared against B5 object_fit on identical full acquired positive-track support. Source truth remains evaluator-only.',
            'Per-object supported/realized weights are alternative hypothetical integrals, not unique physical area. Native reference recall/increments remain in the original stage fields.',
            'All B6 stages already share B5 observed-object admission. Lower prediction mass or higher supported fraction alone is not effectiveness evidence.',
            'No sparse-quadrature nearest-neighbor recall is used.'])
    (args.output/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=result['status'],stages=len(results),elapsed_s=result['elapsed_s'])))

if __name__=='__main__':main()
