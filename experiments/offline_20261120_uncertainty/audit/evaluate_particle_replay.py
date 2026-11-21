#!/usr/bin/env python3
"""Independent B8 five-arm prediction audit on the unchanged 66 real branches.

Source object surfaces are evaluator-only. Explicit point/object/particle IDs
replace the old two-hypotheses-per-object indexing assumption.
"""
import argparse,hashlib,importlib.util,json,os,sys,time
from pathlib import Path
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
B5=ROOT.parent/'dual_template_objectfit_20261120'
HELPER=B5/'audit/evaluate_objectfit.py'
HELPER_SHA='e230e1fa1c46993ccd79bdcb3fc35ee3a8bf7bbb09466da4f0dd401b762ca6e7'
ARMS={'G_fixed':('G','fixed'),'S_fixed':('S','fixed'),'G_uncertainty':('G','uncertainty'),'S_uncertainty':('S','uncertainty'),'S_mean_collapse':('S','mean_collapse')}

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(HELPER)==HELPER_SHA
spec=importlib.util.spec_from_file_location('B8_frozen_object_attribution',HELPER)
b5=importlib.util.module_from_spec(spec);sys.modules[spec.name]=b5;spec.loader.exec_module(b5)
old=b5.old

def point_tracks(arrays,name,objects):
    ids=np.asarray(arrays[name+'_object_ids'],np.int64)
    points=np.asarray(arrays[name+'_points']);n=len(points)
    assert ids.shape==(n,)
    if n:assert ids.min()>=0 and ids.max()<len(objects)
    expected=np.asarray([int(o['support_track'])for o in objects],np.int64)[ids] if n else np.empty(0,np.int64)
    declared=np.asarray(arrays[name+'_support_tracks'],np.int64)
    np.testing.assert_array_equal(declared,expected)
    for field in ('particle_ids','shape_ids','hypothesis_ids'):
        assert np.asarray(arrays[name+'_'+field]).shape==(n,)
    shape=np.asarray(arrays[name+'_shape_ids'],np.int64)
    assert np.all((shape==0)|(shape==1))
    assert np.all(expected>0)
    return expected

class Evaluator(b5.Evaluator):
    def __init__(self,scale_resource):
        super().__init__()
        path=old.checked(scale_resource,self.core.inputs)
        selected=json.loads(path.read_text());self.sigma=float(selected['sigma_m'])
        assert selected['status'] in ('selected_on_training_only','selected_provisional_on_training_with_fallback')
        self.shared_point_bases={}
        self.core.inputs[str(HELPER)]=HELPER_SHA

    def validate_support(self,row,name,model):
        basis=row['fit_support_basis'];assert basis=='full';meta=self.support_metadata[basis]
        cache_key=(row['prediction_receipt']['sha256'],name,basis)
        if cache_key in self.support_cache:return self.support_cache[cache_key]
        points=self.support_arrays[meta['point_key']];normals=self.support_arrays[meta['normal_key']]
        labels=self.support_arrays[meta['label_key']];tracks=self.support_arrays[meta['track_key']]
        assert model['schema']=='B8_policy_joint_particles_v1'
        assert model['sigma_m']==self.sigma and model['applied_global_point_cap'] is None
        assert model['old_surface_likelihood_reapplied'] is False and model['source_geometry_consumed'] is False
        assert model['candidate_weights_stops_or_navigation_changed'] is False
        assert model['history_rejection'] and model['measured_surface_consumption'] and model['fit_update']
        identities={}
        for obj in model['objects']:
            track=int(obj['support_track']);selected=tracks==track;base=obj['conditional_geometry']
            assert track>0 and base['observed_points']==int(selected.sum())
            expected={
                'fit_input_points_sha256':hashlib.sha256(np.ascontiguousarray(points[selected],dtype=np.float64).tobytes()).hexdigest(),
                'fit_input_normals_sha256':hashlib.sha256(np.ascontiguousarray(normals[selected],dtype=np.float64).tobytes()).hexdigest(),
                'fit_input_labels_sha256':hashlib.sha256(np.ascontiguousarray(labels[selected],dtype=np.int64).tobytes()).hexdigest()}
            for field,value in expected.items():assert obj[field]==base[field]==value,(track,field)
            identities[str(track)]=expected
            weights=np.asarray(obj['joint_probability_mass'],float)
            assert weights.shape==(len(obj['particles']),2) and np.isfinite(weights).all() and np.all(weights>=0)
            assert np.isclose(weights.sum(),1,atol=1e-12,rtol=0)
            np.testing.assert_allclose(weights.sum(axis=0),obj['shape_probability_mass'],atol=1e-12,rtol=0)
            assert obj['likelihood_applied'] and obj['sigma_m']==self.sigma
        identity_key=(row['layout'],row['state_action'],basis)
        if identity_key in self.basis_identities:assert identities==self.basis_identities[identity_key]
        else:self.basis_identities[identity_key]=identities
        admitted=np.isin(tracks,[int(t)for t in identities]);actual={}
        for target,ray in self.reference_objects(row['layout']).items():
            on_object=old.distance(ray,points)<=old.TOLERANCE;marked=on_object&(tracks>0);kept=on_object&admitted
            actual[target]=dict(acquired_points_within_5cm=int(on_object.sum()),acquired_positive_track_points_within_5cm=int(marked.sum()),
                admitted_track_points_within_5cm=int(kept.sum()),positive_track_point_retention_fraction=float(kept.sum()/marked.sum())if marked.any()else None,
                all_measured_object_point_retention_fraction=float(kept.sum()/on_object.sum())if on_object.any()else None)
        incoming=sorted(set(map(int,tracks))-{0});outgoing=sorted(map(int,identities))
        result=dict(basis=basis,input_points=len(points),input_positive_tracks=incoming,admitted_tracks=outgoing,
            unadmitted_tracks=sorted(set(incoming)-set(outgoing)),fit_input_hashes_verified=True,shared_support_identity=identities,
            source_object_measured_point_retention=actual,unmodeled_tracks=model.get('unmodeled_tracks',[]),
            geometry_posterior_mass_verified=True,interpretation='Geometry samples are alternatives for the same observed object, not additional objects or unique measured area.')
        self.support_cache[cache_key]=result;return result

    def stage(self,row,default_layout):
        result=self.core.stage(row,default_layout)
        variant=row['variant'];assert variant in ARMS
        name,mode=ARMS[variant]
        assert len(row['prediction_groups'])==1 and row['prediction_groups'][0]['name']==name
        arrays,receipt=self.prediction(row['predictions'],row['prediction_receipt'])
        model=receipt['model_audits'][name];assert model['extent_mode']==mode
        support=self.validate_support(row,name,model)
        points=np.asarray(arrays[name+'_points']);weights=np.asarray(arrays[name+f"_gain_{row['candidate_id']}"])
        tracks=point_tracks(arrays,name,model['objects'])
        # Verify G/S share every geometry sample/ID for matched non-collapse arms.
        if mode!='mean_collapse':
            key=(row['layout'],row['state_action'],mode)
            basis={field:np.asarray(arrays[name+'_'+field]) for field in ('points','object_ids','particle_ids','support_tracks','shape_ids','hypothesis_ids')}
            if key in self.shared_point_bases:
                for field,array in basis.items():np.testing.assert_array_equal(array,self.shared_point_bases[key][field])
            else:self.shared_point_bases[key]=basis
        current=self.attributed(points,weights,tracks,row['layout'],row['predictions'],Path(row['before_mesh']['path']),Path(row['after_mesh']['path']))
        result['arm']=variant
        result['observed_object_attribution']={name:dict(admitted_tracks=support['admitted_tracks'],current=current,
            support_validation=support,object_audit=model['objects'],model_audit_metadata={k:v for k,v in model.items()if k!='objects'})}
        return result

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--manifest',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False);start=time.monotonic();mh=sha(args.manifest);manifest=json.loads(args.manifest.read_text())
    e=Evaluator(manifest['selected_scale'])
    e.support_arrays=dict(np.load(old.checked(manifest['observed_support'],e.core.inputs),allow_pickle=False))
    e.support_metadata=json.loads(old.checked(manifest['observed_support_metadata'],e.core.inputs).read_text())
    meta=e.support_metadata['full']
    assert all(len(e.support_arrays[meta[k]])==meta['points']for k in ('point_key','normal_key','label_key','track_key','key_key'))
    rows=manifest['stages'];assert {r['variant']for r in rows}==set(ARMS)
    per_arm={a:{int(r['candidate_id'])for r in rows if r['variant']==a}for a in ARMS}
    assert all(v==per_arm['G_fixed']for v in per_arm.values())
    assert len(rows)==5*len(per_arm['G_fixed'])
    results=[e.stage(row,manifest['layout'])for row in rows]
    for path,h in e.core.inputs.items():assert sha(path)==h,path
    assert sha(args.manifest)==mh
    output=dict(status='completed_five_arm_particle_fixed_branch_evaluation',manifest_sha256=mh,stages=results,input_sha256=e.core.inputs,
        evaluator_sha256={str(p):sha(p)for p in [Path(__file__),HELPER,b5.EVALUATOR,*e.core.helper_paths]},
        threshold_m=.05,no_registration=True,no_actor_executed=True,source_truth_loaded_only_by_evaluator=True,
        common_fixed_and_uncertain_G_S_bases_verified=True,explicit_object_particle_indexing_verified=True,
        no_new_branch_acquisition=True,elapsed_s=time.monotonic()-start,
        interpretation=['Every arm reuses the same actual before/after native TSDF meshes and branch execution receipts.',
            'Per-object supported/realized weights integrate mutually exclusive hypotheses; they are not unique physical area.',
            'Within-B8 fixed/uncertainty controls share conditional geometry and calibrated likelihood; historical O/B6 remain separate development comparisons.',
            'Same six inspected prefixes/66 branches remain dependent diagnostics. No S-superiority or new closed-loop claim follows.'])
    (args.output/'result.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=output['status'],stages=len(results),elapsed_s=output['elapsed_s'])))
if __name__=='__main__':main()
