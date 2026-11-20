#!/usr/bin/env python3
"""Re-score all sealed B8 branches with frozen structural temperature.

No candidate acquisition, policy rollout or robot endpoint. Geometry and
conditional particle support remain B8's complete uncapped quadrature.
"""
import argparse, copy, hashlib, importlib.util, json, os, sys, time
from pathlib import Path
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import numpy as np

HERE=Path(__file__).resolve().parent
B8=HERE.parents[1]/'dual_template_uncertainty_20261120'

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    obj=importlib.util.module_from_spec(spec);sys.modules[name]=obj;spec.loader.exec_module(obj);return obj

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def reweight(model,temperature,calibration):
    """Replace mixture weights only; history and every geometry point persist."""
    revised=load('B9_structural_posterior',HERE/'model.py')
    data=model.bank.materialize('uncertainty')
    mass=np.zeros(len(model.points));objects=copy.deepcopy(model.objects)
    for oi,obj in enumerate(objects):
        result=revised.posterior(obj['energy_m2'],obj['prior_mass'],obj['shape_prior'],temperature)
        joint=result['joint_probability_mass'];obj['joint_probability_mass']=joint.tolist()
        obj['shape_probability_mass']=joint.sum(axis=0).tolist()
        ds=np.asarray([p['depth_m'] for p in obj['particles']]);unique=np.unique(ds)
        obj['depth_grid_m']=unique.tolist();obj['probability_mass']=[float(joint[ds==d].sum()) for d in unique]
        obj['structural_temperature']=temperature
        obj['neutral_log_evidence_difference']=result['neutral_log_evidence_difference']
        for i,ki,hi,start,end in data['slices']:
            if i==oi:mass[start:end]=joint[ki,hi]
    base=data['areas']*mass;weights=base*model.history_keep
    if temperature==1.:
        np.testing.assert_array_equal(base,model.base_weights)
        np.testing.assert_array_equal(weights,model.weights)
    model.base_weights=base;model.weights=weights;model.objects=objects
    model.audit.update(objects=objects,structural_temperature=temperature,
        B9_modular_structure_revision=True,calibration=calibration,
        conditional_geometry_unchanged=True,marginal_depth_can_change=True,
        expected_area_before_history_m2=float(base.sum()),expected_area_after_history_m2=float(weights.sum()))
    return model

def main():
    p=argparse.ArgumentParser();p.add_argument('--layout',required=True,choices=('shelf_left','shelf_right'))
    p.add_argument('--state',required=True,type=int);p.add_argument('--temperature-config',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    selected=json.loads(a.temperature_config.read_text());temperature=float(selected['temperature'])
    freeze=json.loads((HERE/'replay_freeze.json').read_text())
    for path,digest in freeze['files_sha256'].items():assert sha(path)==digest,path
    assert freeze['selected_temperature_sha256']==sha(a.temperature_config)
    parent=load('B9_B8_prefix_runner',B8/'replay/run_particle_predictions.py');parent.verify()
    old,b,repair,parts,model=parent.dependencies()
    context=old.sealed_prefix(b,a.layout,a.state)
    prior=B8/'replay'/f'{a.layout}_{a.state}_attempt_001'
    assert context['routes']==json.loads((prior/'fixed_candidates.json').read_text())
    a.output.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    check=a.output/'temporary_prefix.ply';rebuilt=old.save_mesh(context['mapper'],check);check.unlink()
    prefix=json.loads((prior/'prefix_execution.json').read_text())
    assert rebuilt['sha256']==prefix['native_mesh_rebuilt_sha256']
    before=parent.snapshot(context['mapper'],model)
    from utils.reconstruction_metrics import ray_scene
    from nso.cpu_sensor_contract_v10 import json_value
    mapper=context['mapper'];mesh=mapper.mesh();ray=ray_scene(mesh) if len(mesh.triangles) else None
    bank=model.build_particle_bank(mapper);records={}
    for name,semantic in [('G',False),('S',True)]:
        m=model.FullTemplateModel(mapper,semantic=semantic,sigma_m=.01,extent_mode='uncertainty',bank=bank)
        base_npz=prior/(name+'_uncertainty_predictions.npz')
        with np.load(base_npz,allow_pickle=False) as original:
            np.testing.assert_array_equal(m.points,original[name+'_points'])
            np.testing.assert_array_equal(m.weights,original[name+'_weights'])
            np.testing.assert_array_equal(m.history_keep,original[name+'_history_keep'])
            reweight(m,1.,None)
            np.testing.assert_array_equal(m.weights,original[name+'_weights'])
        reweight(m,temperature,dict(path=str(a.temperature_config),sha256=sha(a.temperature_config)))
        cache={};rows=[];unions=[];t=time.monotonic()
        for route,common in zip(context['routes'],context['common_rows']):
            union=np.zeros(len(m.points))
            for state in route['outbound_states'][1:]:
                key=tuple(map(int,state))
                if key not in cache:cache[key]=m.gain(key[:2],key[2],ray)
                np.maximum(union,cache[key],out=union)
            area=float(union.sum());unions.append(union)
            rows.append(dict(candidate_id=int(route['candidate_id']),template_gain_m2=area,
                common_total_proxy=common['common_total_proxy'],score=common['common_total_proxy']+area,
                cost=route['cost'],outbound_actions=len(route['outbound_actions'])))
        points_file=a.output/(name+'_predictions.npz')
        np.savez_compressed(points_file,points=m.points,weights=m.weights,base_weights=m.base_weights,
            history_keep=m.history_keep,object_ids=m.object_ids,particle_ids=m.particle_ids,
            hypothesis_ids=m.hypothesis_ids,route_union_gain_matrix=np.asarray(unions),
            candidate_ids=np.asarray([r['candidate_id'] for r in rows]))
        winner=min(rows,key=lambda r:(-r['score'],r['cost'],r['candidate_id']))
        records[name]=dict(rows=rows,selected_candidate=winner['candidate_id'],model_audit=json_value(m.audit),
            arrays=dict(path=str(points_file),sha256=sha(points_file)),elapsed_s=time.monotonic()-t)
    after=parent.snapshot(mapper,model);assert before==after
    write(a.output/'result.json',dict(status='complete_two_arm_temperature_fixed_replay',layout=a.layout,state=a.state,
        temperature=temperature,candidates=len(context['routes']),arms=records,new_acquisition_actions=0,
        acquired_mapper_unchanged=True,temperature_one_B8_points_weights_exact=True,
        source_candidates_sha256=sha(prior/'fixed_candidates.json'),native_mesh_sha256=rebuilt['sha256'],
        source_B8_result_sha256=sha(prior/'result.json'),elapsed_s=time.monotonic()-start))
    for path,digest in freeze['files_sha256'].items():assert sha(path)==digest,path
    print(json.dumps({'layout':a.layout,'state':a.state,'status':'complete','selected':{k:v['selected_candidate'] for k,v in records.items()},'elapsed_s':time.monotonic()-start}),flush=True)

if __name__=='__main__':main()
