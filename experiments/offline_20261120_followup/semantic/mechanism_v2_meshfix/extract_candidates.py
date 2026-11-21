#!/usr/bin/env python3
"""Original NSO candidate extraction on the frozen cold-start scene.

All class interventions affect only the acquired category prior in one common
particle bank. No source bounds, hidden class or selected goals enter scoring.
"""
import copy,hashlib,importlib.util,json,os,sys,time,traceback
from pathlib import Path
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
import scene
HERE=Path(__file__).resolve().parent
B8=Path('/home/ouc/nso_robot/analyses/dual_template_uncertainty_20261120')
STRUCTURE=HERE.parents[1]/'structure'
SELECTED=STRUCTURE/'audit/calibration_attempt_001/selected_temperature.json'

def load(name,p):
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
    from nso.cpu_sensor_contract_v10 import json_value
    Path(p).write_text(json.dumps(json_value(x),indent=2,allow_nan=False)+'\n')

def dependencies():
    wrapper=load('B9_mechanism_B8_binding',B8/'rollouts/joint_depth/run.py');wrapper.dependencies()
    b=wrapper.old;original=b.protocol();cfg=copy.deepcopy(original)
    cfg['environment']['initial_position']=scene.SPEC['start'];cfg['environment']['initial_heading']=scene.SPEC['heading']
    cfg['layouts']={k:dict(scene_id='B9_common_top_'+k)for k in scene.SPEC['layouts']}
    b.protocol=lambda:cfg
    return b,load('B9_mechanism_frozen_model',B8/'method/model.py')

def bootstrap(b,layout):
    from nso.cpu_sensor_contract_v10 import GridTransform
    world=scene.MechanismWorld(layout);tr=GridTransform(tuple(world.shape),world.config.resolution_m)
    frame=world.sense();packet=b.packet(world,tr,frame,layout)
    # Original frontend made an instance mask by category because its source
    # scene had exactly one object per category. Use the actual currently
    # rendered instance mask here; it is the same partial sensor channel and
    # does not reveal global object extents or hidden surfaces.
    b.front.owner=lambda raw:world.sense_visible_instances().astype(np.int64)
    return b.bootstrap(layout,'G_uncertainty',shared=(world,tr,packet))

def union(model,route,ray,cache):
    value=np.zeros(len(model.points))
    for raw in route['outbound_states'][1:]:
        state=tuple(map(int,raw))
        if state not in cache:cache[state]=model.gain(state[:2],state[2],ray)
        np.maximum(value,cache[state],out=value)
    return value

def winner(scores,routes):
    i=min(range(len(routes)),key=lambda i:(-scores[i],routes[i]['cost'],routes[i]['candidate_id']))
    return int(routes[i]['candidate_id'])if scores[i]>0 else None

def main():
    scene.verify();accepted=json.loads((HERE/'acceptance_attempt_001/result.json').read_text())
    assert all(value for key,value in accepted['checks'].items()if key!='initial_geometry_and_masks_identical')
    addendum=json.loads((HERE/'render_precision_addendum.json').read_text())
    assert addendum['mapper_occupancy_identical'] and addendum['mapper_quality_voxel_keys_identical']
    assert addendum['paired_laser_identical'] and addendum['paired_visible_owner_identical']
    out=HERE/'candidate_attempt_002';out.mkdir(exist_ok=False)
    write(HERE/'candidate_extraction_freeze_attempt002.json',dict(status='frozen_before_native_candidate_and_cue_scoring',
        files_sha256={str(p):sha(p)for p in (Path(__file__),HERE/'render_precision_addendum.json',HERE/'scene_freeze.json',HERE/'acceptance_attempt_001/result.json',B8/'method/model.py',B8/'rollouts/joint_depth/run.py',STRUCTURE/'replay.py',STRUCTURE/'model.py',SELECTED)},
        layouts=list(scene.SPEC['layouts']),cue_conditions=scene.SPEC['cue_conditions'],sigma_m=.01,temperatures=[1.,16.],prefix_actions=0))
    started=time.monotonic();results=[]
    try:
        b,model_module=dependencies()
        helper=load('B9_reuse_structural_temperature',STRUCTURE/'replay.py')
        assert json.loads(SELECTED.read_text())['temperature']==16.
        calibration=dict(path=str(SELECTED),sha256=sha(SELECTED))
        from nso.cpu_sensor_contract_v10 import digest
        for layout in scene.SPEC['layouts']:
            folder=out/layout;folder.mkdir()
            world,tr,p,comp,runtime,effective,owner=bootstrap(b,layout)
            comp._cpu_backend.select_target(0)
            state=comp._cpu_backend.scenes[0];mapper=state['mapper'];plan=state['last_selection'];routes=plan['candidates']
            cached=mapper._dual_template_state['cache'];bank=cached['models']['G'].bank
            write(folder/'original_plan.json',plan);write(folder/'fixed_candidates.json',routes)
            write(folder/'initial_owner_receipt.json',owner)
            write(folder/'common_bank.json',bank.distribution(.01,False))
            # All six scenes/cues remain even if representation or candidate
            # opportunity is absent. No candidate insertion or scene tuning.
            conditions=[]
            for temperature in (1.,16.):
                for cue in scene.SPEC['cue_conditions']:
                    modified=copy.copy(bank);modified.objects=[dict(x)for x in bank.objects]
                    for obj in modified.objects:
                        original=obj['shelf_prior_probability']
                        obj['shelf_prior_probability']=original if cue=='correct'else .5 if cue=='neutral'else 1-original
                    G=model_module.FullTemplateModel(mapper,False,sigma_m=.01,extent_mode='uncertainty',bank=modified)
                    S=model_module.FullTemplateModel(mapper,True,sigma_m=.01,extent_mode='uncertainty',bank=modified)
                    for m in (G,S):helper.reweight(m,temperature,calibration if temperature!=1. else None)
                    np.testing.assert_array_equal(G.points,S.points);np.testing.assert_array_equal(G.hypothesis_ids,S.hypothesis_ids)
                    arrays={};models={'G':G,'S':S};scores={};score_rows=[]
                    for name,model in models.items():
                        cache={};mat=[]
                        for route in routes:mat.append(union(model,route,cached['ray'],cache))
                        matrix=np.stack(mat)if mat else np.zeros((0,len(model.points)))
                        scores[name]=np.array(plan['scores']['N'])+matrix.sum(axis=1)
                        arrays.update({name+'_points':model.points,name+'_weights':model.weights,name+'_base_weights':model.base_weights,
                            name+'_object_ids':model.object_ids,name+'_particle_ids':model.particle_ids,name+'_shape_ids':model.hypothesis_ids%2,
                            name+'_route_union_gain_matrix':matrix})
                        for i,route in enumerate(routes):arrays[name+f'_gain_{route["candidate_id"]}']=matrix[i]
                    if temperature==1.:np.testing.assert_allclose(scores['G'],plan['scores']['G'],rtol=0,atol=1e-10)
                    if cue=='correct'and temperature==1.:np.testing.assert_allclose(scores['S'],plan['scores']['S'],rtol=0,atol=1e-10)
                    if cue=='neutral':np.testing.assert_array_equal(scores['G'],scores['S'])
                    for i,route in enumerate(routes):
                        score_rows.append(dict(candidate_id=route['candidate_id'],pose=route['pose'],cost=route['cost'],
                            outbound_actions=len(route['outbound_actions']),common=plan['scores']['N'][i],coverage=plan['scores']['C'][i],
                            G_total=scores['G'][i],S_total=scores['S'][i],G_gain=float(arrays['G_route_union_gain_matrix'][i].sum()),S_gain=float(arrays['S_route_union_gain_matrix'][i].sum())))
                    arrays['candidate_ids']=np.array([r['candidate_id']for r in routes]);np.savez_compressed(folder/(f'T{int(temperature)}_'+cue+'_predictions.npz'),**arrays)
                    receipt=dict(layout=layout,cue=cue,model_audits={name:{**m.audit,'objects':m.objects}for name,m in models.items()},
                        candidates=score_rows,G_winner=winner(scores['G'],routes)if routes else None,S_winner=winner(scores['S'],routes)if routes else None,
                        candidate_sha256=digest(routes),common_score_sha256=digest(plan['scores']['N']),observed_input_sha256=bank.observed_input_sha256,
                        shared_basis=True,temperature=temperature,sigma_m=.01,source_truth_to_prediction=False,simulator_actions=world.step_count)
                    write(folder/(f'T{int(temperature)}_'+cue+'_predictions.json'),receipt);conditions.append({k:v for k,v in receipt.items()if k not in('model_audits','candidates')})
            write(folder/'native_map_metadata.json',dict(belief=digest(mapper.belief),native_points=len(mapper.quality),frames=mapper.frames,world_steps=world.step_count))
            row=dict(layout=layout,native_candidates=len(routes),modeled_objects=len(bank.objects),unmodeled_tracks=bank.unmodeled_tracks,
                actual_candidate_poses=[r['pose']for r in routes],conditions=conditions)
            results.append(row);print(json.dumps(row),flush=True)
            assert world.step_count==0
        scene.verify();write(out/'result.json',dict(status='completed_native_candidates_both_temperatures_all_six_cues',layouts=results,
            source_scores_unchanged=True,new_candidate_actions=0,elapsed_s=time.monotonic()-started,
            candidate_source='Unmodified original NSO selector, single cold-start observation',
            complete_exploration_trial=False))
    except BaseException as e:
        write(out/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),completed_layouts=results,elapsed_s=time.monotonic()-started));raise
if __name__=='__main__':main()
