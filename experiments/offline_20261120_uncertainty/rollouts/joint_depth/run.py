#!/usr/bin/env python3
"""Declared B8 extent-distribution binding into the unchanged B policy loop.

No candidate, common score, mapping, navigation, budget or return change.
Only the parent starts complete-policy runs after fixed-branch review.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
B=ROOT.parent/'dual_template_exploration_20261119'
B5=ROOT.parent/'dual_template_objectfit_20261120'
STAGE='joint_depth'
_installed=False


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module


old=load('B8_frozen_parent_policy_runner',B/'run.py')
write,sha=old.write,old.sha


def protocol():return json.loads((HERE/'protocol.json').read_text())
def prepared(layout):return HERE/'layouts'/layout/'prepared'


def verify_parent(layout):
    folder=B/'layouts'/layout/'prepared';freeze=json.loads((folder/'freeze.json').read_text())
    for field in ('code_sha256','geometry_cache_sha256'):
        for path,digest in freeze[field].items():assert sha(Path(path))==digest,(field,path)
    assert old.base.source_hashes()==freeze['frozen_scientific_source_sha256']
    assert old.base.asset_hashes()==freeze['source_asset_sha256']
    assert old.base.library_versions()==freeze['library_versions']
    for name,digest in freeze['prepared_files_sha256'].items():assert sha(folder/name)==digest,name
    return json.loads((folder/'initial_state.json').read_text())


def code_hashes():
    paths=[Path(__file__),HERE/'protocol.json',ROOT/'policy_design.json',
        ROOT/'audit/calibration_attempt_001/selected_scale.json',ROOT/'audit/summary.json',
        ROOT/'method/validation.json',ROOT/'method/METHOD_SPEC.md',
        ROOT/'calibration/inference_freeze.json',B/'run.py',
        B5/'rollouts/object_support/run.py']
    frozen=json.loads((ROOT/'calibration/inference_freeze.json').read_text())
    paths += [Path(path)for path in frozen['files_sha256']]
    return {str(path):sha(path)for path in sorted(set(paths))}


def prepare(layout):
    cfg=protocol();assert cfg['status']=='frozen_wrapper_before_policy_runs'
    first=verify_parent(layout);folder=prepared(layout);folder.mkdir(parents=True,exist_ok=False)
    calibration=json.loads((ROOT/'audit/calibration_attempt_001/selected_scale.json').read_text())
    assert calibration['sigma_m']==cfg['method']['sigma_m']==.01
    frozen=json.loads((ROOT/'calibration/inference_freeze.json').read_text())
    for path,digest in frozen['files_sha256'].items():assert sha(Path(path))==digest,path
    write(folder/'initial_state.json',first)
    write(folder/'freeze.json',dict(schema='B8_five_arm_policy_wrapper_freeze_v1',code_sha256=code_hashes(),
        parent_freeze_sha256=sha(B/'layouts'/layout/'prepared/freeze.json'),initial_state_sha256=sha(folder/'initial_state.json'),
        reference_receipt_sha256=sha(B/'audit/references'/layout/'RECEIPT.json'),frozen_unix_s=time.time(),
        policy_runs_before_freeze=0,policy_start_requires_parent_review=True))
    print(json.dumps(dict(status='prepared',layout=layout,stage=STAGE)),flush=True)


def verify(layout):
    first=verify_parent(layout);folder=prepared(layout);frozen=json.loads((folder/'freeze.json').read_text())
    assert code_hashes()==frozen['code_sha256'],'Frozen B8 wrapper/input/scale changed'
    assert sha(B/'layouts'/layout/'prepared/freeze.json')==frozen['parent_freeze_sha256']
    assert sha(folder/'initial_state.json')==frozen['initial_state_sha256']
    assert sha(B/'audit/references'/layout/'RECEIPT.json')==frozen['reference_receipt_sha256']
    return first


def dependencies():
    global _installed
    if _installed:return
    import numpy as np
    old.dependencies()
    module=load('B8_frozen_joint_depth_policy_model',ROOT/'method/model.py')
    from utils.reconstruction_metrics import ray_scene
    cfg=protocol();arms=cfg['arms'];sigma=float(cfg['method']['sigma_m'])
    selected=json.loads((ROOT/'audit/calibration_attempt_001/selected_scale.json').read_text())
    assert sigma==selected['sigma_m']==.01
    old.method.ARMS=tuple(arms);old.method.MODE={name:row['score_mode']for name,row in arms.items()}
    old.method.FullTemplateModel=module.FullTemplateModel

    def cache(mapper,arm):
        saved=mapper._dual_template_state;assert mapper.frames==len(saved['frames'])
        mode=arms[arm]['extent_mode'];key=(int(mapper.frames),arm,mode,sigma)
        if saved['cache'] is not None and saved['cache']['key']==key:return saved['cache']
        bank=module.build_particle_bank(mapper)
        common=dict(bank=bank,sigma_m=sigma,extent_mode=mode,fit_update=True,history_rejection=True,measured_surface_consumption=True)
        G=module.FullTemplateModel(mapper,semantic=False,**common)
        S=module.FullTemplateModel(mapper,semantic=True,**common)
        assert G.bank is S.bank is bank
        shared_basis=mode!='mean_collapse'
        if shared_basis:
            np.testing.assert_array_equal(G.points,S.points)
            np.testing.assert_array_equal(G.hypothesis_ids,S.hypothesis_ids)
            np.testing.assert_array_equal(G.history_keep,S.history_keep)
        # Mean collapse conditions each structure's geometry on that arm's
        # unchanged joint posterior. Its diagnostic G/S point bases can differ.
        mesh=mapper.mesh();ray=ray_scene(mesh)if len(mesh.triangles)else None
        result=dict(key=key,models={'G':G,'S':S},ray=ray,gains={'G':{},'S':{}},gain_calls={'G':0,'S':0},
            state_sha256=hashlib.sha256((''.join(x['sha256']for x in saved['frames'])+arm+mode+repr(sigma)).encode()).hexdigest(),
            shared_bank=True,shared_point_basis=shared_basis,observed_input_sha256=bank.observed_input_sha256)
        saved['cache']=result;return result

    old.method._cache=cache
    cls=old.method.ExplorationJointCPUFourModules;score_original=cls._scores_v10_1
    def score(self,state,routes,attempted):
        values,rows=score_original(self,state,routes,attempted)
        saved=state['mapper']._dual_template_state;cached=saved['cache']
        saved['latest_call'].update(reused_original_gain_function=False,
            gain_implementation='B8_conditional_joint_particles_with_B2_visibility_and_active_only_exact_gain',
            model_source_sha256=sha(ROOT/'method/model.py'),stage_arms=arms,sigma_m=sigma,
            selected_scale_sha256=sha(ROOT/'audit/calibration_attempt_001/selected_scale.json'),
            shared_observed_instance_support=True,shared_particle_bank=cached['shared_bank'],
            shared_G_S_point_basis=cached['shared_point_basis'],bank_observed_input_sha256=cached['observed_input_sha256'],
            G_definition='neutral structural prior with the same ideal acquired instance support',
            mean_collapse_likelihood_recomputed=False)
        for row in rows:row['selected_stage']=STAGE;row['extent_mode']=arms[self.args.dual_template_arm]['extent_mode']
        return values,rows
    cls._scores_v10_1=score
    old.HERE=HERE;old.protocol=protocol;old.prepared=prepared;old.verify=verify
    _installed=True


def selfcheck(layout):
    import numpy as np
    from nso.cpu_sensor_contract_v10 import digest
    first=verify(layout);shared=old.world_and_packet(layout);rows={}
    for arm,definition in protocol()['arms'].items():
        world,tr,packet,comp,runtime,effective,owner=old.bootstrap(layout,arm,shared)
        assert packet.sha256()==first['initial_packet_sha256']
        comp._cpu_backend.select_target(0)
        state=comp._cpu_backend.scenes[0];plan=state['last_selection'];cached=state['mapper']._dual_template_state['cache'];models=cached['models']
        assert models['G'].bank is models['S'].bank
        for name,model in models.items():
            audit=model.audit
            assert audit['extent_mode']==definition['extent_mode'] and audit['sigma_m']==.01
            assert audit['history_rejection'] and audit['measured_surface_consumption'] and audit['fit_update']
            assert not audit['old_surface_likelihood_reapplied'] and audit['applied_global_point_cap'] is None
            assert not audit['source_geometry_consumed']
            assert model.semantic==(name=='S')
        if definition['extent_mode']!='mean_collapse':
            np.testing.assert_array_equal(models['G'].points,models['S'].points)
            np.testing.assert_array_equal(models['G'].hypothesis_ids,models['S'].hypothesis_ids)
        rows[arm]=dict(candidate_sha256=digest(plan['candidates']),belief_sha256=digest(state['mapper'].belief),
            owner_sha256=owner['owner_sha256'],initial_packet_sha256=packet.sha256(),
            first_plan=plan,model_audits={name:dict(model.audit,objects=model.objects)for name,model in models.items()},
            shared_bank=True,shared_point_basis=cached['shared_point_basis'],paid_actions=world.step_count)
        assert world.step_count==0 and world.collisions==0
    for field in ('candidate_sha256','belief_sha256','owner_sha256','initial_packet_sha256'):
        assert len({row[field]for row in rows.values()})==1,field
    for mode in ('fixed','uncertainty'):
        a,b=rows['G_'+mode]['first_plan'],rows['S_'+mode]['first_plan']
        for score_name in ('C','N','G','S'):np.testing.assert_array_equal(a['scores'][score_name],b['scores'][score_name])
    reference=next(iter(rows.values()))['first_plan']
    for row in rows.values():
        for score_name in ('C','N'):np.testing.assert_array_equal(row['first_plan']['scores'][score_name],reference['scores'][score_name])
    verify(layout)
    write(HERE/'layouts'/layout/'selfcheck.json',dict(status='passed',scope='zero-action shared input/candidate and module-binding check; not policy-effect evidence',
        simulator_actions=0,shared_initial_sensor_geometry_candidates=True,initial_coverage_fraction=first['evaluation']['coverage_fraction'],arms=rows))
    print(json.dumps(dict(status='passed',layout=layout,arms=list(rows),paid_actions=0)),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('prepare','verify','selfcheck','run'))
    parser.add_argument('--layout',required=True,choices=('shelf_left','shelf_right'));parser.add_argument('--arm');args=parser.parse_args()
    dependencies()
    if args.command=='prepare':prepare(args.layout)
    elif args.command=='verify':print(json.dumps(verify(args.layout)))
    elif args.command=='selfcheck':selfcheck(args.layout)
    elif args.arm in protocol()['arms']:old.run(args.layout,args.arm)
    else:parser.error('run requires a declared --arm')


if __name__=='__main__':main()
