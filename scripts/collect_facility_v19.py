#!/usr/bin/env python3
"""Prepare twelve declared options, execute full-budget tasks and fresh replays."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse,gzip,json,sys,subprocess,zipfile
from copy import deepcopy
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from env.facility_documentation_v19 import FacilityWorldV19
from nso.facility_runtime_v19 import facility_components_v19,FacilityRuntimeV19
from nso.cpu_sensor_contract_v10 import GridTransform,digest,json_value
from nso.decision_replay_v13 import save_packet,load_packet,array_hash
from utils.facility_metrics_v19 import FacilityEvaluatorV19
from utils.counterfactual_surface_visibility import reference_visible
from scripts.collect_semantic_gain_v13_history import packet,sha,write

def read(p):return json.loads(p.read_text())
def gzwrite(p,x):p.write_bytes(gzip.compress(json.dumps(json_value(x),sort_keys=True,allow_nan=False).encode(),mtime=0))
def seal(folder):write(folder/'artifact_hashes.json',{str(p.relative_to(folder)):sha(p) for p in sorted(folder.rglob('*')) if p.is_file() and p.name!='artifact_hashes.json'})
def check_sources(m):
    for name,wanted in m['source_sha256'].items():
        if sha(ROOT/name)!=wanted:raise ValueError('source changed: '+name)

def bootstrap(case):
    world=FacilityWorldV19(case['parent'],case['assignment'],sensor_model=case['sensor_model'],noise_seed=case['noise_seed'])
    args=SimpleNamespace(nso_backend='cpu_v10',eval=True,train_global=False,use_open_vocab_semantic=True,
        use_topo_graph=True,use_rpn_uq=True,use_igcr=True,cpu_score_mode='G',cpu_disable_feedback=False,
        cpu_max_candidates=5,cpu_coverage_slots=4,cpu_planner_revision='v10_3_1',
        cpu_semantic_source_schema='inspection_v4',cpu_measured_novelty_floor=.25,cpu_task_asset_count=6)
    comp=facility_components_v19(args,world.shape);runtime=FacilityRuntimeV19(comp,1,world.shape)
    first=packet(world,{'parent':case['parent']},None)
    runtime.start_sensor_episode(0,config=world.config,transform=GridTransform(world.shape,.2),packets=[first],
        total_budget=case['budget'],return_anchor=(*first.position,first.heading))
    return world,runtime,first

def prepare(root,feasibility):
    witness=read(feasibility/'result.json')
    if read(feasibility/'manifest.json')['status']!='complete' or witness.get('status')!='passed':raise ValueError('passed complete feasibility evidence required')
    if not read(feasibility/'manifest.json').get('frozen') or not witness.get('source_frozen'):raise ValueError('final frozen feasibility required')
    check_sources(read(feasibility/'manifest.json'))
    if len(witness.get('parents',[]))!=2 or any(p['budget']['status']!='passed' for p in witness['parents']):raise ValueError('two passing parent budget contracts required')
    for name,wanted in read(feasibility/'artifact_hashes.json').items():
        if sha(feasibility/name)!=wanted:raise ValueError('feasibility artifact changed')
    root.mkdir(parents=True,exist_ok=False);(root/'references').mkdir()
    sources={str(p.relative_to(ROOT)):sha(p) for folder in ('env','nso','utils') for p in sorted((ROOT/folder).glob('*.py'))}
    for n in ('scripts/collect_facility_v19.py','scripts/collect_semantic_gain_v13_history.py','configs/virtual3d/facility_documentation_v19_design.json'):
        sources[n]=sha(ROOT/n)
    manifest=dict(status='preparing',source_sha256=sources,feasibility_root=str(feasibility),
        feasibility_inventory_sha256=sha(feasibility/'artifact_hashes.json'),cases=[],training_allowed=False)
    write(root/'manifest.json',manifest)
    with zipfile.ZipFile(root/'sources.zip','x',zipfile.ZIP_DEFLATED) as z:
        for name in sources:z.write(ROOT/name,name)
    try:
        pairs={}
        for parent in witness['parents']:
            for variant in parent['variants']:
                for option in ('continue_coverage','observe_asset_A','observe_asset_B'):
                    case=dict(index=len(manifest['cases']),parent=parent['parent'],assignment=variant['assignment'],
                        option=option,budget=parent['budget']['budget'],sensor_model='iid_025px',noise_seed=1901)
                    world,runtime,first=bootstrap(case)
                    for seed in (2026,2027,2028):
                        FacilityEvaluatorV19(world,seed=seed,reference_cache=root/'references'/f'{case["parent"]}_{case["assignment"]}_{seed}.npz')
                    selection,receipt=runtime.install_pilot_option(option)
                    b=runtime.components._cpu_backend;s=b.scenes[0]
                    assets=s['assets'];attempted=s['gain'].attempted_camera_mask(s['mapper'],s['ledger'].planning_camera_poses())
                    try:
                        s['assets']=deepcopy(assets)
                        for a in s['assets']:a['class_vote']=-a['class_vote']
                        scores,_=b._scores_v10_1(s,selection['candidates'],attempted)
                        if scores!=selection['scores']:raise ValueError('class changed common task scores')
                    finally:s['assets']=assets
                    geometry=digest(dict(depth=first.frame.depth_m,scan=first.scan.ranges_m,
                        candidates=selection['candidates'],scores=selection['scores']))
                    key=(case['parent'],option)
                    if key in pairs and pairs[key]!=geometry:raise ValueError('new task candidate pairing failed')
                    pairs[key]=geometry
                    case.update(initial_packet_sha256=first.sha256(),shared_geometry_sha256=geometry,
                        pool_sha256=digest(selection['candidates']),receipt=receipt,
                        observed_votes=[a['class_vote'] for a in assets if a['marked_points']>0])
                    audit_start=len(runtime.audit)
                    first_action=runtime.next_local_action(0)
                    denials=[x for x in runtime.audit[audit_start:] if x.get('event')=='v14_route_denied']
                    case['initial_guard']=dict(original_first_action_allowed=first_action is not None and not denials,
                        any_action_authorized=first_action is not None, denials=denials)
                    if first_action is None:raise ValueError('no initial action even after guarded recovery')
                    manifest['cases'].append(case)
                    write(root/'manifest.json',manifest)
        check_sources(manifest)
        manifest['reference_sha256']={str(p.relative_to(root)):sha(p) for p in sorted((root/'references').glob('*.npz'))}
        if len(manifest['reference_sha256'])!=12:raise ValueError('twelve frozen world/reference caches required')
        manifest['status']='prepared'
    except Exception as error:manifest.update(status='failed_preparation',error=repr(error));raise
    finally:write(root/'manifest.json',manifest);seal(root)

def run_case(root,index,replay=False):
    manifest=read(root/'manifest.json');check_sources(manifest);case=manifest['cases'][index]
    references=manifest.get('reference_sha256',{})
    if len(references)!=12:raise ValueError('frozen reference inventory missing')
    for name,wanted in references.items():
        if not (root/name).is_file() or sha(root/name)!=wanted:raise ValueError('frozen reference changed or missing: '+name)
    folder=root/f'case_{index:02d}'
    if not replay:folder.mkdir();(folder/'packets').mkdir()
    world,runtime,first=bootstrap(case);comp=runtime.components;b=comp._cpu_backend;s=runtime.states[0];mapper=s['mapper']
    if first.sha256()!=case['initial_packet_sha256']:raise ValueError('prefix changed')
    evaluators={seed:FacilityEvaluatorV19(world,seed=seed,reference_cache=root/'references'/f'{case["parent"]}_{case["assignment"]}_{seed}.npz') for seed in (2026,2027,2028)}
    e=evaluators[2026];visible=reference_visible(e.global_evaluator.reference,first.frame,e.truth,
        first.frame.world_from_camera,world.config.max_depth_m)
    initial_visible=visible.copy()
    def evaluate(references=(2026,2027,2028)):
        coverage=float(np.count_nonzero((mapper.belief!=-1)&world.reachable)/world.reachable.sum())
        returned=tuple((*world.position,world.heading))==tuple((*world.start,0))
        mesh=mapper.mesh()
        failed=bool(s.get('termination',{}).get('failed',False))
        return {str(k):evaluators[k].evaluate(mesh,coverage,returned=returned,collisions=world.collisions,failed=failed) for k in references}
    before=evaluate();selection,receipt=runtime.install_pilot_option(case['option'])
    if digest(selection['candidates'])!=case['pool_sha256'] or receipt!=case['receipt']:raise ValueError('declared option changed')
    if not replay:save_packet(folder/'packets/0000.npz',first)
    expected=read(folder/'result.json') if replay else None
    actions=[];curve=[dict(action_id=0,metrics=before['2026'])];started=perf_counter()
    while not s['closed']:
        action=runtime.next_local_action(0)
        if action is None:break
        frame,collision,done=world.step(action)
        measured=packet(world,{'parent':case['parent']},action,frame,collision,done)
        path=folder/f'packets/{world.step_count:04d}.npz'
        if replay:
            if measured.sha256()!=load_packet(path).sha256():raise ValueError('fresh world packet differs')
        else:save_packet(path,measured)
        runtime.observe(0,world.step_count,None,None,None,sensor_packet=measured)
        visible|=reference_visible(e.global_evaluator.reference,frame,e.truth,frame.world_from_camera,world.config.max_depth_m)
        actions.append(dict(action=action,pose=[*world.position,world.heading],collision=collision,packet_sha256=measured.sha256()))
        if world.step_count%20==0:curve.append(dict(action_id=world.step_count,metrics=evaluate((2026,))['2026']))
        if world.step_count%50==0:print(f'case={index} replay={replay} actions={world.step_count}',flush=True)
    elapsed=perf_counter()-started
    if not s['closed']:runtime.close_sensor_episode(0,reason='pilot_terminated')
    after=evaluate();mesh=mapper.mesh()
    if curve[-1]['action_id']!=world.step_count:curve.append(dict(action_id=world.step_count,metrics=after['2026']))
    else:curve[-1]['metrics']=after['2026']
    calls=[{k:v for k,v in row.items() if k!='elapsed_s'} for row in b.calls]
    result=dict(case=case,actions=actions,before=before,after=after,quality_curve=curve,
        termination=s['termination'],paid_actions=world.step_count,path_distance_m=world.moves*.2,collisions=world.collisions,
        module_calls_sha256=digest(calls),runtime_audit_sha256=digest(runtime.audit),
        final_mesh_sha256={k:array_hash(np.asarray(getattr(mesh,k))) for k in ('vertices','triangles','vertex_colors')},
        new_visible_surface_m2=float(np.count_nonzero(visible&~initial_visible))*float(world.mesh.get_surface_area())/12000,
        curve_action_stride=20, task='six_facility_external_documentation',sensor_calibration_scope=world.sensor_metadata,
        actual_first_target_reached=any(x['pose']==case['receipt']['selected']['pose'] for x in actions),
        primitive_budget_compliant=world.step_count<=case['budget'],trained=False)
    if replay:
        if result!=expected:raise ValueError('replay actions, metrics, mesh, or module decisions differ')
        write(folder/'verification.json',dict(status='passed',fresh_world_sensor_action_metric_replay=True,actions=world.step_count))
    else:
        write(folder/'result.json',result);gzwrite(folder/'module_calls.json.gz',b.calls);gzwrite(folder/'runtime_audit.json.gz',runtime.audit)
        write(folder/'timing.json',dict(simulation_runtime_s=elapsed,includes_sensor_simulation_and_visibility_tracking=True,
            recorded_module_compute_s=sum(x['elapsed_s'] for x in b.calls),reference_initialization_and_final_evaluation_excluded=True))
        np.savez_compressed(folder/'final_mesh.npz',**{k:np.asarray(getattr(mesh,k)) for k in ('vertices','triangles','vertex_colors')})
    seal(folder);check_sources(manifest)
    print(f'case={index} replay={replay} complete; actions={world.step_count}; returned={s["termination"]["returned_to_anchor"]}',flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--prepare',type=Path);p.add_argument('--case',type=int);p.add_argument('--replay',action='store_true');p.add_argument('--all',action='store_true');a=p.parse_args()
    if a.prepare:prepare(a.output,a.prepare);return
    if a.case is not None:run_case(a.output,a.case,a.replay);return
    if not a.all:raise ValueError('choose prepare, case, or all')
    manifest=read(a.output/'manifest.json');check_sources(manifest)
    if manifest['status']!='prepared':raise ValueError('prepared-only launch; use explicit case replay to inspect existing work')
    manifest['status']='running';write(a.output/'manifest.json',manifest)
    try:
        for case in manifest['cases']:
            command=[sys.executable,str(Path(__file__).resolve()),'--output',str(a.output),'--case',str(case['index'])]
            subprocess.run(command,check=True);subprocess.run(command+['--replay'],check=True)
        manifest['status']='complete'
    except Exception as error:manifest.update(status='failed',error=repr(error));raise
    finally:write(a.output/'manifest.json',manifest);seal(a.output)

if __name__=='__main__':main()
