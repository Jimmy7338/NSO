#!/usr/bin/env python3
"""Execute all scorer-selected routes at every fixed-timing G/S divergence.

This is post-hoc localization on two previously scanned development histories,
not an estimate of general advantage. Prepare and seal all sensor-only choices
before a separate process constructs GT; then a third process independently
replays physical sensors, trajectories, fusion, area and reconstruction metrics.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile


def read(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path, value): Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
def require(value, message):
    if not value: raise ValueError(message)


def check_hashes(root, values):
    for name, expected in values.items():
        path = (root/name).resolve()
        require(path.is_relative_to(root.resolve()), "unsafe manifest path")
        require(sha(path) == expected, f"input hash changed: {name}")


def run(source, timing, output):
    source, timing, output = source.resolve(), timing.resolve(), output.resolve()
    require(not output.exists() and not output.is_relative_to(source) and not output.is_relative_to(timing), "separate new output required")
    source_meta, timing_meta = read(source/'metadata.json'), read(timing/'metadata.json')
    require(source_meta['status']=='complete' and timing_meta['status']=='complete', "sources must be complete")
    require(Path(timing_meta['source']) == source, "timing/physical source mismatch")
    timing_hashes = read(timing/'artifact_hashes.json'); check_hashes(timing, timing_hashes)
    inputs = read(timing/'input_hashes.json'); check_hashes(source, inputs)
    rows = [row for path in sorted(timing.glob('*/timeline.json')) for row in read(path)]
    require(len(rows)==68, "all fixed timing slots required")
    divergent = [row for row in rows if row['status']=='available' and row['selected']['G']!=row['selected']['S']]
    require(len(divergent)==2 and {(r['fixture'], r['step']) for r in divergent} ==
            {('main_751_a5beedada7c6',100),('main_752_1116de472bbb',70)}, "unexpected discovery inventory; do not silently select a subset")
    output.mkdir(parents=True); (output/'audit_code').mkdir()
    shutil.copyfile(source/'sources.zip', output/'sources.zip')
    shutil.copyfile(__file__, output/'audit_code/eval_timing_divergences.py')
    verifier = Path(__file__).resolve().parent/'replay_counterfactual_views.py'
    shutil.copyfile(verifier, output/'audit_code/replay_counterfactual_views.py')
    write(output/'source_input_hashes.json', inputs); write(output/'timing_input_hashes.json', timing_hashes)
    config = read(source/'config.json')
    config.update(scope='all G/S divergences found by fixed 68-slot development timing audit; post-hoc localization only',
                  physical_source=str(source), timing_source=str(timing), new_histories=[{'fixture':r['fixture'],'step':r['step']} for r in divergent])
    write(output/'config.json', config)
    audit = {'status':'running','scope':config['scope'],'source':str(source),'timing_source':str(timing),
             'source_sha256':source_meta['source_sha256'],'source_archive_sha256':sha(output/'sources.zip'),
             'audit_code_sha256':{str(p.relative_to(output)):sha(p) for p in (output/'audit_code').glob('*.py')},
             'workers':1,'started_unix':time.time(),'requested_timing_slots':68,'available_timing_slots':64,
             'discovered_divergence_histories':2,'independent_confirmation':False,'original_protocol_gate_applied':False,
             'selection_rule':'all unique G/O/S/X/M/N choices at every G/S disagreement; no outcome-based filtering'}
    write(output/'metadata.json', audit)
    try:
        with tempfile.TemporaryDirectory(prefix='nso_timing_execution_') as temporary:
            snapshot=Path(temporary)
            with zipfile.ZipFile(output/'sources.zip') as archive:
                require(set(archive.namelist())==set(source_meta['source_sha256']), 'source inventory mismatch')
                for name in archive.namelist():
                    target=(snapshot/name).resolve();require(target.is_relative_to(snapshot), 'unsafe archive')
                    data=archive.read(name);require(hashlib.sha256(data).hexdigest()==source_meta['source_sha256'][name], 'source mismatch')
                    target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
            env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
                                            PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(snapshot))
            for phase in ('prepare','execute','replay'):
                subprocess.run([sys.executable,str(output/'audit_code/eval_timing_divergences.py'),'--phase',phase,
                                '--snapshot',str(snapshot),'--source',str(source),'--timing',str(timing),
                                '--output',str(output)],cwd=snapshot,env=env,check=True)
        check_hashes(source, inputs);check_hashes(timing,timing_hashes)
        check_hashes(output,read(output/'pre_outcome_seal.json'))
        verification=read(output/'verification.json');require(verification['status']=='passed_all_selected_divergences','incomplete replay')
        audit.update(status='complete',elapsed_s=time.time()-audit['started_unix'],branches=4,
                     paid_actions=sum(r['paid_actions'] for r in verification['branches']),replay_verified=True)
    except Exception as error:
        audit.update(status='failed',error=repr(error));write(output/'metadata.json',audit);raise
    write(output/'metadata.json',audit)
    write(output/'artifact_hashes.json',{str(p.relative_to(output)):sha(p) for p in output.rglob('*')
                                        if p.is_file() and p.name!='artifact_hashes.json'})


def worker(source,timing,output,snapshot,phase):
    sys.path.insert(0,str(snapshot))
    import numpy as np
    from env.virtual3d_inspection_v4 import InspectionConfigV4,InspectionWorldV4
    from env.virtual3d import camera_pose
    from nso.counterfactual_view_scoring import score_routes,CounterfactualScoreConfig
    from scripts.eval_counterfactual_views import remap,grid_config,candidate_routes,state_restore,coverage,save_mesh,update_collision
    from utils.rgbd_contract import RGBDFrame,PlanarScan
    from utils.reconstruction_metrics import ReconstructionEvaluator
    from utils.counterfactual_surface_visibility import reference_visible,surface_increment
    for name,module in list(sys.modules.items()):
        if name.split('.')[0] in ('env','nso','utils') and getattr(module,'__file__',None):
            require(Path(module.__file__).resolve().is_relative_to(snapshot),'live dependency imported')
    config=read(output/'config.json')
    def prefix(fixture,index):
        folder=source/fixture;description=read(folder/'fixture.json');c=InspectionConfigV4(**description['environment'])
        rows=read(folder/'prefix/records.json')[:index+1]
        require(len(rows)==index+1 and not rows[-1]['done'],'unavailable prefix')
        frames=[RGBDFrame.load(folder/'prefix/frames'/f'{i:04d}.npz') for i in range(index+1)]
        scans=[PlanarScan.load(folder/'prefix/scans'/f'{i:04d}.npz') for i in range(index+1)]
        return description,c,rows,frames,scans
    if phase=='prepare':
        def forbidden(*args,**kwargs): raise RuntimeError('GT world forbidden before preparation seal')
        InspectionWorldV4.__init__=forbidden
        inventory=[]
        for item in config['new_histories']:
            fixture,index=item['fixture'],item['step'];description,c,rows,frames,scans=prefix(fixture,index)
            folder=output/fixture;folder.mkdir();write(folder/'fixture.json',description)
            history=folder/f'history_{index:04d}';history.mkdir()
            shape=(round(c.height_m/c.resolution_m),round(c.width_m/c.resolution_m))
            mappers={mode:remap(frames,scans,rows,c,shape,mode) for mode in ('aligned','shuffled','absent')}
            last=rows[-1];obs=mappers['aligned'].observation(tuple(last['position']),last['heading'],index,last['collision'])
            routes,candidate_audit=candidate_routes(mappers['aligned'],obs,48,12)
            prediction=json.loads(json.dumps(score_routes(mappers,routes,CounterfactualScoreConfig(grid_config(c),c))))
            recorded=read(timing/fixture/f'step_{index:04d}.json')
            require(prediction['selected']==recorded['selected'] and prediction['invariants']==recorded['invariants'],'timing prediction replay mismatch')
            for route,pred,prior in zip(routes,prediction['candidates'],recorded['candidates']):
                require(pred['scores']==prior['scores'] and pred['states_sha256']==prior['states_sha256'],'timing candidate score changed')
                require(list(route['pose'])==prior['pose'] and route['cost']==prior['cost'],'timing pose/cost changed')
                require(''.join({'forward':'F','left':'L','right':'R'}[a] for a in route['actions'])==prior['paid_actions'],'timing route changed')
            chosen_ids=sorted(set(prediction['selected'].values()))
            selected=[route for route in routes if route['candidate_id'] in chosen_ids]
            require(len(selected)==2,'unique scorer choice inventory changed')
            write(history/'candidates.json',routes);write(history/'candidate_audit.json',candidate_audit)
            write(history/'predictions.json',prediction);write(history/'selected_routes.json',selected)
            write(history/'prefix_records.json',rows)
            save_mesh(history/'prefix_mesh.npz',mappers['aligned'].mesh())
            np.savez_compressed(history/'prefix_map.npz',belief=mappers['aligned'].belief)
            inventory.append({'fixture':fixture,'step':index,'candidate_ids':chosen_ids,'selected':prediction['selected'],
                              'timing_checkpoint_sha256':sha(timing/fixture/f'step_{index:04d}.json')})
            print('prepared_all_selected',fixture,index,chosen_ids,flush=True)
        write(output/'prepared_histories.json',inventory)
        write(output/'pre_outcome_seal.json',{str(p.relative_to(output)):sha(p) for p in output.rglob('*')
                                            if p.is_file() and p.name!='metadata.json'})
        return
    check_hashes(output,read(output/'pre_outcome_seal.json'))
    if phase=='execute':
        histories=[]
        for item in read(output/'prepared_histories.json'):
            fixture,index=item['fixture'],item['step'];description,c,rows,frames,scans=prefix(fixture,index)
            folder=output/fixture;history=folder/f'history_{index:04d}'
            world=InspectionWorldV4(c,seed=description['entry']['seed'])
            evaluator=ReconstructionEvaluator(world,config['reference_samples'])
            point_weight=world.mesh.get_surface_area()/config['reference_samples']
            np.savez_compressed(folder/'reference.npz',points=evaluator.reference,classes=evaluator.classes,
                                weights=np.full(len(evaluator.reference),point_weight),reachable=world.reachable,
                                vertices=np.asarray(world.mesh.vertices),triangles=np.asarray(world.mesh.triangles))
            initial=remap(frames,scans,rows,c,world.shape);c0=coverage(initial,world.reachable)
            before=evaluator.evaluate(initial.mesh(),c0,config['thresholds_m'])
            seen=np.zeros(len(evaluator.reference),bool)
            for frame,row in zip(frames,rows):
                seen|=reference_visible(evaluator.reference,frame,evaluator.truth,camera_pose(tuple(row['position']),row['heading'],c,world.shape[0]),c.max_depth_m)
            outcomes=[]
            for route in read(history/'selected_routes.json'):
                require(shutil.disk_usage(output).free>100*2**20,'disk reserve reached')
                branch_dir=history/f"candidate_{route['candidate_id']:03d}";branch_dir.mkdir()
                (branch_dir/'frames').mkdir();(branch_dir/'scans').mkdir()
                mapper=remap(frames,scans,rows,c,world.shape);state_restore(world,rows[-1],frames[-1],scans[-1])
                masks={stage:np.zeros_like(seen) for stage in ('outbound','endpoint','return')}
                actions=[];metrics=[{'action_index':0,'coverage_2d':c0,**before}];failure=None
                for j,action in enumerate(route['actions'],1):
                    frame,collision,done=world.step(action);scan=world.scan()
                    frame.save(branch_dir/'frames'/f'{j:04d}.npz');scan.save(branch_dir/'scans'/f'{j:04d}.npz')
                    mapper.update(frame,scan);update_collision(mapper,world.position,world.heading,collision)
                    stage='outbound' if j<route['arrival_action'] else 'endpoint' if j==route['arrival_action'] else 'return'
                    masks[stage]|=reference_visible(evaluator.reference,frame,evaluator.truth,camera_pose(world.position,world.heading,c,world.shape[0]),c.max_depth_m)
                    now_c=coverage(mapper,world.reachable)
                    actions.append({'action_index':j,'absolute_step':world.step_count,'action':action,'position':list(world.position),
                                    'heading':world.heading,'collision':collision,'coverage_2d':now_c,'stage':stage})
                    if collision or (*world.position,world.heading)!=tuple(route['states'][j]):failure='collision_or_execution_mismatch'
                    if done and j<len(route['actions']):failure='world_budget_exhausted'
                    if j==route['arrival_action'] or j==len(route['actions']) or failure:
                        metrics.append({'action_index':j,'coverage_2d':now_c,**evaluator.evaluate(mapper.mesh(),now_c,config['thresholds_m'])})
                        save_mesh(branch_dir/('arrival_mesh.npz' if j==route['arrival_action'] else 'final_mesh.npz'),mapper.mesh())
                    if failure:break
                if not (branch_dir/'final_mesh.npz').exists():save_mesh(branch_dir/'final_mesh.npz',mapper.mesh())
                if failure is None:require((*world.position,world.heading)==tuple(route['states'][0]),'round trip failed')
                union=np.logical_or.reduce(list(masks.values()));cumulative=seen.copy();parts={}
                for stage,mask in masks.items():parts[stage]=surface_increment(cumulative,mask,point_weight);cumulative|=mask
                area=surface_increment(seen,union,point_weight);require(abs(sum(parts.values())-area)<1e-10,'area duplicate ledger')
                final=metrics[-1];paid=len(actions)
                result={'candidate_id':route['candidate_id'],'paid_actions':paid,'planned_actions':route['cost'],
                        'arrival_actions':route['arrival_action'],'failure':failure,'new_area_m2':area,'area_per_action':area/paid,
                        'stage_new_area_m2':parts,'f1_gain_05cm':final['f1_05cm']-before['f1_05cm'],
                        'f1_gain_per_action':(final['f1_05cm']-before['f1_05cm'])/paid,
                        'coverage_gain_m2':(final['coverage_2d']-c0)*world.reachable.sum()*c.resolution_m**2,'before':before,'after':final}
                for threshold in config['thresholds_m']:
                    tag=f'{round(threshold*100):02d}cm'
                    result[f'branch_joint_auc_{tag}']=float(np.trapz(np.interp(np.arange(49),[r['action_index'] for r in metrics],[r[f'joint_{tag}'] for r in metrics]))/48)
                write(branch_dir/'actions.json',actions);write(branch_dir/'metrics.json',metrics);write(branch_dir/'outcome.json',result)
                np.savez_compressed(branch_dir/'visibility.npz',prefix=seen,**masks,union=union)
                outcomes.append(result);print('executed',fixture,index,route['candidate_id'],'paid',paid,flush=True)
            write(history/'outcomes.json',outcomes)
            histories.append({**item,'outcomes':outcomes})
        write(output/'outcomes_summary.json',{'scope':config['scope'],'histories':histories,
                                            'all_pool_candidates_executed':False,'ranking_regret_estimated':False})
        return
    require(phase=='replay','unknown phase')
    specification=importlib.util.spec_from_file_location('independent_replay',output/'audit_code/replay_counterfactual_views.py')
    verifier=importlib.util.module_from_spec(specification);specification.loader.exec_module(verifier);verifier.worker_init(str(snapshot))
    verified=[];prefix_checks=[]
    for item in read(output/'prepared_histories.json'):
        fixture,index=item['fixture'],item['step'];description,c,rows,frames,scans=prefix(fixture,index)
        folder=output/fixture;history=folder/f'history_{index:04d}'
        world=InspectionWorldV4(c,seed=description['entry']['seed']);evaluator=ReconstructionEvaluator(world,config['reference_samples'])
        weights=np.full(len(evaluator.reference),world.mesh.get_surface_area()/config['reference_samples'])
        verifier.compare_npz(folder/'reference.npz',{'points':evaluator.reference,'classes':evaluator.classes,'weights':weights,
                                                  'reachable':world.reachable,**verifier.mesh_arrays(world.mesh)})
        mapper=verifier.SemanticHistoryMapperV3(world.shape,c,c.truncation_m);seen=np.zeros(len(evaluator.reference),bool)
        for i,(row,frame,scan) in enumerate(zip(rows,frames,scans)):
            if i==0:generated,collision,done=world.sense(),False,False
            else:generated,collision,done=world.step(row['action'])
            verifier.compare_sensor(generated,frame,f'prefix {fixture}/{i} RGB-D');verifier.compare_sensor(world.scan(),scan,f'prefix {fixture}/{i} scan')
            verifier.compare(row,{'step':i,'position':list(world.position),'heading':world.heading,'collision':collision,
                                  'collisions':world.collisions,'moves':world.moves,'action':'reset' if i==0 else row['action'],'done':done},'prefix trajectory')
            mapper.update(frame,scan);verifier.collision_update(mapper,row)
            seen|=verifier.reference_visible(evaluator.reference,frame,evaluator.truth,camera_pose(world.position,world.heading,c,world.shape[0]),c.max_depth_m)
        verifier.compare_npz(history/'prefix_mesh.npz',verifier.mesh_arrays(mapper.mesh()))
        verifier.compare_npz(history/'prefix_map.npz',{'belief':mapper.belief})
        before=evaluator.evaluate(mapper.mesh(),verifier.occupancy_coverage(mapper,world.reachable),config['thresholds_m'])
        all_routes=read(history/'candidates.json');prediction=read(history/'predictions.json')
        verifier.validate_predictions(prediction,all_routes)
        safe_map=~verifier.inflated_obstacles(mapper.belief!=0,c.robot_radius_m/c.resolution_m);safe_map[tuple(rows[-1]['position'])]=True
        selected_routes=read(history/'selected_routes.json');require({r['candidate_id'] for r in selected_routes}==set(prediction['selected'].values()),'missed scorer choice')
        for route in selected_routes:
            verifier.validate_route(route,(*rows[-1]['position'],rows[-1]['heading']),safe_map,48)
            result=verifier.replay_branch(history/f"candidate_{route['candidate_id']:03d}",route,world,evaluator,frames,scans,rows,config,weights,seen,before)
            result['path']=str(Path(result['path']).relative_to(output));verified.append(result)
            print('independently_replayed',result['path'],flush=True)
        prefix_checks.append({'fixture':fixture,'step':index,'prefix_frames':len(frames),'trajectory_sensors_fusion':'exact_match'})
    require(len(verified)==4,'incomplete selected-branch replay')
    check_hashes(output,read(output/'pre_outcome_seal.json'))
    write(output/'verification.json',{'status':'passed_all_selected_divergences','branches':verified,'prefixes':prefix_checks,
                                     'verifier_sha256':sha(output/'audit_code/replay_counterfactual_views.py'),
                                     'scope':'all four selected branches verified; not the original full-pool protocol or general efficacy'})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--timing',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--phase',choices=('prepare','execute','replay'),help=argparse.SUPPRESS);p.add_argument('--snapshot',type=Path,help=argparse.SUPPRESS)
    a=p.parse_args()
    if a.phase:worker(a.source.resolve(),a.timing.resolve(),a.output.resolve(),a.snapshot.resolve(),a.phase)
    else:run(a.source,a.timing,a.output)


if __name__=='__main__':main()
