#!/usr/bin/env python3
"""Frozen CPU mechanism bridge, with paid acquisition and independent replay.

Preparation constructs no World. Every invocation acquires one real trajectory.
Ground-truth evaluation follows a terminal prediction/decision seal. A system
baseline can subclass this harness under a separate source/configuration freeze.
"""
import argparse
import io
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import time
import traceback
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_online_routes_v36 as prior

OUTPUT = ROOT/'audit_results/v39_external_cpu_20260920'
PROTOCOL = ROOT/'docs/research/V39_CPU_COMPARISON_PROTOCOL_20260920.md'
METHODS = ('G','S','SWAP','VISTA')
RESERVE = 64*1024**2
BUDGET, PREFIX, SECONDS = 42, 18, 600


class ExperimentV39:
    def __init__(self, output=OUTPUT):
        self.output = Path(output)

    def write_bytes(self, path, payload):
        path = Path(path); path.relative_to(self.output)
        if shutil.disk_usage(ROOT).free-len(payload) < RESERVE:
            raise RuntimeError('64 MiB persistent storage reserve')
        if len(payload)>16*1024**2: raise RuntimeError('per-file 16 MiB cap')
        current=sum(p.stat().st_size for p in prior.files_under(self.output))
        if current+len(payload)>128*1024**2:
            raise RuntimeError('128 MiB total experiment output cap')
        with path.open('xb') as f:
            f.write(payload); f.flush(); os.fsync(f.fileno())

    def write(self, path, value):
        self.write_bytes(path,(json.dumps(prior.normalized(value),ensure_ascii=False,
            separators=(',',':'),allow_nan=False)+'\n').encode())

    def arrays(self, path, **arrays):
        import numpy as np
        buf=io.BytesIO(); np.savez_compressed(buf,**arrays)
        self.write_bytes(path,buf.getvalue())

    def seal(self, folder, name, exclude_replay=False):
        self.write(folder/name,{str(p.relative_to(folder)):prior.sha(p)
            for p in prior.files_under(folder) if p.name!=name
            and not (exclude_replay and p.relative_to(folder).parts[0]=='replay')})

    def frozen(self):
        from nso.research_evidence_v31 import verify_sources
        manifest=verify_sources(self.output)
        for rel,sha in prior.read(self.output/'prepare_seal.json').items():
            if prior.sha(self.output/rel)!=sha: raise ValueError('preparation changed: '+rel)
        for rel,sha in manifest['input_sha256'].items():
            if prior.sha(ROOT/rel)!=sha: raise ValueError('public input changed: '+rel)
        return manifest,prior.read(self.output/'config.json')

    def prepare(self, methods=METHODS, extra_sources=(), scope='L1-inspired-CPU', config_overrides=None):
        prior.cpu_contract()
        if self.output.exists(): raise FileExistsError(self.output)
        from nso.research_evidence_v31 import closure
        parents=prior.read(prior.OUTPUT/'config.json')['parents']
        cases=[dict(index=i,parent=p,hypothesis=h,method=m,episode_id=f'v39-{scope}-{p}-{h}-{m}')
            for i,(p,h,m) in enumerate((p,h,m) for p in ('P00','P01') for h in (0,1) for m in methods)]
        config=dict(version='v39-external-bridge-1',scope=scope,parents=parents,physical_cases=cases,
            budget=BUDGET,prefix=PREFIX,maximum_main_starts=len(cases),maximum_fresh_replays=len(cases),
            historical_v36_main_used=35,historical_v36_main_limit=36,
            authorization='2026-09-20 user requested SWAP/VISTA CPU mechanisms and full TARE adapter experiment',
            budget_accounting='new separate V39 allocation; historical V36 35/36 untouched',
            parent_layouts_previously_seen=True,noise_seed_previously_used=True,independent_scene_samples=2,
            inference_scope='descriptive known-layout bridge; not unseen-scene confirmation or SOTA ranking',
            per_case_seconds=SECONDS,reserve_bytes=RESERVE,maximum_total_output_bytes=128*1024**2,
            per_file_bytes=16*1024**2,main_metric='C_map * public-ROI vertical-exterior surface F1@5cm',
            secondary_thresholds_cm=[2,10],all_failures_retained=True,stop_on_ineligible=False,
            stop_on_integrity_or_resource_failure=True,prediction_sealed_before_GT=True,
            no_candidate_world_queries=True,fresh_process_per_case=True,methods=list(methods),
            shared_semantic_channel='paid artificial RGB class cue; both facilities relevant',
            common_backend='ObservedRuntimeMapperV10 Open3D TSDF voxel .04 truncation .12',
            baseline_parameters='source-hashed modules and design protocol; no result-conditioned tuning')
        if config_overrides:
            if set(config_overrides)&set(config): raise ValueError('cannot override shared comparison contract')
            config.update(config_overrides)
        inputs=[prior.OUTPUT/'config.json',ROOT/'configs/virtual3d/v33_direction_scene_r1_20260917.json']
        for p in parents:
            inputs.append(ROOT/'audit_results/v33_direction_information_r1_20260917'/f'{p}_geometry.json')
        inputs+=prior.files_under(prior.INFORMATION)
        preflight=ROOT/'audit_results/v39_saved_prefix_preflight_r1_20260920'
        if prior.read(preflight/'result.json')['status']!='passed':
            raise ValueError('saved paid-prefix interface preflight required')
        inputs+=prior.files_under(preflight)
        sources=closure([Path(__file__),PROTOCOL,ROOT/'nso/external_planners_v39.py',
            ROOT/'nso/observed_geometry_v39.py',ROOT/'docs/research/V39_SEMANTIC_BASELINE_DESIGN_20260920.md',
            ROOT/'tests/test_observed_geometry_v39.py',ROOT/'tests/virtual3d/test_external_planners_v39.py',
            *extra_sources])
        for p in sources+inputs:
            if not p.is_file(): raise FileNotFoundError(p)
        self.output.mkdir(); self.write(self.output/'config.json',config)
        buffer=io.BytesIO()
        with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
            for p in sources: z.write(p,str(p.relative_to(ROOT)))
        self.write_bytes(self.output/'sources.zip',buffer.getvalue())
        self.write(self.output/'manifest.json',dict(status='prepared_immutable',
            source_sha256={str(p.relative_to(ROOT)):prior.sha(p) for p in sources},
            source_archive_sha256=prior.sha(self.output/'sources.zip'),
            input_sha256={str(p.relative_to(ROOT)):prior.sha(p) for p in inputs},
            worlds_during_prepare=0,sensor_queries_during_prepare=0,versions=prior.library_versions(),
            python=sys.version,preparation_time=time.time()))
        self.write(self.output/'prepare_seal.json',{p.name:prior.sha(p) for p in self.output.iterdir() if p.is_file()})
        print(json.dumps(dict(prepared=True,cases=len(cases),scope=scope)),flush=True)

    def create_controller(self, case, acquisition):
        import numpy as np
        from nso.online_planner_v35 import load_public_models_v35
        from nso.observation_belief_v35 import PublicTemplatesV35
        from nso.cpu_four_modules_v35 import CPUFourModuleControllerV35
        models,prefix=load_public_models_v35(ROOT,case['parent'])
        templates=PublicTemplatesV35.from_saved(prior.INFORMATION,case['parent'],models[0].poses)
        method=case['method']
        if method in ('G','S'):
            controller=CPUFourModuleControllerV35(models,templates,prefix,mode=method,
                total_budget=BUDGET,informative_nodes=acquisition['informative_nodes'])
        elif method in ('SWAP','VISTA'):
            from nso.observed_geometry_v39 import ObservedGeometryV39
            from nso.external_planners_v39 import ExternalPlannerV39,ExternalControllerV39
            ledger=ObservedGeometryV39(public_bounds=acquisition['public_bounds'])
            planner=ExternalPlannerV39(models,ledger,method=method,
                intrinsic=np.array([[48.,0.,47.5],[0.,48.,35.5],[0.,0.,1.]]),
                translation=acquisition['translation'][:2],camera_height_m=.9)
            controller=ExternalControllerV39(models,templates,prefix,mode='S',total_budget=BUDGET,
                informative_nodes=acquisition['informative_nodes'],planner=planner)
        else: raise ValueError('system baseline requires separate explicit factory')
        return controller,models

    def after_observation(self, controller, observation, packet):
        if hasattr(controller.planner,'ledger'):
            receipt=controller.planner.ledger.observe(observation,packet.frame.intrinsic,packet.frame.world_from_camera)
            controller.planner.observe(observation)
            return receipt
        return None

    def close_controller(self, controller):
        pass

    def verify_replay_policy(self, trace, evidence, folder):
        if prior.digest(trace)!=prior.digest(prior.read(folder/'trace.json')):
            raise AssertionError('fresh trace differs')
        if prior.digest(evidence)!=prior.digest(prior.read(folder/'controller.json')):
            raise AssertionError('fresh controller differs')

    def decorate_result(self, result):
        return result

    def finish_replay(self, result, expected, counts, folder, work):
        if prior.digest(result)!=prior.digest(expected):
            self.write(work/'mismatch_result.json',result); raise AssertionError('fresh result differs')
        self.write(work/'result.json',dict(passed=True,main_result_sha256=prior.sha(folder/'result.json'),
            fresh_process=True,all_packets_decisions_maps_meshes_and_scores_equal=True,
            policy_reexecuted=True,counts=counts))

    def run_case(self, index, replay=False):
        prior.cpu_contract()
        if (self.output/'result.json').exists(): raise ValueError('already finalized')
        _,config=self.frozen(); case=config['physical_cases'][index]
        acquisition=config['parents'][case['parent']]; folder=self.output/f'case{index:02d}'
        self.replay,self.case_folder=replay,folder
        if replay:
            prior.verify_seal(folder,'main_seal.json','replay')
            if (folder/'failure.json').exists(): raise ValueError('no full replay for incomplete main')
            expected=prior.read(folder/'result.json')
            if prior.same_process(prior.read(folder/'started.json')['process_identity'],prior.PROCESS_IDENTITY):
                raise ValueError('fresh process required')
            work=folder/'replay'
        else:
            dirs=sorted(self.output.glob('case[0-9][0-9]'))
            if len(dirs)!=index: raise ValueError('fixed order, no retries')
            for old in dirs:
                prior.verify_seal(old,'main_seal.json','replay')
                if not (old/'failure.json').exists():
                    prior.verify_seal(old/'replay','seal.json')
                    if not prior.read(old/'replay/result.json')['passed']: raise ValueError('prior replay failed')
            work=folder
        if shutil.disk_usage(ROOT).free<RESERVE+6*1024**2:
            raise RuntimeError('case headroom unavailable; no start counted')
        work.mkdir(); self.write(work/'started.json',dict(case=case,replay=replay,
            process_identity=prior.PROCESS_IDENTITY,counted_on_directory_creation=True))
        if not replay: (folder/'packets').mkdir()
        started=time.monotonic()
        counts=dict(worlds=0,sensor_packets=0,clean_depth_queries=0,scan_queries=0,paid_actions=0,
            mapper_updates=0,TSDF_integrations=0,mapper_mesh_extractions=0,evaluation_stages=0,
            candidate_world_queries=0,controllers=0)
        timing=dict(sensor_s=0.,mapping_s=0.,planning_and_observed_ledger_s=0.,evaluation_s=0.)
        trace,history,actions,snapshots=[],[],[],{}
        world=controller=None
        def deadline(*_): raise TimeoutError('declared whole-case deadline')
        signal.signal(signal.SIGALRM,deadline); signal.alarm(SECONDS)
        try:
            import numpy as np
            from env.information_pixel_v34 import cue_from_rgb,sensor_counts_v34
            from env.information_pixel_v36 import InformationPixelWorldV36
            from nso.decision_replay_v13 import array_hash,load_packet,save_packet
            from nso.observed_runtime_mapper_v10 import ObservedRuntimeMapperV10
            from nso.sensor_contract_v34 import validate_sensor_packet_v34
            from nso.surface_measurement_v34 import extract_observed_asset_mesh,SurfaceMeasurementV34
            from nso.cpu_four_modules_v35 import ObservationV35
            if any(sensor_counts_v34().values()): raise ValueError('fresh sensor process required')
            controller,models=self.create_controller(case,acquisition); counts['controllers']=1
            world=InformationPixelWorldV36(case['parent'],case['hypothesis'],case['episode_id'],
                noise_model=acquisition['noise_model'])
            if (world.noise_seed!=acquisition['noise_seed'] or list(world.shift)!=acquisition['translation']
                    or list(world.shape)!=acquisition['raster_shape']): raise ValueError('frozen calibration mismatch')
            mapper=ObservedRuntimeMapperV10(world.shape,world.config,truncation_m=.12)
            action,collisions=None,0
            for paid in range(BUDGET+1):
                t=time.monotonic(); packet=world.packet() if paid==0 else world.step(action)
                timing['sensor_s']+=time.monotonic()-t
                counts.update(worlds=world.counts['worlds'],sensor_packets=world.counts['packets'],
                    clean_depth_queries=world.counts['clean_depth_queries'],scan_queries=world.counts['scan_queries'],
                    paid_actions=world.counts['step_calls'])
                path=folder/'packets'/f'{paid:03d}.npz'
                if not replay:
                    if shutil.disk_usage(ROOT).free<RESERVE+256*1024: raise RuntimeError('packet reserve')
                    save_packet(path,packet)
                validate_sensor_packet_v34(packet,world.transform,world.config)
                pose=prior.observed_pose(packet,acquisition['translation'])
                if packet.action_id!=paid or packet.action!=action or np.any(packet.frame.semantic):
                    raise AssertionError('paid observation contract')
                if replay and packet.sha256()!=load_packet(path).sha256(): raise AssertionError('fresh packet differs')
                t=time.monotonic(); mapper.update(packet.frame,packet.scan)
                timing['mapping_s']+=time.monotonic()-t
                counts['mapper_updates']+=1; counts['TSDF_integrations']+=1; collisions+=int(packet.collision)
                observation=ObservationV35(frame_id=f'opaque-{paid}',step=paid,pose=pose,action=action,
                    depth=packet.frame.depth_m,rgb=packet.frame.color_rgb,ranges=packet.scan.ranges_m,
                    collision=packet.collision,measured_map_sha256=array_hash(mapper.belief))
                t=time.monotonic(); controller.accept(observation)
                ledger_receipt=self.after_observation(controller,observation,packet)
                following=controller.next_action()
                timing['planning_and_observed_ledger_s']+=time.monotonic()-t
                trace.append(dict(paid=paid,action=action,pose_v33=list(pose),packet_sha256=packet.sha256(),
                    nonsemantic=prior.nonsemantic_fields(packet),collision=packet.collision,
                    collisions_so_far=collisions,measured_map_sha256=array_hash(mapper.belief),
                    cue=cue_from_rgb(packet.frame),next_action=following,observed_ledger=ledger_receipt))
                history.append(dict(step=paid,state=controller.state,posterior=controller.posterior_receipts[-1],
                    next_action=following))
                for stage,take in (('prefix',paid==PREFIX),('final',following is None)):
                    if not take: continue
                    t=time.monotonic(); raw=mapper.mesh(); counts['mapper_mesh_extractions']+=1
                    predicted,crop=extract_observed_asset_mesh(raw,acquisition['public_bounds'])
                    maps={k:v.copy() for k,v in vars(mapper).items() if isinstance(v,np.ndarray)}
                    snapshots[stage]=dict(mesh=predicted,maps=maps,crop=crop,paid=paid)
                    for name,arrays in ((f'{stage}_raw.npz',prior.mesh_arrays(raw)),
                            (f'{stage}_extracted.npz',prior.mesh_arrays(predicted)),(f'{stage}_maps.npz',maps)):
                        if replay: prior.compare_arrays(folder/name,arrays)
                        else: self.arrays(folder/name,**arrays)
                    if replay:
                        if prior.digest(crop)!=prior.digest(prior.read(folder/f'{stage}_crop.json')):
                            raise AssertionError('crop replay differs')
                    else: self.write(folder/f'{stage}_crop.json',crop)
                    timing['mapping_s']+=time.monotonic()-t
                    print(json.dumps(dict(case=index,method=case['method'],replay=replay,stage=stage,
                        paid=paid,seconds=round(time.monotonic()-started,3))),flush=True)
                if following is None: break
                actions.append(following); action=following
            if 'final' not in snapshots: raise AssertionError('no terminal prediction')
            evidence=prior.controller_evidence(controller,history,actions)
            if replay:
                self.verify_replay_policy(trace,evidence,folder)
                for rel,sha in prior.read(folder/'prediction_freeze.json')['artifact_sha256'].items():
                    if prior.sha(folder/rel)!=sha: raise AssertionError('prediction changed')
            else:
                self.write(folder/'trace.json',trace); self.write(folder/'controller.json',evidence)
                self.write(folder/'prediction_freeze.json',dict(before_first_GT_access=True,
                    all_controller_decisions_frozen=True,
                    artifact_sha256={str(p.relative_to(folder)):prior.sha(p) for p in prior.files_under(folder)}))
            # FIRST ground-truth access: all predictions and actual actions are fixed.
            t=time.monotonic(); floor=world.evaluation_floor()
            reachable,grid=floor['reachable'],floor['declared_grid_cells']
            floor_arrays=dict(safe=floor['safe'],reachable=reachable,declared_grid_cells=grid)
            if replay: prior.compare_arrays(folder/'evaluation_floor.npz',floor_arrays)
            else: self.arrays(folder/'evaluation_floor.npz',**floor_arrays)
            evaluator=SurfaceMeasurementV34(world.instance_mesh(0),world.instance_mesh(0,vertical_only=True))
            stages={}
            for stage,snapshot in snapshots.items():
                belief,stage_paid=snapshot['maps']['belief'],snapshot['paid']
                returned=tuple(trace[stage_paid]['pose_v33'])==models[0].poses[models[0].anchor]
                measured=evaluator.evaluate(snapshot['mesh'],float(np.mean(belief[reachable]!=-1)),
                    returned=returned,collisions=trace[stage_paid]['collisions_so_far'],failed=False,
                    paid_actions=stage_paid,budget=BUDGET)
                counts['evaluation_stages']+=1
                stages[stage]=dict(measurement=measured,C_grid_measured=float(np.mean(belief[grid[:,0],grid[:,1]]!=-1)),
                    full_reachable_raster_denominator=int(reachable.sum()),belief_sha256=array_hash(belief))
            timing['evaluation_s']+=time.monotonic()-t
            result=dict(physical_case=case,stages=stages,counts=counts,paid_actions=paid,
                returned=tuple(trace[-1]['pose_v33'])==models[0].poses[models[0].anchor],collisions=collisions,
                actions=actions,controller_summary=controller.summary(),trace_sha256=prior.digest(trace),
                controller_sha256=prior.digest(evidence),exact_simulated_odometry=True,
                natural_semantic_network=False,complete_original_external_method=False,
                full_trained_ANS=False,scope=config['scope'],candidate_world_queries=0)
            result=self.decorate_result(result)
            self.frozen(); timing['total_s']=time.monotonic()-started
            self.write(work/'timing.json',timing)
            if replay:
                self.finish_replay(result,expected,counts,folder,work)
                self.seal(work,'seal.json')
            else:
                self.write(folder/'result.json',result); self.seal(folder,'main_seal.json',True)
            print(json.dumps(dict(case=index,method=case['method'],replay=replay,complete=True,
                eligible=stages['final']['measurement']['eligible'],seconds=round(time.monotonic()-started,3))),flush=True)
        except BaseException:
            signal.alarm(0)
            self.write(work/'failure.json',dict(error=traceback.format_exc(),counts=counts,
                counted_main_start=not replay,retry_allowed=False,partial_evidence_preserved=True))
            self.write(work/'partial_trace.json',trace)
            if controller is not None:
                self.write(work/'partial_controller.json',prior.controller_evidence(controller,history,actions))
            self.seal(work,'seal.json' if replay else 'main_seal.json',not replay)
            raise
        finally:
            signal.alarm(0)
            if controller is not None: self.close_controller(controller)

    def analyze(self):
        _,config=self.frozen(); rows,failures,traces=[],[],{}
        for case in config['physical_cases']:
            folder=self.output/f"case{case['index']:02d}"
            if not folder.exists(): raise ValueError('complete declared matrix before analysis')
            prior.verify_seal(folder,'main_seal.json','replay')
            if (folder/'failure.json').exists():
                failures.append(dict(case=case,evidence=prior.read(folder/'failure.json'))); continue
            result=prior.read(folder/'result.json'); prior.verify_seal(folder/'replay','seal.json')
            replay=prior.read(folder/'replay/result.json')
            if not replay['passed'] or replay['main_result_sha256']!=prior.sha(folder/'result.json'):
                raise AssertionError('replay integrity')
            traces[case['index']]=prior.read(folder/'trace.json'); m=result['stages']['final']['measurement']
            row=dict(**case,C_map=m['C_map'],eligible=m['eligible'],paid_actions=result['paid_actions'],
                returned=result['returned'],collisions=result['collisions'],forward_m=result['actions'].count('forward'),
                turns=len(result['actions'])-result['actions'].count('forward'),packet_count=result['counts']['sensor_packets'],
                timing=prior.read(folder/'timing.json'))
            for cm in (2,5,10):
                for short,key in (('Q','f1'),('J','joint'),('P','precision'),('R','recall')):
                    row[f'{short}{cm}']=m[f'{cm:02d}cm'][key]
            rows.append(row)
        pairchecks=[]
        for parent in ('P00','P01'):
            selected=[c for c in config['physical_cases'] if c['parent']==parent and c['index'] in traces]
            first=traces[selected[0]['index']] if selected else []
            for case in selected[1:]:
                trace=traces[case['index']]
                pairchecks.append(dict(parent=parent,a=selected[0]['index'],b=case['index'],
                    equal=min(len(first),len(trace))>=PREFIX+1 and all(a['nonsemantic']==b['nonsemantic']
                        for a,b in zip(first[:PREFIX+1],trace[:PREFIX+1]))))
        if not all(p['equal'] for p in pairchecks): raise AssertionError('common prefix differs')
        means={}
        for method in config['methods']:
            selected=[r for r in rows if r['method']==method]
            means[method]=dict(n=len(selected),eligible=sum(r['eligible'] for r in selected),
                **{k:sum(r[k] for r in selected)/len(selected) if selected else None
                    for k in ('C_map','Q2','Q5','Q10','J2','J5','J10','paid_actions','forward_m','turns')})
        self.write(self.output/'result.json',dict(status='complete_with_failures' if failures else 'complete',
            scope=config['scope'],rows=rows,means=means,failures=failures,prefix_comparisons=pairchecks,
            main_starts=len(config['physical_cases']),fresh_replays=len(rows),descriptive_only=True,
            parent_layouts_previously_seen=True,independent_layout_count=2,all_completed_cases_replayed=True,
            full_original_SWAP_or_VISTA_reproduction=False))
        self.seal(self.output,'final_seal.json')
        print(json.dumps(dict(status='complete',means=means,failures=len(failures))),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('prepare','main','replay','analyze')); p.add_argument('--index',type=int)
    args=p.parse_args(); experiment=ExperimentV39()
    if args.command=='prepare': experiment.prepare()
    elif args.command=='analyze': experiment.analyze()
    else:
        if args.index is None: p.error('--index required')
        experiment.run_case(args.index,replay=args.command=='replay')


if __name__=='__main__': main()
