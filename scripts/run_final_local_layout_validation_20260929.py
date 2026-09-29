#!/usr/bin/env python3
"""Final fixed local-layout validation; frozen V35 science is imported unchanged.

Configuration and source hashes are frozen before preparation and acquisition.
All declared cases are retained, including infeasible/failed cases; no retry.
"""
import argparse
from collections import defaultdict, deque
from copy import deepcopy
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'audit_results/final_local_layout_validation_20260929'
CONFIG = ROOT / 'configs/virtual3d/final_local_layout_validation_20260929.json'
SCENE = ROOT / 'configs/virtual3d/v33_direction_scene_r1_20260917.json'
GIB = 1024 ** 3
MODES = {'G': 'G', 'S': 'S'}
OUTPUT_CAP = 256 * 1024 ** 2
HEADINGS = ((0, 1), (1, 0), (0, -1), (-1, 0))


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if hasattr(value, 'tolist'):
        return value.tolist()
    return value


def payload(value):
    return (json.dumps(plain(value), sort_keys=True, ensure_ascii=False,
                       separators=(',', ':'), allow_nan=False) + '\n').encode()


def used_bytes():
    return sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file()) if OUT.exists() else 0


def guard(size=0):
    if used_bytes() + size > OUTPUT_CAP:
        raise RuntimeError('new experiment would exceed the separate 256 MiB output cap')
    if shutil.disk_usage(ROOT).free - size < GIB:
        raise RuntimeError('new experiment would violate the 1 GiB disk reserve')


def write_bytes(path, data):
    guard(len(data)); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(data)


def write(path, value):
    write_bytes(path, payload(value))


def arrays(path, **values):
    import numpy as np
    buffer = io.BytesIO(); np.savez_compressed(buffer, **values)
    write_bytes(path, buffer.getvalue())


def validate_config(config):
    required = dict(schema='final.local_layout_validation.v1', main_tasks=8,
        online_task_cap_this_final_goal=16, unallocated_reserve_tasks=8,
        budget=42, prefix_paid_actions=18, noise_model='iid_025px', noise_seed=350918,
        source_parent_for_geometry_and_noise='P00', modes=['G', 'S'],
        per_case_seconds=120, output_cap_bytes=256*1024**2,
        minimum_free_bytes=1024**3,
        automatic_retries=False, report_all_declared_tasks=True,
        select_layouts_by_policy_outcomes=False, new_category_claim=False)
    if any(config.get(k) != v for k, v in required.items()):
        raise ValueError('fixed final-validation contract differs')
    names = [x['id'] for x in config['layouts']]
    if len(names) != 2 or len(set(names)) != 2:
        raise ValueError('exactly two distinct layouts required')
    expected = {(name, h, mode, 42, 350918) for name in names for h in (0, 1) for mode in MODES}
    actual = [(c['scene'], c['hypothesis'], c['mode'], c['budget'], c['noise_seed']) for c in config['cases']]
    if len(actual) != 8 or set(actual) != expected or len({c['id'] for c in config['cases']}) != 8:
        raise ValueError('all eight unique preregistered tasks required')
    if any(x['base_parent'] != 'P00' for x in config['layouts']):
        raise ValueError('equipment and noise parent must remain P00')
    return config


def runtime_metadata():
    import importlib
    import importlib.metadata
    versions={}
    for name in ('numpy','scipy','open3d'):
        try:
            versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            # The frozen scientific environment can expose bundled packages
            # without wheel dist-info, as supported by the V36 collector.
            versions[name]=str(importlib.import_module(name).__version__)
    cpuinfo=Path('/proc/cpuinfo')
    cpu_models=sorted({line.split(':',1)[1].strip() for line in cpuinfo.read_text().splitlines()
        if line.startswith('model name')}) if cpuinfo.exists() else []
    return dict(python=sys.version, executable=sys.executable,
        versions=versions,cpu_models=cpu_models,logical_cpu_count=os.cpu_count(),
        cpu_affinity_count=len(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None,
        simultaneous_cases=1,
        threads={name: os.environ.get(name) for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')},
        gpu_used=False)


def static_clear(start, end, boxes):
    """Pure rectangle/swept-circle arithmetic; no sensor, planner or World."""
    if sum(abs(a-b) > 1e-9 for a, b in zip(start, end)) > 1:
        raise ValueError('axis-aligned primitive required')
    for b in boxes:
        gap = [max(b[2*k]-max(start[k],end[k]), min(start[k],end[k])-b[2*k+1], 0.) for k in (0,1)]
        if sum(v*v for v in gap) < .2**2-1e-7:
            return False
    return True


def static_edges(parent, hypothesis):
    boxes = [b for a in parent['hypotheses'][hypothesis]['assets'] for b in a['boxes']] + parent['background_boxes']
    cells = {tuple(p) for p in parent['nav_cells']}; graph = {}
    for x,y in sorted(cells):
        for heading in range(4):
            links = {'left':(x,y,(heading-1)%4), 'right':(x,y,(heading+1)%4)}
            dx,dy = HEADINGS[heading]; target=(x+dx,y+dy)
            if target in cells and static_clear((x,y),target,boxes):
                links['forward']=(*target,heading)
            graph[x,y,heading]=links
    return graph


def static_distances(graph, start):
    found={start:0}; todo=deque([start])
    while todo:
        state=todo.popleft()
        for target in graph[state].values():
            if target not in found:
                found[target]=found[state]+1; todo.append(target)
    return found


def parents(config=None):
    config = validate_config(read(CONFIG) if config is None else config)
    base = next(p for p in read(SCENE)['parents'] if p['id']=='P00')
    result={}
    for item in config['layouts']:
        p=deepcopy(base); p['validation_layout_id']=item['id']
        p['background_boxes']=deepcopy(item['background_boxes'])
        grid=p['local_grid']; alternatives=[]
        for h in p['hypotheses']:
            boxes=[b for a in h['assets'] for b in a['boxes']]+p['background_boxes']
            if any(len(b)!=6 or any(not isinstance(v,(float,int)) or not math.isfinite(v) for v in b)
                   or any(b[2*k]>=b[2*k+1] for k in range(3)) for b in boxes):
                raise ValueError('positive finite box intervals required')
            cells=[(x,y) for x in range(grid['x_min'],grid['x_max']+1)
                   for y in range(grid['y_min'],grid['y_max']+1) if static_clear((x,y),(x,y),boxes)]
            alternatives.append(cells)
        if alternatives[0]!=alternatives[1]:
            raise ValueError('same complete legal graph required for both hypotheses')
        p['nav_cells']=[list(v) for v in alternatives[0]]
        result[item['id']]=p
    return result


def static_review(config=None):
    config=validate_config(read(CONFIG) if config is None else config); reports=[]
    for name,p in parents(config).items():
        graph=static_edges(p,0)
        if graph!=static_edges(p,1): raise ValueError('paired legal edges differ')
        anchor=tuple(p['anchor']); prefix=[anchor]
        for action in p['prefix_actions']:
            prefix.append(graph[prefix[-1]][action])
        if len(prefix)!=19 or prefix[-1]!=anchor:
            raise ValueError('exact common 18-action returned prefix required')
        reached=static_distances(graph,anchor)
        if len(reached)!=len(graph): raise ValueError('disconnected public graph')
        returns=[static_distances(graph,state).get(anchor) for state in graph]
        if any(x is None for x in returns): raise ValueError('public pose cannot return')
        targets=[b for h in p['hypotheses'] for a in h['assets'] for b in a['boxes']]
        lo=[min(b[2*k] for b in targets)-.2 for k in range(3)]
        hi=[max(b[2*k+1] for b in targets)+.2 for k in range(3)]
        gaps=[]
        for b in p['background_boxes']:
            separations=[max(b[2*k]-hi[k],lo[k]-b[2*k+1],0.) for k in range(3)]
            if not max(separations)>0:
                raise ValueError('new background intersects unchanged padded facility ROI')
            gaps.append(max(separations))
        reports.append(dict(scene=name, safe_centres=len(p['nav_cells']), public_poses=len(graph),
            full_grid_retained=True, both_hypotheses_same_graph=True, all_poses_returnable=True,
            maximum_shortest_return_actions=max(returns), prefix_actions=p['prefix_actions'],
            common_prefix_returned=True, background_roi_separation_m=gaps,
            background_boxes=p['background_boxes']))
    return dict(schema='final.local_layout_static_review.v1', layouts=reports,
        new_online_tasks=0, world_constructions=0, sensor_queries=0, planner_calls=0,
        scope='only geometric graph/ROI/prefix arithmetic; no quality or controller feasibility claim')


def freeze():
    if OUT.exists(): raise FileExistsError('final batch already exists; cannot replace freeze')
    config=validate_config(read(CONFIG)); static=static_review(config)
    legacy=read(ROOT/'audit_results/semantic_development_acquisition_20260923/method_correction.json')['execution_source_sha256']
    for rel,expected in {**legacy,**config['protected_scientific_sha256']}.items():
        if sha(ROOT/rel)!=expected: raise ValueError('protected scientific source changed: '+rel)
    from nso.research_evidence_v31 import closure
    sources=closure([Path(__file__),CONFIG,SCENE])
    archived=io.BytesIO()
    with zipfile.ZipFile(archived,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(sources): z.writestr(str(p.relative_to(ROOT)),p.read_bytes())
    OUT.mkdir(); write(OUT/'protocol.json',config); write(OUT/'static_review.json',static)
    write_bytes(OUT/'source_archive.zip',archived.getvalue())
    write(OUT/'source_freeze.json',dict(
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in sources},
        input_sha256={str(SCENE.relative_to(ROOT)):sha(SCENE)},
        protected_20260923_sources=legacy,config_sha256=sha(CONFIG),
        source_archive_sha256=sha(OUT/'source_archive.zip'),
        frozen_before_public_template_preparation_and_all_new_episodes=True,
        declared_main_tasks=8,online_cap=16,reserve_unallocated=8,legacy_queue_resumed=False))
    print(json.dumps({'frozen_cases':8,'source_count':len(sources),'free_GiB':shutil.disk_usage(ROOT).free/GIB}),flush=True)


def verify_sources():
    record=read(OUT/'source_freeze.json')
    for group in ('source_sha256','input_sha256','protected_20260923_sources'):
        for rel,expected in record[group].items():
            if sha(ROOT/rel)!=expected: raise ValueError('frozen input/source changed: '+rel)
    # The protocol is canonical JSON and config is pretty JSON.
    if sha(CONFIG)!=record['config_sha256'] or read(OUT/'protocol.json')!=read(CONFIG):
        raise ValueError('configuration changed after freeze')
    if sha(OUT/'source_archive.zip')!=record['source_archive_sha256']:
        raise ValueError('source archive changed')
    return validate_config(read(OUT/'protocol.json'))


def configure_world(world,parent,budget,seed):
    """Constructor state only; inherited render/step/physics/evaluation are untouched."""
    import numpy as np
    from dataclasses import replace
    from nso.box_union_geometry_v30 import BoxV30
    from nso.cpu_sensor_contract_v10 import GridTransform
    cells=np.asarray(parent['nav_cells'],int); low=cells.min(axis=0)-.5; high=cells.max(axis=0)+.5
    size=high-low
    world.parent=deepcopy(parent); world.public_layout_id=parent['validation_layout_id']
    world.public_bounds_v33=tuple(float(v) for v in (low[0],high[0],low[1],high[1]))
    world.shift=np.array([-low[0],-low[1],0.])
    world.config=replace(world.config,width_m=float(size[0]),height_m=float(size[1]),max_steps=int(budget))
    world.shape=(int(round(size[1]/.2)),int(round(size[0]/.2)))
    world.transform=GridTransform(world.shape,.2)
    world._nav_cells=frozenset(map(tuple,parent['nav_cells']))
    world._pose=tuple(parent['anchor']); world.position=world.pose_to_cell(world._pose)
    world.start=world.position; world.heading=world._pose[2]
    world.noise_seed=int(seed); world.step_count=world.collisions=0
    world.last_clean_depth=None; world._floor_cache=None
    world._reference_boxes=[]; world.solid_boxes=[]
    for asset in parent['hypotheses'][world.hypothesis]['assets']:
        members=[BoxV30(tuple((np.asarray(b).reshape(3,2)+world.shift[:,None]).ravel()),asset['id']) for b in asset['boxes']]
        world._reference_boxes.extend(members); world.solid_boxes.extend(members)
    for b in parent['background_boxes']:
        world.solid_boxes.append(BoxV30(tuple((np.asarray(b).reshape(3,2)+world.shift[:,None]).ravel()),None))
    world.boxes=tuple(world.solid_boxes)+(BoxV30((0.,world.config.width_m,0.,world.config.height_m,-.1,0.),None),)
    bounds=np.asarray([b.bounds for b in world.boxes]).reshape(-1,3,2)
    world.primitives=np.concatenate((bounds[:,:,0],bounds[:,:,1]-bounds[:,:,0]),axis=1)
    return world


def world_for(parent,hypothesis,episode,budget,seed,noise='iid_025px'):
    from env.information_pixel_v34 import InformationPixelWorldV34
    # P00 is retained solely for the unchanged sensor/noise namespace. The
    # distinct layout is pinned in the case, preparation and complete parent.
    return configure_world(InformationPixelWorldV34('P00',hypothesis,episode,noise_model=noise),parent,budget,seed)


def prepare():
    config=verify_sources()
    import numpy as np
    from nso.direction_information_v33 import DirectionGeometryV33
    from nso.observation_belief_v35 import PublicTemplatesV35
    from nso.pixel_information_v34 import SavedPotentialModelV34
    from scripts.run_online_routes_v36 import public_graph_audit
    prepared={}
    for name,parent in parents(config).items():
        started=time.monotonic(); folder=OUT/'public_templates'/name
        if folder.exists(): raise FileExistsError('preparation cannot overwrite a layout')
        folder.mkdir(parents=True)
        tables=[DirectionGeometryV33(parent,h).tables() for h in (0,1)]
        models=tuple(SavedPotentialModelV34(t,['unused']*len(t['poses'])) for t in tables)
        graph_audit=public_graph_audit(parent,models,parent['prefix_actions'])
        depth=[];ranges=[]
        for h in (0,1):
            world=world_for(parent,h,'public-template-only',42,350918,'clean')
            packets=[world.packet_at(pose,noise_model='clean') for pose in models[0].poses]
            depth.append(np.stack([p.frame.depth_m for p in packets]));ranges.append(np.stack([p.scan.ranges_m for p in packets]))
        templates=PublicTemplatesV35(models[0].poses,np.stack(depth),np.stack(ranges),provenance={'both_candidates_public':True,'scene':name})
        nodes=models[0].prefix_nodes
        common=bool(np.array_equal(templates.depth[0,nodes],templates.depth[1,nodes]) and np.array_equal(templates.ranges[0,nodes],templates.ranges[1,nodes]))
        if not common: raise ValueError('two configurations differ in paid-prefix nonsemantic geometry')
        bounds=np.asarray([b for h in parent['hypotheses'] for a in h['assets'] for b in a['boxes']]).reshape(-1,3,2)
        cells=np.asarray(parent['nav_cells']);shift=np.array([*(.5-cells.min(axis=0)),0.])
        public_bounds=[bounds[:,:,0].min(axis=0)+shift,bounds[:,:,1].max(axis=0)+shift]
        write(folder/'geometry.json',{'tables':tables,'parent':parent})
        arrays(folder/'sensor_templates.npz',depth=templates.depth,ranges=templates.ranges)
        write(folder/'metadata.json',dict(prefix=parent['prefix_actions'],
            informative_nodes=[n for n in range(len(models[0].poses)) if templates.template_information(n)['informative']],
            public_bounds=public_bounds,shift=shift,graph_nodes=len(models[0].poses),graph_audit=graph_audit,
            paired_graph_safe=True,prefix_geometrically_identical=common,prefix_returned=nodes[0]==nodes[-1]==models[0].anchor,
            pure_template_queries=2*len(models[0].poses),actual_episodes=0,elapsed_seconds=time.monotonic()-started))
        prepared[name]={p.name:sha(p) for p in folder.iterdir() if p.is_file()}
        print(json.dumps({'prepared':name,'seconds':time.monotonic()-started,'common_prefix':common}),flush=True)
    verify_sources();write(OUT/'preparation_seal.json',prepared)


def load_scene(name):
    import numpy as np
    from nso.pixel_information_v34 import SavedPotentialModelV34
    from nso.observation_belief_v35 import PublicTemplatesV35
    folder=OUT/'public_templates'/name
    for rel,expected in read(OUT/'preparation_seal.json')[name].items():
        if sha(folder/rel)!=expected:raise ValueError('public preparation changed')
    raw=read(folder/'geometry.json')
    models=tuple(SavedPotentialModelV34(t,['unused']*len(t['poses'])) for t in raw['tables'])
    with np.load(folder/'sensor_templates.npz',allow_pickle=False) as d:
        templates=PublicTemplatesV35(models[0].poses,d['depth'],d['ranges'],provenance={'preparation_sha256':sha(OUT/'preparation_seal.json')})
    return raw['parent'],models,templates,read(folder/'metadata.json')


def nonsemantic_fields(packet):
    from scripts.run_online_routes_v36 import nonsemantic_fields as frozen_nonsemantic_fields
    return frozen_nonsemantic_fields(packet)


def run_case(index):
    config = verify_sources()
    if type(index) is not int or not 0 <= index < len(config['cases']):
        raise ValueError('declared case index required')
    case = config['cases'][index]
    folder = OUT / 'cases' / case['id']
    if folder.exists(): raise FileExistsError('case already started; retry forbidden')
    # Runtime discovery is preflight, before the case claims a slot. Once the
    # directory exists, operational failures remain an unretryable attempt.
    runtime=runtime_metadata()
    start_record=dict(case=case, source_freeze_sha256=sha(OUT / 'source_freeze.json'),
          preparation_sha256=sha(OUT / 'preparation_seal.json'), time_unix=time.time(), pid=os.getpid(),
          counted_online_attempt=True, runtime=runtime,
          scene_noise_parent='P00', packet_scene_namespace='v34-P00',
          layout_identity_source='case.scene and frozen public preparation')
    guard(8 * 1024 ** 2); folder.mkdir(parents=True)
    started = time.monotonic()
    def timeout(*_): raise TimeoutError('120 second complete-case cap')
    trace, history, actions, measurements = [], [], [], {}
    world = controller = mapper = None
    planning_seconds = 0.0; mapping_seconds = 0.0; sensor_seconds = 0.0
    try:
        write(folder / 'started.json',start_record)
        signal.signal(signal.SIGALRM, timeout); signal.alarm(120)
        import numpy as np
        from nso.cpu_four_modules_v35 import CPUFourModuleControllerV35, ObservationV35
        from nso.observed_runtime_mapper_v10 import ObservedRuntimeMapperV10
        from nso.surface_measurement_v34 import extract_observed_asset_mesh, SurfaceMeasurementV34
        from nso.decision_replay_v13 import save_packet, array_hash
        from nso.sensor_contract_v34 import validate_sensor_packet_v34
        from env.information_pixel_v34 import nonsemantic_rgb
        parent, models, templates, info = load_scene(case['scene'])
        controller = CPUFourModuleControllerV35(models, templates, info['prefix'],
            mode=MODES[case['mode']], total_budget=case['budget'], informative_nodes=info['informative_nodes'])
        world = world_for(parent, case['hypothesis'], case['id'], case['budget'], case['noise_seed'])
        mapper = ObservedRuntimeMapperV10(world.shape, world.config, truncation_m=.12)
        folder.joinpath('packets').mkdir()
        following = None; snapshots = {}; collisions = 0
        for paid in range(case['budget'] + 1):
            guard(2 * 1024 ** 2)
            tick = time.monotonic()
            packet = world.packet() if paid == 0 else world.step(following)
            sensor_seconds += time.monotonic() - tick
            save_packet(folder / 'packets' / f'{paid:03d}.npz', packet)
            validate_sensor_packet_v34(packet, world.transform, world.config)
            if packet.action_id != paid or packet.action != following or np.any(packet.frame.semantic):
                raise ValueError('paid observation/action/semantic boundary violated')
            xy = np.asarray(packet.frame.world_from_camera[:2, 3]) - world.shift[:2]
            if not np.allclose(xy, np.rint(xy), atol=1e-9, rtol=0): raise ValueError('off-grid odometry')
            pose = (*map(int, np.rint(xy)), int(packet.heading))
            collisions += int(packet.collision)
            tick = time.monotonic(); mapper.update(packet.frame, packet.scan)
            mapping_seconds += time.monotonic() - tick
            controller.accept(ObservationV35(frame_id=f'paid-{paid}', step=paid, pose=pose,
                action=following, depth=packet.frame.depth_m, rgb=packet.frame.color_rgb,
                ranges=packet.scan.ranges_m, collision=bool(packet.collision), measured_map_sha256=array_hash(mapper.belief)))
            row = dict(paid=paid, action=following, pose=pose, packet_sha256=packet.sha256(),
                collision=bool(packet.collision), collisions=collisions,
                nonsemantic=nonsemantic_fields(packet),
                measured_map_sha256=array_hash(mapper.belief))
            trace.append(row)
            tick = time.monotonic(); next_action = controller.next_action()
            decision_seconds = time.monotonic() - tick
            planning_seconds += decision_seconds; row['decision_seconds'] = decision_seconds
            row['next_action'] = next_action
            history.append(dict(step=paid, posterior=controller.posterior_receipts[-1], state=deepcopy(controller.state), next_action=next_action))
            for stage, take in [('prefix', paid == 18), ('final', next_action is None)]:
                if not take: continue
                raw = mapper.mesh(); mesh, crop = extract_observed_asset_mesh(raw, info['public_bounds'])
                arrays(folder / f'{stage}_raw_mesh.npz', vertices=np.asarray(raw.vertices), triangles=np.asarray(raw.triangles))
                arrays(folder / f'{stage}_mesh.npz', vertices=np.asarray(mesh.vertices), triangles=np.asarray(mesh.triangles))
                arrays(folder / f'{stage}_map.npz', belief=mapper.belief)
                write(folder / f'{stage}_crop.json', crop)
                snapshots[stage] = dict(mesh=mesh, belief=mapper.belief.copy(), paid=paid, pose=pose, collisions=collisions)
            if next_action is None: break
            actions.append(next_action); following = next_action
        write(folder / 'trace.json', trace)
        write(folder / 'controller.json', dict(history=history, plans=controller.plans, calls=controller.calls,
              actions=actions, summary=controller.summary(), truth_or_selected_hypothesis_passed_to_controller=False))
        write(folder / 'prediction_freeze.json', {'before_first_reference_access': True,
              'artifacts': {str(p.relative_to(folder)): sha(p) for p in folder.rglob('*') if p.is_file()}})
        floor = world.evaluation_floor()
        arrays(folder / 'evaluation_floor.npz', reachable=floor['reachable'])
        evaluator = SurfaceMeasurementV34(world.instance_mesh(0), world.instance_mesh(0, vertical_only=True))
        for stage, snapshot in snapshots.items():
            coverage = float(np.mean(snapshot['belief'][floor['reachable']] != -1))
            measurements[stage] = evaluator.evaluate(snapshot['mesh'], coverage,
                returned=tuple(snapshot['pose']) == models[0].poses[models[0].anchor],
                collisions=snapshot['collisions'], failed=False, paid_actions=snapshot['paid'], budget=case['budget'])
        result = dict(case=case, status='completed', measurements=measurements, paid_actions=paid,
            controller_summary=controller.summary(), sensor_counts=dict(world.counts),
            trajectory_sha256=hashlib.sha256(payload(actions)).hexdigest(),
            prefix_nonsemantic_sha256=hashlib.sha256(payload([r['nonsemantic'] for r in trace[:19]])).hexdigest(),
            elapsed_seconds=time.monotonic()-started, real_geometry_used_for_terminal_evaluation=True,
            resource_usage_bytes=sum(p.stat().st_size for p in folder.rglob('*') if p.is_file()),
            execution_costs=dict(forward_actions=actions.count('forward'),
                translation_m=float(actions.count('forward')),
                turn_actions=actions.count('left') + actions.count('right'),
                paid_observations=len(actions), initial_unpaid_observations=1,
                explicit_observe_action_in_action_space=False),
            timings=dict(planning_wall_seconds=planning_seconds, mapping_wall_seconds=mapping_seconds,
                sensor_wall_seconds=sensor_seconds, total_case_seconds=time.monotonic()-started))
        write(folder / 'result.json', result)
    except BaseException as exc:
        signal.alarm(0)
        result = dict(case=case, status='failed', exception=repr(exc), traceback=traceback.format_exc(),
            elapsed_seconds=time.monotonic()-started, paid_actions=world.step_count if world else 0,
            sensor_counts=dict(world.counts) if world else {}, measurements=measurements,
            controller_summary=controller.summary() if controller else None,
            retry_allowed=False, retained_in_declared_denominator=True)
        write(folder / 'failure.json', result)
        if not (folder / 'trace.json').exists(): write(folder / 'partial_trace.json', trace)
        if controller and not (folder / 'controller.json').exists():
            write(folder / 'partial_controller.json', dict(history=history, plans=controller.plans,
                  calls=controller.calls, actions=actions, summary=controller.summary()))
    finally:
        signal.alarm(0)
    verify_sources()
    write(folder / 'seal.json', {str(p.relative_to(folder)): sha(p) for p in folder.rglob('*') if p.is_file()})
    print(json.dumps({'case': case['id'], 'status': result['status'],
          'seconds': round(result['elapsed_seconds'], 2),
          'J5': measurements.get('final', {}).get('main_joint'),
          'eligible': measurements.get('final', {}).get('eligible')}), flush=True)


def run_all():
    config = verify_sources()
    for index, case in enumerate(config['cases']):
        folder = OUT / 'cases' / case['id']
        if folder.exists():
            if (folder / 'seal.json').exists(): continue
            raise RuntimeError('partial case exists; automatic retry prohibited: ' + case['id'])
        guard(8 * 1024 ** 2)
        completed = subprocess.run([sys.executable, '-B', str(Path(__file__)), 'case', '--index', str(index)], cwd=ROOT)
        if completed.returncode != 0:
            raise RuntimeError('case process exited without sealed result: ' + case['id'])
        if (index + 1) % 4 == 0:
            print(json.dumps({'completed_slots': index+1, 'declared_slots': len(config['cases']),
                  'bytes': used_bytes(), 'free_GiB': shutil.disk_usage(ROOT).free/GIB}), flush=True)


def verify_saved_metrics(folder, result):
    """Recheck only saved raster and metric arithmetic; no surface evaluator."""
    import numpy as np
    frozen=read(folder/'prediction_freeze.json')
    if frozen['before_first_reference_access'] is not True:
        raise ValueError('missing prediction-before-reference seal')
    for rel,expected in frozen['artifacts'].items():
        if sha(folder/rel)!=expected: raise ValueError('pre-evaluation prediction changed: '+rel)
    with np.load(folder/'evaluation_floor.npz',allow_pickle=False) as d:
        reachable=d['reachable']
    for stage,measurement in result['measurements'].items():
        with np.load(folder/f'{stage}_map.npz',allow_pickle=False) as d:
            coverage=float(np.mean(d['belief'][reachable]!=-1))
        if abs(coverage-measurement['C_map'])>1e-12:
            raise ValueError('saved raster coverage differs')
        for threshold in ('02cm','05cm','10cm'):
            row=measurement[threshold];p,r=row['precision'],row['recall']
            f1=2*p*r/(p+r) if p+r else 0.
            if not all(0<=v<=1 for v in (p,r,row['f1'],row['joint'])) or abs(f1-row['f1'])>1e-12 or abs(coverage*f1-row['joint'])>1e-12:
                raise ValueError('saved P/R/F1/J arithmetic differs')
        eligible=bool(coverage>=.8 and measurement['returned'] and not measurement['collisions']
            and not measurement['failed'] and measurement['paid_actions']<=42)
        if eligible!=measurement['eligible']:raise ValueError('saved eligibility differs')


def analyze():
    config=verify_sources(); rows=[]; saved={}
    for case in config['cases']:
        folder=OUT/'cases'/case['id']; row=dict(case,status='unstarted',qualified=False)
        if not folder.exists(): rows.append(row);continue
        if not (folder/'seal.json').exists():
            row['status']='started_unsealed';rows.append(row);continue
        for rel,expected in read(folder/'seal.json').items():
            if sha(folder/rel)!=expected:raise ValueError('sealed case changed: '+case['id'])
        result=read(folder/('result.json' if (folder/'result.json').exists() else 'failure.json'))
        if result['case']!=case:raise ValueError('case identity mismatch')
        row.update(status=result['status'],paid_actions=result['paid_actions'],
            elapsed_seconds=result['elapsed_seconds'],failure=result.get('exception'))
        if result['status']=='completed':
            verify_saved_metrics(folder,result);m=result['measurements']['final']
            row.update(qualified=bool(m['eligible']),returned=m['returned'],collisions=m['collisions'],C_map=m['C_map'])
            for threshold in ('02cm','05cm','10cm'):
                for key in ('precision','recall','f1','joint'):row[threshold+'_'+key]=m[threshold][key]
            row.update(result['execution_costs']);row.update(result['timings'])
            saved[case['id']]=dict(result=result,trace=read(folder/'trace.json'),controller=read(folder/'controller.json'))
        rows.append(row)
    pairs=[];prefixes=[]
    for layout in config['layouts']:
        group=[c for c in config['cases'] if c['scene']==layout['id']]
        traces=[saved[c['id']]['trace'][:19] for c in group if c['id'] in saved]
        prefixes.append(dict(scene=layout['id'],declared_arms=4,completed_arms=len(traces),
            all_prefixes_present=len(traces)==4 and all(len(t)==19 for t in traces),
            same_nonsemantic_prefix=len(traces)==4 and len({payload([{'pose':r['pose'],'nonsemantic':r['nonsemantic']} for r in t]) for t in traces})==1))
        for h in (0,1):
            arms={r['mode']:r for r in rows if r['scene']==layout['id'] and r['hypothesis']==h}
            a,b=arms['S'],arms['G']; valid=a['qualified'] and b['qualified']
            pair=dict(scene=layout['id'],hypothesis=h,both_qualified=valid,
                S_status=a['status'],G_status=b['status'],delta_J5=None,
                first_decision_divergence=None,first_executed_action_divergence=None,
                mechanism_scope='only same-prefix divergence supports category attribution')
            if a['status']==b['status']=='completed':
                pair['delta_J5']=a['05cm_joint']-b['05cm_joint']
                left=saved[a['id']];right=saved[b['id']]
                divergence=next((i for i,(x,y) in enumerate(zip(left['controller']['history'],right['controller']['history'])) if x['next_action']!=y['next_action']),None)
                if divergence is not None:
                    x=left['controller']['history'][divergence];y=right['controller']['history'][divergence]
                    pair.update(first_decision_divergence=divergence,first_executed_action_divergence=divergence+1,
                        same_actual_nonsemantic_prefix=all(l['nonsemantic']==r['nonsemantic'] for l,r in zip(left['trace'][:divergence+1],right['trace'][:divergence+1])),
                        S_next_action=x['next_action'],G_next_action=y['next_action'],
                        S_posterior=x['posterior']['probabilities'],G_posterior=y['posterior']['probabilities'],
                        S_semantic_log_odds=x['posterior']['semantic_log_odds'],G_semantic_log_odds=y['posterior']['semantic_log_odds'],
                        S_geometry_log_odds=x['posterior']['geometry_log_odds'],G_geometry_log_odds=y['posterior']['geometry_log_odds'])
            pairs.append(pair)
    out=OUT/'analysis'
    if out.exists():raise FileExistsError('analysis is exclusive; retain earlier snapshot')
    out.mkdir()
    keys=list(dict.fromkeys(k for row in rows for k in row))
    buffer=io.StringIO();writer=csv.DictWriter(buffer,fieldnames=keys);writer.writeheader();writer.writerows(rows)
    write_bytes(out/'episode_metrics.csv',buffer.getvalue().encode())
    write(out/'paired_metrics.json',pairs);write(out/'prefix_checks.json',prefixes)
    completed=sum(r['status']=='completed' for r in rows); qualified=sum(r['qualified'] for r in rows)
    summary=dict(schema='final.local_layout_analysis.v1',declared_tasks=8,completed=completed,qualified=qualified,
        failed=sum(r['status']=='failed' for r in rows),unstarted=sum(r['status']=='unstarted' for r in rows),
        started_unsealed=sum(r['status']=='started_unsealed' for r in rows),new_layout_units=2,
        paired_conditions=4,all_results_retained=True,new_worlds=0,new_fusions=0,new_surface_evaluations=0,
        p_r_source='sealed measured surface scores; only raster/F1/J arithmetic recomputed',
        generalization_scope='two new same-family background arrangements; both equipment templates public',
        pairs=pairs,prefix_checks=prefixes)
    write(out/'summary.json',summary)
    write(out/'manifest.json',dict(source_freeze_sha256=sha(OUT/'source_freeze.json'),
        inputs={c['id']:sha(OUT/'cases'/c['id']/'seal.json') for c in config['cases'] if (OUT/'cases'/c['id']/'seal.json').exists()},
        files={p.name:sha(p) for p in out.iterdir() if p.is_file()}))
    print(json.dumps({'completed':completed,'qualified':qualified,'failed':summary['failed']}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['static','freeze','prepare','case','run','analyze'])
    parser.add_argument('--index',type=int);args=parser.parse_args()
    if args.command not in ('static','freeze'):
        for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
            if os.environ.get(name)!='1':raise ValueError(name+' must equal 1')
    if args.command=='static':print(json.dumps(static_review(),ensure_ascii=False,indent=2))
    elif args.command=='case':run_case(args.index)
    else:{'freeze':freeze,'prepare':prepare,'run':run_all,'analyze':analyze}[args.command]()


if __name__=='__main__':
    main()
