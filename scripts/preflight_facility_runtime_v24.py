#!/usr/bin/env python3
"""Rebuild saved prefixes into ANS N/G states and inspect one global decision.

No simulator is instantiated, no next_local_action is called, and no candidate
is executed. Public task configuration and actual packet history are the sole
runtime inputs. This readiness check is not a task-outcome or semantic test.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import argparse
import gc
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import sys
from time import perf_counter
import traceback
from types import SimpleNamespace
import zipfile

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import open3d as o3d
import scipy
import shapely
from nso.cpu_sensor_contract_v10 import GridTransform, digest, json_value
from nso.decision_replay_v13 import load_packet, array_hash
from nso.facility_runtime_v21 import FacilityRuntimeV21, facility_components_v21
from utils.grid_geometry import DIRECTIONS

DEFAULT_SOURCE = ROOT/'audit_results/facility_choice_v24_paid_prefix_20260915'
DEFAULT_OUTPUT = ROOT/'audit_results/facility_runtime_v24_readiness_20260915'
PROTOCOL = ROOT/'docs/research/V24_FOUR_MODULE_READINESS_PROTOCOL_20260915.md'
PAID_PREFIX = {'D24-P00':234, 'D24-P01':254}
TOTAL_BUDGET = {'D24-P00':387, 'D24-P01':421}
TASK_ASSET_COUNT = 2


def versions():
    return dict(python=sys.version,numpy=np.__version__,open3d=o3d.__version__,
                scipy=scipy.__version__,shapely=shapely.__version__,geos=shapely.geos_version_string)


def require(condition, message):
    if not condition: raise ValueError(message)


def read(path): return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def capacity(path, required_bytes=0):
    ancestor=path.parent
    while not ancestor.exists(): ancestor=ancestor.parent
    if shutil.disk_usage(ancestor).free-required_bytes < 32*1024**2:
        raise OSError('32 MiB preserved-free-space boundary would be crossed by readiness output')


def write(path, value):
    encoded=(json.dumps(json_value(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
    capacity(path,len(encoded)+4096)
    temporary = path.with_name(path.name+'.tmp')
    with temporary.open('xb') as stream: stream.write(encoded)
    os.replace(temporary, path)


def write_gzip(path,value):
    encoded=gzip.compress(json.dumps(json_value(value),ensure_ascii=False,allow_nan=False).encode(),mtime=0)
    capacity(path,len(encoded)+4096)
    temporary=path.with_name(path.name+'.tmp')
    with temporary.open('xb') as stream:stream.write(encoded)
    os.replace(temporary,path)


def inventory(root):
    values = read(root/'artifact_hashes.json')
    actual = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and p != root/'artifact_hashes.json'}
    require(actual == set(values), 'Complete artifact file set required: '+str(root))
    for name, value in values.items():
        path = root/name
        require(path.resolve().is_relative_to(root.resolve()), 'Inventory path escaped evidence root')
        expected = value['sha256'] if isinstance(value, dict) else value
        require(sha(path) == expected, 'Artifact hash mismatch: '+str(path))
        if isinstance(value, dict): require(path.stat().st_size == value['bytes'], 'Artifact size mismatch')
    return sha(root/'artifact_hashes.json')


def verify_source(source):
    manifest = read(source/'manifest.json'); result = read(source/'result.json')
    require(manifest['status'] == result['status'] == 'complete', 'Four prefixes must be complete before readiness execution')
    require(len(manifest['cases']) == result['physical_trajectories'] == result['independent_process_replays'] == 4, 'Four physical prefixes and four verifications required')
    require(result['saved_raw_packets'] == 980 and result['physical_paid_actions'] == result['replay_paid_actions'] == 976, 'Declared prefix counts differ')
    require(manifest['config']['task_asset_count'] == TASK_ASSET_COUNT, 'The public task has exactly two facilities')
    require(versions() == manifest['versions'], 'Dependency environment differs from saved prefix collection')
    require(manifest['config']['tsdf_truncation_m'] == .12 and manifest['config']['tsdf_voxel_m'] == .04,
            'Saved QualityMapper and runtime must use the same TSDF integration contract')
    root_hash = inventory(source)
    require(sha(source/'sources.zip') == manifest['source_archive_sha256'], 'Prefix source archive differs')
    with zipfile.ZipFile(source/'sources.zip') as archive:
        require(set(archive.namelist()) == set(manifest['source_sha256']), 'Prefix archived source set differs')
        for name, expected in manifest['source_sha256'].items():
            require(sha(ROOT/name) == expected and hashlib.sha256(archive.read(name)).hexdigest() == expected,
                    'Frozen prefix source differs: '+name)
    records = []
    for case in manifest['cases']:
        folder = source/f'case_{case["index"]:02d}'
        case_hash = inventory(folder)
        saved, receipt, timing = read(folder/'result.json'), read(folder/'verification.json'), read(folder/'timing.json')
        require(saved['status'] == 'complete' and receipt['status'] == 'passed' and receipt['independent_process'], 'Incomplete prefix case or verification')
        require(receipt['physical_process_id'] == timing['process_id'] and receipt['physical_process_id'] != receipt['replay_process_id'], 'Prefix verification needs a different PID')
        paid = PAID_PREFIX[case['parent']]
        require(saved['parent'] == case['parent'] and saved['assignment'] == case['assignment'] and saved['index'] == case['index'], 'Prefix identity mismatch')
        require(saved['paid_actions'] == receipt['paid_actions'] == paid and saved['raw_frames'] == receipt['raw_packets_verified'] == paid+1, 'Incomplete prefix packet history')
        require(receipt['fresh_sensor_replay'] and receipt['saved_packet_mapper_replay'] and receipt['all_three_mesh_arrays_equal'], 'Prefix sensor/mapper replay proof missing')
        require([r['action_id'] for r in saved['trace']] == list(range(paid+1)), 'Prefix trace is not consecutive')
        require(sorted(p.name for p in (folder/'packets').iterdir()) == [f'{i:04d}.npz' for i in range(paid+1)], 'Prefix packet file sequence differs')
        with np.load(folder/'final_mesh.npz', allow_pickle=False) as data:
            require(set(data.files) == {'vertices','triangles','vertex_colors'}, 'Saved TSDF schema differs')
            require({k:array_hash(data[k]) for k in data.files} == saved['final_mesh_sha256'], 'Saved prefix TSDF geometry differs')
        records.append(dict(case=case, saved=saved, folder=folder, inventory_sha256=case_hash))
    return manifest, result, records, root_hash


def geometry_evidence(mapper):
    # Same documented values as the collector, without importing its simulator.
    surface = [(list(key), value[0].tolist(), int(value[1])) for key,value in sorted(mapper.surface.items())]
    quality = [(list(key), {k:v for k,v in value.items() if k != 'label'}) for key,value in sorted(mapper.quality.items())]
    return dict(surface_without_class=digest(surface), quality_without_class=digest(quality))


def load_history(record):
    case, saved, folder = record['case'], record['saved'], record['folder']
    config = SimpleNamespace(**case['world_config'])
    shape = tuple(case['shape'])
    expected = (round(config.height_m/config.resolution_m), round(config.width_m/config.resolution_m))
    require(shape == expected, 'Public shape and metric task extent differ')
    require(config.voxel_m == .04, 'Readiness uses the saved public TSDF resolution')
    transform = GridTransform(shape, config.resolution_m)
    packets = []
    for action_id, row in enumerate(saved['trace']):
        packet = load_packet(folder/'packets'/f'{action_id:04d}.npz')
        packet.validate(transform, config)
        require(packet.sha256() == row['packet_sha256'] and packet.action_id == action_id, 'Saved paid packet differs')
        require([*packet.position, packet.heading] == row['pose'] and packet.action == row['action'], 'Saved actual pose/action differs')
        packets.append(packet)
    anchor = [*packets[-1].position, packets[-1].heading]
    require(anchor == case['decision_anchor'] and anchor[2] == 2, 'Actual prefix end must be the declared public heading-2 return anchor')
    return config, shape, packets, tuple(anchor)


def make_args(mode):
    return SimpleNamespace(nso_backend='cpu_v10', eval=True, train_global=False,
        use_open_vocab_semantic=True, use_topo_graph=True, use_rpn_uq=True, use_igcr=True,
        cpu_score_mode=mode, cpu_disable_feedback=False, cpu_max_candidates=5, cpu_coverage_slots=4,
        cpu_planner_revision='v10_3_1', cpu_semantic_source_schema='inspection_v4',
        cpu_measured_novelty_floor=.25, cpu_task_asset_count=TASK_ASSET_COUNT,
        cpu_v20_replan_interval=5, cpu_v20_coverage_slots=8)


def route_audit(route, safe, current, anchor, remaining):
    # This is an observation-grid consistency check, not simulated execution.
    errors = []
    states = [tuple(p) for p in route['states']]; actions = route['actions']
    if not states or states[0] != current or states[-1] != anchor: errors.append('round_trip_endpoints')
    if len(states) != len(actions)+1: errors.append('state_action_count')
    if route['cost'] != len(actions) or route['cost'] > remaining: errors.append('round_trip_budget')
    if route['outbound_cost'] != len(route['outbound_actions']) or route['return_cost'] != len(route['return_actions']): errors.append('split_budget')
    if route['outbound_cost']+route['return_cost'] != route['cost']: errors.append('total_budget')
    if route['outbound_states']+route['return_states'][1:] != route['states']: errors.append('split_states')
    if route['outbound_actions']+route['return_actions'] != actions: errors.append('split_actions')
    if route['outbound_states'][-1] != route['pose'] or route['return_states'][0] != route['pose']: errors.append('selected_pose_split')
    if tuple(route['return_anchor']) != anchor: errors.append('declared_return_anchor')
    for index, p in enumerate(states):
        if len(p)!=3 or p[2] not in range(4) or not all(0<=x<n for x,n in zip(p[:2],safe.shape)) or not safe[p[:2]]:
            errors.append('unknown_or_unsafe_footprint'); break
        if index:
            before, action = states[index-1], actions[index-1]
            if action not in ('forward','left','right'): errors.append('invalid_action'); break
            dr,dc = DIRECTIONS[before[2]] if action=='forward' else (0,0)
            expected = (before[0]+int(dr), before[1]+int(dc),
                        (before[2]+(1 if action=='right' else -1 if action=='left' else 0))%4)
            if p != expected: errors.append('action_pose_transition'); break
    return dict(candidate_id=route['candidate_id'], group=route['group'], asset_index=route.get('asset_index'),
        pose=route['pose'], outbound_cost=route['outbound_cost'], return_cost=route['return_cost'],
        cost=route['cost'], observed_grid_route_consistent=not errors, errors=errors,
        future_execution_success_proven=False)


def run_mode(record, mode, config, shape, packets, anchor):
    case, saved = record['case'], record['saved']; paid = saved['paid_actions']; total = TOTAL_BUDGET[case['parent']]
    components = facility_components_v21(make_args(mode), shape)
    runtime = FacilityRuntimeV21(components, 1, shape)
    runtime.start_sensor_episode(0, config=config, transform=GridTransform(shape, config.resolution_m),
        packets=packets, paid_prefix_actions=paid, total_budget=total, return_anchor=anchor)
    backend = components._cpu_backend; state = runtime.states[0]; shared = backend.scenes[0]; mapper = state['mapper']
    ledger = shared['ledger'].snapshot(); coverage = shared['coverage_v21'].snapshot()
    require(ledger['paid_actions'] == paid and ledger['remaining_budget'] == total-paid
        and ledger['total_budget'] == total and ledger['last_action_id'] == paid, 'Paid prefix was not charged exactly')
    require(mapper.frames == paid+1 and state['externally_scripted_prefix_actions'] == paid
        and len(ledger['prefix_frame_ids']) == paid+1, 'All actual prefix observations must be consumed')
    mesh = mapper.mesh()
    actual_mesh = {k:array_hash(np.asarray(getattr(mesh,k))) for k in ('vertices','triangles','vertex_colors')}
    actual_geometry = geometry_evidence(mapper)
    geometry = dict(mesh_sha256=actual_mesh, expected_mesh_sha256=saved['final_mesh_sha256'],
        tsdf_equal_to_saved_quality_mapper=actual_mesh==saved['final_mesh_sha256'],
        geometry_evidence_sha256=actual_geometry, expected_geometry_evidence_sha256=saved['final_geometry_evidence_sha256'],
        class_stripped_surface_quality_equal=actual_geometry==saved['final_geometry_evidence_sha256'],
        belief_sha256=array_hash(mapper.belief), visible_sha256=array_hash(mapper.visible), camera_seen_sha256=array_hash(mapper.camera_seen),
        belief_equal_to_saved_quality_mapper=array_hash(mapper.belief)==saved['trace'][-1]['belief_sha256'],
        visible_equal_to_saved_quality_mapper=array_hash(mapper.visible)==saved['trace'][-1]['visible_sha256'],
        footprint_conflict_count=mapper.current_footprint_conflict_count,
        current_footprint_conflict=mapper.current_footprint_conflict,
        mapper_relation='QualityMapperV2 update chain plus camera_seen, keyframes and observed current-cell obstacle preservation; belief equality is checked, never assumed')
    del mesh
    assets = shared['assets']; marked = [i for i,a in enumerate(assets) if a['marked_points']>0]
    asset_rows = [{k:v for k,v in a.items() if k not in ('points','bits')} for a in assets]
    asset_geometry = [{k:v for k,v in a.items() if k not in ('class_vote','raw_class_vote','semantic_encoding')} for a in assets]
    before = dict(paid=ledger['paid_actions'], frame_id=state['packet'].frame_id, mesh=actual_mesh,
                  belief=array_hash(mapper.belief),frames=mapper.frames,
                  visible=array_hash(mapper.visible),camera_seen=array_hash(mapper.camera_seen),
                  geometry_evidence=actual_geometry)
    runtime.choose_goal(0, list(packets[-1].position), (0,shape[0],0,shape[1]))
    # This public read-only RPN-UQ call records an actual module invocation.
    # It does not authorize a primitive or install a gain prediction.
    returning = components.plan_observed_return(0)
    selection = shared['last_selection']
    safe = shared['guard'].safe_grid(mapper.belief)
    route_rows = [route_audit(r,safe,tuple([*packets[-1].position,packets[-1].heading]),anchor,total-paid)
                  for r in selection['candidates']]
    final_ledger = shared['ledger'].snapshot()
    after_mesh=mapper.mesh()
    after_mesh_hash={k:array_hash(np.asarray(getattr(after_mesh,k))) for k in ('vertices','triangles','vertex_colors')}
    del after_mesh
    require(final_ledger['paid_actions']==before['paid'] and state['packet'].frame_id==before['frame_id']
        and state['pending'] is None and array_hash(mapper.belief)==before['belief']
        and mapper.frames==before['frames'] and array_hash(mapper.visible)==before['visible']
        and array_hash(mapper.camera_seen)==before['camera_seen'] and geometry_evidence(mapper)==before['geometry_evidence']
        and after_mesh_hash==before['mesh'], 'Planning unexpectedly changed measured state or paid action authorization')
    require(not any(a['event']=='paid_action_authorized' for a in runtime.audit), 'Readiness must not authorize execution')
    methods = [(c['module'],c['method']) for c in backend.calls]
    expected_calls = {('IGCR','bootstrap'),('IGCR','coverage_bootstrap_v21'),('OV-SDF','update_semantic'),
                      ('STGHP','select_topo_target'),('RPN-UQ','plan_return')}
    call_check = expected_calls.issubset(set(methods)) and sum(m==('STGHP','select_topo_target') for m in methods)==1
    candidates = selection['candidates']; audits = selection['score_audit']
    budget_allowed = [i for i,a in enumerate(audits) if a['v21_coverage']['coverage_budget']['allowed']]
    positive_quality = [i for i,a in enumerate(audits) if sum(a.get('v19_task_proxy',{}).get(k,0.)
        for k in ('observed_direction_term','observed_precision_term'))>0]
    noncoverage = [i for i,c in enumerate(candidates) if not c['group'].startswith('coverage_')]
    per_asset=[]
    for ai in marked:
        ids=[i for i in noncoverage if candidates[i].get('asset_index')==ai]
        per_asset.append(dict(observed_asset_index=ai,noncoverage_role_candidate_indices=ids,
            budget_allowed_positive_quality_indices=sorted(set(ids)&set(budget_allowed)&set(positive_quality)),
            association_scope='candidate role asset_index; does not establish future observation success'))
    for a in audits:
        require(a['v21_coverage']['class_used'] is False and a['v21_coverage']['evaluation_truth_used'] is False, 'Coverage ranking used forbidden semantic/truth inputs')
        if 'v19_task_proxy' in a:
            require(a['v19_task_proxy']['task_asset_count']==2 and a['v19_task_proxy']['class_used'] is False
                and a['v19_task_proxy']['evaluation_truth_used'] is False, 'Quality scoring did not use the declared common two-asset normalization')
    checks = dict(mapper_surface_and_tsdf_equal=geometry['tsdf_equal_to_saved_quality_mapper'] and geometry['class_stripped_surface_quality_equal'],
        belief_and_visible_equal=geometry['belief_equal_to_saved_quality_mapper'] and geometry['visible_equal_to_saved_quality_mapper'],
        exactly_two_marked_assets=len(marked)==2, four_modules_called=call_check,
        both_observed_assets_have_noncoverage_roles=len(per_asset)==2 and all(r['noncoverage_role_candidate_indices'] for r in per_asset),
        current_observed_return_available=returning.available,
        candidate_routes_observed_safe=all(r['observed_grid_route_consistent'] for r in route_rows),
        has_candidates=bool(candidates), has_selected_candidate=selection['selected'] is not None)
    if mode=='G':
        checks['has_budget_admitted_noncoverage_quality_candidate'] = bool(set(budget_allowed)&set(noncoverage)&set(positive_quality))
        checks['both_observed_assets_have_admitted_positive_quality_roles'] = len(per_asset)==2 and all(r['budget_allowed_positive_quality_indices'] for r in per_asset)
    result = dict(status='complete',case_index=case['index'], parent=case['parent'], assignment=case['assignment'], mode=mode,
        paid_prefix_actions=paid, bootstrap_saved_frames=len(packets), total_diagnostic_budget=total,
        remaining_diagnostic_budget=total-paid, task_asset_count=2, return_anchor=anchor,
        budget_is_formal_task_budget=False, coverage_budget_bootstrap=coverage,
        coverage_prefix_rate_history_reconstructed=False, paid_ledger=ledger, geometry=geometry,
        observed_assets=asset_rows, marked_asset_indices=marked, asset_geometry_sha256=digest(asset_geometry),
        candidate_pool_sha256=digest(candidates), common_N_scores=selection['scores']['N'],
        chosen_scores=selection['scores'][mode], candidate_routes=route_rows,
        all_budget_allowed_indices=budget_allowed, positive_quality_indices=positive_quality,
        per_observed_asset_candidate_capacity=per_asset,
        noncoverage_role_indices=noncoverage, runtime_admitted_indices=selection['quality_budget_admitted_candidates'],
        selected=selection['selected'], selection=selection, return_check=returning.__dict__,
        module_methods=methods, capabilities=components.capabilities, checks=checks,
        readiness_passed=all(checks.values()), semantic_ranking_enabled=False,
        v24_observed_shape_backend_invoked=False, new_physical_actions=0, new_sensor_frames=0,
        primitive_actions_authorized=0, planned_candidate_success_proven=False)
    calls = json_value(backend.calls); audit = json_value(runtime.audit)
    del runtime,components,backend,state,shared,mapper
    gc.collect()
    return json_value(result),calls,audit


def pair_checks(rows, source_result):
    comparisons=[]; by={(r['case_index'],r['mode']):r for r in rows if r['status']=='complete'}
    for index in range(4):
        a,b=by.get((index,'N')),by.get((index,'G'))
        comparisons.append(dict(kind='N_G_common_pool',case_index=index,
            passed=bool(a and b and a['candidate_pool_sha256']==b['candidate_pool_sha256']
                and a['asset_geometry_sha256']==b['asset_geometry_sha256'] and a['common_N_scores']==b['common_N_scores']),
            expected_identical_fields=['candidate_pool','observed_asset_geometry','coverage_scores']))
    for parent in PAID_PREFIX:
        source_pair=next(p for p in source_result['parents'] if p['parent']==parent)
        for mode in ('N','G'):
            group=[r for r in rows if r.get('parent')==parent and r.get('mode')==mode and r['status']=='complete']
            same=len(group)==2 and all(group[0][k]==group[1][k] for k in ('candidate_pool_sha256','asset_geometry_sha256','common_N_scores','chosen_scores'))
            selected_same=len(group)==2 and ((None if group[0]['selected'] is None else group[0]['selected']['candidate_id'])
                ==(None if group[1]['selected'] is None else group[1]['selected']['candidate_id']))
            comparisons.append(dict(kind='observed_class_exchange',parent=parent,mode=mode,
                source_prefix_pair_passed=source_pair['full_pair_gate_passed'],geometry_pool_scores_equal=same,
                selected_candidate_equal=selected_same,passed=bool(source_pair['full_pair_gate_passed'] and same and selected_same)))
    return comparisons


def execute(source,output,progress):
    manifest,source_result,records,input_hash=verify_source(source)
    own_sources={str(p.relative_to(ROOT)):sha(p) for p in (Path(__file__).resolve(),PROTOCOL)}
    capacity(output/'manifest.json',8192)
    output.mkdir(parents=True)
    progress['created_output']=True
    write(output/'manifest.json',dict(status='running',input_root=str(source),input_inventory_sha256=input_hash,
        source_sha256=own_sources,protocol=str(PROTOCOL),total_diagnostic_budget=TOTAL_BUDGET,
        task_asset_count=2,new_physical_actions=0,new_sensor_frames=0))
    started=perf_counter();results=[]
    for record in records:
        progress.update(case_index=record['case']['index'],phase='load_history')
        case=record['case']; config,shape,packets,anchor=load_history(record)
        for mode in ('N','G'):
            progress.update(mode=mode,phase='runtime_bootstrap_and_candidate_generation')
            try:
                result,calls,audit=run_mode(record,mode,config,shape,packets,anchor)
                for label,rows in [('module_calls',calls),('runtime_audit',audit)]:
                    write_gzip(output/f'case_{case["index"]:02d}_{mode}_{label}.json.gz',rows)
            except Exception as error:
                result=dict(status='failed',case_index=case['index'],parent=case['parent'],assignment=case['assignment'],mode=mode,
                    error=repr(error),traceback=traceback.format_exc(),readiness_passed=False,
                    new_physical_actions=0,new_sensor_frames=0)
            results.append(result);write(output/f'case_{case["index"]:02d}_{mode}.json',result)
            print(f'case={case["index"]} mode={mode} status={result["status"]} ready={result["readiness_passed"]}',flush=True)
        del packets;gc.collect()
    comparisons=pair_checks(results,source_result)
    progress['phase']='final_source_and_evidence_verification'
    require(inventory(source)==input_hash,'Saved prefix evidence changed during readiness analysis')
    require(versions()==manifest['versions'],'Dependency environment changed during readiness analysis')
    require(all(sha(ROOT/name)==h for name,h in manifest['source_sha256'].items()),'Frozen runtime source changed during readiness analysis')
    require(all(sha(ROOT/name)==h for name,h in own_sources.items()),'Readiness source changed during execution')
    summary=dict(status='complete',readiness_passed=all(r['readiness_passed'] for r in results) and all(p['passed'] for p in comparisons),
        runtime_bootstraps_attempted=8,saved_frames_per_N_or_G=980,prefix_physical_actions_reused=976,
        new_physical_actions=0,new_sensor_frames=0,primitive_actions_authorized=0,
        initial_candidate_decisions_only=True,semantic_efficacy_proven=False,task_success_proven=False,
        budget_is_formal_task_budget=False,shared_shape_backend_integrated=False,
        pair_checks=comparisons,cases=[{k:r[k] for k in ('case_index','parent','assignment','mode','status','readiness_passed')} for r in results],
        notes=['The 387/421 total budgets are static readiness suggestions, not frozen full-task budgets.',
            'The paid prefix is correctly charged, but the V21 coverage ledger starts at the last prefix frame and does not reconstruct historical coverage-intent rates.',
            'Four CPU interfaces are checked; this is not learned ANS policy, calibrated RPN uncertainty, natural instance recognition or semantic ranking.',
            'RPN-UQ is called for a read-only return plan. No local action authorization/execution or new gain calibration occurs.',
            'V24 observed-shape completion is not invoked by this inherited V21 readiness runtime.',
            'Observed-safe candidate paths are hypothetical and need live per-action checks; no completed option or quality return is demonstrated.'])
    write(output/'result.json',summary)
    write(output/'timing.json',dict(process_id=os.getpid(),elapsed_s=perf_counter()-started))
    current=read(output/'manifest.json');current['status']='complete';write(output/'manifest.json',current)
    capacity(output/'sources.zip',sum((ROOT/name).stat().st_size for name in own_sources)+8192)
    with zipfile.ZipFile(output/'sources.zip','x',zipfile.ZIP_DEFLATED) as archive:
        for name in own_sources:archive.write(ROOT/name,name)
    write(output/'artifact_hashes.json',{str(p.relative_to(output)):sha(p) for p in sorted(output.rglob('*')) if p.is_file() and p!=output/'artifact_hashes.json'})
    print(json.dumps(dict(output=str(output),readiness_passed=summary['readiness_passed']),indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=DEFAULT_SOURCE)
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--run',action='store_true',help='Only invoke after all four saved prefixes and independent verifications are complete')
    args=parser.parse_args()
    if not args.run: parser.error('Explicit --run is required; no waiting loop or automatic experiment dispatch exists')
    source,output=args.source.resolve(),args.output.resolve()
    require(not output.exists(),'New readiness output path required; existing evidence will not be modified')
    progress=dict(created_output=False,phase='verify_complete_prefix_source')
    try:
        execute(source,output,progress)
    except Exception as error:
        if progress['created_output']:
            failure=dict(status='failed',error=repr(error),traceback=traceback.format_exc(),
                         process_id=os.getpid(),progress=progress)
            try:
                write(output/'failure.json',failure)
                path=output/'manifest.json'
                if path.exists():
                    current=read(path);current['status']='failed';write(path,current)
            except Exception as receipt_error:
                print('Unable to persist failure receipt while preserving 32 MiB: '+repr(receipt_error),file=sys.stderr)
                print(json.dumps(failure,ensure_ascii=False),file=sys.stderr)
        raise


if __name__=='__main__':main()
