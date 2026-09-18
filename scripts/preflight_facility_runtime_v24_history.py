#!/usr/bin/env python3
"""Eight saved-prefix initializations, one read-only N/G decision per state.

Compare the full paid history with sealed endpoint-only initialization results.
No simulator, candidate execution, second mapper pass, or saved mesh copy.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import argparse
from collections import Counter
import gc
import gzip
import hashlib
import json
import signal
from pathlib import Path
import sys
from time import perf_counter
import traceback
import zipfile
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np
import preflight_facility_runtime_v24 as old
from nso.cpu_sensor_contract_v10 import GridTransform,digest,json_value
from nso.decision_replay_v13 import array_hash
from nso.facility_runtime_v24 import FacilityRuntimeV24,facility_components_v24,PREFIX_INTENT_SOURCE,VERSION_V24

# Imported helpers remain unchanged; no global/class replacement is performed.
require,read,sha=old.require,old.read,old.sha
capacity=old.capacity
inventory,versions=old.inventory,old.versions
geometry_evidence,make_args,route_audit=old.geometry_evidence,old.make_args,old.route_audit
PAID_PREFIX,TOTAL_BUDGET=dict(old.PAID_PREFIX),dict(old.TOTAL_BUDGET)
DEFAULT_SOURCE=old.DEFAULT_SOURCE
DEFAULT_BASELINE=old.DEFAULT_OUTPUT
DEFAULT_OUTPUT=ROOT/'audit_results/facility_runtime_v24_history_readiness_20260915'
PROTOCOL=ROOT/'docs/research/V24_PREFIX_HISTORY_READINESS_PROTOCOL_20260915.md'
PREFLIGHT=ROOT/'audit_results/facility_runtime_v24_prefix_history_preflight_20260915'
SHAPE_SOURCE_MANIFEST=ROOT/'audit_results/facility_choice_v24_shape_prefix_20260915/manifest.json'
EXPECTED_SHAPE_MANIFEST_SHA='22777ba8918161cf5a963fd837d15c24e6ce43f0b86d3a073faac2c65015a658'
EXPECTED_NEW_SOURCE={
 'nso/facility_runtime_v24.py':'2a9e76786e235cfad88f0f1f35540b460e4946213d150952551ddb0bd1eba6a5',
 'tests/virtual3d/test_facility_runtime_v24.py':'6e3c9550c81124b64721a8f69b67d723281d35b16d9ca19f188f56b26c8f271d'}
MAX_OUTPUT_BYTES=16*1024**2
FAILURE_RESERVE_BYTES=64*1024


class ReadinessInterrupted(BaseException):
    """Bypass per-case error handling so an interrupted batch cannot continue."""


def stop_on_signal(number,frame):
    raise ReadinessInterrupted('Readiness interrupted by signal '+str(number))


def output_capacity(path,required_bytes,*,emergency=False):
    used=sum(p.stat().st_size for p in path.parent.iterdir() if p.is_file())
    ceiling=MAX_OUTPUT_BYTES if emergency else MAX_OUTPUT_BYTES-FAILURE_RESERVE_BYTES
    require(used+required_bytes+4096<=ceiling,'Readiness 16 MiB artifact cap would be exceeded')
    capacity(path,required_bytes+4096)


def write(path,value,*,emergency=False):
    size=len((json.dumps(json_value(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode())
    output_capacity(path,size,emergency=emergency)
    return old.write(path,value)


def write_gzip(path,value):
    size=len(gzip.compress(json.dumps(json_value(value),ensure_ascii=False,allow_nan=False).encode(),mtime=0))
    output_capacity(path,size)
    return old.write_gzip(path,value)


def verify_protection(prefix_manifest):
    require(sha(SHAPE_SOURCE_MANIFEST)==EXPECTED_SHAPE_MANIFEST_SHA,'Prior 147-source manifest changed')
    protected=read(SHAPE_SOURCE_MANIFEST)['source_sha256']
    require(len(protected)==147 and len(prefix_manifest['source_sha256'])==145,'Expected original 145/147 frozen sources')
    require(all(protected.get(k)==v for k,v in prefix_manifest['source_sha256'].items()),'147 source list differs from original145')
    require(all(sha(ROOT/k)==v for k,v in protected.items()),'A prior protected source changed')
    files=read(PREFLIGHT/'inventory.json')
    require({p.name for p in PREFLIGHT.iterdir() if p.is_file() and p.name!='inventory.json'}
            ==set(files)=={'result.json','test_stdout.txt'},'Complete nine-test preflight required')
    for name,row in files.items():
        require(sha(PREFLIGHT/name)==row['sha256'] and (PREFLIGHT/name).stat().st_size==row['bytes'],
                'Preflight artifact changed: '+name)
    proof=read(PREFLIGHT/'result.json'); stdout=(PREFLIGHT/'test_stdout.txt').read_text()
    require(proof['status']=='passed' and proof['exit_code']==0 and 'Ran 9 tests' in stdout
            and stdout.rstrip().endswith('OK'),'Nine analytic tests must have passed')
    require(proof['source_sha256']==EXPECTED_NEW_SOURCE and all(sha(ROOT/k)==v for k,v in EXPECTED_NEW_SOURCE.items()),
            'Runtime/test differs from passing frozen implementation')
    return dict(protected_source_sha256=protected,shape_manifest_sha256=sha(SHAPE_SOURCE_MANIFEST),
        preflight_inventory_sha256=sha(PREFLIGHT/'inventory.json'),preflight_result_sha256=sha(PREFLIGHT/'result.json'),
        preflight_files=files,new_source_sha256=dict(EXPECTED_NEW_SOURCE))


def verify_baseline(root,source,input_hash):
    manifest,summary=read(root/'manifest.json'),read(root/'result.json')
    require(manifest['status']==summary['status']=='complete','Sealed prior readiness required')
    require(Path(manifest['input_root']).resolve()==source.resolve() and manifest['input_inventory_sha256']==input_hash,
            'Old/new readiness require identical physical prefix input')
    require(summary['runtime_bootstraps_attempted']==8 and summary['new_physical_actions']==0,'Old readiness scope differs')
    root_hash=inventory(root)
    with zipfile.ZipFile(root/'sources.zip') as archive:
        require(set(archive.namelist())==set(manifest['source_sha256']),'Old readiness archived source set differs')
        for name,expected in manifest['source_sha256'].items():
            require(sha(ROOT/name)==expected and hashlib.sha256(archive.read(name)).hexdigest()==expected,
                    'Old readiness source changed: '+name)
    expected={(i,mode) for i in range(4) for mode in ('N','G')}
    require(len(summary['cases'])==8 and {(r['case_index'],r['mode']) for r in summary['cases']}==expected,
            'Old readiness row identities differ')
    rows={}
    for index,mode in sorted(expected):
        row=read(root/f'case_{index:02d}_{mode}.json')
        require(row['status']=='complete' and row['case_index']==index and row['mode']==mode,
                'Old initialization must be complete; readiness rejection remains valid')
        require(row['coverage_prefix_rate_history_reconstructed'] is False,'Old endpoint-only comparison required')
        rows[index,mode]=row
    return rows,dict(root=str(root),artifact_inventory_sha256=root_hash,manifest_sha256=sha(root/'manifest.json'),
                     source_sha256=manifest['source_sha256'])


def prefix_coverage_audit(shared,packets,mapper):
    """Check actual receipt arithmetic, without reconstructing a second map."""
    ledger=shared['coverage_v21']; coverage=ledger.snapshot()
    receipt=shared['v24_prefix_coverage_receipt']; paid=len(packets)-1
    require(receipt['version']==VERSION_V24 and receipt['intent_source']==PREFIX_INTENT_SOURCE
            and receipt['external_prefix_coverage_intent'] is True,'Explicit common prefix intent missing')
    require(receipt['mapper_updates']==mapper.frames==paid+1
            and receipt['paid_actions']==coverage['observed_actions']==paid
            and receipt['initial_action_id']==0 and receipt['final_action_id']==ledger.action_id==paid,
            'Prefix action interval or single mapper accounting differs')
    rows=receipt['paid_observations']; count=receipt['initial_coverage']['known_cells']; gains=losses=0
    require(len(rows)==len(ledger.events)==paid,'Each paid packet needs one coverage event')
    for p,row,event in zip(packets[1:],rows,ledger.events):
        require(row['action_id']==p.action_id and row['frame_id']==p.frame_id and row['action']==p.action
                and row['mapper_frames']==p.action_id+1 and row['coverage_event']==event and event['action_id']==p.action_id,
                'Coverage event is not tied to its actual packet')
        require(event['coverage_intent'] is True and event['predicted_union_cells'] is None
                and event['realized_predicted_cells'] is None,'Scripted history cannot invent predictions or select intents')
        require(event['known_cells_before']==count,'Known-cell chain is discontinuous')
        gains+=event['actual_new_known_cells']; losses+=event['known_lost_cells']; count+=event['net_known_gain_cells']
        require(event['net_known_gain_cells']==event['actual_new_known_cells']-event['known_lost_cells']
                and count==event['known_cells_after'],'Coverage gain/loss units differ')
    require(count==coverage['known_cells']==int(np.count_nonzero(mapper.belief!=-1))
            and receipt['initial_coverage']['task_cells']==coverage['task_cells']==mapper.belief.size
            and np.array_equal(ledger.belief,mapper.belief),'ROI progress differs from authoritative belief')
    require(receipt['final_coverage']==coverage and coverage['pending_prefix'] is None
            and coverage['scan_observations']==paid+1 and coverage['union_yield']==.5,
            'Final flush, real radar count or untouched yield prior differs')
    require(rows[-1]['belief_sha256']==digest(mapper.belief),'Final belief receipt differs')
    blocks=[ledger.events[i:i+5] for i in range(0,paid,5)][-6:]
    expected=[(b[0]['action_id'],b[-1]['action_id'],len(b),sum(e['actual_new_known_cells'] for e in b),
               sum(e['known_lost_cells'] for e in b)) for b in blocks]
    actual=[(r['first_action_id'],r['last_action_id'],r['paid_actions'],r['actual_new_known'],r['known_lost'])
            for r in coverage['coverage_prefixes']]
    require(actual==expected,'Original fixed last-six-prefix window changed')
    cost=sum(x[2] for x in expected); net=sum(x[3]-x[4] for x in expected)
    rate=.5*max(0,net)/cost if cost else None
    require(coverage['conservative_rate']==rate and coverage['rate_history_paid_actions']==cost
            and coverage['rate_history_net_known_gain']==net,'Rate omitted zero/loss/paid actions or used lifetime mean')
    return dict(receipt=receipt,full_coverage_event_sha256=digest(ledger.events),all_paid_events_verified=True,
        actual_known_gains=gains,actual_known_losses=losses,actual_net_known_gain=gains-losses,
        all_paid_action_counts=dict(Counter(p.action for p in packets[1:])),
        zero_gain_actions=sum(e['net_known_gain_cells']==0 for e in ledger.events),
        negative_gain_actions=sum(e['net_known_gain_cells']<0 for e in ledger.events),
        final_window_first_action_id=expected[0][0],final_window_paid_actions=cost,
        final_window_net_known_gain=net,independently_checked_rate=rate,
        all_prefix_actions_have_public_coverage_intent=True,no_predictions_invented=True,
        class_or_evaluator_used=False,reconstructed_with_second_mapper=False)


def compare_baseline(current,previous):
    require(tuple(current[k] for k in ('case_index','parent','assignment','mode'))
            ==tuple(previous[k] for k in ('case_index','parent','assignment','mode')),'Old/new row identity differs')
    prior_coverage,current_coverage=previous['coverage_budget_bootstrap'],current['coverage_budget_bootstrap']
    invariants={k:current[k]==previous[k] for k in ('candidate_pool_sha256','asset_geometry_sha256',
        'common_N_scores','chosen_scores','paid_ledger','geometry')}
    invariants['same_ROI_progress']=all(prior_coverage[k]==current_coverage[k] for k in
        ('known_cells','task_cells','deficit_cells','planning_coverage'))
    def budget_rows(row):
        return {a['candidate_id']:a['v21_coverage']['coverage_budget'] for a in row['selection']['score_audit']}
    def selected(row):
        value=row['selected']
        return None if value is None else {k:value.get(k) for k in
            ('candidate_id','group','asset_index','pose','outbound_cost','return_cost','cost','v21_coverage_intent')}
    prior,now=budget_rows(previous),budget_rows(current)
    return dict(comparison_scope='sealed endpoint-only versus full paid history',old_runtime_reexecuted=False,
        invariants=invariants,only_declared_initialization_changed=all(invariants.values()),
        old_coverage=prior_coverage,new_coverage=current_coverage,
        old_all_budget_allowed_indices=previous['all_budget_allowed_indices'],new_all_budget_allowed_indices=current['all_budget_allowed_indices'],
        old_runtime_admitted_indices=previous['runtime_admitted_indices'],new_runtime_admitted_indices=current['runtime_admitted_indices'],
        old_denial_reason_counts=dict(Counter(v['reason'] for v in prior.values() if not v['allowed'])),
        new_denial_reason_counts=dict(Counter(v['reason'] for v in now.values() if not v['allowed'])),
        old_selected=selected(previous),new_selected=selected(current),actual_selected_changed=selected(previous)!=selected(current),
        candidate_budget_comparison=[dict(candidate_id=i,old_budget=prior.get(i),new_budget=now.get(i))
                                    for i in sorted(set(prior)|set(now))],
        later_quality_actions=None,target_arrival=None,terminal_quality=None,
        budget_is_formal_task_budget=False,task_or_semantic_efficacy_proven=False)


# run_mode preserves the frozen prior diagnostic body; only V24 startup,
# complete receipt checks, and the IGCR bootstrap method identity are added.
def run_mode(record, mode, config, shape, packets, anchor):
    case, saved = record['case'], record['saved']; paid = saved['paid_actions']; total = TOTAL_BUDGET[case['parent']]
    components = facility_components_v24(make_args(mode), shape)
    runtime = FacilityRuntimeV24(components, 1, shape)
    runtime.start_sensor_episode(0, config=config, transform=GridTransform(shape, config.resolution_m),
        packets=packets, paid_prefix_actions=paid, total_budget=total, return_anchor=anchor,
        external_prefix_coverage_intent=True)
    backend = components._cpu_backend; state = runtime.states[0]; shared = backend.scenes[0]; mapper = state['mapper']
    ledger = shared['ledger'].snapshot(); coverage = shared['coverage_v21'].snapshot()
    history_audit = prefix_coverage_audit(shared, packets, mapper)
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
    require(shared['coverage_v21'].snapshot() == coverage, 'Planning changed the observed coverage ledger')
    methods = [(c['module'],c['method']) for c in backend.calls]
    expected_calls = {('IGCR','bootstrap'),('IGCR','coverage_bootstrap_v24'),('OV-SDF','update_semantic'),
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
        coverage_prefix_rate_history_reconstructed=True, prefix_history_audit=history_audit,
        coverage_event_sha256=history_audit['full_coverage_event_sha256'], paid_ledger=ledger, geometry=geometry,
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




def pair_checks(rows,source_result):
    checks=old.pair_checks(rows,source_result)
    by={(r['case_index'],r['mode']):r for r in rows if r['status']=='complete'}
    for row in checks:
        if row['kind']=='N_G_common_pool':
            group=[by.get((row['case_index'],mode)) for mode in ('N','G')]
        else:
            group=[r for r in rows if r.get('parent')==row['parent'] and r.get('mode')==row['mode'] and r['status']=='complete']
        equal=len(group)==2 and all(group) and all(group[0][k]==group[1][k] for k in
            ('coverage_event_sha256','coverage_budget_bootstrap'))
        row['actual_coverage_history_equal']=bool(equal); row['passed']=bool(row['passed'] and equal)
    return checks


def execute(source,baseline,output,progress):
    # Existing capacity helper reserves 32 MiB beyond requested bytes: require
    # 64 MiB before any expensive bootstrap, retain 32 MiB at every write.
    capacity(output/'manifest.json',32*1024**2)
    manifest,source_result,records,input_hash=old.verify_source(source)
    protection=verify_protection(manifest)
    previous,old_proof=verify_baseline(baseline,source,input_hash)
    paths=(Path(__file__).resolve(),PROTOCOL,Path(old.__file__).resolve(),old.PROTOCOL,
           *(ROOT/name for name in EXPECTED_NEW_SOURCE))
    own_sources={str(p.relative_to(ROOT)):sha(p) for p in paths}
    capacity(output/'manifest.json',16384); output.mkdir(parents=True); progress['created_output']=True
    write(output/'manifest.json',dict(status='running',input_root=str(source),input_inventory_sha256=input_hash,
        old_readiness=old_proof,protection=protection,source_sha256=own_sources,protocol=str(PROTOCOL),
        runtime_version=VERSION_V24,total_diagnostic_budget=TOTAL_BUDGET,task_asset_count=2,
        output_cap_bytes=MAX_OUTPUT_BYTES,soft_wall_time_limit_s=progress['max_seconds'],
        public_external_prefix_coverage_intent=True,new_physical_actions=0,new_sensor_frames=0,
        exactly_eight_new_mapper_initializations=True,old_runtime_reexecuted=False))
    started=perf_counter(); results=[]
    for record in records:
        case=record['case']; progress.update(case_index=case['index'],phase='load_saved_actual_history')
        config,shape,packets,anchor=old.load_history(record)
        for mode in ('N','G'):
            progress.update(mode=mode,phase='single_mapper_initialization_and_one_read_only_decision')
            try:
                result,calls,audit=run_mode(record,mode,config,shape,packets,anchor)
                comparison=compare_baseline(result,previous[case['index'],mode])
                result['endpoint_baseline_comparison']=comparison
                result['checks']['only_declared_history_initialization_changed']=comparison['only_declared_initialization_changed']
                result['readiness_passed']=all(result['checks'].values())
                for label,rows in [('module_calls',calls),('runtime_audit',audit)]:
                    write_gzip(output/f'case_{case["index"]:02d}_{mode}_{label}.json.gz',rows)
            except Exception as error:
                result=dict(status='failed',case_index=case['index'],parent=case['parent'],assignment=case['assignment'],mode=mode,
                    error=repr(error),traceback=traceback.format_exc(),readiness_passed=False,new_physical_actions=0,new_sensor_frames=0)
            results.append(result); write(output/f'case_{case["index"]:02d}_{mode}.json',result)
            print(f'case={case["index"]} mode={mode} status={result["status"]} ready={result["readiness_passed"]}',flush=True)
        del packets; gc.collect()
    comparisons=pair_checks(results,source_result); progress['phase']='final_source_and_evidence_verification'
    require(inventory(source)==input_hash,'Saved physical prefix evidence changed')
    require(inventory(baseline)==old_proof['artifact_inventory_sha256'],'Old readiness artifacts changed')
    require(versions()==manifest['versions'],'Dependency environment changed')
    require(verify_protection(manifest)==protection,'Frozen original147 or new runtime/test sources changed')
    require(all(sha(ROOT/name)==h for name,h in own_sources.items()),'New readiness source changed during execution')
    statuses=[{k:r[k] for k in ('case_index','parent','assignment','mode','status','readiness_passed')} for r in results]
    diagnostics=[{k:r[k] for k in ('case_index','parent','assignment','mode','selected','coverage_budget_bootstrap',
        'per_observed_asset_candidate_capacity','endpoint_baseline_comparison')} for r in results if r['status']=='complete']
    summary=dict(status='complete',readiness_passed=all(r['readiness_passed'] for r in results) and all(p['passed'] for p in comparisons),
        runtime_bootstraps_attempted=8,runtime_bootstraps_complete=sum(r['status']=='complete' for r in results),
        saved_frames_per_N_or_G=980,actual_saved_frame_integrations_if_all_complete=1960,prefix_physical_actions_reused=976,
        new_physical_actions=0,new_sensor_frames=0,primitive_actions_authorized=0,old_runtime_reexecuted=False,
        second_mapper_for_history_or_comparison=False,saved_mesh_copies_created=0,initial_candidate_decisions_only=True,
        full_paid_prefix_coverage_history_reconstructed=all(r['status']=='complete'
            and r['coverage_prefix_rate_history_reconstructed'] for r in results),
        semantic_efficacy_proven=False,task_success_proven=False,
        budget_is_formal_task_budget=False,shared_shape_backend_integrated=False,pair_checks=comparisons,cases=statuses,diagnostics=diagnostics,
        notes=[
            'Only initialization changes: every actual prefix turn, return and zero/loss action has explicit public coverage intent.',
            'The same authoritative mapper and same V21 coverage ledger consume each prefix once; the original last-six-prefix rate remains.',
            'The resulting rate can still be zero; G can still reject every quality role or select a short rotation instead of a facility.',
            'Candidate geometry and N/G score values must match sealed old readiness; only budget decisions may change. No candidate is executed.',
            'C_ROI is known cells over a fixed public ROI, not true reachable coverage, a bound, or evaluator-based correction.',
            'The 387/421 total budgets remain diagnostic suggestions. Actual future per-action safety, full-task success and early semantic information need a separate protocol.',
            'No prefix warm-up, hidden candidate-window rays, GT service regions, inferred shape or learned semantic ranking is used.',
            'Prior145/147 sources, frozen passing new runtime/tests and old evidence are preserved. Failures and null/zero rates remain.'])
    write(output/'result.json',summary); write(output/'timing.json',dict(process_id=os.getpid(),elapsed_s=perf_counter()-started))
    current=read(output/'manifest.json'); current['status']='complete'; write(output/'manifest.json',current)
    output_capacity(output/'sources.zip',sum((ROOT/name).stat().st_size for name in own_sources)+1024**2)
    with zipfile.ZipFile(output/'sources.zip','x',zipfile.ZIP_DEFLATED) as archive:
        for name in own_sources: archive.write(ROOT/name,name)
    write(output/'artifact_hashes.json',{str(p.relative_to(output)):sha(p) for p in sorted(output.rglob('*'))
        if p.is_file() and p!=output/'artifact_hashes.json'})
    print(json.dumps(dict(output=str(output),readiness_passed=summary['readiness_passed']),indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=DEFAULT_SOURCE)
    parser.add_argument('--baseline',type=Path,default=DEFAULT_BASELINE)
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--max-seconds',type=int,default=600,help='Soft wall-time limit; records failure at the next interruptible Python boundary')
    parser.add_argument('--run',action='store_true',help='Explicit eight-state saved-history readiness; no candidate execution')
    args=parser.parse_args()
    if not args.run: parser.error('Explicit --run required; no automatic execution or wait loop')
    require(args.max_seconds>0,'Positive wall-time limit required')
    source,baseline,output=args.source.resolve(),args.baseline.resolve(),args.output.resolve()
    require(not output.exists(),'Fresh readiness output required; existing evidence will not be replaced')
    progress=dict(created_output=False,phase='verify_frozen_inputs',max_seconds=args.max_seconds)
    for number in (signal.SIGINT,signal.SIGTERM,signal.SIGALRM):
        signal.signal(number,stop_on_signal)
    signal.setitimer(signal.ITIMER_REAL,args.max_seconds)
    try:
        execute(source,baseline,output,progress)
    except BaseException as error:
        signal.setitimer(signal.ITIMER_REAL,0)
        if progress['created_output']:
            failure=dict(status='failed',error=repr(error),traceback=traceback.format_exc(),process_id=os.getpid(),progress=progress)
            try:
                # Retain an interrupted atomic manifest temporary file without
                # letting its exclusive-create name prevent the failure seal.
                temporary=output/'manifest.json.tmp'
                if temporary.exists(): temporary.replace(output/'interrupted_manifest.json.tmp')
                write(output/'failure.json',failure,emergency=True)
                path=output/'manifest.json'
                if path.exists():
                    current=read(path); current['status']='failed'; write(path,current,emergency=True)
            except Exception as receipt_error:
                print('Unable to persist failure receipt while retaining 32 MiB: '+repr(receipt_error),file=sys.stderr)
                print(json.dumps(failure,ensure_ascii=False),file=sys.stderr)
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)


if __name__=='__main__': main()
