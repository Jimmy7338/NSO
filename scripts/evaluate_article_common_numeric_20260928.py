#!/usr/bin/env python3
"""Explicitly frozen common numerical-face evaluation, outside online sources.

draft only writes a protocol-level plan. prepare seals a terminal input snapshot
and audits saved motion without evaluating quality. measure is a separate,
single-attempt operation requiring the exact externally approved plan SHA.
"""
import argparse
from collections import Counter
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from nso.article_prediction_mesh_adapter_v1 import prepare_prediction_mesh_v1, MAX_REMOVED_FACE_AREA_M2
from nso.article_experiment_v1 import EVALUATION, load_reference, source_hashes
from nso.offline_evaluation_v44 import measure_navigation_coverage_v44
from nso.surface_evaluation_v40 import evaluate_surface_v40
from scripts import review_article_episode_20260928 as review_api

VERSION='article.common_numeric_face_evaluation.v1'
PLAN_SCHEMA='article.common_numeric_evaluation_plan.v1'


def sha(path):
    digest=hashlib.sha256()
    with regular(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024**2),b''):digest.update(chunk)
    return digest.hexdigest()


def regular(path):
    path=Path(path).absolute()
    if path.is_symlink() or any(p.is_symlink() for p in path.parents) or not path.is_file():
        raise ValueError('regular file without symlink ancestors required: '+str(path))
    return path


def record(path):
    path=regular(path);return dict(bytes=path.stat().st_size,sha256=sha(path))


def read(path):return json.loads(regular(path).read_text())


def encoded(value):return (json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()


def write(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.is_symlink() or any(p.is_symlink() for p in path.parents):raise ValueError('output links rejected')
    with path.open('xb') as stream:stream.write(data);stream.flush();os.fsync(stream.fileno())


def save(path,value):write(path,encoded(value))


def rooted(name):
    p=Path(name)
    if p.is_absolute() or '..' in p.parts:raise ValueError('workspace-relative path required')
    result=ROOT/p
    if result.is_symlink() or any(a.is_symlink() for a in result.parents):raise ValueError('workspace path links rejected')
    return result


def inventory(root):
    files={}
    for p in sorted(root.rglob('*')):
        if p.is_symlink():raise ValueError('input/output symlink rejected')
        if p.is_file():files[str(p.relative_to(root))]=record(p)
        elif not p.is_dir():raise ValueError('nonregular member rejected')
    if len(files)>1200:raise ValueError('bounded episode inventory required')
    return files


def check_inventory(root,files):
    if inventory(root)!=files:raise ValueError('sealed forensic inventory changed')


def source_pins():
    pins=source_hashes()
    for p in [Path(__file__),ROOT/'nso/article_prediction_mesh_adapter_v1.py',Path(review_api.__file__)]:
        pins[str(p.relative_to(ROOT))]=sha(p)
    return pins


def draft(protocol_path,reviews_root,output_root,plan_path):
    protocol_path=regular(protocol_path);protocol=read(protocol_path)
    if protocol['phase']!='development' or len(protocol['slots'])!=12:
        raise ValueError('fixed original twelve-slot development protocol required')
    if protocol['evaluation']!=EVALUATION or protocol['source_sha256']!=source_hashes():
        raise ValueError('original evaluator settings and source closure must remain unchanged')
    for p in (reviews_root,output_root):
        if not Path(p).absolute().is_relative_to(ROOT):raise ValueError('workspace output/review root required')
    output=Path(output_root).absolute()
    if output.is_relative_to(rooted(protocol['output_relative_path'])):
        raise ValueError('supplement must live outside the original phase directory')
    plan=dict(schema=PLAN_SCHEMA,measurement_version=VERSION,
        activation='External approval of this exact file SHA is required for prepare and measure.',
        protocol_path=str(protocol_path.relative_to(ROOT)),protocol_sha256=sha(protocol_path),
        source_sha256=source_pins(),original_online_source_sha256=protocol['source_sha256'],
        asset_manifest_sha256=protocol['asset_manifest_sha256'],references=protocol['references'],
        slots=protocol['slots'],phase='development',evaluation=EVALUATION,
        maximum_removed_face_area_m2=MAX_REMOVED_FACE_AREA_M2,
        episode_root=protocol['output_relative_path']+'/episodes',
        reviews_root=str(Path(reviews_root).absolute().relative_to(ROOT)),
        output_root=str(output.relative_to(ROOT)),
        selection='Every original protocol slot, independently of outcome. Terminal inputs bind later; pending inputs are not measured.',
        maximum_new_surface_evaluations=12,maximum_measurement_attempts_per_slot=1,
        reuse_rule='Only zero removed faces plus exactly bound original parameters, reference, prediction, coverage and passed independent qualification review.',
        references_changed=False,original_ledgers_changed=False,original_episodes_changed=False,
        new_worlds=0,new_policy_runs=0,new_tsdf_integrations=0,
        changed_faces_interpretation='Derived measurement version; removed faces may shift subsequent random samples.')
    save(plan_path,plan)
    return dict(plan=str(plan_path),plan_sha256=sha(plan_path),measurement_version=VERSION,new_surface_evaluations=0)


def validated_plan(path,expected_sha):
    if sha(path)!=expected_sha:raise ValueError('explicit frozen plan SHA does not match')
    plan=read(path)
    if plan.get('schema')!=PLAN_SCHEMA or plan.get('measurement_version')!=VERSION:
        raise ValueError('supported explicit common measurement version required')
    protocol=read(rooted(plan['protocol_path']))
    if (sha(rooted(plan['protocol_path']))!=plan['protocol_sha256'] or plan['slots']!=protocol['slots']
            or len(plan['slots'])!=12 or plan['phase']!='development'
            or plan['references']!=protocol['references'] or plan['asset_manifest_sha256']!=protocol['asset_manifest_sha256']
            or plan['evaluation']!=EVALUATION or protocol['evaluation']!=EVALUATION
            or plan['maximum_removed_face_area_m2']!=5e-13 or plan['source_sha256']!=source_pins()
            or plan['original_online_source_sha256']!=protocol['source_sha256']
            or plan['maximum_new_surface_evaluations']!=12 or plan['maximum_measurement_attempts_per_slot']!=1):
        raise ValueError('frozen common measurement contract or source changed')
    if plan['episode_root']!=protocol['output_relative_path']+'/episodes':raise ValueError('original episode root required')
    if rooted(plan['output_root']).is_relative_to(rooted(protocol['output_relative_path'])):
        raise ValueError('external supplement output required')
    return plan,protocol


def motion_audit(episode,slot):
    """Actual saved packet/motion checks, distinct from a full policy review."""
    before=review_api.runtime_counts_v41();result=read(episode/'result.json');checks=review_api.Checks()
    count=result['acquired_and_saved_packets'];budget=slot['budget']
    checks.require(type(count) is int and 1<=count<=budget+1,'bounded packet inventory')
    steps=read(episode/'encoding.json')['steps'];encoding={r['artifact']:r for r in steps}
    checks.require(set(encoding)=={f'steps/{i:03d}.json.gz' for i in range(count)},'continuous paid-step inventory')
    checks.require(len(result['actions'])==count-1,'one terminal action record per paid step')
    graph=review_api.PublicPrimitiveGraphV41(read(episode/'public_graph.json'));distance=review_api.distance_queries(graph)
    previous=None;previous_action=None;home=None;first=None;action_counts=Counter();path_length=0.;collisions=0
    minimum_slack=budget
    for i in range(count):
        step,packet=review_api.packet_at(episode,i)
        plain=gzip.decompress(regular(episode/f'steps/{i:03d}.json.gz').read_bytes())
        checks.require(len(plain)==encoding[f'steps/{i:03d}.json.gz']['uncompressed_bytes'],'step decoded length')
        log=json.loads(plain);action='initial_observation' if i==0 else review_api.ACTION_TO_SENSOR_V43[previous_action]
        verified=review_api.validate_step_v43(step,expected_step=i,expected_action=action,expected_previous_pose=previous)
        checks.check(verified==log['accounting'],'saved physical accounting')
        obs_hash=step.rgbd.sha256();checks.check(packet['observation_sha256']==obs_hash,'saved observation hash')
        for kind in ('rgbd','scan'):
            checks.check(record(episode/f'packets/{i:03d}_{kind}.npz')==packet[kind+'_artifact'],'packet binary hash')
        _,scan_hash=review_api._scan_copy(step.scan,None if i==0 else float(i-1))
        mapping=log['mapper'];checks.check(mapping['observation_sha256']==obs_hash and mapping['scan_sha256']==scan_hash,'mapping packet hashes')
        checks.check(mapping['tsdf_integrated'] and mapping['tsdf_integration_count']==i+1,'saved one-to-one integration receipt')
        if i:
            checks.check(result['actions'][i-1]==dict(paid_step=i,controller_action=previous_action,sensor_action=action,
                observation_sha256=obs_hash,collision=step.receipt['collision']),'terminal action bound to actual packet')
            action_counts[action]+=1
        current=graph.state_from_observation(step.rgbd);pose=step.receipt['pose_xyyaw_rad']
        blocked=log['controller_evidence']['safety'].get('newly_blocked_edges',[])
        for item in blocked:graph.block_observed_edge(*item['edge'])
        if blocked:distance=review_api.distance_queries(graph)
        if home is None:home=current;first=list(pose)
        back=distance(current,home);checks.check(back is not None and back<=budget-i,'paid full-pose return reserve')
        if back is not None:minimum_slack=min(minimum_slack,budget-i-back)
        if previous is not None:path_length+=float(np.linalg.norm(np.asarray(pose[:2])-previous[:2]))
        collisions+=bool(step.receipt['collision']);previous=list(pose);previous_action=log['decision']['action']
    returned=review_api.return_pose_matches(previous,first)
    checks.check(result['sensor_status']['returned_xy_and_yaw']==returned,'actual start-end full pose')
    checks.check(result['executed_paid_actions']==count-1==sum(action_counts.values()),'paid-action total')
    checks.check(result['mapper_frames']==result['mapper_tsdf_integrations']==count,'one frame per saved integration')
    checks.check(result['collisions']==collisions,'actual collisions')
    checks.check(result['sensor_status']['remaining_actions']==budget-count+1,'remaining paid budget')
    checks.check(review_api.runtime_counts_v41()==before,'no new physical runtime counters')
    verified=not checks.failures
    return dict(schema='article.posthoc_motion_audit.v1',checks=checks.count,all_checks_passed=verified,failures=checks.failures,
        scope='Saved physical packets, action costs, full-pose return and fusion receipts; not a new full policy-method audit.',
        motion_completion_verified=verified and result['status']=='controller_stop' and returned and collisions==0,
        original_driver_status=result['status'],paid_actions=count-1,budget=budget,collisions=collisions,
        returned_xy_and_yaw=returned,translation_m=path_length,action_counts=dict(action_counts),
        minimum_return_slack_actions=minimum_slack,rgbd_frames=count,scan_frames=count,
        new_worlds=0,new_tsdf_integrations=0,new_surface_evaluations=0)


def prepare(plan_path,plan_sha,run_id):
    plan,protocol=validated_plan(plan_path,plan_sha)
    if run_id not in plan['slots']:raise ValueError('run outside unconditional twelve-slot plan')
    episode=rooted(plan['episode_root'])/run_id;ledger_path=episode.parent.parent/'start_ledger.json'
    ledger_bytes=regular(ledger_path).read_bytes();ledger=json.loads(ledger_bytes)
    entries=[r for r in ledger['entries'] if r['run_id']==run_id]
    if len(entries)>1:raise ValueError('unique terminal ledger entry required')
    if not entries or entries[0]['status']=='reserved':
        return dict(run_id=run_id,status='pending_terminal_input',new_surface_evaluations=0)
    if len(entries)!=1:raise ValueError('unique terminal ledger entry required')
    entry=entries[0];out=rooted(plan['output_root'])/run_id/'prepared'
    if out.exists():raise FileExistsError('forensic preparation is immutable and cannot be replaced')
    started=read(episode/'started.json')
    if (sha(episode/'protocol.json')!=plan['protocol_sha256'] or started['slot']!=plan['slots'][run_id]
            or ledger['protocol_sha256']!=plan['protocol_sha256'] or started['protocol_sha256']!=plan['protocol_sha256']
            or started['source_sha256']!=plan['original_online_source_sha256']):
        raise ValueError('saved protocol/slot must bind frozen plan')
    files=inventory(episode)
    if 'artifact_manifest.json' in files:
        original=read(episode/'artifact_manifest.json')
        if sha(episode/'artifact_manifest.json')!=entry['artifact_manifest_sha256']:
            raise ValueError('original artifact manifest differs from original terminal ledger')
        if files!={**original['files'],'artifact_manifest.json':record(episode/'artifact_manifest.json')}:
            raise ValueError('original complete artifact inventory mismatch')
        if sha(episode/'result.json')!=entry['result_sha256']:raise ValueError('original result ledger pin differs')
    else:
        if entry['status']!='attempt_failed' or sha(episode/'attempt_failure.json')!=entry['result_sha256']:
            raise ValueError('failed attempt must retain its original ledger-bound failure')
    seal=read(episode/'prediction_seal.json') if (episode/'prediction_seal.json').is_file() else None
    availability=bool(seal and set(seal)=={'mesh.npz','mapper.json','occupancy.npz'})
    if availability:
        for name,pin in seal.items():
            if sha(episode/'prediction'/name)!=pin:raise ValueError('pre-existing prediction seal mismatch')
    out.mkdir(parents=True,exist_ok=False)
    write(out/'frozen_plan.json',regular(plan_path).read_bytes());write(out/'captured_phase_ledger.json',ledger_bytes)
    save(out/'original_ledger_entry.json',entry)
    copied=[]
    for name in ('protocol.json','started.json','result.json','attempt_failure.json','artifact_manifest.json',
        'prediction_seal.json','evaluation.json','runtime.json','encoding.json','controller_final.json','public_spec.json','public_workspace.json','public_graph.json'):
        if (episode/name).is_file():write(out/'original_metadata'/name,regular(episode/name).read_bytes());copied.append(name)
    try:
        motion=motion_audit(episode,plan['slots'][run_id]) if availability else dict(motion_completion_verified=False,reason='sealed prediction unavailable')
        receipt=None
        if availability:
            with np.load(episode/'prediction/mesh.npz',allow_pickle=False) as data:
                _,_,receipt=prepare_prediction_mesh_v1(data['vertices'],data['triangles'])
        check_inventory(episode,files)
        payload=dict(schema='article.common_numeric_input_snapshot.v1',run_id=run_id,measurement_version=VERSION,
            frozen_plan_sha256=plan_sha,original_episode=str(episode.relative_to(ROOT)),original_end_to_end_status=entry['status'],
            original_end_to_end_qualified=entry.get('qualified',False),original_artifact_manifest_available='artifact_manifest.json' in files,
            sealed_prediction_available=availability,original_prediction_seal=seal,input_files=files,
            motion_completion=motion,adapter=receipt,quality_measurement_available=False,new_surface_evaluations=0,
            original_episode_modified=False,original_ledger_modified=False,forensic_capture_after_terminal_failure=entry['status']=='attempt_failed')
        save(out/'input_snapshot.json',payload)
    except BaseException as exc:
        save(out/'preparation_failure.json',dict(type=type(exc).__name__,message=str(exc),no_automatic_retry=True));raise
    save(out/'manifest.json',dict(schema='article.posthoc_forensic_manifest.v1',files=inventory(out)))
    return dict(run_id=run_id,status='prepared_no_quality_evaluation',prepared=str(out),sealed_prediction_available=availability,
        motion_completion_verified=motion.get('motion_completion_verified',False),removed_faces=None if receipt is None else receipt['removed_faces'],new_surface_evaluations=0)


def previous_quality(episode,plan,snapshot,reference,coverage):
    """Strictly reuse a prior independent result only for identical eval inputs."""
    if snapshot['adapter']['removed_faces'] or not snapshot['motion_completion']['motion_completion_verified']:
        return None
    if not snapshot['original_end_to_end_qualified'] or not (episode/'artifact_manifest.json').is_file():return None
    folder=rooted(plan['reviews_root'])/episode.name
    if not (folder/'manifest.json').is_file():return None
    review_manifest=read(folder/'manifest.json')
    if review_manifest['reviewer_source_sha256']!=plan['source_sha256']['scripts/review_article_episode_20260928.py']:
        raise ValueError('independent review must use the declared unchanged reviewer')
    for name,row in review_manifest['files'].items():
        if record(folder/name)!=row:raise ValueError('original independent review artifact changed')
    report=read(folder/'review.json');evaluation=read(episode/'evaluation.json');metrics=evaluation['metrics']
    if not report.get('all_checks_passed') or not report.get('qualified'):return None
    if (report['input_manifest_sha256']!=sha(episode/'artifact_manifest.json') or report['protocol_sha256']!=plan['protocol_sha256']
            or report['metrics']!=metrics or not evaluation['qualified']
            or evaluation['source_prediction_sha256']!=snapshot['original_prediction_seal']
            or evaluation['reference_manifest_sha256']!=plan['references'][plan['slots'][episode.name]['scene_id']]['manifest_sha256']
            or metrics['reference_fingerprint']!=reference.fingerprint or evaluation['coverage']!=coverage
            or metrics['threshold_m']!=plan['evaluation']['threshold_m']
            or metrics['prediction_sample_spacing_m']!=plan['evaluation']['sample_spacing_m']
            or metrics['prediction_seed']!=plan['evaluation']['seed']
            or evaluation['no_roi_crop'] is not True or evaluation['semantic_weights_used'] is not False
            or metrics['C_nav']!=coverage['C_nav']):
        raise ValueError('old score cannot be reused under unequal reference/parameters/prediction/qualification')
    return dict(metrics=metrics,source_review=str(folder.relative_to(ROOT)),source_review_manifest=record(folder/'manifest.json'),
        source_review_files={name:record(folder/name) for name in review_manifest['files']},source_evaluation_sha256=sha(episode/'evaluation.json'))


def measure(plan_path,plan_sha,run_id):
    plan,_=validated_plan(plan_path,plan_sha)
    if run_id not in plan['slots']:raise ValueError('run outside frozen inventory')
    folder=rooted(plan['output_root'])/run_id;prepared=folder/'prepared'
    forensic=read(prepared/'manifest.json')
    for name,row in forensic['files'].items():
        if record(prepared/name)!=row:raise ValueError('forensic preparation changed')
    snapshot=read(prepared/'input_snapshot.json')
    if snapshot['frozen_plan_sha256']!=plan_sha or snapshot['run_id']!=run_id:raise ValueError('forensic plan/slot binding mismatch')
    episode=rooted(snapshot['original_episode']);check_inventory(episode,snapshot['input_files'])
    out=folder/'measurement'
    if out.exists():raise FileExistsError('one measurement attempt per slot; existing attempt cannot be retried')
    out.mkdir(parents=True,exist_ok=False)
    save(out/'attempt.json',dict(run_id=run_id,measurement_version=VERSION,frozen_plan_sha256=plan_sha,
        input_snapshot_sha256=sha(prepared/'input_snapshot.json'),automatic_retry=False))
    calls=0;began=time.monotonic()
    result=dict(schema='article.common_numeric_measurement.v1',run_id=run_id,measurement_version=VERSION,
        original_end_to_end_status=snapshot['original_end_to_end_status'],original_end_to_end_qualified=snapshot['original_end_to_end_qualified'],
        motion_completion_verified=snapshot['motion_completion'].get('motion_completion_verified',False),
        quality_measurement_available=False,original_episode_modified=False,original_ledger_modified=False,
        new_worlds=0,new_policy_runs=0,new_tsdf_integrations=0,original_prediction_seal=snapshot['original_prediction_seal'])
    try:
        if not snapshot['sealed_prediction_available']:
            result.update(status='unavailable_sealed_prediction',new_surface_evaluations=0)
        else:
            entry=plan['references'][plan['slots'][run_id]['scene_id']]
            reference,domain,reference_record=load_reference(entry)
            if reference_record['asset_manifest_sha256']!=plan['asset_manifest_sha256']:raise ValueError('reference asset binding changed')
            with np.load(episode/'prediction/occupancy.npz',allow_pickle=False) as data:
                coverage=measure_navigation_coverage_v44(data['belief'],domain,reference_record['coverage'])
            with np.load(episode/'prediction/mesh.npz',allow_pickle=False) as data:
                vertices,triangles,adapter=prepare_prediction_mesh_v1(data['vertices'],data['triangles'])
            if adapter!=snapshot['adapter']:raise ValueError('deterministic adapter differs from prepared diagnosis')
            reused=previous_quality(episode,plan,snapshot,reference,coverage)
            if reused is not None:
                metrics=reused['metrics'];mode='verified_exact_input_reuse'
                save(out/'reuse_proof.json',{k:v for k,v in reused.items() if k!='metrics'})
            else:
                calls=1
                metrics=evaluate_surface_v40(reference,vertices,triangles,C_map=coverage['C_nav'],**plan['evaluation'])
                metrics.pop('C_map');metrics.pop('J');metrics.update(C_nav=coverage['C_nav'],J_nav=coverage['C_nav']*metrics['Q'])
                mode='new_derived_measurement'
            check_inventory(episode,snapshot['input_files'])
            result.update(status='derived_quality_available',quality_measurement_available=True,mode=mode,metrics=metrics,
                coverage=coverage,adapter=adapter,new_surface_evaluations=calls,reference_manifest=entry,
                performance_comparison_motion_gate=result['motion_completion_verified'],
                original_failed_attempt_remains_failed=snapshot['original_end_to_end_status']=='attempt_failed',
                limitation='Posthoc numerical compatibility supplement; does not retroactively repair end-to-end status or prove semantic advantage.')
    except BaseException as exc:
        result.update(status='derived_measurement_failed',error=dict(type=type(exc).__name__,message=str(exc)),
            new_surface_evaluations=calls,automatic_retry=False)
        save(out/'result.json',result);save(out/'manifest.json',dict(files=inventory(out)));raise
    result['elapsed_s']=time.monotonic()-began
    save(out/'result.json',result);save(out/'manifest.json',dict(schema='article.common_numeric_measurement_manifest.v1',files=inventory(out)))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('draft')
    for name in ('protocol','reviews-root','output-root','plan'):p.add_argument('--'+name,type=Path,required=True)
    for command in ('prepare','measure'):
        p=sub.add_parser(command);p.add_argument('--plan',type=Path,required=True);p.add_argument('--plan-sha256',required=True);p.add_argument('--run-id',required=True)
    args=parser.parse_args()
    value=(draft(args.protocol,args.reviews_root,args.output_root,args.plan) if args.command=='draft' else
        prepare(args.plan,args.plan_sha256,args.run_id) if args.command=='prepare' else measure(args.plan,args.plan_sha256,args.run_id))
    print(json.dumps(value,ensure_ascii=False,allow_nan=False))
