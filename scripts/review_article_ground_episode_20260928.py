#!/usr/bin/env python3
"""Independent read-only Ground V2 episode contract and receipt audit.

The unchanged ArticleV1 reviewer verifies compatible physics/packet/map/route
receipts in a nested report. Extra checks bind the new protocol, common ground
frontend, paired baseline, global start cap and numerical-face preprocessing.
No World, controller execution, TSDF integration or quality remeasurement.
"""
import argparse
from collections import Counter
from copy import deepcopy
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
import zipfile

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import review_article_episode_20260928 as base
from nso.observed_instances_article_ground_v2 import GROUND_POLICY, ground_mask_from_observation

SCHEMA='article.experiment_protocol.ground_v2'
PREPROCESSING_SCHEMA='article.prediction_numeric_face_adapter.v1'
METRIC_VERSION='article.common_numeric_face_evaluation.v1'
EVALUATION=dict(threshold_m=.05,sample_spacing_m=.3,seed=4002,max_samples=1000000)
CAPS=dict(development=12,main=96,ablation=24)
GROUND_FIELDS={'common_ground_association','ground_association_policy','ground_removal_scope'}
METHODS={'G','B','S','NBV'}


def sha(path):return base.sha(path)
def read(path):return base.read(path)
def require(value,message):
    if not value:raise ValueError(message)


def array_sha(value):
    value=np.asarray(value)
    header=json.dumps(dict(dtype=value.dtype.str,shape=list(value.shape)),sort_keys=True,separators=(',',':')).encode()+b'\n'
    return hashlib.sha256(header+np.ascontiguousarray(value).tobytes()).hexdigest()


def exact_or_numeric(actual,expected,path='receipt',*,atol=1e-12):
    """Exact discrete fields; bounded scalar roundoff, never altered gates."""
    if isinstance(expected,dict):
        require(isinstance(actual,dict) and set(actual)==set(expected),path+': dictionary keys')
        for key,value in expected.items():exact_or_numeric(actual[key],value,path+'/'+key,atol=atol)
    elif isinstance(expected,list):
        require(isinstance(actual,list) and len(actual)==len(expected),path+': list length')
        for index,(a,b) in enumerate(zip(actual,expected)):exact_or_numeric(a,b,path+'/'+str(index),atol=atol)
    elif type(expected) is float:
        require(type(actual) is float and math.isfinite(actual) and abs(actual-expected)<=atol,path+': finite float')
    else:
        require(type(actual) is type(expected) and actual==expected,path+': exact value')


def verify_pairing(protocol,baseline):
    """Pure independent protocol comparison; no runner validator imported."""
    require(protocol.get('schema')==SCHEMA and protocol.get('status')=='frozen','strict frozen ground protocol schema')
    require(protocol.get('phase')=='ablation','this reviewer is for declared front-end ablation only')
    require(protocol.get('frozen_before_world_creation') is True,'pre-world freeze required')
    require(baseline.get('schema')=='article.experiment_protocol.v1' and baseline.get('status')=='frozen'
            and baseline.get('phase')=='development','original development baseline schema')
    require(protocol.get('metric_preprocessing')==dict(schema=PREPROCESSING_SCHEMA,
        applied_equally_to_paired_baseline=True),'explicit common derived preprocessing version')
    require(protocol['evaluation']==baseline['evaluation']==EVALUATION,'common frozen evaluation settings')
    for key in ('asset_root','asset_manifest_sha256','mapper','references'):
        require(protocol[key]==baseline[key],'paired common '+key)
    common=dict(protocol['controller'])
    require(common.pop('ground_association',None) is True,'common ground switch required')
    require(common==baseline['controller'],'only ground association changes controller options')
    slots=protocol['slots'];expected={(f'ART1_{family}_DEV',method) for family in ('AISLE','CELL','LOOP') for method in METHODS}
    require(len(slots)==12 and {(s['scene_id'],s['method']) for s in slots.values()}==expected,'complete declared twelve-cell matrix')
    paired=[]
    for slot in slots.values():
        require(set(slot)=={'scene_id','method','budget','noise_seed','paired_baseline_run_id'},'exact paired slot fields; no override')
        require(type(slot['budget']) is int and slot['budget']==160 and type(slot['noise_seed']) is int
                and slot['noise_seed']==92801,'fixed paid budget and noise seed')
        key=slot['paired_baseline_run_id'];paired.append(key)
        require(isinstance(key,str) and key in baseline['slots'],'unknown paired baseline run')
        target=baseline['slots'][key]
        require(set(target)=={'scene_id','method','budget','noise_seed'},'baseline has no per-slot override')
        require(all(slot[k]==target[k] for k in target),'paired method/scene/budget/noise')
    require(len(set(paired))==12 and set(paired)==set(baseline['slots']),'one-to-one full baseline pairing')
    runtime=protocol.get('numerical_runtime')
    require(isinstance(runtime,dict) and all(runtime.get(k) for k in
        ('python_executable','python_version','numpy_version','numpy_path','numpy_build','scipy_version','scipy_path','scipy_build')),
        'complete declared numerical runtime build and path fingerprint')
    require(runtime.get('thread_environment')==dict(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1'),
        'three single-thread numerical controls')
    return True


def verify_preprocessing_arrays(vertices,triangles,receipt):
    """Independent face gate and SHA arithmetic, without calling the adapter."""
    v=np.asarray(vertices);t=np.asarray(triangles)
    require(v.ndim==2 and v.shape[1]==3 and v.dtype.kind in 'fiu' and len(v)<=2000000
        and np.isfinite(v).all(),'bounded finite raw vertices')
    require(t.ndim==2 and t.shape[1]==3 and t.dtype.kind in 'iu' and len(t)<=1000000,'bounded integer raw faces')
    require(not t.size or (t.min()>=0 and t.max()<len(v)),'raw face indices in range')
    xyz=np.asarray(v,dtype=np.float64)[t]
    with np.errstate(over='ignore',invalid='ignore'):
        twice=np.linalg.norm(np.cross(xyz[:,1]-xyz[:,0],xyz[:,2]-xyz[:,0]),axis=1)
    require(np.isfinite(twice).all(),'nonfinite face geometry rejected, not removed')
    area=twice*.5;remove=np.flatnonzero(twice<=1e-12);keep=np.flatnonzero(twice>1e-12)
    retained=t[keep];removed_xyz=xyz[remove]
    edge=np.linalg.norm(removed_xyz-np.roll(removed_xyz,1,axis=1),axis=2)
    expected=dict(schema=PREPROCESSING_SCHEMA,status='geometry_prepared_no_quality_evaluation',
        fixed_maximum_removed_face_area_m2=5e-13,equivalent_frozen_v40_rule='reject twice_area <= 1e-12',
        original_vertices=len(v),original_faces=len(t),retained_faces=len(keep),removed_faces=len(remove),
        removed_original_face_indices=remove.tolist(),removed_total_area_m2=float(area[remove].sum()),
        removed_maximum_face_area_m2=float(area[remove].max()) if len(remove) else 0.,
        removed_maximum_edge_m=float(edge.max()) if edge.size else 0.,
        original_total_area_m2=float(area.sum()),retained_total_area_m2=float(area[keep].sum()),
        original_vertices_array_sha256=array_sha(v),original_triangles_array_sha256=array_sha(t),
        adapted_vertices_array_sha256=array_sha(v),adapted_triangles_array_sha256=array_sha(retained),
        retained_original_face_indices_sha256=array_sha(keep),
        original_vertices_values_order_and_dtype_unchanged=True,retained_faces_values_order_and_dtype_unchanged=True,
        prediction_arrays_exactly_unchanged=not len(remove),gt_reference_read=False,semantic_labels_read=False,roi_applied=False,
        new_surface_evaluations=0,new_tsdf_integrations=0,new_worlds=0,
        quality_reuse_eligibility='requires independent equality of prediction, reference and all evaluator settings' if not len(remove)
            else 'new derived measurement required; cannot reuse original score',
        sampling_warning='Removing a face may shift later random samples in the frozen evaluator; no metric-equivalence claim is made.')
    allowed=set(expected)|{'source_prediction_sha256','original_prediction_seal_sha256','comparison_rule'}
    require(set(expected)<=set(receipt)<=allowed,'strict adapter receipt fields')
    # Exact masks, counts, threshold and hash bindings; small aggregate sums
    # can differ only by ordinary arithmetic roundoff, not face eligibility.
    for key,value in expected.items():
        tolerance=(0. if key=='fixed_maximum_removed_face_area_m2' else
            max(abs(value)*1e-12,1e-30) if type(value) is float else 0.)
        exact_or_numeric(receipt[key],value,'preprocessing/'+key,atol=tolerance)
    return dict(raw_vertices=len(v),raw_faces=len(t),removed_faces=len(remove),retained_faces=len(keep),
        removed_original_face_indices=remove.tolist(),removed_total_area_m2=float(area[remove].sum()),
        adapted_vertices_array_sha256=array_sha(v),adapted_triangles_array_sha256=array_sha(retained))


def verify_evaluation_metadata(evaluation,protocol,receipt_sha):
    require(evaluation.get('schema')=='article.ground_v2.canonical_evaluation.v1'
        and evaluation.get('metric_version')==METRIC_VERSION,'explicit common derived metric version')
    require(evaluation.get('evaluation_settings')==protocol['evaluation']==EVALUATION,'evaluation actual/settings declaration')
    expected_pins={name:protocol['source_sha256'][name] for name in
        ('nso/surface_evaluation_v40.py','nso/article_prediction_mesh_adapter_v1.py')}
    require(evaluation.get('evaluation_source_sha256')==expected_pins,'derived evaluator and adapter source pins')
    require(evaluation.get('metric_preprocessing')==dict(schema=PREPROCESSING_SCHEMA,
        receipt='prediction_preprocessing.json',receipt_sha256=receipt_sha),'derived preprocessing artifact binding')
    require(evaluation.get('original_prediction_files_unchanged') is True and evaluation.get('no_roi_crop') is True
        and evaluation.get('semantic_weights_used') is False and evaluation.get('fixed_target_instances')==4,
        'unchanged whole prediction, no ROI/semantic weighting, four-instance denominator')


def verify_archive(protocol,checks,label):
    pins=protocol['source_sha256'];archive=base.safe_file(ROOT,protocol['source_archive'])
    checks.require(sha(archive)==protocol['source_archive_sha256'],label+' source archive SHA')
    with zipfile.ZipFile(archive) as stream:
        checks.require(len(stream.namelist())==len(set(stream.namelist())) and set(stream.namelist())==set(pins),label+' complete source closure')
        for name,pin in pins.items():
            checks.require(hashlib.sha256(stream.read(name)).hexdigest()==pin,label+' archived source '+name)
            path=base.safe_file(ROOT,name)
            checks.require(path.is_file() and sha(path)==pin,label+' live source '+name)


def verify_source_relationship(current,baseline):
    overlap=set(current)&set(baseline)
    require(all(current[name]==baseline[name] for name in overlap)
        and set(baseline)-set(current)=={'scripts/run_article_experiment_20260928.py'},
        'shared execution sources unchanged; only replaced old CLI omitted from new dependency closure')


def verify_registry(phase,protocol_sha,entry,checks):
    ledger_path=phase/'start_ledger.json';path=phase.parent/'execution_registry.json';registry=read(path)
    checks.require(registry['schema']=='article.execution_registry.v1' and registry['phase_caps']==CAPS,'shared cumulative registry schema/caps')
    counts=Counter(r['phase'] for r in registry['entries'])
    checks.require(all(k in CAPS and v<=CAPS[k] for k,v in counts.items()),'cumulative starts obey all phase caps')
    matches=[r for r in registry['entries'] if r['run_id']==entry['run_id'] and r['ledger']==str(ledger_path.resolve())]
    checks.require(len(matches)==1 and matches[0]['phase']=='ablation' and
        matches[0]['protocol_sha256']==protocol_sha,'this attempt consumes one global ablation slot')
    identities=[(r['ledger'],r['run_id']) for r in registry['entries']]
    checks.require(len(set(identities))==len(identities),'no duplicated global reservation identity')
    return dict(path=str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
                sha256=sha(path),counts=dict(counts),phase_caps=CAPS,total_reserved_attempts=len(identities))


def paired_configuration(baseline_phase,entry,checks):
    """Use original closed seal or explicitly later forensic preservation."""
    episode=baseline_phase/'episodes'/entry['run_id']
    if (episode/'artifact_manifest.json').is_file():
        checks.require(sha(episode/'artifact_manifest.json')==entry['artifact_manifest_sha256'],'baseline closed manifest seal')
        row=read(episode/'artifact_manifest.json')['files']['controller_final.json'];path=episode/'controller_final.json'
        checks.require(path.stat().st_size==row['bytes'] and sha(path)==row['sha256'],'baseline controller configuration original seal')
        return read(path)['configuration'],dict(kind='original_complete_artifact_seal',
            manifest_sha256=sha(episode/'artifact_manifest.json'),controller_final_sha256=sha(path))
    checks.require(entry['status']=='attempt_failed','unsealed baseline explicitly remains failed')
    prepared=ROOT/'audit_results/article_stage_20260928/common_evaluation_v1'/entry['run_id']/'prepared'
    checks.require((prepared/'manifest.json').is_file(),'failed baseline requires separate terminal forensic preparation')
    manifest=read(prepared/'manifest.json')
    checks.require(manifest['schema']=='article.posthoc_forensic_manifest.v1','posthoc forensic seal schema')
    base.verify_files(prepared,manifest['files'],checks,'baseline terminal forensic')
    snapshot=read(prepared/'input_snapshot.json');original=read(prepared/'original_ledger_entry.json')
    checks.require(original==entry and snapshot['run_id']==entry['run_id'] and snapshot['forensic_capture_after_terminal_failure'] is True
        and snapshot['original_end_to_end_status']=='attempt_failed' and snapshot['original_artifact_manifest_available'] is False,
        'failure preserved; later forensic capture never presented as original completion seal')
    checks.require(snapshot['frozen_plan_sha256']==sha(prepared/'frozen_plan.json'),'forensic plan binding')
    plan=read(prepared/'frozen_plan.json')
    checks.require(plan['protocol_sha256']==sha(prepared/'original_metadata/protocol.json')
        and plan['slots'][entry['run_id']]==read(prepared/'original_metadata/protocol.json')['slots'][entry['run_id']],
        'forensic paired protocol/slot binding')
    for name in ('controller_final.json','prediction_seal.json','attempt_failure.json'):
        row=snapshot['input_files'][name];path=prepared/'original_metadata'/name
        checks.require(path.stat().st_size==row['bytes'] and sha(path)==row['sha256'],'forensic original bytes '+name)
    checks.require(sha(prepared/'original_metadata/attempt_failure.json')==entry['result_sha256'],'original failure still ledger-bound')
    return read(prepared/'original_metadata/controller_final.json')['configuration'],dict(kind='post_terminal_forensic_capture',
        manifest_sha256=sha(prepared/'manifest.json'),input_snapshot_sha256=sha(prepared/'input_snapshot.json'),
        original_status='attempt_failed',original_qualified=False,motion_completion=snapshot['motion_completion'],
        original_prediction_seal=snapshot['original_prediction_seal'],controller_final_sha256=sha(prepared/'original_metadata/controller_final.json'))


def review(episode,output,*,protocol_path=None):
    episode=Path(episode).resolve();output=Path(output).resolve();phase=episode.parent.parent
    require(output!=episode and not output.is_relative_to(episode),'review output outside episode')
    ledger_path=phase/'start_ledger.json'
    if not ledger_path.exists():return dict(status='unstarted',run_id=episode.name,reason='no ledger reservation',new_worlds=0)
    ledger=read(ledger_path);matches=[r for r in ledger['entries'] if r['run_id']==episode.name]
    if not matches:return dict(status='unstarted',run_id=episode.name,reason='no ledger reservation',new_worlds=0)
    require(len(matches)==1,'exactly one ledger row')
    entry=matches[0]
    if entry['status']=='reserved':return dict(status='pending',run_id=episode.name,reason='active reservation; no incomplete artifacts inspected',new_worlds=0)
    require(not output.exists(),'exclusive review output')
    checks=base.Checks();before=base.runtime_counts_v41();base_report=None;manifest_sha=None;protocol_sha=None
    summary=dict(schema='article.saved_episode_review.ground_v2',run_id=episode.name,online_status=entry['status'],
        new_worlds=0,new_sensor_queries=0,new_policy_runs=0,new_tsdf_integrations=0,new_surface_evaluations=0,
        scope='Saved receipt/physics compatibility plus independent ground/paired-protocol/derived-mesh checks; no new quality measurement')
    ground_rows=[]
    try:
        archived=episode/'protocol.json'
        selected=archived if archived.is_file() else Path(protocol_path) if protocol_path else None
        checks.require(selected is not None and selected.is_file(),'known protocol even for terminal failure')
        protocol=read(selected);protocol_sha=sha(selected)
        checks.require(protocol_sha==ledger['protocol_sha256'],'protocol bound to episode ledger')
        if protocol_path is not None:checks.require(sha(protocol_path)==protocol_sha,'external and archived protocol SHA')
        pin=protocol['paired_baseline_protocol']
        checks.require(set(pin)=={'path','sha256'},'strict nested paired baseline protocol pin')
        baseline_path=base.safe_file(ROOT,pin['path'])
        checks.require(sha(baseline_path)==pin['sha256'],'original baseline protocol SHA')
        baseline=read(baseline_path);verify_pairing(protocol,baseline);checks.check(True,'independent single-component paired design')
        checks.require(ledger['schema']=='article.start_ledger.v1' and ledger['phase']=='ablation' and ledger['slots']==protocol['slots'],
                       'ledger phase and complete slot binding')
        checks.require(len(ledger['entries'])<=12 and len({r['run_id'] for r in ledger['entries']})==len(ledger['entries'])
            and all(r['run_id'] in protocol['slots'] for r in ledger['entries']),'twelve named starts, no undeclared retry')
        slot=protocol['slots'][episode.name]
        checks.require((ROOT/protocol['output_relative_path']).resolve()==phase,'episode belongs to declared phase output')
        verify_source_relationship(protocol['source_sha256'],baseline['source_sha256'])
        checks.check(True,'shared execution sources unchanged; old CLI separately preserved in baseline seal')
        checks.require({'nso/article_experiment_ground_v2.py','nso/controller_article_ground_v2.py',
            'nso/observed_instances_article_ground_v2.py','nso/article_prediction_mesh_adapter_v1.py',
            'scripts/run_article_ground_experiment_20260928.py'}<=set(protocol['source_sha256']),
            'all new execution modules included in source closure')
        verify_archive(protocol,checks,'ground');verify_archive(baseline,checks,'baseline')
        summary['source_archive_sha256']=protocol['source_archive_sha256']
        summary['baseline_source_archive_sha256']=baseline['source_archive_sha256']
        summary['cumulative_registry']=verify_registry(phase,protocol_sha,entry,checks)
        baseline_phase=base.safe_file(ROOT,baseline['output_relative_path'])
        baseline_ledger=read(baseline_phase/'start_ledger.json')
        checks.require(baseline_ledger['protocol_sha256']==pin['sha256'],'baseline ledger protocol binding')
        pairs=[r for r in baseline_ledger['entries'] if r['run_id']==slot['paired_baseline_run_id']]
        checks.require(len(pairs)==1 and pairs[0]['status']!='reserved','known terminal baseline attempt; no outcome eligibility filter')
        summary['paired_baseline']=dict(protocol=pin,run_id=slot['paired_baseline_run_id'],online_status=pairs[0]['status'],
            quality_comparison_requires_common_derived_evaluation=True,baseline_failure_retained=pairs[0]['status']=='attempt_failed')
        if not (episode/'artifact_manifest.json').exists():
            failure=episode/'attempt_failure.json'
            if not failure.is_file():failure=phase/'failures'/(episode.name+'.json')
            checks.require(failure.is_file() and sha(failure)==entry['result_sha256'],'terminal failure retained and bound')
            summary.update(status='failed_attempt_preserved',qualified=False,failure=read(failure),
                limitation='No complete artifact manifest; no incomplete per-frame analysis or quality claim.')
        else:
            manifest_sha=sha(episode/'artifact_manifest.json')
            checks.require(manifest_sha==entry['artifact_manifest_sha256'],'terminal complete artifact seal')
            # The original reader is actually executed. Its detailed compatible
            # checks remain visible in a separately hashed nested report.
            base_report=base.review(episode,output/'compatible_v1_receipts')
            checks.require(base_report['status']=='reviewed' and base_report['all_checks_passed'],
                'original physics/packet/map/route reader passes this new episode',base_report.get('failures'))
            manifest=read(episode/'artifact_manifest.json');started=read(episode/'started.json')
            checks.require(manifest['schema']=='article.episode_artifacts.v1','compatible artifact inventory schema')
            checks.require(started['numerical_runtime']==protocol['numerical_runtime'],'actual numerical backend bound before execution')
            expected_pair=dict(protocol_relative_path=pin['path'],protocol_sha256=pin['sha256'],original_sources_preserved=True,
                old_scores_not_overwritten=True,comparison_requires_canonical_metric_on_both_arms=True)
            checks.require(started['paired_baseline']==expected_pair,'started paired comparison declaration')
            final=read(episode/'controller_final.json');configuration=final['configuration']
            checks.require(configuration['article_version']=='article_ground_v2' and configuration['common_ground_association'] is True,
                'actual controller ground version and enabled common frontend')
            checks.require(configuration['ground_association_policy']==dict(GROUND_POLICY),'same fixed ground constants for every method')
            digest=hashlib.sha256(json.dumps(configuration,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
            checks.require(digest==final['configuration_sha256'],'actual controller configuration digest')
            original_configuration,provenance=paired_configuration(baseline_phase,pairs[0],checks)
            common_configuration={k:v for k,v in configuration.items() if k not in GROUND_FIELDS}
            common_configuration['article_version']='article_v1'
            checks.require(common_configuration==original_configuration,'entire actual paired configuration differs only in declared ground fields')
            summary['paired_baseline']['configuration_provenance']=provenance
            for key,value in baseline['controller'].items():
                key={'cache_feedback_repair':'common_cache_feedback_repair','multi_view_planes':'common_multiview_plane_estimation'}.get(key,key)
                checks.require(configuration[key]==value,'actual common controller option '+key)
            count=base_report['rgbd_frames'];ground_counts=Counter()
            for i in range(count):
                with np.load(episode/f'packets/{i:03d}_rgbd.npz',allow_pickle=False) as raw:
                    checks.require(set(raw.files)==set(base.PaidRGBDObservationV40.__dataclass_fields__),
                        'strict saved public RGBD field whitelist:'+str(i))
                sensor,_=base.packet_at(episode,i)
                evidence=json.loads(gzip.decompress((episode/f'steps/{i:03d}.json.gz').read_bytes()))['controller_evidence']
                ground=evidence['association']['article_ground_association']
                _,computed=ground_mask_from_observation(sensor.rgbd)
                exact_or_numeric(ground,computed,'current_paid_ground/'+str(i))
                checks.check(True,'current raw paid packet recomputes exact ground mask/eligibility:'+str(i))
                ground_counts[ground['reason']]+=1
                ground_rows.append(dict(paid_step=i,accepted=ground['accepted'],reason=ground['reason'],
                    excluded_pixels=ground['excluded_pixels'],observation_sha256=ground['observation_sha256'],mask_sha256=ground['mask_sha256']))
            evaluation=read(episode/'evaluation.json');receipt_path=episode/'prediction_preprocessing.json'
            verify_evaluation_metadata(evaluation,protocol,sha(receipt_path))
            checks.check(True,'derived metric version/settings/source/preprocessing links')
            seal=read(episode/'prediction_seal.json');receipt=read(receipt_path)
            checks.require(receipt['source_prediction_sha256']==seal==evaluation['source_prediction_sha256'],'raw sealed array file pins')
            checks.require(receipt['original_prediction_seal_sha256']==sha(episode/'prediction_seal.json'),'original prediction seal provenance')
            checks.require(receipt['comparison_rule']=='same numeric-face adapter and frozen metric on both paired arms','common metric declaration')
            with np.load(episode/'prediction/mesh.npz',allow_pickle=False) as arrays:
                preprocessing=verify_preprocessing_arrays(arrays['vertices'],arrays['triangles'],receipt)
            checks.check(True,'independent raw-array geometry threshold/removed-index/hash audit')
            checks.require(sha(episode/'artifact_manifest.json')==manifest_sha,'episode seal unchanged during combined review')
            summary.update(status='reviewed',qualified=base_report['qualified'],method=slot['method'],scene_id=slot['scene_id'],
                phase='ablation',metrics=evaluation['metrics'],metric_version=METRIC_VERSION,
                metric_settings=protocol['evaluation'],ground_reason_counts=dict(ground_counts),mesh_preprocessing=preprocessing,
                compatible_reader_checks=base_report['checks'],physics_metrics={k:base_report[k] for k in
                    ('paid_actions','budget','translation_m','action_counts','rgbd_frames','collisions','returned_xy_and_yaw')})
    except Exception as exc:
        checks.check(False,'ground review execution completed',dict(type=type(exc).__name__,message=str(exc)))
        summary.update(status='review_error',qualified=False,error=dict(type=type(exc).__name__,message=str(exc)))
    checks.check(base.runtime_counts_v41()==before,'no World or paid action created by review')
    summary.update(checks=checks.count,all_checks_passed=not checks.failures,failures=checks.failures,
        input_manifest_sha256=manifest_sha,protocol_sha256=protocol_sha,
        reviewer_source_sha256={str(Path(__file__).relative_to(ROOT)):sha(__file__),
            str(Path(base.__file__).relative_to(ROOT)):sha(base.__file__)})
    output.mkdir(parents=True,exist_ok=True)
    for name,value in [('review.json',summary),('ground_receipts.json',ground_rows)]:
        with (output/name).open('xb') as stream:stream.write(base.canonical(value))
    files={str(p.relative_to(output)):dict(bytes=p.stat().st_size,sha256=sha(p))
           for p in output.rglob('*') if p.is_file()}
    with (output/'manifest.json').open('xb') as stream:
        stream.write(base.canonical(dict(schema='article.ground_episode_review_manifest.v2',files=files,
            input_episode_manifest_sha256=manifest_sha,protocol_sha256=protocol_sha,
            reviewer_source_sha256=summary['reviewer_source_sha256'],new_worlds=0,new_surface_evaluations=0,new_tsdf_integrations=0)))
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episode',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--protocol',type=Path)
    args=parser.parse_args()
    output=args.output or ROOT/'audit_results/article_stage_20260928/episode_reviews_ground_v2'/args.episode.name
    result=review(args.episode,output,protocol_path=args.protocol)
    print(json.dumps(result,ensure_ascii=False,allow_nan=False))
    if result['status'] not in ('pending','unstarted','reviewed','failed_attempt_preserved') or result.get('all_checks_passed') is False:
        raise SystemExit(1)
