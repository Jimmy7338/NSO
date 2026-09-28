"""Explicit SEM scene adapter around the existing paid driver and components.

The old integration runner and its archived closure are unchanged. This module
owns only the new protocol/input binding, orchestration and review boundary;
physics, controller construction, compression, finite reservations and actual
saved-packet/TSDF replay all reuse the existing implementations.
"""
import ast
from pathlib import Path
import re
import time

import numpy as np

from env.development_sensor_v41 import runtime_counts_v41, storage_report_v41
from env.semantic_scene_sensor import create_semantic_scene_sensor, validate_sensor_contract
from nso.episode_driver_v43 import canonical_bytes, execute_episode_v43, file_sha256
from nso.evidence_writer_v44 import CompressedStepWriterV44
from nso.semantic_experiment import (
    COMPLETE, EVALUATION, ROOT, PhaseLedger, _repo_path, make_components,
    read_json, reference_private_inputs,
)
from nso.semantic_experiment_replay import replay_saved_episode
from nso.semantic_scene_assets import checked_manifest, parse_asset_id
from nso.semantic_scene_evaluation import (
    EVALUATION_PARAMETERS, PREDICTION_FILES,
    load_semantic_scene_reference,
)
from nso.semantic_scene_navigation import load_semantic_navigation_bundle


SCHEMA='semantic.scene_experiment.protocol.v1'
ARTIFACT_SCHEMA='semantic.scene_experiment.artifacts.v1'
ENTRY_SOURCES=('nso/semantic_scene_experiment.py','scripts/run_semantic_scene_experiment.py')


def _pin(value):
    if not isinstance(value,str) or re.fullmatch('[0-9a-f]{64}',value) is None:
        raise ValueError('external SHA256 pin required')
    return value


def validate_protocol(protocol):
    from nso.controller_semantic_mechanism import METHODS
    if protocol.get('schema')!=SCHEMA or protocol.get('status') not in ('draft','frozen'):
        raise ValueError('explicit new-scene draft/frozen protocol required')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,79}',protocol.get('phase_id','')):
        raise ValueError('bounded phase identifier required')
    ledger=Path(protocol['ledger_relative_path'])
    if (len(ledger.parts)!=3 or ledger.parts[0]!='audit_results'
            or not ledger.parts[1].startswith('semantic_') or ledger.parts[2]!='start_ledger.json'):
        raise ValueError('separate semantic phase ledger required; old ledgers forbidden')
    for key in ('ledger_relative_path','output_relative_path','asset_root','navigation_root','reference_index_path'):
        _repo_path(protocol[key])
    if Path(protocol['output_relative_path'])!=ledger.parent/'episodes':
        raise ValueError('episode output must be the declared phase episodes directory')
    for key in ('asset_manifest_sha256','navigation_manifest_sha256','reference_index_sha256'):
        _pin(protocol.get(key))
    slots=protocol.get('slots')
    if not isinstance(slots,dict) or not 1<=len(slots)<=512 or protocol.get('maximum_new_world_slots')!=len(slots):
        raise ValueError('finite exact declared slot inventory required')
    for name,slot in slots.items():
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,79}',name):raise ValueError('bounded run ID required')
        parse_asset_id(slot.get('asset_id'))
        if slot.get('method') not in (*METHODS,'known_structure_reference'):raise ValueError('declared method required')
        if type(slot.get('budget')) is not int or not 1<=slot['budget']<=160:raise ValueError('paid budget must be 1..160')
        for key in ('noise_seed','tie_seed'):
            if type(slot.get(key)) is not int or slot[key]<0:raise ValueError('fixed nonnegative seed required')
        if slot.get('tie_rule') not in ('lexicographic','reverse','seeded_random'):raise ValueError('declared tie rule required')
    for key in ('expected_batch_peak_bytes','maximum_episode_bytes','maximum_file_bytes','terminal_record_reserve_bytes'):
        if type(protocol.get(key)) is not int or protocol[key]<=0:raise ValueError('positive byte allowance required')
    if not (protocol['terminal_record_reserve_bytes']<protocol['maximum_episode_bytes']
            and protocol['maximum_file_bytes']<=protocol['maximum_episode_bytes']):raise ValueError('invalid output caps')
    if not 0<protocol.get('maximum_elapsed_s',0)<=600:raise ValueError('positive <=600s wall-time cap required')
    if protocol.get('evaluation')!=EVALUATION:raise ValueError('unchanged complete-mesh metric with 1m capacity required')
    if any(key in protocol['controller'] for key in ('mode','method','budget','tie_rule','tie_seed')):
        raise ValueError('common kwargs cannot shadow per-slot method, budget or ties')
    if protocol['controller'].get('observation_frontend','bounded_local')!='bounded_local':
        raise ValueError('common bounded local frontend required')
    if not isinstance(protocol.get('scope'),str) or not protocol['scope']:
        raise ValueError('explicit experiment scope required')
    return protocol


def source_names():
    """New entry roots, recursively including every statically imported helper."""
    pending=list(ENTRY_SOURCES);seen=set()
    while pending:
        name=pending.pop()
        if name in seen:continue
        path=ROOT/name
        if not path.is_file():raise ValueError('missing required source: '+name)
        seen.add(name)
        if len(seen)>512:raise ValueError('source inventory exceeds bound')
        package=Path(name).with_suffix('').parts[:-1];modules=[]
        for node in ast.walk(ast.parse(path.read_text(),filename=name)):
            if isinstance(node,ast.Import):modules.extend(alias.name for alias in node.names)
            elif isinstance(node,ast.ImportFrom):
                base='.'.join(package[:len(package)-node.level+1]) if node.level else ''
                module='.'.join(x for x in (base,node.module or '') if x)
                modules.extend([module]+[module+'.'+a.name for a in node.names if a.name!='*'])
        for module in modules:
            chunks=module.split('.')
            if chunks[0] not in ('nso','env','utils','scripts'):continue
            base=Path(*chunks)
            for choice in (base.with_suffix('.py'),base/'__init__.py'):
                if (ROOT/choice).is_file():pending.append(str(choice))
            for length in range(1,len(chunks)):
                path=Path(*chunks[:length])/'__init__.py'
                if (ROOT/path).is_file():pending.append(str(path))
    return sorted(seen)


def load_inputs(protocol,slot):
    """Public runtime bundle plus an offline, pinned endpoint reference check."""
    asset_root=_repo_path(protocol['asset_root'])
    asset_manifest=checked_manifest(asset_root,protocol['asset_manifest_sha256'])
    bundle=load_semantic_navigation_bundle(_repo_path(protocol['navigation_root'])/slot['asset_id'],
        expected_manifest_sha256=protocol['navigation_manifest_sha256'])
    if bundle['asset_manifest_sha256']!=protocol['asset_manifest_sha256']:
        raise ValueError('navigation and asset pins refer to different geometry')
    validate_sensor_contract(bundle['public_spec'])
    if slot['budget']>bundle['public_spec']['task']['max_actions']:raise ValueError('budget exceeds pinned task')
    index_path=_repo_path(protocol['reference_index_path'])
    if index_path.is_symlink() or file_sha256(index_path)!=protocol['reference_index_sha256']:
        raise ValueError('reference index differs from protocol pin')
    index=read_json(index_path)
    if (index.get('schema')!='semantic_scene.reference_index.v1'
            or index.get('asset_manifest_sha256')!=protocol['asset_manifest_sha256']
            or set(index.get('references',{}))!=set(asset_manifest['assets'])):
        raise ValueError('complete reference index and asset inventory must agree')
    ref=index['references'][slot['asset_id']];_pin(ref['manifest_sha256'])
    load_semantic_scene_reference(_repo_path(ref['root']),manifest_sha256=ref['manifest_sha256'],
        asset_id=slot['asset_id'],asset_root=asset_root,asset_manifest_sha256=protocol['asset_manifest_sha256'])
    if EVALUATION_PARAMETERS!=protocol['evaluation']:
        raise ValueError('reference evaluator has not frozen the declared prediction capacity')
    inputs={
        'asset_manifest.json':(asset_root/'manifest.json',protocol['asset_manifest_sha256']),
        'navigation_manifest.json':(_repo_path(protocol['navigation_root'])/'manifest.json',protocol['navigation_manifest_sha256']),
        'reference_index.json':(index_path,protocol['reference_index_sha256']),
        'reference_manifest.json':(_repo_path(ref['root'])/'manifest.json',ref['manifest_sha256']),
    }
    return dict(bundle=bundle,reference=ref,archived_inputs=inputs,
        declared_structure_reference_inputs=reference_private_inputs(protocol,slot))


def run_experiment(protocol_path,run_id,*,preflight_only=False):
    protocol_path=Path(protocol_path).resolve();protocol=validate_protocol(read_json(protocol_path))
    if run_id not in protocol['slots']:raise ValueError('undeclared slot')
    if not preflight_only and protocol['status']!='frozen':raise ValueError('freeze the finite protocol before any reservation')
    slot=protocol['slots'][run_id];output=_repo_path(protocol['output_relative_path'])/run_id
    resource=storage_report_v41(output,protocol['expected_batch_peak_bytes'])
    if not resource['passed']:
        return dict(status='blocked_before_world_creation',resource=resource,start_slot_reserved=False,runtime=runtime_counts_v41())
    protocol_sha=file_sha256(protocol_path);ledger=PhaseLedger(protocol,protocol_sha);ledger.check_existing()
    inputs=load_inputs(protocol,slot);bundle=inputs['bundle']
    import open3d as o3d
    sources={name:file_sha256(ROOT/name) for name in source_names()}
    if preflight_only:
        return dict(status='ready_without_world_creation',run_id=run_id,protocol_sha256=protocol_sha,
            resource=resource,backend='Open3D '+o3d.__version__,source_files=len(sources),
            start_slot_reserved=False,protocol_status=protocol['status'],runtime=runtime_counts_v41(),
            input_sha256={name:pin for name,(_,pin) in inputs['archived_inputs'].items()})
    if output.exists():raise FileExistsError('retained output cannot be overwritten')
    metadata=dict(run_id=run_id,slot=slot,output=str(output),protocol_path=str(protocol_path),
        protocol_sha256=protocol_sha,source_sha256=sources,public_graph_sha256=bundle['graph'].input_sha256,
        input_sha256={name:pin for name,(_,pin) in inputs['archived_inputs'].items()},
        declared_structure_reference_inputs=inputs['declared_structure_reference_inputs'])
    # No policy/sensor is constructed until the durable reservation.
    output.parent.mkdir(parents=True,exist_ok=True)
    ledger.reserve(run_id,metadata)
    sensor=None;world_created=False;writer=None;output_created=False;before=runtime_counts_v41()
    try:
        output.mkdir(exist_ok=False)
        output_created=True
        writer=CompressedStepWriterV44(output,maximum_bytes=protocol['maximum_episode_bytes'],
            maximum_file_bytes=protocol['maximum_file_bytes'],terminal_reserve_bytes=protocol['terminal_record_reserve_bytes'])
        writer.json('started.json',dict(metadata,phase_id=protocol['phase_id'],resource=resource,
            scope=protocol['scope'],task_kind='semantic_scene_mechanism_development',exact_pose_model=True,
            known_coarse_navigation_prior=True,primary_experiment_started=protocol.get('primary_experiment_started',False)))
        writer._write('protocol.json',protocol_path.read_bytes())
        if file_sha256(output/'protocol.json')!=protocol_sha:raise ValueError('protocol changed while archiving')
        for name,pin in sources.items():
            writer._write('source/'+name,(ROOT/name).read_bytes())
            if file_sha256(output/'source'/name)!=pin:raise ValueError('source changed while archiving')
        for name,(path,pin) in inputs['archived_inputs'].items():
            writer._write('inputs/'+name,path.read_bytes())
            if file_sha256(output/'inputs'/name)!=pin:raise ValueError('input changed while archiving')
        for name,key in (('public_graph.json','graph_spec'),('public_spec.json','public_spec'),('public_workspace.json','workspace')):
            writer.json(name,bundle[key])
        controller,mapper=make_components(protocol,slot,bundle)
        sensor=create_semantic_scene_sensor(_repo_path(protocol['asset_root'])/slot['asset_id'],bundle['public_spec'],
            expected_manifest_sha256=protocol['asset_manifest_sha256'],episode_id=run_id,noise_seed=slot['noise_seed'],
            persistent_output_root=output,expected_batch_peak_bytes=protocol['expected_batch_peak_bytes'])
        world_created=True
        result=execute_episode_v43(sensor,controller,mapper,writer,budget=slot['budget'],maximum_elapsed_s=protocol['maximum_elapsed_s'])
        if (file_sha256(protocol_path)!=protocol_sha or any(file_sha256(ROOT/name)!=pin for name,pin in sources.items())
                or any(file_sha256(path)!=pin for path,pin in inputs['archived_inputs'].values())):
            raise ValueError('frozen source, protocol or input changed during execution')
        writer.json('encoding.json',dict(schema='v44.step_encoding.v1',steps=writer.step_encoding),terminal=True)
        writer.json('runtime.json',dict(before=before,after=runtime_counts_v41(),world_created=True,
            source_unchanged=True,evaluation_executed=False),terminal=True)
        writer.json('artifact_manifest.json',dict(schema=ARTIFACT_SCHEMA,files=dict(writer.files),
            source_sha256=sources,protocol_sha256=protocol_sha,input_sha256=metadata['input_sha256']),terminal=True)
        ledger.finish(run_id,status=result['status'],world_created=True,result_sha256=file_sha256(output/'result.json'))
        return dict(status=result['status'],run_id=run_id,output=str(output),world_created=True,
            executed_paid_actions=result['executed_paid_actions'],elapsed_s=result['elapsed_s'],artifact_bytes=writer.bytes_written,
            artifact_manifest_sha256=file_sha256(output/'artifact_manifest.json'),
            primary_experiment_started=protocol.get('primary_experiment_started',False))
    except Exception as exc:
        close_error=None
        if sensor is not None:
            try:sensor.close()
            except Exception as closing:close_error=str(closing)
        failure=dict(status='experiment_attempt_failed',type=type(exc).__name__,message=str(exc),run_id=run_id,
            world_created=world_created,close_error=close_error,automatic_retry=False)
        failure_path=output/'attempt_failure.json'
        if writer is not None:writer.json('attempt_failure.json',failure,terminal=True)
        else:
            if not output_created:failure_path=ledger.path.parent/'failures'/(run_id+'.json')
            failure_path.parent.mkdir(parents=True,exist_ok=True)
            with failure_path.open('xb') as stream:stream.write(canonical_bytes(failure))
        ledger.finish(run_id,status=failure['status'],world_created=world_created,result_sha256=file_sha256(failure_path))
        return failure


def _inspect_archive(root,pin):
    _pin(pin)
    if (root/'artifact_manifest.json').is_symlink() or file_sha256(root/'artifact_manifest.json')!=pin:
        raise ValueError('external episode manifest pin mismatch')
    manifest=read_json(root/'artifact_manifest.json')
    if manifest.get('schema')!=ARTIFACT_SCHEMA:raise ValueError('explicit new-scene artifact schema required')
    files=manifest['files'];actual={str(path.relative_to(root)) for path in root.rglob('*') if path.is_file()}
    if actual!=set(files)|{'artifact_manifest.json'}:raise ValueError('complete episode inventory mismatch')
    for name,row in files.items():
        path=root/name
        if Path(name).is_absolute() or '..' in Path(name).parts or path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError('plain contained episode file required')
        if path.stat().st_size!=row['bytes'] or file_sha256(path)!=row['sha256']:raise ValueError('artifact changed: '+name)
    if file_sha256(root/'protocol.json')!=manifest['protocol_sha256']:raise ValueError('archived protocol changed')
    sources=manifest['source_sha256']
    if ({name[7:] for name in files if name.startswith('source/')}!=set(sources)
            or not set(ENTRY_SOURCES)<=set(sources)):
        raise ValueError('archived source inventory does not bind this entry')
    differences=[]
    for name,pin in sources.items():
        if file_sha256(root/'source'/name)!=pin:raise ValueError('archived source changed')
        if not (ROOT/name).is_file() or file_sha256(ROOT/name)!=pin:differences.append(name)
    return manifest,differences


def inspect_experiment(root,expected_manifest_sha256):
    """Archive/ledger validation survives later code changes; replay does not."""
    root=Path(root).resolve();manifest,differences=_inspect_archive(root,expected_manifest_sha256)
    protocol=validate_protocol(read_json(root/'protocol.json'))
    if protocol['status']!='frozen':raise ValueError('executed protocol was not frozen')
    started,result,runtime=(read_json(root/name) for name in ('started.json','result.json','runtime.json'))
    slot=protocol['slots'].get(started['run_id']);sources=manifest['source_sha256']
    if (slot!=started['slot'] or started['source_sha256']!=sources or started['protocol_sha256']!=manifest['protocol_sha256']
            or not runtime.get('world_created') or not runtime.get('source_unchanged')):
        raise ValueError('episode provenance binding mismatch')
    index=read_json(root/'inputs/reference_index.json');ref=index['references'][slot['asset_id']]
    pins=dict(zip(('asset_manifest.json','navigation_manifest.json','reference_index.json','reference_manifest.json'),
        (protocol['asset_manifest_sha256'],protocol['navigation_manifest_sha256'],protocol['reference_index_sha256'],ref['manifest_sha256'])))
    if (pins!=manifest.get('input_sha256') or pins!=started.get('input_sha256')
            or {name[7:] for name in manifest['files'] if name.startswith('inputs/')}!=set(pins)
            or any(file_sha256(root/'inputs'/name)!=pin for name,pin in pins.items())):
        raise ValueError('archived asset/navigation/reference pins disagree')
    ledger=PhaseLedger(protocol,manifest['protocol_sha256']);ledger.check_existing();saved=read_json(ledger.path)
    if (saved.get('schema')!='semantic.phase_start_ledger.v1' or saved.get('phase_id')!=protocol['phase_id']
            or any(saved.get(key)!=value for key,value in ledger.binding.items())):
        raise ValueError('frozen finite phase binding differs')
    entries=[row for row in saved['entries'] if row['run_id']==started['run_id']]
    if (len(entries)!=1 or entries[0]['status']!=result['status'] or entries[0]['world_created'] is not True
            or entries[0]['result_sha256']!=file_sha256(root/'result.json')
            or any(entries[0]['metadata'].get(key)!=started.get(key) for key in
                ('slot','protocol_sha256','source_sha256','input_sha256','declared_structure_reference_inputs'))
            or entries[0]['metadata']['output']!=str(root)):
        raise ValueError('durable reservation does not bind this episode')
    if result['status'] not in COMPLETE or result['error'] is not None or result['finalization_errors']:
        raise ValueError('failed/partial attempt is retained; complete endpoint required')
    count=result['acquired_and_saved_packets']
    if (type(count) is not int or not 1<=count<=slot['budget']+1
            or any(result[key]!=count-1 for key in ('executed_paid_actions','submitted_paid_actions'))
            or result['received_sensor_packets']!=count or result['mapper_frames']!=count):
        raise ValueError('saved paid acquisition counts disagree')
    expected=dict(worlds_created=1,rgbd_frames=count,scans=count,paid_actions=count-1,blocked_before_world_creation=0)
    if any(runtime['after'][key]-runtime['before'][key]!=value for key,value in expected.items()):
        raise ValueError('physical runtime counters disagree')
    bundle=None
    if not differences:
        inputs=load_inputs(protocol,slot);bundle=inputs['bundle']
        if started['declared_structure_reference_inputs']!=inputs['declared_structure_reference_inputs']:
            raise ValueError('declared structure-reference sidecars changed')
        for name,key in (('public_graph.json','graph_spec'),('public_spec.json','public_spec'),('public_workspace.json','workspace')):
            if read_json(root/name)!=bundle[key]:raise ValueError('saved public input differs from pinned runtime bundle')
    return dict(root=root,protocol=protocol,slot=slot,started=started,result=result,runtime=runtime,
        manifest=manifest,manifest_sha256=expected_manifest_sha256,bundle=bundle,reference=ref,
        current_sources_match=not differences,current_source_differences=differences)


def evaluate_endpoint(episode):
    """Keep all positive-area prediction faces; preserve frozen reference math.

    The input checks mirror the sealed new-scene endpoint. Its absolute tiny-
    face rejection cannot be replaced through a global patch, so only this
    adapter selects the independently tested complete-surface evaluator.
    """
    from nso.complete_surface_evaluation import evaluate_complete_surface
    from nso.offline_evaluation_v44 import _array_sha, measure_navigation_coverage_v44
    from nso.saved_replay_v44 import _arrays
    if episode.get('current_sources_match') is not True:
        raise ValueError('evaluation requires executed sources or isolated archived execution')
    protocol=episode['protocol'];ref=episode['reference'];root=episode['root'];slot=episode['slot']
    if protocol['evaluation']!=EVALUATION_PARAMETERS:raise ValueError('evaluation capacity/metric differs from protocol')
    if file_sha256(root/'artifact_manifest.json')!=episode['manifest_sha256']:raise ValueError('inspected manifest changed')
    reference,domain,record=load_semantic_scene_reference(_repo_path(ref['root']),manifest_sha256=ref['manifest_sha256'],
        asset_id=slot['asset_id'],asset_root=_repo_path(protocol['asset_root']),
        asset_manifest_sha256=protocol['asset_manifest_sha256'])
    pins={name:episode['manifest']['files']['prediction/'+name]['sha256'] for name in PREDICTION_FILES}
    for name,pin in pins.items():
        path=root/'prediction'/name
        if path.is_symlink() or file_sha256(path)!=pin:raise ValueError('saved prediction differs from inspected pin')
    workspace_path=root/'public_workspace.json';descriptor=record['coverage']
    if (file_sha256(workspace_path)!=episode['manifest']['files']['public_workspace.json']['sha256']
            or read_json(workspace_path)!=record['public_workspace']):raise ValueError('reference workspace differs')
    mapper=read_json(root/'prediction/mapper.json');count=episode['result']['acquired_and_saved_packets']
    if (type(count) is not int or count<1 or mapper.get('backend_poisoned') is not False or mapper.get('frames')!=count
            or any(mapper.get(key)!=descriptor[key] for key in ('shape','resolution_m','origin_xy_m','grid_convention'))):
        raise ValueError('saved mapper coordinate/history contract differs')
    occupancy=_arrays(root/'prediction/occupancy.npz',('belief','observed'));belief=occupancy['belief'];observed=occupancy['observed']
    if (_array_sha(belief)!=mapper.get('occupancy_sha256') or observed.dtype!=np.bool_
            or observed.shape!=belief.shape or np.any(observed&(belief<0))):raise ValueError('occupancy/observation mask mismatch')
    coverage=measure_navigation_coverage_v44(belief,domain,descriptor)
    mesh=_arrays(root/'prediction/mesh.npz',('vertices','triangles','vertex_colors'))
    if mesh['vertex_colors'].shape!=mesh['vertices'].shape or not np.isfinite(mesh['vertex_colors']).all():
        raise ValueError('complete aligned finite mesh colors required')
    metrics=evaluate_complete_surface(reference,mesh['vertices'],mesh['triangles'],C_map=coverage['C_nav'],**protocol['evaluation'])
    metrics.pop('C_map');metrics.pop('J');metrics.update(C_nav=coverage['C_nav'],J_nav=coverage['C_nav']*metrics['Q'])
    if any(file_sha256(root/'prediction'/name)!=pin for name,pin in pins.items()):raise ValueError('prediction changed during evaluation')
    result=episode['result']
    return dict(schema='semantic_scene.complete_saved_prediction_evaluation.v1',metrics=metrics,coverage=coverage,
        input_prediction_sha256=pins,asset_id=slot['asset_id'],asset_manifest_sha256=protocol['asset_manifest_sha256'],
        episode_manifest_sha256=episode['manifest_sha256'],reference_manifest_sha256=ref['manifest_sha256'],
        evaluator_source_sha256=file_sha256(ROOT/'nso/complete_surface_evaluation.py'),
        all_task_instances_in_macro_denominator=True,prediction_roi_cropped=False,prediction_reintegrated=False,
        new_worlds=0,new_sensor_packets=0,new_tsdf_integrations=0,independent_replay_performed_here=False,
        formal_performance_evidence=False,task_success=result['status']=='controller_stop'
            and result['sensor_status'].get('returned_xy_and_yaw') is True,original_episode_status=result['status'])


def review_experiment(root,*,expected_manifest_sha256,output,replay_only=False):
    root,output=Path(root).resolve(),Path(output).resolve()
    if output.is_relative_to(root) or output.exists():raise ValueError('new review outside immutable episode required')
    episode=inspect_experiment(root,expected_manifest_sha256)
    result=dict(schema='semantic.scene_experiment.review.v1',status='experiment_review_failed',
        run_id=episode['started']['run_id'],phase_id=episode['protocol']['phase_id'],
        episode_manifest_sha256=expected_manifest_sha256,source_sha256=episode['manifest']['source_sha256'],
        input_sha256=episode['manifest']['input_sha256'],automatic_retry=False,
        current_sources_match=episode['current_sources_match'],current_source_differences=episode['current_source_differences'],
        primary_experiment_started=episode['protocol'].get('primary_experiment_started',False))
    began=time.monotonic();before=runtime_counts_v41()
    try:
        result['replay']=replay_saved_episode(episode)
        if not replay_only:result['evaluation']=evaluate_endpoint(episode)
        if not inspect_experiment(root,expected_manifest_sha256)['current_sources_match']:
            raise ValueError('executed source changed during review')
        result['status']='experiment_replay_verified' if replay_only else 'experiment_reviewed'
    except Exception as exc:result['error']=dict(type=type(exc).__name__,message=str(exc))
    after=runtime_counts_v41()
    result['runtime']=dict(before=before,after=after,no_new_world_or_sensor_action=before==after)
    if before!=after:result['status']='experiment_review_failed'
    result['elapsed_s']=time.monotonic()-began
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('xb') as stream:stream.write(canonical_bytes(result))
    return result
