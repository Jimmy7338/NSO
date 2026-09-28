"""Finite new-phase experiment orchestration using the unchanged physical driver.

The protocol, exact slot inventory and source closure bind a permanent phase
reservation. No prior pilot schema, output or ledger is reused or overwritten.
"""
import ast
from copy import deepcopy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time

import numpy as np

from env.development_sensor_v41 import create_development_sensor, runtime_counts_v41, storage_report_v41
from nso.episode_driver_v43 import canonical_bytes, execute_episode_v43, file_sha256
from nso.evidence_writer_v44 import CompressedStepWriterV44

ROOT=Path(__file__).resolve().parents[1]
REGISTRY='audit_results/semantic_experiment_registry.json'
COMPLETE={'controller_stop','controller_blocked','budget_exhausted','stopped_without_confirmed_return'}
EVALUATION=dict(max_samples=1000000,sample_spacing_m=.3,seed=4002,threshold_m=.05)


def read_json(path):
    def unique(pairs):
        out={}
        for key,value in pairs:
            if key in out:raise ValueError('duplicate JSON key: '+key)
            out[key]=value
        return out
    return json.loads(Path(path).read_text(),object_pairs_hook=unique,
        parse_constant=lambda value:(_ for _ in ()).throw(ValueError('nonfinite JSON: '+value)))


def _repo_path(value):
    path=Path(value)
    if path.is_absolute() or '..' in path.parts or not path.parts:
        raise ValueError('bounded repository-relative path required')
    result=(ROOT/path).resolve()
    if not result.is_relative_to(ROOT.resolve()):raise ValueError('repository path escapes root')
    return result


def validate_protocol(protocol):
    from nso.controller_semantic_mechanism import METHODS
    if protocol.get('schema')!='semantic.experiment.protocol.v1':raise ValueError('new experiment protocol required')
    phase=protocol.get('phase_id','')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,79}',phase):raise ValueError('bounded phase identifier required')
    slots=protocol.get('slots')
    if not isinstance(slots,dict) or not 1<=len(slots)<=512:raise ValueError('finite explicit slot inventory required')
    if protocol.get('maximum_new_world_slots')!=len(slots):raise ValueError('slot count must equal frozen inventory')
    ledger=Path(protocol['ledger_relative_path'])
    if (len(ledger.parts)!=3 or ledger.parts[0]!='audit_results' or not ledger.parts[1].startswith('semantic_')
            or ledger.parts[2]!='start_ledger.json'):
        raise ValueError('new semantic phase ledger path required; old ledgers forbidden')
    _repo_path(ledger)
    for name,slot in slots.items():
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,79}',name):raise ValueError('bounded run ID required')
        if not re.fullmatch(r'DEV_[A-F]_00',slot.get('asset_id','')):raise ValueError('registered development asset required')
        if slot.get('method') not in (*METHODS,'known_structure_reference'):raise ValueError('unsupported declared controller method')
        if type(slot.get('budget')) is not int or not 1<=slot['budget']<=160:raise ValueError('paid budget must be 1..160')
        for key in ('noise_seed','tie_seed'):
            if type(slot.get(key)) is not int or slot[key]<0:raise ValueError('nonnegative fixed seed required')
        if slot.get('tie_rule') not in ('lexicographic','reverse','seeded_random'):raise ValueError('declared tie rule required')
    for key in ('expected_batch_peak_bytes','maximum_episode_bytes','maximum_file_bytes','terminal_record_reserve_bytes'):
        if type(protocol.get(key)) is not int or protocol[key]<=0:raise ValueError('positive fixed byte allowance required')
    if not (protocol['terminal_record_reserve_bytes']<protocol['maximum_episode_bytes']
            and protocol['maximum_file_bytes']<=protocol['maximum_episode_bytes']):raise ValueError('invalid output caps')
    if not 0<protocol.get('maximum_elapsed_s',0)<=600:raise ValueError('positive wall-time allowance <=600s required')
    if protocol.get('evaluation')!=EVALUATION:raise ValueError('fixed complete-mesh evaluation settings required')
    if any(key in protocol['controller'] for key in ('mode','method','budget','tie_rule','tie_seed')):
        raise ValueError('per-slot method/budget/ties cannot be shadowed by shared controller kwargs')
    if protocol['controller'].get('observation_frontend','bounded_local')!='bounded_local':
        raise ValueError('the bounded local frontend must be common to all methods')
    return protocol


def source_names():
    """Static recursive local imports include helper additions before freezing."""
    pending=['nso/semantic_experiment.py','nso/semantic_experiment_replay.py','scripts/run_semantic_experiment.py']
    seen=set()
    while pending:
        name=pending.pop()
        if name in seen:continue
        path=ROOT/name
        if not path.is_file():raise ValueError('missing required local source: '+name)
        seen.add(name)
        if len(seen)>512:raise ValueError('local source closure exceeds bounded inventory')
        parts=Path(name).with_suffix('').parts
        package=parts[:-1]
        tree=ast.parse(path.read_text(),filename=name)
        modules=[]
        for node in ast.walk(tree):
            if isinstance(node,ast.Import):modules.extend(alias.name for alias in node.names)
            elif isinstance(node,ast.ImportFrom):
                base='.'.join(package[:len(package)-node.level+1]) if node.level else ''
                module='.'.join(x for x in (base,node.module or '') if x)
                modules.append(module)
                modules.extend(module+'.'+alias.name for alias in node.names if alias.name!='*')
        for module in modules:
            chunks=module.split('.')
            if not chunks or chunks[0] not in ('nso','env','utils','scripts'):continue
            base=Path(*chunks)
            choices=[base.with_suffix('.py'),base/'__init__.py']
            for choice in choices:
                if (ROOT/choice).is_file():pending.append(str(choice))
            for length in range(1,len(chunks)):
                init=Path(*chunks[:length])/'__init__.py'
                if (ROOT/init).is_file():pending.append(str(init))
    return sorted(seen)


def _atomic_json(path,value):
    temporary=path.with_suffix(path.suffix+'.new')
    with temporary.open('xb') as stream:
        stream.write(canonical_bytes(value));stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)
    fd=os.open(path.parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)


class PhaseLedger:
    def __init__(self,protocol,protocol_sha256):
        self.protocol=deepcopy(protocol);self.sha=protocol_sha256
        self.path=_repo_path(protocol['ledger_relative_path'])
        self.binding=dict(protocol_sha256=self.sha,ledger_relative_path=protocol['ledger_relative_path'],
            slots=deepcopy(protocol['slots']),maximum_slots=len(protocol['slots']))

    def check_existing(self):
        registry=ROOT/REGISTRY
        if registry.exists():
            phases=read_json(registry)['phases']
            existing=phases.get(self.protocol['phase_id'])
            if existing is not None and existing!=self.binding:
                raise ValueError('phase already bound to another protocol, slot inventory or ledger path')
            if existing is None and self.path.exists():
                raise ValueError('unregistered existing ledger cannot be adopted')
            for name,row in phases.items():
                if name!=self.protocol['phase_id'] and row['ledger_relative_path']==self.protocol['ledger_relative_path']:
                    raise ValueError('ledger path already belongs to another phase')
        elif self.path.exists():
            raise ValueError('unregistered existing ledger cannot be adopted')

    def _mutate(self,callback):
        registry=ROOT/REGISTRY;registry.parent.mkdir(parents=True,exist_ok=True)
        with registry.with_suffix('.lock').open('a+b') as lock:
            fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
            self.check_existing()
            data=read_json(registry) if registry.exists() else dict(schema='semantic.phase_registry.v1',phases={})
            if self.protocol['phase_id'] not in data['phases']:
                if self.path.exists():raise ValueError('existing ledger cannot be rebound')
                data['phases'][self.protocol['phase_id']]=self.binding
                _atomic_json(registry,data)
            expected=dict(schema='semantic.phase_start_ledger.v1',phase_id=self.protocol['phase_id'],**self.binding)
            ledger=read_json(self.path) if self.path.exists() else dict(**expected,entries=[])
            if any(ledger.get(key)!=value for key,value in expected.items()):raise ValueError('frozen ledger binding changed')
            result=callback(ledger)
            self.path.parent.mkdir(parents=True,exist_ok=True)
            _atomic_json(self.path,ledger)
            return result

    def reserve(self,run_id,metadata):
        def change(data):
            if run_id not in self.protocol['slots']:raise ValueError('undeclared run ID')
            if any(row['run_id']==run_id for row in data['entries']):raise ValueError('run already reserved; no automatic retry')
            if len(data['entries'])>=len(self.protocol['slots']):raise ValueError('finite slot cap exhausted')
            if data['entries'] and data['entries'][0]['metadata']['source_sha256']!=metadata['source_sha256']:
                raise ValueError('source closure differs across this frozen phase')
            row=dict(run_id=run_id,status='reserved_before_factory',world_created=False,
                metadata=deepcopy(metadata),reservation_time_unix_s=time.time())
            data['entries'].append(row);return deepcopy(row)
        return self._mutate(change)

    def finish(self,run_id,*,status,world_created,result_sha256):
        def change(data):
            rows=[r for r in data['entries'] if r['run_id']==run_id]
            if len(rows)!=1 or rows[0]['status']!='reserved_before_factory':raise ValueError('only open reservation can finish')
            rows[0].update(status=status,world_created=bool(world_created),result_sha256=result_sha256,
                           finished_time_unix_s=time.time())
            return deepcopy(rows[0])
        return self._mutate(change)


def reference_spec(protocol,asset_id):
    return (protocol['references'][asset_id] if 'references' in protocol else
            dict(root=protocol['reference_root'],manifest_sha256=protocol['reference_manifest_sha256']))


def load_bundle(protocol,slot):
    from nso.public_navigation_v43 import load_public_navigation_bundle_v43
    return load_public_navigation_bundle_v43(_repo_path(protocol['navigation_root'])/slot['asset_id'],
        _repo_path(protocol['asset_root'])/slot['asset_id'],protocol_path=_repo_path(protocol['scene_protocol_path']))


def make_components(protocol,slot,bundle):
    from nso.controller_semantic_mechanism import SemanticMechanismController
    from nso.observed_mapper_v42 import ObservedMapperV42
    public,workspace=bundle['public_spec'],bundle['workspace'];prior=public['structure_prior']
    if slot['budget']>public['task']['max_actions']:raise ValueError('episode budget exceeds shared public task maximum')
    kwargs=dict(protocol['controller']);kwargs['observation_frontend']='bounded_local'
    common=dict(home=bundle['home_state'],budget=slot['budget'],
        palette=workspace['marker_palette'],structure_names=prior['abstract_structures'],
        class_structure_prior=prior['probability_by_category'],
        tie_rule=slot['tie_rule'],tie_seed=slot['tie_seed'],**kwargs)
    if slot['method']=='known_structure_reference':
        from nso.known_structure_reference import KnownStructureReferenceController,PaidInstanceStructureMatcher
        matcher=PaidInstanceStructureMatcher.from_asset(_repo_path(protocol['asset_root'])/slot['asset_id'],
                                                       structure_names=prior['abstract_structures'])
        controller=KnownStructureReferenceController(bundle['graph'],structure_matcher=matcher,**common)
    else:
        controller=SemanticMechanismController(bundle['graph'],method=slot['method'],**common)
    bounds=workspace['bounds_xy_m'];origin=bounds[0];resolution=protocol['mapper']['resolution_m']
    shape=[math.ceil((bounds[1][1]-origin[1])/resolution),math.ceil((bounds[1][0]-origin[0])/resolution)]
    return controller,ObservedMapperV42(shape=shape,origin_xy_m=origin,**protocol['mapper'])


def reference_private_inputs(protocol,slot):
    """Only the declared structure reference may read/pin these two sidecars."""
    if slot['method']!='known_structure_reference':return {}
    base=_repo_path(protocol['asset_root']); manifest=read_json(base/'manifest.json')
    records={}
    for suffix in ('renderer_private/markers.json','evaluation_private/instances.json'):
        relative=slot['asset_id']+'/'+suffix;path=base/relative
        digest=file_sha256(path)
        if digest!=manifest['artifact_sha256'].get(relative):raise ValueError('structure-reference private asset differs from frozen manifest')
        records[str(path.relative_to(ROOT))]=digest
    return records


def run_experiment(protocol_path,run_id,*,output_root=None,preflight_only=False):
    protocol_path=Path(protocol_path).resolve();protocol=validate_protocol(read_json(protocol_path))
    if run_id not in protocol['slots']:raise ValueError('run must be in frozen inventory')
    output=(Path(output_root).resolve() if output_root is not None else _repo_path(protocol['output_relative_path']))/run_id
    resource=storage_report_v41(output,protocol['expected_batch_peak_bytes'])
    if not resource['passed']:
        return dict(status='blocked_before_world_creation',resource=resource,start_slot_reserved=False,runtime=runtime_counts_v41())
    sha=file_sha256(protocol_path);ledger=PhaseLedger(protocol,sha);ledger.check_existing()
    slot=protocol['slots'][run_id];bundle=load_bundle(protocol,slot)
    from nso.offline_evaluation_v44 import load_reference_v44
    ref=reference_spec(protocol,slot['asset_id'])
    load_reference_v44(_repo_path(ref['root']),manifest_sha256=ref['manifest_sha256'],asset_id=slot['asset_id'])
    import open3d as o3d
    if preflight_only:
        return dict(status='ready_without_world_creation',run_id=run_id,protocol_sha256=sha,
            resource=resource,backend='Open3D '+o3d.__version__,start_slot_reserved=False,
            source_files=len(source_names()),runtime=runtime_counts_v41())
    if output.exists():raise FileExistsError('retained episode output cannot be overwritten')
    sources={name:file_sha256(ROOT/name) for name in source_names()}
    metadata=dict(run_id=run_id,slot=slot,output=str(output),protocol_path=str(protocol_path),
        protocol_sha256=sha,source_sha256=sources,public_graph_sha256=bundle['graph'].input_sha256,
        declared_structure_reference_inputs=reference_private_inputs(protocol,slot))
    ledger.reserve(run_id,metadata)
    output.mkdir(parents=True,exist_ok=False)
    writer=CompressedStepWriterV44(output,maximum_bytes=protocol['maximum_episode_bytes'],
        maximum_file_bytes=protocol['maximum_file_bytes'],terminal_reserve_bytes=protocol['terminal_record_reserve_bytes'])
    sensor=None;world_created=False;before=runtime_counts_v41()
    try:
        writer.json('started.json',dict(metadata,resource=resource,phase_id=protocol['phase_id'],
            scope=protocol['scope'],task_kind='semantic_mechanism_development',exact_pose_model=True,
            known_coarse_navigation_prior=True,primary_experiment_started=protocol.get('primary_experiment_started',False)))
        writer._write('protocol.json',protocol_path.read_bytes())
        if file_sha256(output/'protocol.json')!=sha:raise ValueError('protocol changed while archiving')
        for name,digest in sources.items():
            writer._write('source/'+name,(ROOT/name).read_bytes())
            if file_sha256(output/'source'/name)!=digest:raise ValueError('source changed while archiving')
        for name,key in (('public_graph.json','graph_spec'),('public_spec.json','public_spec'),('public_workspace.json','workspace')):
            writer.json(name,bundle[key])
        controller,mapper=make_components(protocol,slot,bundle)
        sensor=create_development_sensor(_repo_path(protocol['asset_root'])/slot['asset_id'],bundle['public_spec'],
            episode_id=run_id,noise_seed=slot['noise_seed'],persistent_output_root=output,
            expected_batch_peak_bytes=protocol['expected_batch_peak_bytes'])
        world_created=True
        result=execute_episode_v43(sensor,controller,mapper,writer,budget=slot['budget'],maximum_elapsed_s=protocol['maximum_elapsed_s'])
        if file_sha256(protocol_path)!=sha or any(file_sha256(ROOT/name)!=digest for name,digest in sources.items()):
            raise ValueError('executed source/protocol changed during run')
        writer.json('encoding.json',dict(schema='v44.step_encoding.v1',steps=writer.step_encoding),terminal=True)
        writer.json('runtime.json',dict(before=before,after=runtime_counts_v41(),world_created=True,
            source_unchanged=True,evaluation_executed=False),terminal=True)
        writer.json('artifact_manifest.json',dict(schema='semantic.experiment.artifacts.v1',
            files=dict(writer.files),source_sha256=sources,protocol_sha256=sha),terminal=True)
        ledger.finish(run_id,status=result['status'],world_created=True,result_sha256=file_sha256(output/'result.json'))
        return dict(status=result['status'],run_id=run_id,output=str(output),
            executed_paid_actions=result['executed_paid_actions'],elapsed_s=result['elapsed_s'],
            artifact_bytes=writer.bytes_written,artifact_manifest_sha256=file_sha256(output/'artifact_manifest.json'),
            world_created=True,primary_experiment_started=protocol.get('primary_experiment_started',False))
    except Exception as exc:
        close_error=None
        if sensor is not None:
            try:sensor.close()
            except Exception as close_exc:close_error=str(close_exc)
        failure=dict(status='experiment_attempt_failed',type=type(exc).__name__,message=str(exc),
            run_id=run_id,world_created=world_created,close_error=close_error,automatic_retry=False)
        writer.json('attempt_failure.json',failure,terminal=True)
        ledger.finish(run_id,status=failure['status'],world_created=world_created,result_sha256=file_sha256(output/'attempt_failure.json'))
        return failure


def inspect_experiment(root,expected_manifest_sha256):
    root=Path(root).resolve()
    if not re.fullmatch('[0-9a-f]{64}',expected_manifest_sha256 or ''):raise ValueError('external manifest SHA256 required')
    if file_sha256(root/'artifact_manifest.json')!=expected_manifest_sha256:raise ValueError('external manifest pin mismatch')
    manifest=read_json(root/'artifact_manifest.json')
    if manifest.get('schema')!='semantic.experiment.artifacts.v1':raise ValueError('new episode artifact schema required')
    files=manifest['files'];actual={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}
    if actual!=set(files)|{'artifact_manifest.json'}:raise ValueError('complete episode inventory mismatch')
    for name,row in files.items():
        path=root/name
        if Path(name).is_absolute() or '..' in Path(name).parts or path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError('plain contained artifact path required')
        if path.stat().st_size!=row['bytes'] or file_sha256(path)!=row['sha256']:raise ValueError('artifact changed: '+name)
    if file_sha256(root/'protocol.json')!=manifest['protocol_sha256']:raise ValueError('archived protocol mismatch')
    protocol=validate_protocol(read_json(root/'protocol.json'))
    sources=manifest['source_sha256']
    if {name[7:] for name in files if name.startswith('source/')}!=set(sources):
        raise ValueError('archived source closure inventory differs from manifest')
    current_source_differences=[]
    for name,digest in sources.items():
        if file_sha256(root/'source'/name)!=digest:raise ValueError('archived source changed: '+name)
        if not (ROOT/name).is_file() or file_sha256(ROOT/name)!=digest:current_source_differences.append(name)
    started,result,runtime=(read_json(root/name) for name in ('started.json','result.json','runtime.json'))
    slot=protocol['slots'].get(started['run_id'])
    if (slot!=started['slot'] or started['protocol_sha256']!=manifest['protocol_sha256']
            or started['source_sha256']!=sources or not runtime.get('world_created') or not runtime.get('source_unchanged')):
        raise ValueError('episode provenance binding mismatch')
    if started.get('declared_structure_reference_inputs')!=reference_private_inputs(protocol,slot):
        raise ValueError('declared structure-reference inputs differ')
    ledger=PhaseLedger(protocol,manifest['protocol_sha256']);ledger.check_existing()
    ledger_data=read_json(ledger.path)
    if (ledger_data.get('schema')!='semantic.phase_start_ledger.v1'
            or ledger_data.get('phase_id')!=protocol['phase_id']
            or any(ledger_data.get(key)!=value for key,value in ledger.binding.items())):
        raise ValueError('recorded phase ledger differs from frozen protocol binding')
    entries=[row for row in ledger_data['entries'] if row['run_id']==started['run_id']]
    if (len(entries)!=1 or entries[0]['status']!=result['status'] or entries[0]['world_created'] is not True
            or entries[0]['result_sha256']!=file_sha256(root/'result.json')
            or entries[0]['metadata']['source_sha256']!=sources or entries[0]['metadata']['output']!=str(root)
            or entries[0]['metadata']['slot']!=slot
            or entries[0]['metadata']['protocol_sha256']!=manifest['protocol_sha256']):
        raise ValueError('finite phase reservation does not bind this episode')
    if result['status'] not in COMPLETE or result['error'] is not None or result['finalization_errors']:
        raise ValueError('failed or partial episode retained; complete endpoint required')
    count=result['acquired_and_saved_packets']
    if (type(count) is not int or not 1<=count<=slot['budget']+1
            or any(result[key]!=count-1 for key in ('executed_paid_actions','submitted_paid_actions'))
            or result['received_sensor_packets']!=count or result['mapper_frames']!=count):
        raise ValueError('complete paid acquisition counts disagree')
    expected=dict(worlds_created=1,rgbd_frames=count,scans=count,paid_actions=count-1,blocked_before_world_creation=0)
    if any(runtime['after'][key]-runtime['before'][key]!=value for key,value in expected.items()):
        raise ValueError('physical runtime counters disagree with saved frames')
    bundle=None
    if not current_source_differences:
        bundle=load_bundle(protocol,slot)
        for name,key in (('public_graph.json','graph_spec'),('public_spec.json','public_spec'),('public_workspace.json','workspace')):
            if read_json(root/name)!=bundle[key]:raise ValueError('saved shared public input differs')
    return dict(root=root,protocol=protocol,slot=slot,started=started,result=result,runtime=runtime,
                manifest=manifest,manifest_sha256=expected_manifest_sha256,bundle=bundle,
                current_sources_match=not current_source_differences,
                current_source_differences=current_source_differences)


def evaluate_endpoint(episode):
    from nso.offline_evaluation_v44 import load_reference_v44,measure_navigation_coverage_v44
    from nso.saved_replay_v44 import _arrays
    from nso.surface_evaluation_v40 import evaluate_surface_v40
    if not episode.get('current_sources_match'):raise ValueError('numerical evaluation requires executed sources or isolated archived execution')
    protocol,slot,root=episode['protocol'],episode['slot'],episode['root']
    ref=reference_spec(protocol,slot['asset_id'])
    reference,domain,record=load_reference_v44(_repo_path(ref['root']),manifest_sha256=ref['manifest_sha256'],asset_id=slot['asset_id'])
    mapper=read_json(root/'prediction/mapper.json');descriptor=record['coverage']
    for key in ('shape','resolution_m','origin_xy_m','grid_convention'):
        if mapper[key]!=descriptor[key]:raise ValueError('evaluation and mapping coordinates differ')
    occupancy=_arrays(root/'prediction/occupancy.npz',('belief','observed'))
    coverage=measure_navigation_coverage_v44(occupancy['belief'],domain,descriptor)
    mesh=_arrays(root/'prediction/mesh.npz',('vertices','triangles','vertex_colors'))
    metrics=evaluate_surface_v40(reference,mesh['vertices'],mesh['triangles'],C_map=coverage['C_nav'],**protocol['evaluation'])
    metrics.pop('C_map');metrics.pop('J');metrics.update(C_nav=coverage['C_nav'],J_nav=coverage['C_nav']*metrics['Q'])
    result=episode['result']
    return dict(metrics=metrics,coverage=coverage,reference_manifest_sha256=ref['manifest_sha256'],
        task_success=result['status']=='controller_stop' and result['sensor_status'].get('returned_xy_and_yaw') is True,
        original_episode_status=result['status'],all_task_instances_in_macro_denominator=True,
        prediction_roi_cropped=False,new_worlds=0,new_sensor_packets=0,new_tsdf_integrations=0)
