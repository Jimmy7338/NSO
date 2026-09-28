"""Finite article experiments with independent ledgers and common CPU physics.

No old queue, phase registry, source seal or resource gate is mutated. A run is
reserved before sensor construction, and every reserved attempt is retained.
"""
import ast
from copy import deepcopy
import fcntl
import io
import json
import math
import os
from pathlib import Path
import re
import time
import zipfile

import numpy as np

from env.article_scene_sensor_v1 import ArticleSceneSensorV1, article_storage_report
from env.development_sensor_v41 import runtime_counts_v41
from nso.article_scene_assets_v1 import load_public_article_scene, load_private_article_scene
from nso.controller_article_v1 import ArticleControllerV1, METHOD_MAP
from nso.episode_driver_v43 import canonical_bytes, execute_episode_v43, file_sha256
from nso.evidence_writer_v44 import CompressedStepWriterV44
from nso.observed_mapper_v42 import ObservedMapperV42
from nso.offline_evaluation_v44 import (REFERENCE_ARRAYS, conservative_floor_domain_v44,
    measure_navigation_coverage_v44)
from nso.surface_evaluation_v40 import (CandidateViewV40, ReferenceSurfaceV40,
    freeze_reference_v40, evaluate_surface_v40)
from scripts.verify_development_surface_pipeline_v40 import reachable_candidates


ROOT = Path(__file__).resolve().parents[1]
PHASE_CAPS = {'development':12, 'main':96, 'ablation':24}
EVALUATION = dict(threshold_m=.05, sample_spacing_m=.3, seed=4002, max_samples=1000000)


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as stream:
        stream.write(canonical_bytes(value));stream.flush();os.fsync(stream.fileno())


def source_hashes():
    pending = ['nso/article_experiment_v1.py','scripts/run_article_experiment_20260928.py']
    seen = set()
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        path = ROOT/name
        if not path.is_file():
            raise ValueError('Missing imported local source: '+name)
        seen.add(name)
        package = Path(name).with_suffix('').parts[:-1]
        for node in ast.walk(ast.parse(path.read_text())):
            modules = []
            if isinstance(node,ast.Import):
                modules = [a.name for a in node.names]
            elif isinstance(node,ast.ImportFrom):
                base = '.'.join(package[:len(package)-node.level+1]) if node.level else ''
                module = '.'.join(x for x in (base,node.module or '') if x)
                modules = [module]+[module+'.'+a.name for a in node.names if a.name!='*']
            for module in modules:
                parts = module.split('.')
                if parts[0] not in ('nso','env','utils','scripts'):
                    continue
                candidates = [Path(*parts).with_suffix('.py'),Path(*parts)/'__init__.py']
                candidates += [Path(*parts[:i])/'__init__.py' for i in range(1,len(parts))]
                pending.extend(str(p) for p in candidates if (ROOT/p).is_file())
    return {name:file_sha256(ROOT/name) for name in sorted(seen)}


def validate_protocol(protocol):
    if protocol.get('schema') != 'article.experiment_protocol.v1':
        raise ValueError('new independent article protocol required')
    phase = protocol.get('phase')
    if phase not in PHASE_CAPS or protocol.get('status') != 'frozen':
        raise ValueError('declared frozen phase required')
    slots = protocol.get('slots',{})
    if not 1 <= len(slots) <= PHASE_CAPS[phase]:
        raise ValueError('phase slot ceiling exceeded')
    output = Path(protocol['output_relative_path'])
    if (output.is_absolute() or '..' in output.parts or len(output.parts)!=3
            or output.parts[:2] != ('audit_results','article_stage_20260928')):
        raise ValueError('isolated article output root required')
    for name,slot in slots.items():
        if not re.fullmatch(r'[A-Za-z0-9_.-]{1,80}',name) or slot['method'] not in METHOD_MAP:
            raise ValueError('bounded run identifier and declared method required')
        if not re.fullmatch(r'ART1_(AISLE|CELL|LOOP)_(DEV|T0|T1)',slot['scene_id']):
            raise ValueError('new article scene identifier required')
        if (phase=='main') != (not slot['scene_id'].endswith('_DEV')):
            raise ValueError('test layouts only in main phase; development layouts for other phases')
        if type(slot['budget']) is not int or not 1 <= slot['budget'] <= 160:
            raise ValueError('finite physical paid budget required')
        if type(slot['noise_seed']) is not int or slot['noise_seed'] < 0:
            raise ValueError('fixed noise seed required')
    if protocol['evaluation'] != EVALUATION or protocol['reserve_bytes'] < 1024**3:
        raise ValueError('declared common metric and persistent reserve required')
    if not 2*1024**2 <= protocol['maximum_episode_bytes'] <= 64*1024**2:
        raise ValueError('bounded episode allowance required')
    if not 0 < protocol['maximum_elapsed_s'] <= 900:
        raise ValueError('finite <=900 second episode limit required')
    if protocol['controller'].keys() & {'method','budget','mode','home','tie_rule','tie_seed'}:
        raise ValueError('common controller options shadow per-run contract')
    return protocol


class ArticleLedger:
    def __init__(self, output, protocol, protocol_sha, *, registry_path=None):
        self.path = Path(output)/'start_ledger.json'
        self.protocol = protocol
        self.binding = dict(schema='article.start_ledger.v1',protocol_sha256=protocol_sha,
            phase=protocol['phase'],slots=protocol['slots'])
        self.registry_path = (Path(registry_path) if registry_path is not None
                              else Path(output).parent/'execution_registry.json')

    def mutate(self,callback):
        self.path.parent.mkdir(parents=True,exist_ok=True)
        with self.path.with_suffix('.lock').open('a+b') as lock:
            fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
            ledger = read(self.path) if self.path.exists() else dict(**self.binding,entries=[])
            if any(ledger.get(k)!=v for k,v in self.binding.items()):
                raise ValueError('existing ledger bound to different configuration')
            returned = callback(ledger)
            temporary = self.path.with_suffix('.new')
            write_new(temporary,ledger)
            os.replace(temporary,self.path)
            descriptor = os.open(self.path.parent,os.O_RDONLY)
            try:os.fsync(descriptor)
            finally:os.close(descriptor)
            return returned

    def reserve(self,run_id):
        def change(data):
            if run_id not in self.protocol['slots'] or any(r['run_id']==run_id for r in data['entries']):
                raise ValueError('undeclared or already reserved run; no automatic retry')
            # A revision or new output directory cannot reset the night's cap.
            # Reserve the global slot first; a crash between the two durable
            # writes conservatively consumes that slot rather than refunding it.
            registry_path = self.registry_path
            registry_path.parent.mkdir(parents=True,exist_ok=True)
            with registry_path.with_suffix('.lock').open('a+b') as lock:
                fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
                registry = read(registry_path) if registry_path.exists() else dict(
                    schema='article.execution_registry.v1',phase_caps=PHASE_CAPS,entries=[])
                if registry.get('schema')!='article.execution_registry.v1' or registry.get('phase_caps')!=PHASE_CAPS:
                    raise ValueError('cumulative study registry contract differs')
                phase = self.protocol['phase']
                if sum(r['phase']==phase for r in registry['entries'])>=PHASE_CAPS[phase]:
                    raise ValueError('cumulative phase start ceiling exhausted')
                identity = dict(ledger=str(self.path.resolve()),run_id=run_id)
                if any(all(r[k]==v for k,v in identity.items()) for r in registry['entries']):
                    raise ValueError('globally reserved attempt cannot be retried')
                registry['entries'].append(dict(**identity,phase=phase,
                    protocol_sha256=self.binding['protocol_sha256'],reserved_unix_s=time.time()))
                temporary = registry_path.with_suffix('.new')
                write_new(temporary,registry);os.replace(temporary,registry_path)
                fd=os.open(registry_path.parent,os.O_RDONLY)
                try:os.fsync(fd)
                finally:os.close(fd)
            data['entries'].append(dict(run_id=run_id,status='reserved',world_created=False,
                                       started_unix_s=time.time()))
        self.mutate(change)

    def finish(self,run_id,**result):
        def change(data):
            row = next(r for r in data['entries'] if r['run_id']==run_id)
            if row['status'] != 'reserved':
                raise ValueError('only an active reservation can finish')
            row.update(**result,finished_unix_s=time.time())
        self.mutate(change)


def prepare_reference(asset_dir, output, *, expected_manifest_sha256):
    """Static, trajectory-independent reference before any policy execution."""
    asset = load_private_article_scene(asset_dir,expected_manifest_sha256=expected_manifest_sha256)
    arrays,metadata = asset['arrays'],asset['metadata']
    views,candidates = reachable_candidates(metadata,{'public_defaults':asset['public_spec']})
    reference = freeze_reference_v40(arrays['vertices'],arrays['triangles'],arrays['triangle_instance_id'],
        views,sample_spacing_m=.3,seed=4001,max_samples=50000)
    if reference.manifest()['target_instances'] != list(range(4)):
        raise ValueError('all four task instances must remain in reference denominator')
    domain,coverage = conservative_floor_domain_v44(metadata)
    output = Path(output);output.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(output/'surface.npz',**{key:getattr(reference,key) for key in REFERENCE_ARRAYS})
    np.savez_compressed(output/'coverage_domain.npz',domain=domain)
    write_new(output/'candidate_views.json',candidates)
    record = dict(schema='article.reference.v1',scene_id=Path(asset_dir).name,
        asset_manifest_sha256=expected_manifest_sha256,surface=reference.manifest(),coverage=coverage,
        evaluation=EVALUATION,public_workspace=asset['public_workspace'],route_independent=True,
        new_worlds=0,reference_uses_semantic_weights=False)
    write_new(output/'reference.json',record)
    manifest = {p.name:dict(bytes=p.stat().st_size,sha256=file_sha256(p)) for p in sorted(output.iterdir())}
    write_new(output/'manifest.json',manifest)
    return dict(root=str(output.relative_to(ROOT)),manifest_sha256=file_sha256(output/'manifest.json'),
                observable_reference_samples=len(reference.points),reference_area_m2=reference.area_weights.sum())


def load_reference(entry):
    root = ROOT/entry['root']
    if file_sha256(root/'manifest.json') != entry['manifest_sha256']:
        raise ValueError('reference manifest differs from protocol pin')
    for name,row in read(root/'manifest.json').items():
        if (root/name).stat().st_size != row['bytes'] or file_sha256(root/name)!=row['sha256']:
            raise ValueError('reference artifact differs from manifest')
    record = read(root/'reference.json'); surface = record['surface']
    views = tuple(CandidateViewV40(**r) for r in read(root/'candidate_views.json')['candidate_views'])
    with np.load(root/'surface.npz',allow_pickle=False) as data:
        arrays = {key:data[key].copy() for key in REFERENCE_ARRAYS}
    reference = ReferenceSurfaceV40(**arrays,candidate_views=views,
        sample_spacing_m=surface['sample_spacing_m'],seed=surface['seed'],
        full_target_sample_count=surface['full_target_sample_count'],fingerprint=surface['fingerprint'])
    if reference.manifest()!=surface:
        raise ValueError('reference contents disagree with pinned fingerprint')
    with np.load(root/'coverage_domain.npz',allow_pickle=False) as data:
        domain = data['domain'].copy()
    return reference,domain,record


def make_components(protocol,slot,bundle):
    public,workspace = bundle['public_spec'],bundle['workspace']
    prior = public['structure_prior']
    options = dict(protocol['controller']);options.update(slot.get('controller_overrides',{}))
    if slot.get('controller_overrides') and protocol['phase']!='ablation':
        raise ValueError('controller overrides reserved for declared ablations')
    controller = ArticleControllerV1(bundle['graph'],method=slot['method'],home=bundle['home_state'],
        budget=slot['budget'],palette=workspace['marker_palette'],
        structure_names=prior['abstract_structures'],class_structure_prior=prior['probability_by_category'],
        tie_rule='lexicographic',tie_seed=0,**options)
    bounds = workspace['bounds_xy_m']; origin = bounds[0]
    resolution = protocol['mapper']['resolution_m']
    shape = [math.ceil((bounds[1][1]-origin[1])/resolution),math.ceil((bounds[1][0]-origin[0])/resolution)]
    mapper = ObservedMapperV42(shape=shape,origin_xy_m=origin,**protocol['mapper'])
    return controller,mapper


def run_episode(protocol_path,run_id):
    protocol_path = Path(protocol_path);protocol = validate_protocol(read(protocol_path))
    if run_id not in protocol['slots']:
        raise ValueError('undeclared run')
    output = ROOT/protocol['output_relative_path']; episode = output/'episodes'/run_id
    slot = protocol['slots'][run_id]
    source_pins = source_hashes()
    if source_pins!=protocol['source_sha256']:
        raise ValueError('execution sources differ from frozen protocol')
    if episode.exists():
        raise FileExistsError('retained attempt cannot be overwritten')
    report = article_storage_report(output,expected_episode_bytes=protocol['maximum_episode_bytes'],
                                    reserve_bytes=protocol['reserve_bytes'])
    current_bytes = sum(p.stat().st_size for p in output.rglob('*') if p.is_file()) if output.exists() else 0
    if not report['passed'] or current_bytes+protocol['maximum_episode_bytes']>protocol['maximum_phase_bytes']:
        return dict(status='resources_blocked_before_reservation',run_id=run_id,resource=report)
    bundle = load_public_article_scene(ROOT/protocol['asset_root']/slot['scene_id'],
        expected_manifest_sha256=protocol['asset_manifest_sha256'])
    bundle['public_spec'] = deepcopy(bundle['public_spec'])
    if slot['budget'] > bundle['public_spec']['task']['max_actions']:
        raise ValueError('slot budget exceeds asset maximum')
    bundle['public_spec']['task']['budget_tier'] = ('standard' if slot['budget']==
        bundle['public_spec']['task']['max_actions'] else 'short')
    bundle['public_spec']['task']['max_actions'] = slot['budget']
    reference,domain,reference_record = load_reference(protocol['references'][slot['scene_id']])
    if reference_record['asset_manifest_sha256']!=protocol['asset_manifest_sha256']:
        raise ValueError('reference bound to a different scene asset')
    protocol_sha = file_sha256(protocol_path)
    ledger = ArticleLedger(output,protocol,protocol_sha)
    ledger.reserve(run_id)
    sensor = None; writer = None; episode_created = False; before = runtime_counts_v41()
    try:
        episode.mkdir(parents=True,exist_ok=False)
        episode_created = True
        writer = CompressedStepWriterV44(episode,maximum_bytes=protocol['maximum_episode_bytes'],
            maximum_file_bytes=min(32*1024**2,protocol['maximum_episode_bytes']),
            terminal_reserve_bytes=512*1024)
        writer.json('started.json',dict(slot=slot,protocol_sha256=protocol_sha,resource=report,
                                      source_sha256=source_pins,scope='CPU article virtual experiment'))
        writer._write('protocol.json',protocol_path.read_bytes())
        for name,key in [('public_spec.json','public_spec'),('public_workspace.json','workspace'),('public_graph.json','graph_spec')]:
            writer.json(name,bundle[key])
        controller,mapper = make_components(protocol,slot,bundle)
        sensor = ArticleSceneSensorV1(ROOT/protocol['asset_root']/slot['scene_id'],bundle['public_spec'],
            expected_manifest_sha256=protocol['asset_manifest_sha256'],episode_id=run_id,
            noise_seed=slot['noise_seed'],persistent_output_root=episode,
            expected_episode_bytes=protocol['maximum_episode_bytes'],reserve_bytes=protocol['reserve_bytes'])
        result = execute_episode_v43(sensor,controller,mapper,writer,budget=slot['budget'],
                                    maximum_elapsed_s=protocol['maximum_elapsed_s'])
        if source_pins!=source_hashes() or file_sha256(protocol_path)!=protocol_sha:
            raise ValueError('source or protocol changed during execution')
        writer.json('encoding.json',dict(steps=writer.step_encoding),terminal=True)
        writer.json('controller_final.json',controller.snapshot(),terminal=True)
        writer.json('runtime.json',dict(before=before,after=runtime_counts_v41(),new_worlds=1),terminal=True)
        # Seal predictions before the offline evaluator reads the saved mesh.
        prediction_pins = {name:file_sha256(episode/'prediction'/name) for name in
                           ('mesh.npz','occupancy.npz','mapper.json') if (episode/'prediction'/name).exists()}
        writer.json('prediction_seal.json',prediction_pins,terminal=True)
        evaluation = dict(status='prediction_unavailable',qualified=False)
        if len(prediction_pins)==3:
            with np.load(episode/'prediction/occupancy.npz',allow_pickle=False) as data:
                coverage = measure_navigation_coverage_v44(data['belief'],domain,reference_record['coverage'])
            with np.load(episode/'prediction/mesh.npz',allow_pickle=False) as data:
                metrics = evaluate_surface_v40(reference,data['vertices'],data['triangles'],
                    C_map=coverage['C_nav'],**EVALUATION)
            metrics.pop('C_map');metrics.pop('J')
            metrics.update(C_nav=coverage['C_nav'],J_nav=coverage['C_nav']*metrics['Q'])
            qualified = (result['status']=='controller_stop' and not result['collisions']
                and result['sensor_status'].get('returned_xy_and_yaw') is True
                and result['mapper_frames']==result['acquired_and_saved_packets']==result['mapper_tsdf_integrations'])
            evaluation = dict(status='measured',qualified=qualified,metrics=metrics,coverage=coverage,
                source_prediction_sha256=prediction_pins,reference_manifest_sha256=
                protocol['references'][slot['scene_id']]['manifest_sha256'],fixed_target_instances=4,
                no_roi_crop=True,semantic_weights_used=False,exact_simulated_pose=True)
            if any(file_sha256(episode/'prediction'/name)!=pin for name,pin in prediction_pins.items()):
                raise ValueError('sealed prediction changed during evaluation')
        writer.json('evaluation.json',evaluation,terminal=True)
        writer.json('artifact_manifest.json',dict(schema='article.episode_artifacts.v1',files=dict(writer.files),
            protocol_sha256=protocol_sha,source_sha256=source_pins),terminal=True)
        summary = dict(status=result['status'],world_created=True,qualified=evaluation['qualified'],
            elapsed_s=result['elapsed_s'],artifact_bytes=writer.bytes_written,
            executed_paid_actions=result['executed_paid_actions'],result_sha256=file_sha256(episode/'result.json'),
            artifact_manifest_sha256=file_sha256(episode/'artifact_manifest.json'))
        ledger.finish(run_id,**summary)
        return dict(run_id=run_id,**summary)
    except Exception as exc:
        close_error = None
        if sensor is not None:
            try:sensor.close()
            except Exception as closing:close_error=str(closing)
        failure = dict(status='attempt_failed',world_created=sensor is not None,
                       type=type(exc).__name__,message=str(exc),close_error=close_error,automatic_retry=False)
        failure_path = episode/'attempt_failure.json' if episode_created else output/'failures'/(run_id+'.json')
        write_new(failure_path,failure)
        ledger.finish(run_id,**failure,result_sha256=file_sha256(failure_path))
        return dict(run_id=run_id,**failure)
