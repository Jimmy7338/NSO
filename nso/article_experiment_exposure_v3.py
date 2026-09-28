"""Exposure V3 episodes with the unchanged cumulative ledger and common metric.

Mechanically adapted from the frozen Ground V2 lifecycle: only schema,
component construction, source entry points, pairing validation and scope labels
change. The paid driver, original RGB-D mapper, failure ledger and canonical
evaluation lifecycle below are unchanged; no frozen module globals are patched.
"""
import ast
from contextlib import redirect_stdout
from copy import deepcopy
import io
import math
import os
from pathlib import Path
import sys

import numpy as np
import scipy

from nso.article_experiment_v1 import (
    ArticleLedger, EVALUATION, PHASE_CAPS, read, write_new, load_reference,
    validate_protocol as validate_v1_protocol,
)
from nso.article_experiment_ground_v2 import validate_protocol as validate_ground_protocol
from env.article_scene_sensor_v1 import ArticleSceneSensorV1, article_storage_report
from env.development_sensor_v41 import runtime_counts_v41
from nso.article_scene_assets_v1 import load_public_article_scene
from nso.controller_article_exposure_v3 import ArticleControllerExposureV3
from nso.episode_driver_v43 import execute_episode_v43, file_sha256
from nso.evidence_writer_v44 import CompressedStepWriterV44
from nso.observed_mapper_v42 import ObservedMapperV42
from nso.offline_evaluation_v44 import measure_navigation_coverage_v44
from nso.article_prediction_mesh_adapter_v1 import (
    SCHEMA as MESH_ADAPTER_SCHEMA, prepare_prediction_mesh_v1,
)
from nso.surface_evaluation_v40 import evaluate_surface_v40

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'article.experiment_protocol.exposure_v3'
METHODS = frozenset(('G', 'B', 'S', 'NBV'))
METRIC_PREPROCESSING = dict(schema=MESH_ADAPTER_SCHEMA,
    applied_equally_to_paired_baseline=True)


def runtime_metadata():
    """Build/path fingerprint, not merely library version labels."""
    numpy_build = io.StringIO()
    scipy_build = io.StringIO()
    with redirect_stdout(numpy_build):
        np.show_config()
    with redirect_stdout(scipy_build):
        scipy.show_config()
    return dict(python_executable=sys.executable, python_version=sys.version,
        numpy_version=np.__version__, numpy_path=np.__file__, numpy_build=numpy_build.getvalue(),
        scipy_version=scipy.__version__, scipy_path=scipy.__file__, scipy_build=scipy_build.getvalue(),
        thread_environment={name:os.environ.get(name) for name in
            ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')})


def source_hashes():
    pending = ['nso/article_experiment_exposure_v3.py',
               'scripts/run_article_exposure_experiment_20260928.py']
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
            if isinstance(node, ast.Import):
                modules = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = '.'.join(package[:len(package)-node.level+1]) if node.level else ''
                module = '.'.join(x for x in (base, node.module or '') if x)
                modules = [module]+[module+'.'+a.name for a in node.names if a.name != '*']
            for module in modules:
                parts = module.split('.')
                if parts[0] not in ('nso', 'env', 'utils', 'scripts'):
                    continue
                candidates = [Path(*parts).with_suffix('.py'), Path(*parts)/'__init__.py']
                candidates += [Path(*parts[:i])/'__init__.py' for i in range(1, len(parts))]
                pending.extend(str(p) for p in candidates if (ROOT/p).is_file())
    return {name:file_sha256(ROOT/name) for name in sorted(seen)}


def validate_protocol(protocol):
    if protocol.get('schema') != SCHEMA:
        raise ValueError('explicit Exposure V3 protocol required')
    compatible = deepcopy(protocol)
    compatible['schema'] = 'article.experiment_protocol.v1'
    validate_v1_protocol(compatible)
    if protocol['phase'] not in ('ablation', 'main'):
        raise ValueError('Exposure V3 uses existing ablation/main caps, never a fresh development phase')
    if protocol['phase'] == 'ablation' and len(protocol['slots']) > 12:
        raise ValueError('at most twelve declared paired exposure ablations')
    if protocol['controller'].get('ground_association') is not True:
        raise ValueError('this frozen frontend version requires common ground_association=True')
    if protocol['controller'].get('same_center_exposure_dedup') is not True:
        raise ValueError('common same_center_exposure_dedup=True required')
    if protocol['controller'].get('cache_feedback_repair', True) is not True:
        raise ValueError('unchanged ArticleV1 cache repair required')
    if protocol.get('metric_preprocessing') != METRIC_PREPROCESSING:
        raise ValueError('declared common canonical numeric-face metric required')
    if any(slot['method'] not in METHODS or slot.get('controller_overrides')
           for slot in protocol['slots'].values()):
        raise ValueError('four common-configuration methods only; no slot overrides')
    if protocol['phase'] == 'ablation':
        baseline = protocol.get('paired_baseline_protocol', {})
        path = Path(baseline.get('path', ''))
        if (not path.parts or path.is_absolute() or '..' in path.parts
                or not isinstance(baseline.get('sha256'), str) or len(baseline['sha256']) != 64):
            raise ValueError('pinned existing paired baseline protocol required')
        ids = [slot.get('paired_baseline_run_id') for slot in protocol['slots'].values()]
        if any(not isinstance(value, str) or not value for value in ids) or len(ids) != len(set(ids)):
            raise ValueError('unique paired baseline run identifiers required')
    return protocol


def validate_pairing(protocol):
    """Verify the declared single-change comparison before any reservation."""
    if protocol['phase'] != 'ablation':
        return None
    pin = protocol['paired_baseline_protocol']
    path = ROOT/pin['path']
    if not path.resolve().is_relative_to(ROOT.resolve()) or file_sha256(path) != pin['sha256']:
        raise ValueError('paired baseline protocol pin differs')
    baseline = validate_ground_protocol(read(path))
    if baseline['phase'] != 'ablation':
        raise ValueError('exposure ablations pair the existing Ground V2 cohort')
    for key in ('asset_root', 'asset_manifest_sha256', 'mapper', 'evaluation', 'references',
                'metric_preprocessing', 'numerical_runtime', 'maximum_elapsed_s',
                'maximum_episode_bytes', 'reserve_bytes'):
        if protocol[key] != baseline[key]:
            raise ValueError('paired experiment changes common '+key)
    common = dict(protocol['controller'])
    common.pop('same_center_exposure_dedup')
    if common != baseline['controller']:
        raise ValueError('paired experiment must change only the exposure deduplication switch')
    if ({slot['paired_baseline_run_id'] for slot in protocol['slots'].values()}
            != set(baseline['slots'])):
        raise ValueError('complete Ground baseline cohort must remain paired')
    for slot in protocol['slots'].values():
        paired = baseline['slots'].get(slot['paired_baseline_run_id'])
        if paired is None or any(slot[key] != paired[key]
            for key in ('method', 'scene_id', 'budget', 'noise_seed')) or paired.get('controller_overrides'):
            raise ValueError('paired slot method/scene/budget/noise differs')
    return dict(protocol_relative_path=pin['path'], protocol_sha256=pin['sha256'],
        original_sources_preserved=True, old_scores_not_overwritten=True,
        comparison_requires_canonical_metric_on_both_arms=True,
        baseline_version='article_ground_v2',
        sole_common_component_change='same_center_exposure_dedup')


def make_components(protocol, slot, bundle):
    public, workspace = bundle['public_spec'], bundle['workspace']
    prior = public['structure_prior']
    controller = ArticleControllerExposureV3(bundle['graph'], method=slot['method'],
        home=bundle['home_state'], budget=slot['budget'], palette=workspace['marker_palette'],
        structure_names=prior['abstract_structures'],
        class_structure_prior=prior['probability_by_category'],
        tie_rule='lexicographic', tie_seed=0, **protocol['controller'])
    bounds = workspace['bounds_xy_m']; origin = bounds[0]
    resolution = protocol['mapper']['resolution_m']
    shape = [math.ceil((bounds[1][1]-origin[1])/resolution),
             math.ceil((bounds[1][0]-origin[0])/resolution)]
    mapper = ObservedMapperV42(shape=shape, origin_xy_m=origin, **protocol['mapper'])
    return controller, mapper


def run_episode(protocol_path, run_id):
    protocol_path = Path(protocol_path)
    protocol = validate_protocol(read(protocol_path))
    if run_id not in protocol['slots']:
        raise ValueError('undeclared run')
    pairing = validate_pairing(protocol)
    numerical_runtime = runtime_metadata()
    if 'numerical_runtime' in protocol and protocol['numerical_runtime'] != numerical_runtime:
        raise ValueError('numerical runtime differs from frozen build/path fingerprint')
    output = ROOT/protocol['output_relative_path']; episode = output/'episodes'/run_id
    slot = protocol['slots'][run_id]
    source_pins = source_hashes()
    if source_pins != protocol['source_sha256']:
        raise ValueError('execution sources differ from frozen protocol')
    if episode.exists():
        raise FileExistsError('retained attempt cannot be overwritten')
    report = article_storage_report(output, expected_episode_bytes=protocol['maximum_episode_bytes'],
                                    reserve_bytes=protocol['reserve_bytes'])
    current_bytes = sum(p.stat().st_size for p in output.rglob('*') if p.is_file()) if output.exists() else 0
    if not report['passed'] or current_bytes+protocol['maximum_episode_bytes'] > protocol['maximum_phase_bytes']:
        return dict(status='resources_blocked_before_reservation', run_id=run_id, resource=report)
    bundle = load_public_article_scene(ROOT/protocol['asset_root']/slot['scene_id'],
        expected_manifest_sha256=protocol['asset_manifest_sha256'])
    bundle['public_spec'] = deepcopy(bundle['public_spec'])
    if slot['budget'] > bundle['public_spec']['task']['max_actions']:
        raise ValueError('slot budget exceeds asset maximum')
    bundle['public_spec']['task']['budget_tier'] = ('standard' if slot['budget'] ==
        bundle['public_spec']['task']['max_actions'] else 'short')
    bundle['public_spec']['task']['max_actions'] = slot['budget']
    reference, domain, reference_record = load_reference(protocol['references'][slot['scene_id']])
    if reference_record['asset_manifest_sha256'] != protocol['asset_manifest_sha256']:
        raise ValueError('reference bound to a different scene asset')
    protocol_sha = file_sha256(protocol_path)
    ledger = ArticleLedger(output, protocol, protocol_sha)
    ledger.reserve(run_id)
    sensor = None; episode_created = False; before = runtime_counts_v41()
    try:
        episode.mkdir(parents=True, exist_ok=False)
        episode_created = True
        writer = CompressedStepWriterV44(episode, maximum_bytes=protocol['maximum_episode_bytes'],
            maximum_file_bytes=min(32*1024**2, protocol['maximum_episode_bytes']),
            terminal_reserve_bytes=512*1024)
        writer.json('started.json', dict(slot=slot, protocol_sha256=protocol_sha, resource=report,
            source_sha256=source_pins, numerical_runtime=numerical_runtime, paired_baseline=pairing,
            scope='Exposure V3 CPU article virtual experiment; common paid-exposure ablation'))
        writer._write('protocol.json', protocol_path.read_bytes())
        for name, key in [('public_spec.json','public_spec'), ('public_workspace.json','workspace'),
                          ('public_graph.json','graph_spec')]:
            writer.json(name, bundle[key])
        controller, mapper = make_components(protocol, slot, bundle)
        sensor = ArticleSceneSensorV1(ROOT/protocol['asset_root']/slot['scene_id'], bundle['public_spec'],
            expected_manifest_sha256=protocol['asset_manifest_sha256'], episode_id=run_id,
            noise_seed=slot['noise_seed'], persistent_output_root=episode,
            expected_episode_bytes=protocol['maximum_episode_bytes'], reserve_bytes=protocol['reserve_bytes'])
        result = execute_episode_v43(sensor, controller, mapper, writer, budget=slot['budget'],
                                    maximum_elapsed_s=protocol['maximum_elapsed_s'])
        if source_pins != source_hashes() or file_sha256(protocol_path) != protocol_sha:
            raise ValueError('source or protocol changed during execution')
        if pairing is not None:
            validate_pairing(protocol)
        writer.json('encoding.json', dict(steps=writer.step_encoding), terminal=True)
        writer.json('controller_final.json', controller.snapshot(), terminal=True)
        writer.json('runtime.json', dict(before=before, after=runtime_counts_v41(), new_worlds=1), terminal=True)
        prediction_pins = {name:file_sha256(episode/'prediction'/name) for name in
            ('mesh.npz','occupancy.npz','mapper.json') if (episode/'prediction'/name).exists()}
        writer.json('prediction_seal.json', prediction_pins, terminal=True)
        evaluation = dict(status='prediction_unavailable', qualified=False)
        if len(prediction_pins) == 3:
            with np.load(episode/'prediction/occupancy.npz', allow_pickle=False) as data:
                coverage = measure_navigation_coverage_v44(data['belief'], domain, reference_record['coverage'])
            with np.load(episode/'prediction/mesh.npz', allow_pickle=False) as data:
                vertices, triangles, preprocessing = prepare_prediction_mesh_v1(data['vertices'], data['triangles'])
                preprocessing.update(source_prediction_sha256=prediction_pins,
                    original_prediction_seal_sha256=file_sha256(episode/'prediction_seal.json'),
                    comparison_rule='same numeric-face adapter and frozen metric on both paired arms')
                writer.json('prediction_preprocessing.json', preprocessing, terminal=True)
                metrics = evaluate_surface_v40(reference, vertices, triangles,
                    C_map=coverage['C_nav'], **EVALUATION)
            metrics.pop('C_map'); metrics.pop('J')
            metrics.update(C_nav=coverage['C_nav'], J_nav=coverage['C_nav']*metrics['Q'])
            qualified = (result['status'] == 'controller_stop' and not result['collisions']
                and result['sensor_status'].get('returned_xy_and_yaw') is True
                and result['mapper_frames'] == result['acquired_and_saved_packets'] == result['mapper_tsdf_integrations'])
            evaluation = dict(schema='article.exposure_v3.canonical_evaluation.v1',
                metric_version='article.common_numeric_face_evaluation.v1',
                status='measured', qualified=qualified, metrics=metrics, coverage=coverage,
                source_prediction_sha256=prediction_pins, reference_manifest_sha256=
                protocol['references'][slot['scene_id']]['manifest_sha256'], fixed_target_instances=4,
                no_roi_crop=True, semantic_weights_used=False, exact_simulated_pose=True,
                metric_preprocessing=dict(schema=MESH_ADAPTER_SCHEMA,
                    receipt='prediction_preprocessing.json',
                    receipt_sha256=file_sha256(episode/'prediction_preprocessing.json')),
                evaluation_settings=dict(EVALUATION), evaluation_source_sha256={
                    name:source_pins[name] for name in ('nso/surface_evaluation_v40.py',
                        'nso/article_prediction_mesh_adapter_v1.py')},
                original_prediction_files_unchanged=True)
            if any(file_sha256(episode/'prediction'/name) != pin for name, pin in prediction_pins.items()):
                raise ValueError('sealed prediction changed during evaluation')
        writer.json('evaluation.json', evaluation, terminal=True)
        writer.json('artifact_manifest.json', dict(schema='article.episode_artifacts.v1', files=dict(writer.files),
            protocol_sha256=protocol_sha, source_sha256=source_pins), terminal=True)
        summary = dict(status=result['status'], world_created=True, qualified=evaluation['qualified'],
            elapsed_s=result['elapsed_s'], artifact_bytes=writer.bytes_written,
            executed_paid_actions=result['executed_paid_actions'], result_sha256=file_sha256(episode/'result.json'),
            artifact_manifest_sha256=file_sha256(episode/'artifact_manifest.json'))
        ledger.finish(run_id, **summary)
        return dict(run_id=run_id, **summary)
    except Exception as exc:
        close_error = None
        if sensor is not None:
            try:
                sensor.close()
            except Exception as closing:
                close_error = str(closing)
        failure = dict(status='attempt_failed', world_created=sensor is not None,
            type=type(exc).__name__, message=str(exc), close_error=close_error, automatic_retry=False)
        failure_path = episode/'attempt_failure.json' if episode_created else output/'failures'/(run_id+'.json')
        write_new(failure_path, failure)
        ledger.finish(run_id, **failure, result_sha256=file_sha256(failure_path))
        return dict(run_id=run_id, **failure)
