#!/usr/bin/env python3
"""Replay sealed paired V7 branches using frozen dependencies and raw sensors.

The new world is reconstructed in one isolated subprocess. Prefix actions are
executed manually; collect_response_prefix and the execution runner are never
called. Branch physics/mesh/metrics/weighted unions/AUC reuse the existing
independent counterfactual verifier, whose exact source hash is also recorded.
No raw artifact, prediction, metric or mesh is changed or removed.
"""
import argparse
from dataclasses import asdict, replace
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import traceback

for _thread_limit in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_thread_limit] = '1'


ROOT = Path(__file__).resolve().parents[1]
BASE_VERIFIER = ROOT / 'scripts/replay_counterfactual_views.py'
CHANNELS = ('G', 'O', 'S', 'X', 'M', 'N', 'G_capacity')


def load_base(filename):
    spec = importlib.util.spec_from_file_location('independent_counterfactual_replay_base', filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def geometry_hash(mapper):
    """Match the recorded geometry-only preparation digest; labels/colors excluded."""
    import numpy as np
    mesh = mapper.mesh()
    quality = mapper.quality_evidence(max_points=10**9)
    arrays = [('belief', mapper.belief), ('camera_seen', mapper.camera_seen),
              ('vertices', np.asarray(mesh.vertices)), ('triangles', np.asarray(mesh.triangles)),
              ('normals', np.asarray(mesh.vertex_normals))]
    if quality is not None:
        arrays += [(key, quality[key]) for key in sorted(quality) if key != 'label']
    digest = hashlib.sha256()
    for name, value in arrays:
        array = np.ascontiguousarray(value)
        digest.update(json.dumps([name, array.dtype.str, array.shape]).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def replay_prefix(base, folder, world, evaluator):
    """Independent initial+20 paid-action replay, without a collection helper."""
    import numpy as np
    prefix = folder / 'prefix'
    records = base.read_json(prefix / 'records.json')
    base.require(len(records) == 21 and records[-1]['step'] == 20, 'V7 prefix is not initial plus 20 actions')
    base.compare([row['action'] for row in records[1:]], list(world.prefix_actions), 'fixed prefix action contract')
    for name in ('frames', 'scans'):
        base.inventory_npz(prefix / name, range(21))
    if (prefix / 'source_hashes.json').exists():
        hashes = base.read_json(prefix / 'source_hashes.json')
        expected = {f'{kind}/{i:04d}.npz' for kind in ('frames', 'scans') for i in range(21)}
        base.require(set(hashes) == expected, 'prefix hash inventory differs')
        for name, value in hashes.items():
            base.require(base.file_hash(base.relative_file(prefix, name)) == value, f'prefix source hash mismatch: {name}')
    frames, scans = [], []
    mapper = base.SemanticHistoryMapperV3(world.shape, world.config, world.config.truncation_m)
    seen = np.zeros(len(evaluator.reference), bool)
    for index, row in enumerate(records):
        base.require(index == 0 or not records[index - 1]['done'], 'prefix continues after terminal state')
        if index == 0:
            generated, collision, done, action = world.sense(), False, False, 'reset'
        else:
            action = world.prefix_actions[index - 1]
            generated, collision, done = world.step(action)
        frame = base.RGBDFrame.load(prefix / 'frames' / f'{index:04d}.npz')
        scan = base.PlanarScan.load(prefix / 'scans' / f'{index:04d}.npz')
        base.compare_sensor(generated, frame, f'{folder.name}/prefix/{index}/RGB-D')
        base.compare_sensor(world.scan(), scan, f'{folder.name}/prefix/{index}/scan')
        expected = {'step': index, 'position': list(world.position), 'heading': world.heading,
                    'collision': collision, 'collisions': world.collisions, 'moves': world.moves,
                    'action': action, 'done': done}
        base.compare(row, expected, f'{folder.name}/prefix record')
        base.require(not collision and not done, 'predeclared paired prefix collided or exhausted its budget')
        frames.append(frame); scans.append(scan)
        mapper.update(frame, scan); base.collision_update(mapper, row)
        pose = base.camera_pose(world.position, world.heading, world.config, world.shape[0])
        seen |= base.reference_visible(evaluator.reference, frame, evaluator.truth, pose, world.config.max_depth_m)
    base.compare_npz(folder / 'prefix_mesh.npz', base.mesh_arrays(mapper.mesh()))
    base.compare_npz(folder / 'prefix_map.npz', {'belief': mapper.belief})
    if (folder / 'prefix_visibility.npz').exists():
        base.compare_npz(folder / 'prefix_visibility.npz', {'seen': seen})
    return frames, scans, records, mapper, seen


def verify_features(base, folder, mapper, frames, scans, records, routes):
    """Frozen implementation replay, explicitly not an independent V7 derivation."""
    import numpy as np
    from nso.response_features_v7 import response_features, ResponseFeatureConfig
    from nso.counterfactual_view_scoring import _mapper_snapshot
    audit = base.read_json(folder / 'feature_audit.json')
    feature_config = ResponseFeatureConfig(**audit['feature_config'])
    actual, actual_audit = response_features(mapper, routes, feature_config)
    base.compare(actual_audit, audit, f'{folder.name}/V7 feature audit', tolerance=1e-12)
    base.compare_npz(folder / 'features.npz', actual)
    base.require(set(actual) == set(CHANNELS), 'V7 channel inventory differs')
    np.testing.assert_array_equal(actual['G'], actual['M'])
    np.testing.assert_array_equal(actual['S'][:, :8], actual['O'][:, :8])
    np.testing.assert_array_equal(actual['S'][:, :8], actual['X'][:, :8])
    np.testing.assert_array_equal(actual['S'][:, 8:], -actual['X'][:, 8:])
    for name, matrix in actual.items():
        base.require(matrix.shape == (len(routes), 12) and np.isfinite(matrix).all(), 'invalid V7 matrix')
        np.testing.assert_array_equal(matrix[:, :4], actual['G'][:, :4])
    legacy = base.read_json(folder / 'legacy_predictions.json')
    base.validate_predictions(legacy, routes)
    hashes = {'aligned': _mapper_snapshot(mapper)['hashes']}
    base.compare(hashes['aligned'], legacy['invariants']['mapper_hashes']['aligned'], 'aligned mapper fingerprint', tolerance=0)
    for condition in ('shuffled', 'absent'):
        changed = []
        for frame in frames:
            labels = (np.where(frame.semantic == 2, 3, np.where(frame.semantic == 3, 2, frame.semantic))
                      if condition == 'shuffled' else np.zeros_like(frame.semantic)).astype(frame.semantic.dtype)
            changed.append(replace(frame, semantic=labels))
        other = base.remap_prefix(changed, scans, records, mapper.config, mapper.shape)
        hashes[condition] = _mapper_snapshot(other)['hashes']
        base.compare(hashes[condition], legacy['invariants']['mapper_hashes'][condition],
                     f'{condition} mapper fingerprint', tolerance=0)
        matrices, _ = response_features(other, routes, feature_config)
        if condition == 'shuffled':
            np.testing.assert_array_equal(actual['X'], matrices['S'])
        else:
            np.testing.assert_array_equal(actual['M'], matrices['G'])
            np.testing.assert_array_equal(actual['M'], matrices['M'])
        del other
    return actual, {'status': 'passed_frozen_implementation_replay', 'actual_label_interventions_remapped': True,
                    'independent_descriptor_implementation': False, 'mapper_hashes': hashes}


def verify_family(base, run, family, context, config, selected):
    import numpy as np
    from env.virtual3d_response_v7 import ResponseWorldV7, ResponseContextV7, ResponseConfigV7
    folder = run / family
    fixture = base.read_json(folder / 'fixture.json')
    base.compare(fixture['context'], asdict(context), 'family context', tolerance=0)
    base.compare(fixture['family'], family, 'family name')
    c = ResponseConfigV7(**fixture['config'])
    base.require(c.pose_noise_m == 0 and c.max_steps >= 68, 'invalid pose/paid action budget')
    base.compare(c.depth_sigma_m, context.depth_sigma_m, 'context sensor noise', tolerance=0)
    world = ResponseWorldV7(ResponseContextV7(**fixture['context']), family, c)
    evaluator = base.ReconstructionEvaluator(world, config['reference_samples'])
    weights = np.full(len(evaluator.reference), world.mesh.get_surface_area() / config['reference_samples'])
    base.compare_npz(folder / 'reference.npz', {'points': evaluator.reference, 'classes': evaluator.classes,
        'weights': weights, 'reachable': world.reachable, **base.mesh_arrays(world.mesh)})
    frames, scans, records, mapper, seen = replay_prefix(base, folder, world, evaluator)
    geometry = geometry_hash(mapper)
    preparation = base.read_json(folder / 'preparation.json')
    base.compare(geometry, preparation['nonlabel_geometry_sha256'], 'prepared nonlabel geometry', tolerance=0)
    routes = base.read_json(folder / 'candidates.json')
    base.compare([r['candidate_id'] for r in routes], list(range(len(routes))), 'V7 candidate identifiers')
    base.require(len(routes) == 6, 'prepared V7 six-role structure gate is incomplete')
    base.require(len({tuple(r['pose']) for r in routes}) == len(routes), 'duplicate candidate endpoint')
    expected_branches = {f"candidate_{r['candidate_id']:03d}" for r in routes}
    base.require({p.name for p in folder.glob('candidate_[0-9]*')} == expected_branches, 'V7 branch inventory differs')
    start = (*records[-1]['position'], records[-1]['heading'])
    safe = ~base.inflated_obstacles(mapper.belief != 0, c.robot_radius_m / c.resolution_m)
    # The legacy helper permits an initial-cell exception. Explicitly prohibit
    # it here for V7 before invoking that otherwise independent path validator.
    base.require(bool(safe[start[:2]]), 'V7 initial footprint not entirely known free')
    audit = base.read_json(folder / 'candidate_audit.json')
    base.compare(audit['safe_hash'], hashlib.sha256(safe.tobytes()).hexdigest(), 'V7 safe map hash')
    base.compare(audit['semantics_used_for_selection'], False, 'label-agnostic candidate contract')
    base.compare(audit['outcome_used_for_selection'], False, 'outcome-free candidate contract')
    for route in routes:
        base.validate_route(route, start, safe, config['branch_actions'])
        base.require(all(bool(safe[tuple(state[:2])]) for state in route['states']), 'V7 footprint exception attempted')
    features, feature_review = verify_features(base, folder, mapper, frames, scans, records, routes)
    coverage = base.occupancy_coverage(mapper, world.reachable)
    before = evaluator.evaluate(mapper.mesh(), coverage, config['thresholds_m'])
    outcomes = base.read_json(folder / 'outcomes.json')
    base.compare([r['candidate_id'] for r in outcomes], [r['candidate_id'] for r in routes], 'family outcome identifiers')
    checked = []
    for route, outcome in zip(routes, outcomes):
        branch = folder / f"candidate_{route['candidate_id']:03d}"
        base.compare(base.read_json(branch / 'outcome.json'), outcome, 'family/branch outcome copy', tolerance=0)
        relative = str(branch.relative_to(run))
        if relative in selected:
            result = base.replay_branch(branch, route, world, evaluator, frames, scans, records, config, weights, seen, before)
            result['path'] = relative; checked.append(result)
    report = {'family': family, 'context_id': context.context_id, 'role': context.role, 'prefix_frames': 21,
              'branches_total': len(routes), 'checked_branches': checked, 'nonlabel_geometry_sha256': geometry,
              'reference_sha256': base.file_hash(folder / 'reference.npz'),
              'reference_weight_definition': 'GT unique exterior mesh area / 32000 original requested samples; no filtered-area renormalization',
              'safe_footprint_exception_used': False, 'feature_review': feature_review,
              'dedicated_prefix_visibility_file_checked': (folder / 'prefix_visibility.npz').exists(),
              'prefix_visibility_checked_against_every_replayed_branch': True,
              'simple_recall_means': 'class 2 storage_shelves', 'complex_recall_means': 'class 3 ventilation_baffles'}
    print('replayed', context.context_id, family, len(checked), '/', len(routes), 'branches', flush=True)
    return report, (frames, scans, records, routes, features, geometry)


def worker(args):
    import numpy as np
    base = load_base(args.base)
    base.worker_init(str(args.snapshot))
    from env.virtual3d_response_v7 import get_response_context, RESPONSE_FAMILIES, load_response_contexts
    run = args.run
    config = base.read_json(run / 'config.json')
    metadata = base.read_json(run / 'metadata.json')
    base.compare(config['thresholds_m'], [.02, .05], 'fixed metric thresholds', tolerance=0)
    base.compare(config['reference_samples'], 32000, 'fixed reference sample request')
    base.compare(config['branch_actions'], 48, 'fixed paid branch budget')
    base.compare(config['candidate_limit'], 6, 'fixed candidate cap')
    base.compare(config, base.read_json(args.snapshot / 'configs/virtual3d/response_v7_pipeline.json'),
                 'run/frozen pipeline configuration', tolerance=0)
    families = [p.parent.name for p in run.glob('*/fixture.json')]
    base.compare(sorted(families), sorted(RESPONSE_FAMILIES), 'paired family inventory')
    first_fixture = base.read_json(run / RESPONSE_FAMILIES[0] / 'fixture.json')
    context_id = first_fixture['context']['context_id']
    context = get_response_context(context_id)
    base.compare(first_fixture['context'], asdict(context), 'exact frozen parent manifest membership', tolerance=0)
    base.require(context.outer_seed in (751, 752), 'unopened evaluation seed')
    allowed_contexts = config['train_contexts'] if context.role == 'train' else config['calibration_contexts']
    base.require(context_id in allowed_contexts, 'parent role differs from frozen acquisition matrix')
    construction = load_response_contexts()
    base.compare(list(RESPONSE_FAMILIES), construction['family_order'], 'frozen family order')
    if 'context' in metadata:
        base.compare(metadata['context'], asdict(context), 'metadata parent context', tolerance=0)
    seal = base.read_json(run / 'pre_outcome_seal.json')
    base.require(isinstance(seal, dict) and bool(seal), 'missing pre-outcome seal')
    prepared = Path(metadata['prepared'])
    base.require(base.file_hash(prepared / 'artifact_hashes.json') == metadata['prepared_manifest_sha256'],
                 'original preparation manifest changed')
    preparation_manifest = base.read_json(prepared / 'artifact_hashes.json')
    preparation_meta = base.read_json(prepared / 'metadata.json')
    base.require(preparation_meta['status'] == 'complete' and preparation_meta['candidate_structure_gate_passed'],
                 'original preparation did not pass its declared structure gate')
    if context.role == 'calibration':
        base.require(preparation_meta['bank_sha256'] is not None, 'calibration lacks a frozen trained-bank hash')
        for family in RESPONSE_FAMILIES:
            base.require({f'{family}/predictions.json', f'{family}/choices.json'} <= set(seal),
                         'calibration input seal omitted predictions or selections')
    for family in RESPONSE_FAMILIES:
        required = {f'{family}/{name}' for name in ('fixture.json', 'candidates.json', 'candidate_audit.json',
                     'feature_audit.json', 'features.npz', 'legacy_predictions.json', 'prefix/records.json')}
        base.require(required <= set(seal), 'pre-outcome seal omitted a decision/prefix asset')
    for name, expected in seal.items():
        base.require(base.file_hash(base.relative_file(run, name)) == expected, f'pre-outcome seal mismatch: {name}')
        base.require(preparation_manifest.get(name) == expected and
                     base.file_hash(base.relative_file(prepared, name)) == expected,
                     f'execution decision input differs from original preparation: {name}')
    branches = sorted(str(p.relative_to(run)) for p in run.glob('*/candidate_[0-9]*') if p.is_dir())
    selected = set(branches if args.max_branches is None else branches[:args.max_branches])
    reports, pairs = [], []
    for family in RESPONSE_FAMILIES:
        report, pair = verify_family(base, run, family, context, config, selected)
        reports.append(report); pairs.append(pair)
    a, b = pairs
    base.compare(a[2], b[2], 'paired prefix action records', tolerance=0)
    base.compare(a[3], b[3], 'paired complete candidate routes', tolerance=0)
    base.compare(a[5], b[5], 'paired nonlabel measured geometry')
    for fa, fb, sa, sb in zip(a[0], b[0], a[1], b[1]):
        for field in ('timestamp_s', 'depth_m', 'intrinsic', 'world_from_camera'):
            base.compare_array(getattr(fa, field), getattr(fb, field), f'paired prefix {field}')
        base.compare_sensor(sa, sb, 'paired prefix laser')
        base.compare_array(fa.semantic > 0, fb.semantic > 0, 'paired visible marker mask')
        nonmarker = (fa.semantic == 0) & (fb.semantic == 0)
        base.compare_array(fa.color_rgb[nonmarker], fb.color_rgb[nonmarker], 'paired nonmarker RGB')
    for name in ('G', 'O', 'N', 'M', 'G_capacity'):
        base.compare_array(a[4][name], b[4][name], f'paired common feature {name}')
    base.compare_array(a[4]['S'][:, :8], b[4]['S'][:, :8], 'paired S common columns')
    base.compare_array(a[4]['S'][:, 8:], -b[4]['S'][:, 8:], 'paired S category sign')
    checked = [r for family in reports for r in family['checked_branches']]
    base.require({row['path'] for row in checked} == selected, 'missing or duplicated requested branch replay')
    base.require(sum(r['branches_total'] for r in reports) == len(branches), 'global branch inventory differs')
    if 'branches' in metadata:
        base.compare(metadata['branches'], len(branches), 'metadata branch count')
    all_outcomes = [{'family': family, **row} for family in RESPONSE_FAMILIES
                    for row in base.read_json(run / family / 'outcomes.json')]
    base.compare(metadata['physical_branches'], len(all_outcomes), 'metadata physical branch count')
    base.compare(metadata['paid_actions'], sum(r['paid_actions'] for r in all_outcomes), 'metadata paid action count')
    base.compare(metadata['failures'], sum(r['failure'] is not None for r in all_outcomes), 'metadata branch failures')
    base.compare(base.read_json(run / 'summary.json'), {'context': context_id, 'outcomes': all_outcomes,
                 'scope': 'all predeclared candidates, no fitted V7 superiority claim'}, 'complete V7 aggregate summary')
    for name, expected in seal.items():
        base.require(base.file_hash(run / name) == expected, 'sealed decision assets changed during replay')
        base.require(base.file_hash(prepared / name) == expected, 'original preparation inputs changed during replay')
    base.require(base.file_hash(prepared / 'artifact_hashes.json') == metadata['prepared_manifest_sha256'],
                 'original preparation manifest changed during replay')
    # All env/nso/utils imports must resolve to the verified immutable snapshot.
    for name, module in list(sys.modules.items()):
        if name.split('.')[0] in ('env', 'nso', 'utils') and getattr(module, '__file__', None):
            base.require(Path(module.__file__).resolve().is_relative_to(args.snapshot), f'live module imported: {name}')
    import open3d
    import scipy
    result = base.completion_status(args.max_branches, len(checked), len(branches))
    result.update(families=reports, context_id=context_id, role=context.role,
                  paid_branch_actions_checked=sum(r['paid_actions'] for r in checked), prefix_actions_checked=40,
                  paired_geometric_prefix_exact=True, paired_route_pools_exact=True, paired_nonsemantic_features_exact=True,
                  new_physical_routes_generated=False, prefix_collection_helper_used=False,
                  feature_validation_scope='frozen V7 implementation replay plus label invariants; not independent descriptor mathematics',
                  physics_and_accounting_scope='all recorded RGB-D/scans, coverage each action, reference and prefix/arrival/final meshes, weighted area union, P/R/F1 and branch J-AUC',
                  dependencies={'numpy': np.__version__, 'scipy': scipy.__version__, 'open3d': open3d.__version__})
    write_json(args.result, result)


def verify(args):
    base = load_base(BASE_VERIFIER)
    base.require(args.workers == 1, 'this bounded V7 replay uses exactly one CPU worker')
    base.require(args.max_branches is None or args.max_branches >= 0, 'negative replay cap')
    run = args.run
    base.require(run.is_dir(), 'run does not exist')
    own_sha, base_sha = base.file_hash(__file__), base.file_hash(BASE_VERIFIER)
    started = time.monotonic()
    report = {'schema_version': 'response_v7_replay/1', 'status': 'failed', 'passed_full': False,
              'run': str(run), 'workers': 1, 'max_branches': args.max_branches, 'partial': args.max_branches is not None,
              'verifier_sha256': own_sha, 'base_verifier_sha256': base_sha, 'python': platform.python_version(),
              'raw_artifacts_modified': False, 'meshes_deleted': False, 'independent_confirmation': False,
              'scope': 'frozen-source paired raw-sensor and paid-branch replay; no efficacy conclusion'}
    try:
        manifest = base.read_json(run / 'artifact_hashes.json')
        manifest_sha = base.file_hash(run / 'artifact_hashes.json')
        base.check_manifest(run, manifest)
        metadata = base.read_json(run / 'metadata.json')
        base.require(metadata['status'] == 'complete', 'V7 run is not complete')
        if 'config_sha256' in metadata:
            base.compare(metadata['config_sha256'], base.json_hash(base.read_json(run / 'config.json')), 'config hash')
        with tempfile.TemporaryDirectory(prefix='response-v7-independent-replay-') as name:
            temporary = Path(name)
            snapshot = temporary / 'sources'; snapshot.mkdir()
            base.extract_sources(run, snapshot, metadata['source_sha256'])
            copied_base = temporary / 'independent_base.py'; shutil.copyfile(BASE_VERIFIER, copied_base)
            base.require(base.file_hash(copied_base) == base_sha, 'independent helper changed while copying')
            result = temporary / 'result.json'
            command = [sys.executable, str(Path(__file__).resolve()), '--worker', '--run', str(run),
                       '--snapshot', str(snapshot), '--base', str(copied_base), '--result', str(result)]
            if args.max_branches is not None:
                command += ['--max-branches', str(args.max_branches)]
            env = os.environ | {'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
                                'NUMEXPR_NUM_THREADS': '1', 'PYTHONDONTWRITEBYTECODE': '1'}
            completed = subprocess.run(command, cwd=temporary, env=env, check=False)
            if completed.returncode:
                worker_failure = base.read_json(result) if result.exists() else {}
                report['worker_failure'] = worker_failure
                raise ValueError(worker_failure.get('error', f'isolated replay exited {completed.returncode}'))
            report.update(base.read_json(result))
        base.check_manifest(run, manifest)
        base.require(base.file_hash(run / 'artifact_hashes.json') == manifest_sha, 'artifact manifest changed during replay')
        base.require(base.file_hash(__file__) == own_sha and base.file_hash(BASE_VERIFIER) == base_sha,
                     'independent replay source changed during replay')
        report.update(artifact_manifest_sha256=manifest_sha, source_archive_sha256=base.file_hash(run / 'sources.zip'),
                      source_sha256=metadata['source_sha256'], raw_hashes_rechecked_after_replay=True)
    except Exception as error:
        report.update(status='failed', passed_full=False, error=str(error), traceback=traceback.format_exc())
    report['elapsed_s'] = time.monotonic() - started
    with tempfile.NamedTemporaryFile(mode='w', prefix='.response-v7-verification-', suffix='.json', dir=run, delete=False) as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n'); temporary = Path(stream.name)
    temporary.replace(run / 'verification.json')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=lambda s: Path(s).resolve())
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--max-branches', type=int)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--snapshot', type=Path)
    parser.add_argument('--base', type=Path)
    parser.add_argument('--result', type=Path)
    args = parser.parse_args()
    if args.worker:
        try:
            worker(args)
        except Exception as error:
            write_json(args.result, {'status': 'failed', 'error': str(error), 'traceback': traceback.format_exc()})
            raise
    else:
        result = verify(args)
        print(json.dumps({key: result[key] for key in ('status', 'passed_full', 'elapsed_s')}
                         | {'verification': str(args.run / 'verification.json')}, ensure_ascii=False), flush=True)
        if result['status'] == 'failed':
            print(result['error'], file=sys.stderr)
            sys.exit(1)
