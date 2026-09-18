"""Replay every V19 sensor condition from saved depth in a fresh process.

This verifies reproducibility of the frozen implementation and measurements;
it is not an independent reconstruction algorithm or a sensor calibration.
The original artifact directory is read-only throughout this script.
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def differences(expected, actual, path=''):
    """Keep exact equality as the criterion; expose every mismatch plainly."""
    if isinstance(expected, dict) and isinstance(actual, dict):
        result = []
        for key in sorted(set(expected) | set(actual)):
            here = f'{path}.{key}' if path else key
            if key not in expected or key not in actual:
                result.append(dict(path=here, expected=expected.get(key), actual=actual.get(key)))
            else:
                result.extend(differences(expected[key], actual[key], here))
        return result
    if expected == actual:
        return []
    result = dict(path=path, expected=expected, actual=actual)
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        result['absolute_difference'] = abs(actual - expected)
    return [result]


def audit_source(source):
    manifest = json.loads((source / 'manifest.json').read_text())
    if manifest.get('status') != 'complete' or manifest.get('reconstructions') != 60:
        raise ValueError('Source must be the complete frozen 60-condition V19 probe')
    checks = []
    for name, expected in manifest['artifact_sha256'].items():
        path = source / name
        if path.parent != source or not path.is_file():
            raise ValueError(f'Invalid or missing artifact: {name}')
        actual = sha(path)
        checks.append(dict(kind='artifact', path=str(path), sha256=actual, expected=expected))
        if actual != expected:
            raise ValueError(f'Artifact hash mismatch: {path}')
    if sha(source / 'config.json') != manifest['configuration_sha256']:
        raise ValueError('Configuration hash mismatch')
    with zipfile.ZipFile(source / 'sources.zip') as archive:
        for name, expected in manifest['source_sha256'].items():
            path = Path(name)
            if not path.is_absolute():
                path = ROOT / path
            actual = sha(path)
            archived = hashlib.sha256(archive.read(path.name)).hexdigest()
            checks.append(dict(kind='source', path=str(path), sha256=actual,
                               archived_sha256=archived, expected=expected))
            if actual != expected or archived != expected:
                raise ValueError(f'Source or archived source hash mismatch: {path}')
    return manifest, checks


def worker(source, output):
    import numpy as np
    import open3d as o3d
    manifest, checks = audit_source(source)
    config = json.loads((source / 'config.json').read_text())
    if o3d.__version__ != manifest['open3d_version'] or np.__version__ != manifest['numpy_version']:
        raise ValueError('Open3D or NumPy version differs from the frozen probe')
    spec = importlib.util.spec_from_file_location('frozen_probe_v19', ROOT / 'scripts/probe_stereo_tsdf_v19.py')
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    if probe.CONFIG != config:
        raise ValueError('Imported probe CONFIG differs from frozen config.json')

    def forbid_generation(*args, **kwargs):
        raise AssertionError('Replay must read saved depth; generation is forbidden')

    probe.depth_sequence = forbid_generation
    original = json.loads((source / 'measurements.json').read_text())
    expected = {(r['model'], r['seed'], r['plan']): r for r in original}
    keys = {(model, seed, plan) for model in config['models']
            for seed in config['seeds'] for plan in config['plans']}
    if len(original) != 60 or len(expected) != 60 or set(expected) != keys:
        raise ValueError('Frozen measurement matrix has missing or duplicate conditions')
    replay = []
    comparisons = []
    raw_inputs = []
    for model in config['models']:
        for seed in config['seeds']:
            raw_path = source / f'{model}_{seed}_depth.npz'
            with np.load(raw_path, allow_pickle=False) as saved:
                if set(saved.files) != {'far', 'near'}:
                    raise ValueError(f'Unexpected saved depth arrays: {raw_path}')
                far, near = saved['far'], saved['near']
            wanted_shape = (config['frames'], config['height'], config['width'])
            for label, array in [('far', far), ('near', near)]:
                if array.shape != wanted_shape or array.dtype != np.dtype('float32'):
                    raise ValueError(f'Raw depth shape/dtype mismatch: {raw_path}/{label}')
                if not np.isfinite(array).all() or np.any(array < 0) or np.any(array >= config['max_depth_m']):
                    raise ValueError(f'Invalid raw depth values: {raw_path}/{label}')
                array.setflags(write=False)
            raw_inputs.append(dict(path=str(raw_path), sha256=sha(raw_path),
                far_array_sha256=hashlib.sha256(far.tobytes()).hexdigest(),
                near_array_sha256=hashlib.sha256(near.tobytes()).hexdigest()))
            sequences = [(far[:1], 4.), (np.repeat(far[:1], config['frames'], axis=0), 4.),
                         (far, 4.), (near[:1], 1.), (near, 1.)]
            for plan, (sequence, distance) in zip(config['plans'], sequences):
                mesh = probe.integrate(sequence, distance)
                row = dict(model=model, seed=seed, plan=plan, distance_m=distance,
                           frame_count=len(sequence), metrics=probe.evaluate(mesh))
                row['mesh_sha256'] = hashlib.sha256(np.asarray(mesh.vertices).tobytes()
                    + np.asarray(mesh.triangles).tobytes()).hexdigest()
                old = expected[(model, seed, plan)]
                mismatch = differences(old, row)
                comparisons.append(dict(model=model, seed=seed, plan=plan,
                    metrics_exact=old['metrics'] == row['metrics'],
                    mesh_hash_exact=old['mesh_sha256'] == row['mesh_sha256'],
                    all_fields_exact=not mismatch, differences=mismatch))
                replay.append(row)
                del mesh
            write(output / 'replayed_measurements.json', replay)
            write(output / 'comparisons.json', comparisons)
            print(f'{model} seed={seed}: replayed 5 saved-depth conditions', flush=True)
    # Recompute published summaries too, retaining the exact original reduction.
    summary = []
    for model in config['models']:
        for plan in config['plans']:
            metrics = [r['metrics'] for r in replay if r['model'] == model and r['plan'] == plan]
            summary.append(dict(model=model, plan=plan,
                mean_abs_error_mm=1000*np.mean([m['mean_abs_error_m'] for m in metrics]),
                p95_abs_error_mm=1000*np.mean([m['p95_abs_error_m'] for m in metrics]),
                signed_mean_error_mm=1000*np.mean([m['signed_mean_error_m'] for m in metrics]),
                f1_05cm=np.mean([m['05cm']['f1'] for m in metrics]),
                f1_02cm=np.mean([m['02cm']['f1'] for m in metrics])))
    published = json.loads((source / 'summary.json').read_text())
    if len(published) != len(summary):
        summary_differences = [dict(path='summary_length', expected=len(published), actual=len(summary))]
    else:
        summary_differences = [diff for i, pair in enumerate(zip(published, summary))
                               for diff in differences(*pair, path=f'summary[{i}]')]
    final_manifest, final_checks = audit_source(source)
    if final_manifest != manifest or final_checks != checks:
        raise ValueError('Frozen source changed during verification')
    write(output / 'replayed_summary.json', summary)
    write(output / 'input_checks.json', dict(before=checks, after=final_checks, raw_inputs=raw_inputs))
    result = dict(status='complete', passed=all(r['all_fields_exact'] for r in comparisons) and not summary_differences,
        reconstructions=len(replay), exact_metric_conditions=sum(r['metrics_exact'] for r in comparisons),
        exact_mesh_conditions=sum(r['mesh_hash_exact'] for r in comparisons),
        exact_all_field_conditions=sum(r['all_fields_exact'] for r in comparisons),
        summaries_exact=not summary_differences, summary_differences=summary_differences,
        raw_depth_files=len(raw_inputs), sensor_depth_regenerated=False,
        worker_pid=os.getpid(), parent_pid=os.getppid(),
        source_manifest_sha256=sha(source / 'manifest.json'),
        open3d_version=o3d.__version__, numpy_version=np.__version__,
        scope='Fresh-process saved-depth replay of the same frozen TSDF/evaluator implementation; no navigation or semantic efficacy claim')
    write(output / 'result.json', result)
    return 0 if result['passed'] else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worker', action='store_true')
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if output == source or source in output.parents:
        raise ValueError('Replay output must be separate from the frozen source directory')
    if args.worker:
        try:
            return worker(source, output)
        except Exception as error:
            write(output / 'failure.json', dict(error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc()))
            raise
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    manifest = dict(status='running', source=str(source), source_manifest_sha256=sha(source / 'manifest.json'),
                    verifier_source_sha256=sha(__file__), parent_pid=os.getpid())
    write(output / 'manifest.json', manifest)
    with zipfile.ZipFile(output / 'sources.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        archive.write(__file__, Path(__file__).name)
    process = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker',
        '--source', str(source), '--output', str(output)], cwd=ROOT, check=False)
    manifest.update(status='complete' if process.returncode == 0 else 'failed',
        worker_exit_code=process.returncode, elapsed_wall_s=time.perf_counter()-started,
        verifier_source_unchanged=sha(__file__) == manifest['verifier_source_sha256'])
    if not manifest['verifier_source_unchanged']:
        manifest['status'] = 'failed'
    manifest['artifact_sha256'] = {p.name: sha(p) for p in output.iterdir() if p.name != 'manifest.json'}
    write(output / 'manifest.json', manifest)
    print(json.dumps(manifest), flush=True)
    return 0 if manifest['status'] == 'complete' else 1


if __name__ == '__main__':
    sys.exit(main())
