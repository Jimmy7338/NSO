#!/usr/bin/env python3
"""Reversible storage compaction for three fixed, fully replayed NSO runs.

Every removable mesh is regenerated from archived mappers and retained raw
RGB-D/scans, then its complete compressed NPZ hash is checked before unlink.
Original manifests and replay receipts are immutable; witnesses live outside
the runs. No simulator, reference, reward, planner or scorer is called.
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / 'audit_results/storage_recovery_20260911'
RUNS = {
    'counterfactual_view_development_first_20260910': (72, 144, 'inspection'),
    'counterfactual_candidate_augmentation_v6_20260911': (16, 32, 'inspection'),
    'competition_v8_1_execution_20260911': (24, 48, 'competition'),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def relative(root, name):
    path = Path(name)
    require(not path.is_absolute() and '..' not in path.parts and '\\' not in name,
            'invalid artifact path')
    result = root / path
    require(result.resolve().is_relative_to(root.resolve()) and not result.is_symlink(),
            'artifact escaped run')
    return result


def targets(manifest, kind):
    result = []
    for name in manifest:
        p = Path(name)
        if p.name not in ('arrival_mesh.npz', 'final_mesh.npz'):
            continue
        require(len(p.parts) == 4 and p.parts[-2].startswith('candidate_'), 'unexpected mesh layout')
        if kind == 'inspection':
            require(p.parts[0].startswith('main_') and p.parts[1].startswith('history_'), 'unexpected history')
        else:
            require(p.parts[0] in ('P0', 'P1') and p.parts[1] in ('shelf_west', 'shelf_east'), 'unexpected context')
        result.append(name)
    return sorted(result)


def bindings(run):
    require(run.parent == ROOT / 'eval_results' and run.name in RUNS, 'run is not whitelisted')
    branches, count, kind = RUNS[run.name]
    manifest = read(run / 'artifact_hashes.json')
    verification = read(run / 'verification.json')
    require(verification['status'] == 'passed_full' and verification['passed_full']
            and not verification['partial'] and verification['max_branches'] is None
            and verification['branches_checked'] == verification['branches_total'] == branches
            and verification['raw_hashes_rechecked_after_replay'], 'full replay prerequisite missing')
    require(verification['artifact_manifest_sha256'] == sha(run / 'artifact_hashes.json')
            and verification['source_archive_sha256'] == sha(run / 'sources.zip'), 'replay binding mismatch')
    selected = targets(manifest, kind)
    require(len(selected) == count, 'fixed mesh inventory changed')
    require(len({str(Path(n).parent) for n in selected}) == branches, 'branch inventory changed')
    hashes = {name: sha(run / name) for name in ('artifact_hashes.json', 'verification.json', 'sources.zip', 'metadata.json')}
    return manifest, selected, hashes


def validate_assets(run, manifest, selected, compact=None):
    missing = []
    for name, digest in manifest.items():
        path = relative(run, name)
        if path.is_file():
            require(sha(path) == digest, 'asset hash mismatch: ' + name)
        else:
            require(compact is not None and compact['status'] == 'complete'
                    and name in selected and name in compact['deleted']
                    and compact['meshes'][name]['file_sha256'] == digest,
                    'missing asset without complete reconstruction witness: ' + name)
            missing.append(name)
    return missing


def validate_witness(compact, audit, hashes, manifest, selected):
    require(compact['schema_version'] == 'completed_branch_mesh_compaction/1'
            and compact['status'] == 'complete' and compact['bindings'] == hashes
            and compact['raw_observations_removed'] == 0 and compact['prefix_and_reference_retained']
            and set(compact['meshes']) == set(selected) and set(compact['deleted']) == set(selected),
            'compaction witness binding mismatch')
    source = audit / 'compaction_tool.py'
    receipt_path = audit / 'raw_reconstruction.json'
    require(sha(source) == compact['tool_sha256']
            and sha(receipt_path) == compact['raw_reconstruction_sha256'], 'reconstruction/tool witness hash mismatch')
    receipt = read(receipt_path)
    require(receipt['status'] == 'passed_all_raw_to_original_npz_bytes'
            and receipt['bindings'] == hashes and receipt['meshes'] == compact['meshes']
            and receipt['tool_sha256'] == compact['tool_sha256']
            and not receipt['world_constructed'] and not receipt['gt_or_outcomes_read'],
            'reconstruction witness incomplete')
    for name, item in compact['meshes'].items():
        require(item['file_sha256'] == manifest[name] and item['file_bytes'] > 0
                and set(item['arrays']) == {'vertices', 'triangles'}
                and item['status'] == 'passed_raw_to_original_npz_bytes', 'mesh witness differs')
        for array in item['arrays'].values():
            require(len(array['sha256']) == 64 and array['nbytes'] >= 0, 'invalid array identity')
    require(receipt['input_hashes'], 'empty raw input witness')
    for name, digest in receipt['input_hashes'].items():
        require(name in manifest and manifest[name] == digest, 'raw witness differs from original manifest')


def extract(run, snapshot):
    expected = read(run / 'metadata.json')['source_sha256']
    with zipfile.ZipFile(run / 'sources.zip') as archive:
        require(len(archive.namelist()) == len(set(archive.namelist()))
                and set(archive.namelist()) == set(expected), 'archive inventory mismatch')
        for name in archive.namelist():
            data = archive.read(name)
            require(hashlib.sha256(data).hexdigest() == expected[name], 'archived source hash mismatch')
            path = relative(snapshot, name)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)


def worker(args):
    sys.path.insert(0, str(args.snapshot))
    import numpy as np
    import open3d
    import scipy
    from nso.semantic_completion_v3 import SemanticHistoryMapperV3
    from utils.rgbd_contract import RGBDFrame, PlanarScan
    from utils.grid_geometry import DIRECTIONS
    kind = RUNS[args.run.name][2]
    if kind == 'inspection':
        from env.virtual3d_inspection_v4 import InspectionConfigV4 as Config
    else:
        from env.virtual3d_competition_v8 import CompetitionConfigV8 as Config
    manifest = read(args.run / 'artifact_hashes.json')
    selected = targets(manifest, kind)
    if args.sample:
        selected = selected[:1]
    evidence = {}
    restored = []
    raw_inputs = set()

    def input_json(path):
        raw_inputs.add(str(path.relative_to(args.run)))
        return read(path)

    def assimilate(mapper, base, index, row):
        paths = [base / modality / f'{index:04d}.npz' for modality in ('frames', 'scans')]
        raw_inputs.update(str(p.relative_to(args.run)) for p in paths)
        mapper.update(RGBDFrame.load(paths[0]), PlanarScan.load(paths[1]))
        if row['collision']:
            dr, dc = DIRECTIONS[row['heading']]
            r, col = row['position'][0] + dr, row['position'][1] + dc
            if 0 <= r < mapper.shape[0] and 0 <= col < mapper.shape[1]:
                mapper.belief[r, col] = 1

    for branch_name in sorted({str(Path(n).parent) for n in selected}):
        branch = args.run / branch_name
        history = branch.parent
        fixture = history.parent if kind == 'inspection' else history
        config = Config(**input_json(fixture / 'fixture.json')['environment' if kind == 'inspection' else 'config'])
        prefix = fixture / 'prefix'
        rows = input_json(prefix / 'records.json')
        last_step = int(history.name.split('_')[1]) if kind == 'inspection' else rows[-1]['step']
        rows = [row for row in rows if row['step'] <= last_step]
        require([row['step'] for row in rows] == list(range(last_step + 1)), 'prefix inventory mismatch')
        mapper = SemanticHistoryMapperV3((round(config.height_m / config.resolution_m),
                                         round(config.width_m / config.resolution_m)), config, config.truncation_m)
        for row in rows:
            assimilate(mapper, prefix, row['step'], row)
        actions = input_json(branch / 'actions.json')
        candidates = input_json(history / 'candidates.json')
        candidate_id = int(branch.name.split('_')[1])
        route = next(row for row in candidates if row['candidate_id'] == candidate_id)
        require(len(actions) == len(route['actions']) == route['cost'], 'only complete routes are compactable')
        arrival = route['arrival_action'] if kind == 'inspection' else next(
            row['action_index'] for row in actions if row['stage'] == 'endpoint')
        for index, row in enumerate(actions, 1):
            require(row['action_index'] == index and row['absolute_step'] == last_step + index
                    and row['action'] == route['actions'][index - 1] and not row['collision']
                    and [*row['position'], row['heading']] == route['states'][index], 'raw paid route ledger mismatch')
            assimilate(mapper, branch, index, row)
            for stage, step in (('arrival', arrival), ('final', len(actions))):
                name = f'{branch_name}/{stage}_mesh.npz'
                if index != step or name not in selected:
                    continue
                mesh = mapper.mesh()
                arrays = {'vertices': np.asarray(mesh.vertices), 'triangles': np.asarray(mesh.triangles)}
                identities = {}
                for key, array in arrays.items():
                    array = np.ascontiguousarray(array)
                    identities[key] = {'dtype': array.dtype.str, 'shape': list(array.shape), 'nbytes': array.nbytes,
                                       'sha256': hashlib.sha256(array.tobytes()).hexdigest()}
                stream = io.BytesIO()
                np.savez_compressed(stream, **arrays)
                data = stream.getvalue()
                require(hashlib.sha256(data).hexdigest() == manifest[name], 'raw-to-NPZ byte identity failed: ' + name)
                evidence[name] = {'file_sha256': manifest[name], 'file_bytes': len(data), 'arrays': identities,
                                  'status': 'passed_raw_to_original_npz_bytes'}
                if args.restore and not (args.run / name).exists():
                    require(shutil.disk_usage(args.run).free >= 400 * 1048576 + len(data), 'restoration reserve exceeded')
                    target = args.run / name
                    with tempfile.NamedTemporaryFile(dir=target.parent, prefix='.restoring-', delete=False) as tmp:
                        tmp.write(data)
                        tmp.flush()
                        os.fsync(tmp.fileno())
                        temporary = Path(tmp.name)
                    require(sha(temporary) == manifest[name], 'restoration temporary hash mismatch')
                    require(not target.exists(), 'target appeared concurrently')
                    # Hard-link publication fails rather than overwrite an existing path.
                    os.link(temporary, target)
                    temporary.unlink()
                    restored.append(name)
            del row
        del mapper
        print('raw regeneration checked', branch_name, flush=True)
    require(set(evidence) == set(selected), 'not all selected meshes reconstructed')
    for name, module in list(sys.modules.items()):
        if name.split('.')[0] in ('nso', 'env', 'utils') and getattr(module, '__file__', None):
            require(Path(module.__file__).resolve().is_relative_to(args.snapshot), 'live source import escaped archive')
    write(args.result, {'status': 'passed_sample' if args.sample else 'passed_all_raw_to_original_npz_bytes',
                       'meshes': evidence, 'input_hashes': {name: manifest[name] for name in sorted(raw_inputs)},
                       'restored': restored, 'world_constructed': False, 'gt_or_outcomes_read': False,
                       'dependencies': {'numpy': np.__version__, 'open3d': open3d.__version__, 'scipy': scipy.__version__}})


def execute(args):
    run = args.run.resolve()
    manifest, selected, hashes = bindings(run)
    audit = AUDIT / run.name
    compact_path = audit / 'compaction.json'
    compact = read(compact_path) if compact_path.exists() else None
    if compact:
        validate_witness(compact, audit, hashes, manifest, selected)
    missing = validate_assets(run, manifest, selected, compact)
    if args.validate_only:
        print(json.dumps({'status': 'validated', 'missing_reconstructable_meshes': len(missing),
                          'original_manifest_entries': len(manifest)}))
        return
    if not args.restore and not args.sample:
        require(compact is None, 'already compacted; use validate-only or restore')
    original_presence = {name: (run / name).exists() for name in selected}
    started = time.monotonic()
    own_sha = sha(__file__)
    write(audit / 'plan.json', {'run': str(run), 'bindings': hashes, 'selected': selected,
          'policy': 'all branch arrival/final TSDF meshes; every prefix mesh and raw observation retained',
          'selected_bytes': sum((run / n).stat().st_size for n in selected if (run / n).exists()),
          'source_sha256': own_sha, 'sample': args.sample, 'restore': args.restore})
    with tempfile.TemporaryDirectory(prefix='nso-branch-mesh-') as tmp:
        temp = Path(tmp)
        snapshot = temp / 'sources'
        extract(run, snapshot)
        result = temp / 'result.json'
        command = [sys.executable, str(Path(__file__).resolve()), '--worker', '--run', str(run),
                   '--snapshot', str(snapshot), '--result', str(result)]
        if args.restore:
            command.append('--restore')
        if args.sample:
            command.append('--sample')
        completed = subprocess.run(command, cwd=temp, env=os.environ | {'PYTHONDONTWRITEBYTECODE': '1'})
        require(completed.returncode == 0 and result.exists(), 'raw mesh regeneration failed')
        checked = read(result)
    require(bindings(run)[2] == hashes and sha(__file__) == own_sha, 'sources/manifests changed during reconstruction')
    validate_assets(run, manifest, selected, compact)
    report = {**checked, 'bindings': hashes, 'tool_sha256': own_sha,
              'elapsed_seconds': time.monotonic() - started, 'original_manifest_modified': False}
    if args.sample or args.restore:
        if args.sample and not args.restore:
            require(original_presence == {n: (run / n).exists() for n in selected}, 'sample changed originals')
        target = audit / ('sample_verification.json' if args.sample else f'restoration_{time.time_ns()}.json')
        write(target, report)
        print(json.dumps({k: v for k, v in report.items() if k not in ('meshes', 'input_hashes')}))
        return
    require(set(report['meshes']) == set(selected), 'full reconstruction gate failed')
    write(audit / 'raw_reconstruction.json', report)
    shutil.copyfile(__file__, audit / 'compaction_tool.py')
    record = {'schema_version': 'completed_branch_mesh_compaction/1', 'status': 'planned', 'run': str(run),
              'bindings': hashes, 'tool_sha256': own_sha,
              'raw_reconstruction_sha256': sha(audit / 'raw_reconstruction.json'),
              'meshes': report['meshes'], 'deleted': [], 'raw_observations_removed': 0,
              'prefix_and_reference_retained': True, 'started_unix_s': time.time(),
              'free_before': shutil.disk_usage(run).free}
    write(compact_path, record)
    for name in selected:
        path = relative(run, name)
        require(sha(path) == manifest[name], 'target changed before unlink')
        path.unlink()
        record['deleted'].append(name)
        write(compact_path, record)
    record.update(status='complete', saved_file_bytes=sum(x['file_bytes'] for x in report['meshes'].values()),
                  completed_unix_s=time.time(), free_after=shutil.disk_usage(run).free)
    write(compact_path, record)
    validate_assets(run, manifest, selected, record)
    print(json.dumps({'status': 'compacted_all_exactly_reconstructable', 'meshes': len(selected),
                      'saved_file_bytes': record['saved_file_bytes'], 'free_after': record['free_after']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--sample', action='store_true')
    parser.add_argument('--restore', action='store_true')
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--snapshot', type=Path)
    parser.add_argument('--result', type=Path)
    args = parser.parse_args()
    worker(args) if args.worker else execute(args)
