#!/usr/bin/env python3
"""Bounded, offline V37 asset inventory; never download or execute a model."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nso.research_evidence_v31 import freeze, seal, verify_inventory, verify_sources, write

OUTPUT = ROOT / 'audit_results/v37_semantic_assets_20260918'
SCOPES = ('data', 'docs', 'semantic', 'nso', 'scripts', 'configs', 'third_party')
EXCLUDED = {'.git', '__pycache__', 'build', 'node_modules', '.venv', '.venv-cpu', '.venv-3d'}
MEDIA = {'.jpg', '.jpeg', '.png', '.webp'}
MODULES = ('numpy', 'PIL', 'torch', 'torchvision', 'ultralytics', 'cv2',
           'onnxruntime', 'transformers', 'open_clip', 'rospy', 'tf2_ros',
           'message_filters', 'cv_bridge', 'sensor_msgs', 'geometry_msgs', 'rosbag')


def digest(path):
    hasher = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def files_under(base):
    for directory, directories, names in os.walk(base, followlinks=False):
        directories[:] = sorted(d for d in directories if d not in EXCLUDED)
        for name in sorted(names):
            path = Path(directory) / name
            if path.is_file() and not path.is_symlink():
                yield path


def probe_environment(relative):
    interpreter = ROOT / relative / 'bin/python'
    if not interpreter.is_file():
        return {'interpreter': str(interpreter), 'available': False}
    code = ('import importlib.util,importlib.metadata,json,sys; '
            'out={"python":sys.version,"modules":{}}\n'
            f'for name in {MODULES!r}:\n'
            ' spec=importlib.util.find_spec(name)\n'
            ' out["modules"][name]={"discoverable":spec is not None,'
            '"origin":None if spec is None else spec.origin}\n'
            'print(json.dumps(out))')
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='',
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    result = subprocess.run([str(interpreter), '-B', '-c', code], env=env,
                            capture_output=True, text=True, timeout=30, check=True)
    record = json.loads(result.stdout)
    record.update(interpreter=str(interpreter), available=True,
                  discovery_is_not_runtime_compatibility_test=True)
    if record['modules']['torch']['discoverable']:
        smoke = ('import json\ntry:\n import torch\n '
                 'print(json.dumps(dict(imported=True,version=torch.__version__,'
                 'cuda_available=torch.cuda.is_available(),cpu_tensor=(torch.tensor([1,2])+1).tolist())))'
                 '\nexcept Exception as e: print(json.dumps(dict(imported=False,'
                 'error_type=type(e).__name__,error=str(e)[:1000])))')
        result = subprocess.run([str(interpreter), '-B', '-c', smoke], env=env,
                                capture_output=True, text=True, timeout=30, check=True)
        record['torch_cpu_arithmetic_smoke'] = json.loads(result.stdout)
    return record


def inventory():
    media = []
    for scope in SCOPES:
        for path in files_under(ROOT / scope):
            if path.suffix.lower() in MEDIA:
                rel = str(path.relative_to(ROOT))
                role = ('project_research_figure' if rel.startswith('docs/research/figures/')
                        else 'third_party_documentation_or_example' if rel.startswith('third_party/')
                        else 'unqualified_media_asset')
                media.append(dict(path=rel, bytes=path.stat().st_size, sha256=digest(path),
                                  role=role, qualified_device_instance_dataset=False))
    bags = [dict(path=str(p.relative_to(ROOT)), bytes=p.stat().st_size)
            for p in files_under(ROOT)
            if p.suffix.lower() in {'.bag', '.mcap'} or p.name.endswith('.bag.active')]
    checkpoint = ROOT / 'yolov8n.pt'
    pointer = checkpoint.read_text()
    match = re.fullmatch(r'version https://git-lfs.github.com/spec/v1\noid sha256:([a-f0-9]{64})\nsize (\d+)\n?', pointer)
    if not match:
        raise ValueError('Expected existing YOLO LFS pointer; audit contract changed')
    oid, size = match[1], int(match[2])
    cache = ROOT / '.git/lfs/objects' / oid[:2] / oid[2:4] / oid
    cache_hash = digest(cache) if cache.is_file() else None
    model = dict(path='yolov8n.pt', pointer_sha256=digest(checkpoint), lfs_oid=oid,
                 expected_bytes=size, local_cache_present=cache.is_file(),
                 local_cache_sha256=cache_hash,
                 local_cache_bytes=cache.stat().st_size if cache.is_file() else None,
                 cache_matches_pointer=cache_hash == oid and cache.stat().st_size == size
                     if cache.is_file() else False,
                 weights_deserialized=False, detector_executed=False, network_access=False)
    environments = {name: probe_environment(name) for name in ('.venv', '.venv-cpu', '.venv-3d')}
    return dict(schema='nso_v37_semantic_assets/1', media_scan_roots=list(SCOPES),
                directory_exclusions=sorted(EXCLUDED), symlinks_followed=False,
                media_scan_excludes_frozen_audit_and_eval_frames=True,
                scoped_media_count=len(media), media=media,
                raw_bag_scan_root='workspace except listed exclusions', raw_bag_files=bags,
                qualified_natural_device_instances=0,
                qualification_note='No task device-instance labels, split or class-to-configuration evidence supplied; incidental media are not a validation dataset.',
                checkpoint=model, environments=environments,
                ros_commands={c: shutil.which(c) for c in ('roscore', 'rosbag', 'rostopic', 'rosparam', 'rosrun')},
                natural_frontend_inference_executed=False, natural_accuracy=None,
                natural_rejection_rate=None, model_inference_latency_ms=None,
                ros_communication_executed=False, new_main_tasks=0,
                new_world_TSDF_quality_calls=0, main_tasks_used=35, main_tasks_limit=36,
                free_bytes=shutil.disk_usage(ROOT).free)


def run():
    if OUTPUT.exists():
        raise FileExistsError('exclusive-create asset audit: ' + str(OUTPUT))
    result = inventory()
    OUTPUT.mkdir()
    freeze(OUTPUT, [Path(__file__), ROOT / 'yolov8n.pt'],
           scope='Read-only inventory plus CPU tensor arithmetic; no model execution or download')
    write(OUTPUT, OUTPUT / 'result.json', result)
    verify_sources(OUTPUT)
    seal(OUTPUT)
    verify_inventory(OUTPUT)
    print(json.dumps(dict(output=str(OUTPUT.relative_to(ROOT)), media_count=result['scoped_media_count'],
                          bag_count=len(result['raw_bag_files']), local_yolo_cache_verified=result['checkpoint']['cache_matches_pointer'],
                          bytes=sum(p.stat().st_size for p in OUTPUT.iterdir()))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    if parser.parse_args().run:
        run()
    else:
        parser.print_help()
