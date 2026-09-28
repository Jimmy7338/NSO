#!/usr/bin/env python3
"""Administrative Exposure V3 adapter over the unchanged bounded article scheduler.

Loads a private module instance: existing scheduler processes and the original
Python module are never modified. All claims, locks and cumulative ledgers use
the original implementation. Scientific validation runs in the cached runtime.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
RUNNER = 'scripts/run_article_exposure_experiment_20260928.py'
SCHEMA = 'article.experiment_protocol.exposure_v3'
CORE_PATH = ROOT/'scripts/run_article_batch_20260928.py'
_spec = importlib.util.spec_from_file_location('_article_exposure_private_scheduler', CORE_PATH)
_core = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_core)


def load_protocol(path, root, expected_sha=None):
    content = Path(path).read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    if expected_sha is not None and digest != expected_sha:
        raise ValueError('protocol changed after batch start')
    protocol = json.loads(content)
    if (protocol.get('schema') != SCHEMA or protocol.get('status') != 'frozen'
            or protocol.get('phase') not in ('ablation','main')):
        raise ValueError('frozen Exposure V3 ablation/main protocol required')
    slots = protocol.get('slots', {})
    limit = 12 if protocol['phase'] == 'ablation' else _core.CAPS['main']
    if not isinstance(slots,dict) or not 1 <= len(slots) <= limit:
        raise ValueError('invalid finite declared exposure slots')
    if any(not re.fullmatch(r'[A-Za-z0-9_.-]{1,80}',key) for key in slots):
        raise ValueError('invalid run identifier')
    output = _core.within(root, protocol['output_relative_path'])
    if (Path(protocol['output_relative_path']).parts[:2] != ('audit_results','article_stage_20260928')
            or len(Path(protocol['output_relative_path']).parts) != 3):
        raise ValueError('isolated article phase output required')
    if (protocol['reserve_bytes'] < _core.GIB
            or not 2*1024**2 <= protocol['maximum_episode_bytes'] <= _core.EPISODE_ALLOWANCE):
        raise ValueError('resource contract differs')
    pins = protocol.get('source_sha256', {})
    if not isinstance(pins,dict) or RUNNER not in pins:
        raise ValueError('frozen exposure runner source pin required')
    for name,pin in pins.items():
        if _core.sha(_core.within(root,name)) != pin:
            raise ValueError('frozen source changed: '+name)
    if _core.sha(_core.within(root,protocol['source_archive'])) != protocol['source_archive_sha256']:
        raise ValueError('frozen source archive changed')
    return protocol,digest,output


_core.RUNNER = RUNNER
_core.load_protocol = load_protocol
# Reports identify the wrapper as the entry point; the reused scheduler pin is
# additionally retained below. This changes a private module variable only.
_core.__file__ = __file__


def run_batch(protocol_path, output, *, workers=2, max_wall_seconds=None,
              root=ROOT, preflight=True, **test_hooks):
    root = Path(root).resolve(); protocol_path = Path(protocol_path).resolve()
    output = Path(output).resolve()
    if type(workers) is not int or not 1 <= workers <= 2:
        raise ValueError('workers must be 1 or 2')
    protocol,pin,_ = load_protocol(protocol_path,root)
    if output.exists():
        raise FileExistsError('fresh batch output required')
    if preflight:
        python = root/'.venv-3d/bin/python'  # retain venv symlink semantics
        if not python.exists():
            raise ValueError('cached .venv-3d interpreter unavailable')
        code = ('import json,sys; from nso.article_experiment_exposure_v3 import '
            'validate_protocol,validate_pairing,source_hashes,runtime_metadata; '
            'p=validate_protocol(json.load(open(sys.argv[1]))); validate_pairing(p); '
            'assert source_hashes()==p["source_sha256"],"source closure changed"; '
            'assert "numerical_runtime" not in p or p["numerical_runtime"]==runtime_metadata(),'
            '"numerical runtime differs"')
        output.parent.mkdir(parents=True,exist_ok=True)
        log = output.with_name(output.name+'.preflight.log')
        with log.open('xb') as stream:
            checked = subprocess.run([str(python),'-B','-c',code,str(protocol_path)],
                cwd=root,env=_core.child_environment(),stdout=stream,stderr=subprocess.STDOUT,timeout=90)
        if checked.returncode:
            raise ValueError('exposure preflight failed; see '+str(log))
    result = _core.run_batch(protocol_path,output,workers=workers,
        max_wall_seconds=max_wall_seconds,root=root,preflight=False,**test_hooks)
    result.update(exposure_runner=RUNNER,exposure_protocol_schema=SCHEMA,
                  reused_scheduler_sha256=_core.sha(CORE_PATH))
    _core.save(output/'batch.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workers',type=int,choices=(1,2),default=2)
    parser.add_argument('--max-wall-seconds',type=float)
    args=parser.parse_args()
    result=run_batch(args.protocol,args.output,workers=args.workers,max_wall_seconds=args.max_wall_seconds)
    print(json.dumps(result,ensure_ascii=False),flush=True)
    if result['status']!='complete':
        raise SystemExit(1)


if __name__=='__main__':
    main()
