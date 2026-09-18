#!/usr/bin/env python3
"""Output-only recovery wrapper for the frozen V27 matched-fixed shadow.

The failed attempt remains intact. --self-test uses synthetic JSON values only.
--run (separate authorization required) repeats the original 401-packet pass,
with exactly its gates/limits; this is not recovery of the lost in-memory rows.
"""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import argparse
import hashlib
import json as standard_json
from pathlib import Path
import shutil
import sys
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUTPUT = ROOT/'audit_results/observed_feedback_v27_shadow_recovered_20260916'
TEST_OUTPUT = ROOT/'audit_results/v27_shadow_serialization_preflight_20260916'
FAILED = ROOT/'audit_results/observed_feedback_v27_shadow_20260916'
PINS = {
    'scripts/audit_v27_feedback_shadow.py': '0430504591e7805dfbcabbd29f491c38d9fafda864f8275bcebeaf054c2463db',
    'scripts/audit_v26_feedback_support.py': '8bb7d9b78b72ab1bec6e1ad63615d981b2c28f192002c92741d19cfffd5942a0',
    'nso/cpu_sensor_contract_v10.py': '28d0870913d67bd102b83b61025a992cf68c76021c270eabd267fb021f14db02',
    'docs/research/V27_SHADOW_SERIALIZATION_RECOVERY_20260916.md': '7f89f2944457f4c24d822d92d150a640b606cffed930c51a92b23d03418cd9cd',
}
FAILURE_INVENTORIES = {
    'audit_results/observed_feedback_v27_shadow_failure_20260916': '58d9e08a575d288b2eeb9ea903b58485199743c6dc560e12284cf79a9b2c7e27',
    'audit_results/observed_feedback_v27_shadow_failure_stack_20260916': 'f6d70bfefa276248bde1eb969b2d01fbf3892c33ed9d0d1d66c274bff6680649',
}
FAILED_PINS = {
    'artifact_hashes.json': '91de924953e0df1a3e6d15e0bbd38a267fd53899ba9a85886a33e7fc87f7d1a0',
    'manifest.json': '570311e0bae70bd302ec64a58b9365255e3450a4e2d77b552ce006fb59ed6717',
    'input_sha256.json': 'd54bc221b6dbb071906dcdabf1e15b90fd92c34e1509562f67116b97463e8e90',
}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def verify_pins():
    for name, expected in PINS.items():
        require(sha(ROOT/name) == expected, 'Frozen recovery dependency changed: '+name)
    require({p.name for p in FAILED.iterdir()} == set(FAILED_PINS), 'Original failed artifact set changed')
    for name, expected in FAILED_PINS.items():
        require(sha(FAILED/name) == expected, 'Original failed artifact changed: '+name)
    inventory = standard_json.loads((FAILED/'artifact_hashes.json').read_text())
    require(inventory == {k:v for k,v in FAILED_PINS.items() if k != 'artifact_hashes.json'},
            'Original failed inventory mismatch')
    # Its last persisted status is running because both result and failure
    # serialization failed. Absence of a result is not treated as completion.
    require(not (FAILED/'result.json').exists(), 'This wrapper only recovers the declared uncompleted attempt')
    for folder, expected in FAILURE_INVENTORIES.items():
        path = ROOT/folder
        require(sha(path/'artifact_hashes.json') == expected, 'Failure receipt inventory changed')
        items = standard_json.loads((path/'artifact_hashes.json').read_text())
        require({p.name for p in path.iterdir()} == set(items) | {'artifact_hashes.json'}, 'Failure receipt set changed')
        for name, digest in items.items():
            require(sha(path/name) == digest, 'Failure receipt/terminal changed')


class LocalJSONProxy:
    """Replace only shadow.json, never mutate the shared stdlib json module."""
    def __init__(self, normalize):
        self.normalize = normalize

    def dumps(self, value, *args, **kwargs):
        kwargs['allow_nan'] = False
        return standard_json.dumps(self.normalize(value), *args, **kwargs)

    @staticmethod
    def loads(*args, **kwargs):
        return standard_json.loads(*args, **kwargs)


def install_output_recovery():
    verify_pins()
    from scripts import audit_v27_feedback_shadow as shadow
    from nso.cpu_sensor_contract_v10 import json_value
    require(not getattr(shadow, '_serialization_recovery_installed', False), 'Recovery may be installed only once')
    original_write = shadow.base.write
    original_inputs = shadow.checked_inputs
    original_snapshot = shadow.source_snapshot

    def normalized_write(out, name, value):
        # All byte/disk/atomic-write guards remain in the original writer.
        return original_write(out, name, json_value(value))

    def inputs_with_failure_provenance():
        verify_pins()
        public = original_inputs()
        for name, expected in FAILED_PINS.items():
            public['input_sha256'][str((FAILED/name).relative_to(ROOT))] = expected
        for folder, expected in FAILURE_INVENTORIES.items():
            public['input_sha256'][folder+'/artifact_hashes.json'] = expected
            for name, digest in standard_json.loads((ROOT/folder/'artifact_hashes.json').read_text()).items():
                public['input_sha256'][folder+'/'+name] = digest
        return public

    def snapshot_with_wrapper():
        sources = original_snapshot()
        sources[str(Path(__file__).resolve().relative_to(ROOT))] = sha(__file__)
        for name, expected in PINS.items():
            require(sha(ROOT/name) == expected, 'Recovery source snapshot mismatch: '+name)
            sources[name] = expected
        return sources

    shadow.base.write = normalized_write
    shadow.json = LocalJSONProxy(json_value)
    shadow.checked_inputs = inputs_with_failure_provenance
    shadow.source_snapshot = snapshot_with_wrapper
    shadow._serialization_recovery_installed = True
    return shadow


def self_test(out):
    """No checked_inputs(), mapper, world, packet loading, TSDF or Q calls."""
    require(not out.exists(), 'Fresh serialization preflight directory required')
    require(shutil.disk_usage(ROOT).free >= 64*1024**2+512*1024, '64 MiB reserve plus 512 KiB allowance required')
    original_dumps = standard_json.dumps
    shadow = install_output_recovery()
    import numpy as np
    out.mkdir(parents=True)
    checks = []
    try:
        payload = {'status':'synthetic_only', 'rows':[{'sector':np.int64(3),
            'count':np.int64(2**53+1), 'fraction':np.float32(.1), 'ok':np.bool_(True),
            'nested':(np.int64(-7), [np.float64(.25), None]), 'array':np.array([2,3], dtype=np.int64)}]}
        expected = {'status':'synthetic_only', 'rows':[{'sector':3,
            'count':2**53+1, 'fraction':float(np.float32(.1)), 'ok':True,
            'nested':[-7,[.25,None]], 'array':[2,3]}]}
        encoded = shadow.json.dumps(payload, sort_keys=True)
        require(standard_json.loads(encoded) == expected, 'NumPy scalar value/field changed')
        require(isinstance(standard_json.loads(encoded)['rows'][0]['count'], int), 'Large integer became floating')
        checks.append('nested NumPy scalar/list/array values and field names preserved; int64 remains exact')
        shadow.base.write(out, 'synthetic_result.json', payload)
        require(standard_json.loads((out/'synthetic_result.json').read_text()) == expected, 'File writer changed payload')
        failure = {'status':'synthetic_failed', 'error':'test only', 'completed_rows':payload['rows'],
                   'counts':{'offline_mapper_updates':np.int64(0)}}
        shadow.base.write(out, 'synthetic_failure.json', failure)
        require(standard_json.loads((out/'synthetic_failure.json').read_text()) ==
                standard_json.loads(shadow.json.dumps(failure)), 'Failure stdout/file serialization mismatch')
        checks.append('original guarded base writer and shadow stdout/failure proxy normalize identically')
        native = {'result':{'sector':3,'ratio':.125,'enabled':True},'failure':{'rows':[],'error':'x'}}
        require(shadow.json.dumps(native, sort_keys=True) == original_dumps(native, sort_keys=True, allow_nan=False),
                'Standard result/failure JSON changed')
        require(shadow.json.loads(encoded) == standard_json.loads(encoded), 'loads behavior changed')
        checks.append('standard nested result/failure serialization and original loads preserved')
        for value in (np.int64(3),):
            try:
                standard_json.dumps(value)
            except TypeError:
                pass
            else:
                raise AssertionError('Global json module was patched')
        require(standard_json.dumps is original_dumps and shadow.base.json is standard_json,
                'Global/base JSON binding changed')
        checks.append('stdlib json module unmodified; only shadow.json has local proxy')
        for bad in (float('nan'), np.float32('inf'), np.float64('-inf'), np.array([float('nan')])):
            for writer in ('proxy', 'file'):
                try:
                    if writer == 'proxy':shadow.json.dumps({'nested':[bad]}, allow_nan=True)
                    else:shadow.base.write(out,'must_not_exist.json',{'nested':[bad]})
                except ValueError:
                    pass
                else:
                    raise AssertionError('Non-finite JSON was accepted')
        require(not (out/'must_not_exist.json').exists(), 'Rejected nonfinite data reached disk')
        checks.append('nonfinite Python/NumPy scalar and array values rejected by both output paths')
        sources = shadow.source_snapshot()
        require(str(Path(__file__).relative_to(ROOT)) in sources, 'Wrapper omitted from frozen source inventory')
        require(all(value == 0 for value in shadow.COUNTERS.values()) and
                all(value == 0 for value in shadow.base.COUNTERS.values()), 'Non-serialization work occurred')
        verify_pins()
        checks.append('wrapper/source/failure provenance pinned; all processing counters remain zero')
        shadow.base.write(out,'result.json',dict(status='passed_serialization_preflight_only',checks=checks,
            source_sha256=sources,original_failed_evidence_sha256=FAILED_PINS,
            original_failure_receipt_inventories=FAILURE_INVENTORIES,
            original_failure_status='uncompleted; persisted manifest says running; no result receipt',
            original_shadow_bytes_unchanged=True,processing_counters=shadow.COUNTERS,
            recovery_replay_executed=False))
    except BaseException as error:
        shadow.base.write(out,'failure.json',dict(status='failed_serialization_preflight',error=repr(error),checks=checks))
        raise
    finally:
        inventory={p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file() and p.name!='artifact_hashes.json'}
        shadow.base.write(out,'artifact_hashes.json',inventory)
    print(shadow.json.dumps({'status':'passed_serialization_preflight_only','checks':len(checks),'output':str(out)}))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--self-test',action='store_true')
    mode.add_argument('--run',action='store_true')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.self_test:
        self_test((args.output or TEST_OUTPUT).resolve())
        return
    output=(args.output or OUTPUT).resolve()
    require(not output.is_relative_to(FAILED.resolve()), 'Original failed directory must remain intact')
    shadow=install_output_recovery()
    # Delegate CLI execution to the frozen original, preserving single CPU,
    # 600 s alarm, exactly 401 observations, budget and all existing assertions.
    sys.argv=[str(ROOT/'scripts/audit_v27_feedback_shadow.py'),'--run','--output',str(output)]
    shadow.main()
    verify_pins()


if __name__=='__main__':
    main()
