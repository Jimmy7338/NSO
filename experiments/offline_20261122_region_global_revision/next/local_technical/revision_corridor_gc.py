#!/usr/bin/env python3
"""Run the old corridor arm with only a verified cache-lifetime intervention."""

import faulthandler
import gc
from hashlib import sha256
import json
from pathlib import Path
import runpy
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FROZEN = ROOT / 'FORMAL_FREEZE.json'
ACTOR = ROOT / 'runner.py'
ARCHIVE = ROOT / 'execution' / 'formal' / 'rev26_corridor_receiving' / 'region_S'
LAYOUT = 'rev26_corridor_receiving'
ARM = 'region_S'


def digest(path):
    with Path(path).open('rb') as stream:
        return sha256(stream.read()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def main():
    spec = read(FROZEN)
    assert any(job['layout']==LAYOUT and job['arm']==ARM for job in spec['jobs'])
    for name, expected in spec['actor_files'].items():
        assert digest(Path(name)) == expected
    output = HERE / 'rev26_corridor_region_S_gc'
    output.mkdir(exist_ok=False)
    native_output = output / 'run'

    sys.path.insert(0, str(ROOT.parent / 'paired_unseen_gain_20261120' / 'execution'))
    import runtime_binding

    original_install = runtime_binding.install
    collections = []

    def install():
        old = original_install()
        method = old.method
        original_cache = method._cache

        def cache(mapper, arm):
            previous = mapper._dual_template_state['cache']
            if previous is None or previous['key']!=(int(mapper.frames),arm):
                collections.append({'frames':int(mapper.frames),'unreachable_collected':gc.collect()})
            return original_cache(mapper, arm)

        method._cache = cache
        return old

    recipe = [str(ACTOR), LAYOUT, ARM, '--freeze', str(FROZEN), '--out', str(native_output)]
    with (output/'faulthandler.log').open('x') as fault_log:
        faulthandler.enable(file=fault_log)
        runtime_binding.install = install
        sys.argv = recipe
        runpy.run_path(str(ACTOR),run_name='__main__')
        faulthandler.disable()

    comparisons = {}
    for old in sorted(ARCHIVE.glob('attempt_*/run')):
        comparable = {}
        for name in ('start','paid_48','paid_96','paid_144','paid_180'):
            captured = old/name
            if not (captured/'checkpoint.json').is_file():
                comparable[name] = {'status':'original_did_not_seal_checkpoint'}
                continue
            assert (native_output/name/'checkpoint.json').is_file()
            fields = ('checkpoint.json','belief.npz','native_mesh.ply')
            comparable[name] = {field:digest(captured/field)==digest(native_output/name/field) for field in fields}
        comparisons[old.parent.name] = comparable
    manifest = read(native_output/'manifest.json')
    assert manifest['runner_freeze_sha256'] == digest(FROZEN)
    with (output/'receipt.json').open('x') as stream:
        json.dump({
            'scope':'Supplemental simulator-only runtime diagnostic. Original failures retained.',
            'intervention':'gc.collect before stale cache rebuild, using the same original cache and native actor',
            'exact_actor_configuration_sha256':digest(FROZEN),
            'exact_actor_source_sha256':digest(ACTOR),
            'command':recipe,
            'matching_preserved_checkpoints_by_attempt':comparisons,
            'collections':collections,
            'returned':manifest['returned'],
            'collisions':manifest['collisions'],
            'paid_actions':manifest['paid_actions'],
            'native_manifest_sha256':digest(native_output/'manifest.json'),
        },stream,indent=2,allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()
