#!/usr/bin/env python3
"""Repeat easterly screened S with the previously tested cache-lifetime intervention."""

import faulthandler
import gc
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OLD = HERE.parents[2] / 'region_structure_floor_20261122' / 'local'
sys.path.insert(0, str(OLD))

from run import FREEZE, execute, native, read, sha, write


def main():
    directory = HERE / 'east_screened_S_gc'
    directory.mkdir(exist_ok=False)
    output = directory / 'native'
    output.mkdir()
    original_setup = native.setup_runtime
    collections = []
    with (directory / 'faulthandler.log').open('x') as fault_log:
        faulthandler.enable(file=fault_log)

        def setup(*args):
            context = original_setup(*args)
            import runtime_binding
            inherited_method = runtime_binding.old.method
            ordinary_cache = inherited_method._cache

            def cache(mapper, arm):
                saved = mapper._dual_template_state['cache']
                if saved is None or saved['key'] != (int(mapper.frames), arm):
                    collections.append({
                        'mapper_frames': int(mapper.frames),
                        'unreachable_collected': gc.collect(),
                    })
                    write(directory / 'collections.json', collections)
                return ordinary_cache(mapper, arm)

            inherited_method._cache = cache
            return context

        native.setup_runtime = setup
        execute('east_screened', 'S', output)
        faulthandler.disable()

    reference_g = OLD / 'evidence_checks' / 'gc_lifetime_diagnosis' / 'native'
    partial = OLD / 'execution_attempt_002' / 'east_screened_S' / 'partial_native'
    old_checkpoints = ('start', 'fork', 'paid_48', 'paid_96', 'paid_144')
    checkpoint_matches = {}
    for name in old_checkpoints:
        checkpoint_matches[name] = {
            'checkpoint_bytes': sha(partial / name / 'checkpoint.json') == sha(output / name / 'checkpoint.json'),
            'belief_bytes': sha(partial / name / 'belief.npz') == sha(output / name / 'belief.npz'),
            'native_mesh_bytes': sha(partial / name / 'native_mesh.ply') == sha(output / name / 'native_mesh.ply'),
        }
    keys = ('action', 'position', 'heading', 'packet_sha256', 'packet_chain_sha256',
            'depth_values_sha256', 'rgb_values_sha256', 'scan_ranges_sha256')
    new_steps = read(output / 'steps.json')
    prior_steps = read(reference_g / 'steps.json')
    same = [all(a[k] == b[k] for k in keys) for a, b in zip(prior_steps, new_steps)]
    episode = read(output / 'manifest.json')
    import open3d
    write(directory / 'receipt.json', {
        'scope': 'simulator diagnostic. Original freeze and two original missing endpoints remain unchanged.',
        'only_intervention': 'gc.collect before invalidated template cache rebuild; original cache invoked with same mapper and arm.',
        'source_bound_by_original_freeze_sha256': sha(FREEZE),
        'open3d_version': open3d.__version__,
        'previous_G_gc_manifest_sha256': sha(reference_g / 'manifest.json'),
        'old_S_retry_after_sigsegv_prefix_matches': checkpoint_matches,
        'S_original_saved_prefix_matches': all(all(c.values()) for c in checkpoint_matches.values()),
        'G_and_S_motion_and_sensor_ledger_matches_count': sum(same),
        'G_and_S_first_difference_paid': next((i for i, equal in enumerate(same, 1) if not equal), None),
        'G_and_S_terminal_belief_bytes_match': sha(reference_g / 'terminal' / 'belief.npz') == sha(output / 'terminal' / 'belief.npz'),
        'G_and_S_terminal_mesh_bytes_match': sha(reference_g / 'terminal' / 'native_mesh.ply') == sha(output / 'terminal' / 'native_mesh.ply'),
        'paid_actions': episode['paid_actions'],
        'returned': episode['returned'],
        'collisions': episode['collisions'],
        'manifest_sha256': sha(output / 'manifest.json'),
    })


if __name__ == '__main__':
    main()
