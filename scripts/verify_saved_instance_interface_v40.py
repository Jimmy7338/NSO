#!/usr/bin/env python3
"""Check the new instance interface on sealed V39 paid RGB-D, without a World.

Each saved route is independently fed to S and G ledgers. This is an interface
compatibility check, not a newly executed planner or multi-facility experiment.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from nso.instance_belief_v40 import InstanceBeliefV40, PaidRGBDObservationV40


SOURCE = ROOT / 'audit_results/v39_external_cpu_20260920'
CASES = ('case00', 'case01', 'case04', 'case05')
PACKET_FIELDS = ('frame__color_rgb', 'frame__depth_m', 'frame__intrinsic',
                 'frame__world_from_camera')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def verify(output):
    output = Path(output)
    if output.exists():
        raise FileExistsError('Preserve earlier checks; choose a new output path.')
    if shutil.disk_usage(ROOT).free < 80 * 1024**2:
        raise RuntimeError('Insufficient persistent space for this saved-data check.')
    output.mkdir(parents=True)
    inventory = json.loads((SOURCE / 'final_seal.json').read_text())
    inputs = {}
    rows = []
    configuration = dict(palette={2: [40, 100, 220], 3: [220, 60, 40]},
                         structure_names=['h0', 'h1'],
                         class_structure_prior={2: [.9, .1], 3: [.1, .9]})
    for case in CASES:
        ledgers = {mode: InstanceBeliefV40(mode=mode, **configuration) for mode in ('S', 'G')}
        paths = sorted((SOURCE / case / 'packets').glob('*.npz'))
        if [p.name for p in paths] != [f'{i:03d}.npz' for i in range(43)]:
            raise ValueError('Expected complete original 0..42 paid packet sequence.')
        events = []
        for step, path in enumerate(paths):
            relative = str(path.relative_to(SOURCE))
            digest = sha(path)
            if inventory.get(relative) != digest:
                raise ValueError('Saved packet differs from original final seal: ' + relative)
            inputs[str(path.relative_to(ROOT))] = digest
            with np.load(path, allow_pickle=False) as packet:
                # Do not decode metadata, simulation owner/semantic fields, or Q.
                selected = {key: np.array(packet[key], copy=True) for key in PACKET_FIELDS}
            observation = PaidRGBDObservationV40.from_mapping(dict(
                frame_id=f'paid_{step:03d}', paid_step=step,
                rgb=selected['frame__color_rgb'], depth_m=selected['frame__depth_m'],
                intrinsic=selected['frame__intrinsic'],
                world_from_camera=selected['frame__world_from_camera']))
            receipts = {mode: ledger.observe(observation) for mode, ledger in ledgers.items()}
            geometry_equal = ledgers['S'].geometry_snapshot() == ledgers['G'].geometry_snapshot()
            if not geometry_equal:
                raise AssertionError('S/G observation/association geometry differs.')
            if any(i['semantic_conditioning_used'] for i in ledgers['G'].snapshot()['instances']):
                raise AssertionError('Geometry control conditioned on class.')
            events.append(dict(step=step, observation_sha256=observation.sha256(),
                               geometry_equal=geometry_equal, receipts=receipts))
        final = {mode: ledger.snapshot() for mode, ledger in ledgers.items()}
        write(output / f'{case}.json', dict(source_case=case, events=events, final=final))
        rows.append(dict(case=case, paid_packets=len(paths), s_g_geometry_equal_all_steps=True,
                         discovered_marker_instances=len(final['S']['instances']),
                         accepted_components=sum(len(e['receipts']['S']['accepted']) for e in events),
                         rejected_components=sum(len(e['receipts']['S']['rejected']) for e in events),
                         semantic_active_steps=sum(any(i['semantic_conditioning_used'] for i in
                             e['receipts']['S']['instances']) for e in events),
                         final_semantic_snapshot=final['S']))
    for path in (Path(__file__), ROOT / 'nso/instance_belief_v40.py', SOURCE / 'final_seal.json'):
        inputs[str(path.relative_to(ROOT))] = sha(path)
    # Preserve the exact mutable new implementation used in this compatibility check.
    (output / 'sources').mkdir()
    for path in (Path(__file__), ROOT / 'nso/instance_belief_v40.py'):
        shutil.copy2(path, output / 'sources' / path.name)
    result = dict(status='passed', scope='saved single-facility RGB-D interface compatibility only',
                  rows=rows, source_cases=len(CASES), unique_saved_packet_reads=sum(r['paid_packets'] for r in rows),
                  ledger_observe_calls=2 * sum(r['paid_packets'] for r in rows),
                  loaded_packet_fields=list(PACKET_FIELDS), metadata_decoded=False,
                  new_worlds=0, new_physical_trajectories=0, planner_calls=0,
                  mapper_calls=0, metric_calls=0, applied_geometry_feedback_calls=0,
                  multi_facility_task_effectiveness_proven=False,
                  natural_semantic_detection_proven=False, configuration=configuration)
    write(output / 'input_sha256.json', inputs)
    write(output / 'result.json', result)
    write(output / 'artifact_sha256.json', {str(p.relative_to(output)): sha(p)
          for p in sorted(output.rglob('*')) if p.is_file()})
    print(json.dumps({k: v for k, v in result.items() if k not in ('rows', 'configuration')},
                     ensure_ascii=False, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'audit_results/v40_saved_instance_compatibility_20260920')
    verify(parser.parse_args().output)
