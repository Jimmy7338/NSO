#!/usr/bin/env python3
"""Independently score the supplemental engineering run using the frozen evaluator."""

import json
from pathlib import Path
import sys

from revision_corridor_gc import ARM, FROZEN, HERE, LAYOUT, ROOT, digest, read

PREVIOUS = ROOT.parent / 'region_structure_floor_20261122'
sys.path.insert(0, str(PREVIOUS))
from evaluate import WEIGHTED_METRIC, import_file, measure_episode


def main():
    base = HERE / 'rev26_corridor_region_S_gc'
    raw_trial, destination = base / 'run', base / 'measurement.json'
    engineering = read(base / 'receipt.json')
    manifest = raw_trial / 'manifest.json'
    id_hash = digest(manifest)
    assert id_hash == engineering['native_manifest_sha256']
    frozen = read(FROZEN)
    job = next(j for j in frozen['jobs'] if (j['layout'],j['arm'])==(LAYOUT,ARM))
    observer = import_file('supplemental_corridor_reference_measurement', WEIGHTED_METRIC)
    scene_root = Path(frozen['scene_root'])
    observer.SCENES = scene_root
    model = import_file('supplemental_corridor_hidden_world_evaluation', scene_root/'scene.py')
    metric = observer.files.load_metric(read(observer.files.SCENES/'freeze.json'),read(observer.files.OLD_EVAL/'freeze.json'))
    worker = {'exit_code':0,'result_sha256':id_hash,'result_path':str(manifest)}
    summary, evidence, _ = measure_episode(observer,raw_trial,job,worker,frozen,digest(FROZEN),scene_root,model,metric)
    measurements = ('C_free','F1_scene_05cm','J','A95_paid','actual_paid','returned','collisions','repeat_distance_m')
    official = read(ROOT/'results/formal/summary.json')
    assert official['run_freeze_sha256']==digest(FROZEN)
    by_arm = {r['arm']:r for r in official['episodes'] if r['layout']==LAYOUT}
    with destination.open('x') as out:
        json.dump({
            'scope':'Supplemental engineering-only paired sensitivity. The original failure remains in the frozen formal table.',
            'formal_freeze_sha256':digest(FROZEN),
            'formal_summary_sha256':digest(ROOT/'results/formal/summary.json'),
            'regional_native_recovery_measurement':{k:summary[k] for k in measurements},
            'native_recovery_receipt_sha256':digest(base/'receipt.json'),
            'official_same_layout_endpoints':{arm:{k:values.get(k) for k in measurements}
                                              for arm,values in by_arm.items()},
            'checkpoints':[{k:row.get(k) for k in ('checkpoint','actual_paid','C_free','F1_scene_05cm','J')}
                           for row in evidence['checkpoints']],
        },out,indent=2,allow_nan=False)
        out.write('\n')


if __name__=='__main__':
    main()
