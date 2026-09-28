#!/usr/bin/env python3
"""Independent read-only sealed V24.1 prefix inventory and compact summary."""
import hashlib
import json
from pathlib import Path
import shutil

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
SOURCE = ROOT/'audit_results/facility_choice_v24_paid_prefix_20260915'
STATIC = ROOT/'audit_results/facility_choice_v24_1_high_backstops_20260915/result.json'
REPORT = ROOT/'docs/research/V24_PAIRED_PREFIX_RESULT_20260915.md'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text())


def main():
    if (OUT/'result.json').exists():
        raise ValueError('retain existing analysis')
    manifest = read(SOURCE/'manifest.json'); result = read(SOURCE/'result.json')
    assert manifest['status']==result['status']=='complete'
    inventory = read(SOURCE/'artifact_hashes.json')
    for name,expected in inventory.items():
        assert sha(SOURCE/name)==expected,name
    for name,expected in manifest['source_sha256'].items():
        assert sha(ROOT/name)==expected,name
    assert not list(SOURCE.glob('failure_*.json'))
    cases = []
    for case in result['cases']:
        folder = SOURCE/f"case_{case['index']:02d}"
        verification = read(folder/'verification.json')
        assert verification['status']=='passed'
        assert verification['physical_process_id'] != verification['replay_process_id']
        detailed = read(folder/'result.json')
        cases.append(dict(index=case['index'],parent=case['parent'],assignment=case['assignment'],
            paid_actions=case['paid_actions'],coverage=case['final_coverage_2d'],
            known_reachable_cells=detailed['trace'][-1]['known_reachable_cells'],
            reachable_cells=detailed['trace'][-1]['reachable_cells'],
            returned=case['returned_to_anchor'],collisions=case['collisions'],
            marker_tracks=[{k:row[k] for k in ('observed_track_index','frames','distinct_paid_poses','total_valid_pixels')}
                           for row in case['marker_tracks']],verification=verification))
    inputs = [SOURCE/name for name in ('manifest.json','result.json','artifact_hashes.json','sources.zip')]
    inputs += [STATIC,REPORT,Path(__file__).resolve()]
    summary = dict(status='passed_read_only_inventory_review',checked_artifacts=len(inventory),
        checked_frozen_sources=len(manifest['source_sha256']),physical_paid_actions=result['physical_paid_actions'],
        replay_paid_actions=result['replay_paid_actions'],raw_packets=result['saved_raw_packets'],
        raw_mib=result['raw_packet_bytes']/1024**2,free_mib_at_audit=shutil.disk_usage(ROOT).free/1024**2,
        prefix_sensor_feasibility_passed=result['prefix_sensor_feasibility_passed'],
        all_prefix_coverage_at_least_80=result['all_prefix_coverage_at_least_80'],
        parents=result['parents'],cases=cases,
        static_height_revision_gates_passed=read(STATIC)['static_revision_gates_passed'],
        input_sha256={str(path.relative_to(ROOT)):sha(path) for path in inputs},
        new_physical_actions=0,new_fusion_or_quality_evaluation=False,
        semantic_or_autonomous_efficacy_proven=False)
    (OUT/'result.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    shutil.copyfile(REPORT,OUT/REPORT.name)
    (OUT/'artifact_hashes.json').write_text(json.dumps({path.name:sha(path) for path in sorted(OUT.iterdir())
        if path.is_file() and path.name!='artifact_hashes.json'},indent=2)+'\n')
    print('verified',len(inventory),'artifacts;',len(manifest['source_sha256']),'sources;',OUT)


if __name__ == '__main__':
    main()
