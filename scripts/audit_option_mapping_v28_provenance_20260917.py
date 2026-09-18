#!/usr/bin/env python3
"""Read-only V28 cue/measurement provenance; no mapper, mesh, or world calls.

Only the two frozen marker trackers are replayed over 802 saved packets.
This is identity auditing, not TSDF replay, policy execution, or Q evaluation.
"""
import os
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import argparse
from dataclasses import asdict
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import sys
from time import perf_counter
import traceback
import zipfile

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from nso.cpu_sensor_contract_v10 import json_value
from nso.decision_replay_v13 import load_packet
from nso.observed_state_v26 import VisibleSemanticMemoryV26
from scripts.probe_facility_choice_v24_prefix import MarkerTracks
from scripts.replay_facility_choice_shape_v24 import component_seed_hits

INPUT = ROOT/'audit_results/observed_autonomous_v27_20260916'
DIAGNOSTIC = ROOT/'audit_results/v28_option_feedback_diagnostic_r1_20260916'
FAILED = ROOT/'audit_results/v28_option_feedback_diagnostic_20260916'
DEFAULT = ROOT/'audit_results/v28_provenance_20260917'
TRACK_CONFIG = dict(marker_minimum_valid_pixels_per_component=16,
    marker_track_association_distance_m=.75, minimum_supported_distinct_poses_per_track=2)
CAP = 100*1024
RESERVE = 64*1024**2


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def canonical(value):
    return json.dumps(json_value(value), sort_keys=True, separators=(',', ':'), allow_nan=False)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def safe_write(root, name, value):
    data = value.encode() if isinstance(value, str) else (canonical(value)+'\n').encode()
    require(sum(p.stat().st_size for p in root.iterdir() if p.is_file())+len(data) < CAP,
            'audit 100 KiB cap')
    require(shutil.disk_usage(root).free-len(data)-4096 >= RESERVE, '64 MiB reserve')
    with (root/name).open('xb') as stream:
        stream.write(data)


def inventory(root):
    expected = read(root/'artifact_hashes.json')
    actual = {str(p.relative_to(root)) for p in root.rglob('*')
              if p.is_file() and p != root/'artifact_hashes.json'}
    require(actual == set(expected), 'inventory file set: '+str(root))
    for name, digest in expected.items():
        require(sha(root/name) == digest, 'inventory hash: '+name)
    return dict(entries=len(expected), inventory_sha256=sha(root/'artifact_hashes.json'))


def frozen_sources():
    manifest = read(INPUT/'manifest.json')
    for key in ('source_sha256', 'input_sha256'):
        for name, expected in manifest[key].items():
            require(sha(ROOT/name) == expected, 'frozen '+key+': '+name)
    require(sha(INPUT/'sources.zip') == manifest['source_archive_sha256'], 'source archive hash')
    with zipfile.ZipFile(INPUT/'sources.zip') as archive:
        require(set(archive.namelist()) == set(manifest['source_sha256']), 'source archive names')
        for name, expected in manifest['source_sha256'].items():
            require(hashlib.sha256(archive.read(name)).hexdigest() == expected, 'archive source: '+name)
    recovered = read(DIAGNOSTIC/'manifest.json')
    for key in ('script', 'protocol', 'recovery'):
        require(sha(ROOT/recovered[key]) == recovered[key+'_sha256'], 'V28 recovered '+key)
    require(recovered['input_inventory_sha256'] == sha(INPUT/'artifact_hashes.json'), 'V28 input link')
    require(recovered['failed_predecessor_inventory_sha256'] == sha(FAILED/'artifact_hashes.json'),
            'V28 predecessor link')
    return dict(source_count=len(manifest['source_sha256']), input_count=len(manifest['input_sha256']),
        archived_sources=len(manifest['source_sha256']), source_sha256=manifest['source_sha256'],
        input_sha256=manifest['input_sha256'], source_archive_sha256=manifest['source_archive_sha256'],
        recovered_v28_sources={recovered[key]: recovered[key+'_sha256'] for key in ('script','protocol','recovery')})


def case_audit(index, samples):
    folder = INPUT/f'case_{index:02d}'
    result = read(folder/'result.json')
    snapshot = read(folder/'measured_snapshot.json')
    seeds = {row['observed_slot']: row['seed'] for row in snapshot['instances']}
    associations = result['evaluation_audit']['fixed_seed_association']
    require(associations['seed_association_gate_passed'], 'stored seed association failed')
    references = {row['observed_slot']: row['reference_id'] for row in associations['rows']}
    memory, marker = VisibleSemanticMemoryV26(), MarkerTracks(dict(TRACK_CONFIG))
    mapping, first, histories = {}, [], {}
    observation_events, maximum_center_offset = 0, 0.
    min_foreign_center_distance = float('inf')
    event_digest = hashlib.sha256()
    for action, trace in enumerate(result['trace']):
        require(trace['action_id'] == action, 'noncontiguous source trace')
        packet = load_packet(folder/'packets'/f'{action:04d}.npz')
        require(packet.action_id == action and packet.sha256() == trace['packet_sha256'], 'packet SHA/action')
        with gzip.open(folder/'audit'/f'{action:04d}.json.gz', 'rt') as stream:
            audit = json.load(stream)
        cues = [asdict(cue) for cue in memory.update(packet)]
        require(canonical(cues) == canonical(audit['cues']), 'saved cue history differs at '+str(action))
        histories[action] = {cue['cue_id']: cue for cue in cues}
        rows = marker.consume(packet)
        hits = component_seed_hits(packet.frame, 16)
        require(len(rows) == len(hits), 'marker component order')
        for row, hit in zip(rows, hits):
            slot = row['observed_track_index']
            current = [cue for cue in cues if json.loads(cue['source'])['last_action'] == action
                       and json.loads(cue['source'])['last_pixel'] == hit['pixel']]
            require(len(current) == 1, 'component does not uniquely identify an observed cue')
            cue = current[0]; cue_id = cue['cue_id']; source = json.loads(cue['source'])
            require(int(cue_id.split('-')[1]) == slot, 'numeric cue/slot shortcut invalid for this history')
            require(mapping.setdefault(cue_id, slot) == slot, 'cue/slot association changed')
            require(source['last_valid_pixels'] == hit['component_pixels'] == row['valid_rgb_marker_depth_pixels'],
                    'component support differs')
            if seeds[slot]['action_id'] == action:
                saved = seeds[slot]
                require(saved['frame_id'] == packet.frame_id and saved['packet_sha256'] == packet.sha256(), 'seed packet')
                require(saved['pixel'] == hit['pixel'] == source['first_pixel'] and source['first_action'] == action,
                        'first observed identity differs')
                for name in hit:
                    require(canonical(saved[name]) == canonical(hit[name]), 'seed receipt differs: '+name)
                xyz_error = float(np.linalg.norm(np.asarray(cue['center'])-saved['observed_seed_xyz']))
                require(xyz_error <= 1e-12, 'cue first point differs from measured seed')
                require(cue['class_id'] == saved['actual_marker_code'], 'observed first color differs')
                first.append(dict(cue_id=cue_id, slot=slot, reference_id=references[slot],
                    first_action=action, pixel=hit['pixel'], observed_seed_xyz=hit['observed_seed_xyz'],
                    cue_seed_distance_m=xyz_error, observed_class_id=cue['class_id'],
                    seed_packet_sha256=packet.sha256(), valid_pixels=hit['component_pixels']))
            event_digest.update(canonical([action,cue_id,slot,hit]).encode())
            observation_events += 1
        for cue in cues:
            slot = int(cue['cue_id'].split('-')[1])
            center = np.asarray(cue['center'])
            maximum_center_offset=max(maximum_center_offset,float(np.linalg.norm(center-marker.tracks[slot]['centre'])))
            for other, track in enumerate(marker.tracks):
                if other != slot:
                    min_foreign_center_distance=min(min_foreign_center_distance,float(np.linalg.norm(center-track['centre'])))
    require(canonical(marker.summary()) == canonical(snapshot['observed_track_summary']), 'final measurement tracks')
    require(len(first) == len(mapping) == len(marker.tracks) == 2, 'exactly two persistent observed identities required')
    require(min_foreign_center_distance > .75 and maximum_center_offset < .75, 'actual cross-tracker identities ambiguous')
    selected=[]
    for row in samples:
        cue=histories[row['start_action']][row['cue_id']];slot=mapping[row['cue_id']]
        require(canonical(cue['center']) == canonical(row['cue_center']), 'sample cue center')
        require(cue['class_id'] == row['class_id'] and references[slot] == row['reference_id'], 'sample class/reference')
        selected.append(dict(option_id=row['option_id'],cue_id=row['cue_id'],observed_slot=slot,
            reference_id=references[slot],start_action=row['start_action'],end_action=row['end_action']))
    return dict(case_index=index, saved_packets_and_cue_audits_checked=len(result['trace']),
        binary_marker_tracker_updates=len(result['trace']), semantic_memory_updates=len(result['trace']),
        all_saved_cues_exactly_reproduced=True, final_binary_marker_summary_exactly_reproduced=True,
        observed_component_identity_events=observation_events, observed_component_event_sha256=event_digest.hexdigest(),
        cue_to_slot=mapping, first_seed_correspondences=first,
        maximum_cue_vs_same_slot_running_center_offset_m=maximum_center_offset,
        minimum_cue_to_other_slot_running_center_distance_m=min_foreign_center_distance,
        all_v28_sample_identities_checked=len(selected), sample_mappings=selected,
        gt_references_from_sealed_receipts_only=True,
        source_files_sha256={name:sha(folder/name) for name in ('result.json','measured_snapshot.json','artifact_hashes.json')})


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=DEFAULT)
    args=parser.parse_args()
    require(not args.output.exists(), 'refuse existing audit directory')
    require(shutil.disk_usage(ROOT).free >= CAP+RESERVE, 'capacity before audit')
    args.output.mkdir(parents=True)
    started=perf_counter()
    safe_write(args.output,'manifest.json',dict(status='started_read_only_identity_audit',
        script=str(Path(__file__).resolve().relative_to(ROOT)),script_sha256=sha(__file__),
        input=str(INPUT.relative_to(ROOT)), saved_cases=[1,3], physical_actions=0,tsdf_updates=0,
        mesh_extractions=0,quality_evaluations=0,output_cap_bytes=CAP))
    try:
        inventories={str(p.relative_to(ROOT)):inventory(p) for p in (INPUT,DIAGNOSTIC,FAILED)}
        frozen=frozen_sources()
        diagnostic=read(DIAGNOSTIC/'result.json')
        cases=[case_audit(index,[row for row in diagnostic['samples'] if row['case_index']==index]) for index in (1,3)]
        require([case['all_v28_sample_identities_checked'] for case in cases]==[13,11], 'sample counts')
        cost=diagnostic['execution_cost']
        require([row['saved_packets_mapped_once'] for row in cost]==[401,401], 'recovery count')
        require(diagnostic['total_saved_packets_mapped']==802 and all(row['mapper_updates']==row['measurement_observes']==401 for row in cost), 'recovery calls')
        old=read(FAILED/'manifest.json');failure=read(FAILED/'failure.json')
        require(failure['error']=="ValueError('terminal outline score differs from sealed V27 evidence')", 'predecessor failure changed')
        result=dict(status='passed_case_specific_provenance_not_general_numbering_contract',cases=cases,
            frozen_sources_verified=frozen,inventories_verified=inventories,
            historical_diagnostic_cost=dict(recovered_saved_packets_mapped=802,recovered_cases=[401,401],
                first_failed_attempt_inferred_saved_packets_mapped=401, combined_mapping_count_including_failed_attempt=1203,
                first_failed_count_basis='Failure at first-case terminal validation after full history; ordered cases [1,3] and recovery explanation. Failure receipt itself lacks case/packet counter; original failed source bytes not present in its three-file evidence directory. Count is a control-flow/protocol inference, not a direct logged counter.',
                first_failed_original_script_sha256=old['script_sha256'],
                original_and_recovered_scripts_are_different=True,
                original_manifest_status_stayed_running_but_failure_receipt_is_present=True),
            this_audit_cost=dict(saved_packets_read=802,semantic_memory_updates=802,binary_marker_tracker_updates=802,
                tsdf_updates=0,mesh_extractions=0,quality_evaluations=0,world_constructions=0,
                planner_calls=0,physical_actions=0,elapsed_s=perf_counter()-started),
            limitations=[
                'Cue suffix equals measurement slot is verified only for these two complete saved histories; the implementations do not define a shared ID contract.',
                'Semantic memory uses nearest real pixel to coordinate median, global one-to-one matching, and range upper bound; binary tracks use coordinate median, sequential nearest matching without one-to-one exclusion, and no explicit range upper bound.',
                'This audit verifies frozen source and saved identity provenance; it does not recompute TSDF or independently reproduce checkpoint quality labels.',
                'Reference IDs are copied from already sealed evaluation-only seed receipts; no world or reference geometry was read into tracking.',
                'Both semantic and measurement tracks rely on artificial RGB marker colors; this is not natural-image semantic recognition evidence.'])
        safe_write(args.output,'result.json',result)
        safe_write(args.output,'artifact_hashes.json',{p.name:sha(p) for p in sorted(args.output.iterdir()) if p.is_file()})
        print(canonical(dict(status=result['status'],cases=[{k:c[k] for k in ('case_index','cue_to_slot','observed_component_identity_events','maximum_cue_vs_same_slot_running_center_offset_m','minimum_cue_to_other_slot_running_center_distance_m')} for c in cases],cost=result['this_audit_cost'])))
    except BaseException as error:
        safe_write(args.output,'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),
            world_constructions=0,tsdf_updates=0,physical_actions=0))
        raise


if __name__=='__main__':
    main()
