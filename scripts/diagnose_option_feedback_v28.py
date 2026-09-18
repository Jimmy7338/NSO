#!/usr/bin/env python3
"""Offline option-level Q labels for the sealed V27 feedback arrivals.

This script never runs the planner, simulator sensor, or world.step().  It maps
the saved packets once, extracts measured-only checkpoint meshes, and uses the
existing evaluator only after each checkpoint representation has been formed.
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import open3d as o3d
from scipy.stats import pearsonr, spearmanr

from env.facility_choice_v25_r1 import FacilityChoiceWorldV25
from nso.cpu_sensor_contract_v10 import json_value
from nso.decision_replay_v13 import array_hash, load_packet
from nso.facility_measurement_v26 import FacilityMeasurementV26, OUTLINE_CONFIG
from nso.observed_feedback_v27 import capture_feedback_support_v27, local_support_v27
from nso.observed_runtime_mapper_v10 import ObservedRuntimeMapperV10
from nso.observed_state_v26 import geometry_state_v26
from scripts.run_observed_autonomous_v26 import verify_inventory
from utils.facility_outline_v23 import OutlineEvaluatorV23


INPUT = ROOT/'audit_results/observed_autonomous_v27_20260916'
DEFAULT = ROOT/'audit_results/v28_option_feedback_diagnostic_r1_20260916'
PROTOCOL = ROOT/'docs/research/V28_OPTION_LEVEL_FEEDBACK_DIAGNOSTIC_PROTOCOL_20260916.md'
RECOVERY = ROOT/'docs/research/V28_OPTION_DIAGNOSTIC_RECOVERY_20260916.md'
FAILED_PREDECESSOR = ROOT/'audit_results/v28_option_feedback_diagnostic_20260916'
CASES = (1, 3)
EXPECTED_SAMPLES = {1: 13, 3: 11}
OUTPUT_CAP = 2*1024**2
FREE_RESERVE = 64*1024**2


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def read_gzip(path):
    with gzip.open(path, 'rt') as stream:
        return json.load(stream)


def write(root, name, value):
    payload = (json.dumps(json_value(value), ensure_ascii=False, sort_keys=True,
                          separators=(',', ':'), allow_nan=False)+'\n').encode()
    used = sum(p.stat().st_size for p in root.iterdir() if p.is_file())
    if used + len(payload) > OUTPUT_CAP:
        raise OSError('2 MiB diagnostic output cap would be exceeded')
    if shutil.disk_usage(root).free - len(payload) - 4096 < FREE_RESERVE:
        raise OSError('64 MiB real free-space reserve would be crossed')
    path = root/name
    temporary = path.with_name(path.name+'.tmp')
    with temporary.open('xb') as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def seal(root):
    write(root, 'artifact_hashes.json', {p.name: sha(p) for p in sorted(root.iterdir())
          if p.is_file() and p.name != 'artifact_hashes.json'})


def mesh_hashes(mesh):
    return {key: array_hash(np.asarray(getattr(mesh, key)))
            for key in ('vertices', 'triangles', 'vertex_colors')}


def joined(meshes):
    result = o3d.geometry.TriangleMesh()
    for mesh in meshes:
        result += mesh
    return result


def qrow(evaluation, reference_id=None):
    if reference_id is None:
        return dict(q05=evaluation['05cm']['outline_macro_quality'],
                    f1_05=evaluation['05cm']['outline_macro_f1'],
                    joint_05=evaluation['05cm']['joint_outline'])
    row = next(x for x in evaluation['instances'] if x['id'] == reference_id)
    return dict(q05=row['05cm']['outline_quality'], f1_05=row['05cm']['outline_f1'],
                missing=row['missing'], completed=row['completed'])


def evaluate_checkpoint(measurement, mapper, evaluator, slot, reference_id):
    raw = mapper.mesh()
    mesh = measurement.backends[slot].snapshot(raw_mesh=raw)['observed_mesh']
    args = dict(coverage=1., returned=True, collisions=0, failed=False)
    instance = evaluator.evaluate(mesh, **args)
    return dict(instance=qrow(instance, reference_id))


def evaluate_terminal(measurement, mapper, evaluator, result):
    raw = mapper.mesh()
    meshes = tuple(backend.snapshot(raw_mesh=raw)['observed_mesh']
                   for backend in measurement.backends)
    evaluation = evaluator.evaluate(joined(meshes), result['final_coverage_2d'],
        returned=result['returned_to_anchor'], collisions=result['collisions'],
        failed=not result['returned_to_anchor'] or bool(result['collisions']),
        paid_actions=result['paid_actions'], budget=400)
    return dict(union=qrow(evaluation), raw_hashes=mesh_hashes(raw),
                instance_hashes=[mesh_hashes(x) for x in meshes])


def frozen_samples(folder):
    result = read(folder/'result.json')
    plans = read_gzip(folder/result['plans_file'])
    calls = read_gzip(folder/result['calls_file'])
    option = {}
    serial = 0
    for plan in plans:
        if plan['selected'] is not None:
            serial += 1
            option[serial] = plan
    samples = []
    for call in calls:
        updates = call.get('directional_updates', ())
        if not updates:
            continue
        if not call.get('selected_endpoint_reached') or call.get('option_id') not in option:
            raise ValueError('directional update lacks a valid reached option')
        plan = option[call['option_id']]
        start = plan['audit']['action_id']
        evidence = {(x['cue_id'], x['sector']): x for x in plan['selected']['semantic_evidence']}
        audit = read_gzip(folder/'audit'/f'{start:04d}.json.gz')
        cues = {x['cue_id']: x for x in audit['cues']}
        for update in updates:
            key = update['cue_id'], update['sector']
            if update.get('status') != 'updated' or key not in evidence or key[0] not in cues:
                raise ValueError('directional update does not match its frozen semantic option')
            if start >= call['action_id']:
                raise ValueError('option feedback must follow its opening observation')
            samples.append(dict(case_index=result['index'], assignment=result['assignment'],
                option_id=call['option_id'], start_action=start, end_action=call['action_id'],
                duration_actions=call['action_id']-start, cue_id=key[0], sector=key[1],
                cue_center=cues[key[0]]['center'], class_id=cues[key[0]]['class_id'],
                endpoint_proxy=update['observed_yield_proxy'],
                endpoint_comparable_patches=update['comparable_patches'],
                endpoint_improved_common_patches=update['improved_common_patches']))
    return result, samples


def analyze_proxy(rows, name):
    x = np.asarray([row[name] for row in rows], float)
    y = np.asarray([row['delta_Q05'] for row in rows], float)
    varied = len(np.unique(x)) > 1 and len(np.unique(y)) > 1
    spearman = float(spearmanr(x, y).statistic) if varied else None
    pearson = float(pearsonr(x, y).statistic) if varied else None
    median = float(np.median(x))
    high = y[x > median]
    consistency = None if not len(high) else float(np.mean(high > 0.))
    return dict(proxy=name, samples=len(rows), mean=float(x.mean()), std=float(x.std()),
        unique_values=len(np.unique(x)), minimum=float(x.min()), maximum=float(x.max()),
        saturation_rate=float(np.mean(x >= 1.-1e-12)), spearman_delta_Q05=spearman,
        pearson_delta_Q05=pearson, median=median, strictly_above_median_samples=len(high),
        high_proxy_positive_Q_rate=consistency, correlation_identifiable=varied)


def validate_final(checkpoint, result):
    if checkpoint['raw_hashes'] != result['mesh_hashes']['raw']:
        raise ValueError('terminal raw mapper mesh differs from sealed V27 evidence')
    if checkpoint['instance_hashes'] != result['mesh_hashes']['observed']:
        raise ValueError('terminal measured instance meshes differ from sealed V27 evidence')
    expected = result['final_main_observed']['05cm']
    actual = checkpoint['union']
    pairs = ((actual['q05'], expected['outline_macro_quality']),
             (actual['f1_05'], expected['outline_macro_f1']),
             (actual['joint_05'], expected['joint_outline']))
    if any(not np.isclose(a, b, atol=1e-12, rtol=0) for a, b in pairs):
        raise ValueError('terminal outline score differs from sealed V27 evidence')


def process_case(index):
    folder = INPUT/f'case_{index:02d}'
    result, samples = frozen_samples(folder)
    if len(samples) != EXPECTED_SAMPLES[index]:
        raise ValueError(f'case {index} sample count changed: {len(samples)}')
    config = SimpleNamespace(**result['public_config'])
    mapper = ObservedRuntimeMapperV10(tuple(result['shape']), config)
    measurement = FacilityMeasurementV26()
    world = FacilityChoiceWorldV25(result['parent'], result['assignment'], 'iid_025px', 1901)
    evaluator = OutlineEvaluatorV23.from_world(world, **OUTLINE_CONFIG)
    association = result['evaluation_audit']['fixed_seed_association']
    if not association['seed_association_gate_passed']:
        raise ValueError('sealed final seed association did not pass')
    reference_by_slot = {row['observed_slot']: row['reference_id'] for row in association['rows']}
    evaluators = {}
    for reference_id in reference_by_slot.values():
        item = next(x for x in world.objects if x['id'] == reference_id)
        evaluators[reference_id] = OutlineEvaluatorV23([dict(id=reference_id,
            mesh=world.instance_mesh(reference_id), bounds=item['evaluation_bounds'])], **OUTLINE_CONFIG)
    boundaries = {x['start_action'] for x in samples} | {x['end_action'] for x in samples}
    supports, checkpoints = {}, {}
    trace = result['trace']
    anchor = tuple(trace[0]['pose'])
    for action_id in range(len(trace)):
        packet = load_packet(folder/'packets'/f'{action_id:04d}.npz')
        if packet.sha256() != trace[action_id]['packet_sha256']:
            raise ValueError('saved packet hash differs from sealed trace')
        mapper.update(packet.frame, packet.scan)
        measurement.observe(packet)
        if action_id in boundaries:
            state = geometry_state_v26(mapper, packet, anchor, 400-action_id)
            supports[action_id] = capture_feedback_support_v27(
                mapper, packet, anchor, 400-action_id, state)
            relevant = [x for x in samples if action_id in (x['start_action'], x['end_action'])]
            slots = {int(x['cue_id'].split('-')[1]) for x in relevant}
            if len(slots) != 1:
                raise ValueError('one checkpoint unexpectedly refers to multiple measured slots')
            slot = slots.pop()
            checkpoints[action_id] = evaluate_checkpoint(
                measurement, mapper, evaluators[reference_by_slot[slot]], slot, reference_by_slot[slot])
    terminal = evaluate_terminal(measurement, mapper, evaluator, result)
    validate_final(terminal, result)
    for row in samples:
        slot = int(row['cue_id'].split('-')[1])
        if slot not in reference_by_slot:
            raise ValueError('cue-to-observed-slot mapping is unavailable')
        before, after = checkpoints[row['start_action']], checkpoints[row['end_action']]
        counts = local_support_v27(supports[row['start_action']], supports[row['end_action']],
                                   row['cue_center'])
        row.update(reference_id=reference_by_slot[slot], option_comparable_patches=counts['comparable_patches'],
            option_improved_common_patches=counts['improved_common_patches'],
            option_direction_improved_patches=counts['direction_improved_patches'],
            option_range_improved_patches=counts['range_improved_patches'],
            option_new_keys=counts['new_keys'],
            option_proxy_16=min(1., counts['improved_common_patches']/16.),
            option_improved_fraction=(counts['improved_common_patches']/counts['comparable_patches']
                                      if counts['comparable_patches'] else 0.),
            start_Q05=before['instance']['q05'], end_Q05=after['instance']['q05'],
            delta_Q05=after['instance']['q05']-before['instance']['q05'],
            start_F1_05=before['instance']['f1_05'], end_F1_05=after['instance']['f1_05'],
            delta_F1_05=after['instance']['f1_05']-before['instance']['f1_05'])
    return result, samples, dict(saved_packets_mapped_once=len(trace), mapper_updates=len(trace),
        measurement_observes=len(trace), planner_calls=0, simulator_sensor_calls=0,
        simulator_steps=0, reference_worlds_constructed=1,
        unique_checkpoint_mesh_extractions=len(boundaries)+1, terminal_match_passed=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=DEFAULT)
    args = parser.parse_args()
    verify_inventory(INPUT)
    if args.output.exists():
        raise FileExistsError(args.output)
    if shutil.disk_usage(ROOT).free < FREE_RESERVE + OUTPUT_CAP:
        raise OSError('diagnostic requires 2 MiB plus 64 MiB reserve')
    args.output.mkdir(parents=True)
    manifest = dict(status='running_readonly_saved_packet_diagnostic', protocol=str(PROTOCOL.relative_to(ROOT)),
        protocol_sha256=sha(PROTOCOL), recovery=str(RECOVERY.relative_to(ROOT)),
        recovery_sha256=sha(RECOVERY), script=str(Path(__file__).resolve().relative_to(ROOT)),
        script_sha256=sha(__file__), input=str(INPUT.relative_to(ROOT)),
        input_inventory_sha256=sha(INPUT/'artifact_hashes.json'), cases=list(CASES),
        failed_predecessor=str(FAILED_PREDECESSOR.relative_to(ROOT)),
        failed_predecessor_inventory_sha256=sha(FAILED_PREDECESSOR/'artifact_hashes.json'),
        expected_samples=EXPECTED_SAMPLES, new_main_attempts=0, autonomous_actions=0,
        output_cap_bytes=OUTPUT_CAP, free_reserve_bytes=FREE_RESERVE)
    write(args.output, 'manifest.json', manifest)
    try:
        rows, costs, case_summaries = [], [], []
        for index in CASES:
            result, samples, cost = process_case(index)
            rows.extend(samples); costs.append(dict(case_index=index, **cost))
            case_summaries.append(dict(case_index=index, assignment=result['assignment'], samples=len(samples),
                paid_packets=len(result['trace']), terminal_Q05=result['final_main_observed']['05cm']['outline_macro_quality']))
        if len(rows) != sum(EXPECTED_SAMPLES.values()):
            raise ValueError('frozen total sample count changed')
        proxies = [analyze_proxy(rows, name) for name in
                   ('endpoint_proxy', 'option_proxy_16', 'option_improved_fraction')]
        endpoint = proxies[0]
        gate = dict(minimum_samples=endpoint['samples'] >= 20,
            minimum_unique_values=endpoint['unique_values'] >= 4,
            saturation_below_half=endpoint['saturation_rate'] < .5,
            spearman_at_least_0p30=(endpoint['spearman_delta_Q05'] is not None
                                    and endpoint['spearman_delta_Q05'] >= .30),
            high_proxy_positive_rate_at_least_0p60=(endpoint['high_proxy_positive_Q_rate'] is not None
                                    and endpoint['high_proxy_positive_Q_rate'] >= .60))
        coverage = dict(by_case=dict(Counter(str(x['case_index']) for x in rows)),
            by_class=dict(Counter(str(x['class_id']) for x in rows)),
            by_cue=dict(Counter(f"{x['case_index']}:{x['cue_id']}" for x in rows)),
            by_class_sector=dict(Counter(f"{x['class_id']}:{x['sector']}" for x in rows)))
        class3_verifiable = coverage['by_class'].get('3', 0) >= 6
        option_fraction = proxies[2]
        result = dict(status='complete_offline_option_feedback_diagnostic', samples=rows,
            proxy_statistics=proxies, endpoint_minimum_calibration_gate=gate,
            endpoint_minimum_calibration_evidence_passed=all(gate.values()),
            sample_coverage=coverage, complex_class_minimum_samples=6,
            complex_class_feedback_verifiable=class3_verifiable,
            option_fraction_positive_association=(option_fraction['spearman_delta_Q05'] is not None
                and option_fraction['spearman_delta_Q05'] > 0),
            case_summaries=case_summaries, execution_cost=costs,
            total_saved_packets_mapped=sum(x['saved_packets_mapped_once'] for x in costs),
            new_main_attempts=0, autonomous_actions=0, simulator_sensor_calls=0,
            evaluation_truth_used_offline_only=True, planner_received_evaluation_truth=False,
            efficacy_claimed=False)
        write(args.output, 'result.json', result)
        manifest['status'] = 'complete'
        manifest['actual_samples'] = len(rows)
        write(args.output, 'manifest.json', manifest)
        seal(args.output)
        print(json.dumps({k: result[k] for k in ('status','endpoint_minimum_calibration_evidence_passed',
              'complex_class_feedback_verifiable','option_fraction_positive_association')}, ensure_ascii=False))
    except Exception as error:
        write(args.output, 'failure.json', dict(status='failed', error=repr(error),
              new_main_attempts=0, autonomous_actions=0))
        seal(args.output)
        raise


if __name__ == '__main__':
    main()
