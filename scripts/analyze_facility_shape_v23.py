#!/usr/bin/env python3
"""Read-only verification and plots for a complete V23 two-shape sensor bench.

No simulator, mapper, evaluator, policy or training is executed. Formal analysis
requires the complete batch, exact inventories and two fresh-process replays.
The negative observation-demand conditions are retained without changing gates.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import zipfile

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT/'audit_results/facility_shape_v23_probe_20260915'
DEFAULT_OUTPUT = ROOT/'audit_results/facility_shape_v23_analysis_20260915'
STAGES = dict(initial=0, front=3, coarse=23, rear=48, extra=73, returned=96)
TAGS = ('02cm', '05cm', '10cm')
PLANES = ('xy', 'xz', 'yz')
PACKET_FIELDS = ('scene_id', 'episode_id', 'frame_id', 'action_id', 'frame', 'scan',
    'position', 'heading', 'sensor_source', 'pose_source', 'action', 'collision', 'done')
FRAME_FIELDS = ('timestamp_s', 'depth_m', 'color_rgb', 'intrinsic', 'world_from_camera', 'semantic')
SCAN_FIELDS = ('timestamp_s', 'ranges_m', 'angle_min_rad', 'angle_increment_rad', 'range_max_m', 'world_from_laser')
NONSEMANTIC = {name: 'frame__'+name for name in ('depth_m', 'intrinsic', 'world_from_camera')}
NONSEMANTIC.update({name: 'scan__'+name for name in ('ranges_m', 'world_from_laser',
    'angle_min_rad', 'angle_increment_rad', 'range_max_m', 'timestamp_s')})
NOTES = [
    'Two shape variants, one layout, one noise seed and one shared paid scripted route. Two replays are equality checks, not additional experiments.',
    'Primary Q is the mean across three fixed orthographic projections of min(boundary-length F1@5cm, projected-envelope IoU), averaged equally across declared facilities. This bench has one facility per shape.',
    'J is true evaluation C2D times primary Q. Boundary F1 is a separate diagnostic and is not substituted for Q.',
    'Three projection envelopes do not identify complete three-dimensional geometry. Raw TSDF is scored without geometric completion; original external-surface F1, precision/recall and continuous errors remain beside the outline score.',
    'The fixed demand conditions are simple completion at action 23 and a positive complex Q increment from action 23 to 73 that exceeds the simple increment over the same paid interval. Failure of either condition is retained.',
    'Each condition uses the frozen tolerance/threshold and the predeclared stages, not the best available stage or a different ending after observing results.',
    'The observed response interval contains 50 paid movements/turns. It is not a per-view causal attribution or evidence that every additional observation helps.',
    'Pairing is checked against this batch’s saved sensors. Initial identity does not establish identity through coarse survey; an earlier reveal prevents claiming that a later decision had only class information.',
    'No autonomous coverage policy, semantic policy, four-module closed loop, learned model, six-workstation scenario, pose drift or calibrated real-ZED sensor was tested here.',
    'Plot lines connect recorded stages only. No unmeasured quality checkpoints or additional independent reference samplings are created.',
]


def require(condition, message):
    if not condition: raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def finite(value, label):
    require(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value), 'Invalid finite number: '+label)
    return float(value)


def close(left, right, label):
    require(left is None and right is None or left is not None and right is not None
        and math.isclose(float(left), float(right), rel_tol=1e-10, abs_tol=1e-10), label+': values differ')


def inside(root, relative):
    path = (root/relative).resolve()
    require(path.is_relative_to(root.resolve()), 'Artifact path escapes evidence: '+relative)
    return path


def inventory(root):
    expected = read(root/'artifact_hashes.json')
    # V23 seals nested inventory files too; exclude only this directory's own.
    actual = {str(p.relative_to(root)) for p in root.rglob('*')
        if p.is_file() and p != root/'artifact_hashes.json'}
    require(set(expected) == actual, 'Artifact inventory differs: '+str(root))
    for name, wanted in expected.items(): require(sha(inside(root, name)) == wanted, 'Artifact hash differs: '+str(root/name))
    return len(expected)


def atomic_json(path, value):
    data = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)+'\n').encode()
    fd, name = tempfile.mkstemp(prefix=path.name+'.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream: stream.write(data); stream.flush(); os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name): os.unlink(name)


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False, allow_nan=False) if isinstance(value, (dict, list)) else value for key, value in row.items()})


def array_hash(value):
    import numpy as np
    array = np.ascontiguousarray(value)
    return hashlib.sha256(f'{array.dtype.str}:{array.shape}:'.encode()+array.tobytes()).hexdigest()


def read_raw_packet(path):
    """Independently parse the fixed raw packet layout and its framed digest."""
    import numpy as np
    with np.load(path, allow_pickle=False) as arrays:
        keys = {'metadata'} | {'frame__'+k for k in FRAME_FIELDS} | {'scan__'+k for k in SCAN_FIELDS}
        require(set(arrays.files) == keys, 'Raw sensor field inventory differs')
        metadata = json.loads(str(arrays['metadata'].item()))
        require(set(metadata) == set(PACKET_FIELDS)-{'frame', 'scan'}, 'Raw metadata fields differ')
        h = hashlib.sha256()
        def append(value): h.update(len(value).to_bytes(8, 'big')); h.update(value)
        for field in PACKET_FIELDS:
            append(field.encode())
            if field in ('frame', 'scan'):
                for inner in FRAME_FIELDS if field == 'frame' else SCAN_FIELDS:
                    arr = np.ascontiguousarray(arrays[field+'__'+inner])
                    require(not arr.dtype.hasobject, 'Object array in sensor packet')
                    append(inner.encode()); append(arr.dtype.str.encode()); append(str(arr.shape).encode()); append(arr.tobytes())
            else: append(digest(metadata[field]).encode())
        nonsemantic = {key: array_hash(arrays[name]) for key, name in NONSEMANTIC.items()}
        semantic = arrays['frame__semantic']
        observed_values, observed_counts = np.unique(semantic, return_counts=True)
        return dict(metadata=metadata, packet_sha256=h.hexdigest(), nonsemantic=nonsemantic,
            camera_timestamp_s=float(arrays['frame__timestamp_s']),
            color_sha256=array_hash(arrays['frame__color_rgb']), semantic_sha256=array_hash(semantic),
            visible_semantic_pixels={str(int(k)): int(v) for k, v in zip(observed_values, observed_counts) if int(k) != 0})


def validate_f1(row):
    p, r, f = [finite(row[key], 'P/R/F1') for key in ('precision', 'recall', 'f1')]
    require(all(0 <= value <= 1 for value in (p, r, f)), 'P/R/F1 out of [0,1]')
    close(f, 2*p*r/(p+r) if p+r else 0., 'F1 formula')


def validate_outline(metric, *, coverage, returned, collisions, failed, paid_actions):
    require(metric['contract'] == 'facility-three-projection-outline-v23-1', 'Outline contract changed')
    require(metric['mission_asset_count'] == len(metric['instances']) == 1 and metric['instances'][0]['id'] == 0, 'Single fixed facility required')
    close(metric['coverage_2d'], coverage, 'Outline coverage')
    require(metric['returned'] == returned and metric['collisions'] == collisions and metric['failed'] == failed, 'Outline qualification fields differ')
    require(metric['budget_verified'] is True and metric['qualification_incomplete'] == [] and metric['primitive_budget_compliant'] == (paid_actions <= 96), 'Outline budget receipt differs')
    eligible = coverage >= .8 and returned and collisions == 0 and not failed and paid_actions <= 96
    require(metric['eligible'] == eligible, 'Outline eligibility differs')
    require(metric['category_or_visit_count_used'] is False and metric['three_dimensional_hole_filling'] is False
        and metric['reference_is_trajectory_independent'] is True and metric['completion_is_primary_reward'] is False, 'Outline scope changed')
    instance = metric['instances'][0]; projections = instance['projections']; dimensions = instance['dimensions']
    require(set(projections) == set(PLANES), 'Three fixed projections required')
    completed = not instance['missing']
    for plane, row in projections.items():
        iou = finite(row['iou'], 'projection IoU'); require(0 <= iou <= 1, 'IoU outside [0,1]')
        missing = row['missing_projection']; lower, upper = row['hausdorff_lower_m'], row['hausdorff_upper_m']
        if missing:
            require(iou == 0 and lower is None and upper is None and row['boundary_symmetric_mean_m'] is None, 'Missing projection must have zero score and unavailable distance')
            completed = False
        else:
            require(0 <= finite(lower, 'Hausdorff lower') <= finite(upper, 'Hausdorff upper'), 'Invalid Hausdorff bounds')
            require(upper-lower <= .005+1e-10, 'Hausdorff enclosure exceeds declared half-spacing')
            require(finite(row['boundary_symmetric_mean_m'], 'mean boundary error') >= 0, 'Negative boundary distance')
            completed = completed and upper <= .05 and iou >= .90
        for tag in TAGS:
            validate_f1(row[tag])
            if missing: require(all(row[tag][key] == 0 for key in ('precision', 'recall', 'f1')), 'Missing projection scored nonzero')
    if instance['missing']:
        require(all(p['missing_projection'] for p in projections.values()), 'Missing asset has a nonempty projection')
        require(dimensions['absolute_size_error_xyz_m'] is None and dimensions['center_error_inf_m'] is None, 'Missing shape has fabricated dimensions')
    else:
        errors = dimensions['absolute_size_error_xyz_m']; require(len(errors) == 3, 'Three size errors required')
        for predicted, reference, error in zip(dimensions['predicted_size_xyz_m'], dimensions['reference_size_xyz_m'], errors):
            close(error, abs(finite(predicted, 'predicted size')-finite(reference, 'reference size')), 'Size error formula')
        center = finite(dimensions['center_error_inf_m'], 'center error'); require(center >= 0, 'Negative center error')
        completed = completed and max(errors) <= .05 and center <= .05
    require(instance['completed'] == bool(completed) and metric['completion_fraction'] == float(bool(completed)), 'Frozen outline completion gate differs')
    require(metric['missing_asset_count'] == int(instance['missing']), 'Missing asset count differs')
    for tag in TAGS:
        boundary = sum(p[tag]['f1'] for p in projections.values())/3
        quality = sum(min(p[tag]['f1'], p['iou']) for p in projections.values())/3
        close(instance[tag]['outline_f1'], boundary, 'Instance boundary diagnostic')
        close(instance[tag]['outline_quality'], quality, 'Instance quality formula')
        close(metric[tag]['outline_macro_f1'], boundary, 'Macro boundary diagnostic')
        close(metric[tag]['outline_macro_quality'], quality, 'Macro outline quality')
        close(metric[tag]['joint_outline'], coverage*quality, 'Joint outline formula')


def validate_surface(metric, coverage, returned, collisions, failed):
    require(metric['mission_asset_count'] == len(metric['instances']) == 1 and metric['instances'][0]['id'] == 0, 'Surface task count differs')
    close(metric['coverage_2d'], coverage, 'Surface coverage')
    require(metric['returned'] == returned and metric['collisions'] == collisions and metric['failed'] == failed, 'Surface qualification fields differ')
    require(metric['eligible'] == (coverage >= .8 and returned and collisions == 0 and not failed), 'Surface eligibility differs')
    instance = metric['instances'][0]; require(metric['missing_asset_count'] == int(instance['missing']), 'Surface missing count differs')
    for tag in TAGS:
        row = instance[tag]; validate_f1(row)
        for alias in ('external_macro_f1', 'asset_macro_f1'): close(metric[tag][alias], row['f1'], 'Surface macro alias')
        for alias in ('joint_external', 'joint_asset'): close(metric[tag][alias], coverage*row['f1'], 'Surface joint alias')
        close(metric[tag]['macro_precision'], row['precision'], 'Surface macro precision')
        close(metric[tag]['macro_recall'], row['recall'], 'Surface macro recall')
        if instance['missing']: require(row['precision'] == row['recall'] == row['f1'] == 0, 'Missing surface scored nonzero')
    for direction in ('accuracy', 'completeness'):
        for statistic in ('mean_m', 'rmse_m', 'p95_m'):
            value = instance[direction][statistic]
            require(value is None or finite(value, 'continuous surface error') >= 0, 'Negative surface error')
            close(metric['continuous_external'][direction]['complete_mission_macro_'+statistic], value, 'Continuous surface macro')
    legacy = metric['global_legacy']
    for tag in TAGS:
        row = {key: legacy[key+'_'+tag] for key in ('precision', 'recall', 'f1')}; validate_f1(row)
        close(legacy['joint_'+tag], coverage*row['f1'], 'Legacy global joint')


def verify_source(source):
    manifest = read(source/'manifest.json')
    require(manifest.get('status') == 'complete', 'Formal analysis requires manifest complete; no partial-result analysis')
    config = manifest['config']; route = manifest['route']
    require(config['version'] == 'facility-paid-shape-response-v23-1' and config['kinds'] == ['simple', 'complex'], 'Expected frozen V23 shape bench')
    require(config['stages'] == route['stage_indices'] == STAGES and config['paid_actions_per_case'] == route['total_paid_actions'] == 96, 'Frozen stage/action contract differs')
    require(config['sensor_model'] == 'iid_025px' and config['noise_seed'] == 1901, 'Sensor condition changed')
    require(config['outline'] == dict(boundary_spacing_m=.01, thresholds=[.02, .05, .10], completion_tolerance_m=.05, completion_minimum_iou=.9), 'Outline thresholds changed')
    require(config['surface'] == dict(count_per_asset=3000, predicted_per_asset=4000, global_count=12000, reference_stride=4, seed=2026), 'Surface reference contract changed')
    require(config['primary_shape_quality'] == 'mean_over_assets_and_xy_xz_yz(min(boundary_length_f1_05cm,projection_iou))', 'Primary shape objective changed')
    require(config['joint'] == 'true_2d_coverage * primary_shape_quality', 'Joint objective changed')
    for key in ('autonomous_planner', 'semantic_policy', 'training', 'new_six_asset_world', 'geometry_completion', 'full_architecture_efficacy_proven'):
        require(config[key] is False, 'Unexpected capability flag: '+key)
    require(config['shared_scripted_route'] and manifest['preparation_paid_actions'] == 0 and not manifest['new_policy_training'], 'Preparation/route scope changed')
    require([(c['index'], c['kind']) for c in manifest['cases']] == [(0, 'simple'), (1, 'complex')], 'Both complete cases required')
    require(manifest['initial_nonsemantic_pair_identical'] and manifest['cases'][0]['nonsemantic_initial'] == manifest['cases'][1]['nonsemantic_initial'], 'Prepared initial pairing differs')
    actions = route['actions']; require(len(actions) == 96 and actions.count('forward') == 84 and actions.count('left')+actions.count('right') == 12, 'Primitive action accounting differs')
    require([a for s in route['segments'] for a in s['actions']] == actions, 'Route segment concatenation differs')
    require(route['identical_across_kinds'] and route['stage_evaluation_acquires_no_new_frame'] and not route['planner'] and not route['semantic_policy'], 'Scripted action boundary differs')
    artifacts = inventory(source); frozen = manifest['source_sha256']
    require(sha(source/'sources.zip') == manifest['source_archive_sha256'], 'Frozen source archive changed')
    with zipfile.ZipFile(source/'sources.zip') as archive:
        require(set(archive.namelist()) == set(frozen), 'Frozen source inventory differs')
        for name, wanted in frozen.items(): require(hashlib.sha256(archive.read(name)).hexdigest() == wanted, 'Frozen source member changed: '+name)
        require(json.loads(archive.read('configs/virtual3d/facility_shape_v23_probe.json')) == config, 'Archived config and manifest differ')
    require('docs/research/V23_OUTLINE_AND_SHAPE_PROBE_PROTOCOL_20260915.md' in frozen and 'tests/virtual3d/test_facility_outline_v23.py' in frozen, 'Protocol/behavior tests not frozen')
    require(manifest['versions']['shapely'] == '2.1.2', 'Planar geometry version changed')
    for case in manifest['cases']:
        require(sha(inside(source, case['surface_reference'])) == case['surface_reference_sha256'], 'Frozen surface reference changed')
        cfg = case['world_config']
        for key, expected in dict(max_depth_m=5., voxel_m=.04, truncation_m=.12, width_px=96, height_px=72).items():
            require(cfg[key] == expected, 'Shared physical setting changed: '+key)
        audit = case['static_route_audit']
        require(audit['static_only'] and audit['no_sense_or_step_called'] and audit['paid_actions'] == 96 and audit['translations'] == 84 and audit['turns'] == 12, 'Static route metadata changed')
        require(len(audit['states']) == 97 and audit['states'][0] == audit['states'][-1], 'Static route pose count/return differs')
        folder = source/f'case_{case["index"]:02d}'; inventory(folder)
        receipt = read(folder/'verification.json')
        require(receipt['status'] == 'passed' and receipt['independent_process'] and receipt['fresh_world_sensor_replay'] and receipt['saved_packet_tsdf_replay'], 'Two fresh sensor and saved-packet fusion replays required')
        require(receipt['physical_process_id'] != receipt['replay_process_id'] and receipt['raw_packets_verified'] == 97 and receipt['paid_actions'] == 96 and receipt['stage_mesh_and_metric_checks'] == 6, 'Replay process/count contract differs')
    return manifest, dict(artifact_files_verified=artifacts, frozen_source_files_verified=len(frozen),
        source_manifest_sha256=sha(source/'manifest.json'), source_inventory_sha256=sha(source/'artifact_hashes.json'),
        source_archive_is_authoritative=True, independent_process_replays_verified=2)


def completion_reasons(instance):
    reasons = []
    if instance['missing']: reasons.append('missing_asset')
    for plane, projection in instance['projections'].items():
        if projection['missing_projection']: reasons.append(plane+':missing_projection')
        else:
            if projection['hausdorff_upper_m'] > .05: reasons.append(plane+':hausdorff_upper_gt_5cm')
            if projection['iou'] < .90: reasons.append(plane+':iou_lt_0.90')
    dims = instance['dimensions']
    if dims['absolute_size_error_xyz_m'] is None: reasons.append('size_unavailable')
    else:
        reasons.extend('size_'+axis+':error_gt_5cm' for axis, error in zip('xyz', dims['absolute_size_error_xyz_m']) if error > .05)
    if dims['center_error_inf_m'] is None: reasons.append('center_unavailable')
    elif dims['center_error_inf_m'] > .05: reasons.append('center:error_gt_5cm')
    return reasons


def analyze_case(source, manifest, case):
    folder = source/f'case_{case["index"]:02d}'
    raw = read(folder/'result.json'); timing = read(folder/'timing.json'); receipt = read(folder/'verification.json')
    require(raw['status'] == 'complete' and raw['index'] == case['index'] and raw['kind'] == case['kind'], 'Physical case identity differs')
    require(raw['paid_actions'] == 96 and raw['raw_frames'] == 97 and raw['translation_actions'] == 84, 'Paid action/frame counts differ')
    close(raw['path_distance_m'], 84*.2, 'Paid translation distance')
    require(raw['shared_scripted_route'] and all(raw[key] is False for key in ('semantic_policy', 'autonomous_planner', 'four_module_closed_loop', 'full_architecture_efficacy_proven')), 'Physical capability flags differ')
    require(timing['process_id'] == receipt['physical_process_id'], 'Physical process receipt differs')
    require(raw['termination'] == 'declared_paid_route_complete' and raw['world_done_at_endpoint'] is False, 'Declared route termination differs')
    trace = raw['trace']; require([row['action_id'] for row in trace] == list(range(97)), 'Paid trace missing/duplicated')
    require(sorted(p.name for p in (folder/'packets').iterdir()) == [f'{i:04d}.npz' for i in range(97)], 'Raw packet sequence differs')
    require(sorted(p.name for p in (folder/'meshes').iterdir()) == sorted(stage+'.npz' for stage in STAGES), 'Stage mesh set differs')
    checks = []; trace_rows = []
    for i, row in enumerate(trace):
        packet = read_raw_packet(folder/f'packets/{i:04d}.npz'); metadata = packet['metadata']
        require(packet['packet_sha256'] == row['packet_sha256'] and packet['nonsemantic'] == row['nonsemantic'], 'Saved sensor digest differs at '+str(i))
        require(metadata['action_id'] == i and metadata['episode_id'] == case['kind'] and metadata['scene_id'] == 'D23-SHAPE', 'Raw frame identity differs')
        require(metadata['frame_id'] == f'frame-{i:04d}', 'Raw frame sequence differs')
        require(metadata['position']+[metadata['heading']] == row['pose'] == case['static_route_audit']['states'][i], 'Measured pose differs from paid route')
        action = None if i == 0 else manifest['route']['actions'][i-1]
        require(row['action'] == metadata['action'] == action, 'Paid action differs from declaration')
        require(row['collision'] == metadata['collision'] is False and row['world_done'] == metadata['done'] is False, 'Completed bench contains collision/early done')
        close(packet['camera_timestamp_s'], i*case['world_config']['action_duration_s'], 'Camera acquisition time')
        require(0 <= finite(row['coverage_2d'], 'coverage trace') <= 1, 'Coverage outside [0,1]')
        if i == 0:
            require(packet['packet_sha256'] == case['initial_packet_sha256'] and packet['nonsemantic'] == case['nonsemantic_initial'], 'Prepared initial input differs')
        checks.append(packet)
        trace_rows.append(dict(kind=case['kind'], **row, color_sha256=packet['color_sha256'],
            semantic_sha256=packet['semantic_sha256'], visible_semantic_pixels=packet['visible_semantic_pixels']))
    returned = trace[-1]['pose'] == trace[0]['pose']
    require(raw['returned_to_anchor'] == returned and raw['collisions'] == 0, 'Endpoint return/collision receipt differs')
    stages = []; projections = []
    require([(row['stage'], row['action_id']) for row in raw['checkpoints']] == list(STAGES.items()), 'Six fixed stage checkpoints required')
    import numpy as np
    with np.load(source/case['surface_reference'], allow_pickle=False) as cache:
        metadata = json.loads(str(cache['metadata'].item()))
        require(metadata['reference_signature'] == case['surface_reference_signature'], 'Surface cache signature differs')
    for checkpoint in raw['checkpoints']:
        stage, action_id = checkpoint['stage'], checkpoint['action_id']; observed = trace[action_id]
        with np.load(folder/f'meshes/{stage}.npz', allow_pickle=False) as mesh:
            require(set(mesh.files) == set(checkpoint['mesh_sha256']) == {'vertices', 'triangles', 'vertex_colors'}, 'Stage mesh fields differ')
            for key, wanted in checkpoint['mesh_sha256'].items(): require(array_hash(mesh[key]) == wanted, 'Stage mesh content differs: '+stage+'/'+key)
        outline, surface = checkpoint['outline'], checkpoint['surface']; at_anchor = observed['pose'] == trace[0]['pose']
        require(outline['reference_signature'] == case['outline_reference_signature'] and surface['reference_signature'] == case['surface_reference_signature'], 'Stage reference signature differs')
        validate_outline(outline, coverage=observed['coverage_2d'], returned=at_anchor, collisions=0, failed=False, paid_actions=action_id)
        validate_surface(surface, observed['coverage_2d'], at_anchor, 0, False)
        instance = outline['instances'][0]; external = surface['instances'][0]; dims = instance['dimensions']
        failure_reasons = completion_reasons(instance)
        require(instance['completed'] == (not failure_reasons), 'Completion reason audit differs')
        row = dict(kind=case['kind'], stage=stage, action_id=action_id, C2D=observed['coverage_2d'],
            Q_outline=outline['05cm']['outline_macro_quality'], boundary_F1=outline['05cm']['outline_macro_f1'],
            mean_projection_IoU=sum(p['iou'] for p in instance['projections'].values())/3,
            J_outline=outline['05cm']['joint_outline'], completed=instance['completed'],
            completion_failure_reasons=failure_reasons, eligible=outline['eligible'], returned=at_anchor,
            budget_verified=outline['budget_verified'], primitive_budget_compliant=outline['primitive_budget_compliant'],
            missing_outline=instance['missing'], missing_surface=external['missing'],
            center_error_inf_m=dims['center_error_inf_m'],
            outline_reference_size_xyz_m=dims['reference_size_xyz_m'], outline_predicted_size_xyz_m=dims['predicted_size_xyz_m'],
            surface_reference_samples=external['reference_samples'], surface_observable_area_m2=external['observable_area_m2'],
            prediction_source='raw paid-sensor Open3D TSDF, no completion')
        for axis, value in zip('xyz', dims['absolute_size_error_xyz_m'] or [None]*3): row['size_error_'+axis+'_m'] = value
        for tag in TAGS:
            row.update({f'Q_outline_{tag}': outline[tag]['outline_macro_quality'],
                f'boundary_F1_{tag}': outline[tag]['outline_macro_f1'],
                f'external_F1_{tag}': external[tag]['f1'], f'external_precision_{tag}': external[tag]['precision'],
                f'external_recall_{tag}': external[tag]['recall'], f'J_external_{tag}': surface[tag]['joint_external']})
        for direction in ('accuracy', 'completeness'):
            for statistic in ('mean_m', 'rmse_m', 'p95_m'): row[f'external_{direction}_{statistic}'] = external[direction][statistic]
        for key, value in surface['global_legacy'].items(): row['global_legacy_'+key] = value
        row['original_surface_dimensions'] = external['dimensions']
        stages.append(row)
        for plane in PLANES:
            p = instance['projections'][plane]
            for tag in TAGS:
                f, iou = p[tag]['f1'], p['iou']
                projections.append(dict(kind=case['kind'], stage=stage, action_id=action_id, plane=plane, threshold=tag,
                    precision=p[tag]['precision'], recall=p[tag]['recall'], boundary_F1=f, projection_IoU=iou,
                    projection_quality=min(f, iou), quality_limiter='boundary_F1' if f < iou else 'projection_IoU' if iou < f else 'equal',
                    boundary_symmetric_mean_m=p['boundary_symmetric_mean_m'], hausdorff_lower_m=p['hausdorff_lower_m'],
                    hausdorff_upper_m=p['hausdorff_upper_m'], missing_projection=p['missing_projection'],
                    predicted_area_m2=p['predicted_area_m2'], reference_area_m2=p['reference_area_m2'],
                    predicted_components=p['predicted_components'], reference_components=p['reference_components']))
    return dict(raw=raw, stages=stages, projections=projections, trace_rows=trace_rows, packets=checks,
        provenance=dict(index=case['index'], kind=case['kind'], inventory_sha256=sha(folder/'artifact_hashes.json'),
            result_sha256=sha(folder/'result.json'), replay_sha256=sha(folder/'verification.json'),
            physical_process_id=receipt['physical_process_id'], replay_process_id=receipt['replay_process_id'],
            raw_packets_independently_hashed=97, stage_meshes_independently_hashed=6))


def demand_conditions(stage_rows):
    by = {(r['kind'], r['stage']): r for r in stage_rows}
    delta = {kind: by[kind, 'extra']['Q_outline']-by[kind, 'coarse']['Q_outline'] for kind in ('simple', 'complex')}
    gates = dict(simple_coarse_completed=by['simple', 'coarse']['completed'],
        complex_positive_and_larger_increment=delta['complex'] > 0 and delta['complex'] > delta['simple'])
    return delta, gates, all(gates.values())


def paired_history(data):
    a, b = [item['raw']['trace'] for item in data]
    differences = [dict(action_id=x['action_id'], fields=[key for key in x['nonsemantic'] if x['nonsemantic'][key] != y['nonsemantic'][key]])
        for x, y in zip(a, b) if x['nonsemantic'] != y['nonsemantic']]
    return dict(initial_identical=not differences or differences[0]['action_id'] > 0,
        first_difference_action=differences[0]['action_id'] if differences else None,
        first_difference_fields=differences[0]['fields'] if differences else [],
        coarse_prefix_identical=not differences or differences[0]['action_id'] > 23,
        differences=differences,
        same_paid_actions_and_poses=all((x['action'], x['pose']) == (y['action'], y['pose']) for x, y in zip(a, b)))


def interval_rows(stages, projections):
    by = {(row['kind'], row['stage']): row for row in stages}
    columns = ('C2D', 'Q_outline', 'boundary_F1', 'mean_projection_IoU', 'J_outline',
        'center_error_inf_m', 'size_error_x_m', 'size_error_y_m', 'size_error_z_m',
        'external_F1_05cm', 'external_precision_05cm', 'external_recall_05cm',
        'external_accuracy_mean_m', 'external_accuracy_rmse_m', 'external_accuracy_p95_m',
        'external_completeness_mean_m', 'external_completeness_rmse_m', 'external_completeness_p95_m')
    rows = []; projected = []
    for kind in ('simple', 'complex'):
        before, after = by[kind, 'coarse'], by[kind, 'extra']
        for name in columns:
            x, y = before[name], after[name]
            rows.append(dict(kind=kind, interval='coarse_23_to_extra_73', paid_actions=50,
                metric=name, coarse_value=x, extra_value=y, delta=None if x is None or y is None else y-x))
        contribution = 0.
        for plane in PLANES:
            pair = [next(r for r in projections if r['kind'] == kind and r['stage'] == s and r['plane'] == plane and r['threshold'] == '05cm') for s in ('coarse', 'extra')]
            x, y = pair; delta = y['projection_quality']-x['projection_quality']; contribution += delta/3
            projected.append(dict(kind=kind, plane=plane, coarse_Q=x['projection_quality'], extra_Q=y['projection_quality'],
                delta_Q=delta, contribution_to_facility_delta_Q=delta/3,
                coarse_boundary_F1=x['boundary_F1'], extra_boundary_F1=y['boundary_F1'],
                coarse_IoU=x['projection_IoU'], extra_IoU=y['projection_IoU'],
                coarse_limiter=x['quality_limiter'], extra_limiter=y['quality_limiter'],
                coarse_hausdorff_upper_m=x['hausdorff_upper_m'], extra_hausdorff_upper_m=y['hausdorff_upper_m']))
        close(contribution, after['Q_outline']-before['Q_outline'], 'Projection delta contributions')
    return rows, projected


def verify_aggregate(source, data, pairing, deltas, conditions, passed):
    aggregate = read(source/'result.json')
    require(aggregate['status'] == 'complete' and aggregate['physical_trajectories'] == aggregate['independent_process_replays'] == 2, 'Complete two-case aggregate required')
    require(aggregate['physical_paid_actions'] == aggregate['replay_paid_actions'] == 192, 'Aggregate paid action counts differ')
    require(aggregate['paired_history'] == pairing, 'Aggregate pairing/revelation differs')
    for kind, delta in deltas.items(): close(aggregate['coarse_to_extra_quality_delta'][kind], delta, 'Aggregate fixed interval delta')
    gates = {**conditions, 'all_returned_without_collision': all(d['raw']['returned_to_anchor'] and d['raw']['collisions'] == 0 for d in data), 'semantic_information_or_efficacy_proven': False}
    require(aggregate['gates'] == gates and aggregate['observation_demand_gate_passed'] == passed, 'Predeclared demand gate differs')
    require(len(aggregate['cases']) == 2, 'Aggregate omitted a shape')
    for recorded, actual in zip(aggregate['cases'], data):
        raw = actual['raw']
        require(recorded == dict(kind=raw['kind'], checkpoints=raw['checkpoints'], path_distance_m=raw['path_distance_m'],
            paid_actions=raw['paid_actions'], returned=raw['returned_to_anchor']), 'Aggregate stage result differs')
    require(aggregate['independent_parent_layouts'] == aggregate['noise_seeds'] == 1 and all(aggregate[key] is False
        for key in ('full_architecture_efficacy_proven', 'semantic_policy', 'training')), 'Aggregate scope differs')
    return gates


def plot_stages(output, stages):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey=True)
    plots = [('Q_outline', 'Primary outline quality Q = mean min(F1, IoU)'),
        ('boundary_F1', 'Outline boundary F1 at 5 cm (diagnostic)'),
        ('mean_projection_IoU', 'Mean orthographic envelope IoU'),
        ('external_F1_05cm', 'Original observable external-surface F1 at 5 cm')]
    for ax, (key, title) in zip(axes.flat, plots):
        for kind, color in (('simple', '#0072B2'), ('complex', '#D55E00')):
            rows = [row for row in stages if row['kind'] == kind]
            ax.plot([r['action_id'] for r in rows], [r[key] for r in rows], '.-', color=color, label=kind)
        ax.axvline(23, color='#999999', linewidth=.8, linestyle=':')
        ax.axvline(73, color='#999999', linewidth=.8, linestyle=':')
        ax.set(title=title, xlabel='Paid actions (all translations and turns)', ylim=(0, 1.02))
        ax.set_xticks(list(STAGES.values()), [s+' ('+str(i)+')' for s, i in STAGES.items()],
            fontsize=8, rotation=90)
        ax.grid(alpha=.2); ax.legend(loc='lower right')
    fig.suptitle('V23 paid observation bench: one layout and noise seed; raw TSDF; no semantic policy', fontsize=11)
    fig.tight_layout(); fig.savefig(output/'shape_stage_quality.png', dpi=140); plt.close(fig)


def self_test():
    from copy import deepcopy
    projection = dict(iou=.2, missing_projection=False, hausdorff_lower_m=0., hausdorff_upper_m=0., boundary_symmetric_mean_m=0.)
    projection.update({tag: dict(precision=1., recall=1., f1=1.) for tag in TAGS})
    instance = dict(id=0, missing=False, completed=False, projections={p: deepcopy(projection) for p in PLANES},
        dimensions=dict(absolute_size_error_xyz_m=[0., 0., 0.], center_error_inf_m=0., predicted_size_xyz_m=[1., 1., 1.], reference_size_xyz_m=[1., 1., 1.]))
    instance.update({tag: dict(outline_f1=1., outline_quality=.2) for tag in TAGS})
    metric = dict(contract='facility-three-projection-outline-v23-1', mission_asset_count=1, instances=[instance],
        coverage_2d=.9, returned=True, collisions=0, failed=False, budget_verified=True, qualification_incomplete=[],
        primitive_budget_compliant=True, eligible=True, category_or_visit_count_used=False,
        three_dimensional_hole_filling=False, reference_is_trajectory_independent=True, completion_is_primary_reward=False,
        completion_fraction=0., missing_asset_count=0)
    metric.update({tag: dict(outline_macro_f1=1., outline_macro_quality=.2, joint_outline=.18) for tag in TAGS})
    validate_outline(metric, coverage=.9, returned=True, collisions=0, failed=False, paid_actions=80)
    wrong = deepcopy(metric); wrong['05cm']['outline_macro_quality'] = 1.
    try: validate_outline(wrong, coverage=.9, returned=True, collisions=0, failed=False, paid_actions=80)
    except ValueError: pass
    else: raise AssertionError('Boundary F1 silently replaced area-constrained Q')
    wrong = deepcopy(metric); wrong['instances'][0]['completed'] = True; wrong['completion_fraction'] = 1.
    try: validate_outline(wrong, coverage=.9, returned=True, collisions=0, failed=False, paid_actions=80)
    except ValueError: pass
    else: raise AssertionError('Low IoU passed completion')
    rows = [dict(kind=k, stage=s, Q_outline=q, completed=completed) for k, values in
        [('simple', [(.4, False), (.5, False)]), ('complex', [(.2, False), (.6, True)])]
        for s, (q, completed) in zip(('coarse', 'extra'), values)]
    _, gates, passed = demand_conditions(rows)
    require(gates['complex_positive_and_larger_increment'] and not passed, 'Failed simple coarse condition was discarded')
    rows[0]['completed'] = True; rows[1]['Q_outline'] = .8
    rows[2]['Q_outline'] = .4; rows[3]['Q_outline'] = .8
    require(not demand_conditions(rows)[2], 'Equal increments passed strict greater condition')
    print('Five synthetic analysis checks passed; no evidence read or written.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test: self_test(); return
    source, output = args.source.resolve(), args.output.resolve()
    require(not output.is_relative_to(source) and not source.is_relative_to(output), 'Output must be separate from evidence')
    script = Path(__file__).resolve(); script_sha256 = sha(script)
    manifest, checks = verify_source(source)
    data = [analyze_case(source, manifest, case) for case in manifest['cases']]
    stages = [r for d in data for r in d['stages']]; projections = [r for d in data for r in d['projections']]
    deltas, conditions, passed = demand_conditions(stages); pairing = paired_history(data)
    gates = verify_aggregate(source, data, pairing, deltas, conditions, passed)
    intervals, contributions = interval_rows(stages, projections)
    coarse_simple = next(r for r in stages if r['kind'] == 'simple' and r['stage'] == 'coarse')
    summary = dict(status='complete', source_root=str(source), verification=checks,
        physical_trajectories=2, independent_process_replays=2, physical_paid_actions=192, replay_paid_actions=192,
        independent_parent_layouts=1, noise_seeds=1, all_shapes_and_negative_conditions_retained=True,
        fixed_stages=STAGES, stage_metrics=stages, coarse_to_extra_quality_delta=deltas,
        projection_quality_delta_contributions=contributions, paired_history=pairing, gates=gates,
        observation_demand_gate_passed=passed, simple_coarse_completion_failures=coarse_simple['completion_failure_reasons'],
        initial_visible_semantic_pixels={d['raw']['kind']: d['packets'][0]['visible_semantic_pixels'] for d in data},
        physical_experiments_started_by_analyzer=False, automatic_six_asset_or_training_expansion_allowed=False,
        semantic_information_or_efficacy_proven=False, full_architecture_efficacy_proven=False,
        geometry_completion_executed=False, autonomous_planner=False, interpretation=NOTES)
    require(sha(script) == script_sha256, 'Analysis source changed during verification')
    compact = dict(status='complete', observation_demand_gate_passed=passed, gates=gates,
        coarse_to_extra_quality_delta=deltas, first_nonsemantic_difference_action=pairing['first_difference_action'])
    if args.verify_only: print(json.dumps({**compact, 'artifacts_written': False}, indent=2)); return
    output.mkdir(parents=True, exist_ok=False)
    try:
        write_csv(output/'stage_metrics.csv', stages)
        write_csv(output/'projection_metrics.csv', projections)
        write_csv(output/'coarse_to_extra_deltas.csv', intervals)
        write_csv(output/'projection_delta_contributions.csv', contributions)
        write_csv(output/'paid_coverage_trace.csv', [r for d in data for r in d['trace_rows']])
        write_csv(output/'paired_nonsemantic_differences.csv', pairing['differences'])
        plot_stages(output, stages)
        atomic_json(output/'result.json', summary)
        atomic_json(output/'source_sha256.json', {str(script.relative_to(ROOT)): script_sha256})
        with zipfile.ZipFile(output/'sources.zip', 'x', zipfile.ZIP_DEFLATED) as archive:
            archive.write(script, str(script.relative_to(ROOT)))
        atomic_json(output/'input_provenance.json', dict(source_root=str(source),
            source_manifest_sha256=sha(source/'manifest.json'), source_inventory_sha256=sha(source/'artifact_hashes.json'),
            source_result_sha256=sha(source/'result.json'), source_archive_sha256=manifest['source_archive_sha256'],
            frozen_source_sha256=manifest['source_sha256'], physical_versions=manifest['versions'],
            cases=[d['provenance'] for d in data], no_simulation_mapping_or_policy_execution=True))
        require(sha(script) == script_sha256, 'Analysis source changed during writing')
        atomic_json(output/'artifact_hashes.json', {str(p.relative_to(output)): sha(p) for p in sorted(output.rglob('*')) if p.is_file()})
        inventory(output)
    except Exception as error:
        atomic_json(output/'analysis_failure.json', dict(status='failed', error=repr(error))); raise
    print(json.dumps({**compact, 'output': str(output)}, indent=2))


if __name__ == '__main__':
    try: main()
    except (ValueError, FileNotFoundError) as error:
        print('Analysis refused: '+str(error), file=sys.stderr); raise SystemExit(2)
