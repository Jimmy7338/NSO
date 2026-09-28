#!/usr/bin/env python3
"""Preregistered V36 frozen confirmation routes: prepare, main, independent replay, analyze.

Preparation uses both public templates and configuration arithmetic only. No
acquisition is executed by import or --help. All artifacts are exclusive-create; a failed main
directory spends its quota. The immutable preparation manifest is never edited.
"""
import argparse
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import time
import traceback
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUTPUT = ROOT / 'audit_results/v36_online_confirmation_20260918'
SCENE = ROOT / 'configs/virtual3d/v33_direction_scene_r1_20260917.json'
INFORMATION = ROOT / 'audit_results/v34_pixel_information_20260918'
INFORMATION_REVIEW = ROOT / 'audit_results/v34_pixel_information_review_20260918'
MEASUREMENT = ROOT / 'audit_results/surface_measurement_v34_unit_20260918'
SENSOR = ROOT / 'audit_results/v34_sensor_contract_tests_20260918'
PROTOCOL = ROOT / 'docs/research/V36_ONLINE_CONFIRMATION_PROTOCOL_20260918.md'
PLAN = ROOT / 'docs/research/V36_FROZEN_CONFIRMATION_PLAN_20260918.md'
V35_BATCH = ROOT / 'audit_results/v35_online_development_20260918'
V35_REVIEW = ROOT / 'audit_results/v35_semantic_chain_review_20260918'
ADAPTER_GATE = ROOT / 'audit_results/v36_seed_adapter_tests_20260918'
ONLINE_GATE = ROOT / 'audit_results/v35_saved_observation_gate_20260918'
ONLINE_REVIEW = ROOT / 'audit_results/v35_saved_observation_review_20260918'
V34_MEASURED = ROOT / 'audit_results/v34_fixed_measurement_20260918'
IMPLEMENTATION_GATES = tuple(ROOT / 'audit_results' / name for name in (
    'v35_online_planner_toy_tests_20260918', 'v35_observation_belief_tests_20260918',
    'v35_four_module_controller_tests_20260918'))
MODES = ('G', 'S')
PARENTS = ('P00', 'P01')
MIB = 1024 ** 2
RESERVE = 64 * MIB
CASE_CAP = 40 * MIB
SHARED_CAP = 2 * MIB
BUDGET = 42
PREFIX = 18
HISTORICAL_MAIN = 27
MAIN_LIMIT = 36
MAX_STARTS = 8
SECONDS = 600
HEADINGS = ((0, 1), (1, 0), (0, -1), (-1, 0))


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalized(value):
    if isinstance(value, dict):
        return {str(k): normalized(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [normalized(v) for v in value]
    if hasattr(value, 'tolist'):
        return value.tolist()
    return value


def digest(value):
    return hashlib.sha256(json.dumps(normalized(value), sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def files_under(folder):
    return sorted(p for p in folder.rglob('*') if p.is_file()) if folder.exists() else []


def case_directories():
    return sorted(p for p in OUTPUT.glob('case*') if p.is_dir())


def guard_write(path, size=0):
    path = Path(path)
    relative = path.relative_to(OUTPUT)
    if relative.parts[0].startswith('case'):
        current = files_under(OUTPUT / relative.parts[0])
        cap = CASE_CAP
    else:
        current = [p for p in files_under(OUTPUT)
            if not p.relative_to(OUTPUT).parts[0].startswith('case')]
        cap = SHARED_CAP
    if sum(p.stat().st_size for p in current) + size > cap:
        raise RuntimeError('declared output cap would be exceeded')
    if shutil.disk_usage(ROOT).free - size < RESERVE:
        raise RuntimeError('64 MiB actual disk reserve would be exceeded')


def write_bytes(path, payload):
    guard_write(path, len(payload))
    with Path(path).open('xb') as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def write(path, value):
    write_bytes(path, (json.dumps(normalized(value), ensure_ascii=False,
        indent=2, allow_nan=False) + '\n').encode())


def seal(folder, name, exclude_prefix=None):
    paths = [p for p in files_under(folder) if p.name != name
        and (exclude_prefix is None or p.relative_to(folder).parts[0] != exclude_prefix)]
    write(folder / name, {str(p.relative_to(folder)): sha(p) for p in paths})


def verify_seal(folder, name, exclude_prefix=None):
    inventory = read(folder / name)
    actual = {str(p.relative_to(folder)) for p in files_under(folder) if p.name != name
        and (exclude_prefix is None or p.relative_to(folder).parts[0] != exclude_prefix)}
    if actual != set(inventory):
        raise ValueError('sealed artifact set changed: ' + str(folder))
    for rel, expected in inventory.items():
        if sha(folder / rel) != expected:
            raise ValueError('sealed artifact bytes changed: ' + str(folder / rel))
    return inventory


def cpu_contract():
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        if os.environ.get(key) != '1':
            raise ValueError(key + '=1 is required before Python imports')
    if not sys.dont_write_bytecode:
        raise ValueError('invoke Python with -B')


def library_versions():
    versions = {name: importlib.metadata.version(name) for name in ('numpy', 'scipy')}
    try:
        versions['open3d'] = importlib.metadata.version('open3d')
    except importlib.metadata.PackageNotFoundError:
        # This workspace's bundled Open3D has no distribution metadata.
        # Reading its version does not construct a World, sensor or mapper.
        import open3d
        versions['open3d'] = open3d.__version__
    return versions


def require_not_finalized():
    if any((OUTPUT / name).exists() for name in ('result.json', 'final_seal.json')):
        raise ValueError('analysis is finalized; no additional main, replay or analysis mutations are allowed')


def gate_inputs(review):
    """Read and verify prerequisites without any scene or mapper construction."""
    from nso.research_evidence_v31 import verify_inventory, verify_sources
    gates = (INFORMATION, review, MEASUREMENT, SENSOR)
    for folder in gates:
        verify_inventory(folder)
    for folder in (INFORMATION, MEASUREMENT):
        verify_sources(folder)
    information, audited, metric, sensor = [read(p / 'result.json') for p in gates]
    if not (information.get('status') == 'complete'
            and information.get('all_information_gates_passed') is True
            and information.get('all_pixel_prerequisites_passed') is True):
        raise ValueError('pixel-information prerequisites did not pass')
    if not (audited.get('status') == 'passed' and audited.get('saved_evidence_verified') is True):
        raise ValueError('independent saved-pixel review did not pass')
    for field, name in (('source_result_sha256', 'result.json'),
            ('source_manifest_sha256', 'manifest.json'), ('source_inventory_sha256', 'artifact_hashes.json')):
        if audited.get(field) != sha(INFORMATION / name):
            raise ValueError('independent review does not identify this pixel evidence: ' + field)
    if not (metric.get('status') == 'passed' and metric.get('tests_run') == 11):
        raise ValueError('eleven surface-measurement fixtures are required')
    if not (sensor.get('status') == 'passed' and sensor.get('tests') == 10
            and sensor.get('exit_code') == 0):
        raise ValueError('sensor contract fixtures did not pass')
    for rel, expected in sensor['source_sha256'].items():
        if sha(ROOT / rel) != expected:
            raise ValueError('sensor contract fixture source changed: ' + rel)
    if (review / 'manifest.json').exists():
        review_manifest = read(review / 'manifest.json')
        for rel, expected in review_manifest['source_sha256'].items():
            if sha(ROOT / rel) != expected:
                raise ValueError('independent review source changed: ' + rel)
        if 'source_archive_sha256' in review_manifest:
            verify_sources(review)
    return {str(p.relative_to(ROOT)): sha(p) for folder in gates for p in files_under(folder)}


def online_prerequisites(review):
    from nso.research_evidence_v31 import verify_inventory, verify_sources
    inputs = gate_inputs(review)
    for folder in (*IMPLEMENTATION_GATES, ONLINE_GATE):
        verify_inventory(folder)
        if (folder / 'manifest.json').exists():
            if 'source_archive_sha256' in read(folder / 'manifest.json'):
                verify_sources(folder)
        result = read(folder / 'result.json')
        if folder == ONLINE_GATE:
            if result.get('status') != 'complete' or result.get('implementation_gate_passed') is not True:
                raise ValueError('formal saved-observation implementation gate failed')
        elif result.get('status') != 'passed' or result.get('exit_code') != 0:
            raise ValueError('online implementation fixture failed')
        for rel, expected in result.get('source_sha256', {}).items():
            if sha(ROOT / rel) != expected: raise ValueError('unit source changed: ' + rel)
        inputs.update({str(p.relative_to(ROOT)): sha(p) for p in files_under(folder)})
    # The public reward loader reads dynamic V33 JSON paths that AST source
    # closure cannot discover. Preserve every preregistered upstream input,
    # including both actual public model tables, in each execution's checks.
    for rel, expected in read(ONLINE_GATE / 'manifest.json')['input_sha256'].items():
        if sha(ROOT / rel) != expected: raise ValueError('cached gate upstream input changed: ' + rel)
        inputs[rel] = expected
    verify_inventory(ONLINE_REVIEW); review_manifest = verify_sources(ONLINE_REVIEW)
    audited = read(ONLINE_REVIEW / 'result.json')
    if (audited.get('status') != 'passed' or audited.get('saved_evidence_verified') is not True or
            audited.get('source_result_sha256') != sha(ONLINE_GATE / 'result.json') or
            audited.get('source_inventory_sha256') != sha(ONLINE_GATE / 'artifact_hashes.json')):
        raise ValueError('independent review does not bind this passed cached gate')
    manifest_path = str((ONLINE_GATE / 'manifest.json').relative_to(ROOT))
    if review_manifest['input_sha256'].get(manifest_path) != sha(ONLINE_GATE / 'manifest.json'):
        raise ValueError('independent cached review manifest binding differs')
    for rel, expected in review_manifest['input_sha256'].items():
        if sha(ROOT / rel) != expected: raise ValueError('reviewed cache input changed: ' + rel)
    inputs.update({str(p.relative_to(ROOT)): sha(p) for p in files_under(ONLINE_REVIEW)})
    verify_sources(V34_MEASURED); verify_seal(V34_MEASURED, 'final_seal.json')
    if read(V34_MEASURED / 'result.json').get('measured_fixed_route_gate_passed') is not True:
        raise ValueError('previous measured fixed-route prerequisite failed')
    for name in ('manifest.json', 'config.json', 'result.json', 'final_seal.json'):
        path = V34_MEASURED / name; inputs[str(path.relative_to(ROOT))] = sha(path)
    return inputs


def public_graph_audit(parent, models, prefix):
    """Static contract arithmetic for BOTH public hypotheses, never a World."""
    from env.information_pixel_v34 import swept_clear_v34
    nav = {tuple(p) for p in parent['nav_cells']}; a, b = models
    if (a.poses, a.edges, a.anchor) != (b.poses, b.edges, b.anchor):
        raise ValueError('public hypotheses must share a legal graph')
    tested = 0
    for hypothesis in parent['hypotheses']:
        boxes = [box for asset in hypothesis['assets'] for box in asset['boxes']]
        boxes += parent['background_boxes']
        for node, row in enumerate(a.edges):
            pose = a.poses[node]
            for action, following in row:
                target = list(pose)
                if action == 'forward':
                    dx, dy = HEADINGS[pose[2]]; target[0] += dx; target[1] += dy
                elif action in ('left', 'right'):
                    target[2] = (pose[2] + (1 if action == 'right' else -1)) % 4
                else: raise ValueError('unsupported public primitive')
                if (tuple(target) != a.poses[following] or tuple(target[:2]) not in nav or
                        not swept_clear_v34(pose[:2], target[:2], boxes)):
                    raise ValueError('public safe graph fails exact swept-footprint contract')
                tested += 1
    node = a.anchor
    for action in prefix: node = dict(a.edges[node])[action]
    if len(prefix) != PREFIX or node != a.anchor:
        raise ValueError('common 18-action prefix must return to exact anchor')
    return dict(public_hypotheses=2, public_nodes=len(a.poses), tested_edges=tested,
        prefix_actions=list(prefix), source='public two-template geometry; no actual hypothesis selection',
        world_constructions=0, sensor_queries=0)


def confirmation_prerequisites(review):
    from nso.research_evidence_v31 import verify_sources, verify_inventory
    inputs = online_prerequisites(review)
    for folder in (V35_BATCH, V35_REVIEW, ADAPTER_GATE):
        manifest = verify_sources(folder)
        if folder == V35_BATCH:
            verify_seal(folder, 'final_seal.json')
        else:
            verify_inventory(folder)
        for relative, expected in manifest.get('input_sha256', {}).items():
            if sha(ROOT / relative) != expected:
                raise ValueError('confirmation prerequisite changed: ' + relative)
            inputs[relative] = expected
        inputs.update({str(p.relative_to(ROOT)): sha(p) for p in files_under(folder)})
    prior, audit, adapter = [read(p / 'result.json') for p in (V35_BATCH, V35_REVIEW, ADAPTER_GATE)]
    if (prior['total_main_used'] != HISTORICAL_MAIN or not prior['qualification_and_replay_gate_passed']
            or audit['complete_development_gate_passed'] is not True
            or audit['source_batch_result_sha256'] != sha(V35_BATCH / 'result.json')):
        raise ValueError('V35 complete development gate is not bound or did not pass')
    if (adapter.get('status') != 'passed' or adapter.get('exit_code') != 0
            or adapter.get('seed') != 350918 or adapter.get('P01_saved_format_passed') is not True
            or adapter.get('tests_run', 0) < 1
            or any(adapter.get(k) != 0 for k in ('new_worlds', 'new_sensor_packets',
                'new_TSDF_integrations', 'new_quality_evaluations'))):
        raise ValueError('zero-physical adapter/P01 format gate did not pass')
    return inputs


def prepare(review):
    cpu_contract()
    if OUTPUT.exists(): raise FileExistsError('preparation already exists; no overwrite or retry')
    inputs = confirmation_prerequisites(review)
    if not PROTOCOL.is_file() or not PLAN.is_file():
        raise ValueError('frozen confirmation plan and protocol required before preparation')
    from nso.research_evidence_v31 import closure
    from nso.online_planner_v35 import load_public_models_v35
    from nso.observation_belief_v35 import PublicTemplatesV35, CONFIG_V35
    from env.information_pixel_v36 import CONFIRMATION_NOISE_SEED_V36
    import numpy as np
    if CONFIRMATION_NOISE_SEED_V36 != 350918: raise ValueError('preregistered seed differs')
    scene = read(SCENE)
    parents, forecasts, audits = {}, {}, {}
    for parent_id in PARENTS:
        parent = next(p for p in scene['parents'] if p['id'] == parent_id)
        models, prefix = load_public_models_v35(ROOT, parent_id)
        templates = PublicTemplatesV35.from_saved(INFORMATION, parent_id, models[0].poses)
        forecast = [templates.template_information(n) for n in range(len(models[0].poses))]
        audits[parent_id] = public_graph_audit(parent, models, prefix)
        forecasts[parent_id] = forecast
        nav = np.asarray(parent['nav_cells']); low, high = nav.min(axis=0) - .5, nav.max(axis=0) + .5
        shift = np.array([-low[0], -low[1], 0.])
        bounds = np.asarray([box for hypothesis in parent['hypotheses']
            for asset in hypothesis['assets'] for box in asset['boxes']]).reshape(-1, 3, 2)
        public_roi = np.stack((bounds[:, :, 0].min(axis=0), bounds[:, :, 1].max(axis=0))) + shift
        parents[parent_id] = dict(parent=parent_id, noise_model='iid_025px', noise_seed=350918,
            paired_noise_depends_on=['parent', 'paid_step'], noise_independent_of_class=True,
            exact_pose=True, pose_source='observed camera extrinsic and heading; exact simulation odometry',
            mapper='ObservedRuntimeMapperV10', voxel_m=.04, truncation_m=.12,
            camera_euclidean_range_m=4., tsdf_axial_depth_trunc_m=4., laser_range_m=8.,
            public_bounds=public_roi.tolist(), translation=shift.tolist(),
            raster_shape=[int(round((high[1]-low[1])/.2)), int(round((high[0]-low[0])/.2))],
            raster_resolution_m=.2, budget=BUDGET, forced_prefix=PREFIX,
            common_prefix_actions=list(prefix), likelihood_config=CONFIG_V35,
            informative_nodes=[r['node'] for r in forecast if r['informative']],
            inference_performed=False, learned_reconstruction=False)
    physical = []
    for parent_id in PARENTS:
        for h in (0, 1):
            for mode in MODES:
                index = len(physical)
                physical.append(dict(index=index, parent=parent_id, hypothesis=h, mode=mode,
                    episode_id=f'v36-confirmation-{parent_id}-case{index:02d}'))
    cells = [dict(parent=c['parent'], hypothesis=c['hypothesis'], mode=c['mode'], physical_case=c['index'],
        alias=False, acquisition_independent=True, independent_scene_sample=False) for c in physical]
    config = dict(version='v36-frozen-confirmation-1', parents=parents, cells=cells, physical_cases=physical,
        historical_main_used=HISTORICAL_MAIN, maximum_new_main_starts=MAX_STARTS,
        total_main_limit=MAIN_LIMIT, per_case_seconds=SECONDS, per_case_bytes=CASE_CAP,
        shared_bytes=SHARED_CAP, reserve_bytes=RESERVE,
        main_metric='C_map times public-ROI external-vertical surface F1@5cm',
        secondary_thresholds_cm=[2, 10], GT_after_both_prediction_snapshots_frozen=True,
        stop_after_any_final_ineligible_or_failure=True,
        require_previous_case_independent_replay_before_next_main=True,
        confirmation_gate='each parent: mean S-G J5 >0 and mean S-G J2/J10 >=0; all eligibility/replay gates; at least one legal observed semantic action divergence',
        control_scope='unchanged V35 online CPU four-module controller; cross-parent physical and held-out-noise confirmation',
        parent_layouts_wholly_unseen=False, feedback_efficacy_retested=False,
        route_aliasing_allowed=False, offline_policy_or_witness_read=False,
        outside_public_roi_geometry_penalized=False, natural_semantic_network=False)
    OUTPUT.mkdir(); (OUTPUT / 'starts').mkdir()
    try:
        write(OUTPUT / 'config.json', config)
        write(OUTPUT / 'public_graph_audit.json', audits)
        write(OUTPUT / 'public_forecast.json', forecasts)
        seeds = [Path(__file__), SCENE, PLAN, PROTOCOL, OUTPUT / 'config.json',
            OUTPUT / 'public_graph_audit.json', OUTPUT / 'public_forecast.json',
            ROOT / 'env/information_pixel_v36.py', ROOT / 'scripts/verify_online_confirmation_v36.py',
            ROOT / 'docs/research/V36_ADAPTER_STATIC_REVIEW_20260918.md',
            ROOT / 'docs/research/V34_PIXEL_INFORMATION_PROTOCOL_20260918.md',
            ROOT / 'docs/research/V34_SURFACE_MEASUREMENT_CONTRACT_20260918.md']
        gates = (INFORMATION, review, MEASUREMENT, SENSOR, ONLINE_GATE, ONLINE_REVIEW,
            *IMPLEMENTATION_GATES, V35_BATCH, V35_REVIEW, ADAPTER_GATE)
        seeds += [folder / name for folder in gates
            for name in ('result.json', 'manifest.json', 'artifact_hashes.json') if (folder / name).exists()]
        seeds += [ROOT / rel for folder in gates if (folder / 'manifest.json').exists()
            for rel in read(folder / 'manifest.json').get('source_sha256', {})]
        sources = closure(seeds); buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for path in sources: archive.write(path, str(path.relative_to(ROOT)))
        write_bytes(OUTPUT / 'sources.zip', buffer.getvalue())
        write(OUTPUT / 'manifest.json', dict(status='prepared_immutable',
            source_sha256={str(p.relative_to(ROOT)): sha(p) for p in sources},
            source_archive_sha256=sha(OUTPUT / 'sources.zip'), input_sha256=inputs,
            information_review=str(review.relative_to(ROOT)), physical_case_count=MAX_STARTS,
            comparison_cell_count=MAX_STARTS, aliases=[],
            historical_main_used=HISTORICAL_MAIN, maximum_new_main_starts=MAX_STARTS,
            main_limit=MAIN_LIMIT, worlds_during_prepare=0, sensor_queries_during_prepare=0,
            controller_DP_calls_during_prepare=0,
            versions=library_versions(), python=sys.version, preparation_time=time.time()))
        write(OUTPUT / 'prepare_seal.json', {p.name: sha(p) for p in OUTPUT.iterdir() if p.is_file()})
    except BaseException:
        write(OUTPUT / 'prepare_failure.json', dict(error=traceback.format_exc(), worlds=0,
            sensor_queries=0, new_main_tasks=0, partial_preparation_preserved=True))
        raise
    print(json.dumps(dict(prepared=True, physical_cases=MAX_STARTS, comparison_cells=MAX_STARTS,
        new_main_tasks=0, worlds=0)), flush=True)


def frozen():
    from nso.research_evidence_v31 import verify_sources
    manifest = verify_sources(OUTPUT)
    for rel, expected in read(OUTPUT / 'prepare_seal.json').items():
        if sha(OUTPUT / rel) != expected:
            raise ValueError('immutable preparation changed: ' + rel)
    for rel, expected in manifest['input_sha256'].items():
        if sha(ROOT / rel) != expected:
            raise ValueError('prerequisite evidence changed: ' + rel)
    if (OUTPUT / 'prepare_failure.json').exists():
        raise ValueError('preparation failed; cannot execute')
    return manifest, read(OUTPUT / 'config.json')


def save_arrays(path, **arrays):
    import numpy as np
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    write_bytes(path, buffer.getvalue())


def mesh_arrays(mesh):
    import numpy as np
    return {name: np.asarray(getattr(mesh, name)).copy()
        for name in ('vertices', 'triangles', 'vertex_colors')}


def compare_arrays(path, arrays):
    import numpy as np
    from nso.decision_replay_v13 import array_hash
    with np.load(path, allow_pickle=False) as stored:
        if set(stored.files) != set(arrays):
            raise AssertionError('array field set differs: ' + str(path))
        for name, array in arrays.items():
            if array_hash(stored[name]) != array_hash(array):
                raise AssertionError('fresh array bytes differ: ' + str(path) + ':' + name)


def save_full_packet(path, packet):
    from dataclasses import fields
    from nso.decision_replay_v13 import save_packet
    import numpy as np
    # Compression cannot be trusted to shrink arbitrary bytes. This bound also
    # allows NPZ headers and all metadata before the old exclusive-create IO.
    upper_bound = 65536 + sum(np.asarray(getattr(record, f.name)).nbytes
        for record in (packet.frame, packet.scan) for f in fields(record))
    guard_write(path, upper_bound)
    save_packet(path, packet)
    guard_write(path)


def nonsemantic_fields(packet):
    import numpy as np
    from env.information_pixel_v34 import nonsemantic_rgb
    from nso.decision_replay_v13 import array_hash
    values = dict(depth=packet.frame.depth_m, rgb_nonsemantic=nonsemantic_rgb(packet.frame.color_rgb),
        intrinsic=packet.frame.intrinsic, camera_pose=packet.frame.world_from_camera,
        ranges=packet.scan.ranges_m, laser_pose=packet.scan.world_from_laser,
        scan_calibration=np.asarray([packet.scan.angle_min_rad, packet.scan.angle_increment_rad,
            packet.scan.range_max_m]), cell=np.asarray(packet.position, dtype=np.int64),
        heading=np.asarray(packet.heading, dtype=np.int64))
    return {key: array_hash(value) for key, value in values.items()}


def require_previous_cases(config, index):
    directories = case_directories()
    if index != len(directories) or len(directories) >= min(MAX_STARTS, len(config['physical_cases'])):
        raise ValueError('fixed acquisition order, no-retry rule or main quota violated')
    if HISTORICAL_MAIN + len(directories) + 1 > MAIN_LIMIT:
        raise ValueError('historical 27/36 main quota exceeded')
    for previous in range(index):
        folder = OUTPUT / f'case{previous:02d}'
        verify_seal(folder, 'main_seal.json', 'replay')
        if ((folder / 'failure.json').exists() or (folder / 'stop.json').exists()
                or not read(folder / 'result.json')['stages']['final']['measurement']['eligible']):
            raise ValueError('retained prior failure/ineligible result stops all remaining acquisitions')
        verify_seal(folder / 'replay', 'seal.json')
        receipt = read(folder / 'replay' / 'result.json')
        if receipt.get('passed') is not True or receipt.get('main_result_sha256') != sha(folder / 'result.json'):
            raise ValueError('previous fresh-process replay is missing or failed')


def process_identity():
    # PID alone may repeat across the tool's PID namespaces. Pair Linux start
    # identity with invocation time; nothing here enters policy or replay data.
    stat = Path('/proc/self/stat').read_text().split(') ', 1)[1].split()
    return dict(pid=os.getpid(), process_start_ticks=stat[19],
        pid_namespace=os.readlink('/proc/self/ns/pid'),
        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        invocation_unix_ns=time.time_ns(), invocation_monotonic_ns=time.monotonic_ns())


PROCESS_IDENTITY = process_identity()


def same_process(left, right):
    return all(left[key] == right[key] for key in
        ('pid', 'process_start_ticks', 'pid_namespace', 'boot_id'))


def observed_pose(packet, translation):
    import numpy as np
    xy = np.asarray(packet.frame.world_from_camera[:2, 3]) - np.asarray(translation[:2])
    rounded = np.rint(xy)
    if not np.allclose(xy, rounded, atol=1e-9, rtol=0.):
        raise ValueError('observed camera odometry is off the declared metre-grid')
    return (int(rounded[0]), int(rounded[1]), int(packet.heading))


def controller_evidence(controller, history, actions):
    return dict(summary=controller.summary(), calls=controller.calls,
        plans=controller.plans, history=history, actions=actions,
        observation_boundary='RGB-D/ranges, observed camera pose/heading, paid step/action and observed map digest',
        scene_episode_hypothesis_to_controller=False, ground_truth_or_metric_to_controller=False)


def paired_prefix_trace_check(trace, parent_id, config):
    checked = []
    cases = {c['index']: c for c in config['physical_cases']}
    for previous in case_directories():
        if cases[int(previous.name[4:])]['parent'] != parent_id: continue
        path = previous / 'trace.json'
        if not path.exists(): continue
        prior = read(path)
        equal = len(prior) > PREFIX and len(trace) > PREFIX and all(
            prior[n]['nonsemantic'] == trace[n]['nonsemantic'] for n in range(PREFIX+1))
        checked.append(dict(case=int(previous.name[4:]), paired_nonsemantic_prefix_equal=equal))
    return dict(comparisons=checked, passed=all(row['paired_nonsemantic_prefix_equal'] for row in checked))


def run_case(index, replay=False):
    cpu_contract(); require_not_finalized()
    manifest, config = frozen()
    if index is None or index not in range(len(config['physical_cases'])):
        raise ValueError('index must identify a declared online physical case')
    case = config['physical_cases'][index]; folder = OUTPUT / f'case{index:02d}'
    acquisition = config['parents'][case['parent']]
    if replay:
        verify_seal(folder, 'main_seal.json', 'replay')
        if (folder / 'failure.json').exists():
            raise ValueError('incomplete acquisition cannot claim a full independent replay')
        expected = read(folder / 'result.json'); main_start = read(folder / 'started.json')
        if same_process(main_start['process_identity'], PROCESS_IDENTITY):
            raise ValueError('replay must be a fresh process invocation')
        work = folder / 'replay'; guard_write(work / 'started.json', 4096)
        work.mkdir()
        write(work / 'started.json', dict(process_identity=PROCESS_IDENTITY,
            main_process_identity=main_start['process_identity'], independent_invocation=True))
    else:
        require_previous_cases(config, index)
        guard_write(folder / 'started.json', 8192); folder.mkdir()
        work = folder
        start = dict(case=index, process_identity=PROCESS_IDENTITY,
            cumulative_main_used=HISTORICAL_MAIN + len(case_directories()), counted_on_directory_creation=True)
        write(OUTPUT / 'starts' / f'main_{index:02d}.json', start)
        write(folder / 'started.json', start); (folder / 'packets').mkdir()
    started = time.monotonic()
    counts = dict(worlds=0, sensor_packets=0, clean_depth_queries=0, scan_queries=0,
        paid_actions=0, mapper_updates=0, TSDF_integrations=0, mapper_mesh_extractions=0,
        evaluation_stages=0, learned_reconstruction_calls=0, controllers=0,
        controller_observations=0, online_planning_calls=0, online_planning_attempts=0,
        online_planner_memo_states=0, IGCR_updates=0, new_geometry_pose_updates=0)
    trace, snapshots, history, actions = [], {}, [], []
    world = controller = None
    def deadline(*_): raise TimeoutError('declared 600 second whole-case limit')
    signal.signal(signal.SIGALRM, deadline); signal.alarm(SECONDS)
    try:
        import numpy as np
        from env.information_pixel_v34 import cue_from_rgb, sensor_counts_v34
        from env.information_pixel_v36 import InformationPixelWorldV36
        from nso.decision_replay_v13 import array_hash, load_packet
        from nso.observed_runtime_mapper_v10 import ObservedRuntimeMapperV10
        from nso.sensor_contract_v34 import validate_sensor_packet_v34
        from nso.surface_measurement_v34 import extract_observed_asset_mesh, SurfaceMeasurementV34
        from nso.online_planner_v35 import load_public_models_v35
        from nso.observation_belief_v35 import PublicTemplatesV35
        from nso.cpu_four_modules_v35 import CPUFourModuleControllerV35, ObservationV35
        if any(sensor_counts_v34().values()): raise ValueError('one fresh process per main/replay required')
        models, prefix = load_public_models_v35(ROOT, case['parent'])
        templates = PublicTemplatesV35.from_saved(INFORMATION, case['parent'], models[0].poses)
        controller = CPUFourModuleControllerV35(models, templates, prefix, mode=case['mode'],
            total_budget=BUDGET, informative_nodes=acquisition['informative_nodes'])
        counts['controllers'] = 1
        # Hidden scenario metadata stays entirely on the sensor/evaluation side.
        world = InformationPixelWorldV36(case['parent'], case['hypothesis'], case['episode_id'],
            noise_model=acquisition['noise_model'])
        counts['worlds'] = 1
        if (world.noise_seed != acquisition['noise_seed'] or world.config.voxel_m != .04
                or world.config.max_depth_m != 4. or list(world.shape) != acquisition['raster_shape']
                or list(world.shift) != acquisition['translation']):
            raise ValueError('execution differs from frozen calibration')
        mapper = ObservedRuntimeMapperV10(world.shape, world.config, truncation_m=.12)
        action = None; collisions = 0
        for paid in range(BUDGET+1):
            if time.monotonic() - started >= SECONDS: raise TimeoutError('declared whole-case time cap')
            packet = world.packet() if paid == 0 else world.step(action)
            counts.update(worlds=world.counts['worlds'], sensor_packets=world.counts['packets'],
                clean_depth_queries=world.counts['clean_depth_queries'], scan_queries=world.counts['scan_queries'],
                paid_actions=world.counts['step_calls'])
            path = folder / 'packets' / f'{paid:03d}.npz'
            # Preserve the generated main packet before calibration, action,
            # collision or replay comparisons can reject the observation.
            if not replay: save_full_packet(path, packet)
            errors = []; pose = None
            try:
                validate_sensor_packet_v34(packet, world.transform, world.config)
                pose = observed_pose(packet, acquisition['translation'])
            except Exception as error:
                errors.append('invalid observed calibration/odometry: ' + repr(error))
            if packet.action_id != paid or packet.action != action:
                errors.append('paid action metadata differs from online issued action')
            if packet.collision: errors.append('actual execution collision')
            if np.any(packet.frame.semantic): errors.append('semantic/GT labels must not enter mapper')
            if replay and not errors:
                stored = validate_sensor_packet_v34(load_packet(path), world.transform, world.config)
                if packet.sha256() != stored.sha256(): errors.append('fresh full packet field bytes differ')
            if errors:
                if replay:
                    path = work / f'abnormal_packet_{paid:03d}.npz'; save_full_packet(path, packet)
                receipt = dict(paid=paid, expected_action=action, action=packet.action,
                    action_id=packet.action_id, observed_pose=pose, collision=packet.collision,
                    packet_sha256=packet.sha256(), saved_packet=str(path.relative_to(folder)),
                    mapper_updated=False, errors=errors)
                write(work / f'abnormal_action_{paid:03d}.json', receipt); trace.append(receipt)
                raise AssertionError('; '.join(errors))
            mapper.update(packet.frame, packet.scan)
            counts['mapper_updates'] += 1; counts['TSDF_integrations'] += 1
            collisions += int(packet.collision)
            trace.append(dict(paid=paid, action=action, pose_v33=list(pose),
                position=packet.position, heading=packet.heading, packet_sha256=packet.sha256(),
                nonsemantic=nonsemantic_fields(packet), clean_depth_sha256=array_hash(world.last_clean_depth),
                cue=cue_from_rgb(packet.frame), collision=packet.collision, collisions_so_far=collisions,
                footprint_conflict=mapper.current_footprint_conflict,
                footprint_conflict_details=mapper.current_footprint_conflict_details,
                measured_map_sha256=array_hash(mapper.belief)))
            observation = ObservationV35(frame_id=f'opaque-{paid}', step=paid, pose=pose,
                action=action, depth=packet.frame.depth_m, rgb=packet.frame.color_rgb,
                ranges=packet.scan.ranges_m, collision=packet.collision,
                measured_map_sha256=array_hash(mapper.belief))
            controller.accept(observation)
            counts['controller_observations'] += 1; counts['IGCR_updates'] += 1
            counts['new_geometry_pose_updates'] += int(controller.posterior_receipts[-1]['new_geometry_pose'])
            before_plans = len(controller.plans)
            if paid >= PREFIX and controller.terminal_reason is None:
                counts['online_planning_attempts'] += 1
            following = controller.next_action()
            for plan in controller.plans[before_plans:]:
                if plan['planning']['phase'] == 'online_belief_global_planning':
                    counts['online_planning_calls'] += 1
                    counts['online_planner_memo_states'] += plan['planning']['memo_states']
            history.append(dict(step=paid, node=controller.state['node'], pose=list(pose),
                posterior=controller.posterior_receipts[-1], state=controller.state, next_action=following))
            trace[-1]['next_action'] = following
            for stage, take in (('prefix', paid == PREFIX), ('final', following is None)):
                if not take: continue
                raw = mapper.mesh(); counts['mapper_mesh_extractions'] += 1
                predicted, crop = extract_observed_asset_mesh(raw, acquisition['public_bounds'])
                maps = {key: value.copy() for key, value in vars(mapper).items() if isinstance(value, np.ndarray)}
                if 'belief' not in maps: raise AssertionError('mapper snapshot has no belief array')
                snapshots[stage] = dict(mesh=predicted, maps=maps, crop=crop, paid=paid)
                for name, arrays in ((f'{stage}_raw.npz', mesh_arrays(raw)),
                        (f'{stage}_extracted.npz', mesh_arrays(predicted)), (f'{stage}_maps.npz', maps)):
                    if replay: compare_arrays(folder / name, arrays)
                    else: save_arrays(folder / name, **arrays)
                if replay:
                    if digest(crop) != digest(read(folder / f'{stage}_crop.json')):
                        raise AssertionError('fresh extraction audit differs')
                else: write(folder / f'{stage}_crop.json', crop)
                print(json.dumps(dict(case=index, replay=replay, stage=stage, paid=paid,
                    seconds=time.monotonic()-started)), flush=True)
            if following is None: break
            actions.append(following); action = following
        if set(snapshots) != {'prefix', 'final'}:
            raise AssertionError('both common-prefix and terminal prediction snapshots are required')
        evidence = controller_evidence(controller, history, actions)
        if replay:
            if digest(trace) != digest(read(folder / 'trace.json')):
                raise AssertionError('fresh complete sensor/action trace differs')
            if digest(evidence) != digest(read(folder / 'controller.json')):
                raise AssertionError('fresh controller posterior/planning/module evidence differs')
            frozen_predictions = read(folder / 'prediction_freeze.json')
            for rel, expected_hash in frozen_predictions['artifact_sha256'].items():
                if sha(folder / rel) != expected_hash: raise AssertionError('frozen prediction artifact changed')
            write(work / 'prediction_comparison.json', dict(passed=True,
                main_prediction_freeze_sha256=sha(folder / 'prediction_freeze.json'),
                packet_count=len(trace), both_raw_and_extracted_meshes_equal=True,
                all_mapper_arrays_equal=True, all_trace_fields_equal=True,
                all_controller_decisions_and_receipts_equal=True))
        else:
            write(folder / 'trace.json', trace); write(folder / 'controller.json', evidence)
            write(folder / 'prediction_freeze.json', dict(before_first_quality_or_GT_reference_access=True,
                artifact_sha256={str(p.relative_to(folder)): sha(p) for p in files_under(folder)},
                stages=['prefix', 'final'], packet_count=len(trace),
                public_roi_sha256=digest(acquisition['public_bounds']),
                controller_decisions_frozen=True))
        # The evaluation reference is accessed only after ALL predictions and
        # actual online decisions have been frozen; the controller is finished.
        floor = world.evaluation_floor(); reachable, grid = floor['reachable'], floor['declared_grid_cells']
        floor_arrays = dict(safe=floor['safe'], reachable=reachable, declared_grid_cells=grid)
        if replay: compare_arrays(folder / 'evaluation_floor.npz', floor_arrays)
        else: save_arrays(folder / 'evaluation_floor.npz', **floor_arrays)
        evaluator = SurfaceMeasurementV34(world.instance_mesh(0), world.instance_mesh(0, vertical_only=True))
        stages = {}
        for stage, snapshot in snapshots.items():
            belief, stage_paid = snapshot['maps']['belief'], snapshot['paid']
            C_map = float(np.mean(belief[reachable] != -1))
            C_grid = float(np.mean(belief[grid[:, 0], grid[:, 1]] != -1))
            returned = tuple(trace[stage_paid]['pose_v33']) == models[0].poses[models[0].anchor]
            measurement = evaluator.evaluate(snapshot['mesh'], C_map, returned=returned,
                collisions=trace[stage_paid]['collisions_so_far'], failed=False,
                paid_actions=stage_paid, budget=BUDGET)
            counts['evaluation_stages'] += 1
            stages[stage] = dict(measurement=measurement, C_grid_measured=C_grid,
                C_grid_denominator=len(grid), full_reachable_raster_denominator=int(reachable.sum()),
                belief_sha256=array_hash(belief), crop_audit_sha256=digest(snapshot['crop']))
        recognized = [r for r in trace[:PREFIX+1] if r['cue']['class_id'] == case['hypothesis']+2]
        cue_positions = sorted({tuple(r['pose_v33'][:2]) for r in recognized})
        contradictory = [r['paid'] for r in trace[:PREFIX+1]
            if r['cue']['class_id'] not in (None, case['hypothesis']+2)]
        if replay:
            prefix_pairing = expected['prefix_pairing_at_acquisition']
        else:
            prefix_pairing = paired_prefix_trace_check(trace, case['parent'], config)
        summary = controller.summary()
        result = dict(physical_case=case, mode=case['mode'], stages=stages, counts=counts,
            collisions=collisions, returned=tuple(trace[-1]['pose_v33']) == models[0].poses[models[0].anchor],
            paid_actions=paid, budget=BUDGET,
            prefix_cue=dict(distinct_recognized_positions=cue_positions,
                contradictory_actions=contradictory, passed=len(cue_positions) >= 2 and not contradictory),
            prefix_pairing_at_acquisition=prefix_pairing,
            controller_summary=summary, controller_sha256=digest(evidence),
            all_four_modules_called=set(summary['modules_called']) == {'OV-SDF','STGHP','RPN-UQ','IGCR'},
            footprint_conflict_count=mapper.current_footprint_conflict_count,
            trajectory_sha256=digest([{k:r[k] for k in ('paid','action','pose_v33')} for r in trace]),
            trace_sha256=digest(trace), inference_performed=False, exact_simulated_pose_not_estimated_SLAM=True,
            adaptive_policy_executed=True, full_architecture_executed=False,
            original_quota=HISTORICAL_MAIN, cumulative_main_used=HISTORICAL_MAIN+index+1,
            prediction_scope='public union ROI; errors outside padded ROI do not affect precision')
        frozen()
        if replay:
            if digest(result) != digest(expected):
                write(work / 'mismatch_result.json', result)
                raise AssertionError('fresh reconstruction/controller/measurement result differs')
            write(work / 'result.json', dict(passed=True, all_packet_field_bytes_equal=True,
                all_raw_extracted_mesh_map_array_bytes_equal=True,
                all_controller_decisions_and_receipts_equal=True, complete_result_equal=True,
                packet_count=len(trace), main_result_sha256=sha(folder / 'result.json'),
                main_process_identity=read(folder / 'started.json')['process_identity'],
                replay_process_identity=PROCESS_IDENTITY,
                fresh_process_identity_verified=True, counts=counts, elapsed_seconds=time.monotonic()-started))
            seal(work, 'seal.json')
        else:
            write(folder / 'result.json', result)
            if (not stages['final']['measurement']['eligible'] or not result['prefix_cue']['passed'] or
                    not prefix_pairing['passed'] or not result['all_four_modules_called']):
                write(folder / 'stop.json', dict(stop_remaining_acquisitions=True,
                    reason='final qualification, RGB cue, paired prefix or module contract failed',
                    actual_C80_passed=stages['final']['measurement']['C_map'] >= .8,
                    result_sha256=sha(folder / 'result.json'), negative_result_retained=True))
            seal(folder, 'main_seal.json', 'replay')
        print(json.dumps(dict(case=index, replay=replay, complete=True,
            eligible=stages['final']['measurement']['eligible'], seconds=time.monotonic()-started)), flush=True)
    except BaseException:
        signal.alarm(0)
        if world is not None:
            counts.update(worlds=world.counts['worlds'], sensor_packets=world.counts['packets'],
                clean_depth_queries=world.counts['clean_depth_queries'], scan_queries=world.counts['scan_queries'],
                paid_actions=world.counts['step_calls'])
        write(work / 'failure.json', dict(error=traceback.format_exc(), counts=counts,
            counted_main_directories=len(case_directories()), cumulative_main_used=HISTORICAL_MAIN+len(case_directories()),
            retry_allowed=False, stop_remaining_acquisitions=True, elapsed_seconds=time.monotonic()-started,
            failed_planner_attempt_may_have_partial_unreported_memo_states=True))
        write(work / 'partial_trace.json', trace)
        if controller is not None:
            write(work / 'partial_controller.json', controller_evidence(controller, history, actions))
        if not (work / ('seal.json' if replay else 'main_seal.json')).exists():
            seal(work, 'seal.json' if replay else 'main_seal.json', None if replay else 'replay')
        raise
    finally:
        signal.alarm(0)


def analyze():
    cpu_contract()
    require_not_finalized()
    _, config = frozen()
    by_index, traces, replay_receipts, failures = {}, {}, {}, []
    main_counts, replay_counts = {}, {}
    for folder in case_directories():
        index = int(folder.name[4:])
        if not (folder / 'main_seal.json').exists():
            failures.append(dict(case=index, kind='main_unsealed',
                reason='counted directory exists but no completed seal; evidence preserved, counters unknown'))
            continue
        verify_seal(folder, 'main_seal.json', 'replay')
        if (folder / 'result.json').exists():
            by_index[index] = read(folder / 'result.json')
            main_counts[index] = by_index[index]['counts']
            traces[index] = read(folder / 'trace.json')
            replay = folder / 'replay'
            if (not (folder / 'failure.json').exists()
                    and (not (replay / 'seal.json').exists()
                        or not any((replay / name).exists() for name in ('result.json', 'failure.json')))):
                raise ValueError('every complete main case, including ineligible cases, needs a terminal sealed replay before analysis')
        if (folder / 'failure.json').exists():
            failure = read(folder / 'failure.json')
            main_counts[index] = failure['counts']
            failures.append(dict(case=index, kind='main', evidence=failure))
        replay = folder / 'replay'
        if replay.exists():
            if not (replay / 'seal.json').exists():
                failures.append(dict(case=index, kind='replay_unsealed',
                    reason='replay directory exists but no completed seal; counters unknown'))
                continue
            verify_seal(replay, 'seal.json')
            if (replay / 'result.json').exists():
                receipt = read(replay / 'result.json')
                if receipt.get('main_result_sha256') != sha(folder / 'result.json'):
                    raise ValueError('replay receipt identifies a different main result')
                replay_receipts[index] = receipt
                replay_counts[index] = receipt['counts']
            if (replay / 'failure.json').exists():
                failure = read(replay / 'failure.json')
                replay_counts[index] = failure['counts']
                failures.append(dict(case=index, kind='replay', evidence=failure))
    incomplete = len(by_index) != len(config['physical_cases'])
    stopped = bool(failures or any(not r['stages']['final']['measurement']['eligible']
        or not r['prefix_cue']['passed'] or not r['prefix_pairing_at_acquisition']['passed']
        or not r['all_four_modules_called'] for r in by_index.values()))
    if incomplete and not stopped:
        raise ValueError('analysis must await all declared cases or a retained stop condition')
    table = []
    for cell in config['cells']:
        result = by_index.get(cell['physical_case'])
        table.append(dict(**cell, measured=result is not None,
            stages=result['stages'] if result is not None else None,
            controller_summary=result['controller_summary'] if result is not None else None,
            prefix_to_final_joint_change={threshold: result['stages']['final']['measurement'][threshold]['joint']
                - result['stages']['prefix']['measurement'][threshold]['joint']
                for threshold in ('02cm', '05cm', '10cm')} if result is not None else None,
            independent_replay_passed=replay_receipts.get(cell['physical_case'], {}).get('passed', False)))
    pairings = []
    for parent_id in PARENTS:
        for mode in MODES:
            cells = [next(c for c in config['cells'] if c['parent'] == parent_id
                and c['hypothesis'] == h and c['mode'] == mode) for h in (0, 1)]
            left, right = [traces.get(c['physical_case']) for c in cells]
            first = None
            if left is not None and right is not None:
                first = {key: next((i for i, (a, b) in enumerate(zip(left, right))
                    if a['nonsemantic'][key] != b['nonsemantic'][key]), None) for key in left[0]['nonsemantic']}
            pairings.append(dict(parent=parent_id, mode=mode,
                physical_cases=[c['physical_case'] for c in cells], first_observed_field_difference=first,
                actual_prefix_nonsemantic_equal=None if first is None else all(v is None or v > PREFIX for v in first.values())))
    comparisons, overall = {}, {}
    if not incomplete:
        for parent_id in PARENTS:
            comparisons[parent_id] = {}
            for threshold in ('02cm', '05cm', '10cm'):
                values = {(c['hypothesis'], c['mode']): by_index[c['physical_case']]['stages']['final']['measurement'][threshold]['joint']
                    for c in config['cells'] if c['parent'] == parent_id}
                means = {mode: sum(values[h, mode] for h in (0, 1)) / 2 for mode in MODES}
                deltas = [values[h, 'S'] - values[h, 'G'] for h in (0, 1)]
                average = sum(deltas) / 2
                comparisons[parent_id][threshold] = dict(mean_joint=means, S_vs_G=dict(
                    treatment='S', baseline='G', paired_joint_differences=deltas,
                    paired_mean_joint_difference=average,
                    relative_mean_joint_difference=average / means['G'] if means['G'] else None))
        for threshold in ('02cm', '05cm', '10cm'):
            means = {mode: sum(comparisons[p][threshold]['mean_joint'][mode] for p in PARENTS)/2 for mode in MODES}
            deltas = [comparisons[p][threshold]['S_vs_G']['paired_mean_joint_difference'] for p in PARENTS]
            average = sum(deltas)/2
            overall[threshold] = dict(mean_joint=means, S_vs_G=dict(
                treatment='S', baseline='G', per_parent_mean_joint_differences=deltas,
                paired_mean_joint_difference=average,
                relative_mean_joint_difference=average/means['G'] if means['G'] else None))
    all_replayed = len(replay_receipts) == len(config['physical_cases']) and all(r['passed'] for r in replay_receipts.values())
    eligible = not incomplete and all(r['stages']['final']['measurement']['eligible'] for r in by_index.values())
    pairing_ok = all(p['actual_prefix_nonsemantic_equal'] is True for p in pairings) and all(
        r['prefix_pairing_at_acquisition']['passed'] for r in by_index.values())
    cue_ok = not incomplete and all(r['prefix_cue']['passed'] for r in by_index.values())
    modules_ok = not incomplete and all(r['all_four_modules_called'] for r in by_index.values())
    prerequisites = eligible and all_replayed and pairing_ok and cue_ok and modules_ok and not failures
    parent_score_gates = {p: bool(p in comparisons
        and comparisons[p]['05cm']['S_vs_G']['paired_mean_joint_difference'] > 0
        and all(comparisons[p][t]['S_vs_G']['paired_mean_joint_difference'] >= 0 for t in ('02cm','10cm')))
        for p in PARENTS}
    result = dict(status='stopped' if stopped else 'complete', table=table,
        actual_noisy_prefix_pairings=pairings, prescribed_paired_comparison=comparisons,
        overall_comparison=overall, per_parent_score_gates=parent_score_gates,
        full_matrix_collected=not incomplete, all_final_eligible=eligible, all_replays_passed=all_replayed,
        actual_C80_verified=not incomplete and all(r['stages']['final']['measurement']['C_map'] >= .8 for r in by_index.values()),
        controlled_marker_prerequisite_passed=cue_ok, all_prefix_nonsemantic_pairings_passed=pairing_ok,
        all_four_modules_called=modules_ok, qualification_and_replay_gate_passed=prerequisites,
        measured_confirmation_score_gate_passed=prerequisites and all(parent_score_gates.values()),
        overall_online_confirmation_gate_passed=None,
        overall_gate_pending='independent saved-evidence arithmetic, per-parent effect and observed semantic action divergence audit; no extra DP or physical calls',
        failures=failures, new_main_tasks=len(case_directories()), total_main_used=HISTORICAL_MAIN+len(case_directories()),
        main_limit=MAIN_LIMIT, unique_physical_cases_declared=len(config['physical_cases']), comparison_cells=MAX_STARTS,
        route_aliasing_used=False, parent_layout_count=2, parent_layouts_wholly_unseen=False,
        independent_scene_sample_count=0, feedback_efficacy_retested=False,
        full_architecture_advantage_proven=False, natural_semantic_innovation_proven=False,
        controlled_online_CPU_four_module_executed=not incomplete,
        pose_scope='exact synthetic odometry; no pose-estimation SLAM evaluated',
        metric_scope='single-facility public union ROI; outside-ROI erroneous geometry excluded from precision',
        acquisition_counts={key:sum(r[key] for r in main_counts.values()) for key in next(iter(main_counts.values()))} if main_counts else {},
        replay_counts={key:sum(r[key] for r in replay_counts.values()) for key in next(iter(replay_counts.values()))} if replay_counts else {},
        counter_accounting_complete=all(not f['kind'].endswith('_unsealed') for f in failures))
    write(OUTPUT / 'result.json', result); seal(OUTPUT, 'final_seal.json')
    print(json.dumps(dict(status=result['status'], new_main_tasks=result['new_main_tasks'],
        measured_confirmation_score_gate_passed=result['measured_confirmation_score_gate_passed'],
        per_parent_score_gates=parent_score_gates, overall_online_confirmation_gate_passed=None)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'main', 'replay', 'analyze'))
    parser.add_argument('--index', type=int, choices=range(MAX_STARTS), help='unique physical case, not comparison cell')
    parser.add_argument('--information-review', type=Path, default=INFORMATION_REVIEW,
        help='sealed independent pixel review directory; frozen by prepare')
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.information_review.resolve())
    elif args.command in ('main', 'replay'):
        run_case(args.index, replay=args.command == 'replay')
    else:
        analyze()


if __name__ == '__main__':
    main()
