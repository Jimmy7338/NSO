#!/usr/bin/env python3
"""Preregistered V34 fixed routes: prepare, main, independent replay, analyze.

Preparation uses saved witnesses and configuration arithmetic only. Nothing is
executed by import or --help. All artifacts are exclusive-create; a failed main
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
OUTPUT = ROOT / 'audit_results/v34_fixed_measurement_20260918'
SCENE = ROOT / 'configs/virtual3d/v33_direction_scene_r1_20260917.json'
INFORMATION = ROOT / 'audit_results/v34_pixel_information_20260918'
INFORMATION_REVIEW = ROOT / 'audit_results/v34_pixel_information_review_20260918'
MEASUREMENT = ROOT / 'audit_results/surface_measurement_v34_unit_20260918'
SENSOR = ROOT / 'audit_results/v34_sensor_contract_tests_20260918'
PROTOCOL = ROOT / 'docs/research/V34_FIXED_MEASUREMENT_PROTOCOL_20260918.md'
MIB = 1024 ** 2
RESERVE = 64 * MIB
CASE_CAP = 40 * MIB
SHARED_CAP = 2 * MIB
BUDGET = 42
PREFIX = 18
HISTORICAL_MAIN = 16
MAIN_LIMIT = 36
MAX_STARTS = 4
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


def static_route(parent, hypothesis, actions):
    """V33 exact axis-aligned swept-footprint arithmetic, never a World."""
    from env.information_pixel_v34 import swept_clear_v34
    boxes = [box for asset in parent['hypotheses'][hypothesis]['assets'] for box in asset['boxes']]
    boxes += parent['background_boxes']
    nav = {tuple(p) for p in parent['nav_cells']}
    pose = list(parent['anchor'])
    poses = [pose.copy()]
    for action in actions:
        start = pose[:2]
        if action == 'forward':
            dx, dy = HEADINGS[pose[2]]
            pose[0] += dx
            pose[1] += dy
        elif action in ('left', 'right'):
            pose[2] = (pose[2] + (1 if action == 'right' else -1)) % 4
        else:
            raise ValueError('unsupported frozen primitive')
        if tuple(pose[:2]) not in nav or not swept_clear_v34(start, pose[:2], boxes):
            raise ValueError('frozen route fails exact swept-footprint contract')
        poses.append(pose.copy())
    if not PREFIX <= len(actions) <= BUDGET or poses[PREFIX] != poses[0] or poses[-1] != poses[0]:
        raise ValueError('18-prefix, 42-budget or exact-return contract failed')
    return poses


def prepare(review):
    cpu_contract()
    if OUTPUT.exists():
        raise FileExistsError('preparation already exists; no overwrite or retry')
    inputs = gate_inputs(review)
    if not PROTOCOL.is_file():
        raise ValueError('fixed measurement protocol must exist before preparation')
    from nso.research_evidence_v31 import closure
    import numpy as np
    scene = read(SCENE)
    parent = next(p for p in scene['parents'] if p['id'] == 'P00')
    policies = read(INFORMATION / 'P00_policies.json')
    if not policies['screening_passed'] or not policies['exact_search_completed']:
        raise ValueError('P00 witnesses are not certified')
    nav = np.asarray(parent['nav_cells'])
    low, high = nav.min(axis=0) - .5, nav.max(axis=0) + .5
    shift = np.array([-low[0], -low[1], 0.])
    bounds = np.asarray([box for hypothesis in parent['hypotheses']
        for asset in hypothesis['assets'] for box in asset['boxes']]).reshape(-1, 3, 2)
    public_roi = np.stack((bounds[:, :, 0].min(axis=0), bounds[:, :, 1].max(axis=0))) + shift
    acquisition = dict(parent='P00', noise_model='iid_025px', noise_seed=1901,
        paired_noise_depends_on=['parent', 'paid_step'], noise_independent_of_class=True,
        exact_pose=True, pose_source='simulator-exact-discrete-pose',
        mapper='ObservedRuntimeMapperV10', voxel_m=.04, truncation_m=.12,
        camera_euclidean_range_m=4., tsdf_axial_depth_trunc_m=4., laser_range_m=8.,
        public_bounds=public_roi.tolist(), translation=shift.tolist(),
        raster_shape=[int(round((high[1]-low[1])/.2)), int(round((high[0]-low[0])/.2))],
        raster_resolution_m=.2, budget=BUDGET, forced_prefix=PREFIX,
        inference_performed=False, learned_reconstruction=False)
    physical, cells, traces, identities = [], [], [], {}
    for h in (0, 1):
        for policy in ('G', 'class_oracle'):
            candidates = [w for w in policies['witnesses']
                if w['actual_hypothesis'] == h and w['policy'] == policy]
            if len(candidates) != 1:
                raise ValueError('missing or duplicated declared witness')
            witness = candidates[0]
            actions = parent['prefix_actions'] + witness['suffix_actions']
            if len(parent['prefix_actions']) != PREFIX or len(actions) != witness['total_paid_actions']:
                raise ValueError('saved witness action accounting changed')
            poses = static_route(parent, h, actions)
            if poses[PREFIX:] != witness['suffix_poses']:
                raise ValueError('saved witness poses disagree with primitive simulation')
            identity = digest(dict(hypothesis=h, actions=actions, acquisition=acquisition))
            duplicate = identity in identities
            if not duplicate:
                index = len(physical)
                identities[identity] = index
                physical.append(dict(index=index, parent='P00', hypothesis=h,
                    episode_id=f'v34-fixed-P00-case{index:02d}', actions=actions,
                    poses=poses, acquisition_identity_sha256=identity))
            index = identities[identity]
            cell = dict(cell_index=len(cells), hypothesis=h, policy=policy,
                physical_case=index, alias=duplicate, independent_sample=not duplicate,
                witness_sha256=digest(witness), route_origin='offline finite pixel-model witness')
            cells.append(cell)
            traces.append(dict(**cell, actions=actions, poses=poses))
    if len(physical) > MAX_STARTS or len(cells) != 4:
        raise ValueError('declared four-cell quota violated')
    config = dict(version='v34-fixed-measurement-1', acquisition=acquisition,
        cells=cells, physical_cases=physical, source_policy_sha256=sha(INFORMATION / 'P00_policies.json'),
        historical_main_used=HISTORICAL_MAIN, maximum_new_main_starts=MAX_STARTS,
        total_main_limit=MAIN_LIMIT, per_case_seconds=SECONDS, per_case_bytes=CASE_CAP,
        shared_bytes=SHARED_CAP, reserve_bytes=RESERVE,
        main_metric='C_map times public-ROI external-vertical surface F1@5cm',
        secondary_thresholds_cm=[2, 10], GT_after_both_prediction_snapshots_frozen=True,
        stop_after_any_final_ineligible_or_failure=True,
        require_previous_case_independent_replay_before_next_main=True,
        fixed_route_scope='offline witnesses; not adaptive measured G, ANS or full architecture',
        outside_public_roi_geometry_penalized=False, natural_semantic_network=False)
    OUTPUT.mkdir()
    (OUTPUT / 'starts').mkdir()
    try:
        write(OUTPUT / 'config.json', config)
        write(OUTPUT / 'static_routes.json', traces)
        seeds = [Path(__file__), SCENE, PROTOCOL, OUTPUT / 'config.json', OUTPUT / 'static_routes.json',
            INFORMATION / 'P00_policies.json',
            ROOT / 'docs/research/V34_PIXEL_INFORMATION_PROTOCOL_20260918.md',
            ROOT / 'docs/research/V34_SURFACE_MEASUREMENT_CONTRACT_20260918.md',
            ROOT / 'docs/research/V34_MEASUREMENT_GATE_PLAN_20260917.md']
        seeds += [folder / name for folder in (INFORMATION, review, MEASUREMENT, SENSOR)
            for name in ('result.json', 'manifest.json', 'artifact_hashes.json') if (folder / name).exists()]
        seeds += [ROOT / rel for folder in (INFORMATION, review, MEASUREMENT, SENSOR)
            if (folder / 'manifest.json').exists()
            for rel in read(folder / 'manifest.json').get('source_sha256', {})]
        sources = closure(seeds)
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for path in sources:
                archive.write(path, str(path.relative_to(ROOT)))
        write_bytes(OUTPUT / 'sources.zip', buffer.getvalue())
        write(OUTPUT / 'manifest.json', dict(status='prepared_immutable',
            source_sha256={str(p.relative_to(ROOT)): sha(p) for p in sources},
            source_archive_sha256=sha(OUTPUT / 'sources.zip'), input_sha256=inputs,
            information_review=str(review.relative_to(ROOT)), physical_case_count=len(physical),
            comparison_cell_count=4, aliases=[c for c in cells if c['alias']],
            historical_main_used=HISTORICAL_MAIN, maximum_new_main_starts=MAX_STARTS,
            main_limit=MAIN_LIMIT, worlds_during_prepare=0, sensor_queries_during_prepare=0,
            versions=library_versions(),
            python=sys.version, preparation_time=time.time()))
        write(OUTPUT / 'prepare_seal.json', {p.name: sha(p) for p in OUTPUT.iterdir() if p.is_file()})
    except BaseException:
        write(OUTPUT / 'prepare_failure.json', dict(error=traceback.format_exc(), worlds=0,
            sensor_queries=0, new_main_tasks=0, partial_preparation_preserved=True))
        raise
    print(json.dumps(dict(prepared=True, physical_cases=len(physical), comparison_cells=4,
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
        raise ValueError('historical 16/36 main quota exceeded')
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


def run_case(index, replay=False):
    cpu_contract()
    require_not_finalized()
    manifest, config = frozen()
    if index is None or index not in range(len(config['physical_cases'])):
        raise ValueError('index must identify a unique physical case in the frozen config')
    case = config['physical_cases'][index]
    folder = OUTPUT / f'case{index:02d}'
    if replay:
        verify_seal(folder, 'main_seal.json', 'replay')
        if (folder / 'failure.json').exists():
            raise ValueError('failed or incomplete acquisition cannot claim a full independent replay')
        expected = read(folder / 'result.json')
        if read(folder / 'started.json')['pid'] == os.getpid():
            raise ValueError('replay must be in a fresh process')
        work = folder / 'replay'
        guard_write(work / 'started.json', 4096)
        work.mkdir()  # Existing failed replay is retained; never retried.
        write(work / 'started.json', dict(pid=os.getpid(), time=time.time(), main_pid=read(folder / 'started.json')['pid']))
    else:
        require_previous_cases(config, index)
        guard_write(folder / 'started.json', 8192)
        folder.mkdir()  # This creation spends quota, even if the next write fails.
        work = folder
        start = dict(case=index, pid=os.getpid(), time=time.time(),
            cumulative_main_used=HISTORICAL_MAIN + len(case_directories()), counted_on_directory_creation=True)
        write(OUTPUT / 'starts' / f'main_{index:02d}.json', start)
        write(folder / 'started.json', start)
        (folder / 'packets').mkdir()
    started = time.monotonic()
    counts = dict(worlds=0, sensor_packets=0, clean_depth_queries=0, scan_queries=0,
        paid_actions=0, mapper_updates=0, TSDF_integrations=0, mapper_mesh_extractions=0,
        evaluation_stages=0, learned_reconstruction_calls=0)
    trace, snapshots = [], {}
    world = None
    def deadline(*_):
        raise TimeoutError('declared 600 second whole-case limit')
    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(SECONDS)
    try:
        import numpy as np
        from env.information_pixel_v34 import InformationPixelWorldV34, cue_from_rgb, sensor_counts_v34
        from nso.decision_replay_v13 import array_hash, load_packet
        from nso.observed_runtime_mapper_v10 import ObservedRuntimeMapperV10
        from nso.sensor_contract_v34 import validate_sensor_packet_v34
        from nso.surface_measurement_v34 import extract_observed_asset_mesh, SurfaceMeasurementV34
        if any(sensor_counts_v34().values()):
            raise ValueError('one fresh process per acquisition or replay is required')
        world = InformationPixelWorldV34(case['parent'], case['hypothesis'], case['episode_id'],
            noise_model=config['acquisition']['noise_model'])
        counts['worlds'] = 1
        if (world.noise_seed != config['acquisition']['noise_seed']
                or world.config.voxel_m != .04 or world.config.max_depth_m != 4.
                or list(world.shape) != config['acquisition']['raster_shape']
                or list(world.shift) != config['acquisition']['translation']):
            raise ValueError('execution differs from frozen calibration')
        mapper = ObservedRuntimeMapperV10(world.shape, world.config, truncation_m=.12)
        total = len(case['actions'])
        for paid, action in enumerate([None] + case['actions']):
            if time.monotonic() - started >= SECONDS:
                raise TimeoutError('declared whole-case time cap')
            packet = world.packet() if action is None else world.step(action)
            counts.update(worlds=world.counts['worlds'], sensor_packets=world.counts['packets'],
                clean_depth_queries=world.counts['clean_depth_queries'], scan_queries=world.counts['scan_queries'],
                paid_actions=world.counts['step_calls'])
            validate_sensor_packet_v34(packet, world.transform, world.config)
            path = folder / 'packets' / f'{paid:03d}.npz'
            # Preserve every valid generated main packet BEFORE checking the
            # witness, including a collision that leaves the pose unchanged.
            if not replay:
                save_full_packet(path, packet)
            errors = []
            if list(world.pose) != case['poses'][paid] or packet.action_id != paid or packet.action != action:
                errors.append('actual primitive/pose differs from frozen witness')
            if packet.collision:
                errors.append('actual execution collision')
            if np.any(packet.frame.semantic):
                errors.append('semantic/GT labels must not enter the mapper')
            if replay and not errors:
                stored = validate_sensor_packet_v34(load_packet(path), world.transform, world.config)
                if packet.sha256() != stored.sha256():
                    errors.append('fresh full packet field bytes differ from saved main')
            if errors:
                if replay:
                    # Keep the unexpected fresh frame without touching main evidence.
                    path = work / f'abnormal_packet_{paid:03d}.npz'
                    save_full_packet(path, packet)
                receipt = dict(paid=paid, expected_action=action, expected_pose_v33=case['poses'][paid],
                    action=packet.action, action_id=packet.action_id, pose_v33=list(world.pose),
                    position=packet.position, heading=packet.heading, collision=packet.collision,
                    collisions_so_far=world.collisions, packet_sha256=packet.sha256(),
                    saved_packet=str(path.relative_to(folder)), nonsemantic=nonsemantic_fields(packet),
                    clean_depth_sha256=array_hash(world.last_clean_depth), cue=cue_from_rgb(packet.frame),
                    mapper_updated=False, errors=errors)
                write(work / f'abnormal_action_{paid:03d}.json', receipt)
                trace.append(receipt)
                raise AssertionError('; '.join(errors))
            mapper.update(packet.frame, packet.scan)
            counts['mapper_updates'] += 1
            counts['TSDF_integrations'] += 1
            trace.append(dict(paid=paid, action=action, pose_v33=list(world.pose),
                position=packet.position, heading=packet.heading, packet_sha256=packet.sha256(),
                nonsemantic=nonsemantic_fields(packet), clean_depth_sha256=array_hash(world.last_clean_depth),
                cue=cue_from_rgb(packet.frame), collision=packet.collision, collisions_so_far=world.collisions,
                footprint_conflict=mapper.current_footprint_conflict,
                footprint_conflict_details=mapper.current_footprint_conflict_details))
            for stage, stage_paid in (('prefix', PREFIX), ('final', total)):
                if paid != stage_paid:
                    continue
                raw = mapper.mesh()
                counts['mapper_mesh_extractions'] += 1
                predicted, crop = extract_observed_asset_mesh(raw, config['acquisition']['public_bounds'])
                maps = {key: value.copy() for key, value in vars(mapper).items() if isinstance(value, np.ndarray)}
                if 'belief' not in maps:
                    raise AssertionError('belief array not present in mapper snapshot')
                snapshots[stage] = dict(mesh=predicted, maps=maps, crop=crop, paid=paid)
                for name, arrays in ((f'{stage}_raw.npz', mesh_arrays(raw)),
                        (f'{stage}_extracted.npz', mesh_arrays(predicted)), (f'{stage}_maps.npz', maps)):
                    if replay:
                        compare_arrays(folder / name, arrays)
                    else:
                        save_arrays(folder / name, **arrays)
                if replay:
                    if digest(crop) != digest(read(folder / f'{stage}_crop.json')):
                        raise AssertionError('fresh extraction audit differs')
                else:
                    write(folder / f'{stage}_crop.json', crop)
                print(json.dumps(dict(case=index, replay=replay, stage=stage, paid=paid,
                    seconds=time.monotonic()-started)), flush=True)
        # This seal precedes the FIRST evaluation truth access and surface score.
        if replay:
            if digest(trace) != digest(read(folder / 'trace.json')):
                raise AssertionError('fresh complete trace differs')
            frozen_predictions = read(folder / 'prediction_freeze.json')
            for rel, expected_hash in frozen_predictions['artifact_sha256'].items():
                if sha(folder / rel) != expected_hash:
                    raise AssertionError('frozen prediction artifact changed')
            write(work / 'prediction_comparison.json', dict(passed=True,
                main_prediction_freeze_sha256=sha(folder / 'prediction_freeze.json'),
                packet_count=len(trace), both_raw_and_extracted_meshes_equal=True,
                all_mapper_arrays_equal=True, all_trace_fields_equal=True))
        else:
            write(folder / 'trace.json', trace)
            write(folder / 'prediction_freeze.json', dict(before_first_quality_or_GT_reference_access=True,
                artifact_sha256={str(p.relative_to(folder)): sha(p) for p in files_under(folder)},
                stages=['prefix', 'final'], packet_count=len(trace),
                public_roi_sha256=digest(config['acquisition']['public_bounds'])))
        # GT reachable raster and full union surfaces are evaluation-side only.
        floor = world.evaluation_floor()
        reachable, grid = floor['reachable'], floor['declared_grid_cells']
        floor_arrays = dict(safe=floor['safe'], reachable=reachable, declared_grid_cells=grid)
        if replay:
            compare_arrays(folder / 'evaluation_floor.npz', floor_arrays)
        else:
            save_arrays(folder / 'evaluation_floor.npz', **floor_arrays)
        evaluator = SurfaceMeasurementV34(world.instance_mesh(0), world.instance_mesh(0, vertical_only=True))
        stages = {}
        for stage, snapshot in snapshots.items():
            belief, paid = snapshot['maps']['belief'], snapshot['paid']
            C_map = float(np.mean(belief[reachable] != -1))
            C_grid = float(np.mean(belief[grid[:, 0], grid[:, 1]] != -1))
            returned = trace[paid]['pose_v33'] == case['poses'][0]
            measurement = evaluator.evaluate(snapshot['mesh'], C_map, returned=returned,
                collisions=trace[paid]['collisions_so_far'], failed=False, paid_actions=paid, budget=BUDGET)
            counts['evaluation_stages'] += 1
            stages[stage] = dict(measurement=measurement, C_grid_measured=C_grid,
                C_grid_denominator=len(grid), full_reachable_raster_denominator=int(reachable.sum()),
                belief_sha256=array_hash(belief), crop_audit_sha256=digest(snapshot['crop']))
        recognized = [r for r in trace[:PREFIX+1] if r['cue']['class_id'] == case['hypothesis']+2]
        cue_positions = sorted({tuple(r['pose_v33'][:2]) for r in recognized})
        contradictory = [r['paid'] for r in trace[:PREFIX+1]
            if r['cue']['class_id'] not in (None, case['hypothesis']+2)]
        result = dict(physical_case=case, stages=stages, counts=counts,
            collisions=world.collisions, returned=world.pose == tuple(case['poses'][0]),
            paid_actions=world.step_count, budget=BUDGET,
            prefix_cue=dict(distinct_recognized_positions=cue_positions,
                contradictory_actions=contradictory, passed=len(cue_positions) >= 2 and not contradictory),
            footprint_conflict_count=mapper.current_footprint_conflict_count,
            trajectory_sha256=digest([{k: r[k] for k in ('paid', 'action', 'pose_v33')} for r in trace]),
            trace_sha256=digest(trace), inference_performed=False, exact_simulated_pose_not_estimated_SLAM=True,
            adaptive_policy_executed=False, full_architecture_executed=False,
            original_quota=HISTORICAL_MAIN, cumulative_main_used=HISTORICAL_MAIN+index+1,
            prediction_scope='public union ROI; errors outside the padded ROI do not affect precision')
        frozen()
        if replay:
            if digest(result) != digest(expected):
                write(work / 'mismatch_result.json', result)
                raise AssertionError('fresh reconstruction or complete measurement result differs')
            write(work / 'result.json', dict(passed=True, all_packet_field_bytes_equal=True,
                all_raw_extracted_mesh_map_array_bytes_equal=True, complete_result_equal=True,
                packet_count=len(trace), main_result_sha256=sha(folder / 'result.json'),
                main_pid=read(folder / 'started.json')['pid'], replay_pid=os.getpid(),
                counts=counts, elapsed_seconds=time.monotonic()-started))
            seal(work, 'seal.json')
        else:
            write(folder / 'result.json', result)
            if not stages['final']['measurement']['eligible'] or not result['prefix_cue']['passed']:
                write(folder / 'stop.json', dict(stop_remaining_acquisitions=True,
                    reason='final qualification or controlled RGB cue prerequisite failed',
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
            retry_allowed=False, stop_remaining_acquisitions=True, elapsed_seconds=time.monotonic()-started))
        write(work / 'partial_trace.json', trace)
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
        or not r['prefix_cue']['passed'] for r in by_index.values()))
    if incomplete and not stopped:
        raise ValueError('analysis must await all declared cases or a retained stop condition')
    table = []
    for cell in config['cells']:
        result = by_index.get(cell['physical_case'])
        table.append(dict(**cell, measured=result is not None,
            stages=result['stages'] if result is not None else None,
            prefix_to_final_joint_change={threshold: result['stages']['final']['measurement'][threshold]['joint']
                - result['stages']['prefix']['measurement'][threshold]['joint']
                for threshold in ('02cm', '05cm', '10cm')} if result is not None else None,
            independent_replay_passed=replay_receipts.get(cell['physical_case'], {}).get('passed', False)))
    pairings = []
    for policy in ('G', 'class_oracle'):
        cells = [next(c for c in config['cells'] if c['hypothesis'] == h and c['policy'] == policy) for h in (0, 1)]
        left, right = [traces.get(c['physical_case']) for c in cells]
        first = None
        if left is not None and right is not None:
            fields = set(left[0]['nonsemantic'])
            first = {key: next((i for i, (a, b) in enumerate(zip(left, right))
                if a['nonsemantic'][key] != b['nonsemantic'][key]), None) for key in fields}
        pairings.append(dict(policy=policy, physical_cases=[c['physical_case'] for c in cells],
            first_observed_field_difference=first,
            actual_prefix_nonsemantic_equal=None if first is None else all(v is None or v > PREFIX for v in first.values())))
    comparison = {}
    if not incomplete:
        for threshold in ('02cm', '05cm', '10cm'):
            values = {(c['hypothesis'], c['policy']): by_index[c['physical_case']]['stages']['final']['measurement'][threshold]
                for c in config['cells']}
            comparison[threshold] = dict(
                G_mean_joint=sum(values[h, 'G']['joint'] for h in (0, 1))/2,
                class_oracle_mean_joint=sum(values[h, 'class_oracle']['joint'] for h in (0, 1))/2,
                paired_mean_joint_difference=sum(values[h, 'class_oracle']['joint']-values[h, 'G']['joint'] for h in (0, 1))/2,
                paired_joint_differences=[values[h, 'class_oracle']['joint']-values[h, 'G']['joint'] for h in (0, 1)])
            baseline = comparison[threshold]['G_mean_joint']
            comparison[threshold]['relative_mean_joint_difference'] = (
                comparison[threshold]['paired_mean_joint_difference']/baseline if baseline else None)
    all_replayed = len(replay_receipts) == len(config['physical_cases']) and all(r['passed'] for r in replay_receipts.values())
    eligible = not incomplete and all(r['stages']['final']['measurement']['eligible'] for r in by_index.values())
    pairing_ok = all(p['actual_prefix_nonsemantic_equal'] is True for p in pairings)
    cue_ok = not incomplete and all(r['prefix_cue']['passed'] for r in by_index.values())
    positive = bool(comparison and comparison['05cm']['paired_mean_joint_difference'] > 0
        and all(comparison[t]['paired_mean_joint_difference'] >= 0 for t in ('02cm', '10cm')))
    result = dict(status='stopped' if stopped else 'complete', table=table,
        actual_noisy_prefix_pairings=pairings, prescribed_paired_comparison=comparison,
        full_matrix_collected=not incomplete, all_final_eligible=eligible, all_replays_passed=all_replayed,
        actual_C80_verified=not incomplete and all(r['stages']['final']['measurement']['C_map'] >= .8 for r in by_index.values()),
        controlled_marker_prerequisite_passed=cue_ok, all_prefix_nonsemantic_pairings_passed=pairing_ok,
        measured_fixed_route_gate_passed=eligible and all_replayed and pairing_ok and cue_ok and positive and not failures,
        failures=failures, new_main_tasks=len(case_directories()), total_main_used=HISTORICAL_MAIN+len(case_directories()),
        main_limit=MAIN_LIMIT, unique_physical_cases_declared=len(config['physical_cases']), comparison_cells=4,
        aliases_are_not_independent_samples=True, adaptive_G_or_ANS_advantage_proven=False,
        full_architecture_advantage_proven=False, natural_semantic_innovation_proven=False,
        autonomous_trials_started=0, pose_scope='exact synthetic pose; no pose-estimation SLAM evaluated',
        metric_scope='single-facility public union ROI; outside-ROI erroneous geometry excluded from precision',
        acquisition_counts={key: sum(r[key] for r in main_counts.values())
            for key in next(iter(main_counts.values()))} if main_counts else {},
        replay_counts={key: sum(r[key] for r in replay_counts.values())
            for key in next(iter(replay_counts.values()))} if replay_counts else {},
        counter_accounting_complete=all(not f['kind'].endswith('_unsealed') for f in failures))
    write(OUTPUT / 'result.json', result)
    seal(OUTPUT, 'final_seal.json')
    print(json.dumps(dict(status=result['status'], new_main_tasks=result['new_main_tasks'],
        measured_fixed_route_gate_passed=result['measured_fixed_route_gate_passed'])), flush=True)


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
