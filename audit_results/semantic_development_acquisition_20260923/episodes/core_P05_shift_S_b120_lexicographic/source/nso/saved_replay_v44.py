"""Read-only saved-observation verification, never counterfactual simulation.

Artifacts are integrity checked before construction of any mapping backend.
No World, sensor factory, renderer, or action execution is called here.
"""
from dataclasses import dataclass
import hashlib
import gzip
import json
import math
from pathlib import Path, PurePosixPath
import re
import zipfile

import numpy as np

from env.development_sensor_v41 import SensorStepV41
from nso.episode_driver_v43 import ACTION_TO_SENSOR_V43, canonical_bytes, validate_step_v43
from nso.instance_belief_v40 import PaidRGBDObservationV40
from utils.rgbd_contract import PlanarScan

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_NAME = 'configs/virtual3d/v43_runtime_protocol_20260921.json'
SOURCE_NAMES = frozenset(('nso/instance_belief_v40.py', 'nso/observed_instances_v41.py',
    'nso/observed_residual_v41.py', 'nso/primitive_navigation_v41.py',
    'nso/development_geometry_v40.py', 'nso/surface_evaluation_v40.py',
    'nso/observed_mapper_v42.py', 'nso/view_quality_v42.py',
    'env/development_sensor_v41.py', 'nso/scene_contract_v40.py', 'utils/rgbd_contract.py',
    'configs/virtual3d/v40_scene_protocol_20260920.json', PROTOCOL_NAME,
    'scripts/run_development_v43.py', 'nso/controller_v43.py',
    'nso/diagnostic_policy_v43.py', 'nso/episode_driver_v43.py',
    'nso/observed_safety_v43.py', 'nso/public_navigation_v43.py'))
COMPLETE_STATUSES = frozenset(('controller_stop', 'controller_blocked', 'budget_exhausted',
                              'stopped_without_confirmed_return'))
FAILED_STATUSES = frozenset(('episode_error', 'artifact_limit', 'wall_time_limit',
                            'prediction_save_failed', 'development_attempt_failed'))


class SavedEpisodeIntegrityErrorV44(ValueError):
    pass


class FailedSavedEpisodeV44(SavedEpisodeIntegrityErrorV44):
    def __init__(self, status):
        self.status = status
        super().__init__('failed or incomplete episode is not an eligible complete replay: '+status)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _object_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise SavedEpisodeIntegrityErrorV44('duplicate JSON field: '+key)
        result[key] = value
    return result


def _json(path):
    def invalid(value):
        raise SavedEpisodeIntegrityErrorV44('nonfinite JSON number: '+value)
    try:
        if str(path).endswith('.gz'):
            with gzip.open(path, 'rb') as stream:
                content = stream.read(32*1024**2+1)
            if len(content) > 32*1024**2:
                raise SavedEpisodeIntegrityErrorV44('decompressed step exceeds 32 MiB cap')
        else:
            content = Path(path).read_bytes()
        return json.loads(content, object_pairs_hook=_object_pairs, parse_constant=invalid)
    except (OSError, EOFError, UnicodeError, json.JSONDecodeError) as exc:
        raise SavedEpisodeIntegrityErrorV44('unreadable JSON artifact: '+str(path)) from exc


def _relative(name):
    if (not isinstance(name, str) or '\\' in name or not name
            or str(PurePosixPath(name)) != name or PurePosixPath(name).is_absolute()
            or '..' in PurePosixPath(name).parts):
        raise SavedEpisodeIntegrityErrorV44('unsafe or noncanonical artifact path')
    return name


def inspect_saved_inventory_v44(root, *, source_root=None, expected_manifest_sha256=None):
    """Validate hashes and source copies, including failed-episode inventories.

    Without an external expected digest this detects accidental corruption,
    not coordinated replacement of both artifacts and their manifest.
    """
    root = Path(root).resolve(strict=True)
    source_root = ROOT if source_root is None else Path(source_root).resolve(strict=True)
    manifest_path = root/'artifact_manifest.json'
    if not root.is_dir() or not manifest_path.is_file():
        raise SavedEpisodeIntegrityErrorV44('complete terminal artifact manifest required')
    entries = list(root.rglob('*'))
    if any(path.is_symlink() for path in entries):
        raise SavedEpisodeIntegrityErrorV44('symlink artifacts are forbidden')
    if sum(path.stat().st_size for path in entries if path.is_file()) > 64*1024**2:
        raise SavedEpisodeIntegrityErrorV44('episode exceeds frozen 64 MiB cap')
    manifest_sha = _sha(manifest_path)
    if expected_manifest_sha256 is not None and manifest_sha != expected_manifest_sha256:
        raise SavedEpisodeIntegrityErrorV44('external manifest SHA256 mismatch')
    manifest = _json(manifest_path)
    files = manifest.get('files')
    if not isinstance(files, dict) or not files or len(files) > 1200:
        raise SavedEpisodeIntegrityErrorV44('bounded nonempty file inventory required')
    disk = {str(path.relative_to(root)) for path in entries if path.is_file()}
    if disk != set(files)|{'artifact_manifest.json'}:
        raise SavedEpisodeIntegrityErrorV44('missing or unmanifested episode artifact')
    for name, row in files.items():
        _relative(name)
        if (not isinstance(row, dict) or set(row) != {'bytes', 'sha256'}
                or type(row['bytes']) is not int or not 0 <= row['bytes'] <= 32*1024**2
                or not isinstance(row['sha256'], str) or not re.fullmatch('[0-9a-f]{64}', row['sha256'])):
            raise SavedEpisodeIntegrityErrorV44('invalid artifact hash record')
        path = root/name
        if path.stat().st_size != row['bytes'] or _sha(path) != row['sha256']:
            raise SavedEpisodeIntegrityErrorV44('artifact SHA256/size mismatch: '+name)
    sources = manifest.get('source_sha256')
    v44_sources = SOURCE_NAMES|{'scripts/run_development_v44.py', 'nso/evidence_writer_v44.py'}
    if not isinstance(sources, dict) or set(sources) not in (SOURCE_NAMES, v44_sources):
        raise SavedEpisodeIntegrityErrorV44('complete frozen V43 source closure required')
    for name, digest in sources.items():
        _relative(name)
        if ('source/'+name not in files or files['source/'+name]['sha256'] != digest
                or not (source_root/name).is_file() or _sha(source_root/name) != digest):
            raise SavedEpisodeIntegrityErrorV44('current/archive source mismatch: '+name)
    required = {'started.json', 'result.json', 'runtime.json', 'public_graph.json',
                'public_spec.json', 'public_workspace.json'}
    if not required <= set(files):
        raise SavedEpisodeIntegrityErrorV44('missing complete episode metadata')
    started, result, runtime = (_json(root/name) for name in ('started.json', 'result.json', 'runtime.json'))
    protocol = _json(root/'source'/PROTOCOL_NAME)
    if (started.get('source_sha256') != sources
            or started.get('protocol_sha256') != sources[PROTOCOL_NAME]
            or result.get('schema') != 'v43.episode_result.v1'
            or runtime.get('source_unchanged') is not True or 'integrity_failure.json' in files):
        raise SavedEpisodeIntegrityErrorV44('episode/source provenance mismatch')
    run_id = started.get('run_id')
    if (run_id not in protocol['slots'] or started.get('slot') != protocol['slots'][run_id]
            or protocol.get('maximum_replays_authorized_here') != 0):
        raise SavedEpisodeIntegrityErrorV44('episode must retain its declared V43 slot and zero new-World replay quota')
    if started.get('finite_fixture') is True:
        if runtime.get('finite_fixture') is not True or runtime.get('world_created') is not False:
            raise SavedEpisodeIntegrityErrorV44('finite fixture provenance is inconsistent')
    else:
        if runtime.get('world_created') is not True:
            raise SavedEpisodeIntegrityErrorV44('live study episode requires actual World receipt')
        ledger_path = source_root/protocol['ledger_relative_path']
        if not ledger_path.is_file():
            raise SavedEpisodeIntegrityErrorV44('live episode requires original durable start ledger')
        ledger = _json(ledger_path)
        matches = [row for row in ledger.get('entries', []) if row.get('run_id') == run_id]
        if (len(matches) != 1 or matches[0].get('world_created') is not True
                or matches[0].get('status') != result.get('status')
                or matches[0].get('result_sha256') != _sha(root/'result.json')
                or matches[0].get('metadata', {}).get('source_sha256') != sources):
            raise SavedEpisodeIntegrityErrorV44('live episode is not bound to durable completed reservation')
        before, after = runtime.get('before', {}), runtime.get('after', {})
        for key, delta in (('worlds_created', 1), ('rgbd_frames', result.get('received_sensor_packets')),
                           ('scans', result.get('received_sensor_packets')), ('paid_actions', result.get('executed_paid_actions'))):
            if type(before.get(key)) is not int or type(after.get(key)) is not int or after[key]-before[key] != delta:
                raise SavedEpisodeIntegrityErrorV44('live runtime counters disagree with saved episode')
    return dict(root=root, manifest=manifest, manifest_sha256=manifest_sha,
                externally_pinned_manifest=expected_manifest_sha256 is not None,
                started=started, result=result, runtime=runtime, protocol=protocol)


def _arrays(path, fields):
    try:
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            if (len(infos) != len(fields) or {row.filename for row in infos} != {name+'.npy' for name in fields}
                    or sum(row.file_size for row in infos) > 32*1024**2):
                raise SavedEpisodeIntegrityErrorV44('bounded exact-key numeric NPZ required')
        with np.load(path, allow_pickle=False) as data:
            values = {name: data[name].copy() for name in fields}
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        raise SavedEpisodeIntegrityErrorV44('invalid saved numeric arrays: '+str(path)) from exc
    return values


@dataclass(frozen=True)
class SavedFrameV44:
    index: int
    rgbd: PaidRGBDObservationV40
    scan: PlanarScan
    receipt: dict
    record: dict


@dataclass(frozen=True)
class SavedEpisodeV44:
    root: Path
    started: dict
    result: dict
    runtime: dict
    protocol: dict
    public_graph: dict
    public_spec: dict
    public_workspace: dict
    manifest: dict
    manifest_sha256: str
    frames: tuple
    externally_pinned_manifest: bool

    @property
    def status(self):
        return self.result['status']

    @property
    def eligible_study_episode(self):
        return self.started.get('finite_fixture') is not True

    @property
    def task_success(self):
        return self.status == 'controller_stop' and self.result['sensor_status'].get('returned_xy_and_yaw') is True


def load_saved_episode_v44(root, *, source_root=None, expected_manifest_sha256=None):
    """Read complete saved packets; reject partial runs rather than score a prefix."""
    inventory = inspect_saved_inventory_v44(root, source_root=source_root,
                                           expected_manifest_sha256=expected_manifest_sha256)
    root, result = inventory['root'], inventory['result']
    status = result.get('status')
    if status in FAILED_STATUSES:
        raise FailedSavedEpisodeV44(status)
    if status not in COMPLETE_STATUSES or result.get('error') is not None or result.get('finalization_errors') != []:
        raise SavedEpisodeIntegrityErrorV44('unknown terminal status or hidden failure')
    count = result.get('acquired_and_saved_packets')
    if type(count) is not int or not 1 <= count <= 161:
        raise SavedEpisodeIntegrityErrorV44('bounded consecutive saved packets required')
    files = set(inventory['manifest']['files'])
    packet_files = {f'packets/{i:03d}_{suffix}' for i in range(count)
                    for suffix in ('rgbd.npz', 'scan.npz', 'receipt.json')}
    step_names = []
    for i in range(count):
        aliases = {f'steps/{i:03d}.json', f'steps/{i:03d}.json.gz'} & files
        if len(aliases) != 1:
            raise SavedEpisodeIntegrityErrorV44('exactly one plain or gzip step artifact required')
        step_names.append(aliases.pop())
    step_files = set(step_names)
    if any(name.endswith('.gz') for name in step_names):
        if set(inventory['manifest']['source_sha256']) != SOURCE_NAMES|{'scripts/run_development_v44.py', 'nso/evidence_writer_v44.py'} or 'encoding.json' not in files:
            raise SavedEpisodeIntegrityErrorV44('compressed step requires archived V44 writer and encoding ledger')
        encoding = _json(root/'encoding.json')
        rows = encoding.get('steps')
        if encoding.get('schema') != 'v44.step_encoding.v1' or not isinstance(rows, list) or len(rows) != count:
            raise SavedEpisodeIntegrityErrorV44('complete compression ledger required')
        for name, row in zip(step_names, rows):
            if (row.get('artifact') != name or row.get('stored_bytes') != inventory['manifest']['files'][name]['bytes']
                    or row.get('encoding') != 'gzip-json-v1' or row.get('compression_level') != 6
                    or row.get('mtime') != 0 or type(row.get('uncompressed_bytes')) is not int
                    or not 0 < row['uncompressed_bytes'] <= 32*1024**2):
                raise SavedEpisodeIntegrityErrorV44('compression ledger mismatch')
            with gzip.open(root/name, 'rb') as stream:
                content = stream.read(32*1024**2+1)
            if len(content) != row['uncompressed_bytes']:
                raise SavedEpisodeIntegrityErrorV44('decompressed size mismatch')
    if ({name for name in files if name.startswith('packets/')} != packet_files
            or {name for name in files if name.startswith('steps/')} != step_files):
        raise SavedEpisodeIntegrityErrorV44('missing, duplicate, or unordered packet/step sequence')
    prediction = {'prediction/mesh.npz', 'prediction/occupancy.npz', 'prediction/mapper.json'}
    if not prediction <= files or result.get('artifacts') != {'mesh_saved': True, 'occupancy_saved': True}:
        raise SavedEpisodeIntegrityErrorV44('complete saved prediction required')
    if (result.get('received_sensor_packets') != count or result.get('executed_paid_actions') != count-1
            or result.get('submitted_paid_actions') != count-1 or result.get('mapper_frames') != count):
        raise SavedEpisodeIntegrityErrorV44('complete packet/action accounting mismatch')
    public_graph, public_spec, public_workspace = (_json(root/name) for name in
        ('public_graph.json', 'public_spec.json', 'public_workspace.json'))
    from nso.primitive_navigation_v41 import PublicPrimitiveGraphV41
    graph = PublicPrimitiveGraphV41(public_graph)
    if graph.input_sha256 != inventory['started'].get('public_graph_sha256'):
        raise SavedEpisodeIntegrityErrorV44('saved graph binding mismatch')
    if inventory['started'].get('finite_fixture') is not True:
        from nso.public_navigation_v43 import load_public_navigation_bundle_v43
        checked_root = ROOT if source_root is None else Path(source_root).resolve(strict=True)
        protocol = inventory['protocol']; asset_id = inventory['started']['slot']['asset_id']
        frozen = load_public_navigation_bundle_v43(
            checked_root/protocol['navigation_root']/asset_id,
            checked_root/protocol['asset_root']/asset_id,
            protocol_path=checked_root/protocol['scene_protocol_path'])
        if (public_graph != frozen['graph_spec'] or public_spec != frozen['public_spec']
                or public_workspace != frozen['workspace']):
            raise SavedEpisodeIntegrityErrorV44('live episode public contract differs from frozen shared bundle')
    if count-1 > public_spec['task']['max_actions']:
        raise SavedEpisodeIntegrityErrorV44('saved trajectory exceeds declared action budget')
    frames, seen_ids, actions, submitted, collisions, previous_pose = [], set(), [], [], 0, None
    for index in range(count):
        prefix = f'packets/{index:03d}'
        values = _arrays(root/(prefix+'_rgbd.npz'), PaidRGBDObservationV40.__dataclass_fields__)
        for name in ('frame_id', 'paid_step'):
            if values[name].shape != ():
                raise SavedEpisodeIntegrityErrorV44('scalar packet identity required')
            values[name] = values[name].item()
        rgbd = PaidRGBDObservationV40.from_mapping(values)
        scan_values = _arrays(root/(prefix+'_scan.npz'), PlanarScan.__dataclass_fields__)
        for name in set(scan_values)-{'ranges_m', 'world_from_laser'}:
            if scan_values[name].shape != ():
                raise SavedEpisodeIntegrityErrorV44('scalar scan metadata required')
            scan_values[name] = scan_values[name].item()
        scan = PlanarScan(**scan_values)
        from nso.observed_mapper_v42 import _scan_copy
        _scan_copy(scan, None if index == 0 else frames[-1].scan.timestamp_s)
        receipt = _json(root/(prefix+'_receipt.json'))
        record = _json(root/step_names[index])
        if (set(receipt) != {'execution', 'observation_sha256', 'rgbd_artifact', 'scan_artifact'}
                or set(record) != {'accounting', 'mapper', 'controller_evidence', 'decision'}
                or receipt['observation_sha256'] != rgbd.sha256() or rgbd.frame_id in seen_ids
                or receipt['rgbd_artifact'] != inventory['manifest']['files'][prefix+'_rgbd.npz']
                or receipt['scan_artifact'] != inventory['manifest']['files'][prefix+'_scan.npz']):
            raise SavedEpisodeIntegrityErrorV44('duplicate packet or packet/receipt binding mismatch')
        expected_action = 'initial_observation'
        if index:
            chosen = frames[-1].record['decision'].get('action')
            if chosen not in ACTION_TO_SENSOR_V43:
                raise SavedEpisodeIntegrityErrorV44('saved frames continue after terminal decision')
            expected_action = ACTION_TO_SENSOR_V43[chosen]
        accounting = validate_step_v43(SensorStepV41(rgbd, scan, receipt['execution']),
            expected_step=index, expected_action=expected_action, expected_previous_pose=previous_pose)
        if record['accounting'] != accounting:
            raise SavedEpisodeIntegrityErrorV44('saved accounting disagrees with packet')
        for field in ('mapper', 'controller_evidence'):
            if record[field].get('observation_sha256') != rgbd.sha256():
                raise SavedEpisodeIntegrityErrorV44('current packet is not bound to '+field)
        previous_pose = receipt['execution']['pose_xyyaw_rad']
        seen_ids.add(rgbd.frame_id); collisions += int(accounting['collision'])
        if index:
            actions.append(dict(paid_step=index, controller_action=chosen, sensor_action=expected_action,
                                observation_sha256=rgbd.sha256(), collision=accounting['collision']))
            submitted.append(dict(expected_paid_step=index, controller_action=chosen, sensor_action=expected_action))
        frames.append(SavedFrameV44(index, rgbd, scan, receipt, record))
    if result.get('actions') != actions or result.get('submitted_action_records') != submitted or result.get('collisions') != collisions:
        raise SavedEpisodeIntegrityErrorV44('terminal action/collision ledger mismatch')
    last = frames[-1].record['decision']
    terminal_last = {key: last[key] for key in ('action', 'reason', 'target', 'paid_step') if key in last}
    if result.get('last_decision') != terminal_last:
        raise SavedEpisodeIntegrityErrorV44('terminal decision mismatch')
    if ((status in ('controller_stop', 'stopped_without_confirmed_return') and last.get('action') != 'stop')
            or (status == 'controller_stop' and result.get('sensor_status', {}).get('returned_xy_and_yaw') is not True)
            or (status == 'controller_blocked' and last.get('action') != 'blocked')
            or (status == 'budget_exhausted' and (count-1 != public_spec['task']['max_actions']
                or last.get('action') not in ACTION_TO_SENSOR_V43))):
        raise SavedEpisodeIntegrityErrorV44('terminal status disagrees with saved decisions')
    mapper = _json(root/'prediction/mapper.json')
    occupancy = _arrays(root/'prediction/occupancy.npz', ('belief', 'observed'))
    from nso.observed_mapper_v42 import _array_sha256
    if (mapper.get('frames') != count or mapper.get('receipts') != [frame.record['mapper'] for frame in frames]
            or mapper.get('tsdf_integration_count') != result.get('mapper_tsdf_integrations')
            or mapper.get('occupancy_sha256') != _array_sha256(occupancy['belief'])):
        raise SavedEpisodeIntegrityErrorV44('saved final mapper/prediction binding mismatch')
    return SavedEpisodeV44(**inventory, public_graph=public_graph, public_spec=public_spec,
                           public_workspace=public_workspace, frames=tuple(frames))


def _first_difference(expected, actual, path='root'):
    """Exact JSON-equivalent comparison, no tolerance for policy score drift."""
    if isinstance(expected, dict) and isinstance(actual, dict):
        if set(expected) != set(actual):
            return path+':keys'
        for key in sorted(expected):
            found = _first_difference(expected[key], actual[key], path+'.'+key)
            if found:
                return found
        return None
    if isinstance(expected, (list, tuple)) and isinstance(actual, (list, tuple)):
        if len(expected) != len(actual):
            return path+':length'
        for index, (left, right) in enumerate(zip(expected, actual)):
            found = _first_difference(left, right, path+'['+str(index)+']')
            if found:
                return found
        return None
    return None if canonical_bytes(expected) == canonical_bytes(actual) else path


def _canonical_mesh(arrays):
    vertices, triangles, colors = (np.asarray(arrays[name]) for name in ('vertices', 'triangles', 'vertex_colors'))
    if (vertices.ndim != 2 or vertices.shape[1:] != (3,) or triangles.ndim != 2 or triangles.shape[1:] != (3,)
            or colors.shape != vertices.shape or not np.isfinite(vertices).all() or not np.isfinite(colors).all()
            or triangles.dtype.kind not in 'iu' or (triangles.size and (triangles.min() < 0 or triangles.max() >= len(vertices)))):
        raise SavedEpisodeIntegrityErrorV44('invalid complete triangle mesh arrays')
    # Backend extraction can enumerate identical vertices/triangles differently.
    # Compare geometric triangle corners and colors; never discard far vertices.
    rows = []
    for triangle in triangles:
        corners = np.concatenate((vertices[triangle], colors[triangle]), axis=1)
        order = np.lexsort(corners.T[::-1])
        rows.append(corners[order].reshape(-1))
    all_vertices = np.concatenate((vertices, colors), axis=1)
    all_vertices = all_vertices[np.lexsort(all_vertices.T[::-1])]
    rows = np.asarray(rows, dtype=float).reshape(-1, 18)
    if len(rows):
        rows = rows[np.lexsort(rows.T[::-1])]
    return all_vertices, rows


def _verify_loaded_v44(episode, mapper, controller):
    """Internal loop; finite doubles may exercise this without any live sensor."""
    checked = 0
    result = dict(schema='v44.saved_observation_verification.v1',
        verification_kind='saved_observation_determinism_only',
        source_manifest_sha256=episode.manifest_sha256, source_episode_status=episode.status,
        source_task_success=episode.task_success, eligible_study_episode=episode.eligible_study_episode, source_is_finite_fixture=episode.started.get('finite_fixture') is True,
        externally_pinned_manifest=episode.externally_pinned_manifest,
        new_worlds=0, new_sensor_queries=0, physical_actions=0, counterfactual_trajectory=False,
        semantic_performance_claim=False, frames_verified=0,
        comparison='exact JSON values; occupancy exact; mesh order-invariant coordinates/colors atol=1e-9, rtol=0')
    try:
        for frame in episode.frames:
            actual_map = mapper.update(frame.rgbd, frame.scan)
            actual_evidence = controller.accept(frame.rgbd, mapper,
                execution_outcome='collision' if frame.receipt['execution']['collision'] else 'success')
            actual_decision = controller.choose()
            for name, actual in (('mapper', actual_map), ('controller_evidence', actual_evidence), ('decision', actual_decision)):
                difference = _first_difference(frame.record[name], actual, name)
                if difference:
                    return dict(result, status='diverged', frames_verified=checked,
                                divergent_step=frame.index, difference=difference, future_saved_frames_processed=False)
            checked += 1
        difference = _first_difference(_json(episode.root/'prediction/mapper.json'), mapper.snapshot(), 'final_mapper')
        if difference:
            return dict(result, status='diverged', frames_verified=checked, difference=difference)
        occupancy = _arrays(episode.root/'prediction/occupancy.npz', ('belief', 'observed'))
        if any(not np.array_equal(occupancy[name], value) for name, value in zip(('belief', 'observed'), mapper.occupancy_arrays())):
            return dict(result, status='diverged', frames_verified=checked, difference='final_occupancy')
        expected_mesh = _canonical_mesh(_arrays(episode.root/'prediction/mesh.npz', ('vertices', 'triangles', 'vertex_colors')))
        actual_mesh = _canonical_mesh(mapper.mesh_arrays())
        if any(left.shape != right.shape or not np.allclose(left, right, atol=1e-9, rtol=0)
               for left, right in zip(expected_mesh, actual_mesh)):
            return dict(result, status='diverged', frames_verified=checked, difference='final_mesh')
        inspect_saved_inventory_v44(episode.root, expected_manifest_sha256=episode.manifest_sha256)
        return dict(result, status='verified', frames_verified=checked, prediction_verified=True, source_unchanged_after_verification=True)
    except Exception as exc:
        return dict(result, status='verification_error', frames_verified=checked,
                    error=dict(type=type(exc).__name__, message=str(exc)), automatic_retry=False)


def verify_saved_observations_v44(root, *, source_root=None, expected_manifest_sha256=None):
    episode = load_saved_episode_v44(root, source_root=source_root,
                                    expected_manifest_sha256=expected_manifest_sha256)
    from nso.controller_v43 import ANSControllerV43
    from nso.diagnostic_policy_v43 import DiagnosticPolicyV43
    from nso.observed_mapper_v42 import ObservedMapperV42
    from nso.primitive_navigation_v41 import PublicPrimitiveGraphV41, PrimitiveStateV41
    graph = PublicPrimitiveGraphV41(episode.public_graph)
    home = PrimitiveStateV41('home', 0)
    budget = episode.public_spec['task']['max_actions']
    mode = episode.started['slot']['mode']
    if mode == 'diagnostic':
        controller = DiagnosticPolicyV43(graph, home=home, budget=budget,
                                        actions=episode.protocol['diagnostic_actions'])
    else:
        prior = episode.public_spec['structure_prior']
        controller = ANSControllerV43(graph, home=home, budget=budget,
            palette=episode.public_workspace['marker_palette'], structure_names=prior['abstract_structures'],
            class_structure_prior=prior['probability_by_category'], mode=mode, **episode.protocol['controller'])
    bounds = episode.public_workspace['bounds_xy_m']; origin = bounds[0]
    resolution = episode.protocol['mapper']['resolution_m']
    shape = [math.ceil((bounds[1][1]-origin[1])/resolution), math.ceil((bounds[1][0]-origin[0])/resolution)]
    mapper = ObservedMapperV42(shape=shape, origin_xy_m=origin, **episode.protocol['mapper'])
    return _verify_loaded_v44(episode, mapper, controller)
