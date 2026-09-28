"""V26 planning boundary: measured geometry and separate artificial RGB cues.

No world, reference, task bounds, mesh evaluation or hidden category is read.
Geometry extraction never reads RGB, semantic labels, marker pixels or quality
labels. Cue confidence is observed exact-color consistency, NOT calibrated
network confidence. Cue centers are observed marker surface locations, never
hidden object centers. Call geometry_state after mapper.update on that packet.
"""
from dataclasses import dataclass
import hashlib
import json
from collections.abc import Mapping
import numpy as np
from scipy.ndimage import label
from nso.cpu_sensor_contract_v10 import SensorPacket
from utils.rgbd_contract import RGBDFrame, PlanarScan


MARKER_COLORS_V26 = ((2, (40, 100, 220)), (3, (220, 60, 40)))
MINIMUM_MARKER_PIXELS = 16
TRACK_DISTANCE_M = .75


def _integer(value, name, minimum=0, maximum=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name+' must be an integer')
    result = int(value)
    if result < minimum or (maximum is not None and result > maximum):
        raise ValueError(name+' outside allowed range')
    return result


def _number(value, name, minimum=0., strict=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)):
        raise ValueError(name+' must be a finite number')
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(name+' must be a real finite number') from error
    if not np.isfinite(result) or (result <= minimum if strict else result < minimum):
        raise ValueError(name+' outside allowed range')
    return result


def _vector(value, size, name):
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(name+' must be finite geometry') from error
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(name+' must be finite geometry')
    return tuple(map(float, result))


def _cell(value, name, shape=None):
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError(name+' requires row and column')
    result = tuple(_integer(x, name) for x in value)
    if shape is not None and any(x >= limit for x, limit in zip(result, shape)):
        raise ValueError(name+' outside map')
    return result


def _rigid(value, name):
    a = np.asarray(value)
    if (a.shape != (4, 4) or not np.issubdtype(a.dtype, np.floating)
            or not np.isfinite(a).all() or not np.allclose(a[3], [0, 0, 0, 1], atol=1e-7, rtol=0)
            or not np.allclose(a[:3, :3].T@a[:3, :3], np.eye(3), atol=1e-6, rtol=0)
            or not np.isclose(np.linalg.det(a[:3, :3]), 1., atol=1e-6, rtol=0)):
        raise ValueError(name+' must be a finite rigid transform')
    return a


def _packet_geometry(packet):
    """Validate only geometric channels: do not call label-dependent validate()."""
    if not isinstance(packet, SensorPacket):
        raise ValueError('SensorPacket required')
    for name in ('scene_id', 'episode_id', 'frame_id', 'sensor_source', 'pose_source'):
        if not isinstance(getattr(packet, name), str) or not getattr(packet, name):
            raise ValueError('nonempty packet identity/provenance required')
    _integer(packet.action_id, 'action_id')
    _cell(packet.position, 'position'); _integer(packet.heading, 'heading', maximum=3)
    if packet.action not in (None, 'forward', 'left', 'right'):
        raise ValueError('unsupported action')
    if type(packet.collision) is not bool or type(packet.done) is not bool:
        raise ValueError('collision/done must be bool')
    f, scan = packet.frame, packet.scan
    if not isinstance(f, RGBDFrame) or not isinstance(scan, PlanarScan):
        raise ValueError('typed RGBDFrame and PlanarScan required')
    d, k = np.asarray(f.depth_m), np.asarray(f.intrinsic)
    if (d.ndim != 2 or min(d.shape) < 3 or not np.issubdtype(d.dtype, np.floating)
            or not np.isfinite(d).all() or (d < 0).any()):
        raise ValueError('finite nonnegative floating metric depth required')
    if (k.shape != (3, 3) or not np.issubdtype(k.dtype, np.floating) or not np.isfinite(k).all()
            or k[0, 0] <= 0 or k[1, 1] <= 0 or not np.allclose(k[2], [0, 0, 1], atol=1e-8, rtol=0)
            or np.linalg.det(k) <= 0 or np.linalg.cond(k) > 1e12):
        raise ValueError('valid pinhole calibration required')
    t = _rigid(f.world_from_camera, 'camera pose')
    timestamp = _number(f.timestamp_s, 'timestamp_s')
    ranges = np.asarray(scan.ranges_m)
    maximum = _number(scan.range_max_m, 'scan range', strict=True)
    if (ranges.ndim != 1 or not ranges.size or not np.issubdtype(ranges.dtype, np.floating)
            or not np.isfinite(ranges).all() or (ranges <= 0).any() or (ranges > maximum+1e-6).any()):
        raise ValueError('valid measured scan ranges required')
    _rigid(scan.world_from_laser, 'laser pose')
    if (not np.isfinite(scan.angle_min_rad)
            or _number(scan.angle_increment_rad, 'scan angular increment', strict=True) <= 0
            or abs(_number(scan.timestamp_s, 'scan timestamp')-timestamp) > 1e-6):
        raise ValueError('synchronized calibrated scan required')
    return d, k, t, timestamp


def _belief_copy(value):
    array = np.asarray(value)
    if array.ndim != 2 or min(array.shape) < 1 or not np.issubdtype(array.dtype, np.integer) or not np.isin(array, [-1, 0, 1]).all():
        raise ValueError('integer unknown/free/occupied belief required')
    # bytes owns the backing store, so even setflags(write=True) cannot mutate it.
    return np.frombuffer(np.ascontiguousarray(array, dtype=np.int8).tobytes(), dtype=np.int8).reshape(array.shape)


@dataclass(frozen=True)
class GeometryPatchV26:
    key: tuple
    point: tuple
    normal: tuple
    n: int
    bits: int
    best_range: float
    residual: float


@dataclass(frozen=True)
class GeometryStateV26:
    """Immutable G whitelist; use geometry_state_v26 or equivalent input validation."""
    belief: np.ndarray
    resolution_m: float
    robot_radius_m: float
    max_depth_m: float
    fov_deg: float
    width_px: int
    height_px: int
    camera_height_m: float
    position: tuple
    heading: int
    anchor: tuple
    remaining_budget: int
    action_id: int
    patches: tuple
    geometry_sha256: str

    def __post_init__(self):
        object.__setattr__(self, 'belief', _belief_copy(self.belief))
        for name in ('position', 'anchor', 'patches'):
            object.__setattr__(self, name, tuple(getattr(self, name)))


def geometry_state_v26(mapper, packet, anchor, remaining_budget, max_patches=256):
    """Snapshot the explicit G whitelist, without category-based patch discovery.

    All measured, finite, nonzero-normal patches inside the public map extent
    are eligible, including label-0/1 support. Stable voxel-key order followed
    by uniform index thinning is the only sampling; no object/marker filter.
    Neither scene/episode/frame text nor color/labels enters geometry_sha256.
    """
    d, k, t, timestamp = _packet_geometry(packet)
    belief = _belief_copy(mapper.belief); shape = belief.shape
    c = mapper.config
    public = dict(resolution_m=_number(c.resolution_m, 'resolution', strict=True),
        robot_radius_m=_number(c.robot_radius_m, 'robot radius'),
        max_depth_m=_number(c.max_depth_m, 'camera range', strict=True),
        fov_deg=_number(c.fov_deg, 'fov', strict=True),
        width_px=_integer(c.width_px, 'width', 8), height_px=_integer(c.height_px, 'height', 8),
        camera_height_m=_number(c.camera_height_m, 'camera height', strict=True))
    if public['fov_deg'] >= 180 or d.shape != (public['height_px'], public['width_px']):
        raise ValueError('camera dimensions/FOV disagree')
    focal = public['width_px']/(2*np.tan(np.deg2rad(public['fov_deg'])/2))
    expected_k = [[focal, 0., (public['width_px']-1)/2], [0., focal, (public['height_px']-1)/2], [0., 0., 1.]]
    if not np.allclose(k, expected_k, atol=1e-6, rtol=0) or (d > public['max_depth_m']+1e-6).any():
        raise ValueError('camera calibration/range disagrees with public configuration')
    position = _cell(packet.position, 'position', shape)
    heading = _integer(packet.heading, 'heading', maximum=3)
    if not isinstance(anchor, (tuple, list)) or len(anchor) != 3:
        raise ValueError('anchor includes row, column and heading')
    anchor = (*_cell(anchor[:2], 'anchor', shape), _integer(anchor[2], 'anchor heading', maximum=3))
    remaining = _integer(remaining_budget, 'remaining_budget'); limit = _integer(max_patches, 'max_patches', 1)
    expected_xyz = [(position[1]+.5)*public['resolution_m'],
        (shape[0]-position[0]-.5)*public['resolution_m'], public['camera_height_m']]
    forward = np.asarray(((0., 1., 0.), (1., 0., 0.), (0., -1., 0.), (-1., 0., 0.))[heading])
    rotation = np.column_stack((np.cross(forward, [0., 0., 1.]), [0., 0., -1.], forward))
    if not np.allclose(t[:3, 3], expected_xyz, atol=1e-6, rtol=0) or not np.allclose(t[:3, :3], rotation, atol=1e-6, rtol=0):
        raise ValueError('packet pose disagrees with the discrete camera/base state')
    details = getattr(mapper, 'current_footprint_conflict_details', None)
    if (not isinstance(details, Mapping) or details.get('map_version') != _integer(mapper.frames, 'mapper frames', 1)
            or details.get('timestamp_s') != timestamp or tuple(details.get('current_cell', ())) != position):
        raise ValueError('mapper must have consumed this packet before state extraction')
    if not isinstance(mapper.quality, Mapping):
        raise ValueError('observed quality mapping required')
    patches = []
    for raw_key, row in mapper.quality.items():
        if not isinstance(raw_key, tuple) or len(raw_key) != 3:
            raise ValueError('quality key must contain three integer voxel coordinates')
        key = tuple(_integer(x, 'quality key', -(2**63)) for x in raw_key)
        point = _vector(row['point'], 3, 'quality point'); normal = np.asarray(_vector(row['normal'], 3, 'quality normal'))
        length = float(np.linalg.norm(normal))
        n = _integer(row['n'], 'quality count', 1); bits = _integer(row['bits'], 'view bits', 1, 255)
        best = _number(row['best_range'], 'best range', strict=True); residual = _number(row['residual'], 'residual')
        if length < 1e-8:
            raise ValueError('measured quality patch requires a nonzero normal')
        normal /= length
        if not (0 <= point[0] < shape[1]*public['resolution_m'] and 0 <= point[1] < shape[0]*public['resolution_m']):
            continue
        patches.append(GeometryPatchV26(key, point, tuple(map(float, normal)), n, bits, best, residual))
    patches.sort(key=lambda patch: patch.key)
    if len(patches) > limit:
        patches = [patches[i] for i in np.linspace(0, len(patches)-1, limit, dtype=int)]
    patches = tuple(patches)
    payload = dict(**public, position=position, heading=heading, anchor=anchor,
        remaining_budget=remaining, action_id=int(packet.action_id),
        patches=[vars(p) for p in patches], belief_shape=shape, belief_dtype=belief.dtype.str)
    h = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False).encode())
    h.update(belief.tobytes())
    return GeometryStateV26(belief=belief, **public, position=position, heading=heading, anchor=anchor,
        remaining_budget=remaining, action_id=int(packet.action_id), patches=patches, geometry_sha256=h.hexdigest())


@dataclass(frozen=True)
class SemanticCueV26:
    cue_id: str
    center: tuple
    outward: tuple
    class_id: int
    confidence: float
    action_id: int
    source: str

    def __post_init__(self):
        if not isinstance(self.cue_id, str) or not self.cue_id or not isinstance(self.source, str) or not self.source:
            raise ValueError('nonempty cue identity/source required')
        center = _vector(self.center, 3, 'cue center')
        raw = np.asarray(self.outward)
        if raw.shape == (2,):
            raw = np.r_[raw, 0.]
        outward = _vector(raw, 3, 'cue outward')
        if not np.isclose(np.linalg.norm(outward), 1., atol=1e-6, rtol=0) or np.linalg.norm(outward[:2]) < 1e-8:
            raise ValueError('cue outward must be unit length with a horizontal direction')
        category = _integer(self.class_id, 'cue class', 2, 3)
        confidence = _number(self.confidence, 'cue confidence')
        if confidence > 1.:
            raise ValueError('cue confidence must lie in [0,1]')
        action = _integer(self.action_id, 'cue action')
        for name, value in [('center', center), ('outward', outward), ('class_id', category),
                            ('confidence', confidence), ('action_id', action)]:
            object.__setattr__(self, name, value)


class VisibleSemanticMemoryV26:
    """Cumulative, position-associated artificial-color observations.

    Input order must be a contiguous paid history after the first supplied
    packet, with increasing time and unique frame IDs. An exact latest-packet
    retry is idempotent; semantic-channel-only edits are intentionally ignored.
    Returned cues are immutable snapshots, sorted by stable observation ID.
    """
    def __init__(self):
        self._identity = None; self._last_action = None; self._last_time = None
        self._last_fingerprint = None; self._frames = set(); self._tracks = []
        self._last_cues = ()

    def update(self, packet):
        d, k, t, timestamp = _packet_geometry(packet)
        rgb = np.asarray(packet.frame.color_rgb)
        if rgb.shape != (*d.shape, 3) or rgb.dtype != np.uint8:
            raise ValueError('aligned uint8 RGB required for actual color cues')
        identity = (packet.scene_id, packet.episode_id)
        action_id = int(packet.action_id)
        h = hashlib.sha256()
        for value in (d, k, t, rgb, packet.scan.ranges_m, packet.scan.world_from_laser):
            a = np.ascontiguousarray(value); h.update(str((a.dtype.str, a.shape)).encode()); h.update(a.tobytes())
        h.update(json.dumps([identity, packet.frame_id, action_id, timestamp, tuple(map(int, packet.position)),
            int(packet.heading), packet.action, packet.collision, packet.done, packet.sensor_source, packet.pose_source,
            float(packet.scan.timestamp_s), float(packet.scan.angle_min_rad), float(packet.scan.angle_increment_rad), float(packet.scan.range_max_m)],
            separators=(',', ':')).encode())
        fingerprint = h.hexdigest()
        if self._identity is not None:
            if identity != self._identity:
                raise ValueError('semantic memory cannot cross scene/episode identity')
            if packet.action_id == self._last_action and fingerprint == self._last_fingerprint:
                return self._last_cues
            if (packet.action_id != self._last_action+1 or timestamp <= self._last_time
                    or packet.frame_id in self._frames):
                raise ValueError('semantic memory requires unique, ordered, contiguous observed packets')
        code = np.zeros(d.shape, np.uint8)
        for category, color in MARKER_COLORS_V26:
            code[np.all(rgb == color, axis=2)] = category
        valid = (code != 0) & (d > .15) & (d <= packet.scan.range_max_m+1e-6)
        groups, count = label(valid, np.ones((3, 3), bool)); detections = []
        for component in range(1, count+1):
            rr, cc = np.nonzero(groups == component)
            if len(rr) < MINIMUM_MARKER_PIXELS:
                continue
            rays = np.column_stack((cc, rr, np.ones(len(rr))))@np.linalg.inv(k).T
            xyz = (rays*d[rr, cc, None])@t[:3, :3].T+t[:3, 3]
            median = np.median(xyz, axis=0); chosen = int(np.argmin(np.linalg.norm(xyz-median, axis=1)))
            center = xyz[chosen]; delta = t[:3, 3]-center; delta[2] = 0.
            norm = float(np.linalg.norm(delta))
            if norm < 1e-8:
                continue
            detections.append(dict(center=center, outward=delta/norm,
                votes={category: int(np.count_nonzero(code[rr, cc] == category)) for category, _ in MARKER_COLORS_V26},
                pixel=[int(rr[chosen]), int(cc[chosen])], valid_pixels=len(rr)))
        # One-to-one nearest-position matching; class never enters association.
        possibilities = sorted((float(np.linalg.norm(detection['center']-track['center'])), di, ti)
            for di, detection in enumerate(detections) for ti, track in enumerate(self._tracks)
            if np.linalg.norm(detection['center']-track['center']) <= TRACK_DISTANCE_M)
        matches = {}; used = set()
        for _, di, ti in possibilities:
            if di not in matches and ti not in used:
                matches[di] = ti; used.add(ti)
        for di, detection in enumerate(detections):
            if di in matches:
                track = self._tracks[matches[di]]; count = track['observations']+1
                track['center'] += (detection['center']-track['center'])/count
                track['observations'] = count
                for category in (2, 3):track['votes'][category] += detection['votes'][category]
            else:
                track = dict(cue_id=f'cue-{len(self._tracks):04d}', center=detection['center'].copy(),
                    outward=detection['outward'].copy(), observations=1, votes=detection['votes'].copy(),
                    first_frame=packet.frame_id, first_action=action_id, first_pixel=detection['pixel'])
                self._tracks.append(track)
            track.update(last_action=action_id, last_frame=packet.frame_id, last_pixel=detection['pixel'],
                last_valid_pixels=detection['valid_pixels'])
        cues = []
        for track in self._tracks:
            category = max((2, 3), key=lambda value: (track['votes'][value], -value))
            source = json.dumps(dict(kind='artificial_rgb_exact_color_consistency_not_calibrated_network',
                first_action=track['first_action'], first_pixel=track['first_pixel'], last_action=track['last_action'],
                last_pixel=track['last_pixel'], last_valid_pixels=track['last_valid_pixels'],
                observations=track['observations'], outward='first_observed_horizontal_line_of_sight'),
                sort_keys=True, separators=(',', ':'))
            cues.append(SemanticCueV26(track['cue_id'], tuple(map(float, track['center'])),
                tuple(map(float, track['outward'])), category,
                track['votes'][category]/sum(track['votes'].values()), track['last_action'], source))
        self._identity = identity; self._last_action = action_id; self._last_time = timestamp
        self._last_fingerprint = fingerprint; self._frames.add(packet.frame_id); self._last_cues = tuple(cues)
        return self._last_cues
