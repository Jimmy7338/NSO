"""Validate recorded ROS source receipts without importing ROS or opening devices.

Receipts describe actual source headers and historical TF lookup responses.
They cannot recover provenance absent from an older recording, and do not
independently establish the physical camera extrinsic calibration.
"""
import hashlib
import math

import numpy as np


SCHEMA = 'nso.paid_rgbd_source.v1'


def array_receipt(array):
    """Hash the exact C-order array bytes; preserve dtype and shape separately."""
    value = np.asarray(array)
    if value.dtype.hasobject:
        raise ValueError('object arrays cannot be source evidence')
    return dict(dtype=value.dtype.str, shape=list(value.shape),
                sha256=hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest())


def tf_timestamp_rounding_bound_ns(query_stamp_ns):
    """Two epoch-double ULPs, capped at 1 us, for tf2 Python serialization.

    ROS image headers remain exact integers. tf2's Python/C++ lookup boundary
    converts time through an epoch double, so its returned integer timestamp
    may differ at submicrosecond precision. Both raw integers remain recorded.
    """
    return min(1000, max(1, int(math.ceil(abs(float(np.spacing(query_stamp_ns/1e9)))*2e9))))


def _integer(value, name, positive=False):
    if type(value) is not int or value < (1 if positive else 0):
        raise ValueError(name+' requires a raw integer source value')
    return value


def _seconds_match(value, stamp_ns, name):
    if not isinstance(value, (int, float, np.number)) or not math.isfinite(float(value)):
        raise ValueError(name+' requires a finite component timestamp')
    expected = stamp_ns / 1e9
    # Epoch doubles cannot retain nanosecond precision. The original integer
    # fields are checked exactly; floating fields may differ by two ULPs.
    tolerance = max(1e-12, abs(float(np.spacing(expected)))*2.)
    if abs(float(value)-expected) > tolerance:
        raise ValueError(name+' differs from its actual source stamp')


def _matches(receipt, array, name):
    if not isinstance(receipt, dict) or set(receipt) != {'dtype', 'shape', 'sha256'}:
        raise ValueError(name+' array receipt missing or malformed')
    if receipt != array_receipt(array):
        raise ValueError(name+' source payload hash/dtype/shape mismatch')


def validate_source_provenance(frame, maximum_slop_s=.05):
    """Return an audit summary after checking a newly recorded paid RGB-D frame.

    ``frame`` is the undecimated RGBDFrameInput. Old recordings with no receipt
    must be explicitly handled by the caller, rather than retroactively filled.
    An odom<-camera transform has a dynamic component on this robot; a zero
    response timestamp therefore cannot prove a usable live scientific pose.
    """
    if not math.isfinite(maximum_slop_s) or not 0 <= maximum_slop_s <= .1:
        raise ValueError('bounded source synchronization slop required')
    p = getattr(frame, 'source_provenance', None)
    if not isinstance(p, dict) or p.get('schema') != SCHEMA:
        raise ValueError('paid RGBD source provenance absent or unsupported')
    if p.get('frame_id') != frame.frame_id:
        raise ValueError('source receipt frame ID differs')
    if p.get('world_frame') != 'odom' or p.get('navigation_frame') != 'map':
        raise ValueError('source world/navigation frame declaration differs')
    sources, synchronization = p.get('sources'), p.get('synchronization')
    if not isinstance(sources, dict) or set(sources) != {'rgb', 'depth', 'camera_info'}:
        raise ValueError('all actual RGB/depth/CameraInfo source headers required')
    if not isinstance(synchronization, dict) or synchronization.get('policy') != 'exact_header_stamp':
        raise ValueError('exact original header synchronization receipt required')
    if synchronization.get('all_header_stamps_equal') is not True or synchronization.get('all_optical_frames_equal') is not True:
        raise ValueError('source headers were not exactly aligned')
    query = _integer(synchronization.get('query_stamp_ns'), 'TF query stamp', True)
    stamps, optical_frames = [], []
    for kind in ('rgb', 'depth', 'camera_info'):
        row = sources[kind]
        if not isinstance(row, dict):raise ValueError('source header receipt requires an object')
        stamps.append(_integer(row.get('stamp_ns'), kind+' stamp', True))
        optical_frames.append(row.get('frame_id'))
    if any(stamp != query for stamp in stamps):
        raise ValueError('original source header stamps differ from exact TF query')
    if any(not isinstance(name, str) or not name for name in optical_frames) or len(set(optical_frames)) != 1:
        raise ValueError('RGB/depth/CameraInfo optical frames differ')
    optical = optical_frames[0]
    if max(stamps)-min(stamps) > maximum_slop_s*1e9:
        raise ValueError('source synchronization slop exceeded')
    _seconds_match(frame.timestamp_s, query, 'frame timestamp')
    _seconds_match(frame.rgb_timestamp_s, sources['rgb']['stamp_ns'], 'RGB timestamp')
    _seconds_match(frame.depth_timestamp_s, sources['depth']['stamp_ns'], 'depth timestamp')
    rgb, depth, k = np.asarray(frame.rgb), np.asarray(frame.depth_m), np.asarray(frame.intrinsic)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or depth.shape != rgb.shape[:2]:
        raise ValueError('paid RGB/depth shapes differ')
    height, width = depth.shape
    for kind in ('rgb', 'depth', 'camera_info'):
        if (sources[kind].get('height'), sources[kind].get('width')) != (height, width):
            raise ValueError(kind+' original source image dimensions differ')
    calibration = sources['camera_info']
    for name, size in [('K', 9), ('P', 12), ('R', 9)]:
        values = np.asarray(calibration.get(name), dtype=float)
        if values.shape != (size,) or not np.isfinite(values).all():
            raise ValueError('actual CameraInfo '+name+' required')
    if k.shape != (3, 3) or not np.array_equal(np.asarray(calibration['K']).reshape(3, 3), k):
        raise ValueError('paid intrinsic differs from actual CameraInfo K')
    if not isinstance(calibration.get('D'), list) or not np.isfinite(np.asarray(calibration['D'], dtype=float)).all():
        raise ValueError('actual CameraInfo distortion coefficients required')
    if not isinstance(calibration.get('distortion_model'), str):
        raise ValueError('actual CameraInfo distortion model required')
    payloads = p.get('payload_hashes', {})
    for name, value in [('rgb', rgb), ('depth_m', depth), ('intrinsic', k),
                        ('world_from_camera', frame.world_from_camera)]:
        _matches(payloads.get(name), value, name)
    transforms = p.get('transforms')
    if not isinstance(transforms, dict) or set(transforms) != {'world_from_camera', 'map_from_camera'}:
        raise ValueError('scientific and navigation historical TF receipts required')
    response_stamps = {}
    for name, target in [('world_from_camera', 'odom'), ('map_from_camera', 'map')]:
        row = transforms[name]
        if not isinstance(row, dict):raise ValueError('TF source receipt requires an object')
        if (row.get('target_frame'), row.get('source_frame'), row.get('response_target_frame'),
                row.get('response_source_frame')) != (target, optical, target, optical):
            raise ValueError(name+' historical TF frames differ')
        if _integer(row.get('query_stamp_ns'), name+' query stamp', True) != query:
            raise ValueError(name+' historical TF queried at a different source time')
        response = _integer(row.get('response_stamp_ns'), name+' response stamp')
        bound=tf_timestamp_rounding_bound_ns(query)
        difference=response-query
        if response<=0 or abs(difference)>bound:
            raise ValueError(name+' dynamic historical TF response exceeds serialization rounding bound')
        if row.get('response_query_delta_ns')!=difference or row.get('maximum_serialization_rounding_ns')!=bound:
            raise ValueError(name+' TF response rounding receipt differs from actual raw stamps')
        response_stamps[name] = response
        matrix = np.asarray(row.get('matrix'), dtype=float)
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all() or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-10):
            raise ValueError(name+' invalid historical TF matrix')
        rotation = matrix[:3, :3]
        if not np.allclose(rotation.T@rotation, np.eye(3), atol=1e-5) or abs(np.linalg.det(rotation)-1.) > 1e-5:
            raise ValueError(name+' historical TF matrix is not rigid')
        if row.get('timestamp_semantics') != 'historical_lookup_response_header':
            raise ValueError(name+' actual TF response timestamp semantics missing')
        _matches(payloads.get(name), matrix, name)
        if name == 'world_from_camera' and not np.array_equal(matrix, np.asarray(frame.world_from_camera)):
            raise ValueError('scientific pose differs from historical TF response')
    _seconds_match(frame.pose_timestamp_s, response_stamps['world_from_camera'], 'pose response timestamp')
    return dict(schema=SCHEMA, validated=True, exact_header_synchronization=True,
                frame_id=frame.frame_id, query_stamp_ns=query, optical_frame=optical,
                world_frame='odom', navigation_frame='map',
                pose_response_stamp_ns=response_stamps['world_from_camera'],
                tf_response_serialization_bound_ns=tf_timestamp_rounding_bound_ns(query),
                maximum_observed_tf_stamp_delta_ns=max(abs(value-query) for value in response_stamps.values()),
                physical_extrinsic_calibration_verified=False,
                proof_scope='exact recorded source headers and payload hashes; historical TF responses within two epoch-double ULPs capped at 1us')
