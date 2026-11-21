"""Static A/B/A camera-input diagnostic; no actuator, map clear or policy edit."""
import json
import hashlib
from pathlib import Path
import sys
import threading
import time

import numpy as np
import rospy
import tf2_ros
import message_filters
import dynamic_reconfigure.client
from sensor_msgs.msg import Image, CameraInfo
from tf.transformations import quaternion_matrix
from depth_mask import anchor_depth_support

OUT = Path(__file__).parent / 'confidence_probe_50_100'
OUT.mkdir(exist_ok=False)
BASE = Path('/home/ouc/nso_robot')
CFG = json.loads((BASE/'analyses/expand_scene_20261110/configuration.json').read_text())
PLANNING = np.asarray(CFG['configuration']['planning_from_odom'])
rospy.init_node('nso_static_depth_confidence_probe', disable_signals=True, disable_rosout=True)
buffer = tf2_ros.Buffer(cache_time=rospy.Duration(30))
listener = tf2_ros.TransformListener(buffer)
client = dynamic_reconfigure.client.Client('/zed2i/zed_node', timeout=5)
lock = threading.Lock()
pending = []

def f32_image(msg):
    assert msg.encoding == '32FC1' and msg.step >= msg.width * 4
    assert len(msg.data) == msg.height * msg.step
    dtype = np.dtype('>f4' if msg.is_bigendian else '<f4')
    return np.ndarray((msg.height, msg.width), dtype=dtype, buffer=msg.data,
                      strides=(msg.step, 4)).astype(np.float32, copy=True)

def receive(depth, confidence, info):
    with lock:
        if len(pending) < 30:
            pending.append((depth, confidence, info))

subs = [message_filters.Subscriber('/zed2i/zed_node/depth/depth_registered', Image),
        message_filters.Subscriber('/zed2i/zed_node/confidence/confidence_map', Image),
        message_filters.Subscriber('/zed2i/zed_node/depth/camera_info', CameraInfo)]
sync = message_filters.TimeSynchronizer(subs, 10)
sync.registerCallback(receive)
original = client.get_configuration(timeout=4)
old = {k: original[k] for k in ('depth_confidence', 'depth_texture_conf')}
receipt = dict(scope='Static input diagnostic at user-confirmed empty old anchor; no robot motion or research-policy changes',
               original=old, phases=[], errors=[], full_frames_retained='first/last canonical depth and confidence only per phase')

def save():
    (OUT/'receipt.json').write_text(json.dumps(receipt, indent=2, allow_nan=False)+'\n')

def configure(settings):
    # Installed wrapper uses switch(level), not a bit mask; 1|2 selects
    # POINTCLOUD_FREQ. Submit each parameter separately and verify readback.
    for key, value in settings.items():
        client.update_configuration({key: value})
    time.sleep(.4)
    actual = client.get_configuration(timeout=3)
    assert all(actual[k] == v for k, v in settings.items()), 'SDK reconfigure reverted'
    return actual

def phase(name, settings):
    actual = configure(settings)
    assert all(actual[k] == v for k, v in settings.items())
    start = time.time() + .8
    with lock:
        pending.clear()
    rows = []
    deadline = time.monotonic() + 15
    while len(rows) < 20 and time.monotonic() < deadline:
        with lock:
            item = pending.pop(0) if pending else None
        if item is None:
            time.sleep(.02)
            continue
        depth, confidence, info = item
        if depth.header.stamp.to_sec() < start:
            continue
        assert depth.header.stamp == confidence.header.stamp == info.header.stamp
        assert depth.header.frame_id == confidence.header.frame_id == info.header.frame_id
        raw = f32_image(depth)
        conf = f32_image(confidence)
        assert raw.shape == conf.shape
        d = raw[::4, ::4].copy()
        with np.errstate(invalid='ignore'):
            d[~np.isfinite(d) | (d <= 0) | (d > 4)] = 0
        c = conf[::4, ::4].copy()
        intrinsic = np.asarray(info.K, dtype=float).reshape(3, 3).copy()
        intrinsic[:2] /= 4
        t = buffer.lookup_transform('odom', depth.header.frame_id, depth.header.stamp, rospy.Duration(2))
        q = t.transform.rotation
        matrix = quaternion_matrix([q.x, q.y, q.z, q.w])
        p = t.transform.translation
        matrix[:3, 3] = [p.x, p.y, p.z]
        support = anchor_depth_support(d, intrinsic, PLANNING @ matrix)
        uv = np.asarray(support['uv'], dtype=int).reshape(-1, 2)
        values = c[uv[:, 1], uv[:, 0]] if len(uv) else np.array([], dtype=float)
        row = dict(stamp_ns=depth.header.stamp.to_nsec(), frame=depth.header.frame_id,
                   source_publisher=depth._connection_header.get('callerid'), valid_fraction=float(np.mean(d > 0)),
                   depth_sha256=hashlib.sha256(d.tobytes()).hexdigest(),
                   confidence_sha256=hashlib.sha256(c.tobytes()).hexdigest(),
                   planning_from_camera=(PLANNING @ matrix).tolist(), anchor=support,
                   anchor_confidence=values.tolist())
        rows.append(row)
        if len(rows) in (1, 20):
            np.savez_compressed(OUT/(name+'_'+str(len(rows))+'.npz'), depth=d, confidence=c, intrinsic=intrinsic,
                                planning_from_camera=PLANNING @ matrix, stamp_ns=np.asarray(row['stamp_ns']))
    assert len(rows) == 20, 'Synchronized input timeout: '+str(len(rows))
    result = dict(name=name, settings=settings, frames=rows,
                  mean_valid_fraction=float(np.mean([x['valid_fraction'] for x in rows])),
                  occupied_anchor_cells=[len(x['anchor']['cells']) for x in rows],
                  occupied_anchor_points=[x['anchor']['point_count'] for x in rows])
    receipt['phases'].append(result)
    save()
    print(json.dumps({k:v for k,v in result.items() if k != 'frames'}), flush=True)

try:
    phase('baseline', old)
    phase('candidate_50_100', dict(depth_confidence=50, depth_texture_conf=100))
    phase('restored', old)
except Exception as exc:
    receipt['errors'].append(repr(exc))
    raise
finally:
    final = configure(old)
    receipt['restored'] = all(final[k] == v for k,v in old.items())
    receipt['finished_unix_s'] = time.time()
    save()
    for sub in subs:
        sub.unregister()
    rospy.signal_shutdown('Static depth-input probe complete')
