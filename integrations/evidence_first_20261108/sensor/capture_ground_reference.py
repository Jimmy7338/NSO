"""Three passive original sensor packets; no command publisher or CAN writes."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
from pathlib import Path
import copy
import hashlib
import json
import socket
import struct
import sys
import threading
import time
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'current_capture'
WORK = Path('/home/ouc/nso_robot/workspaces/nso_hybrid_20261029')
sys.path.insert(0, str(WORK / 'runtime'))
import rospy
import tf2_ros
from reference_observation import ReferenceObservationCollector

OUT.mkdir(exist_ok=False)
protocol = dict(
    schema='nso.passive_floor_reference_protocol.v1',
    written_unix_s=time.time(), number_of_frames=3,
    roles=['training', 'heldout_stationary', 'heldout_stationary'],
    delay_after_previous_capture_s=3.,
    ROI_uv=[360, 480, 1100, 710], stride=4, seed=0,
    ransac_trials=120, residual_threshold_m=.01,
    calibration_parameter='scalar floor Z at actual base XY from training-frame fitted plane',
    ROI_identity='visually verify after capture; do not refit alternate crops if rejected',
    no_policy_change=True, no_motion_commands=True, no_sensor_parameter_change=True,
    camera_mount='user-requested legacy transform; verify actual capture TF',
    scope='Sensor-internal consistency candidate only; same stationary pose, not independent physical accuracy truth',
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(OUT / 'protocol.json').write_text(json.dumps(protocol, ensure_ascii=False, indent=2)+'\n')
rospy.init_node('nso_passive_floor_reference_capture', disable_signals=True, disable_rosout=True)
tfbuf = tf2_ros.Buffer(cache_time=rospy.Duration(30))
listener = tf2_ros.TransformListener(tfbuf)
stop = threading.Event()
lock = threading.Lock()
motion = {}
can = socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
can.setsockopt(socket.SOL_CAN_RAW, socket.CAN_RAW_FILTER, b''.join(
    struct.pack('=II', cid, 0xc00007ff) for cid in (0x211, 0x221)))
can.setsockopt(socket.SOL_SOCKET, 35, 1)
can.bind(('can0',))
can.settimeout(.1)

def receive():
    while not stop.is_set():
        try:
            raw, anc, flags, _ = can.recvmsg(16, 128)
        except socket.timeout:
            continue
        cid, dlc, payload = struct.unpack('=IB3x8s', raw)
        if flags & (8 | 32) or dlc != 8 or cid not in (0x211, 0x221):
            continue
        stamp = next((struct.unpack('@ll', v[:struct.calcsize('@ll')])
                      for level, kind, v in anc if level == socket.SOL_SOCKET and kind == 35), None)
        if stamp is None:
            continue
        row = dict(receipt_monotonic=time.monotonic(), kernel_unix_s=stamp[0]+stamp[1]/1e9,
                   raw_frame_hex=raw.hex())
        if cid == 0x211:
            row.update(state=payload[0], mode=payload[1], error=struct.unpack('>H', payload[4:6])[0])
        else:
            v, w = struct.unpack('>hh', payload[:4]); row.update(v=v*.001, w=w*.001)
        with lock:
            motion['status' if cid == 0x211 else 'velocity'] = row

def snapshot():
    with lock:
        return copy.deepcopy(motion)

thread = threading.Thread(target=receive, daemon=True)
thread.start()
collector = ReferenceObservationCollector(tfbuf)
records = []
try:
    for index in range(3):
        if index:
            time.sleep(3.)
        path = Path(collector.capture_live_joint(OUT / 'raw', lambda: None, snapshot))
        with np.load(path, allow_pickle=False) as z:
            receipt = json.loads(str(z['receipt_json']))
        tf = receipt['historical_tf']
        base = np.asarray(tf['odom_from_base']['matrix'])
        expected = np.array([[0.,0.,1.,.061],[-1.,0.,0.,.0591],[0.,-1.,0.,.303],[0.,0.,0.,1.]])
        actual = np.asarray(tf['base_from_camera']['matrix'])
        row = dict(index=index, role=protocol['roles'][index], path=str(path),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            depth_stamp_ns=receipt['headers']['depth']['stamp_ns'],
            base_odom_xy_yaw=[float(base[0,3]), float(base[1,3]), float(np.arctan2(base[1,0],base[0,0]))],
            camera_legacy_max_abs_error=float(np.max(np.abs(actual-expected))),
            CAN_before=receipt['motion_before'], CAN_after=receipt['motion_after'])
        records.append(row)
        (OUT / 'receipt.json').write_text(json.dumps(dict(protocol=protocol,frames=records,
            no_motion_commands=True, complete=len(records)==3),ensure_ascii=False,indent=2)+'\n')
        print(json.dumps(row,ensure_ascii=False), flush=True)
finally:
    collector.close()
    stop.set()
    thread.join(1)
    can.close()
    rospy.signal_shutdown('passive three-packet capture ended')
