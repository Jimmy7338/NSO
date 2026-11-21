"""Bounded isolated native SDK probe: no guarded publisher, so only zero commands."""
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import struct
import subprocess
import threading
import time
import rospy
from std_msgs.msg import Int32

D = Path(__file__).resolve().parent
OUT = D / 'zero_sdk_probe'
OUT.mkdir(exist_ok=False)
v = json.loads((D / 'validation_v2.json').read_text())
binary = Path(v['candidate_binary'])
assert hashlib.sha256(binary.read_bytes()).hexdigest() == v['candidate_binary_sha256']
assert os.environ['ROS_MASTER_URI'] == 'http://127.0.0.1:11382'
rospy.init_node('nso_native_zero_sdk_observer', anonymous=True, disable_signals=True, disable_rosout=True)
can = socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
can.setsockopt(socket.SOL_CAN_RAW, socket.CAN_RAW_FILTER, b''.join(struct.pack('=II', cid, 0xc00007ff) for cid in (0x211, 0x221)))
can.setsockopt(socket.SOL_SOCKET, 35, 1)
can.bind(('can0',))
can.settimeout(.1)
stop = threading.Event()
lock = threading.Lock()
data = dict(velocity_frames=0, nonzero_velocity_frames=0, max_abs_v=0., max_abs_w=0., last_velocity=None, last_status=None, errors=[])

def receive():
    while not stop.is_set():
        try:
            raw, anc, flags, _ = can.recvmsg(16, 128)
        except socket.timeout:
            continue
        except Exception as exc:
            with lock: data['errors'].append(str(exc))
            return
        cid, dlc, payload = struct.unpack('=IB3x8s', raw)
        if dlc != 8 or flags & (8 | 32): continue
        stamp = next((struct.unpack('@ll', value[:struct.calcsize('@ll')]) for level, kind, value in anc if level == socket.SOL_SOCKET and kind == 35), None)
        if stamp is None: continue
        row = dict(kernel_stamp_ns=stamp[0]*1000000000+stamp[1], receipt_monotonic=time.monotonic(), raw_hex=raw.hex())
        with lock:
            if cid == 0x221:
                vx, w = struct.unpack('>hh', payload[:4]); row.update(v=vx*.001, w=w*.001)
                data['velocity_frames'] += 1
                data['nonzero_velocity_frames'] += int(vx != 0 or w != 0)
                data['max_abs_v'] = max(data['max_abs_v'], abs(row['v']))
                data['max_abs_w'] = max(data['max_abs_w'], abs(row['w']))
                data['last_velocity'] = row
            elif cid == 0x211:
                row.update(state=payload[0], mode=payload[1], error=struct.unpack('>H', payload[4:6])[0])
                data['last_status'] = row

def snapshot():
    with lock: return json.loads(json.dumps(data))

thread = threading.Thread(target=receive, daemon=True)
thread.start()
child = None
receipt = dict(scope='90s isolated actual SDK zero-only sender timing; no guarded publisher or navigation', duration_requested_s=90, candidate_binary_sha256=v['candidate_binary_sha256'], frozen_pilot_resumed=False)
try:
    time.sleep(1)
    before = snapshot()
    assert before['last_velocity'] and before['last_status']
    assert before['nonzero_velocity_frames'] == 0
    assert before['last_status']['state'] == 0 and before['last_status']['mode'] == 1 and before['last_status']['error'] == 0
    receipt['before'] = before
    prefix = '/nso_native_zero_sdk_probe'
    argv = [str(binary), '_execute:=true', '_session_id:=native_zero_sdk_probe',
            '_expected_boot_id:=' + Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            '_can_interface:=can0', '_guard_node:=' + prefix + '_unused_guard',
            '_timing_output:=' + str(OUT / 'timing.json'), '__name:=nso_native_zero_sdk_probe',
            '/nso_radius/guarded_cmd:=' + prefix + '/guarded_cmd',
            '/nso_radius/stop:=' + prefix + '/stop',
            '/nso_radius/driver_fault:=' + prefix + '/driver_fault']
    receipt['argv'] = argv
    with (OUT / 'native.log').open('xb') as log:
        child = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        receipt['pid'] = child.pid
        (OUT / 'running.json').write_text(json.dumps(receipt, indent=2)+'\n')
        first = rospy.wait_for_message(prefix + '/driver_fault', Int32, timeout=10)
        assert first.data == 0, ('startup fault', first.data)
        faults = []
        subscriber = rospy.Subscriber(prefix + '/driver_fault', Int32, lambda m: faults.append(dict(code=m.data, monotonic=time.monotonic())) if m.data and not faults else None, queue_size=1)
        started = time.monotonic()
        while time.monotonic()-started < 90 and child.poll() is None and not faults:
            current = snapshot()
            if current['errors'] or current['nonzero_velocity_frames']:
                receipt['early_stop_reason'] = 'nonzero_feedback_or_reader_error'
                break
            time.sleep(.1)
        receipt.update(elapsed_observation_s=time.monotonic()-started, pre_shutdown_faults=faults)
        os.kill(child.pid, signal.SIGINT)
        receipt['exit_code'] = child.wait(timeout=10)
        subscriber.unregister()
    time.sleep(1)
    receipt['after'] = snapshot()
except Exception as exc:
    receipt['error'] = type(exc).__name__ + ': ' + str(exc)
    raise
finally:
    if child is not None and child.poll() is None:
        os.kill(child.pid, signal.SIGINT)
        child.wait(timeout=10)
    stop.set(); thread.join(1); can.close()
    receipt['final_feedback'] = snapshot()
    receipt['limitations'] = ['Zero-only stationary test does not establish loaded motion stability or identify the earlier B1_C fault cause.', 'Ring contains last512 cycles and full first fault; no full-duration percentile claim.']
    (OUT / 'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
print(json.dumps(receipt))
