"""Authorized snapshot sender: 30 s isolated SDK zero only, no command publisher."""
from dataclasses import asdict
from collections import deque
import json
from pathlib import Path
import signal
import subprocess
import sys
import time
import rospy
from std_msgs.msg import Int32

BASE = Path('/home/ouc/nso_robot')
OUT = Path(__file__).parent/'zero_sdk_probe'
OUT.mkdir(exist_ok=False)
sys.path.insert(0, str(BASE/'workspaces/nso_hybrid_20261029/runtime'))
from can_control_feedback import CanVelocityFeedback

rospy.init_node('nso_snapshot_zero_observer', disable_signals=True, disable_rosout=True)
faults = []
sub = rospy.Subscriber('/nso_snapshot_zero/driver_fault', Int32,
                       lambda m: faults.append(dict(monotonic=time.monotonic(), code=m.data)) if m.data else None,
                       queue_size=10)
reader = CanVelocityFeedback('can0')
reader.samples = deque(maxlen=2200)
reader.start()
child = None
receipt = dict(scope='Actual SDK zero-only isolated topic validation; no command publisher or navigation',
               user_authorization='采用新版本做短程实测', requested_seconds=30, errors=[])
try:
    rospy.sleep(1)
    assert reader.health_reason() is None
    with reader.lock:
        assert reader.samples and all(s.vx == 0 and s.w == 0 for s in reader.samples)
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    args = [str(BASE/'staging/ans-live-20261110-snapshot-v1/bin/nso_scout_attended_node'),
            '_execute:=true', '_session_id:=snapshot_zero_sdk_probe', '_expected_boot_id:='+boot,
            '_can_interface:=can0', '_guard_node:=/nso_snapshot_zero_unused_guard',
            '_timing_output:='+str(OUT/'timing.json'), '__name:=nso_snapshot_zero',
            '/nso_radius/guarded_cmd:=/nso_snapshot_zero/guarded_cmd',
            '/nso_radius/stop:=/nso_snapshot_zero/stop',
            '/nso_radius/driver_fault:=/nso_snapshot_zero/driver_fault']
    with (OUT/'native.log').open('xb') as log:
        child = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT)
    receipt.update(argv=args, pid=child.pid, started_unix_s=time.time())
    (OUT/'running.json').write_text(json.dumps(receipt, indent=2)+'\n')
    start = time.monotonic()
    while time.monotonic()-start < 30:
        if child.poll() is not None:
            raise RuntimeError('Native exited: '+str(child.returncode))
        if faults:
            raise RuntimeError('Native pre-stop fault: '+str(faults[-1]))
        if reader.health_reason() is not None:
            raise RuntimeError(reader.health_reason())
        with reader.lock:
            if any(s.vx != 0 or s.w != 0 for s in reader.samples):
                raise RuntimeError('Nonzero CAN feedback during zero-only probe')
        rospy.sleep(.1)
    receipt['elapsed_s'] = time.monotonic()-start
    receipt['pre_shutdown_faults'] = list(faults)
except Exception as exc:
    receipt['errors'].append(repr(exc))
finally:
    if child is not None and child.poll() is None:
        child.send_signal(signal.SIGINT)
        child.wait(timeout=8)
    receipt['returncode'] = None if child is None else child.returncode
    rospy.sleep(.5)
    reader.close()
    with reader.lock:
        samples = [asdict(s) for s in reader.samples]
    receipt.update(CAN_samples=len(samples), nonzero_CAN_samples=sum(s['vx'] != 0 or s['w'] != 0 for s in samples),
                   source=reader.source_receipt(), finished_unix_s=time.time())
    (OUT/'can.json').write_text(json.dumps(samples, separators=(',', ':'))+'\n')
    (OUT/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    sub.unregister()
    rospy.signal_shutdown('Zero-only probe finished')
print(json.dumps({k:v for k,v in receipt.items() if k not in ['argv','source']}), flush=True)
if receipt['errors']:
    raise SystemExit(1)
