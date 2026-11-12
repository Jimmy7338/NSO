"""Run the declared NSO episode; keep bounded raw sensor data only on interruption.

Motion, collision checking, goals and shutdown belong to the existing mission.
This observer adds no policy condition and never publishes a command.
"""
from collections import deque
import io
import json
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan

ROOT=Path(__file__).resolve().parent
REPORT=Path('/home/ouc/nso_robot/reports/nso-hybrid-20261108/calibrated_S24_02')
sys.path.insert(0,'/home/ouc/nso_robot/analyses/forward_controller_20261107')
from bounded_can_capture import BoundedCanCapture

def main():
    assert not REPORT.exists(), 'Existing report: inspect its owner/result instead of restarting'
    rospy.init_node('nso_bounded_fault_observer',anonymous=True,disable_signals=True)
    spec={'scan':('/scan',LaserScan), 'csm_pose':('/nso_csm/pose_stamped',PoseStamped),
          'raw_odom':('/nso_csm/odom',Odometry)}
    rings={k:deque(maxlen=150) for k in spec}; lock=threading.Lock()
    def receive(name,msg):
        with lock: rings[name].append((time.time_ns(),msg))
    subs=[rospy.Subscriber(topic,kind,lambda m,k=k:receive(k,m),queue_size=10)
          for k,(topic,kind) in spec.items()]
    can=None; process=None; result=None
    try:
        can=BoundedCanCapture()
        with (ROOT/'mission.log').open('xb',buffering=0) as log:
            process=subprocess.Popen([str(ROOT/'RUN_NEXT.sh')],stdout=log,stderr=subprocess.STDOUT)
            (ROOT/'process.json').write_text(json.dumps(dict(pid=process.pid,
                argv=[str(ROOT/'RUN_NEXT.sh')],report=str(REPORT),started_unix_s=time.time(),
                scope='Declared S24 calibrated-camera integration plus passive bounded fault observer'),indent=2))
            while process.poll() is None: time.sleep(.25)
        result_path=REPORT/'result.json'
        if result_path.exists():result=json.loads(result_path.read_text())
        interrupted=(process.returncode!=0 or not result or bool(result.get('execution_interrupted')) or bool(result.get('method_failed')))
        if interrupted:
            out=ROOT/'fault_ring';out.mkdir(exist_ok=False)
            packed={};meta=[]
            with lock: copied={k:list(v) for k,v in rings.items()}
            for name,rows in copied.items():
                for i,(received,msg) in enumerate(rows):
                    key='%s_%03d'%(name,i);buffer=io.BytesIO();msg.serialize(buffer)
                    packed[key]=np.frombuffer(buffer.getvalue(),dtype=np.uint8)
                    meta.append(dict(key=key,topic=spec[name][0],type=msg._type,
                                     source_stamp_ns=msg.header.stamp.to_nsec(),receipt_unix_ns=received,
                                     caller=msg._connection_header.get('callerid')))
            np.savez_compressed(out/'failure_scan_ring.npz',**packed)
            (out/'failure_scan_ring.json').write_text(json.dumps(dict(messages=meta,
                scope='At most150 original source messages per topic around this execution interruption; fresh-epoch raw odom and original matcher pose are retained without coordinate alignment'),indent=2))
            can.save_and_close(out/'failure_can_ring.json');can=None
        summary=dict(returncode=process.returncode,status=None if result is None else result.get('status'),
                     reason=None if result is None else result.get('reason'),
                     raw_fault_ring_retained=interrupted,report=str(REPORT))
        (ROOT/'observer_result.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
    finally:
        if process is not None and process.poll() is None:
            process.send_signal(signal.SIGINT);process.wait(timeout=20)
        if can is not None:can.reader.close()
        for sub in subs:sub.unregister()
        rospy.signal_shutdown('bounded episode observer finished')

if __name__=='__main__':main()
