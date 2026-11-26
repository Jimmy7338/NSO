"""Execute actual observer main with fake ROS/process/CAN; no robot or ROS import."""
import json
from pathlib import Path
import runpy
import signal
import sys
import tempfile
import types
import unittest
import numpy as np
from unittest.mock import patch

OBSERVER=Path('/home/ouc/nso_robot/analyses/sensor_repair_20261108/run_with_fault_ring.py')

class TestActualObserver(unittest.TestCase):
    def test_sensor_callback_then_repeated_signal(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);command=root/'command';command.mkdir();report=root/'report'
            callbacks=[];handlers={};sent=[];polls=[];closed=[]
            class Msg:
                _type='test/ActualSource'
                _connection_header={'callerid':'/test_source'}
                header=types.SimpleNamespace(stamp=types.SimpleNamespace(to_nsec=lambda:1234))
                def serialize(self,b):b.write(b'original_source')
            class Subscriber:
                def __init__(self,topic,kind,cb,**kw):callbacks.append(cb)
                def unregister(self):closed.append('subscriber')
            rospy=types.ModuleType('rospy');rospy.init_node=lambda *a,**k:None
            rospy.Subscriber=Subscriber;rospy.signal_shutdown=lambda reason:closed.append('shutdown')
            class FakeProcess:
                pid=12345;returncode=None
                def __init__(self,*a,**k):pass
                def poll(self):
                    polls.append(1)
                    if len(polls)==1:
                        for cb in callbacks:cb(Msg())
                        # Sensor callback must not invoke signal handler or cause forwarding.
                        self_test.assertEqual(sent,[])
                        self_test.assertEqual(list(command.glob('observer_lifecycle/*signal_received*')),[])
                        return None
                    if len(polls)==2:
                        self_test.assertEqual(sent,[])
                        handlers[signal.SIGINT](signal.SIGINT,None)
                        handlers[signal.SIGTERM](signal.SIGTERM,None)
                        return None
                    self.returncode=0;return 0
                def send_signal(self,sig):sent.append(sig)
                def wait(self,timeout=None):self.returncode=0;return 0
            class Capture:
                def __init__(self):self.reader=types.SimpleNamespace(close=lambda:closed.append('can'))
                def save_and_close(self,path):Path(path).write_text('{}');closed.append('can_saved')
            mods={'rospy':rospy}
            for package,name in [('geometry_msgs','PoseStamped'),('nav_msgs','Odometry'),('sensor_msgs','LaserScan')]:
                m=types.ModuleType(package+'.msg');setattr(m,name,Msg)
                mods[package]=types.ModuleType(package);mods[package+'.msg']=m
            canmod=types.ModuleType('bounded_can_capture');canmod.BoundedCanCapture=Capture;mods['bounded_can_capture']=canmod
            self_test=self
            with patch.dict(sys.modules,mods),patch.object(sys,'argv',[str(OBSERVER),'--report',str(report),'--command-dir',str(command)]),patch('signal.signal',side_effect=lambda sig,handler:handlers.__setitem__(sig,handler)),patch('subprocess.Popen',FakeProcess),patch('time.sleep',lambda _:None):
                runpy.run_path(str(OBSERVER),run_name='__main__')
            self.assertEqual(sent,[signal.SIGINT])
            summary=json.loads((command/'observer_result.json').read_text())
            self.assertTrue(summary['raw_fault_ring_retained']);self.assertEqual(summary['returncode'],0)
            ring=json.loads((command/'fault_ring/failure_scan_ring.json').read_text())
            self.assertEqual(len(ring['messages']),3)
            self.assertEqual({r['source_stamp_ns'] for r in ring['messages']},{1234})
            self.assertTrue((command/'fault_ring/failure_can_ring.json').exists())
            self.assertIn('shutdown',closed)
            events=[json.loads(p.read_text())['event'] for p in (command/'observer_lifecycle').glob('*.json')]
            self.assertEqual(events.count('signal_received'),2)
            self.assertEqual(events.count('mission_sigint_forward_begin'),1)

if __name__=='__main__':unittest.main()
