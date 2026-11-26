"""Tick evidence and actual instrumented main-loop body; no ROS/devices."""
import ast
import importlib.util
import json
from pathlib import Path
import tempfile
import threading
import time
import types
import unittest
from unittest.mock import Mock
P=Path(__file__).resolve().parents[1]/'runtime/visual_velocity_adapter.py'
spec=importlib.util.spec_from_file_location('adapter_timing',P);a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)

class TimingTests(unittest.TestCase):
    def test_original_boundary_and_backward(self):
        g=a.Gate();g.last_tick=0.
        self.assertEqual(g.decision(.08,1000.08),(0.,0.,'not_armed'))
        self.assertIsNone(g.fault)
        g=a.Gate();g.last_tick=0.;self.assertEqual(g.decision(.080001,1000.080001)[2],'adapter_control_tick_invalid')
        self.assertEqual(g.fault_evidence['delta_s'],.080001)
        self.assertEqual(g.fault_evidence['classification'],'tick_overrun')
        g=a.Gate();g.last_tick=1.;g.decision(.99,1000.99)
        self.assertEqual(g.fault_evidence['classification'],'clock_regressed')
        self.assertEqual(g.fault_evidence['previous_tick_monotonic'],1.)

    def test_preserve_first_fault_and_evidence(self):
        g=a.Gate();g.fail('original',{'source':'first'});g.last_tick=0.
        self.assertEqual(g.decision(1.,1001.),(0.,0.,'original'))
        self.assertEqual(g.fault_evidence,{'source':'first'})

    def test_bounded_same_outputs_and_once_save(self):
        ring=a.ControlTimingRing(4);plain=a.Gate();instrumented=a.Gate()
        for n in range(10):
            ring.begin();ring.mark('lock_wait');expected=plain.decision(n*.02,1000+n*.02)
            actual=instrumented.decision(n*.02,1000+n*.02);ring.mark('decision');ring.finish('continue')
            self.assertEqual(actual,expected)
        self.assertEqual(len(ring.rows),4)
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'fault.json';self.assertFalse(ring.save_fault_once(p,None,None));self.assertFalse(p.exists())
            self.assertTrue(ring.save_fault_once(p,'test',{}));before=p.read_bytes()
            self.assertFalse(ring.save_fault_once(p,'different',{}));self.assertEqual(before,p.read_bytes())
            self.assertEqual(len(json.loads(before)['rows']),4)

    def test_actual_loop_fault_row_and_stop_despite_diagnostic_io_error(self):
        tree=ast.parse(P.read_text());main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
        loop=next(n for n in ast.walk(main) if isinstance(n,ast.While) and isinstance(n.test,ast.UnaryOp) and isinstance(n.test.operand,ast.Call) and isinstance(n.test.operand.func,ast.Attribute) and n.test.operand.func.attr=='is_shutdown')
        module=ast.Module(body=[loop],type_ignores=[]);ast.fix_missing_locations(module)
        ring=a.ControlTimingRing(4);sent=[];calls=[]
        class Gate:
            fault=None;ever_armed=True;zero_published=0;proposal=None;odom=None;scan=None;depth_health=None;odom_health=None
            def decision(self,now,wall):
                calls.append(now)
                if len(calls)==2:self.fault='test_tick_fault'
                return 0.,0.,self.fault or 'not_armed'
            def fail(self,reason):self.fault=reason
        gate=Gate()
        def message():return types.SimpleNamespace(header=types.SimpleNamespace(),twist=types.SimpleNamespace(linear=types.SimpleNamespace(),angular=types.SimpleNamespace()))
        class Wall:
            def to_sec(self):return time.time()
            def to_nsec(self):return time.time_ns()
        rospy=types.SimpleNamespace(is_shutdown=lambda:False,Time=types.SimpleNamespace(now=Wall))
        publisher=types.SimpleNamespace(get_num_connections=lambda:1,publish=lambda m:sent.append((m.twist.linear.x,m.twist.angular.z)))
        ctx=dict(rospy=rospy,control_timing=ring,lock=threading.Lock(),time=types.SimpleNamespace(monotonic=time.monotonic,sleep=lambda _:None),gate=gate,publisher=publisher,journal=types.SimpleNamespace(error=None,add=lambda row:None),pending_deadline=0,last_stamp=0,sequence=0,TwistStamped=message,session='dry',next_state_monotonic=0,snapshot=lambda *args:{},publish_state=lambda state:None)
        stop=Mock()
        try:exec(compile(module,str(P),'exec'),ctx)
        finally:
            stop() # Existing authority must remain available despite diagnostic failures.
            with tempfile.TemporaryDirectory() as td:
                self.assertFalse(ring.save_fault_once(td,gate.fault,{}))
        stop.assert_called_once();self.assertEqual(sent,[(0.,0.),(0.,0.)]);self.assertEqual(len(ring.rows),2)
        fault=ring.rows[-1];self.assertEqual(fault['outcome'],'test_tick_fault')
        for k in ['lock_wait_ns','decision_ns','guarded_publish_ns','journal_add_ns','snapshot_ns','state_publish_ns']:self.assertIn(k,fault)
        self.assertIn('sleep_ns',ring.rows[0]);self.assertNotIn('sleep_ns',fault)
        # Diagnostic clock failure is non-throwing at every entry point.
        bad=a.ControlTimingRing(clock=lambda:(_ for _ in ()).throw(OSError('clock unavailable')))
        bad.begin();bad.mark('lock_wait');bad.finish('fault');self.assertTrue(bad.errors)

if __name__=='__main__':unittest.main()
