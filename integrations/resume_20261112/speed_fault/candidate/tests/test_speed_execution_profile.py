"""Offline replay of S96 command/feedback boundary, no ROS or hardware."""
import ast
import json
import struct
import unittest
from pathlib import Path
from test_visual_velocity_adapter import adapter, ready_gate, hardware, vision

class ExecutionProfileTests(unittest.TestCase):
    def test_s96_raw_proposal_and_legacy(self):
        for cap,expected in [(.05,.05),(.1,.1)]:
            g=adapter.Gate(maximum_linear_m_s=cap)
            g.propose([.12,0,0,0,0,0],10.,'/nso_visual/move_base')
            self.assertEqual(g.proposal['v'],expected)
            self.assertEqual(g.proposal['receipt'],10.)
            self.assertEqual(adapter.snapshot(g,10.)['maximum_linear_m_s'],cap)
        self.assertEqual(adapter.configuration({})['maximum_linear_m_s'],.1)

    def test_ordinary_proposal_identical_and_reverse_phase_preserved(self):
        for cap in [.05,.1]:
            g=adapter.Gate(maximum_linear_m_s=cap,nso_correction_reverse=True,mission_mode='nso')
            g.session_id='s';g.armed=True
            for v in [0,.01111111138,.04066666588,.05]:
                g.propose([v,0,0,0,0,-.17],10.,'/nso_visual/move_base')
                self.assertEqual((g.proposal['v'],g.proposal['w']),(v,-.17))
            self.assertTrue(g.set_correction_reverse(True,'/nso_onboard_supervisor','s')[0])
            g.propose([-.12,0,0,0,0,.4],10.,'/nso_visual/move_base')
            self.assertEqual((g.proposal['v'],g.proposal['w']),(-.05,.3))
            g.set_correction_reverse(False,'/nso_onboard_supervisor','s')
            self.assertEqual(g.proposal['v'],0)

    def test_exact_limit_and_recorded_fault_preserved_with_immutable_evidence(self):
        for cap in [.05,.1]:
            g=adapter.Gate(maximum_linear_m_s=cap);g.ever_armed=True
            g.propose([.12,0,0,0,0,0],9.9,'/nso_visual/move_base')
            g.hardware(0x211,bytes([0,1,0,0,0,0,0,0]),9.99,9.991)
            g.hardware(0x221,struct.pack('>hh',150,450)+bytes(4),10.,10.001)
            self.assertIsNone(g.fault)
            g.hardware(0x221,bytes.fromhex('009affd800000000'),10.02,10.021)
            self.assertEqual(g.fault,'chassis_speed_feedback_limit')
            e=g.fault_evidence
            self.assertEqual(e['can_id'],'0x221')
            self.assertEqual(e['payload_hex'],'009affd800000000')
            self.assertEqual(e['limits'],dict(maximum_abs_v_m_s=.15,maximum_abs_w_rad_s=.45))
            self.assertEqual((e['can_velocity']['v'],e['can_velocity']['w']),(.154,-.04))
            self.assertEqual((e['source_monotonic'],e['receipt_monotonic']),(10.02,10.021))
            self.assertEqual(e['proposal']['v'],cap)
            self.assertEqual(e['can_status']['mode'],1)
            g.proposal['v']=0;g.can_status['mode']=0
            self.assertEqual(e['proposal']['v'],cap);self.assertEqual(e['can_status']['mode'],1)
            self.assertEqual(adapter.snapshot(g,10.02)['fault_evidence'],e)

    def test_cap_does_not_refresh_expired_proposal(self):
        g=ready_gate();g.maximum_linear_m_s=.05
        self.assertEqual(g.arm(2.,1002.),(True,'armed_once'))
        g.propose([.12,0,0,0,0,0],2.01,'/nso_visual/move_base')
        hardware(g,2.1);hardware(g,2.22);vision(g,2.22)
        self.assertEqual(g.decision(2.22,1002.22),(0.,0.,'proposal_expired'))
        self.assertEqual(g.proposal['receipt'],2.01)

    def test_bad_caps_rejected_both_entrypoints(self):
        for value in [None,True,False,'0.05',0,-.05,.1001,float('inf'),float('nan'),[],{}]:
            for make in [lambda:adapter.Gate(maximum_linear_m_s=value),lambda:adapter.configuration({'maximum_linear_m_s':value})]:
                with self.assertRaises(ValueError):make()

    def test_mission_metadata_and_started_envelope_use_actual_cap(self):
        runtime=Path(__file__).resolve().parents[1]/'runtime'
        a=(runtime/'visual_velocity_adapter.py').read_text();m=(runtime/'mission.py').read_text()
        self.assertIn('motion_envelope=dict(max_linear_m_s=gate.maximum_linear_m_s',a)
        tree=ast.parse(m)
        assignments=[x for x in ast.walk(tree) if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='execution_max_linear' for t in x.targets)]
        self.assertEqual(len(assignments),1)
        expression=compile(ast.Expression(assignments[0].value),'mission cap','eval')
        from types import SimpleNamespace
        for mode in ['nso','joint','navigation','reference']:
            self.assertEqual(eval(expression,{'args':SimpleNamespace(mode=mode)}),.05 if mode in ('nso','joint') else .1)
        self.assertIn('maximum_linear_m_s=execution_max_linear,execution_clipping_profile=execution_profile',m)

if __name__=='__main__':unittest.main()
