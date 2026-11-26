"""Offline threshold counterfactuals only; not validation of physical safety."""
import ast,json,struct,unittest
from pathlib import Path
from visual_velocity_adapter import Gate,configuration,snapshot
ROOT=Path('/home/ouc/nso_robot/analyses/resume_20261112')
TRACES=[ROOT/'run/fault_ring/failure_can_ring.json',ROOT/'linear005_S40/run/fault_ring/failure_can_ring.json']
class FeedbackProposalTests(unittest.TestCase):
    def test_defaults_and_invalid_values(self):
        self.assertEqual(Gate().maximum_feedback_linear_m_s,.15)
        self.assertEqual(configuration({})['maximum_feedback_linear_m_s'],.15)
        for value in [True,False,None,'0.2',0,-.2,.2001,float('nan'),float('inf'),[],{}]:
            for fn in [lambda:Gate(maximum_feedback_linear_m_s=value),lambda:configuration({'maximum_feedback_linear_m_s':value})]:
                with self.assertRaises(ValueError):fn()
    def test_boundaries_and_angular_stop_unchanged(self):
        for limit,boundary in [(.15,150),(.20,200)]:
            for sign in [-1,1]:
                for value,expected in [(boundary,None),(boundary+1,'chassis_speed_feedback_limit')]:
                    g=Gate(maximum_feedback_linear_m_s=limit);g.ever_armed=True
                    g.hardware(0x221,struct.pack('>hh',sign*value,450)+bytes(4),1.,1.001)
                    self.assertEqual(g.fault,expected)
                    if expected:self.assertEqual(g.fault_evidence['limits']['maximum_abs_v_m_s'],limit)
            g=Gate(maximum_feedback_linear_m_s=limit);g.ever_armed=True
            g.hardware(0x221,struct.pack('>hh',0,-451)+bytes(4),1.,1.)
            self.assertEqual(g.fault,'chassis_speed_feedback_limit')
    def test_actual_trace_counterfactuals(self):
        for path in TRACES:
            data=json.loads(path.read_text());samples=data['samples']
            for limit in [.15,.20]:
                g=Gate(maximum_linear_m_s=.05,maximum_feedback_linear_m_s=limit);g.ever_armed=True
                for row in samples:
                    frame=bytes.fromhex(row['raw_frame_hex']);source=row['kernel_stamp_ns']/1e9
                    g.hardware(0x221,frame[8:16],source,source+.001)
                self.assertEqual(g.fault,'chassis_speed_feedback_limit' if limit==.15 else None,str(path))
                self.assertEqual(snapshot(g,source)['maximum_feedback_linear_m_s'],limit)
                g.propose([.12,0,0,0,0,.4],source,'/nso_visual/move_base')
                self.assertEqual(g.proposal['v'],.05);self.assertEqual(g.proposal['w'],.3)
    def test_cli_default_and_opt_in_mapping(self):
        tree=ast.parse((Path(__file__).resolve().parents[1]/'runtime/mission.py').read_text())
        call=next(n for n in ast.walk(tree) if isinstance(n,ast.Call) and n.args and isinstance(n.args[0],ast.Constant) and n.args[0].value=='--feedback-speed-profile')
        kw={x.arg:ast.literal_eval(x.value) for x in call.keywords if x.arg in ('default','choices')}
        self.assertEqual(kw,dict(default='original_015_v1',choices=['original_015_v1','measured_020_v1']))
        assign=next(n for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='maximum_feedback_linear' for t in n.targets))
        from types import SimpleNamespace
        for profile,value in [('original_015_v1',.15),('measured_020_v1',.20)]:
            self.assertEqual(eval(compile(ast.Expression(assign.value),'profile','eval'),{'args':SimpleNamespace(feedback_speed_profile=profile)}),value)
if __name__=='__main__':unittest.main()
