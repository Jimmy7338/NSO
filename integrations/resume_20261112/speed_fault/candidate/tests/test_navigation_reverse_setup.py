"""Setup-only reverse opt-in; no ROS, hardware or motion."""
import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'runtime'))
from visual_velocity_adapter import Gate, configuration
import mission

class NavigationReverseTests(unittest.TestCase):
    def test_default_reverse_stays_zero(self):
        gate=Gate();gate.propose([-.05,0,0,0,0,.2],1.,'/nso_visual/move_base')
        self.assertEqual(gate.proposal['v'],0.);self.assertEqual(gate.proposal['w'],.2)

    def test_setup_reverse_bounded_and_existing_maxima_unchanged(self):
        gate=Gate(navigation_allow_reverse=True,mission_mode='navigation')
        gate.propose([-.2,0,0,0,0,-.4],1.,'/nso_visual/move_base')
        self.assertEqual(gate.proposal['v'],-.05);self.assertEqual(gate.proposal['w'],-.3)
        gate.propose([.2,0,0,0,0,.4],2.,'/nso_visual/move_base')
        self.assertEqual(gate.proposal['v'],.1);self.assertEqual(gate.proposal['w'],.3)

    def test_nonplanar_or_nonfinite_proposals_still_rejected(self):
        for proposal in [[-.04,.01,0,0,0,0],[float('nan'),0,0,0,0,0]]:
            gate=Gate(navigation_allow_reverse=True,mission_mode='navigation')
            gate.propose(proposal,1.,'/nso_visual/move_base');self.assertIsNotNone(gate.fault)
        for mode in ['visual','nso','joint','reference',None]:
            with self.assertRaises(ValueError):Gate(navigation_allow_reverse=True,mission_mode=mode)
            with self.assertRaises(ValueError):configuration(dict(navigation_allow_reverse=True,mission_mode=mode))

    def test_cli_rejects_flag_outside_navigation_before_initialization(self):
        for mode in ['visual','nso','joint','reference']:
            with self.assertRaisesRegex(ValueError,'restricted to navigation setup'):
                mission.main(['--mode',mode,'--navigation-allow-reverse'])


class MissionReverseLifecycleTests(unittest.TestCase):
    def test_actual_mission_apply_arm_goal_stop_restore_cleanup_control_flow(self):
        """Execute actual mission lifecycle AST with sensor waits mocked out.

        Preserve the source try/finally and reverse/arm/navigation/restore/child
        cleanup statements, so moving apply into finally fails this test.
        """
        import ast,copy,types,signal
        from unittest.mock import patch
        tree=ast.parse(Path(mission.__file__).read_text())
        main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
        block=copy.deepcopy(next(n for n in main.body if isinstance(n,ast.Try) and n.finalbody))
        def names(node):return {n.id for n in ast.walk(node) if isinstance(n,ast.Name)}
        def attrs(node):return {n.attr for n in ast.walk(node) if isinstance(n,ast.Attribute)}
        def selected(node,final=False):
            if isinstance(node,ast.If) and 'navigation_allow_reverse' in attrs(node):return True
            if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='armed' for t in node.targets):return True
            if isinstance(node,ast.If) and 'run_standard_navigation_goal' in names(node):return True
            if final and isinstance(node,ast.Expr) and 'publish' in attrs(node):return True
            if final and isinstance(node,ast.If) and 'navigation_restore' in names(node):return True
            if final and isinstance(node,ast.For) and 'children' in names(node):return True
            if final and isinstance(node,ast.FunctionDef) and node.name=='close_actuator_children':return True
            if final and isinstance(node,ast.Assign) and 'stop_feedback_cleanup_sequence' in names(node):return True
            return False
        block.body=[n for n in block.body if selected(n)]
        block.finalbody=[n for n in block.finalbody if selected(n,True)]
        code=compile(ast.fix_missing_locations(ast.Module(body=[block],type_ignores=[])),mission.__file__,'exec')
        for failure in (None,'apply','restore'):
            with self.subTest(failure=failure):
                events=[]
                class Config:
                    def get_configuration(self,**kwargs):return {'min_vel_x':0.}
                    def update_configuration(self,change):
                        kind='apply' if change['min_vel_x']<0 else 'restore';events.append(kind)
                        if failure==kind:raise RuntimeError(kind+' failed')
                        return dict(change)
                config_module=types.ModuleType('dynamic_reconfigure.client');config_module.Client=lambda *a,**kw:Config()
                nav_module=types.ModuleType('standard_navigation_goal');nav_module.run_standard_navigation_goal=lambda ctx:events.append('goal')
                class Child:
                    returncode=0
                    def poll(self):return None
                    def send_signal(self,value):events.append('cleanup')
                    def wait(self,**kwargs):return 0
                ns=dict(args=types.SimpleNamespace(mode='navigation',navigation_allow_reverse=True),
                    navigation_restore=None,result={'cleanup':[]},output=Path('/unused'),
                    save=lambda *a:None,rospy=types.SimpleNamespace(ServiceProxy=lambda *a:lambda:events.append('arm')),
                    Trigger=object,Bool=lambda **kw:kw,stop=types.SimpleNamespace(publish=lambda msg:events.append('stop')),
                    navigation_goal=[0,0,0],client=None,capture_odom_pose=None,can_snapshot=None,journal=None,
                    execution_interruption=lambda result,msg:events.append('error'),children=[('native',Child())],
                    stop_feedback_sample=lambda now:dict(confirmed=True,reason=None,snapshot={}),
                    stop_feedback_cleanup_sequence=__import__('stop_feedback_wait').stop_feedback_cleanup_sequence,
                    signal=signal,subprocess=types.SimpleNamespace(TimeoutExpired=TimeoutError))
                with patch.dict(sys.modules,{'dynamic_reconfigure':types.ModuleType('dynamic_reconfigure'),
                    'dynamic_reconfigure.client':config_module,'standard_navigation_goal':nav_module}):exec(code,ns)
                if failure=='apply':
                    self.assertNotIn('arm',events);self.assertNotIn('goal',events)
                else:self.assertEqual(events[:3],['apply','arm','goal'])
                self.assertLess(events.index('stop'),events.index('restore'))
                self.assertLess(events.index('restore'),events.index('cleanup'))
                self.assertEqual(events.count('apply'),1)
                if failure=='restore':self.assertIn('navigation_restore_error',ns['result'])

if __name__=='__main__':unittest.main()
