import ast,copy,sys,types,unittest
from pathlib import Path
from unittest.mock import patch
from visual_velocity_adapter import Gate,configuration
import mission
class OrdinaryReverseTests(unittest.TestCase):
    def test_bidirectional_bounds_modes_and_legacy(self):
        for mode in ['nso','joint','navigation']:
            g=Gate(ordinary_reverse=True,mission_mode=mode)
            for v,w,expected in [(-.2,-.5,(-.05,-.3)),(.2,.5,(.05,.3)),(.03,.1,(.03,.1))]:
                g.propose([v,0,0,0,0,w],10.,g.proposal_publisher)
                self.assertEqual((g.proposal['v'],g.proposal['w']),expected)
            self.assertEqual(configuration(dict(ordinary_reverse=True,mission_mode=mode))['maximum_linear_m_s'],.05)
        g=Gate(navigation_allow_reverse=True,mission_mode='navigation')
        g.propose([.2,0,0,0,0,0],10.,g.proposal_publisher);self.assertEqual(g.proposal['v'],.1)
    def test_phase_exit_preserves_only_authorized_ordinary_reverse(self):
        for ordinary in [False,True]:
            g=Gate(ordinary_reverse=ordinary,nso_correction_reverse=True,mission_mode='nso');g.session_id='s';g.armed=True
            self.assertTrue(g.set_correction_reverse(True,g.correction_supervisor,'s')[0])
            g.propose([-.04,0,0,0,0,0],10.,g.proposal_publisher)
            g.set_correction_reverse(False,g.correction_supervisor,'s')
            self.assertEqual(g.proposal['v'],-.04 if ordinary else 0.)
            self.assertEqual(g.proposal['receipt'],10.)
            g.fail('explicit_stop');self.assertEqual(g.decision(10.,1010.)[:2],(0.,0.))
    def test_validation_scope_and_cli(self):
        for value in [1,'true',None]:
            with self.assertRaises(ValueError):Gate(ordinary_reverse=value,mission_mode='nso')
        for mode in [None,'reference','visual']:
            with self.assertRaises(ValueError):configuration(dict(ordinary_reverse=True,mission_mode=mode))
        self.assertFalse(mission.argument_parser().parse_args([]).ros_allow_reverse)
        for mode in ['nso','joint','navigation']:
            self.assertTrue(mission.argument_parser().parse_args(['--mode',mode,'--ros-allow-reverse']).ros_allow_reverse)
    def test_real_mission_dynamic_scope_applies_and_restores(self):
        tree=ast.parse(Path(mission.__file__).read_text());main=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='main')
        block=next(x for x in main.body if isinstance(x,ast.Try) and x.finalbody)
        dynamic=next(x for x in block.body if isinstance(x,ast.If) and 'ros_allow_reverse' in {n.attr for n in ast.walk(x.test) if isinstance(n,ast.Attribute)})
        restore=next(x for x in block.finalbody if isinstance(x,ast.If) and 'navigation_restore' in {n.id for n in ast.walk(x.test) if isinstance(n,ast.Name)})
        code=compile(ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(dynamic),copy.deepcopy(restore)],type_ignores=[])),'mission_scope','exec')
        for mode in ['navigation','nso','joint']:
            values=dict(min_vel_x=0.,max_vel_x=.1,max_vel_trans=.12);updates=[]
            class Config:
                def get_configuration(self,**kw):return dict(values)
                def update_configuration(self,change):values.update(change);updates.append(dict(change));return dict(values)
            mod=types.ModuleType('dynamic_reconfigure.client');mod.Client=lambda *a,**k:Config()
            ns=dict(args=types.SimpleNamespace(mode=mode,navigation_allow_reverse=False,ros_allow_reverse=True),navigation_restore=None,result={},output=Path('/unused'),save=lambda *x:None)
            with patch.dict(sys.modules,{'dynamic_reconfigure':types.ModuleType('dynamic_reconfigure'),'dynamic_reconfigure.client':mod}):exec(code,ns)
            self.assertEqual(updates,[dict(min_vel_x=-.05,max_vel_x=.05,max_vel_trans=.05),dict(min_vel_x=0.,max_vel_x=.1,max_vel_trans=.12)])
            self.assertEqual(values,dict(min_vel_x=0.,max_vel_x=.1,max_vel_trans=.12))
if __name__=='__main__':unittest.main()
