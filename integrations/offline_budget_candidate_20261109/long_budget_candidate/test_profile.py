import pathlib,sys,unittest,math,importlib.util,json,ast,hashlib
from unittest.mock import patch
ROOT=pathlib.Path(__file__).parent;RUNTIME=ROOT/'candidate/runtime';PROD=pathlib.Path('/home/ouc/nso_robot/workspaces/nso_hybrid_20261029/runtime')
sys.path[:0]=[str(RUNTIME),str(PROD)]
import duration_profile as profile
import visual_velocity_adapter as adapter
import mission
class Profiles(unittest.TestCase):
 def test_default600(self):
  self.assertEqual(adapter.Gate().duration,600.)
  self.assertEqual(mission.argument_parser().parse_args([]).duration_profile,profile.STANDARD)
  for d in (600.00001,1200):
   with self.subTest(d=d),self.assertRaises(ValueError):adapter.Gate(d)
   with self.subTest(d=d),self.assertRaises(ValueError):adapter.configuration({'duration_s':d})
 def test_explicit1200_both_modes_and_configuration(self):
  for mode in ('nso','joint'):
   for d in (1.,600.,600.00001,1200.):
    with self.subTest(mode=mode,d=d):
     gate=adapter.Gate(d,duration_profile=profile.LONG,mission_mode=mode)
     cfg=adapter.configuration({'duration_s':d,'duration_profile':profile.LONG,'mission_mode':mode})
     self.assertEqual(gate.duration,d);self.assertEqual(cfg['duration_profile'],profile.LONG)
     self.assertEqual(gate.duration_profile['maximum_duration_s'],1200.)
 def test_rejections_all_layers(self):
  for d in (0.,1200.00001,math.nan,math.inf,-math.inf):
   for fn in (lambda:profile.validate_duration(d,profile.LONG,'nso'),lambda:adapter.Gate(d,duration_profile=profile.LONG,mission_mode='nso'),lambda:adapter.configuration({'duration_s':d,'duration_profile':profile.LONG,'mission_mode':'nso'})):
    with self.subTest(d=d),self.assertRaises(ValueError):fn()
   with self.subTest(d=d),self.assertRaises(ValueError):mission.main(['--mode','nso','--duration-profile',profile.LONG,'--duration='+str(d)])
 def test_wrong_mode_even_with_short_duration(self):
  for mode in ('navigation','visual','reference',None):
   with self.subTest(mode=mode),self.assertRaises(ValueError):adapter.Gate(600,duration_profile=profile.LONG,mission_mode=mode)
   if mode:
    with self.subTest(mode=mode),self.assertRaises(ValueError):mission.main(['--mode',mode,'--duration-profile',profile.LONG])
 def test_unknown_profile(self):
  for p in ('bogus',None):
   with self.assertRaises(ValueError):adapter.Gate(600,duration_profile=p,mission_mode='nso')
 def test_real_decision_deadline_latches_zero(self):
  for d,p,mode in ((600.,profile.STANDARD,None),(1200.,profile.LONG,'joint')):
   g=adapter.Gate(d,duration_profile=p,mission_mode=mode);g.armed=g.ever_armed=True;g.started=10.;g.last_tick=10.+d-.02
   # Call unchanged production decision method at the precise deadline; no ROS or socket.
   self.assertEqual(g.decision(10.+d,1010.+d),(0.,0.,'duration_complete'))
   self.assertFalse(g.armed);self.assertEqual(g.decision(10.+d+.02,1010.+d+.02),(0.,0.,'duration_complete'))
 def test_freshness_still_bounded(self):
  for kw in ({'proposal_ttl_s':.251},{'observation_ttl_s':.501}):
   with self.assertRaises(ValueError):adapter.Gate(1200,duration_profile=profile.LONG,mission_mode='nso',**kw)
 def test_unchanged_decision_and_execution_deadlines(self):
  before=ast.parse((ROOT/'original/runtime/visual_velocity_adapter.py').read_text());after=ast.parse((RUNTIME/'visual_velocity_adapter.py').read_text())
  def method(tree,name):return next(n for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==name)
  for name in ('decision','arm','propose','observe','sources_reason','hardware_reason'):
   self.assertEqual(ast.dump(method(before,name)),ast.dump(method(after,name)))
  text=(RUNTIME/'mission.py').read_text();self.assertIn('stop_gate=mod.Gate(args.duration, duration_profile=args.duration_profile, mission_mode=args.mode)',text)
  self.assertIn('duration_profile=args.duration_profile,mission_mode=args.mode,output_directory',text)
  self.assertEqual(text.count("if time.monotonic()-run_start>=args.duration"),3)
  self.assertIn("duration_profile=duration_profile,\n        successful_arrivals",text)
if __name__=='__main__':
 suite=unittest.defaultTestLoader.loadTestsFromTestCase(Profiles);r=unittest.TextTestRunner(verbosity=2).run(suite)
 (ROOT/'validation.json').write_text(json.dumps({'tests_run':r.testsRun,'failures':len(r.failures),'errors':len(r.errors),'passed':r.wasSuccessful(),'no_motion':True,'production_hashes_unchanged':all(hashlib.sha256((PROD/f.name).read_bytes()).hexdigest()==hashlib.sha256(f.read_bytes()).hexdigest() for f in (ROOT/'original/runtime').glob('*.py'))},indent=2)+'\n');sys.exit(not r.wasSuccessful())
