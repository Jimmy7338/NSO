import ast, inspect, sys, textwrap, unittest
from pathlib import Path
from types import SimpleNamespace as NS
SOURCE=Path('/home/ouc/nso_robot/analyses/reproduction_20261104/simulation_publication_stage/experiments/unknown_exploration_20261104/source')
sys.path.insert(0,str(SOURCE))
import numpy as np
from scipy.ndimage import binary_dilation
from nso.frontier_policy import FrontierPolicy as Original, CROSS
from strict_policy import FrontierPolicy as Strict
from env.grid_exploration import GridObservation, readonly_copy

class StrictTests(unittest.TestCase):
    def config(self):return NS(resolution_m=.2,robot_radius_m=0.,sensor_range_m=8.,sensor_fov_deg=90.)
    def obs(self,belief,position):return GridObservation(readonly_copy(belief),readonly_copy(belief!=-1),position,0,0,False)
    def test_select_ast_only_changes_initial_candidate_set(self):
        def tree(cls):return ast.parse(textwrap.dedent(inspect.getsource(cls._select)))
        a,b=tree(Original),tree(Strict)
        assignments=[]
        for t in (a,b):
            x=next(n for n in ast.walk(t) if isinstance(n,ast.Assign) and any(isinstance(z,ast.Name)and z.id=='candidates'for z in n.targets))
            assignments.append(ast.dump(x.value));x.value=ast.Constant(None)
        self.assertNotEqual(*assignments)
        self.assertEqual(ast.dump(a),ast.dump(b))
        self.assertIs(Strict.act,Original.act)
    def test_nonfrontier_current_pose_not_forced(self):
        belief=np.full((11,11),-1);belief[3:8,3:8]=0;obs=self.obs(belief,(5,5))
        old=Original(self.config());new=Strict(self.config())
        old.act(obs);new.act(obs)
        self.assertEqual(old.goal,obs.position)
        self.assertNotEqual(new.goal,obs.position)
        frontier=(belief==0)&binary_dilation(belief==-1,structure=CROSS)
        self.assertTrue(frontier[new.goal]);self.assertGreater(new._distance,0)
    def test_true_frontier_current_pose_remains_eligible(self):
        belief=np.full((11,11),-1);belief[3:8,3:8]=0;obs=self.obs(belief,(3,5))
        a,b=Original(self.config()),Strict(self.config())
        self.assertEqual(a.act(obs),b.act(obs));self.assertEqual(b.goal,obs.position)
    def test_empty_frontier_preserves_four_scan_then_stop(self):
        obs=self.obs(np.zeros((11,11),int),(5,5));p=Strict(self.config())
        self.assertEqual([p.act(obs).action for _ in range(5)],['right']*4+['stop'])
    def test_unreachable_free_frontier_is_not_accepted(self):
        belief=np.ones((11,11),int);belief[5,5]=0;belief[1:3,1:3]=0;belief[0,1:3]=-1
        p=Strict(self.config());obs=self.obs(belief,(5,5));p.act(obs)
        self.assertIsNone(p.goal)

if __name__=='__main__':unittest.main()
