import unittest
import numpy as np
from scipy.sparse.csgraph import dijkstra
from nso.route_coverage_v2 import orientation_graph
from scripts.eval_counterfactual_views import path_states, states_to_actions, advance_states


class PaidRouteTests(unittest.TestCase):
    def test_directed_return_is_real_forward_and_turn_path(self):
        safe=np.zeros((7,8),bool);safe[2:5,2:6]=True;safe[3,4]=False
        graph,cells,ids=orientation_graph(safe)
        start=int(ids[3,2])*4+1;end=int(ids[3,5])*4+3
        out,prev=dijkstra(graph,directed=True,indices=start,return_predecessors=True)
        back,backprev=dijkstra(graph.T.tocsr(),directed=True,indices=start,return_predecessors=True)
        states=path_states(prev,start,end,cells)+path_states(backprev,start,end,cells,reverse=True)[1:]
        actions=states_to_actions(states)
        self.assertEqual(len(actions),int(out[end]+back[end]))
        self.assertEqual(states,advance_states(states[0],actions))
        self.assertEqual(states[0],states[-1])
        self.assertTrue(all(safe[r,c] for r,c,_ in states))

    def test_teleport_and_unpaid_rotation_rejected(self):
        with self.assertRaises(ValueError):states_to_actions([(1,1,1),(1,3,1)])
        with self.assertRaises(ValueError):states_to_actions([(1,1,1),(1,1,3)])
        with self.assertRaises(ValueError):advance_states((1,1,1),['stop'])


if __name__=='__main__':unittest.main()
