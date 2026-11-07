"""Synthetic unit test only: no robot, recorded sensors or desk templates.

The two in-memory templates and two-pose turn graph exercise the unmodified
default V35 planner through its observation/action/return lifecycle. Passing
this test provides no physical deployment or method-efficacy evidence.
"""
from pathlib import Path
import sys
import unittest

import numpy as np

VENDOR_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VENDOR_ROOT))

from nso.cpu_four_modules_v35 import CPUFourModuleControllerV35, ObservationV35
from nso.observation_belief_v35 import PublicTemplatesV35
from nso.online_planner_v35 import ForecastBeliefPlannerV35
from nso.pixel_information_v34 import SavedPotentialModelV34


def synthetic_unit_fixture():
    poses = ((0, 0, 0), (0, 0, 1))
    # A right turn followed by a left turn restores the complete anchor pose.
    # Bit 0 is surface; bits 1 and 2 are the two declared floor cells.
    table = dict(
        poses=poses, edges=((('right', 1),), (('left', 0),)), anchor=0,
        observed_masks_hex=('0x2', '0x5'), target_bits=1,
        targets=[dict(owner=0, area=1.0)],
        exact_reference_vertical_areas={'0': 1.0},
        floor_cells=((0, 0), (0, 1)), prefix_mask_hex='0x2',
        prefix_nodes=(0,),
    )
    models = tuple(SavedPotentialModelV34(table, ('unit', 'unit'))
                   for _ in range(2))
    depth = np.full((2, 2, 4, 4), 2.0, dtype=np.float32)
    ranges = np.full((2, 2, 3), 1.5, dtype=np.float32)
    depth[1] = 3.0
    ranges[1] = 2.5
    templates = PublicTemplatesV35(
        poses, depth, ranges,
        provenance={'scope': 'synthetic unit test only; not physical evidence'},
    )
    return models, templates


def synthetic_observation(step, pose, action=None):
    return ObservationV35(
        frame_id=f'synthetic-unit-only-{step}', step=step, pose=pose,
        action=action, depth=np.full((4, 4), 2.0, dtype=np.float32),
        rgb=np.zeros((4, 4, 3), dtype=np.uint8),
        ranges=np.full(3, 1.5, dtype=np.float32),
    )


class VendorV35DefaultControllerTest(unittest.TestCase):
    def test_default_planner_observation_action_and_qualified_return(self):
        models, templates = synthetic_unit_fixture()
        # No planner stub or informative-node override: exercise default imports
        # and information-node derivation, then actual finite-horizon planning.
        core = CPUFourModuleControllerV35(models, templates, (), total_budget=3)
        self.assertIsInstance(core.planner, ForecastBeliefPlannerV35)
        self.assertEqual(
            Path(sys.modules[ForecastBeliefPlannerV35.__module__].__file__).resolve(),
            VENDOR_ROOT / 'nso' / 'online_planner_v35.py',
        )
        core.accept(synthetic_observation(0, templates.poses[0]))
        self.assertFalse(models[0].terminal(core.masks[0])['feasible'])
        self.assertEqual(core.next_action(), 'right')

        # A report of arrival at the wrong graph pose must not advance state.
        with self.assertRaisesRegex(ValueError, 'odometry disagrees'):
            core.accept(synthetic_observation(1, templates.poses[0], 'right'))
        self.assertEqual(core.state['step'], 0)
        core.accept(synthetic_observation(1, templates.poses[1], 'right'))
        self.assertEqual(core.next_action(), 'left')
        core.accept(synthetic_observation(2, templates.poses[0], 'left'))

        self.assertEqual(core.state['pose'], list(templates.poses[0]))
        self.assertTrue(all(model.terminal(mask)['feasible']
                            for model, mask in zip(models, core.masks)))
        self.assertIsNone(core.next_action())
        self.assertIsNone(core.pending)
        self.assertEqual(core.summary()['actual_paid_actions'], 2)
        self.assertEqual(core.summary()['modules_called'],
                         ['IGCR', 'OV-SDF', 'RPN-UQ', 'STGHP'])
        self.assertFalse(core.summary()['full_method_efficacy_proven'])


if __name__ == '__main__':
    unittest.main()
