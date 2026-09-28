import unittest
from utils.virtual_scene_factory import scene_configuration, scene_key


class SceneIdentityTests(unittest.TestCase):
    def test_archived_keys_preserved(self):
        config = {'environment': {'width_m': 16, 'height_m': 12}}
        entry = {'layout': 'rooms', 'seed': 202, 'environment': {'occluded_objects': True}}
        self.assertEqual(scene_key(config, entry, 1), 'rooms_202_aligned_d0.01_p0.0')
        self.assertEqual(scene_key(config, entry, 2), 'rooms_202_aligned_d0.01_p0.0_opaque1')
        self.assertEqual(scene_key(config, entry, 3), scene_key(config, entry, 2))

    def test_all_facility_controls_separate_identity(self):
        config = {'environment': {'appearance': 'marked', 'semantic_source': 'rgb_marker'}}
        entry = {'layout': 'inspection', 'seed': 811}
        original = scene_key(config, entry)
        keys = [original]
        for control, value in [('hidden_parts', 'vertical_baffles'), ('semantic_relation', 'reversed'),
                               ('observation_condition', 'open'), ('appearance', 'gray'),
                               ('max_steps', 42), ('bays_per_side', 3)]:
            keys.append(scene_key(config, entry | {'environment': {control: value}}))
        self.assertEqual(len(set(keys)), len(keys))
        self.assertEqual(original, scene_key(config, entry | {'environment': {'appearance': 'marked'}}))

    def test_replay_uses_derived_dimensions_and_entry_budget(self):
        config = {'environment': {'bays_per_side': 1, 'max_steps': 40}}
        family, c = scene_configuration(config, {'layout': 'inspection', 'seed': 811,
                                                'environment': {'max_steps': 24}})
        self.assertEqual(family, 'inspection_v4')
        self.assertAlmostEqual(c.width_m, 9.2)
        self.assertAlmostEqual(c.height_m, 13.2)
        self.assertEqual(c.max_steps, 24)
        self.assertEqual(c.resolution_m, .2)

    def test_reject_incompatible_family_and_old_identity(self):
        with self.assertRaises(ValueError):
            scene_key({'environment': {}}, {'layout': 'inspection', 'seed': 811}, 2)
        with self.assertRaises(ValueError):
            scene_configuration({'environment': {}, 'world_family': 'inspection_v4'},
                                {'layout': 'rooms', 'seed': 811})


if __name__ == '__main__':
    unittest.main()
