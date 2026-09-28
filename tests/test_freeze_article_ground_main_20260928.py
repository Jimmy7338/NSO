"""Synthetic configuration checks; never freeze a real protocol or run a World."""
from collections import Counter
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from env.development_sensor_v41 import runtime_counts_v41
from scripts import freeze_article_ground_main_20260928 as freeze
from tests.test_article_ground_experiment_v2 import ground_protocol


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


class MainFreezeTests(unittest.TestCase):
    def setUp(self):self.before=runtime_counts_v41()
    def tearDown(self):self.assertEqual(self.before,runtime_counts_v41())

    def test_exact_preselected_factorial_and_lexical_complete_blocks(self):
        slots,blocks=freeze.declared_slots()
        self.assertEqual(len(slots),48);self.assertEqual(len(blocks),12)
        self.assertEqual([name for block in blocks for name in block['run_ids']],sorted(slots))
        self.assertEqual(Counter(s['budget'] for s in slots.values()),{120:24,160:24})
        self.assertEqual({s['noise_seed'] for s in slots.values()},{92811})
        self.assertEqual(Counter((s['scene_id'],s['budget'],s['method']) for s in slots.values()),
            {(scene,budget,method):1 for scene in freeze.LAYOUTS for budget in (120,160) for method in freeze.METHODS})
        for block in blocks:self.assertEqual({slots[name]['method'] for name in block['run_ids']},set(freeze.METHODS))
        self.assertEqual({block['scene_id'] for block in blocks[:6]},set(freeze.LAYOUTS))

    def synthetic(self,root):
        ground=ground_protocol(12)
        source={f'nso/fake{i:02d}.py':f'# synthetic {i}\n'.encode() for i in range(62)}
        pins={name:__import__('hashlib').sha256(data).hexdigest() for name,data in source.items()}
        ground['source_sha256']=pins
        archive=root/'sources.zip'
        with zipfile.ZipFile(archive,'w') as stream:
            for name,data in source.items():stream.writestr(name,data)
        ground['source_archive']='sources.zip';ground['source_archive_sha256']=freeze.sha(archive)
        ground['numerical_runtime']={'mock_only':True}
        ground['asset_root']='assets';write(root/'assets/manifest.json',{'mock_only':True})
        ground['asset_manifest_sha256']=freeze.sha(root/'assets/manifest.json')
        ground['references']={f'ART1_{family}_DEV':dict(root=f'refs/ART1_{family}_DEV',manifest_sha256='a'*64)
                              for family in ('AISLE','CELL','LOOP')}
        for scene in freeze.LAYOUTS:
            reference=dict(scene_id=scene,asset_manifest_sha256=ground['asset_manifest_sha256'],
                evaluation=freeze.EVALUATION,route_independent=True,reference_uses_semantic_weights=False,
                surface={'target_instances':list(range(4))})
            path=root/'refs'/scene;write(path/'reference.json',reference)
            write(path/'manifest.json',{'reference.json':dict(bytes=(path/'reference.json').stat().st_size,
                  sha256=freeze.sha(path/'reference.json'))})
        path=root/'ground.json';write(path,ground)
        return path,ground,pins

    def test_config_reuses_same_sources_and_declares_gate_without_passing_it(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path,ground,pins=self.synthetic(root)
            with patch.object(freeze,'source_hashes',return_value=pins), \
                 patch.object(freeze,'runtime_metadata',return_value=ground['numerical_runtime']):
                result=freeze.build_protocol(path,root=root)
            self.assertEqual(result['phase'],'main')
            self.assertEqual(result['source_sha256'],ground['source_sha256'])
            self.assertEqual(result['source_archive_sha256'],ground['source_archive_sha256'])
            self.assertEqual(result['controller'],ground['controller'])
            self.assertNotIn('paired_baseline_protocol',result)
            self.assertEqual(set(result['references']),set(freeze.LAYOUTS))
            self.assertTrue(result['scientific_release_gate']['required_before_any_world'])
            self.assertFalse(result['scientific_release_gate']['positive_endpoint_difference_required'])
            self.assertFalse(result['scientific_release_gate']['automatic_release'])
            self.assertTrue(result['matrix']['second_noise_explicitly_deferred'])
            self.assertFalse((root/result['output_relative_path']).exists())

    def test_changed_environment_reference_or_archive_rejected(self):
        for fault in ('runtime','reference','archive'):
            with self.subTest(fault=fault),tempfile.TemporaryDirectory() as directory:
                root=Path(directory);path,ground,pins=self.synthetic(root)
                runtime=ground['numerical_runtime'] if fault!='runtime' else {'wrong':True}
                if fault=='reference':(root/'refs'/freeze.LAYOUTS[0]/'reference.json').write_text('{}')
                if fault=='archive':(root/'sources.zip').write_bytes(b'changed')
                with patch.object(freeze,'source_hashes',return_value=pins), \
                     patch.object(freeze,'runtime_metadata',return_value=runtime),self.assertRaises((ValueError,KeyError)):
                    freeze.build_protocol(path,root=root)


if __name__=='__main__':unittest.main()
