"""Synthetic freeze inputs only; no real configuration, archive or World created."""
from contextlib import ExitStack
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from scripts import freeze_article_exposure_ablation_20260928 as freeze
from tests.test_article_ground_experiment_v2 import NoWorld


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))


class ExposureFreezeTests(NoWorld):
    def fixture(self,stack):
        root=Path(stack.enter_context(tempfile.TemporaryDirectory()))
        ground=deepcopy(freeze.read(freeze.ROOT/freeze.GROUND_PATH))
        ground['numerical_runtime']={'mock_runtime':True}
        sources={freeze.PREDICTOR_PATH:b'# mock predictor\n','nso/mock_old_source.py':b'# old source\n'}
        for name,data in sources.items():
            (root/name).parent.mkdir(parents=True,exist_ok=True);(root/name).write_bytes(data)
        pins={name:hashlib.sha256(data).hexdigest() for name,data in sources.items()}
        ground['source_sha256']={'nso/mock_old_source.py':pins['nso/mock_old_source.py']}
        ground['source_archive']='original.zip'
        with zipfile.ZipFile(root/'original.zip','w') as stream:
            stream.writestr('nso/mock_old_source.py',sources['nso/mock_old_source.py'])
        ground['source_archive_sha256']=freeze.sha(root/'original.zip')
        write(root/freeze.DECLARATION_PATH,{'mock_only':True})
        write(root/ground['asset_root']/'manifest.json',{'mock_only':True})
        ground['asset_manifest_sha256']=freeze.sha(root/ground['asset_root']/'manifest.json')
        for entry in ground['references'].values():
            write(root/entry['root']/'manifest.json',{'mock_only':True})
            entry['manifest_sha256']=freeze.sha(root/entry['root']/'manifest.json')
        write(root/freeze.GROUND_PATH,ground);pin=freeze.sha(root/freeze.GROUND_PATH)
        output=root/ground['output_relative_path']
        entries=[dict(run_id=name,status='controller_stop') for name in ground['slots']]
        entries[0]['status']='attempt_failed'  # failures must remain in the cohort
        write(output/'start_ledger.json',dict(schema='article.start_ledger.v1',phase='ablation',
            protocol_sha256=pin,slots=ground['slots'],entries=entries))
        write(output.parent/'execution_registry.json',dict(schema='article.execution_registry.v1',
            phase_caps={'development':12,'main':96,'ablation':24},entries=[dict(phase='ablation',
                ledger=str((output/'start_ledger.json').resolve()),run_id=name,protocol_sha256=pin)
                for name in ground['slots']]))
        for key,value in (('GROUND_SHA',pin),('DECLARATION_SHA',freeze.sha(root/freeze.DECLARATION_PATH)),
                          ('PREDICTOR_SHA',pins[freeze.PREDICTOR_PATH])):
            stack.enter_context(patch.object(freeze,key,value))
        stack.enter_context(patch.object(freeze,'source_hashes',return_value=pins))
        stack.enter_context(patch.object(freeze,'runtime_metadata',return_value=ground['numerical_runtime']))
        # Detailed runtime pairing is covered by the separate runner tests.
        stack.enter_context(patch.object(freeze,'validate_pairing',return_value={'mock_only':True}))
        return root,ground,output,pins

    def test_preserves_full_matrix_failures_and_only_adds_common_switch(self):
        with ExitStack() as stack:
            root,ground,output,pins=self.fixture(stack)
            result=freeze.build_protocol(root=root)
            self.assertEqual(len(result['slots']),12)
            self.assertEqual(result['phase'],'ablation')
            self.assertEqual(result['source_sha256'],pins)
            self.assertEqual({s['paired_baseline_run_id'] for s in result['slots'].values()},set(ground['slots']))
            fields=dict(result['controller']);self.assertTrue(fields.pop('same_center_exposure_dedup'))
            self.assertEqual(fields,ground['controller'])
            self.assertIn('attempt_failed',result['predecessor_terminal_cohort']['retained_statuses'].values())
            self.assertEqual(result['metric_preprocessing'],ground['metric_preprocessing'])
            self.assertFalse((root/result['source_archive']).exists())
            self.assertFalse((root/result['output_relative_path']).exists())

    def test_unfinished_unknown_and_extra_reservation_prevent_freeze(self):
        for fault in ('reserved','unknown','missing','extra_global'):
            with self.subTest(fault=fault),ExitStack() as stack:
                root,ground,output,pins=self.fixture(stack)
                if fault=='extra_global':
                    path=output.parent/'execution_registry.json';record=freeze.read(path)
                    record['entries'].append(dict(record['entries'][0],run_id='extra'));write(path,record)
                else:
                    path=output/'start_ledger.json';record=freeze.read(path)
                    if fault=='missing':record['entries'].pop()
                    else:record['entries'][0]['status']=fault
                    write(path,record)
                with self.assertRaises(ValueError):freeze.build_protocol(root=root)

    def test_old_source_archive_predictor_and_runtime_must_match(self):
        for fault in ('old_source','archive','predictor','runtime'):
            with self.subTest(fault=fault),ExitStack() as stack:
                root,ground,output,pins=self.fixture(stack)
                if fault=='old_source':(root/'nso/mock_old_source.py').write_text('changed')
                if fault=='archive':(root/'original.zip').write_bytes(b'changed')
                if fault=='predictor':(root/freeze.PREDICTOR_PATH).write_text('changed')
                if fault=='runtime':stack.enter_context(patch.object(freeze,'runtime_metadata',return_value={'changed':True}))
                with self.assertRaises(ValueError):freeze.build_protocol(root=root)


if __name__=='__main__':unittest.main()
