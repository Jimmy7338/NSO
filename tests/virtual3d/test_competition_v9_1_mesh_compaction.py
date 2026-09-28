"""Evidence gates for destructive derived-mesh storage operations."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location(
    'competition_v9_1_mesh_compaction', Path(__file__).resolve().parents[2] / 'scripts/compact_competition_v9_1_meshes.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class CompactionEvidenceTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.run = Path(self.temp.name) / 'run'
        self.run.mkdir()
        self.audit = Path(self.temp.name) / 'audit'
        self.audit.mkdir()
        self.mesh = 'Q0/shelf_west/candidate_000/final_mesh.npz'
        self.raw = 'Q0/shelf_west/prefix/frames/0000.npz'
        for name, data in ((self.mesh, b'mesh'), (self.raw, b'raw')):
            path = self.run / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        self.manifest = {name: MODULE.sha(self.run / name) for name in (self.mesh, self.raw)}
        self.selected = [self.mesh]
        self.bindings = {'artifact_hashes.json': 'a' * 64}
        tool = self.audit / 'compaction_tool.py'
        tool.write_bytes(b'archived tool')
        item = {'file_sha256': self.manifest[self.mesh], 'file_bytes': 4,
                'arrays': {key: {'dtype': '<f8', 'shape': [0, 3], 'nbytes': 0, 'sha256': 'b' * 64}
                           for key in ('vertices', 'triangles')},
                'status': 'passed_raw_to_original_npz_bytes'}
        self.receipt = {'status': 'passed_all_raw_to_original_npz_bytes', 'bindings': self.bindings,
                        'meshes': {self.mesh: item}, 'tool_sha256': MODULE.sha(tool),
                        'world_constructed': False, 'gt_or_outcomes_read': False,
                        'input_hashes': {self.raw: self.manifest[self.raw]}}
        MODULE.write(self.audit / 'raw_reconstruction.json', self.receipt)
        self.compact = {'schema_version': 'completed_branch_mesh_compaction/1', 'status': 'complete',
                        'bindings': self.bindings, 'raw_observations_removed': 0,
                        'prefix_and_reference_retained': True, 'meshes': self.receipt['meshes'],
                        'deleted': self.selected, 'tool_sha256': MODULE.sha(tool),
                        'raw_reconstruction_sha256': MODULE.sha(self.audit / 'raw_reconstruction.json')}

    def witness(self):
        MODULE.validate_witness(self.compact, self.audit, self.bindings, self.manifest, self.selected)

    def test_only_witnessed_derived_mesh_can_be_missing(self):
        (self.run / self.mesh).unlink()
        with self.assertRaises(ValueError):
            MODULE.validate_assets(self.run, self.manifest, self.selected)
        self.witness()
        self.assertEqual(MODULE.validate_assets(self.run, self.manifest, self.selected, self.compact), self.selected)

    def test_missing_raw_never_receives_mesh_exception(self):
        (self.run / self.raw).unlink()
        with self.assertRaises(ValueError):
            MODULE.validate_assets(self.run, self.manifest, self.selected, self.compact)

    def test_changed_retained_raw_is_rejected(self):
        (self.run / self.raw).write_bytes(b'wrong raw')
        with self.assertRaises(ValueError):
            MODULE.validate_assets(self.run, self.manifest, self.selected, self.compact)

    def test_partial_compaction_is_not_a_complete_witness(self):
        self.compact['status'] = 'planned'
        with self.assertRaises(ValueError):
            self.witness()

    def test_reconstruction_receipt_corruption_is_rejected(self):
        (self.audit / 'raw_reconstruction.json').write_text('{}')
        with self.assertRaises(ValueError):
            self.witness()

    def test_archived_tool_corruption_is_rejected(self):
        (self.audit / 'compaction_tool.py').write_text('changed')
        with self.assertRaises(ValueError):
            self.witness()

    def test_unbound_raw_receipt_is_rejected_even_after_rehash(self):
        self.receipt['input_hashes'][self.raw] = 'c' * 64
        MODULE.write(self.audit / 'raw_reconstruction.json', self.receipt)
        self.compact['raw_reconstruction_sha256'] = MODULE.sha(self.audit / 'raw_reconstruction.json')
        with self.assertRaises(ValueError):
            self.witness()

    def test_unlisted_mesh_and_path_escape_are_rejected(self):
        bad = dict(self.manifest, **{'Q0/shelf_west/unlisted/final_mesh.npz': 'd' * 64})
        with self.assertRaises(ValueError):
            MODULE.targets(bad, 'competition')
        with self.assertRaises(ValueError):
            MODULE.relative(self.run, '../outside')

    def test_complete_variable_pool_whitelist(self):
        manifest = {name: 'a' * 64 for name in MODULE.REMOVABLE}
        self.assertEqual(len(MODULE.targets(manifest, 'competition')), 88)
        missing = dict(manifest)
        missing.pop(next(iter(missing)))
        with self.assertRaises(ValueError):
            MODULE.targets(missing, 'competition')
        extra = dict(manifest)
        extra['Q3/shelf_west/candidate_004/final_mesh.npz'] = 'b' * 64
        with self.assertRaises(ValueError):
            MODULE.targets(extra, 'competition')

    def test_symlink_artifact_is_rejected(self):
        (self.run / self.mesh).unlink()
        (self.run / self.mesh).symlink_to(self.run / self.raw)
        with self.assertRaises(ValueError):
            MODULE.validate_assets(self.run, self.manifest, self.selected, self.compact)


if __name__ == '__main__':
    unittest.main()
