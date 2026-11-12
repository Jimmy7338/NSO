import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'runtime'))
from rtabmap_loop_session import prepare_session


class SessionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / 'owned'
        self.addCleanup(self.tmp.cleanup)

    def test_mapping_fresh_and_no_silent_reuse(self):
        output = prepare_session(self.root, 'mapping', 'new', boot_id='test')
        self.assertFalse((output/'rtabmap.db').exists())
        with self.assertRaises(FileExistsError):
            prepare_session(self.root, 'mapping', 'new', boot_id='test')

    def test_mapping_rejects_database(self):
        with self.assertRaises(ValueError):
            prepare_session(self.root, 'mapping', 'new', source='/tmp/old.db', boot_id='test')

    def test_localization_requires_owned_database(self):
        for source in [None, '/tmp/not_our_database.db']:
            with self.assertRaises(ValueError):
                prepare_session(self.root, 'localization', 'new', source=source, boot_id='test')

    def test_localization_copy_is_independent(self):
        original = self.root/'boot'/'mapped'/'rtabmap.db'
        original.parent.mkdir(parents=True)
        with sqlite3.connect(str(original)) as db:
            db.execute('CREATE TABLE record (value INTEGER)')
            db.execute('INSERT INTO record VALUES (1)')
        output = prepare_session(self.root, 'localization', 'loc', original, boot_id='test')
        with sqlite3.connect(str(output/'rtabmap.db')) as db:
            db.execute('INSERT INTO record VALUES (2)')
        with sqlite3.connect(str(original)) as db:
            self.assertEqual(db.execute('SELECT COUNT(*) FROM record').fetchone()[0], 1)

    def test_path_traversal_rejected(self):
        with self.assertRaises(ValueError):
            prepare_session(self.root, 'mapping', '../elsewhere', boot_id='test')


if __name__ == '__main__':
    unittest.main()
