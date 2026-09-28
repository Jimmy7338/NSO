"""Encoding/cap regressions; no sensor acquisition or study trajectory."""
import gzip
import hashlib
from pathlib import Path
import tempfile
import unittest

from nso.episode_driver_v43 import OutputLimitExceededV43, canonical_bytes
from nso.evidence_writer_v44 import CompressedStepWriterV44


class EvidenceWriterV44Tests(unittest.TestCase):
    def test_lossless_deterministic_encoding_and_compressed_hash(self):
        payload = {'decision': {'action': 'left', 'unicode': '完整保留'},
                   'repeated': [dict(area=.123, unknown=False)]*100}
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            a, b = CompressedStepWriterV44(first), CompressedStepWriterV44(second)
            receipt = a.json('steps/003.json', payload)
            b.json('steps/003.json', payload)
            raw = (Path(first)/'steps/003.json.gz').read_bytes()
            self.assertEqual(raw, (Path(second)/'steps/003.json.gz').read_bytes())
            self.assertEqual(gzip.decompress(raw), canonical_bytes(payload))
            self.assertEqual(receipt['sha256'], hashlib.sha256(raw).hexdigest())
            self.assertEqual(a.bytes_written, len(raw))
            self.assertFalse((Path(first)/'steps/003.json').exists())

    def test_uncompressed_size_limit_is_not_bypassed_by_compressibility(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = CompressedStepWriterV44(directory, maximum_bytes=4096,
                                           maximum_file_bytes=1024, terminal_reserve_bytes=256)
            with self.assertRaises(OutputLimitExceededV43):
                writer.json('steps/000.json', {'highly_compressible': 'x'*1024})
            self.assertEqual(writer.files, {})
            self.assertEqual(writer.step_encoding, [])

    def test_terminal_reserve_plain_json_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = CompressedStepWriterV44(directory, maximum_bytes=1024,
                                           maximum_file_bytes=1024, terminal_reserve_bytes=256)
            writer.json('steps/000.json', {'value': 1})
            with self.assertRaises(FileExistsError):
                writer.json('steps/000.json', {'value': 2})
            writer._write('fill.bin', b'x'*(768-writer.bytes_written))
            with self.assertRaises(OutputLimitExceededV43):
                writer.json('steps/001.json', {'value': 3})
            writer.json('result.json', {'status': 'artifact_limit'}, terminal=True)
            self.assertEqual((Path(directory)/'result.json').read_bytes(),
                             canonical_bytes({'status': 'artifact_limit'}))
            self.assertLessEqual(writer.bytes_written, 1024)

    def test_conflicting_plain_alias_and_step_reserve_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = CompressedStepWriterV44(directory)
            writer._write('steps/000.json', b'{}')
            with self.assertRaises(FileExistsError):
                writer.json('steps/000.json', {})
            with self.assertRaises(ValueError):
                writer.json('steps/001.json', {}, terminal=True)
            with self.assertRaises(ValueError):
                writer.json('steps/001.json.gz', {})
            with self.assertRaises(ValueError):
                writer.json('../escape.json', {})


if __name__ == '__main__':
    unittest.main()
