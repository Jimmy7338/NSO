"""Lossless bounded encoding for V43 step receipts; no planning changes."""
import gzip
from pathlib import PurePosixPath
import re

from nso.episode_driver_v43 import (
    BoundedRunWriterV43, OutputLimitExceededV43, canonical_bytes,
)


class CompressedStepWriterV44(BoundedRunWriterV43):
    """Compress only full step JSON, keeping metadata and terminal JSON plain.

    Stored bytes count against the unchanged episode allowance. The original
    per-file allowance also bounds each uncompressed JSON document, so gzip
    cannot admit an unbounded record. Compression uses no timestamps or file
    names. The inherited manifest binds the actual compressed bytes.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.step_encoding = []

    def json(self, name, value, *, terminal=False):
        name = str(PurePosixPath(name))
        if not name.startswith('steps/'):
            return super().json(name, value, terminal=terminal)
        if not re.fullmatch(r'steps/[0-9]{3}\.json', name):
            raise ValueError('step JSON must use steps/NNN.json')
        if terminal:
            raise ValueError('step receipts cannot use terminal reserve')
        plain = canonical_bytes(value)
        if len(plain) > self.maximum_file_bytes:
            raise OutputLimitExceededV43('uncompressed step exceeds per-file allowance')
        compressed = gzip.compress(plain, compresslevel=6, mtime=0)
        stored_name = name + '.gz'
        if (self.root/name).exists():
            raise FileExistsError('uncompressed alias already exists')
        receipt = self._write(stored_name, compressed)
        self.step_encoding.append(dict(
            artifact=stored_name, uncompressed_bytes=len(plain),
            stored_bytes=len(compressed), encoding='gzip-json-v1',
            compression_level=6, mtime=0,
        ))
        return receipt
