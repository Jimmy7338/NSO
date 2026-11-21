#!/usr/bin/env python3
"""Build the preserved offline decoder snapshot without ROS or a robot workspace."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess

DIRECTORY = Path(__file__).resolve().parent
CPP_SHA256 = '8c7527ec18e418590e1653d4385ebec900d6e18ee25756916d2e1370cc84090c'
VENDOR_SOURCE_SHA256 = '6a60a2aef316b285aacf1908d8f58004d11157cdde9c1a9056e890641938ae94'
FUNCTION_SHA256 = [
    '1f34e71b497861ad9ac0e44106b2fd83952afe3a81e03f79d02d5c96f25a208a',
    '21f3db8f54996127947e1b456e62a07bbcf0202d19b812bc02237c5a1975b945',
    '634b21ad4922d69e21f4332560b52ee636a6665713074c9e5d7d2cdfbf39d001',
]


def build(output, source=None, compiler='c++'):
    source = (DIRECTORY/'vendor_oracle.cpp' if source is None else Path(source)).resolve()
    actual = hashlib.sha256(source.read_bytes()).hexdigest()
    if actual != CPP_SHA256:
        raise ValueError('Preserved C++ snapshot checksum differs; review provenance before building')
    executable = shutil.which(compiler)
    if executable is None:
        raise FileNotFoundError('A local C++11 compiler is required')
    output = Path(output).resolve()
    # Require a separate new directory; never replace the live capture oracle.
    output.mkdir(parents=True, exist_ok=False)
    library = output/'vendor_oracle.so'
    command = [executable, '-std=c++11', '-O2', '-shared', '-fPIC', str(source), '-o', str(library)]
    receipt = dict(schema='nso.offline_snapshot_build.v1',
        scope='Offline snapshot only; no live driver, ROS, network or capture access',
        status='building', source=str(source), cpp_sha256=actual,
        original_vendor_source_sha256=VENDOR_SOURCE_SHA256,
        preserved_function_body_sha256=FUNCTION_SHA256,
        original_vendor_tree_reverified=False,
        provenance='Original extraction is recorded by build_oracle.py; this command checks the complete preserved snapshot',
        command=command, platform=platform.platform(), machine=platform.machine())
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=45, check=True)
        receipt.update(status='built', library_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
                       compiler_stdout=completed.stdout, compiler_stderr=completed.stderr)
    except Exception as exc:
        receipt.update(status='failed', error=str(exc))
        raise
    finally:
        (output/'snapshot_build.json').write_text(json.dumps(receipt, indent=2)+'\n')
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source', type=Path, help='Preserved vendor_oracle.cpp (exact checksum required)')
    parser.add_argument('--compiler', default='c++')
    args = parser.parse_args()
    result = build(args.output, args.source, args.compiler)
    print(json.dumps(dict(status=result['status'], library_sha256=result['library_sha256'])))
