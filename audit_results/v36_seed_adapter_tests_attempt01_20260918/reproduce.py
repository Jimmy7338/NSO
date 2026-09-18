"""One array-only adapter admission run; records stdout before sealing."""
import io
from pathlib import Path
import sys
import time
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from nso.research_evidence_v31 import (freeze, read, seal, sha, verify_inventory,
                                     verify_sources, write, write_bytes)
from tests.virtual3d import test_information_pixel_v36 as tests


def main():
    directory = Path(__file__).resolve().parent
    if (directory/'manifest.json').exists():
        raise FileExistsError('admission cannot be silently repeated')
    inputs = {p: sha(ROOT/p) for p in tests.FROZEN_INPUTS_V36}
    manifest = freeze(directory, [Path(__file__), ROOT/'env/information_pixel_v36.py',
        ROOT/'tests/virtual3d/test_information_pixel_v36.py'], input_sha256=inputs,
        scope='mock constructor and synthetic-array noise functions; P01 saved-array format only; zero real World, packet, policy, TSDF or quality',
        seed=350918, compared_existing_seed=1901)
    transcript = io.StringIO()
    started = time.monotonic()
    result = unittest.TextTestRunner(stream=transcript, verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(tests.InformationPixelV36Tests))
    duration = time.monotonic()-started
    write_bytes(directory, directory/'stdout.txt', transcript.getvalue().encode())
    verify_sources(directory)
    for p, expected in inputs.items():
        if sha(ROOT/p) != expected:
            raise ValueError('admission input changed: '+p)
    receipt = dict(tests.RECEIPT_V36)
    passed = (result.wasSuccessful() and not any(receipt['sensor_counter_delta'].values())
              and receipt.get('P01_saved_format_passed') is True)
    output = dict(status='passed' if passed else 'failed', exit_code=0 if passed else 1,
        tests_run=result.testsRun, test_duration_seconds=duration, test_execution_count=1,
        seed=350918, compared_existing_seed=1901,
        new_worlds=0, new_sensor_packets=0, new_TSDF_integrations=0,
        new_quality_evaluations=0, new_main_tasks=0, new_controller_calls=0,
        new_DP_calls=0, P01_saved_format_passed=receipt.get('P01_saved_format_passed', False),
        scope=manifest['scope'], receipt=receipt)
    write(directory, directory/'result.json', output)
    seal(directory)
    verify_inventory(directory)
    print(transcript.getvalue(), end='')
    print('result_sha256='+sha(directory/'result.json'))
    print('artifact_hashes_sha256='+sha(directory/'artifact_hashes.json'))
    return output['exit_code']


if __name__ == '__main__':
    raise SystemExit(main())
